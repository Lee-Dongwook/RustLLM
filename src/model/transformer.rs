use crate::error::{Result, TinyError};
use crate::model::linear_scale_name;
use std::path::Path;

use crate::metal::MetalContext;

use crate::nn::{
    Embedding, Linear, Mlp, RmsNorm, RotaryEmbedding, SelfAttention, TransformerBlock,
};

use crate::tensor::{DType, Tensor};

use super::weights::WeightData;
use super::{KvCache, ModelConfig, ModelWeights, WeightTensor};

mod cache;

pub struct Transformer {
    config: ModelConfig,

    token_embedding: Embedding,

    blocks: Vec<TransformerBlock>,

    final_norm: RmsNorm,

    lm_head: Linear,
}

impl Transformer {
    pub fn new(
        config: ModelConfig,
        token_embedding: Embedding,
        blocks: Vec<TransformerBlock>,
        final_norm: RmsNorm,
        lm_head: Linear,
    ) -> Result<Self> {
        config.validate()?;

        if token_embedding.vocab_size() != config.vocab_size {
            return Err(TinyError::InvalidShape(format!(
                "embedding vocabulary size {} does not match model vocabulary size {}",
                token_embedding.vocab_size(),
                config.vocab_size,
            )));
        }

        if token_embedding.hidden_size() != config.hidden_size {
            return Err(TinyError::InvalidShape(format!(
                "embedding hidden size {} does not match model hidden size {}",
                token_embedding.hidden_size(),
                config.hidden_size,
            )));
        }

        if blocks.len() != config.num_layers {
            return Err(TinyError::InvalidShape(format!(
                "model expects {} transformer layers, got {}",
                config.num_layers,
                blocks.len(),
            )));
        }

        for (index, block) in blocks.iter().enumerate() {
            if block.hidden_size() != config.hidden_size {
                return Err(TinyError::InvalidShape(format!(
                    "transformer block {index} has hidden size {}, expected {}",
                    block.hidden_size(),
                    config.hidden_size,
                )));
            }
        }

        if final_norm.hidden_size() != config.hidden_size {
            return Err(TinyError::InvalidShape(format!(
                "final RMSNorm hidden size {} does not match model hidden size {}",
                final_norm.hidden_size(),
                config.hidden_size,
            )));
        }

        if lm_head.in_features() != config.hidden_size {
            return Err(TinyError::InvalidShape(format!(
                "LM head input size {} does not match model hidden size {}",
                lm_head.in_features(),
                config.hidden_size,
            )));
        }

        if lm_head.out_features() != config.vocab_size {
            return Err(TinyError::InvalidShape(format!(
                "LM head output size {} does not match vocabulary size {}",
                lm_head.out_features(),
                config.vocab_size,
            )));
        }

        Ok(Self {
            config,
            token_embedding,
            blocks,
            final_norm,
            lm_head,
        })
    }

    pub fn config(&self) -> &ModelConfig {
        &self.config
    }

    pub fn dtype(&self) -> DType {
        self.token_embedding.dtype()
    }

    /// Returns true when this package uses INT8 weight-only Linear layers.
    pub fn has_int8_weights(&self) -> bool {
        self.lm_head.is_quantized()
    }

    /// Converts runtime weights while preserving the on-disk F32 model format.
    pub fn to_dtype(&self, context: &MetalContext, dtype: DType) -> Result<Self> {
        let blocks = self
            .blocks
            .iter()
            .map(|block| block.to_dtype(context, dtype))
            .collect::<Result<Vec<_>>>()?;

        Self::new(
            self.config.clone(),
            self.token_embedding.to_dtype(context, dtype)?,
            blocks,
            self.final_norm.to_dtype(context, dtype)?,
            self.lm_head.to_dtype(context, dtype)?,
        )
    }

    pub fn forward(&self, context: &MetalContext, token_ids: &[u32]) -> Result<Tensor> {
        if token_ids.is_empty() {
            return Err(TinyError::InvalidShape(
                "Transformer input cannot be empty".to_string(),
            ));
        }

        if token_ids.len() > self.config.max_seq_len {
            return Err(TinyError::PositionOutOfRange {
                start_pos: 0,
                seq_len: token_ids.len(),
                max_seq_len: self.config.max_seq_len,
            });
        }

        // -----------------------------------------------
        // Token IDs
        // ↓
        // Embedding
        //
        // [sequence, hidden]
        // -----------------------------------------------

        let mut hidden = self.token_embedding.forward(context, token_ids)?;

        // -----------------------------------------------
        // Transformer layers
        // -----------------------------------------------

        for block in &self.blocks {
            hidden = block.forward(context, &hidden)?;
        }

        // -----------------------------------------------
        // Final RMSNorm
        // -----------------------------------------------

        let hidden = self.final_norm.forward(context, &hidden)?;

        // -----------------------------------------------
        // LM Head
        //
        // [sequence, hidden]
        //
        // ×
        //
        // [hidden, vocab]
        //
        // →
        //
        // [sequence, vocab]
        // -----------------------------------------------

        self.lm_head.forward(context, &hidden)
    }

    pub fn load(context: &MetalContext, model_dir: impl AsRef<Path>) -> Result<Self> {
        let model_dir = model_dir.as_ref();

        let config = ModelConfig::load_json(model_dir.join("config.json"))?;

        let weights = ModelWeights::load(model_dir.join("model.bin"))?;

        Self::from_weights(context, config, weights)
    }

    pub fn from_weights(
        context: &MetalContext,
        config: ModelConfig,
        mut weights: ModelWeights,
    ) -> Result<Self> {
        config.validate()?;
        let token_embedding = Embedding::new(take_tensor(
            context,
            &mut weights,
            "token_embedding.weight",
            &[config.vocab_size, config.hidden_size],
        )?)?;

        let rope = RotaryEmbedding::new(
            context,
            config.head_dim(),
            config.max_seq_len,
            config.rope_theta,
        )?;

        let mut blocks = Vec::with_capacity(config.num_layers);

        for layer_index in 0..config.num_layers {
            let prefix = format!("layers.{layer_index}");

            let q_proj = take_linear(
                context,
                &mut weights,
                &format!("{prefix}.attention.q_proj.weight"),
                &[config.hidden_size, config.q_proj_size()],
            )?;

            let k_proj = take_linear(
                context,
                &mut weights,
                &format!("{prefix}.attention.k_proj.weight"),
                &[config.hidden_size, config.kv_proj_size()],
            )?;

            let v_proj = take_linear(
                context,
                &mut weights,
                &format!("{prefix}.attention.v_proj.weight"),
                &[config.hidden_size, config.kv_proj_size()],
            )?;

            let out_proj = take_linear(
                context,
                &mut weights,
                &format!("{prefix}.attention.out_proj.weight"),
                &[config.q_proj_size(), config.hidden_size],
            )?;

            let attention = SelfAttention::new(
                q_proj,
                k_proj,
                v_proj,
                out_proj,
                rope.clone(),
                config.num_heads,
                config.num_kv_heads,
            )?;

            let attention_norm = RmsNorm::new(
                take_tensor(
                    context,
                    &mut weights,
                    &format!("{prefix}.attention_norm.weight"),
                    &[config.hidden_size],
                )?,
                config.rms_norm_eps,
            )?;
            let mlp_norm = RmsNorm::new(
                take_tensor(
                    context,
                    &mut weights,
                    &format!("{prefix}.mlp_norm.weight"),
                    &[config.hidden_size],
                )?,
                config.rms_norm_eps,
            )?;

            let gate_proj = take_linear(
                context,
                &mut weights,
                &format!("{prefix}.mlp.gate_proj.weight"),
                &[config.hidden_size, config.intermediate_size],
            )?;

            let up_proj = take_linear(
                context,
                &mut weights,
                &format!("{prefix}.mlp.up_proj.weight"),
                &[config.hidden_size, config.intermediate_size],
            )?;

            let down_proj = take_linear(
                context,
                &mut weights,
                &format!("{prefix}.mlp.down_proj.weight"),
                &[config.intermediate_size, config.hidden_size],
            )?;

            let mlp = Mlp::new(gate_proj, up_proj, down_proj)?;
            let block = TransformerBlock::new(attention_norm, attention, mlp_norm, mlp)?;

            blocks.push(block);
        }

        let final_norm = RmsNorm::new(
            take_tensor(
                context,
                &mut weights,
                "final_norm.weight",
                &[config.hidden_size],
            )?,
            config.rms_norm_eps,
        )?;

        let lm_head = take_linear(
            context,
            &mut weights,
            "lm_head.weight",
            &[config.hidden_size, config.vocab_size],
        )?;

        Self::new(config, token_embedding, blocks, final_norm, lm_head)
    }
}

fn take_tensor(
    context: &MetalContext,
    weights: &mut ModelWeights,
    name: &str,
    expected_shape: &[usize],
) -> Result<Tensor> {
    let weight: WeightTensor = weights.take(name)?;

    if weight.shape() != expected_shape {
        return Err(TinyError::InvalidShape(format!(
            "weight {name} has shape {:?}, expected {:?}",
            weight.shape(),
            expected_shape,
        )));
    }

    let (shape, data) = weight.into_storage_parts();

    match data {
        WeightData::F32(data) => Tensor::from_f32_slice(context, &data, &shape),
        WeightData::F16(data) => Tensor::from_f16_slice(context, &data, &shape),
        WeightData::I8(_) => Err(TinyError::ModelFormat(format!(
            "INT8 weight {name} must be loaded through QuantizedLinear"
        ))),
    }
}

fn take_linear(
    context: &MetalContext,
    weights: &mut ModelWeights,
    name: &str,
    expected_shape: &[usize],
) -> Result<Linear> {
    let weight = weights.take(name)?;
    if weight.shape() != expected_shape {
        return Err(TinyError::InvalidShape(format!(
            "weight {name} has shape {:?}, expected {:?}",
            weight.shape(),
            expected_shape,
        )));
    }

    let (shape, data) = weight.into_storage_parts();
    match data {
        WeightData::F32(values) => Linear::new(Tensor::from_f32_slice(context, &values, &shape)?),
        WeightData::F16(values) => Linear::new(Tensor::from_f16_slice(context, &values, &shape)?),
        WeightData::I8(values) => {
            let scale_name = linear_scale_name(name);
            let scale_weight = weights.take(&scale_name)?;
            let expected_scale_shape = [shape[1]];
            if scale_weight.shape() != expected_scale_shape {
                return Err(TinyError::InvalidShape(format!(
                    "INT8 scale {scale_name} shape mismatch: expected {:?}, got {:?}",
                    expected_scale_shape,
                    scale_weight.shape(),
                )));
            }
            let (_, scale_data) = scale_weight.into_storage_parts();
            let scales = match scale_data {
                WeightData::F32(scales) => scales,
                _ => {
                    return Err(TinyError::ModelFormat(format!(
                        "INT8 scale {scale_name} must use F32 storage"
                    )));
                }
            };
            Linear::from_i8(context, &shape, &values, &scales)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{ModelConfig, Transformer, take_linear};
    use crate::{
        error::TinyError,
        metal::MetalContext,
        model::{ModelWeights, quantize_model_i8},
        nn::{Embedding, Linear, Mlp, RmsNorm, RotaryEmbedding, SelfAttention, TransformerBlock},
        profile::DecodeProfile,
        tensor::{DType, Tensor},
    };

    fn metal_context() -> Option<MetalContext> {
        match MetalContext::new() {
            Ok(context) => Some(context),
            Err(TinyError::Metal(message)) => {
                eprintln!("skipping Metal Transformer test: {message}");
                None
            }
            Err(error) => panic!("failed to create Metal context: {error}"),
        }
    }

    fn identity_linear(context: &MetalContext) -> Linear {
        Linear::new(
            Tensor::from_f32_slice(
                context,
                &[
                    1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0,
                ],
                &[4, 4],
            )
            .unwrap(),
        )
        .unwrap()
    }

    fn tiny_transformer(context: &MetalContext) -> Transformer {
        let config = ModelConfig {
            vocab_size: 5,
            hidden_size: 4,
            intermediate_size: 4,
            num_layers: 1,
            num_heads: 2,
            num_kv_heads: 2,
            max_seq_len: 8,
            rms_norm_eps: 1e-5,
            rope_theta: 10_000.0,
        };
        let embedding = Embedding::new(
            Tensor::from_f32_slice(
                context,
                &[
                    0.1, 0.2, 0.3, 0.4, 0.2, 0.3, 0.4, 0.5, -0.1, 0.3, 0.5, 0.2, 0.4, -0.2, 0.1,
                    0.6, 0.5, 0.1, -0.3, 0.2,
                ],
                &[5, 4],
            )
            .unwrap(),
        )
        .unwrap();
        let attention = SelfAttention::new(
            identity_linear(context),
            identity_linear(context),
            identity_linear(context),
            identity_linear(context),
            RotaryEmbedding::new(context, 2, 8, 10_000.0).unwrap(),
            2,
            2,
        )
        .unwrap();
        let block = TransformerBlock::new(
            RmsNorm::new(
                Tensor::from_f32_slice(context, &[1.0; 4], &[4]).unwrap(),
                1e-5,
            )
            .unwrap(),
            attention,
            RmsNorm::new(
                Tensor::from_f32_slice(context, &[1.0; 4], &[4]).unwrap(),
                1e-5,
            )
            .unwrap(),
            Mlp::new(
                identity_linear(context),
                identity_linear(context),
                identity_linear(context),
            )
            .unwrap(),
        )
        .unwrap();
        let final_norm = RmsNorm::new(
            Tensor::from_f32_slice(context, &[1.0; 4], &[4]).unwrap(),
            1e-5,
        )
        .unwrap();
        let lm_head = Linear::new(
            Tensor::from_f32_slice(
                context,
                &[
                    0.1, 0.2, -0.1, 0.3, 0.2, -0.2, 0.4, 0.1, 0.3, 0.1, 0.2, -0.3, -0.1, 0.5, 0.2,
                    0.4, 0.2, -0.1, 0.3, 0.1,
                ],
                &[4, 5],
            )
            .unwrap(),
        )
        .unwrap();

        Transformer::new(config, embedding, vec![block], final_norm, lm_head).unwrap()
    }

    fn assert_close(expected: &Tensor, actual: &Tensor, tolerance: f32) {
        assert_eq!(expected.shape().dims(), actual.shape().dims());
        for (expected, actual) in expected
            .to_f32_vec()
            .unwrap()
            .iter()
            .zip(actual.to_f32_vec().unwrap().iter())
        {
            let error = (expected - actual).abs();
            assert!(
                error < tolerance,
                "Transformer logits mismatch: expected={expected}, actual={actual}, error={error}",
            );
        }
    }

    #[test]
    fn loads_int8_linear_with_scale() {
        let Some(context) = metal_context() else {
            return;
        };
        let mut weights = ModelWeights::new();
        weights
            .insert_i8("test.weight", &[2, 2], vec![127, -127, 64, -64])
            .unwrap();
        weights
            .insert_f32("test.scale", &[2], vec![0.01, 0.02])
            .unwrap();

        let linear = take_linear(&context, &mut weights, "test.weight", &[2, 2]).unwrap();

        assert_eq!(linear.in_features(), 2);
        assert_eq!(linear.out_features(), 2);
        assert_eq!(linear.dtype(), DType::F16);
        assert!(weights.is_empty());
    }

    #[test]
    fn rejects_int8_linear_with_wrong_scale_shape() {
        let Some(context) = metal_context() else {
            return;
        };
        let mut weights = ModelWeights::new();
        weights
            .insert_i8("test.weight", &[2, 3], vec![0; 6])
            .unwrap();
        weights
            .insert_f32("test.scale", &[2], vec![1.0; 2])
            .unwrap();

        let error = match take_linear(&context, &mut weights, "test.weight", &[2, 3]) {
            Ok(_) => panic!("INT8 Linear with an invalid scale shape must be rejected"),
            Err(error) => error,
        };

        assert!(
            matches!(error, TinyError::InvalidShape(message) if message.contains("expected [3]") && message.contains("got [2]"))
        );
    }

    #[test]
    fn f16_transformer_matches_f32_and_returns_finite_logits() {
        let Some(context) = metal_context() else {
            return;
        };
        let model = tiny_transformer(&context);
        let expected = model.forward(&context, &[1, 2]).unwrap();
        let model_f16 = model.to_dtype(&context, DType::F16).unwrap();
        let actual = model_f16.forward(&context, &[1, 2]).unwrap();

        assert_eq!(model.dtype(), DType::F32);
        assert_eq!(model_f16.dtype(), DType::F16);
        assert_eq!(actual.dtype(), DType::F16);
        assert!(
            actual
                .to_f32_vec()
                .unwrap()
                .iter()
                .all(|value| value.is_finite())
        );
        assert_close(&expected, &actual, 0.2);
    }

    #[test]
    fn f16_transformer_uses_f16_cache_for_prefill_and_decode() {
        let Some(context) = metal_context() else {
            return;
        };
        let model_f16 = tiny_transformer(&context)
            .to_dtype(&context, DType::F16)
            .unwrap();
        let mut cache = model_f16.new_kv_cache(&context).unwrap();
        assert_eq!(cache.layer_mut(0).unwrap().dtype(), DType::F16);

        let prefill = model_f16
            .forward_with_cache(&context, &[1, 2, 3], &mut cache)
            .unwrap();
        assert_eq!(prefill.dtype(), DType::F16);
        assert_eq!(cache.layer_mut(0).unwrap().len(), 3);

        let decode = model_f16
            .forward_with_cache(&context, &[4], &mut cache)
            .unwrap();
        assert_eq!(decode.dtype(), DType::F16);
        assert!(
            decode
                .to_f32_vec()
                .unwrap()
                .iter()
                .all(|value| value.is_finite())
        );
        assert_eq!(cache.layer_mut(0).unwrap().len(), 4);
    }

    #[test]
    fn profiled_decode_matches_normal_decode() {
        let Some(context) = metal_context() else {
            return;
        };
        let model = tiny_transformer(&context)
            .to_dtype(&context, DType::F16)
            .unwrap();
        let mut normal_cache = model.new_kv_cache(&context).unwrap();
        let mut profiled_cache = model.new_kv_cache(&context).unwrap();
        model
            .forward_with_cache(&context, &[1, 2], &mut normal_cache)
            .unwrap();
        model
            .forward_with_cache(&context, &[1, 2], &mut profiled_cache)
            .unwrap();

        let expected = model
            .forward_with_cache(&context, &[3], &mut normal_cache)
            .unwrap();
        let mut profile = DecodeProfile::default();
        let actual = model
            .forward_with_cache_profiled(&context, &[3], &mut profiled_cache, &mut profile)
            .unwrap();

        assert_close(&expected, &actual, 0.001);
        assert_eq!(profile.sampled_tokens, 0);
        assert!(!profile.profiled_operations().is_zero());
    }

    #[test]
    fn loads_mixed_int8_weights_and_runs_forward() {
        let Some(context) = metal_context() else {
            return;
        };
        let config = ModelConfig {
            vocab_size: 5,
            hidden_size: 4,
            intermediate_size: 4,
            num_layers: 1,
            num_heads: 2,
            num_kv_heads: 2,
            max_seq_len: 8,
            rms_norm_eps: 1e-5,
            rope_theta: 10_000.0,
        };
        let mut weights = ModelWeights::new();
        weights
            .insert_f32("token_embedding.weight", &[5, 4], vec![0.1; 20])
            .unwrap();
        weights
            .insert_f32("layers.0.attention_norm.weight", &[4], vec![1.0; 4])
            .unwrap();
        for projection in [
            "attention.q_proj.weight",
            "attention.k_proj.weight",
            "attention.v_proj.weight",
            "attention.out_proj.weight",
            "mlp.gate_proj.weight",
            "mlp.up_proj.weight",
            "mlp.down_proj.weight",
        ] {
            weights
                .insert_f32(format!("layers.0.{projection}"), &[4, 4], vec![0.1; 16])
                .unwrap();
        }
        weights
            .insert_f32("layers.0.mlp_norm.weight", &[4], vec![1.0; 4])
            .unwrap();
        weights
            .insert_f32("final_norm.weight", &[4], vec![1.0; 4])
            .unwrap();
        weights
            .insert_f32("lm_head.weight", &[4, 5], vec![0.1; 20])
            .unwrap();

        let model =
            Transformer::from_weights(&context, config, quantize_model_i8(weights, 1).unwrap())
                .unwrap();
        let logits = model.forward(&context, &[1, 2]).unwrap();

        assert!(model.has_int8_weights());
        assert_eq!(model.dtype(), DType::F16);
        assert_eq!(logits.dtype(), DType::F16);
        assert!(
            logits
                .to_f32_vec()
                .unwrap()
                .iter()
                .all(|value| value.is_finite())
        );
    }
}
