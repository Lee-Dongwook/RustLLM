use crate::error::{Result, TinyError};
use std::path::Path;

use crate::metal::MetalContext;

use crate::nn::{
    Embedding, Linear, Mlp, RmsNorm, RotaryEmbedding, SelfAttention, TransformerBlock,
};

use crate::tensor::Tensor;

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

            let q_proj = Linear::new(take_tensor(
                context,
                &mut weights,
                &format!("{prefix}.attention.q_proj.weight"),
                &[config.hidden_size, config.hidden_size],
            )?)?;

            let k_proj = Linear::new(take_tensor(
                context,
                &mut weights,
                &format!("{prefix}.attention.k_proj.weight"),
                &[config.hidden_size, config.hidden_size],
            )?)?;

            let v_proj = Linear::new(take_tensor(
                context,
                &mut weights,
                &format!("{prefix}.attention.v_proj.weight"),
                &[config.hidden_size, config.hidden_size],
            )?)?;

            let out_proj = Linear::new(take_tensor(
                context,
                &mut weights,
                &format!("{prefix}.attention.out_proj.weight"),
                &[config.hidden_size, config.hidden_size],
            )?)?;

            let attention = SelfAttention::new(
                q_proj,
                k_proj,
                v_proj,
                out_proj,
                rope.clone(),
                config.num_heads,
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

            let gate_proj = Linear::new(take_tensor(
                context,
                &mut weights,
                &format!("{prefix}.mlp.gate_proj.weight"),
                &[config.hidden_size, config.intermediate_size],
            )?)?;

            let up_proj = Linear::new(take_tensor(
                context,
                &mut weights,
                &format!("{prefix}.mlp.up_proj.weight"),
                &[config.hidden_size, config.intermediate_size],
            )?)?;

            let down_proj = Linear::new(take_tensor(
                context,
                &mut weights,
                &format!("{prefix}.mlp.down_proj.weight"),
                &[config.intermediate_size, config.hidden_size],
            )?)?;

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

        let lm_head = Linear::new(take_tensor(
            context,
            &mut weights,
            "lm_head.weight",
            &[config.hidden_size, config.vocab_size],
        )?)?;

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

    let (shape, data) = weight.into_parts();

    Tensor::from_f32_slice(context, &data, &shape)
}
