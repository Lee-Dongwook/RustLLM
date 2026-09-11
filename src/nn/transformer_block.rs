use crate::error::{Result, TinyError};

use crate::metal::MetalContext;
use crate::model::LayerKvCache;
use crate::profile::DecodeProfile;
use crate::tensor::{DType, Tensor};
use std::time::Instant;

use super::{Mlp, RmsNorm, SelfAttention};

pub struct TransformerBlock {
    attention_norm: RmsNorm,
    attention: SelfAttention,

    mlp_norm: RmsNorm,
    mlp: Mlp,
}

impl TransformerBlock {
    pub fn new(
        attention_norm: RmsNorm,
        attention: SelfAttention,
        mlp_norm: RmsNorm,
        mlp: Mlp,
    ) -> Result<Self> {
        let hidden_size = attention.hidden_size();

        if attention_norm.hidden_size() != hidden_size {
            return Err(TinyError::InvalidShape(format!(
                "attention RMSNorm hidden size {} does not match attention hidden size {hidden_size}",
                attention_norm.hidden_size(),
            )));
        }

        if mlp_norm.hidden_size() != hidden_size {
            return Err(TinyError::InvalidShape(format!(
                "MLP RMSNorm hidden size {} does not match hidden size {hidden_size}",
                mlp_norm.hidden_size(),
            )));
        }

        if mlp.input_size() != hidden_size {
            return Err(TinyError::InvalidShape(format!(
                "MLP input size {} does not match hidden size {hidden_size}",
                mlp.input_size(),
            )));
        }

        if mlp.output_size() != hidden_size {
            return Err(TinyError::InvalidShape(format!(
                "MLP output size {} does not match hidden size {hidden_size}",
                mlp.output_size(),
            )));
        }

        if attention_norm.dtype() != attention.dtype()
            || mlp_norm.dtype() != attention.dtype()
            || mlp.dtype() != attention.dtype()
        {
            return Err(TinyError::UnsupportedDType(format!(
                "TransformerBlock component dtypes must match, got attention_norm={:?}, attention={:?}, mlp_norm={:?}, mlp={:?}",
                attention_norm.dtype(),
                attention.dtype(),
                mlp_norm.dtype(),
                mlp.dtype(),
            )));
        }

        Ok(Self {
            attention_norm,
            attention,
            mlp_norm,
            mlp,
        })
    }

    pub fn hidden_size(&self) -> usize {
        self.attention.hidden_size()
    }

    pub fn dtype(&self) -> DType {
        self.attention.dtype()
    }

    pub fn to_dtype(&self, context: &MetalContext, dtype: DType) -> Result<Self> {
        Self::new(
            self.attention_norm.to_dtype(context, dtype)?,
            self.attention.to_dtype(context, dtype)?,
            self.mlp_norm.to_dtype(context, dtype)?,
            self.mlp.to_dtype(context, dtype)?,
        )
    }

    pub fn forward(&self, context: &MetalContext, input: &Tensor) -> Result<Tensor> {
        self.validate_input_dtype(input)?;

        let normalized = self.attention_norm.forward(context, input)?;

        let attention_output = self.attention.forward(context, &normalized)?;

        let hidden = input.add(context, &attention_output)?;

        let normalized = self.mlp_norm.forward(context, &hidden)?;

        let mlp_output = self.mlp.forward(context, &normalized)?;

        hidden.add(context, &mlp_output)
    }
    pub fn forward_with_cache(
        &self,
        context: &MetalContext,
        x: &Tensor,
        cache: &mut LayerKvCache,
    ) -> Result<Tensor> {
        self.validate_input_dtype(x)?;

        let normalized = self.attention_norm.forward(context, x)?;

        let attention = self
            .attention
            .forward_with_cache(context, &normalized, cache)?;

        let hidden = x.add(context, &attention)?;

        let normalized = self.mlp_norm.forward(context, &hidden)?;

        let mlp = self.mlp.forward(context, &normalized)?;

        hidden.add(context, &mlp)
    }

    /// Decode-only profiled variant. The normal inference method above remains
    /// branch-free when profiling is disabled.
    pub fn forward_with_cache_profiled(
        &self,
        context: &MetalContext,
        x: &Tensor,
        cache: &mut LayerKvCache,
        profile: &mut DecodeProfile,
    ) -> Result<Tensor> {
        self.validate_input_dtype(x)?;

        let started = Instant::now();
        let normalized = self.attention_norm.forward(context, x)?;
        profile.norm += started.elapsed();

        let attention =
            self.attention
                .forward_with_cache_profiled(context, &normalized, cache, profile)?;

        let started = Instant::now();
        let hidden = x.add(context, &attention)?;
        profile.residual += started.elapsed();

        let started = Instant::now();
        let normalized = self.mlp_norm.forward(context, &hidden)?;
        profile.norm += started.elapsed();

        let mlp = self.mlp.forward_profiled(context, &normalized, profile)?;

        let started = Instant::now();
        let output = hidden.add(context, &mlp)?;
        profile.residual += started.elapsed();
        Ok(output)
    }

    fn validate_input_dtype(&self, input: &Tensor) -> Result<()> {
        if input.dtype() != self.dtype() {
            return Err(TinyError::UnsupportedDType(format!(
                "TransformerBlock input dtype {:?} does not match block dtype {:?}",
                input.dtype(),
                self.dtype(),
            )));
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::{Mlp, RmsNorm, SelfAttention, TransformerBlock};
    use crate::{
        error::TinyError,
        metal::MetalContext,
        model::LayerKvCache,
        nn::{Linear, RotaryEmbedding},
        tensor::{DType, Tensor},
    };

    fn metal_context() -> Option<MetalContext> {
        match MetalContext::new() {
            Ok(context) => Some(context),
            Err(TinyError::Metal(message)) => {
                eprintln!("skipping Metal TransformerBlock test: {message}");
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

    fn block(context: &MetalContext) -> TransformerBlock {
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
        let attention_norm = RmsNorm::new(
            Tensor::from_f32_slice(context, &[1.0; 4], &[4]).unwrap(),
            1e-5,
        )
        .unwrap();
        let mlp_norm = RmsNorm::new(
            Tensor::from_f32_slice(context, &[1.0; 4], &[4]).unwrap(),
            1e-5,
        )
        .unwrap();
        let mlp = Mlp::new(
            identity_linear(context),
            identity_linear(context),
            identity_linear(context),
        )
        .unwrap();

        TransformerBlock::new(attention_norm, attention, mlp_norm, mlp).unwrap()
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
                "TransformerBlock mismatch: expected={expected}, actual={actual}, error={error}",
            );
        }
    }

    #[test]
    fn f16_transformer_block_matches_f32() {
        let Some(context) = metal_context() else {
            return;
        };
        let block = block(&context);
        let input = Tensor::from_f32_slice(
            &context,
            &[0.1, 0.2, 0.3, 0.4, -0.2, 0.5, 0.7, -0.1],
            &[2, 4],
        )
        .unwrap();

        let expected = block.forward(&context, &input).unwrap();
        let block_f16 = block.to_dtype(&context, DType::F16).unwrap();
        let actual = block_f16
            .forward(&context, &input.to_dtype(&context, DType::F16).unwrap())
            .unwrap();

        assert_eq!(block_f16.dtype(), DType::F16);
        assert_eq!(actual.dtype(), DType::F16);
        assert_close(&expected, &actual, 0.2);
    }

    #[test]
    fn f16_transformer_block_supports_decode_cache() {
        let Some(context) = metal_context() else {
            return;
        };
        let block_f16 = block(&context).to_dtype(&context, DType::F16).unwrap();
        let token_one = Tensor::from_f32_slice(&context, &[0.1, 0.2, 0.3, 0.4], &[1, 4])
            .unwrap()
            .to_dtype(&context, DType::F16)
            .unwrap();
        let token_two = Tensor::from_f32_slice(&context, &[-0.3, 0.6, 0.2, 0.8], &[1, 4])
            .unwrap()
            .to_dtype(&context, DType::F16)
            .unwrap();
        let mut cache = LayerKvCache::new(&context, 2, 8, 2, DType::F16).unwrap();

        block_f16
            .forward_with_cache(&context, &token_one, &mut cache)
            .unwrap();
        assert_eq!(cache.len(), 1);

        let output = block_f16
            .forward_with_cache(&context, &token_two, &mut cache)
            .unwrap();
        assert_eq!(cache.len(), 2);
        assert_eq!(output.dtype(), DType::F16);
        assert_eq!(output.shape().dims(), &[1, 4]);
    }

    #[test]
    fn f16_transformer_block_rejects_f32_input() {
        let Some(context) = metal_context() else {
            return;
        };
        let block_f16 = block(&context).to_dtype(&context, DType::F16).unwrap();
        let input = Tensor::from_f32_slice(&context, &[0.1, 0.2, 0.3, 0.4], &[1, 4]).unwrap();

        assert!(block_f16.forward(&context, &input).is_err());
    }
}
