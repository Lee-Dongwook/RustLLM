use crate::error::{Result, TinyError};

use crate::metal::MetalContext;
use crate::model::LayerKvCache;
use crate::tensor::{DType, Tensor};

use super::{Linear, RotaryEmbedding};

pub struct SelfAttention {
    q_proj: Linear,
    k_proj: Linear,
    v_proj: Linear,
    out_proj: Linear,

    rope: RotaryEmbedding,

    hidden_size: usize,
    num_heads: usize,
    head_dim: usize,
}

impl SelfAttention {
    pub fn new(
        q_proj: Linear,
        k_proj: Linear,
        v_proj: Linear,
        out_proj: Linear,
        rope: RotaryEmbedding,
        num_heads: usize,
    ) -> Result<Self> {
        if num_heads == 0 {
            return Err(TinyError::InvalidShape(
                "attention num_heads cannot be zero".to_string(),
            ));
        }

        let hidden_size = q_proj.in_features();

        if q_proj.out_features() != hidden_size {
            return Err(TinyError::InvalidShape(format!(
                "q_proj output size must equal hidden size {hidden_size}"
            )));
        }

        if k_proj.in_features() != hidden_size || k_proj.out_features() != hidden_size {
            return Err(TinyError::InvalidShape(
                "k_proj shape must match hidden size".to_string(),
            ));
        }

        if v_proj.in_features() != hidden_size || v_proj.out_features() != hidden_size {
            return Err(TinyError::InvalidShape(
                "v_proj shape must match hidden size".to_string(),
            ));
        }

        if out_proj.in_features() != hidden_size || out_proj.out_features() != hidden_size {
            return Err(TinyError::InvalidShape(
                "out_proj shape must match hidden size".to_string(),
            ));
        }

        if !hidden_size.is_multiple_of(num_heads) {
            return Err(TinyError::InvalidShape(format!(
                "hidden size {hidden_size} must be divisible by num_heads {num_heads}"
            )));
        }

        let head_dim = hidden_size / num_heads;

        if rope.head_dim() != head_dim {
            return Err(TinyError::InvalidShape(format!(
                "RoPE head dimension {} does not match attention head dimension {head_dim}",
                rope.head_dim(),
            )));
        }

        if k_proj.dtype() != q_proj.dtype()
            || v_proj.dtype() != q_proj.dtype()
            || out_proj.dtype() != q_proj.dtype()
        {
            return Err(TinyError::UnsupportedDType(format!(
                "SelfAttention projection dtypes must match, got Q={:?}, K={:?}, V={:?}, O={:?}",
                q_proj.dtype(),
                k_proj.dtype(),
                v_proj.dtype(),
                out_proj.dtype(),
            )));
        }

        Ok(Self {
            q_proj,
            k_proj,
            v_proj,
            out_proj,
            rope,
            hidden_size,
            num_heads,
            head_dim,
        })
    }

    pub fn hidden_size(&self) -> usize {
        self.hidden_size
    }

    pub fn num_heads(&self) -> usize {
        self.num_heads
    }

    pub fn head_dim(&self) -> usize {
        self.head_dim
    }

    pub fn dtype(&self) -> DType {
        self.q_proj.dtype()
    }

    pub fn to_dtype(&self, context: &MetalContext, dtype: DType) -> Result<Self> {
        Self::new(
            self.q_proj.to_dtype(context, dtype)?,
            self.k_proj.to_dtype(context, dtype)?,
            self.v_proj.to_dtype(context, dtype)?,
            self.out_proj.to_dtype(context, dtype)?,
            self.rope.clone(),
            self.num_heads,
        )
    }

    pub fn forward(&self, context: &MetalContext, input: &Tensor) -> Result<Tensor> {
        self.validate_input_dtype(input)?;

        let mut cache = LayerKvCache::new(
            context,
            self.num_heads,
            self.rope.max_seq_len(),
            self.head_dim,
            input.dtype(),
        )?;
        self.forward_with_cache(context, input, &mut cache)
    }

    pub fn forward_with_cache(
        &self,
        context: &MetalContext,
        x: &Tensor,
        cache: &mut LayerKvCache,
    ) -> Result<Tensor> {
        self.validate_input_dtype(x)?;

        if cache.dtype() != self.dtype() {
            return Err(TinyError::UnsupportedDType(format!(
                "SelfAttention/cache dtype mismatch: attention={:?}, cache={:?}",
                self.dtype(),
                cache.dtype(),
            )));
        }

        if x.rank() != 2 {
            return Err(TinyError::ModelFormat(format!(
                "attention expects [sequence, hidden], got {:?}",
                x.shape().dims(),
            )));
        }

        let seq_len = x.dim(0)?;
        let hidden_size = x.dim(1)?;

        if hidden_size != self.hidden_size {
            return Err(TinyError::InvalidShape(format!(
                "attention expected hidden size {}, got {hidden_size}",
                self.hidden_size,
            )));
        }

        let start_pos = cache.len();

        let q = self.q_proj.forward(context, x)?;
        let k = self.k_proj.forward(context, x)?;
        let v = self.v_proj.forward(context, x)?;

        // --------------------------------------------
        // [S, hidden]
        //
        // ↓
        //
        // [1, S, H, D]
        //
        // ↓ permute
        //
        // [1, H, S, D]
        // --------------------------------------------

        let q = q
            .reshape(&[1, seq_len, self.num_heads, self.head_dim])?
            .permute(&[0, 2, 1, 3])?;
        let k = k
            .reshape(&[1, seq_len, self.num_heads, self.head_dim])?
            .permute(&[0, 2, 1, 3])?;
        let v = v
            .reshape(&[1, seq_len, self.num_heads, self.head_dim])?
            .permute(&[0, 2, 1, 3])?;

        // --------------------------------------------
        // RoPE
        //
        // 여기 start_pos가 핵심
        // --------------------------------------------

        let q = self.rope.forward(context, &q, start_pos)?;

        let k = self.rope.forward(context, &k, start_pos)?;

        cache.append(context, k, v)?;

        let all_k = cache.key()?;

        let all_v = cache.value()?;

        let k_transposed = all_k.transpose(2, 3)?;

        let scores = q.batched_matmul(context, &k_transposed)?;

        let scale = 1.0f32 / (self.head_dim as f32).sqrt();

        let scores = scores.attention_scale_mask(context, scale, start_pos)?;

        let probabilities = scores.softmax_last_dim(context)?;

        let attention = probabilities.batched_matmul(context, &all_v)?;
        let attention = attention.permute(&[0, 2, 1, 3])?.contiguous(context)?;

        let attention = attention.reshape(&[seq_len, self.hidden_size])?;

        self.out_proj.forward(context, &attention)
    }

    fn validate_input_dtype(&self, input: &Tensor) -> Result<()> {
        if input.dtype() != self.dtype() {
            return Err(TinyError::UnsupportedDType(format!(
                "SelfAttention input dtype {:?} does not match attention dtype {:?}",
                input.dtype(),
                self.dtype(),
            )));
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::{Linear, RotaryEmbedding, SelfAttention};
    use crate::{
        error::TinyError,
        metal::MetalContext,
        model::LayerKvCache,
        tensor::{DType, Tensor},
    };

    fn metal_context() -> Option<MetalContext> {
        match MetalContext::new() {
            Ok(context) => Some(context),
            Err(TinyError::Metal(message)) => {
                eprintln!("skipping Metal SelfAttention test: {message}");
                None
            }
            Err(error) => panic!("failed to create Metal context: {error}"),
        }
    }

    fn attention(context: &MetalContext) -> SelfAttention {
        let identity = Tensor::from_f32_slice(
            context,
            &[
                1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0,
            ],
            &[4, 4],
        )
        .unwrap();
        SelfAttention::new(
            Linear::new(identity.clone()).unwrap(),
            Linear::new(identity.clone()).unwrap(),
            Linear::new(identity.clone()).unwrap(),
            Linear::new(identity).unwrap(),
            RotaryEmbedding::new(context, 2, 8, 10_000.0).unwrap(),
            2,
        )
        .unwrap()
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
                "attention mismatch: expected={expected}, actual={actual}, error={error}",
            );
        }
    }

    #[test]
    fn f16_self_attention_matches_f32_without_external_cache() {
        let Some(context) = metal_context() else {
            return;
        };
        let attention = attention(&context);
        let input = Tensor::from_f32_slice(
            &context,
            &[0.1, 0.2, 0.3, 0.4, -0.2, 0.5, 0.7, -0.1],
            &[2, 4],
        )
        .unwrap();

        let expected = attention.forward(&context, &input).unwrap();
        let attention_f16 = attention.to_dtype(&context, DType::F16).unwrap();
        let actual = attention_f16
            .forward(&context, &input.to_dtype(&context, DType::F16).unwrap())
            .unwrap();

        assert_eq!(attention_f16.dtype(), DType::F16);
        assert_eq!(actual.dtype(), DType::F16);
        assert_close(&expected, &actual, 0.1);
    }

    #[test]
    fn f16_self_attention_matches_f32_across_decode_cache() {
        let Some(context) = metal_context() else {
            return;
        };
        let attention = attention(&context);
        let attention_f16 = attention.to_dtype(&context, DType::F16).unwrap();
        let token_one = Tensor::from_f32_slice(&context, &[0.1, 0.2, 0.3, 0.4], &[1, 4]).unwrap();
        let token_two = Tensor::from_f32_slice(&context, &[-0.3, 0.6, 0.2, 0.8], &[1, 4]).unwrap();
        let mut cache_f32 = LayerKvCache::new(&context, 2, 8, 2, DType::F32).unwrap();
        let mut cache_f16 = LayerKvCache::new(&context, 2, 8, 2, DType::F16).unwrap();

        attention
            .forward_with_cache(&context, &token_one, &mut cache_f32)
            .unwrap();
        attention_f16
            .forward_with_cache(
                &context,
                &token_one.to_dtype(&context, DType::F16).unwrap(),
                &mut cache_f16,
            )
            .unwrap();
        assert_eq!(cache_f32.len(), 1);
        assert_eq!(cache_f16.len(), 1);

        let expected = attention
            .forward_with_cache(&context, &token_two, &mut cache_f32)
            .unwrap();
        let actual = attention_f16
            .forward_with_cache(
                &context,
                &token_two.to_dtype(&context, DType::F16).unwrap(),
                &mut cache_f16,
            )
            .unwrap();

        assert_eq!(cache_f16.len(), 2);
        assert_eq!(actual.dtype(), DType::F16);
        assert_close(&expected, &actual, 0.1);
    }

    #[test]
    fn f16_self_attention_supports_prefill_then_decode() {
        let Some(context) = metal_context() else {
            return;
        };
        let attention_f16 = attention(&context).to_dtype(&context, DType::F16).unwrap();
        let prefill = Tensor::from_f32_slice(
            &context,
            &[0.1, 0.2, 0.3, 0.4, 0.5, -0.1, 0.2, 0.6, -0.2, 0.4, 0.8, 0.1],
            &[3, 4],
        )
        .unwrap()
        .to_dtype(&context, DType::F16)
        .unwrap();
        let decode = Tensor::from_f32_slice(&context, &[0.3, -0.5, 0.1, 0.9], &[1, 4])
            .unwrap()
            .to_dtype(&context, DType::F16)
            .unwrap();
        let mut cache = LayerKvCache::new(&context, 2, 8, 2, DType::F16).unwrap();

        attention_f16
            .forward_with_cache(&context, &prefill, &mut cache)
            .unwrap();
        assert_eq!(cache.len(), 3);

        let output = attention_f16
            .forward_with_cache(&context, &decode, &mut cache)
            .unwrap();
        assert_eq!(cache.len(), 4);
        assert_eq!(output.dtype(), DType::F16);
        assert_eq!(output.shape().dims(), &[1, 4]);
    }

    #[test]
    fn f16_self_attention_rejects_f32_input_and_cache() {
        let Some(context) = metal_context() else {
            return;
        };
        let attention_f16 = attention(&context).to_dtype(&context, DType::F16).unwrap();
        let input = Tensor::from_f32_slice(&context, &[0.1, 0.2, 0.3, 0.4], &[1, 4]).unwrap();
        let mut cache_f32 = LayerKvCache::new(&context, 2, 8, 2, DType::F32).unwrap();

        assert!(attention_f16.forward(&context, &input).is_err());
        let input_f16 = input.to_dtype(&context, DType::F16).unwrap();
        assert!(
            attention_f16
                .forward_with_cache(&context, &input_f16, &mut cache_f32)
                .is_err()
        );
    }
}
