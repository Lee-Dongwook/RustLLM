use crate::error::{Result, TinyError};

use crate::metal::{MetalContext, MetalExecution};
use crate::model::LayerKvCache;
use crate::profile::DecodeProfile;
use crate::tensor::{DType, Tensor};
use std::time::Instant;
use ::metal::CommandBufferRef;

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

    num_kv_heads: usize,
}

impl SelfAttention {
    pub fn new(
        q_proj: Linear,
        k_proj: Linear,
        v_proj: Linear,
        out_proj: Linear,
        rope: RotaryEmbedding,
        num_heads: usize,
        num_kv_heads: usize,
    ) -> Result<Self> {
        if num_heads == 0 {
            return Err(TinyError::InvalidShape(
                "attention num_heads cannot be zero".to_string(),
            ));
        }

        if num_kv_heads == 0 {
            return Err(TinyError::InvalidShape(
                "attention num_kv_heads cannot be zero".to_string(),
            ));
        }

        let hidden_size = q_proj.in_features();

        if q_proj.out_features() != hidden_size {
            return Err(TinyError::InvalidShape(format!(
                "q_proj output size must equal hidden size {hidden_size}"
            )));
        }

        if !hidden_size.is_multiple_of(num_heads) {
            return Err(TinyError::InvalidShape(format!(
                "hidden size {hidden_size} must be divisible by num_heads {num_heads}"
            )));
        }

        let head_dim = hidden_size / num_heads;
        let q_proj_size = num_heads * head_dim;
        let kv_proj_size = num_kv_heads * head_dim;

        if k_proj.in_features() != hidden_size || k_proj.out_features() != kv_proj_size {
            return Err(TinyError::InvalidShape(format!(
                "k_proj shape must be [{hidden_size}, {kv_proj_size}]"
            )));
        }

        if v_proj.in_features() != hidden_size || v_proj.out_features() != kv_proj_size {
            return Err(TinyError::InvalidShape(format!(
                "v_proj shape must be [{hidden_size}, {kv_proj_size}]"
            )));
        }

        if out_proj.in_features() != q_proj_size || out_proj.out_features() != hidden_size {
            return Err(TinyError::InvalidShape(format!(
                "out_proj shape must be [{q_proj_size}, {hidden_size}]"
            )));
        }

        if !num_heads.is_multiple_of(num_kv_heads) {
            return Err(TinyError::InvalidShape(format!(
                "attention num_heads {num_heads} must be divisible by num_kv_heads {num_kv_heads}"
            )));
        }

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
            num_kv_heads,
        })
    }

    pub fn hidden_size(&self) -> usize {
        self.hidden_size
    }

    pub fn num_heads(&self) -> usize {
        self.num_heads
    }

    pub fn num_kv_heads(&self) -> usize {
        self.num_kv_heads
    }

    pub fn head_dim(&self) -> usize {
        self.head_dim
    }

    pub fn dtype(&self) -> DType {
        self.q_proj.dtype()
    }

    pub fn num_kv_groups(&self) -> usize {
        self.num_heads / self.num_kv_heads
    }

    pub fn to_dtype(&self, context: &MetalContext, dtype: DType) -> Result<Self> {
        Self::new(
            self.q_proj.to_dtype(context, dtype)?,
            self.k_proj.to_dtype(context, dtype)?,
            self.v_proj.to_dtype(context, dtype)?,
            self.out_proj.to_dtype(context, dtype)?,
            self.rope.clone(),
            self.num_heads,
            self.num_kv_heads,
        )
    }

    pub fn forward(&self, context: &MetalContext, input: &Tensor) -> Result<Tensor> {
        self.validate_input_dtype(input)?;

        let mut cache = LayerKvCache::new(
            context,
            self.num_kv_heads,
            self.rope.max_seq_len(),
            self.head_dim,
            input.dtype(),
        )?;
        self.forward_with_cache(context, input, &mut cache)
    }

    fn project_qkv_encode(
        &self,
        context: &MetalContext,
        command_buffer: &CommandBufferRef,
        input: &Tensor,
    ) -> Result<(Tensor, Tensor, Tensor)> {
        let q = self.q_proj.forward_encode(context, command_buffer, input)?;

        let k = self.k_proj.forward_encode(context, command_buffer, input)?;

        let v = self.v_proj.forward_encode(context, command_buffer, input)?;

        Ok((q, k, v))
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

        let execution = MetalExecution::new(context);
        let (q, k, v) =
        self.project_qkv_encode(
            context,
            execution.command_buffer(),
            x,
        )?;
        execution.finish();
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

        let q = self.prepare_heads(&q, seq_len, self.num_heads)?;
        let k = self.prepare_heads(&k, seq_len, self.num_kv_heads)?;
        let v = self.prepare_heads(&v, seq_len, self.num_kv_heads)?;

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

        let is_gqa = self.num_heads != self.num_kv_heads;
        let scores = if is_gqa {
            q.gqa_qk_matmul(context, &all_k)?
        } else {
            let k_transposed = all_k.transpose(2, 3)?;
            q.batched_matmul(context, &k_transposed)?
        };

        let scale = 1.0f32 / (self.head_dim as f32).sqrt();

        let scores = scores.attention_scale_mask(context, scale, start_pos)?;

        let probabilities = scores.softmax_last_dim(context)?;

        let attention = if is_gqa {
            probabilities.gqa_pv_matmul(context, &all_v)?
        } else {
            probabilities.batched_matmul(context, &all_v)?
        };
        let attention = attention.permute(&[0, 2, 1, 3])?.contiguous(context)?;

        let attention = attention.reshape(&[seq_len, self.hidden_size])?;

        self.out_proj.forward(context, &attention)
    }

    pub fn forward_with_cache_profiled(
        &self,
        context: &MetalContext,
        x: &Tensor,
        cache: &mut LayerKvCache,
        profile: &mut DecodeProfile,
    ) -> Result<Tensor> {
        self.validate_input_dtype(x)?;
        if cache.dtype() != self.dtype() || x.rank() != 2 || x.dim(1)? != self.hidden_size {
            return self.forward_with_cache(context, x, cache);
        }
        let seq_len = x.dim(0)?;
        let start_pos = cache.len();

        let started = Instant::now();
        let execution = MetalExecution::new(context);
        let (q, k, v) = self.project_qkv_encode(context, execution.command_buffer(), x)?;
        
        execution.finish();
        profile.qkv_projection += started.elapsed();

        let started = Instant::now();
        let q = self.prepare_heads(&q, seq_len, self.num_heads)?;
        let k = self.prepare_heads(&k, seq_len, self.num_kv_heads)?;
        let v = self.prepare_heads(&v, seq_len, self.num_kv_heads)?;
        let q = self.rope.forward(context, &q, start_pos)?;
        let k = self.rope.forward(context, &k, start_pos)?;
        cache.append(context, k, v)?;
        let all_k = cache.key()?;
        let all_v = cache.value()?;
        let scores = if self.num_heads != self.num_kv_heads {
            q.gqa_qk_matmul(context, &all_k)?
        } else {
            q.batched_matmul(context, &all_k.transpose(2, 3)?)?
        };
        let scores = scores.attention_scale_mask(
            context,
            1.0f32 / (self.head_dim as f32).sqrt(),
            start_pos,
        )?;
        let probabilities = scores.softmax_last_dim(context)?;
        let attention = if self.num_heads != self.num_kv_heads {
            probabilities.gqa_pv_matmul(context, &all_v)?
        } else {
            probabilities.batched_matmul(context, &all_v)?
        };
        let attention = attention.permute(&[0, 2, 1, 3])?.contiguous(context)?;
        let attention = attention.reshape(&[seq_len, self.hidden_size])?;
        profile.attention += started.elapsed();

        let started = Instant::now();
        let output = self.out_proj.forward(context, &attention)?;
        profile.output_projection += started.elapsed();
        Ok(output)
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

    fn prepare_heads(&self, tensor: &Tensor, seq_len: usize, num_heads: usize) -> Result<Tensor> {
        tensor
            .reshape(&[1, seq_len, num_heads, self.head_dim])?
            .permute(&[0, 2, 1, 3])
    }
}

#[cfg(test)]
mod tests {
    use super::{Linear, RotaryEmbedding, SelfAttention};
    use crate::{
        error::TinyError,
        metal::{MetalExecution, MetalContext},
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
    fn gqa_projection_and_head_layout_use_kv_head_count() {
        let Some(context) = metal_context() else {
            return;
        };

        let linear = |out_features| {
            Linear::new(
                Tensor::from_f32_slice(
                    &context,
                    &vec![0.0; 12 * out_features],
                    &[12, out_features],
                )
                .unwrap(),
            )
            .unwrap()
        };
        let attention = SelfAttention::new(
            linear(12),
            linear(4),
            linear(4),
            linear(12),
            RotaryEmbedding::new(&context, 4, 32, 10_000.0).unwrap(),
            3,
            1,
        )
        .unwrap();
        let input = Tensor::from_f32_slice(&context, &[0.0; 24], &[2, 12]).unwrap();

        let q = attention.q_proj.forward(&context, &input).unwrap();
        let k = attention.k_proj.forward(&context, &input).unwrap();
        let v = attention.v_proj.forward(&context, &input).unwrap();

        assert_eq!(q.shape().dims(), &[2, 12]);
        assert_eq!(k.shape().dims(), &[2, 4]);
        assert_eq!(v.shape().dims(), &[2, 4]);
        assert_eq!(attention.num_heads(), 3);
        assert_eq!(attention.num_kv_heads(), 1);
        assert_eq!(attention.num_kv_groups(), 3);
        assert_eq!(
            attention
                .prepare_heads(&q, 2, attention.num_heads())
                .unwrap()
                .shape()
                .dims(),
            &[1, 3, 2, 4]
        );
        assert_eq!(
            attention
                .prepare_heads(&k, 2, attention.num_kv_heads())
                .unwrap()
                .shape()
                .dims(),
            &[1, 1, 2, 4]
        );
        assert_eq!(
            attention
                .prepare_heads(&v, 2, attention.num_kv_heads())
                .unwrap()
                .shape()
                .dims(),
            &[1, 1, 2, 4]
        );
    }

    #[test]
    fn gqa_forward_uses_compact_cached_kv_heads() {
        let Some(context) = metal_context() else {
            return;
        };

        let q_and_out = Tensor::from_f32_slice(&context, &vec![0.0; 12 * 12], &[12, 12]).unwrap();
        let kv = Tensor::from_f32_slice(&context, &vec![0.0; 12 * 4], &[12, 4]).unwrap();
        let attention = SelfAttention::new(
            Linear::new(q_and_out.clone()).unwrap(),
            Linear::new(kv.clone()).unwrap(),
            Linear::new(kv).unwrap(),
            Linear::new(q_and_out).unwrap(),
            RotaryEmbedding::new(&context, 4, 8, 10_000.0).unwrap(),
            3,
            1,
        )
        .unwrap();
        let input = Tensor::from_f32_slice(&context, &[0.0; 24], &[2, 12]).unwrap();
        let mut cache = LayerKvCache::new(&context, 1, 8, 4, DType::F32).unwrap();

        let output = attention
            .forward_with_cache(&context, &input, &mut cache)
            .unwrap();

        assert_eq!(cache.key().unwrap().shape().dims(), &[1, 1, 2, 4]);
        assert_eq!(output.shape().dims(), &[2, 12]);
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

    #[test]
fn batched_qkv_projection_matches_sync_path() {
    let Some(context) =
        metal_context()
    else {
        return;
    };

    /*
     * F16으로 테스트.
     *
     * Linear::forward_encode()의
     * Dense F16 path도 검증할 수 있다.
     */
    let attention =
        attention(&context)
            .to_dtype(
                &context,
                DType::F16,
            )
            .unwrap();

    let input =
        Tensor::from_f32_slice(
            &context,
            &[
                0.1,
                0.2,
                0.3,
                0.4,
                -0.2,
                0.5,
                0.7,
                -0.1,
            ],
            &[2, 4],
        )
        .unwrap()
        .to_dtype(
            &context,
            DType::F16,
        )
        .unwrap();

    /*
     * 기존 synchronous reference
     */
    let expected_q =
        attention
            .q_proj
            .forward(
                &context,
                &input,
            )
            .unwrap();

    let expected_k =
        attention
            .k_proj
            .forward(
                &context,
                &input,
            )
            .unwrap();

    let expected_v =
        attention
            .v_proj
            .forward(
                &context,
                &input,
            )
            .unwrap();

    /*
     * batched path
     */
    let execution =
        MetalExecution::new(
            &context,
        );

    let (actual_q, actual_k, actual_v) =
        attention
            .project_qkv_encode(
                &context,
                execution.command_buffer(),
                &input,
            )
            .unwrap();

    /*
     * Q/K/V 세 개를 encode하고
     * 여기서 단 한 번 commit/wait.
     */
    execution.finish();

    assert_close(
        &expected_q,
        &actual_q,
        0.01,
    );

    assert_close(
        &expected_k,
        &actual_k,
        0.01,
    );

    assert_close(
        &expected_v,
        &actual_v,
        0.01,
    );
}
}
