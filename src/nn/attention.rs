use crate::error::{Result, TinyError};

use crate::metal::MetalContext;
use crate::model::LayerKvCache;
use crate::tensor::Tensor;

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

    pub fn forward(&self, context: &MetalContext, input: &Tensor) -> Result<Tensor> {
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
}
