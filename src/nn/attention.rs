use crate::error::{
    Result,
    TinyError,
};

use crate::metal::MetalContext;
use crate::tensor::Tensor;

use super::{
    Linear,
    RotaryEmbedding,
};

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
            return Err(
                TinyError::InvalidShape(
                    "attention num_heads cannot be zero"
                        .to_string(),
                ),
            );
        }

        let hidden_size =
            q_proj.in_features();

        if q_proj.out_features()
            != hidden_size
        {
            return Err(
                TinyError::InvalidShape(
                    format!(
                        "q_proj output size must equal hidden size {hidden_size}"
                    ),
                ),
            );
        }

        if k_proj.in_features()
            != hidden_size
            || k_proj.out_features()
                != hidden_size
        {
            return Err(
                TinyError::InvalidShape(
                    "k_proj shape must match hidden size"
                        .to_string(),
                ),
            );
        }

        if v_proj.in_features()
            != hidden_size
            || v_proj.out_features()
                != hidden_size
        {
            return Err(
                TinyError::InvalidShape(
                    "v_proj shape must match hidden size"
                        .to_string(),
                ),
            );
        }

        if out_proj.in_features()
            != hidden_size
            || out_proj.out_features()
                != hidden_size
        {
            return Err(
                TinyError::InvalidShape(
                    "out_proj shape must match hidden size"
                        .to_string(),
                ),
            );
        }

        if hidden_size
            % num_heads
            != 0
        {
            return Err(
                TinyError::InvalidShape(
                    format!(
                        "hidden size {hidden_size} must be divisible by num_heads {num_heads}"
                    ),
                ),
            );
        }

        let head_dim =
            hidden_size
            / num_heads;

        if rope.head_dim()
            != head_dim
        {
            return Err(
                TinyError::InvalidShape(
                    format!(
                        "RoPE head dimension {} does not match attention head dimension {head_dim}",
                        rope.head_dim(),
                    ),
                ),
            );
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

    pub fn hidden_size(
        &self,
    ) -> usize {
        self.hidden_size
    }

    pub fn num_heads(
        &self,
    ) -> usize {
        self.num_heads
    }

    pub fn head_dim(
        &self,
    ) -> usize {
        self.head_dim
    }

    pub fn forward(
        &self,
        context: &MetalContext,
        input: &Tensor,
    ) -> Result<Tensor> {
        // 현재 v0.1 Attention은
        //
        // [sequence, hidden]
        //
        // 형태를 받는다.
        if input.rank() != 2 {
            return Err(
                TinyError::InvalidDimension(
                    format!(
                        "SelfAttention currently expects [sequence, hidden], got shape {:?}",
                        input.shape().dims(),
                    ),
                ),
            );
        }

        let seq_len =
            input.dim(0)?;

        let hidden_size =
            input.dim(1)?;

        if hidden_size
            != self.hidden_size
        {
            return Err(
                TinyError::InvalidShape(
                    format!(
                        "attention expected hidden size {}, got {}",
                        self.hidden_size,
                        hidden_size,
                    ),
                ),
            );
        }

        // -----------------------------------------------
        // 1. Q / K / V projection
        // -----------------------------------------------

        let q =
            self.q_proj.forward(
                context,
                input,
            )?;

        let k =
            self.k_proj.forward(
                context,
                input,
            )?;

        let v =
            self.v_proj.forward(
                context,
                input,
            )?;

        // 현재:
        //
        // [sequence, hidden]
        //
        // ↓
        //
        // [1, sequence, heads, head_dim]

        let q =
            q.reshape(
                &[
                    1,
                    seq_len,
                    self.num_heads,
                    self.head_dim,
                ],
            )?;

        let k =
            k.reshape(
                &[
                    1,
                    seq_len,
                    self.num_heads,
                    self.head_dim,
                ],
            )?;

        let v =
            v.reshape(
                &[
                    1,
                    seq_len,
                    self.num_heads,
                    self.head_dim,
                ],
            )?;

        // -----------------------------------------------
        // 2. Head를 앞으로 이동
        //
        // [B,S,H,D]
        //
        // →
        //
        // [B,H,S,D]
        // -----------------------------------------------

        let q =
            q.permute(
                &[0, 2, 1, 3],
            )?;

        let k =
            k.permute(
                &[0, 2, 1, 3],
            )?;

        let v =
            v.permute(
                &[0, 2, 1, 3],
            )?;

        // -----------------------------------------------
        // 3. RoPE
        // -----------------------------------------------

        let q =
            self.rope.forward(
                context,
                &q,
                0,
            )?;

        let k =
            self.rope.forward(
                context,
                &k,
                0,
            )?;

        // -----------------------------------------------
        // 4. K transpose
        //
        // K
        // [B,H,S,D]
        //
        // →
        //
        // K^T
        // [B,H,D,S]
        // -----------------------------------------------

        let k_t =
            k.transpose(
                2,
                3,
            )?;

        // -----------------------------------------------
        // 5. Q × Kᵀ
        //
        // [B,H,S,D]
        // ×
        // [B,H,D,S]
        //
        // →
        //
        // [B,H,S,S]
        // -----------------------------------------------

        let scores =
            q.batched_matmul(
                context,
                &k_t,
            )?;

        // -----------------------------------------------
        // 6. Scale + Causal Mask
        // -----------------------------------------------

        let scale =
            1.0
            / (self.head_dim as f32)
                .sqrt();

        let scores =
            scores.attention_scale_mask(
                context,
                scale,
                0,
            )?;

        // -----------------------------------------------
        // 7. Softmax
        // -----------------------------------------------

        let probabilities =
            scores.softmax_last_dim(
                context,
            )?;

        // -----------------------------------------------
        // 8. Attention probabilities × V
        //
        // [B,H,S,S]
        // ×
        // [B,H,S,D]
        //
        // →
        //
        // [B,H,S,D]
        // -----------------------------------------------

        let context_tensor =
            probabilities
                .batched_matmul(
                    context,
                    &v,
                )?;

        // -----------------------------------------------
        // 9. Heads 다시 합치기
        //
        // [B,H,S,D]
        //
        // →
        //
        // [B,S,H,D]
        // -----------------------------------------------

        let context_tensor =
            context_tensor
                .permute(
                    &[0, 2, 1, 3],
                )?;

        // permute 결과는 non-contiguous다.
        let context_tensor =
            context_tensor
                .contiguous(
                    context,
                )?;

        // -----------------------------------------------
        // 10. [B,S,H,D] → [S,hidden]
        //
        // 현재 batch=1
        // -----------------------------------------------

        let context_tensor =
            context_tensor
                .reshape(
                    &[
                        seq_len,
                        self.hidden_size,
                    ],
                )?;

        // -----------------------------------------------
        // 11. Output Projection
        // -----------------------------------------------

        self.out_proj.forward(
            context,
            &context_tensor,
        )
    }
}
