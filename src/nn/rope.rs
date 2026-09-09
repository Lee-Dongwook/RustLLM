use crate::error::{
    Result,
    TinyError,
};

use crate::metal::MetalContext;
use crate::ops::rope_f32;

use crate::tensor::{
    DType,
    Tensor,
};

#[derive(Clone)]
pub struct RotaryEmbedding {
    cos_table: Tensor,
    sin_table: Tensor,

    head_dim: usize,
    max_seq_len: usize,
    theta: f32,
}

impl RotaryEmbedding {
    pub fn new(
        context: &MetalContext,
        head_dim: usize,
        max_seq_len: usize,
        theta: f32,
    ) -> Result<Self> {
        if head_dim == 0
            || head_dim % 2 != 0
        {
            return Err(
                TinyError::InvalidShape(
                    format!(
                        "RoPE head dimension must be positive and even, got {head_dim}"
                    ),
                ),
            );
        }

        if max_seq_len == 0 {
            return Err(
                TinyError::InvalidShape(
                    "RoPE max sequence length cannot be zero"
                        .to_string(),
                ),
            );
        }

        if theta <= 0.0 {
            return Err(
                TinyError::InvalidShape(
                    format!(
                        "RoPE theta must be positive, got {theta}"
                    ),
                ),
            );
        }

        let half_dim =
            head_dim / 2;

        let mut cos =
            Vec::with_capacity(
                max_seq_len * half_dim,
            );

        let mut sin =
            Vec::with_capacity(
                max_seq_len * half_dim,
            );

        for position in 0..max_seq_len {
            for i in 0..half_dim {
                let exponent =
                    (2 * i) as f32
                    / head_dim as f32;

                let inverse_frequency =
                    1.0
                    / theta.powf(exponent);

                let angle =
                    position as f32
                    * inverse_frequency;

                cos.push(
                    angle.cos(),
                );

                sin.push(
                    angle.sin(),
                );
            }
        }

        let cos_table =
            Tensor::from_f32_slice(
                context,
                &cos,
                &[
                    max_seq_len,
                    half_dim,
                ],
            )?;

        let sin_table =
            Tensor::from_f32_slice(
                context,
                &sin,
                &[
                    max_seq_len,
                    half_dim,
                ],
            )?;

        Ok(Self {
            cos_table,
            sin_table,
            head_dim,
            max_seq_len,
            theta,
        })
    }

    pub fn head_dim(
        &self,
    ) -> usize {
        self.head_dim
    }

    pub fn max_seq_len(
        &self,
    ) -> usize {
        self.max_seq_len
    }

    pub fn theta(
        &self,
    ) -> f32 {
        self.theta
    }

    pub fn forward(
        &self,
        context: &MetalContext,
        input: &Tensor,
        start_pos: usize,
    ) -> Result<Tensor> {
        if input.dtype()
            != DType::F32
        {
            return Err(
                TinyError::UnsupportedDType(
                    format!(
                        "{:?}",
                        input.dtype(),
                    ),
                ),
            );
        }

        if input.rank() < 2 {
            return Err(
                TinyError::InvalidDimension(
                    format!(
                        "RoPE expects [..., sequence, head_dim], got rank {}",
                        input.rank(),
                    ),
                ),
            );
        }

        let seq_len =
            input.dim(
                input.rank() - 2,
            )?;

        let input_head_dim =
            input.dim(
                input.rank() - 1,
            )?;

        if input_head_dim
            != self.head_dim
        {
            return Err(
                TinyError::InvalidShape(
                    format!(
                        "RoPE expected head dimension {}, got {}",
                        self.head_dim,
                        input_head_dim,
                    ),
                ),
            );
        }

        let end_pos =
            start_pos
                .checked_add(seq_len)
                .ok_or_else(|| {
                    TinyError::InvalidShape(
                        "RoPE position overflow"
                            .to_string(),
                    )
                })?;

        if end_pos
            > self.max_seq_len
        {
            return Err(
                TinyError::PositionOutOfRange {
                    start_pos,
                    seq_len,
                    max_seq_len:
                        self.max_seq_len,
                },
            );
        }

        let input =
            if input.is_contiguous() {
                input.clone()
            } else {
                input.contiguous(
                    context,
                )?
            };

        let output =
            rope_f32(
                context,
                input.metal_buffer()?,
                self.cos_table
                    .metal_buffer()?,
                self.sin_table
                    .metal_buffer()?,
                seq_len,
                self.head_dim,
                start_pos,
            )?;

        Tensor::from_metal_buffer(
            output,
            input.shape().dims(),
            DType::F32,
        )
    }
}
