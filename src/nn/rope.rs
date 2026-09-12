use crate::error::{Result, TinyError};

use crate::metal::MetalContext;
use crate::ops::rope;
use metal::CommandBufferRef;

use crate::tensor::{DType, Tensor};

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
        if head_dim == 0 || !head_dim.is_multiple_of(2) {
            return Err(TinyError::InvalidShape(format!(
                "RoPE head dimension must be positive and even, got {head_dim}"
            )));
        }

        if max_seq_len == 0 {
            return Err(TinyError::InvalidShape(
                "RoPE max sequence length cannot be zero".to_string(),
            ));
        }

        if theta <= 0.0 {
            return Err(TinyError::InvalidShape(format!(
                "RoPE theta must be positive, got {theta}"
            )));
        }

        let half_dim = head_dim / 2;

        let mut cos = Vec::with_capacity(max_seq_len * half_dim);

        let mut sin = Vec::with_capacity(max_seq_len * half_dim);

        for position in 0..max_seq_len {
            for i in 0..half_dim {
                let exponent = (2 * i) as f32 / head_dim as f32;

                let inverse_frequency = 1.0 / theta.powf(exponent);

                let angle = position as f32 * inverse_frequency;

                cos.push(angle.cos());

                sin.push(angle.sin());
            }
        }

        let cos_table = Tensor::from_f32_slice(context, &cos, &[max_seq_len, half_dim])?;

        let sin_table = Tensor::from_f32_slice(context, &sin, &[max_seq_len, half_dim])?;

        Ok(Self {
            cos_table,
            sin_table,
            head_dim,
            max_seq_len,
            theta,
        })
    }

    pub fn head_dim(&self) -> usize {
        self.head_dim
    }

    pub fn max_seq_len(&self) -> usize {
        self.max_seq_len
    }

    pub fn theta(&self) -> f32 {
        self.theta
    }

    pub fn forward(
        &self,
        context: &MetalContext,
        input: &Tensor,
        start_pos: usize,
    ) -> Result<Tensor> {
        if !matches!(input.dtype(), DType::F32 | DType::F16) {
            return Err(TinyError::UnsupportedDType(format!("{:?}", input.dtype(),)));
        }

        if input.rank() < 2 {
            return Err(TinyError::InvalidDimension(format!(
                "RoPE expects [..., sequence, head_dim], got rank {}",
                input.rank(),
            )));
        }

        let seq_len = input.dim(input.rank() - 2)?;

        let input_head_dim = input.dim(input.rank() - 1)?;

        if input_head_dim != self.head_dim {
            return Err(TinyError::InvalidShape(format!(
                "RoPE expected head dimension {}, got {}",
                self.head_dim, input_head_dim,
            )));
        }

        let end_pos = start_pos
            .checked_add(seq_len)
            .ok_or_else(|| TinyError::InvalidShape("RoPE position overflow".to_string()))?;

        if end_pos > self.max_seq_len {
            return Err(TinyError::PositionOutOfRange {
                start_pos,
                seq_len,
                max_seq_len: self.max_seq_len,
            });
        }

        rope(context, input, &self.cos_table, &self.sin_table, start_pos)
    }

    pub(crate) fn forward_encode(
        &self,
        context: &MetalContext,
        command_buffer: &CommandBufferRef,
        input: &Tensor,
        start_pos: usize,
    ) -> Result<Tensor> {
        if !matches!(input.dtype(), DType::F32 | DType::F16) {
            return Err(TinyError::UnsupportedDType(format!("{:?}", input.dtype())));
        }

        crate::ops::rope_encode(
            context,
            command_buffer,
            input,
            &self.cos_table,
            &self.sin_table,
            start_pos,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::RotaryEmbedding;
    use crate::{
        error::TinyError,
        metal::MetalContext,
        tensor::{DType, Tensor},
    };

    fn metal_context() -> Option<MetalContext> {
        match MetalContext::new() {
            Ok(context) => Some(context),
            Err(TinyError::Metal(message)) => {
                eprintln!("skipping Metal RoPE test: {message}");
                None
            }
            Err(error) => panic!("failed to create Metal context: {error}"),
        }
    }

    fn assert_close(actual: &[f32], expected: &[f32], tolerance: f32) {
        assert_eq!(actual.len(), expected.len());
        for (index, (actual, expected)) in actual.iter().zip(expected).enumerate() {
            let error = (actual - expected).abs();
            assert!(
                error <= tolerance,
                "value {index} differs by {error}: actual={actual}, expected={expected}",
            );
        }
    }

    #[test]
    fn f16_rope_stays_close_to_f32_and_returns_f16() {
        let Some(context) = metal_context() else {
            return;
        };
        let rope = RotaryEmbedding::new(&context, 4, 8, 10_000.0).unwrap();
        let input = Tensor::from_f32_slice(
            &context,
            &[0.125, -1.25, 2.5, 3.75, -0.5, 1.5, -2.25, 4.0],
            &[1, 1, 2, 4],
        )
        .unwrap();

        let output_f32 = rope
            .forward(&context, &input, 1)
            .unwrap()
            .to_f32_vec()
            .unwrap();
        let input_f16 = input.to_dtype(&context, DType::F16).unwrap();
        let output_f16 = rope.forward(&context, &input_f16, 1).unwrap();

        assert_eq!(output_f16.dtype(), DType::F16);
        assert_close(&output_f16.to_f32_vec().unwrap(), &output_f32, 0.01);
    }

    #[test]
    fn f16_rope_materializes_non_contiguous_input() {
        let Some(context) = metal_context() else {
            return;
        };
        let rope = RotaryEmbedding::new(&context, 4, 8, 10_000.0).unwrap();
        let input = Tensor::from_f32_slice(
            &context,
            &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
            &[1, 2, 1, 4],
        )
        .unwrap()
        .to_dtype(&context, DType::F16)
        .unwrap()
        .permute(&[0, 2, 1, 3])
        .unwrap();

        assert!(!input.is_contiguous());

        let output = rope.forward(&context, &input, 0).unwrap();

        assert_eq!(output.dtype(), DType::F16);
        assert!(output.is_contiguous());
    }
}
