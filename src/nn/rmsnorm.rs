use crate::error::{Result, TinyError};

use crate::metal::MetalContext;

use crate::ops::rmsnorm_f32;

use crate::tensor::{DType, Tensor};

pub struct RmsNorm {
    weight: Tensor,
    hidden_size: usize,
    epsilon: f32,
}

impl RmsNorm {
    pub fn new(weight: Tensor, epsilon: f32) -> Result<Self> {
        if weight.rank() != 1 {
            return Err(TinyError::InvalidDimension(format!(
                "RMSNorm weight must have rank 1, got rank {}",
                weight.rank(),
            )));
        }

        if weight.dtype() != DType::F32 {
            return Err(TinyError::UnsupportedDType(
                format!("{:?}", weight.dtype(),),
            ));
        }

        if epsilon <= 0.0 {
            return Err(TinyError::InvalidShape(format!(
                "RMSNorm epsilon must be positive, got {epsilon}"
            )));
        }

        let hidden_size = weight.dim(0)?;

        Ok(Self {
            weight,
            hidden_size,
            epsilon,
        })
    }

    pub fn hidden_size(&self) -> usize {
        self.hidden_size
    }

    pub fn epsilon(&self) -> f32 {
        self.epsilon
    }

    pub fn weight(&self) -> &Tensor {
        &self.weight
    }

    pub fn forward(&self, context: &MetalContext, input: &Tensor) -> Result<Tensor> {
        if input.dtype() != DType::F32 {
            return Err(TinyError::UnsupportedDType(format!("{:?}", input.dtype(),)));
        }

        if input.rank() == 0 {
            return Err(TinyError::InvalidDimension(
                "RMSNorm input must have at least one dimension".to_string(),
            ));
        }

        let input_hidden_size = input.dim(input.rank() - 1)?;

        if input_hidden_size != self.hidden_size {
            return Err(TinyError::ShapeMismatch {
                left: input.shape().dims().to_vec(),

                right: self.weight.shape().dims().to_vec(),
            });
        }

        let input = if input.is_contiguous() {
            input.clone()
        } else {
            input.contiguous(context)?
        };

        let weight = if self.weight.is_contiguous() {
            self.weight.clone()
        } else {
            self.weight.contiguous(context)?
        };

        let rows = input.numel() / self.hidden_size;

        let output = rmsnorm_f32(
            context,
            input.metal_buffer()?,
            weight.metal_buffer()?,
            rows,
            self.hidden_size,
            self.epsilon,
        )?;

        Tensor::from_metal_buffer(output, input.shape().dims(), DType::F32)
    }
}
