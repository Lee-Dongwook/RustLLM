use crate::error::{Result, TinyError};

use crate::metal::MetalContext;
use crate::tensor::Tensor;

pub struct Linear {
    weight: Tensor,
    in_features: usize,
    out_features: usize,
}

impl Linear {
    pub fn new(weight: Tensor) -> Result<Self> {
        if weight.rank() != 2 {
            return Err(TinyError::InvalidDimension(format!(
                "Linear weight must have rank 2, got rank {}",
                weight.rank(),
            )));
        }

        let in_features = weight.dim(0)?;
        let out_features = weight.dim(1)?;

        Ok(Self {
            weight,
            in_features,
            out_features,
        })
    }

    pub fn in_features(&self) -> usize {
        self.in_features
    }

    pub fn out_features(&self) -> usize {
        self.out_features
    }

    pub fn weight(&self) -> &Tensor {
        &self.weight
    }

    pub fn forward(&self, context: &MetalContext, input: &Tensor) -> Result<Tensor> {
        if input.rank() != 2 {
            return Err(TinyError::InvalidDimension(format!(
                "Linear currently requires rank-2 input, got rank {}",
                input.rank(),
            )));
        }

        let input_features = input.dim(1)?;

        if input_features != self.in_features {
            return Err(TinyError::ShapeMismatch {
                left: input.shape().dims().to_vec(),

                right: self.weight.shape().dims().to_vec(),
            });
        }

        input.matmul(context, &self.weight)
    }
}
