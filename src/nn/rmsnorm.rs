use crate::error::{Result, TinyError};

use crate::metal::MetalContext;

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

        if !matches!(weight.dtype(), DType::F32 | DType::F16) {
            return Err(TinyError::UnsupportedDType(
                format!("{:?}", weight.dtype(),),
            ));
        }

        if !epsilon.is_finite() || epsilon <= 0.0 {
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
        if input.dtype() != self.weight.dtype() {
            return Err(TinyError::UnsupportedDType(format!(
                "RMSNorm input dtype {:?} does not match weight dtype {:?}",
                input.dtype(),
                self.weight.dtype(),
            )));
        }

        crate::ops::rmsnorm(context, input, &self.weight, self.epsilon)
    }

    pub fn to_dtype(&self, context: &MetalContext, dtype: DType) -> Result<Self> {
        Ok(Self {
            weight: self.weight.to_dtype(context, dtype)?,
            hidden_size: self.hidden_size,
            // Epsilon is passed to the shader as F32 for numerical stability.
            epsilon: self.epsilon,
        })
    }

    pub fn dtype(&self) -> DType {
        self.weight.dtype()
    }
}

#[cfg(test)]
mod tests {
    use super::RmsNorm;
    use crate::{
        error::TinyError,
        metal::MetalContext,
        tensor::{DType, Tensor},
    };

    fn metal_context() -> Option<MetalContext> {
        match MetalContext::new() {
            Ok(context) => Some(context),
            Err(TinyError::Metal(message)) => {
                eprintln!("skipping Metal RMSNorm test: {message}");
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
    fn f16_rmsnorm_stays_close_to_f32_and_returns_f16() {
        let Some(context) = metal_context() else {
            return;
        };
        let input = Tensor::from_f32_slice(
            &context,
            &[0.125, -1.25, 2.5, 3.75, -0.5, 1.5, -2.25, 4.0],
            &[2, 4],
        )
        .unwrap();
        let weight = Tensor::from_f32_slice(&context, &[1.0, 0.5, 1.25, 0.75], &[4]).unwrap();
        let norm = RmsNorm::new(weight, 1e-5).unwrap();

        let output_f32 = norm
            .forward(&context, &input)
            .unwrap()
            .to_f32_vec()
            .unwrap();
        let norm_f16 = norm.to_dtype(&context, DType::F16).unwrap();
        let input_f16 = input.to_dtype(&context, DType::F16).unwrap();
        let output_f16 = norm_f16.forward(&context, &input_f16).unwrap();

        assert_eq!(norm_f16.dtype(), DType::F16);
        assert_eq!(output_f16.dtype(), DType::F16);
        assert_close(&output_f16.to_f32_vec().unwrap(), &output_f32, 0.01);
    }
}
