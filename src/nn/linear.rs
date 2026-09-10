use crate::error::{Result, TinyError};

use crate::metal::MetalContext;
use crate::tensor::{DType, Tensor};

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
            return Err(TinyError::ModelFormat(format!(
                "Linear expects rank-2 input, got {:?}",
                input.shape().dims(),
            )));
        }

        if input.dim(1)? != self.in_features {
            return Err(TinyError::ModelFormat(format!(
                "Linear input feature mismatch: expected {}, got {}",
                self.in_features,
                input.dim(1)?,
            )));
        }

        if input.dtype() != self.weight.dtype() {
            return Err(TinyError::UnsupportedDType(format!(
                "Linear input dtype {:?} does not match weight dtype {:?}",
                input.dtype(),
                self.weight.dtype(),
            )));
        }

        input.matmul(context, &self.weight)
    }

    pub fn dtype(&self) -> DType {
        self.weight.dtype()
    }

    pub fn to_dtype(&self, context: &MetalContext, dtype: DType) -> Result<Self> {
        let weight = self.weight.to_dtype(context, dtype)?;

        Ok(Self {
            weight,
            in_features: self.in_features,
            out_features: self.out_features,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::Linear;
    use crate::error::TinyError;
    use crate::metal::MetalContext;
    use crate::tensor::{DType, Tensor};

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

    fn metal_context() -> Option<MetalContext> {
        match MetalContext::new() {
            Ok(context) => Some(context),
            Err(TinyError::Metal(message)) => {
                eprintln!("skipping Metal Linear test: {message}");
                None
            }
            Err(error) => panic!("failed to create Metal context: {error}"),
        }
    }

    #[test]
    fn f16_linear_reuses_converted_weight_and_returns_f16_output() {
        let Some(context) = metal_context() else {
            return;
        };
        let input = Tensor::from_f32_slice(&context, &[1.0, 2.0, 3.0, 4.0], &[2, 2]).unwrap();
        let weight = Tensor::from_f32_slice(&context, &[1.0, 2.0, 3.0, 4.0], &[2, 2]).unwrap();
        let linear = Linear::new(weight).unwrap();

        let output_f32 = linear.forward(&context, &input).unwrap();
        assert_eq!(output_f32.dtype(), DType::F32);
        assert_close(
            &output_f32.to_f32_vec().unwrap(),
            &[7.0, 10.0, 15.0, 22.0],
            1e-6,
        );

        let linear_f16 = linear.to_dtype(&context, DType::F16).unwrap();
        let input_f16 = input.to_dtype(&context, DType::F16).unwrap();

        assert_eq!(linear_f16.dtype(), DType::F16);
        assert_eq!(linear_f16.weight().dtype(), DType::F16);
        assert_eq!(input_f16.dtype(), DType::F16);

        let output_f16 = linear_f16.forward(&context, &input_f16).unwrap();
        assert_eq!(output_f16.dtype(), DType::F16);
        assert_close(
            &output_f16.to_f32_vec().unwrap(),
            &[7.0, 10.0, 15.0, 22.0],
            0.01,
        );
    }

    #[test]
    fn f16_linear_stays_close_to_f32_for_fractional_values() {
        let Some(context) = metal_context() else {
            return;
        };
        let input =
            Tensor::from_f32_slice(&context, &[0.1234, 1.2345, 2.3456, 3.4567], &[2, 2]).unwrap();
        let weight = Tensor::from_f32_slice(&context, &[1.0, 2.0, 3.0, 4.0], &[2, 2]).unwrap();
        let linear = Linear::new(weight).unwrap();

        let output_f32 = linear
            .forward(&context, &input)
            .unwrap()
            .to_f32_vec()
            .unwrap();
        let linear_f16 = linear.to_dtype(&context, DType::F16).unwrap();
        let input_f16 = input.to_dtype(&context, DType::F16).unwrap();
        let output_f16 = linear_f16
            .forward(&context, &input_f16)
            .unwrap()
            .to_f32_vec()
            .unwrap();

        assert_close(&output_f16, &output_f32, 0.01);
    }

    #[test]
    fn f16_linear_rejects_f32_input_without_implicit_cast() {
        let Some(context) = metal_context() else {
            return;
        };
        let input = Tensor::from_f32_slice(&context, &[1.0, 2.0, 3.0, 4.0], &[2, 2]).unwrap();
        let weight = Tensor::from_f32_slice(&context, &[1.0, 2.0, 3.0, 4.0], &[2, 2]).unwrap();
        let linear_f16 = Linear::new(weight)
            .unwrap()
            .to_dtype(&context, DType::F16)
            .unwrap();

        let error = match linear_f16.forward(&context, &input) {
            Ok(_) => panic!("F16 Linear must not implicitly cast an F32 input"),
            Err(error) => error,
        };

        assert!(
            matches!(error, TinyError::UnsupportedDType(message) if message.contains("input dtype F32 does not match weight dtype F16"))
        );
    }
}
