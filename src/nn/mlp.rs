use crate::error::{Result, TinyError};

use crate::metal::MetalContext;
use crate::profile::DecodeProfile;
use crate::tensor::{DType, Tensor};
use std::time::Instant;

use super::{Linear, SwiGlu};

pub struct Mlp {
    gate_proj: Linear,
    up_proj: Linear,
    down_proj: Linear,
}

impl Mlp {
    pub fn new(gate_proj: Linear, up_proj: Linear, down_proj: Linear) -> Result<Self> {
        if gate_proj.in_features() != up_proj.in_features() {
            return Err(TinyError::InvalidShape(
                "gate_proj and up_proj must have the same input size".to_string(),
            ));
        }

        if gate_proj.out_features() != up_proj.out_features() {
            return Err(TinyError::InvalidShape(
                "gate_proj and up_proj must have the same output size".to_string(),
            ));
        }

        if down_proj.in_features() != gate_proj.out_features() {
            return Err(TinyError::InvalidShape(
                "down_proj input size must match intermediate size".to_string(),
            ));
        }

        Ok(Self {
            gate_proj,
            up_proj,
            down_proj,
        })
    }

    pub fn input_size(&self) -> usize {
        self.gate_proj.in_features()
    }

    pub fn intermediate_size(&self) -> usize {
        self.gate_proj.out_features()
    }

    pub fn output_size(&self) -> usize {
        self.down_proj.out_features()
    }

    pub fn dtype(&self) -> DType {
        self.gate_proj.dtype()
    }

    pub fn to_dtype(&self, context: &MetalContext, dtype: DType) -> Result<Self> {
        Ok(Self {
            gate_proj: self.gate_proj.to_dtype(context, dtype)?,
            up_proj: self.up_proj.to_dtype(context, dtype)?,
            down_proj: self.down_proj.to_dtype(context, dtype)?,
        })
    }

    pub fn forward(&self, context: &MetalContext, input: &Tensor) -> Result<Tensor> {
        let gate = self.gate_proj.forward(context, input)?;

        let up = self.up_proj.forward(context, input)?;

        let hidden = SwiGlu::forward(context, &gate, &up)?;

        self.down_proj.forward(context, &hidden)
    }

    pub fn forward_profiled(
        &self,
        context: &MetalContext,
        input: &Tensor,
        profile: &mut DecodeProfile,
    ) -> Result<Tensor> {
        let started = Instant::now();
        let gate = self.gate_proj.forward(context, input)?;
        let up = self.up_proj.forward(context, input)?;
        profile.mlp_gate_up += started.elapsed();

        let started = Instant::now();
        let hidden = SwiGlu::forward(context, &gate, &up)?;
        profile.mlp_activation += started.elapsed();

        let started = Instant::now();
        let output = self.down_proj.forward(context, &hidden)?;
        profile.mlp_down += started.elapsed();
        Ok(output)
    }
}

#[cfg(test)]
mod tests {
    use super::{Linear, Mlp};
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
                eprintln!("skipping Metal MLP test: {message}");
                None
            }
            Err(error) => panic!("failed to create Metal context: {error}"),
        }
    }

    #[test]
    fn f16_mlp_stays_close_to_f32_and_returns_f16() {
        let Some(context) = metal_context() else {
            return;
        };

        let input = Tensor::from_f32_slice(&context, &[0.25, -0.5, 1.25, 0.75], &[2, 2]).unwrap();
        let gate_weight =
            Tensor::from_f32_slice(&context, &[0.5, -0.25, 0.75, 1.0, 0.5, -0.5], &[2, 3]).unwrap();
        let up_weight =
            Tensor::from_f32_slice(&context, &[0.25, 0.5, -0.75, 0.5, -1.0, 0.25], &[2, 3])
                .unwrap();
        let down_weight =
            Tensor::from_f32_slice(&context, &[0.5, -0.25, 0.75, 0.5, -0.5, 1.0], &[3, 2]).unwrap();
        let mlp = Mlp::new(
            Linear::new(gate_weight).unwrap(),
            Linear::new(up_weight).unwrap(),
            Linear::new(down_weight).unwrap(),
        )
        .unwrap();

        let output_f32 = mlp.forward(&context, &input).unwrap();
        let mlp_f16 = mlp.to_dtype(&context, DType::F16).unwrap();
        let input_f16 = input.to_dtype(&context, DType::F16).unwrap();
        let output_f16 = mlp_f16.forward(&context, &input_f16).unwrap();

        assert_eq!(output_f16.dtype(), DType::F16);
        assert_close(
            &output_f16.to_f32_vec().unwrap(),
            &output_f32.to_f32_vec().unwrap(),
            5e-2,
        );
    }
}
