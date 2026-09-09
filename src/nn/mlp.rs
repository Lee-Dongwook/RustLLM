use crate::error::{
    Result,
    TinyError,
};

use crate::metal::MetalContext;
use crate::tensor::Tensor;

use super::{
    Linear,
    SwiGlu,
};

pub struct Mlp {
    gate_proj: Linear,
    up_proj: Linear,
    down_proj: Linear,
}

impl Mlp {
    pub fn new(
        gate_proj: Linear,
        up_proj: Linear,
        down_proj: Linear,
    ) -> Result<Self> {
        if gate_proj.in_features()
            != up_proj.in_features()
        {
            return Err(
                TinyError::InvalidShape(
                    "gate_proj and up_proj must have the same input size"
                        .to_string(),
                ),
            );
        }

        if gate_proj.out_features()
            != up_proj.out_features()
        {
            return Err(
                TinyError::InvalidShape(
                    "gate_proj and up_proj must have the same output size"
                        .to_string(),
                ),
            );
        }

        if down_proj.in_features()
            != gate_proj.out_features()
        {
            return Err(
                TinyError::InvalidShape(
                    "down_proj input size must match intermediate size"
                        .to_string(),
                ),
            );
        }

        Ok(Self {
            gate_proj,
            up_proj,
            down_proj,
        })
    }

    pub fn forward(
        &self,
        context: &MetalContext,
        input: &Tensor,
    ) -> Result<Tensor> {
        let gate =
            self.gate_proj
                .forward(
                    context,
                    input,
                )?;

        let up =
            self.up_proj
                .forward(
                    context,
                    input,
                )?;

        let hidden =
            SwiGlu::forward(
                context,
                &gate,
                &up,
            )?;

        self.down_proj
            .forward(
                context,
                &hidden,
            )
    }
}
