use crate::error::Result;
use crate::metal::MetalContext;
use crate::tensor::Tensor;

pub struct SwiGlu;

impl SwiGlu {
    pub fn forward(
        context: &MetalContext,
        gate: &Tensor,
        up: &Tensor,
    ) -> Result<Tensor> {
        let gate =
            gate.silu(
                context,
            )?;

        gate.mul(
            context,
            up,
        )
    }
}
