use crate::error::Result;
use crate::metal::MetalContext;
use crate::tensor::Tensor;
use metal::CommandBufferRef;

pub struct SwiGlu;

impl SwiGlu {
    pub fn forward(context: &MetalContext, gate: &Tensor, up: &Tensor) -> Result<Tensor> {
        let gate = gate.silu(context)?;

        gate.mul(context, up)
    }

    pub(crate) fn forward_encode(
        context: &MetalContext,
        command_buffer: &CommandBufferRef,
        gate: &Tensor,
        up: &Tensor,
    ) -> Result<Tensor> {
        if gate.dtype() == crate::tensor::DType::F16
            && gate.is_contiguous()
            && up.is_contiguous()
            && gate.shape() == up.shape()
        {
            let output = crate::ops::swiglu_f16_encode(
                context,
                command_buffer,
                gate.metal_buffer()?,
                up.metal_buffer()?,
            )?;
            Tensor::from_metal_buffer(output, gate.shape().dims(), crate::tensor::DType::F16)
        } else {
            gate.silu_encode(context, command_buffer)?
                .mul_encode(context, command_buffer, up)
        }
    }
}
