use metal::{CommandBufferRef, MTLSize};

use crate::{
    error::{Result, TinyError},
    metal::{MetalBuffer, MetalContext, MetalExecution},
};

pub(crate) fn swiglu_f16_encode(
    context: &MetalContext,
    command_buffer: &CommandBufferRef,
    gate: &MetalBuffer,
    up: &MetalBuffer,
) -> Result<MetalBuffer> {
    if gate.len() != up.len() || gate.byte_len() != gate.len() * 2 || up.byte_len() != up.len() * 2
    {
        return Err(TinyError::InvalidShape(
            "SwiGLU requires equal F16 buffers".into(),
        ));
    }
    let output = MetalBuffer::empty_with_element_size(context, gate.len(), 2);
    let pipeline = context.pipeline(include_str!("../../kernels/silu.metal"), "swiglu_f16");
    let encoder = command_buffer.new_compute_command_encoder();
    encoder.set_compute_pipeline_state(pipeline.as_ref());
    encoder.set_buffer(0, Some(gate.raw()), 0);
    encoder.set_buffer(1, Some(up.raw()), 0);
    encoder.set_buffer(2, Some(output.raw()), 0);
    encoder.dispatch_threads(
        MTLSize::new(gate.len() as u64, 1, 1),
        MTLSize::new(gate.len().min(256) as u64, 1, 1),
    );
    encoder.end_encoding();
    Ok(output)
}

#[allow(dead_code)]
pub fn swiglu_f16(
    context: &MetalContext,
    gate: &MetalBuffer,
    up: &MetalBuffer,
) -> Result<MetalBuffer> {
    let execution = MetalExecution::new(context);
    let output = swiglu_f16_encode(context, execution.command_buffer(), gate, up)?;
    execution.finish();
    Ok(output)
}
