use crate::{
    error::{Result, TinyError},
    metal::{MetalBuffer, MetalContext, MetalExecution},
};
use metal::{CommandBufferRef, MTLSize};
use std::{ffi::c_void, mem};
pub fn matmul_f16(
    context: &MetalContext,
    a: &MetalBuffer,
    b: &MetalBuffer,
    m: usize,
    k: usize,
    n: usize,
) -> Result<MetalBuffer> {
    let execution = MetalExecution::new(context);
    let output = matmul_f16_encode(context, execution.command_buffer(), a, b, m, k, n)?;
    execution.finish();
    Ok(output)
}

pub(crate) fn matmul_f16_encode(
    context: &MetalContext,
    command_buffer: &CommandBufferRef,
    a: &MetalBuffer,
    b: &MetalBuffer,
    m: usize,
    k: usize,
    n: usize,
) -> Result<MetalBuffer> {
    if a.len() != m * k || b.len() != k * n {
        return Err(TinyError::InvalidShape(
            "F16 matmul buffer shape mismatch".into(),
        ));
    }
    let output = MetalBuffer::empty_with_element_size(context, m * n, 2);
    let pipeline = context.pipeline(include_str!("../../kernels/matmul_f16.metal"), "matmul_f16");
    let encoder = command_buffer.new_compute_command_encoder();
    encoder.set_compute_pipeline_state(pipeline.as_ref());
    encoder.set_buffer(0, Some(a.raw()), 0);
    encoder.set_buffer(1, Some(b.raw()), 0);
    encoder.set_buffer(2, Some(output.raw()), 0);
    for (index, value) in [m, k, n].iter().enumerate() {
        let value = u32::try_from(*value)
            .map_err(|_| TinyError::InvalidShape("matmul dimension exceeds u32".into()))?;
        encoder.set_bytes(
            (index + 3) as u64,
            mem::size_of::<u32>() as u64,
            &value as *const u32 as *const c_void,
        );
    }
    encoder.dispatch_threads(MTLSize::new(n as u64, m as u64, 1), MTLSize::new(16, 16, 1));
    encoder.end_encoding();
    Ok(output)
}
