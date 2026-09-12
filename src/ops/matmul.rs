use std::ffi::c_void;
use std::mem;

use ::metal::{CommandBufferRef, MTLSize};

use crate::error::{Result, TinyError};

use crate::metal::{MetalBuffer, MetalContext, MetalExecution};

pub fn matmul_f32(
    context: &MetalContext,
    a: &MetalBuffer,
    b: &MetalBuffer,
    m: usize,
    k: usize,
    n: usize,
) -> Result<MetalBuffer> {
    let execution = MetalExecution::new(context);

    let output = matmul_f32_encode(context, execution.command_buffer(), a, b, m, k, n)?;

    execution.finish();

    Ok(output)
}

pub(crate) fn matmul_f32_encode(
    context: &MetalContext,
    command_buffer: &CommandBufferRef,
    a: &MetalBuffer,
    b: &MetalBuffer,
    m: usize,
    k: usize,
    n: usize,
) -> Result<MetalBuffer> {
    if a.len() != m * k {
        return Err(TinyError::InvalidShape(format!(
            "left buffer has {} elements, expected {} for shape [{m}, {k}]",
            a.len(),
            m * k,
        )));
    }

    if b.len() != k * n {
        return Err(TinyError::InvalidShape(format!(
            "right buffer has {} elements, expected {} for shape [{k}, {n}]",
            b.len(),
            k * n,
        )));
    }

    let result = MetalBuffer::empty(context, m * n);

    let shader_source = include_str!("../../kernels/matmul.metal");

    let pipeline = context.pipeline(shader_source, "matmul_f32");

    let encoder = command_buffer.new_compute_command_encoder();

    encoder.set_compute_pipeline_state(pipeline.as_ref());

    encoder.set_buffer(0, Some(a.raw()), 0);

    encoder.set_buffer(1, Some(b.raw()), 0);

    encoder.set_buffer(2, Some(result.raw()), 0);

    for (index, value) in [m, k, n].iter().enumerate() {
        let value = u32::try_from(*value)
            .map_err(|_| TinyError::InvalidShape("matmul dimension exceeds u32".to_string()))?;

        encoder.set_bytes(
            (index + 3) as u64,
            mem::size_of::<u32>() as u64,
            &value as *const u32 as *const c_void,
        );
    }

    const TILE_SIZE: u64 = 16;

    encoder.dispatch_thread_groups(
        MTLSize::new(
            (n as u64).div_ceil(TILE_SIZE),
            (m as u64).div_ceil(TILE_SIZE),
            1,
        ),
        MTLSize::new(TILE_SIZE, TILE_SIZE, 1),
    );

    encoder.end_encoding();

    Ok(result)
}
