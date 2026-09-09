use std::ffi::c_void;
use std::mem;

use ::metal::MTLSize;

use crate::error::{Result, TinyError};

use crate::metal::{MetalBuffer, MetalContext};

const THREADGROUP_SIZE: u64 = 256;

pub fn rmsnorm_f32(
    context: &MetalContext,
    input: &MetalBuffer,
    weight: &MetalBuffer,
    rows: usize,
    hidden_size: usize,
    epsilon: f32,
) -> Result<MetalBuffer> {
    if hidden_size == 0 {
        return Err(TinyError::InvalidShape(
            "RMSNorm hidden size cannot be zero".to_string(),
        ));
    }

    if input.len() != rows * hidden_size {
        return Err(TinyError::InvalidShape(format!(
            "RMSNorm input has {} elements, expected {}",
            input.len(),
            rows * hidden_size,
        )));
    }

    if weight.len() != hidden_size {
        return Err(TinyError::InvalidShape(format!(
            "RMSNorm weight has {} elements, expected {}",
            weight.len(),
            hidden_size,
        )));
    }

    let output = MetalBuffer::empty(context, input.len());

    let shader_source = include_str!("../../kernels/rmsnorm.metal");

    let pipeline = context.pipeline(shader_source, "rmsnorm_f32");

    if THREADGROUP_SIZE > pipeline.max_total_threads_per_threadgroup() {
        return Err(TinyError::Metal(format!(
            "RMSNorm requires {THREADGROUP_SIZE} threads per group, but pipeline supports at most {}",
            pipeline.max_total_threads_per_threadgroup(),
        )));
    }

    let command_buffer = context.command_queue.new_command_buffer();

    let encoder = command_buffer.new_compute_command_encoder();

    encoder.set_compute_pipeline_state(pipeline.as_ref());

    encoder.set_buffer(0, Some(input.raw()), 0);

    encoder.set_buffer(1, Some(weight.raw()), 0);

    encoder.set_buffer(2, Some(output.raw()), 0);

    let hidden_size_u32 = u32::try_from(hidden_size)
        .map_err(|_| TinyError::InvalidShape("RMSNorm hidden size exceeds u32".to_string()))?;

    encoder.set_bytes(
        3,
        mem::size_of::<u32>() as u64,
        &hidden_size_u32 as *const u32 as *const c_void,
    );

    encoder.set_bytes(
        4,
        mem::size_of::<f32>() as u64,
        &epsilon as *const f32 as *const c_void,
    );

    let thread_groups = MTLSize::new(rows as u64, 1, 1);

    let threads_per_group = MTLSize::new(THREADGROUP_SIZE, 1, 1);

    encoder.dispatch_thread_groups(thread_groups, threads_per_group);

    encoder.end_encoding();

    command_buffer.commit();

    command_buffer.wait_until_completed();

    Ok(output)
}
