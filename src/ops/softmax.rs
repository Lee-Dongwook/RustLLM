use std::ffi::c_void;
use std::mem;

use ::metal::MTLSize;

use crate::error::{Result, TinyError};

use crate::metal::{MetalBuffer, MetalContext};

const THREADGROUP_SIZE: u64 = 256;

pub fn softmax_f32(
    context: &MetalContext,
    input: &MetalBuffer,
    rows: usize,
    width: usize,
) -> Result<MetalBuffer> {
    if rows == 0 || width == 0 {
        return Err(TinyError::InvalidShape(
            "Softmax rows and width must be greater than zero".to_string(),
        ));
    }

    if input.len() != rows * width {
        return Err(TinyError::InvalidShape(format!(
            "Softmax input has {} elements, expected {}",
            input.len(),
            rows * width,
        )));
    }

    let output = MetalBuffer::empty(context, input.len());

    let shader_source = include_str!("../../kernels/softmax.metal");

    let pipeline = context.pipeline(shader_source, "softmax_f32");

    if THREADGROUP_SIZE > pipeline.max_total_threads_per_threadgroup() {
        return Err(TinyError::Metal(format!(
            "Softmax requires {THREADGROUP_SIZE} threads per group, but pipeline supports at most {}",
            pipeline.max_total_threads_per_threadgroup(),
        )));
    }

    let command_buffer = context.command_queue.new_command_buffer();

    let encoder = command_buffer.new_compute_command_encoder();

    encoder.set_compute_pipeline_state(pipeline.as_ref());

    encoder.set_buffer(0, Some(input.raw()), 0);

    encoder.set_buffer(1, Some(output.raw()), 0);

    let width_u32 = u32::try_from(width)
        .map_err(|_| TinyError::InvalidShape("Softmax width exceeds u32".to_string()))?;

    encoder.set_bytes(
        2,
        mem::size_of::<u32>() as u64,
        &width_u32 as *const u32 as *const c_void,
    );

    // 한 ThreadGroup이 한 row를 담당한다.
    let thread_groups = MTLSize::new(rows as u64, 1, 1);

    let threads_per_group = MTLSize::new(THREADGROUP_SIZE, 1, 1);

    encoder.dispatch_thread_groups(thread_groups, threads_per_group);

    encoder.end_encoding();

    command_buffer.commit();

    command_buffer.wait_until_completed();

    Ok(output)
}
