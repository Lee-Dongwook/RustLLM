use std::ffi::c_void;
use std::mem;

use ::metal::MTLSize;

use crate::error::{Result, TinyError};

use crate::metal::{MetalBuffer, MetalContext};

pub fn mul_f32(context: &MetalContext, a: &MetalBuffer, b: &MetalBuffer) -> Result<MetalBuffer> {
    if a.len() != b.len() {
        return Err(TinyError::InvalidShape(format!(
            "element-wise multiply requires equal buffer lengths, got {} and {}",
            a.len(),
            b.len(),
        )));
    }

    let output = MetalBuffer::empty(context, a.len());

    let shader_source = include_str!("../../kernels/mul.metal");

    let pipeline = context.pipeline(shader_source, "mul_f32");

    let command_buffer = context.command_queue.new_command_buffer();

    let encoder = command_buffer.new_compute_command_encoder();

    encoder.set_compute_pipeline_state(pipeline.as_ref());

    encoder.set_buffer(0, Some(a.raw()), 0);

    encoder.set_buffer(1, Some(b.raw()), 0);

    encoder.set_buffer(2, Some(output.raw()), 0);

    let numel = u32::try_from(a.len())
        .map_err(|_| TinyError::InvalidShape("multiply input size exceeds u32".to_string()))?;

    encoder.set_bytes(
        3,
        mem::size_of::<u32>() as u64,
        &numel as *const u32 as *const c_void,
    );

    let grid = MTLSize::new(a.len() as u64, 1, 1);

    let threads_per_group = MTLSize::new(a.len().min(256) as u64, 1, 1);

    encoder.dispatch_threads(grid, threads_per_group);

    encoder.end_encoding();

    command_buffer.commit();

    command_buffer.wait_until_completed();

    Ok(output)
}

pub fn mul_f16(context: &MetalContext, a: &MetalBuffer, b: &MetalBuffer) -> Result<MetalBuffer> {
    if a.len() != b.len() {
        return Err(TinyError::InvalidShape(format!(
            "element-wise multiply requires equal buffer lengths, got {} and {}",
            a.len(),
            b.len(),
        )));
    }

    let output = MetalBuffer::empty_with_element_size(context, a.len(), 2);

    let shader_source = include_str!("../../kernels/mul.metal");

    let pipeline = context.pipeline(shader_source, "mul_f16");

    let command_buffer = context.command_queue.new_command_buffer();

    let encoder = command_buffer.new_compute_command_encoder();

    encoder.set_compute_pipeline_state(pipeline.as_ref());

    encoder.set_buffer(0, Some(a.raw()), 0);

    encoder.set_buffer(1, Some(b.raw()), 0);

    encoder.set_buffer(2, Some(output.raw()), 0);

    let grid = MTLSize::new(a.len() as u64, 1, 1);

    let threads_per_group = MTLSize::new(a.len().min(256) as u64, 1, 1);

    encoder.dispatch_threads(grid, threads_per_group);

    encoder.end_encoding();

    command_buffer.commit();

    command_buffer.wait_until_completed();

    Ok(output)
}
