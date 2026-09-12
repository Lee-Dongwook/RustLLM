use std::ffi::c_void;
use std::mem;

use ::metal::{CommandBufferRef, MTLSize};

use crate::error::{Result, TinyError};

use crate::metal::{MetalBuffer, MetalContext, MetalExecution};

const THREADGROUP_SIZE: u64 = 256;

pub fn softmax_f32(
    context: &MetalContext,
    input: &MetalBuffer,
    rows: usize,
    width: usize,
) -> Result<MetalBuffer> {
    let execution = MetalExecution::new(context);

    let output = softmax_f32_encode(context, execution.command_buffer(), input, rows, width)?;

    execution.finish();

    Ok(output)
}

pub(crate) fn softmax_f32_encode(
    context: &MetalContext,
    command_buffer: &CommandBufferRef,
    input: &MetalBuffer,
    rows: usize,
    width: usize,
) -> Result<MetalBuffer> {
    validate(input, rows, width, 4)?;

    let output = MetalBuffer::empty(context, input.len());

    encode(
        context,
        command_buffer,
        "softmax_f32",
        input,
        &output,
        rows,
        width,
    )?;

    Ok(output)
}

/// Stable row-wise softmax with F16 storage and FP32 reductions.
pub fn softmax_f16(
    context: &MetalContext,
    input: &MetalBuffer,
    rows: usize,
    width: usize,
) -> Result<MetalBuffer> {
    let execution = MetalExecution::new(context);

    let output = softmax_f16_encode(context, execution.command_buffer(), input, rows, width)?;

    execution.finish();

    Ok(output)
}

pub(crate) fn softmax_f16_encode(
    context: &MetalContext,
    command_buffer: &CommandBufferRef,
    input: &MetalBuffer,
    rows: usize,
    width: usize,
) -> Result<MetalBuffer> {
    validate(input, rows, width, 2)?;

    let output = MetalBuffer::empty_with_element_size(context, input.len(), 2);

    encode(
        context,
        command_buffer,
        "softmax_f16",
        input,
        &output,
        rows,
        width,
    )?;

    Ok(output)
}

fn validate(input: &MetalBuffer, rows: usize, width: usize, element_size: usize) -> Result<()> {
    if rows == 0 || width == 0 {
        return Err(TinyError::InvalidShape(
            "Softmax rows and width must be greater than zero".to_string(),
        ));
    }

    if input.len() != rows * width || input.byte_len() != input.len() * element_size {
        return Err(TinyError::InvalidShape(format!(
            "Softmax input has {} elements ({} bytes), expected {} elements of {element_size} bytes",
            input.len(),
            input.byte_len(),
            rows * width,
        )));
    }

    Ok(())
}

fn encode(
    context: &MetalContext,
    command_buffer: &CommandBufferRef,
    kernel_name: &str,
    input: &MetalBuffer,
    output: &MetalBuffer,
    rows: usize,
    width: usize,
) -> Result<()> {
    let shader_source = include_str!("../../kernels/softmax.metal");

    let pipeline = context.pipeline(shader_source, kernel_name);

    if THREADGROUP_SIZE > pipeline.max_total_threads_per_threadgroup() {
        return Err(TinyError::Metal(format!(
            "Softmax requires {THREADGROUP_SIZE} threads per group, but pipeline supports at most {}",
            pipeline.max_total_threads_per_threadgroup(),
        )));
    }

    let width_u32 = u32::try_from(width)
        .map_err(|_| TinyError::InvalidShape("Softmax width exceeds u32".to_string()))?;

    let encoder = command_buffer.new_compute_command_encoder();

    encoder.set_compute_pipeline_state(pipeline.as_ref());

    encoder.set_buffer(0, Some(input.raw()), 0);

    encoder.set_buffer(1, Some(output.raw()), 0);

    encoder.set_bytes(
        2,
        mem::size_of::<u32>() as u64,
        &width_u32 as *const u32 as *const c_void,
    );

    // 한 ThreadGroup이 한 row를 담당한다.
    encoder.dispatch_thread_groups(
        MTLSize::new(rows as u64, 1, 1),
        MTLSize::new(THREADGROUP_SIZE, 1, 1),
    );

    encoder.end_encoding();

    Ok(())
}
