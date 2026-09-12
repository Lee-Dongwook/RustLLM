use std::ffi::c_void;
use std::mem;

use ::metal::{CommandBufferRef, MTLSize};

use crate::error::{Result, TinyError};

use crate::metal::{MetalBuffer, MetalContext, MetalExecution};

const TILE_SIZE: u64 = 16;

pub fn batched_matmul_f32(
    context: &MetalContext,
    a: &MetalBuffer,
    b: &MetalBuffer,
    batch_count: usize,
    m: usize,
    k: usize,
    n: usize,
) -> Result<MetalBuffer> {
    let execution = MetalExecution::new(context);

    let output = batched_matmul_f32_encode(
        context,
        execution.command_buffer(),
        a,
        b,
        batch_count,
        m,
        k,
        n,
    )?;

    execution.finish();

    Ok(output)
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn batched_matmul_f32_encode(
    context: &MetalContext,
    command_buffer: &CommandBufferRef,
    a: &MetalBuffer,
    b: &MetalBuffer,
    batch_count: usize,
    m: usize,
    k: usize,
    n: usize,
) -> Result<MetalBuffer> {
    validate_operands(a, b, batch_count, m, k, n, 4)?;

    let output = MetalBuffer::empty(context, batch_count * m * n);

    encode(
        context,
        command_buffer,
        "batched_matmul_f32",
        a,
        b,
        &output,
        batch_count,
        m,
        k,
        n,
    )?;

    Ok(output)
}

/// Batched F16 matrix multiplication with FP32 accumulation.
///
/// `a` is laid out as `batch_count` contiguous `[m, k]` matrices and `b` as
/// `batch_count` contiguous `[k, n]` matrices. The result contains contiguous
/// `[m, n]` matrices in F16 storage.
pub fn batched_matmul_f16(
    context: &MetalContext,
    a: &MetalBuffer,
    b: &MetalBuffer,
    batch_count: usize,
    m: usize,
    k: usize,
    n: usize,
) -> Result<MetalBuffer> {
    let execution = MetalExecution::new(context);

    let output = batched_matmul_f16_encode(
        context,
        execution.command_buffer(),
        a,
        b,
        batch_count,
        m,
        k,
        n,
    )?;

    execution.finish();

    Ok(output)
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn batched_matmul_f16_encode(
    context: &MetalContext,
    command_buffer: &CommandBufferRef,
    a: &MetalBuffer,
    b: &MetalBuffer,
    batch_count: usize,
    m: usize,
    k: usize,
    n: usize,
) -> Result<MetalBuffer> {
    validate_operands(a, b, batch_count, m, k, n, 2)?;

    let output = MetalBuffer::empty_with_element_size(context, batch_count * m * n, 2);

    encode(
        context,
        command_buffer,
        "batched_matmul_f16",
        a,
        b,
        &output,
        batch_count,
        m,
        k,
        n,
    )?;

    Ok(output)
}

#[allow(clippy::too_many_arguments)]
fn validate_operands(
    a: &MetalBuffer,
    b: &MetalBuffer,
    batch_count: usize,
    m: usize,
    k: usize,
    n: usize,
    element_size: usize,
) -> Result<()> {
    if batch_count == 0 {
        return Err(TinyError::InvalidShape(
            "batched matmul batch count cannot be zero".to_string(),
        ));
    }

    let expected_a = batch_count * m * k;

    let expected_b = batch_count * k * n;

    if a.len() != expected_a || a.byte_len() != expected_a * element_size {
        return Err(TinyError::InvalidShape(format!(
            "batched matmul left buffer has {} elements ({} bytes), expected {expected_a} elements of {element_size} bytes",
            a.len(),
            a.byte_len(),
        )));
    }

    if b.len() != expected_b || b.byte_len() != expected_b * element_size {
        return Err(TinyError::InvalidShape(format!(
            "batched matmul right buffer has {} elements ({} bytes), expected {expected_b} elements of {element_size} bytes",
            b.len(),
            b.byte_len(),
        )));
    }

    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn encode(
    context: &MetalContext,
    command_buffer: &CommandBufferRef,
    kernel_name: &str,
    a: &MetalBuffer,
    b: &MetalBuffer,
    output: &MetalBuffer,
    batch_count: usize,
    m: usize,
    k: usize,
    n: usize,
) -> Result<()> {
    let shader_source = include_str!("../../kernels/batched_matmul.metal");

    let pipeline = context.pipeline(shader_source, kernel_name);

    let encoder = command_buffer.new_compute_command_encoder();

    encoder.set_compute_pipeline_state(pipeline.as_ref());

    encoder.set_buffer(0, Some(a.raw()), 0);

    encoder.set_buffer(1, Some(b.raw()), 0);

    encoder.set_buffer(2, Some(output.raw()), 0);

    let dimensions = [batch_count, m, k, n].map(|value| {
        u32::try_from(value)
            .map_err(|_| TinyError::InvalidShape("batched matmul dimension exceeds u32".to_string()))
    });

    for (index, value) in dimensions.into_iter().enumerate() {
        let value = value?;

        encoder.set_bytes(
            (index + 3) as u64,
            mem::size_of::<u32>() as u64,
            &value as *const u32 as *const c_void,
        );
    }

    encoder.dispatch_thread_groups(
        MTLSize::new(
            (n as u64).div_ceil(TILE_SIZE),
            (m as u64).div_ceil(TILE_SIZE),
            batch_count as u64,
        ),
        MTLSize::new(TILE_SIZE, TILE_SIZE, 1),
    );

    encoder.end_encoding();

    Ok(())
}
