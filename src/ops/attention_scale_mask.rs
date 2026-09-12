use std::ffi::c_void;
use std::mem;

use ::metal::{CommandBufferRef, MTLSize};

use crate::error::{Result, TinyError};

use crate::metal::{MetalBuffer, MetalContext, MetalExecution};

pub fn attention_scale_mask_f32(
    context: &MetalContext,
    input: &MetalBuffer,
    scale: f32,
    query_len: usize,
    key_len: usize,
    query_start_pos: usize,
) -> Result<MetalBuffer> {
    let execution = MetalExecution::new(context);

    let output = attention_scale_mask_f32_encode(
        context,
        execution.command_buffer(),
        input,
        scale,
        query_len,
        key_len,
        query_start_pos,
    )?;

    execution.finish();

    Ok(output)
}

pub(crate) fn attention_scale_mask_f32_encode(
    context: &MetalContext,
    command_buffer: &CommandBufferRef,
    input: &MetalBuffer,
    scale: f32,
    query_len: usize,
    key_len: usize,
    query_start_pos: usize,
) -> Result<MetalBuffer> {
    validate(input, scale, query_len, key_len, query_start_pos, 4)?;

    let output = MetalBuffer::empty(context, input.len());

    encode(
        context,
        command_buffer,
        "attention_scale_mask_f32",
        input,
        &output,
        scale,
        query_len,
        key_len,
        query_start_pos,
    )?;

    Ok(output)
}

/// Applies attention scaling and a causal mask to F16 scores.
///
/// Multiplication uses FP32 and the result is stored in F16. Masked values are
/// negative infinity so a later FP32 softmax maps them to zero probability.
pub fn attention_scale_mask_f16(
    context: &MetalContext,
    input: &MetalBuffer,
    scale: f32,
    query_len: usize,
    key_len: usize,
    query_start_pos: usize,
) -> Result<MetalBuffer> {
    let execution = MetalExecution::new(context);

    let output = attention_scale_mask_f16_encode(
        context,
        execution.command_buffer(),
        input,
        scale,
        query_len,
        key_len,
        query_start_pos,
    )?;

    execution.finish();

    Ok(output)
}

pub(crate) fn attention_scale_mask_f16_encode(
    context: &MetalContext,
    command_buffer: &CommandBufferRef,
    input: &MetalBuffer,
    scale: f32,
    query_len: usize,
    key_len: usize,
    query_start_pos: usize,
) -> Result<MetalBuffer> {
    validate(input, scale, query_len, key_len, query_start_pos, 2)?;

    let output = MetalBuffer::empty_with_element_size(context, input.len(), 2);

    encode(
        context,
        command_buffer,
        "attention_scale_mask_f16",
        input,
        &output,
        scale,
        query_len,
        key_len,
        query_start_pos,
    )?;

    Ok(output)
}

fn validate(
    input: &MetalBuffer,
    scale: f32,
    query_len: usize,
    key_len: usize,
    query_start_pos: usize,
    element_size: usize,
) -> Result<()> {
    if query_len == 0 || key_len == 0 {
        return Err(TinyError::InvalidShape(
            "attention query/key length cannot be zero".to_string(),
        ));
    }

    if !scale.is_finite() || scale <= 0.0 {
        return Err(TinyError::InvalidShape(format!(
            "attention scale must be positive and finite, got {scale}"
        )));
    }

    let query_end = query_start_pos
        .checked_add(query_len)
        .ok_or_else(|| TinyError::InvalidShape("attention query position overflow".to_string()))?;

    if query_end > key_len {
        return Err(TinyError::InvalidShape(format!(
            "attention query range [{query_start_pos}, {query_end}) exceeds key length {key_len}"
        )));
    }

    if !input.len().is_multiple_of(query_len * key_len)
        || input.byte_len() != input.len() * element_size
    {
        return Err(TinyError::InvalidShape(format!(
            "attention score buffer has {} elements ({} bytes), incompatible with query_len={query_len}, key_len={key_len} and {element_size} byte elements",
            input.len(),
            input.byte_len(),
        )));
    }

    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn encode(
    context: &MetalContext,
    command_buffer: &CommandBufferRef,
    kernel_name: &str,
    input: &MetalBuffer,
    output: &MetalBuffer,
    scale: f32,
    query_len: usize,
    key_len: usize,
    query_start_pos: usize,
) -> Result<()> {
    let shader_source = include_str!("../../kernels/attention_scale_mask.metal");

    let pipeline = context.pipeline(shader_source, kernel_name);

    let encoder = command_buffer.new_compute_command_encoder();

    encoder.set_compute_pipeline_state(pipeline.as_ref());

    encoder.set_buffer(0, Some(input.raw()), 0);

    encoder.set_buffer(1, Some(output.raw()), 0);

    encoder.set_bytes(
        2,
        mem::size_of::<f32>() as u64,
        &scale as *const f32 as *const c_void,
    );

    let values = [query_len, key_len, query_start_pos, input.len()].map(|value| {
        u32::try_from(value)
            .map_err(|_| TinyError::InvalidShape("attention dimension exceeds u32".to_string()))
    });

    for (index, value) in values.into_iter().enumerate() {
        let value = value?;

        encoder.set_bytes(
            (index + 3) as u64,
            mem::size_of::<u32>() as u64,
            &value as *const u32 as *const c_void,
        );
    }

    encoder.dispatch_threads(
        MTLSize::new(input.len() as u64, 1, 1),
        MTLSize::new(input.len().min(256) as u64, 1, 1),
    );

    encoder.end_encoding();

    Ok(())
}
