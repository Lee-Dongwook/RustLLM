use std::ffi::c_void;
use std::mem;

use ::metal::{CommandBufferRef, MTLSize};

use crate::error::{Result, TinyError};

use crate::metal::{MetalBuffer, MetalContext, MetalExecution};

pub fn embedding_f32(
    context: &MetalContext,
    weight: &MetalBuffer,
    token_ids: &[u32],
    vocab_size: usize,
    hidden_size: usize,
) -> Result<MetalBuffer> {
    let execution = MetalExecution::new(context);

    let output = embedding_f32_encode(
        context,
        execution.command_buffer(),
        weight,
        token_ids,
        vocab_size,
        hidden_size,
    )?;

    execution.finish();

    Ok(output)
}

pub(crate) fn embedding_f32_encode(
    context: &MetalContext,
    command_buffer: &CommandBufferRef,
    weight: &MetalBuffer,
    token_ids: &[u32],
    vocab_size: usize,
    hidden_size: usize,
) -> Result<MetalBuffer> {
    validate(weight, token_ids, vocab_size, hidden_size, 4)?;

    let output = MetalBuffer::empty(context, token_ids.len() * hidden_size);

    encode(
        context,
        command_buffer,
        "embedding_f32",
        weight,
        token_ids,
        &output,
        hidden_size,
    )?;

    Ok(output)
}

pub fn embedding_f16(
    context: &MetalContext,
    weight: &MetalBuffer,
    token_ids: &[u32],
    vocab_size: usize,
    hidden_size: usize,
) -> Result<MetalBuffer> {
    let execution = MetalExecution::new(context);

    let output = embedding_f16_encode(
        context,
        execution.command_buffer(),
        weight,
        token_ids,
        vocab_size,
        hidden_size,
    )?;

    execution.finish();

    Ok(output)
}

pub(crate) fn embedding_f16_encode(
    context: &MetalContext,
    command_buffer: &CommandBufferRef,
    weight: &MetalBuffer,
    token_ids: &[u32],
    vocab_size: usize,
    hidden_size: usize,
) -> Result<MetalBuffer> {
    validate(weight, token_ids, vocab_size, hidden_size, 2)?;

    let output = MetalBuffer::empty_with_element_size(context, token_ids.len() * hidden_size, 2);

    encode(
        context,
        command_buffer,
        "embedding_f16",
        weight,
        token_ids,
        &output,
        hidden_size,
    )?;

    Ok(output)
}

fn validate(
    weight: &MetalBuffer,
    token_ids: &[u32],
    vocab_size: usize,
    hidden_size: usize,
    element_size: usize,
) -> Result<()> {
    if token_ids.is_empty() {
        return Err(TinyError::InvalidShape(
            "embedding requires at least one token".to_string(),
        ));
    }

    if weight.len() != vocab_size * hidden_size || weight.byte_len() != weight.len() * element_size
    {
        return Err(TinyError::InvalidShape(format!(
            "embedding weight has {} elements ({} bytes), expected {} elements of {element_size} bytes for shape [{vocab_size}, {hidden_size}]",
            weight.len(),
            weight.byte_len(),
            vocab_size * hidden_size,
        )));
    }

    for &token_id in token_ids {
        if token_id as usize >= vocab_size {
            return Err(TinyError::InvalidTokenId {
                token_id,
                vocab_size,
            });
        }
    }

    Ok(())
}

fn encode(
    context: &MetalContext,
    command_buffer: &CommandBufferRef,
    kernel_name: &str,
    weight: &MetalBuffer,
    token_ids: &[u32],
    output: &MetalBuffer,
    hidden_size: usize,
) -> Result<()> {
    // Metal command buffers retain the resources bound to their encoders, so
    // this token buffer outlives the handle dropped at the end of this call.
    let token_buffer = MetalBuffer::from_u32_slice(context, token_ids);

    let output_elements = token_ids.len() * hidden_size;

    let shader_source = include_str!("../../kernels/embedding.metal");

    let pipeline = context.pipeline(shader_source, kernel_name);

    let encoder = command_buffer.new_compute_command_encoder();

    encoder.set_compute_pipeline_state(pipeline.as_ref());

    encoder.set_buffer(0, Some(weight.raw()), 0);

    encoder.set_buffer(1, Some(token_buffer.raw()), 0);

    encoder.set_buffer(2, Some(output.raw()), 0);

    let values = [hidden_size, output_elements].map(|value| {
        u32::try_from(value)
            .map_err(|_| TinyError::InvalidShape("embedding dimension exceeds u32".to_string()))
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
        MTLSize::new(output_elements as u64, 1, 1),
        MTLSize::new(output_elements.min(256) as u64, 1, 1),
    );

    encoder.end_encoding();

    Ok(())
}
