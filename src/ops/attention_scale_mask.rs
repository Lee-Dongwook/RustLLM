use std::ffi::c_void;
use std::mem;

use ::metal::MTLSize;

use crate::error::{Result, TinyError};

use crate::metal::{MetalBuffer, MetalContext};

pub fn attention_scale_mask_f32(
    context: &MetalContext,
    input: &MetalBuffer,
    scale: f32,
    query_len: usize,
    key_len: usize,
    query_start_pos: usize,
) -> Result<MetalBuffer> {
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

    if !input.len().is_multiple_of(query_len * key_len) {
        return Err(TinyError::InvalidShape(format!(
            "attention score buffer length {} is incompatible with query_len={query_len}, key_len={key_len}",
            input.len(),
        )));
    }

    let output = MetalBuffer::empty(context, input.len());

    let shader_source = include_str!("../../kernels/attention_scale_mask.metal");

    let pipeline = context.pipeline(shader_source, "attention_scale_mask_f32");

    let command_buffer = context.command_queue.new_command_buffer();

    let encoder = command_buffer.new_compute_command_encoder();

    encoder.set_compute_pipeline_state(pipeline.as_ref());

    encoder.set_buffer(0, Some(input.raw()), 0);

    encoder.set_buffer(1, Some(output.raw()), 0);

    let query_len = u32::try_from(query_len)
        .map_err(|_| TinyError::InvalidShape("query length exceeds u32".to_string()))?;

    let key_len = u32::try_from(key_len)
        .map_err(|_| TinyError::InvalidShape("key length exceeds u32".to_string()))?;

    let query_start_pos = u32::try_from(query_start_pos)
        .map_err(|_| TinyError::InvalidShape("query start position exceeds u32".to_string()))?;

    let numel = u32::try_from(input.len())
        .map_err(|_| TinyError::InvalidShape("attention score size exceeds u32".to_string()))?;

    encoder.set_bytes(
        2,
        mem::size_of::<f32>() as u64,
        &scale as *const f32 as *const c_void,
    );

    encoder.set_bytes(
        3,
        mem::size_of::<u32>() as u64,
        &query_len as *const u32 as *const c_void,
    );

    encoder.set_bytes(
        4,
        mem::size_of::<u32>() as u64,
        &key_len as *const u32 as *const c_void,
    );

    encoder.set_bytes(
        5,
        mem::size_of::<u32>() as u64,
        &query_start_pos as *const u32 as *const c_void,
    );

    encoder.set_bytes(
        6,
        mem::size_of::<u32>() as u64,
        &numel as *const u32 as *const c_void,
    );

    let grid = MTLSize::new(input.len() as u64, 1, 1);

    let threads_per_group = MTLSize::new(input.len().min(256) as u64, 1, 1);

    encoder.dispatch_threads(grid, threads_per_group);

    encoder.end_encoding();

    command_buffer.commit();

    command_buffer.wait_until_completed();

    Ok(output)
}
