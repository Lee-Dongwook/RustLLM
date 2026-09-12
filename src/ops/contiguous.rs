use std::ffi::c_void;
use std::mem;

use ::metal::{CommandBufferRef, MTLSize};

use crate::error::{Result, TinyError};

use crate::metal::{MetalBuffer, MetalContext, MetalExecution};

pub fn materialize_contiguous_f32(
    context: &MetalContext,
    source: &MetalBuffer,
    dims: &[usize],
    strides: &[usize],
) -> Result<MetalBuffer> {
    let execution = MetalExecution::new(context);
    let result = materialize_contiguous_f32_encode(
        context,
        execution.command_buffer(),
        source,
        dims,
        strides,
    )?;
    execution.finish();
    Ok(result)
}

pub(crate) fn materialize_contiguous_f32_encode(
    context: &MetalContext,
    command_buffer: &CommandBufferRef,
    source: &MetalBuffer,
    dims: &[usize],
    strides: &[usize],
) -> Result<MetalBuffer> {
    materialize_contiguous_encode(
        context,
        command_buffer,
        source,
        dims,
        strides,
        4,
        "contiguous_f32",
    )
}

pub fn materialize_contiguous_f16(
    context: &MetalContext,
    source: &MetalBuffer,
    dims: &[usize],
    strides: &[usize],
) -> Result<MetalBuffer> {
    let execution = MetalExecution::new(context);

    let result = materialize_contiguous_f16_encode(
        context,
        execution.command_buffer(),
        source,
        dims,
        strides,
    )?;

    execution.finish();

    Ok(result)
}

pub(crate) fn materialize_contiguous_f16_encode(
    context: &MetalContext,
    command_buffer: &CommandBufferRef,
    source: &MetalBuffer,
    dims: &[usize],
    strides: &[usize],
) -> Result<MetalBuffer> {
    materialize_contiguous_encode(
        context,
        command_buffer,
        source,
        dims,
        strides,
        2,
        "contiguous_f16",
    )
}

fn materialize_contiguous_encode(
    context: &MetalContext,
    command_buffer: &CommandBufferRef,
    source: &MetalBuffer,
    dims: &[usize],
    strides: &[usize],
    element_size: usize,
    kernel_name: &str,
) -> Result<MetalBuffer> {
    if dims.len() != strides.len() {
        return Err(TinyError::InvalidShape(
            "shape rank and strides rank differ".to_string(),
        ));
    }

    let numel: usize = dims.iter().product();
    if source.len() < numel || source.byte_len() != source.len() * element_size {
        return Err(TinyError::InvalidShape(
            "source buffer does not match contiguous materialization dtype".to_string(),
        ));
    }

    let result = MetalBuffer::empty_with_element_size(context, numel, element_size);
    let dims_u32: Vec<u32> = dims
        .iter()
        .map(|&value| u32::try_from(value))
        .collect::<std::result::Result<Vec<_>, _>>()
        .map_err(|_| TinyError::InvalidShape("shape dimension exceeds u32".to_string()))?;
    let strides_u32: Vec<u32> = strides
        .iter()
        .map(|&value| u32::try_from(value))
        .collect::<std::result::Result<Vec<_>, _>>()
        .map_err(|_| TinyError::InvalidShape("stride exceeds u32".to_string()))?;
    let rank = u32::try_from(dims.len())
        .map_err(|_| TinyError::InvalidShape("tensor rank exceeds u32".to_string()))?;
    let numel_u32 = u32::try_from(numel)
        .map_err(|_| TinyError::InvalidShape("tensor element count exceeds u32".to_string()))?;

    let pipeline = context.pipeline(include_str!("../../kernels/contiguous.metal"), kernel_name);
    let encoder = command_buffer.new_compute_command_encoder();
    encoder.set_compute_pipeline_state(pipeline.as_ref());
    encoder.set_buffer(0, Some(source.raw()), 0);
    encoder.set_buffer(1, Some(result.raw()), 0);
    encoder.set_bytes(
        2,
        (dims_u32.len() * mem::size_of::<u32>()) as u64,
        dims_u32.as_ptr() as *const c_void,
    );
    encoder.set_bytes(
        3,
        (strides_u32.len() * mem::size_of::<u32>()) as u64,
        strides_u32.as_ptr() as *const c_void,
    );
    encoder.set_bytes(
        4,
        mem::size_of::<u32>() as u64,
        &rank as *const u32 as *const c_void,
    );
    encoder.set_bytes(
        5,
        mem::size_of::<u32>() as u64,
        &numel_u32 as *const u32 as *const c_void,
    );
    encoder.dispatch_threads(
        MTLSize::new(numel as u64, 1, 1),
        MTLSize::new(numel.min(256) as u64, 1, 1),
    );
    encoder.end_encoding();
    Ok(result)
}
