use std::ffi::c_void;
use std::mem;

use metal::{CommandBufferRef, MTLSize};

use crate::error::{Result, TinyError};

use crate::metal::{MetalBuffer, MetalContext, MetalExecution};
use crate::tensor::{DType, Tensor};

pub fn rope_f32(
    context: &MetalContext,
    input: &MetalBuffer,
    cos_table: &MetalBuffer,
    sin_table: &MetalBuffer,
    seq_len: usize,
    head_dim: usize,
    start_pos: usize,
) -> Result<MetalBuffer> {
    if head_dim == 0 || !head_dim.is_multiple_of(2) {
        return Err(TinyError::InvalidShape(format!(
            "RoPE head dimension must be positive and even, got {head_dim}"
        )));
    }

    if seq_len == 0 {
        return Err(TinyError::InvalidShape(
            "RoPE sequence length cannot be zero".to_string(),
        ));
    }

    if !input.len().is_multiple_of(head_dim) {
        return Err(TinyError::InvalidShape(format!(
            "RoPE input length {} is not divisible by head dimension {head_dim}",
            input.len(),
        )));
    }

    let half_dim = head_dim / 2;

    let rows = input.len() / head_dim;

    if !rows.is_multiple_of(seq_len) {
        return Err(TinyError::InvalidShape(
            "RoPE input rows are incompatible with sequence length".to_string(),
        ));
    }

    let total_pairs = rows * half_dim;

    let output = MetalBuffer::empty(context, input.len());

    let shader_source = include_str!("../../kernels/rope.metal");

    let pipeline = context.pipeline(shader_source, "rope_f32");

    let command_buffer = context.command_queue.new_command_buffer();

    let encoder = command_buffer.new_compute_command_encoder();

    encoder.set_compute_pipeline_state(pipeline.as_ref());

    encoder.set_buffer(0, Some(input.raw()), 0);

    encoder.set_buffer(1, Some(cos_table.raw()), 0);

    encoder.set_buffer(2, Some(sin_table.raw()), 0);

    encoder.set_buffer(3, Some(output.raw()), 0);

    let seq_len_u32 = u32::try_from(seq_len)
        .map_err(|_| TinyError::InvalidShape("RoPE sequence length exceeds u32".to_string()))?;

    let head_dim_u32 = u32::try_from(head_dim)
        .map_err(|_| TinyError::InvalidShape("RoPE head dimension exceeds u32".to_string()))?;

    let half_dim_u32 = u32::try_from(half_dim)
        .map_err(|_| TinyError::InvalidShape("RoPE half dimension exceeds u32".to_string()))?;

    let start_pos_u32 = u32::try_from(start_pos)
        .map_err(|_| TinyError::InvalidShape("RoPE start position exceeds u32".to_string()))?;

    let total_pairs_u32 = u32::try_from(total_pairs)
        .map_err(|_| TinyError::InvalidShape("RoPE work size exceeds u32".to_string()))?;

    encoder.set_bytes(
        4,
        mem::size_of::<u32>() as u64,
        &seq_len_u32 as *const u32 as *const c_void,
    );

    encoder.set_bytes(
        5,
        mem::size_of::<u32>() as u64,
        &head_dim_u32 as *const u32 as *const c_void,
    );

    encoder.set_bytes(
        6,
        mem::size_of::<u32>() as u64,
        &half_dim_u32 as *const u32 as *const c_void,
    );

    encoder.set_bytes(
        7,
        mem::size_of::<u32>() as u64,
        &start_pos_u32 as *const u32 as *const c_void,
    );

    encoder.set_bytes(
        8,
        mem::size_of::<u32>() as u64,
        &total_pairs_u32 as *const u32 as *const c_void,
    );

    let grid = MTLSize::new(total_pairs as u64, 1, 1);

    let threads_per_group = MTLSize::new(total_pairs.min(256) as u64, 1, 1);

    encoder.dispatch_threads(grid, threads_per_group);

    encoder.end_encoding();

    command_buffer.commit();
    command_buffer.wait_until_completed();

    Ok(output)
}

pub fn rope_f16(
    context: &MetalContext,
    input: &MetalBuffer,
    cos_table: &MetalBuffer,
    sin_table: &MetalBuffer,
    seq_len: usize,
    head_dim: usize,
    start_pos: usize,
) -> Result<MetalBuffer> {
    let execution = MetalExecution::new(context);
    let output = rope_f16_encode(
        context,
        execution.command_buffer(),
        input,
        cos_table,
        sin_table,
        seq_len,
        head_dim,
        start_pos,
    )?;
    execution.finish();
    Ok(output)
}

pub(crate) fn rope_f16_encode(
    context: &MetalContext,
    command_buffer: &CommandBufferRef,
    input: &MetalBuffer,
    cos_table: &MetalBuffer,
    sin_table: &MetalBuffer,
    seq_len: usize,
    head_dim: usize,
    start_pos: usize,
) -> Result<MetalBuffer> {
    if head_dim == 0 || !head_dim.is_multiple_of(2) {
        return Err(TinyError::InvalidShape(format!(
            "RoPE head dimension must be positive and even, got {head_dim}"
        )));
    }
    if seq_len == 0 {
        return Err(TinyError::InvalidShape(
            "RoPE sequence length cannot be zero".to_string(),
        ));
    }
    if !input.len().is_multiple_of(head_dim) {
        return Err(TinyError::InvalidShape(format!(
            "RoPE input length {} is not divisible by head dimension {head_dim}",
            input.len(),
        )));
    }

    let half_dim = head_dim / 2;

    let rows = input.len() / head_dim;

    if !rows.is_multiple_of(seq_len) {
        return Err(TinyError::InvalidShape(
            "RoPE input rows are incompatible with sequence length".to_string(),
        ));
    }

    let total_pairs = rows * half_dim;

    let output = MetalBuffer::empty_with_element_size(context, input.len(), 2);
    let shader_source = include_str!("../../kernels/rope.metal");
    let pipeline = context.pipeline(shader_source, "rope_f16");
    let encoder = command_buffer.new_compute_command_encoder();
    encoder.set_compute_pipeline_state(pipeline.as_ref());
    encoder.set_buffer(0, Some(input.raw()), 0);
    encoder.set_buffer(1, Some(cos_table.raw()), 0);
    encoder.set_buffer(2, Some(sin_table.raw()), 0);
    encoder.set_buffer(3, Some(output.raw()), 0);

    let seq_len_u32 = u32::try_from(seq_len)
        .map_err(|_| TinyError::InvalidShape("RoPE sequence length exceeds u32".to_string()))?;

    let head_dim_u32 = u32::try_from(head_dim)
        .map_err(|_| TinyError::InvalidShape("RoPE head dimension exceeds u32".to_string()))?;

    let half_dim_u32 = u32::try_from(half_dim)
        .map_err(|_| TinyError::InvalidShape("RoPE half dimension exceeds u32".to_string()))?;

    let start_pos_u32 = u32::try_from(start_pos)
        .map_err(|_| TinyError::InvalidShape("RoPE start position exceeds u32".to_string()))?;

    let total_pairs_u32 = u32::try_from(total_pairs)
        .map_err(|_| TinyError::InvalidShape("RoPE work size exceeds u32".to_string()))?;

    encoder.set_bytes(
        4,
        mem::size_of::<u32>() as u64,
        &seq_len_u32 as *const u32 as *const c_void,
    );

    encoder.set_bytes(
        5,
        mem::size_of::<u32>() as u64,
        &head_dim_u32 as *const u32 as *const c_void,
    );

    encoder.set_bytes(
        6,
        mem::size_of::<u32>() as u64,
        &half_dim_u32 as *const u32 as *const c_void,
    );

    encoder.set_bytes(
        7,
        mem::size_of::<u32>() as u64,
        &start_pos_u32 as *const u32 as *const c_void,
    );

    encoder.set_bytes(
        8,
        mem::size_of::<u32>() as u64,
        &total_pairs_u32 as *const u32 as *const c_void,
    );

    encoder.dispatch_threads(
        MTLSize::new(total_pairs as u64, 1, 1),
        MTLSize::new(total_pairs.min(256) as u64, 1, 1),
    );
    encoder.end_encoding();
    Ok(output)
}

pub fn rope(
    context: &MetalContext,
    input: &Tensor,
    cos: &Tensor,
    sin: &Tensor,
    start_pos: usize,
) -> Result<Tensor> {
    if cos.dtype() != DType::F32 || sin.dtype() != DType::F32 {
        return Err(TinyError::UnsupportedDType(
            "RoPE requires F32 cos/sin tables".to_string(),
        ));
    }
    if input.rank() < 2 {
        return Err(TinyError::InvalidDimension(format!(
            "RoPE expects [..., sequence, head_dim], got rank {}",
            input.rank(),
        )));
    }
    let seq_len = input.dim(input.rank() - 2)?;
    let head_dim = input.dim(input.rank() - 1)?;
    if head_dim == 0 || !head_dim.is_multiple_of(2) {
        return Err(TinyError::InvalidShape(format!(
            "RoPE head dimension must be positive and even, got {head_dim}",
        )));
    }
    if cos.rank() != 2 || sin.rank() != 2 || cos.shape().dims() != sin.shape().dims() {
        return Err(TinyError::InvalidShape(
            "RoPE cos/sin tables must have equal rank-2 shapes".to_string(),
        ));
    }
    let table_seq_len = cos.dim(0)?;
    if cos.dim(1)? != head_dim / 2
        || start_pos
            .checked_add(seq_len)
            .is_none_or(|end| end > table_seq_len)
    {
        return Err(TinyError::InvalidShape(
            "RoPE cos/sin table shape or position range is invalid".to_string(),
        ));
    }

    let input = if input.is_contiguous() {
        input.clone()
    } else {
        input.contiguous(context)?
    };
    let buffer = match input.dtype() {
        DType::F32 => rope_f32(
            context,
            input.metal_buffer()?,
            cos.metal_buffer()?,
            sin.metal_buffer()?,
            seq_len,
            head_dim,
            start_pos,
        )?,
        DType::F16 => rope_f16(
            context,
            input.metal_buffer()?,
            cos.metal_buffer()?,
            sin.metal_buffer()?,
            seq_len,
            head_dim,
            start_pos,
        )?,
    };

    Tensor::from_metal_buffer(buffer, input.shape().dims(), input.dtype())
}
