use std::ffi::c_void;
use std::mem;

use ::metal::MTLSize;

use crate::error::{Result, TinyError};

use crate::metal::{MetalBuffer, MetalContext};
use crate::tensor::{DType, Tensor};

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

pub fn rmsnorm_f16(
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

    let output = MetalBuffer::empty_with_element_size(context, input.len(), 2);
    let shader_source = include_str!("../../kernels/rmsnorm.metal");
    let pipeline = context.pipeline(shader_source, "rmsnorm_f16");

    if THREADGROUP_SIZE > pipeline.max_total_threads_per_threadgroup() {
        return Err(TinyError::Metal(format!(
            "RMSNorm requires {THREADGROUP_SIZE} threads per group, but pipeline supports at most {}",
            pipeline.max_total_threads_per_threadgroup(),
        )));
    }

    let hidden_size_u32 = u32::try_from(hidden_size)
        .map_err(|_| TinyError::InvalidShape("RMSNorm hidden size exceeds u32".to_string()))?;
    let command_buffer = context.command_queue.new_command_buffer();
    let encoder = command_buffer.new_compute_command_encoder();
    encoder.set_compute_pipeline_state(pipeline.as_ref());
    encoder.set_buffer(0, Some(input.raw()), 0);
    encoder.set_buffer(1, Some(weight.raw()), 0);
    encoder.set_buffer(2, Some(output.raw()), 0);
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
    encoder.dispatch_thread_groups(
        MTLSize::new(rows as u64, 1, 1),
        MTLSize::new(THREADGROUP_SIZE, 1, 1),
    );
    encoder.end_encoding();
    command_buffer.commit();
    command_buffer.wait_until_completed();

    Ok(output)
}

pub fn rmsnorm(
    context: &MetalContext,
    input: &Tensor,
    weight: &Tensor,
    epsilon: f32,
) -> Result<Tensor> {
    if input.dtype() != weight.dtype() {
        return Err(TinyError::UnsupportedDType(format!(
            "RMSNorm dtype mismatch: input {:?}, weight {:?}",
            input.dtype(),
            weight.dtype(),
        )));
    }
    if input.rank() == 0 {
        return Err(TinyError::InvalidDimension(
            "RMSNorm input must have at least one dimension".to_string(),
        ));
    }
    if !epsilon.is_finite() || epsilon <= 0.0 {
        return Err(TinyError::InvalidShape(format!(
            "RMSNorm epsilon must be positive and finite, got {epsilon}",
        )));
    }

    let hidden_size = input.dim(input.rank() - 1)?;
    if weight.rank() != 1 || weight.numel() != hidden_size {
        return Err(TinyError::ShapeMismatch {
            left: input.shape().dims().to_vec(),
            right: weight.shape().dims().to_vec(),
        });
    }

    // F16 stride materialization is not implemented yet, so do not silently
    // treat a strided view as contiguous input.
    if input.dtype() == DType::F16 && (!input.is_contiguous() || !weight.is_contiguous()) {
        return Err(TinyError::NonContiguousTensor(
            "F16 RMSNorm currently requires contiguous input and weight".to_string(),
        ));
    }

    let input = input.contiguous(context)?;
    let weight = weight.contiguous(context)?;
    let rows = input.numel() / hidden_size;

    let buffer = match input.dtype() {
        DType::F32 => rmsnorm_f32(
            context,
            input.metal_buffer()?,
            weight.metal_buffer()?,
            rows,
            hidden_size,
            epsilon,
        )?,
        DType::F16 => rmsnorm_f16(
            context,
            input.metal_buffer()?,
            weight.metal_buffer()?,
            rows,
            hidden_size,
            epsilon,
        )?,
    };

    Tensor::from_metal_buffer(buffer, input.shape().dims(), input.dtype())
}
