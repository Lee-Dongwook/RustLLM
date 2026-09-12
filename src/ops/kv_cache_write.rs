use crate::{
    error::{Result, TinyError},
    metal::MetalContext,
    tensor::{DType, Tensor},
};
use metal::{CommandBufferRef, MTLSize};
use std::{ffi::c_void, mem};

pub fn kv_cache_write_f32(
    context: &MetalContext,
    cache: &Tensor,
    source: &Tensor,
    start: usize,
) -> Result<()> {
    let execution = crate::metal::MetalExecution::new(context);
    let source = source.contiguous(context)?;
    kv_cache_write_f32_encode(context, execution.command_buffer(), cache, &source, start)?;
    execution.finish();
    Ok(())
}

pub(crate) fn kv_cache_write_f32_encode(
    context: &MetalContext,
    command_buffer: &CommandBufferRef,
    cache: &Tensor,
    source: &Tensor,
    start: usize,
) -> Result<()> {
    kv_cache_write_encode(
        context,
        command_buffer,
        cache,
        source,
        start,
        DType::F32,
        "kv_cache_write_f32",
    )
}

pub fn kv_cache_write_f16(
    context: &MetalContext,
    cache: &Tensor,
    source: &Tensor,
    start: usize,
) -> Result<()> {
    let execution = crate::metal::MetalExecution::new(context);
    let source = source.contiguous(context)?;
    kv_cache_write_f16_encode(context, execution.command_buffer(), cache, &source, start)?;
    execution.finish();
    Ok(())
}

pub(crate) fn kv_cache_write_f16_encode(
    context: &MetalContext,
    command_buffer: &CommandBufferRef,
    cache: &Tensor,
    source: &Tensor,
    start: usize,
) -> Result<()> {
    kv_cache_write_encode(
        context,
        command_buffer,
        cache,
        source,
        start,
        DType::F16,
        "kv_cache_write_f16",
    )
}

pub(crate) fn kv_cache_write_encode(
    context: &MetalContext,
    command_buffer: &CommandBufferRef,
    cache: &Tensor,
    source: &Tensor,
    start: usize,
    dtype: DType,
    kernel_name: &str,
) -> Result<()> {
    if cache.dtype() != dtype || source.dtype() != dtype {
        return Err(TinyError::UnsupportedDType(format!(
            "{kernel_name} expects {dtype:?} cache and source, got {:?} and {:?}",
            cache.dtype(),
            source.dtype(),
        )));
    }

    if cache.rank() != 4
        || source.rank() != 4
        || cache.dim(0)? != 1
        || source.dim(0)? != 1
        || cache.dim(1)? != source.dim(1)?
        || cache.dim(3)? != source.dim(3)?
    {
        return Err(TinyError::InvalidShape(
            "invalid KV cache write shape".to_string(),
        ));
    }
    let source_seq = source.dim(2)?;
    let max_seq = cache.dim(2)?;
    let head_dim = cache.dim(3)?;
    if start
        .checked_add(source_seq)
        .is_none_or(|end| end > max_seq)
    {
        return Err(TinyError::PositionOutOfRange {
            start_pos: start,
            seq_len: source_seq,
            max_seq_len: max_seq,
        });
    }
    let pipeline = context.pipeline(
        include_str!("../../kernels/kv_cache_write.metal"),
        kernel_name,
    );
    if !source.is_contiguous() {
        return Err(TinyError::NonContiguousTensor(
            "batched KV cache write requires a contiguous source".into(),
        ));
    }
    let encoder = command_buffer.new_compute_command_encoder();
    encoder.set_compute_pipeline_state(pipeline.as_ref());
    encoder.set_buffer(0, Some(cache.metal_buffer()?.raw()), 0);
    encoder.set_buffer(1, Some(source.metal_buffer()?.raw()), 0);
    let values = [start, source_seq, max_seq, head_dim, source.numel()];
    for (index, value) in values.iter().enumerate() {
        let value = u32::try_from(*value)
            .map_err(|_| TinyError::InvalidShape("KV cache dimension exceeds u32".to_string()))?;
        encoder.set_bytes(
            (index + 2) as u64,
            mem::size_of::<u32>() as u64,
            &value as *const u32 as *const c_void,
        );
    }
    encoder.dispatch_threads(
        MTLSize::new(source.numel() as u64, 1, 1),
        MTLSize::new(source.numel().min(256) as u64, 1, 1),
    );
    encoder.end_encoding();
    Ok(())
}
