use std::{ffi::c_void, mem};
use ::metal::MTLSize;
use crate::{error::{Result, TinyError}, metal::MetalContext, tensor::Tensor};

pub fn kv_cache_write_f32(context: &MetalContext, cache: &Tensor, source: &Tensor, start: usize) -> Result<()> {
    if cache.rank() != 4 || source.rank() != 4 || cache.dim(0)? != 1 || source.dim(0)? != 1 || cache.dim(1)? != source.dim(1)? || cache.dim(3)? != source.dim(3)? { return Err(TinyError::InvalidShape("invalid KV cache write shape".to_string())); }
    let source_seq = source.dim(2)?; let max_seq = cache.dim(2)?; let head_dim = cache.dim(3)?;
    if start.checked_add(source_seq).is_none_or(|end| end > max_seq) { return Err(TinyError::PositionOutOfRange { start_pos: start, seq_len: source_seq, max_seq_len: max_seq }); }
    let source = source.contiguous(context)?; let pipeline = context.pipeline(include_str!("../../kernels/kv_cache_write.metal"), "kv_cache_write_f32");
    let command = context.command_queue.new_command_buffer(); let encoder = command.new_compute_command_encoder(); encoder.set_compute_pipeline_state(pipeline.as_ref()); encoder.set_buffer(0, Some(cache.metal_buffer()?.raw()), 0); encoder.set_buffer(1, Some(source.metal_buffer()?.raw()), 0);
    let values = [start, source_seq, max_seq, head_dim, source.numel()];
    for (index, value) in values.iter().enumerate() { let value = u32::try_from(*value).map_err(|_| TinyError::InvalidShape("KV cache dimension exceeds u32".to_string()))?; encoder.set_bytes((index + 2) as u64, mem::size_of::<u32>() as u64, &value as *const u32 as *const c_void); }
    encoder.dispatch_threads(MTLSize::new(source.numel() as u64, 1, 1), MTLSize::new(source.numel().min(256) as u64, 1, 1)); encoder.end_encoding(); command.commit(); command.wait_until_completed(); Ok(())
}
