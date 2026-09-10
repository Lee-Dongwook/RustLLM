use std::{ffi::c_void, mem};

use ::metal::MTLSize;

use crate::{
    error::{Result, TinyError},
    metal::MetalContext,
    tensor::{DType, Tensor},
};

const SHADER_SOURCE: &str = include_str!("../../kernels/repeat_kv.metal");

pub fn repeat_kv(context: &MetalContext, input: &Tensor, repeats: usize) -> Result<Tensor> {
    if input.rank() != 4 {
        return Err(TinyError::ModelFormat(format!(
            "repeat_kv expects [B, KVH, S, D], got {:?}",
            input.shape().dims(),
        )));
    }

    if repeats == 0 {
        return Err(TinyError::ModelFormat(
            "repeat_kv repeats must be greater than zero".to_string(),
        ));
    }

    if repeats == 1 {
        return Ok(input.clone());
    }

    let input = if input.is_contiguous() {
        input.clone()
    } else {
        input.contiguous(context)?
    };

    let batch = input.dim(0)?;
    let kv_heads = input.dim(1)?;
    let seq_len = input.dim(2)?;
    let head_dim = input.dim(3)?;

    let query_heads = kv_heads
        .checked_mul(repeats)
        .ok_or_else(|| TinyError::ModelFormat("repeat_kv head count overflow".to_string()))?;

    let output = Tensor::empty(
        context,
        &[batch, query_heads, seq_len, head_dim],
        input.dtype(),
    )?;

    match input.dtype() {
        DType::F32 => {
            dispatch_repeat_kv(
                context,
                &input,
                &output,
                kv_heads,
                seq_len,
                head_dim,
                repeats,
                "repeat_kv_f32",
            )?;
        }

        DType::F16 => {
            dispatch_repeat_kv(
                context,
                &input,
                &output,
                kv_heads,
                seq_len,
                head_dim,
                repeats,
                "repeat_kv_f16",
            )?;
        }
    }

    Ok(output)
}

fn dispatch_repeat_kv(
    context: &MetalContext,
    input: &Tensor,
    output: &Tensor,
    kv_heads: usize,
    seq_len: usize,
    head_dim: usize,
    repeats: usize,
    kernel_name: &str,
) -> Result<()> {
    let dimensions = [kv_heads, seq_len, head_dim, repeats].map(|value| {
        u32::try_from(value)
            .map_err(|_| TinyError::InvalidShape("repeat_kv dimension exceeds u32".to_string()))
    });
    let [kv_heads, seq_len, head_dim, repeats] = dimensions;

    let input = input.metal_buffer()?;
    let output_buffer = output.metal_buffer()?;
    let pipeline = context.pipeline(SHADER_SOURCE, kernel_name);
    let command_buffer = context.command_queue.new_command_buffer();
    let encoder = command_buffer.new_compute_command_encoder();

    encoder.set_compute_pipeline_state(pipeline.as_ref());
    encoder.set_buffer(0, Some(input.raw()), 0);
    encoder.set_buffer(1, Some(output_buffer.raw()), 0);

    for (index, value) in [kv_heads?, seq_len?, head_dim?, repeats?]
        .iter()
        .enumerate()
    {
        encoder.set_bytes(
            (index + 2) as u64,
            mem::size_of::<u32>() as u64,
            value as *const u32 as *const c_void,
        );
    }

    let total_threads = output.numel();
    encoder.dispatch_threads(
        MTLSize::new(total_threads as u64, 1, 1),
        MTLSize::new(total_threads.min(256) as u64, 1, 1),
    );
    encoder.end_encoding();
    command_buffer.commit();
    command_buffer.wait_until_completed();

    Ok(())
}
