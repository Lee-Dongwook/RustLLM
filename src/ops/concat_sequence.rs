use std::{ffi::c_void, mem};

use ::metal::MTLSize;

use crate::{
    error::{Result, TinyError},
    metal::{MetalBuffer, MetalContext},
    tensor::{DType, Tensor},
};

const SHADER_SOURCE: &str = include_str!("../../kernels/concat_sequence.metal");

pub fn concat_sequence_f32(
    context: &MetalContext,
    left: &Tensor,
    right: &Tensor,
) -> Result<Tensor> {
    if left.rank() != 4 || right.rank() != 4 {
        return Err(TinyError::ModelFormat(
            "concat_sequence expects rank-4 tensors".to_string(),
        ));
    }

    let left = if left.is_contiguous() {
        left.clone()
    } else {
        left.contiguous(context)?
    };

    let right = if right.is_contiguous() {
        right.clone()
    } else {
        right.contiguous(context)?
    };

    let batch = left.dim(0)?;

    let num_heads = left.dim(1)?;

    let left_seq = left.dim(2)?;

    let head_dim = left.dim(3)?;

    let right_batch = right.dim(0)?;

    let right_heads = right.dim(1)?;

    let right_seq = right.dim(2)?;

    let right_head_dim = right.dim(3)?;

    if batch != right_batch || num_heads != right_heads || head_dim != right_head_dim {
        return Err(TinyError::ModelFormat(format!(
            "concat_sequence incompatible shapes: {:?} and {:?}",
            left.shape().dims(),
            right.shape().dims(),
        )));
    }

    let output_seq = left_seq.checked_add(right_seq).ok_or_else(|| {
        TinyError::InvalidShape("concatenated sequence length overflow".to_string())
    })?;

    let output_shape = vec![batch, num_heads, output_seq, head_dim];

    let output_len = batch
        .checked_mul(num_heads)
        .and_then(|value| value.checked_mul(output_seq))
        .and_then(|value| value.checked_mul(head_dim))
        .ok_or_else(|| TinyError::InvalidShape("concatenated tensor size overflow".to_string()))?;

    let output_buffer = MetalBuffer::empty(context, output_len);

    let pipeline = context.pipeline(SHADER_SOURCE, "concat_sequence_f32");

    let left_buffer = left.metal_buffer()?;
    let right_buffer = right.metal_buffer()?;
    let command_buffer = context.command_queue.new_command_buffer();
    let encoder = command_buffer.new_compute_command_encoder();

    encoder.set_compute_pipeline_state(pipeline.as_ref());
    encoder.set_buffer(0, Some(left_buffer.raw()), 0);
    encoder.set_buffer(1, Some(right_buffer.raw()), 0);
    encoder.set_buffer(2, Some(output_buffer.raw()), 0);

    let left_seq = u32::try_from(left_seq)
        .map_err(|_| TinyError::InvalidShape("left sequence length exceeds u32".to_string()))?;
    let right_seq = u32::try_from(right_seq)
        .map_err(|_| TinyError::InvalidShape("right sequence length exceeds u32".to_string()))?;
    let num_heads = u32::try_from(num_heads)
        .map_err(|_| TinyError::InvalidShape("number of heads exceeds u32".to_string()))?;
    let head_dim = u32::try_from(head_dim)
        .map_err(|_| TinyError::InvalidShape("head dimension exceeds u32".to_string()))?;

    for (index, value) in [left_seq, right_seq, num_heads, head_dim]
        .iter()
        .enumerate()
    {
        encoder.set_bytes(
            (index + 3) as u64,
            mem::size_of::<u32>() as u64,
            value as *const u32 as *const c_void,
        );
    }

    encoder.dispatch_threads(
        MTLSize::new(output_len as u64, 1, 1),
        MTLSize::new(output_len.min(256) as u64, 1, 1),
    );
    encoder.end_encoding();
    command_buffer.commit();
    command_buffer.wait_until_completed();

    Tensor::from_metal_buffer(output_buffer, &output_shape, DType::F32)
}
