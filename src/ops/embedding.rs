use std::ffi::c_void;
use std::mem;

use ::metal::MTLSize;

use crate::error::{
    Result,
    TinyError,
};

use crate::metal::{
    MetalBuffer,
    MetalContext,
};

pub fn embedding_f32(
    context: &MetalContext,
    weight: &MetalBuffer,
    token_ids: &[u32],
    vocab_size: usize,
    hidden_size: usize,
) -> Result<MetalBuffer> {
    if token_ids.is_empty() {
        return Err(
            TinyError::InvalidShape(
                "embedding requires at least one token"
                    .to_string(),
            ),
        );
    }

    if weight.len()
        != vocab_size * hidden_size
    {
        return Err(
            TinyError::InvalidShape(
                format!(
                    "embedding weight has {} elements, expected {} for shape [{vocab_size}, {hidden_size}]",
                    weight.len(),
                    vocab_size * hidden_size,
                ),
            ),
        );
    }

    for &token_id in token_ids {
        if token_id as usize >= vocab_size {
            return Err(
                TinyError::InvalidTokenId {
                    token_id,
                    vocab_size,
                },
            );
        }
    }

    let token_buffer = MetalBuffer::from_u32_slice(context, token_ids,);

    let output_elements = token_ids.len() * hidden_size;

    let output = MetalBuffer::empty(
        context,
        output_elements,
    );

    let shader_source = 
        include_str!("../../kernels/embedding.metal");
    
    let pipeline = context.pipeline(shader_source, "embedding_f32");

    let command_buffer = context.command_queue.new_command_buffer();

    let encoder = command_buffer.new_compute_command_encoder();

    encoder.set_compute_pipeline_state(
        pipeline.as_ref(),
    );

    encoder.set_buffer(
        0,
        Some(weight.raw()),
        0,
    );

    encoder.set_buffer(
        1,
        Some(token_buffer.raw()),
        0,
    );

    encoder.set_buffer(
        2,
        Some(output.raw()),
        0,
    );

    let hidden_size =
        u32::try_from(
            hidden_size,
        )
        .map_err(|_| {
            TinyError::InvalidShape(
                "hidden size exceeds u32"
                    .to_string(),
            )
        })?;

    let output_elements_u32 =
        u32::try_from(
            output_elements,
        )
        .map_err(|_| {
            TinyError::InvalidShape(
                "embedding output size exceeds u32"
                    .to_string(),
            )
        })?;

    encoder.set_bytes(
        3,
        mem::size_of::<u32>() as u64,
        &hidden_size as *const u32
            as *const c_void,
    );

    encoder.set_bytes(
        4,
        mem::size_of::<u32>() as u64,
        &output_elements_u32
            as *const u32
            as *const c_void,
    );

    let grid =
        MTLSize::new(
            output_elements as u64,
            1,
            1,
        );

    let threads_per_group =
        MTLSize::new(
            output_elements
                .min(256)
                as u64,
            1,
            1,
        );
    
     encoder.dispatch_threads(
        grid,
        threads_per_group,
    );

    encoder.end_encoding();

    command_buffer.commit();

    command_buffer
        .wait_until_completed();

    Ok(output)
}
