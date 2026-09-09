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

pub fn silu_f32(
    context: &MetalContext,
    input: &MetalBuffer,
) -> Result<MetalBuffer> {
    if input.len() == 0 {
        return Err(
            TinyError::InvalidShape(
                "SiLU input cannot be empty"
                    .to_string(),
            ),
        );
    }

    let output =
        MetalBuffer::empty(
            context,
            input.len(),
        );

    let shader_source =
        include_str!(
            "../../kernels/silu.metal"
        );

    let pipeline =
        context.pipeline(
            shader_source,
            "silu_f32",
        );

    let command_buffer =
        context
            .command_queue
            .new_command_buffer();

    let encoder =
        command_buffer
            .new_compute_command_encoder();

    encoder.set_compute_pipeline_state(
        pipeline.as_ref(),
    );

    encoder.set_buffer(
        0,
        Some(input.raw()),
        0,
    );

    encoder.set_buffer(
        1,
        Some(output.raw()),
        0,
    );

    let numel =
        u32::try_from(
            input.len(),
        )
        .map_err(|_| {
            TinyError::InvalidShape(
                "SiLU input size exceeds u32"
                    .to_string(),
            )
        })?;

    encoder.set_bytes(
        2,
        mem::size_of::<u32>() as u64,
        &numel as *const u32
            as *const c_void,
    );

    let grid =
        MTLSize::new(
            input.len() as u64,
            1,
            1,
        );

    let threads_per_group =
        MTLSize::new(
            input.len()
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
