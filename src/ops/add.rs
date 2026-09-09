use ::metal::MTLSize;

use crate::error::{Result, TinyError};

use crate::metal::{MetalBuffer, MetalContext};

pub fn add_f32(context: &MetalContext, a: &MetalBuffer, b: &MetalBuffer) -> Result<MetalBuffer> {
    if a.len() != b.len() {
        return Err(TinyError::InvalidShape(format!(
            "add requires equal buffer lengths, got {} and {}",
            a.len(),
            b.len(),
        )));
    }

    let output = MetalBuffer::empty(context, a.len());

    let shader_source = include_str!("../../kernels/vector_add.metal");

    let pipeline = context.pipeline(shader_source, "vector_add");

    let command_buffer = context.command_queue.new_command_buffer();

    let encoder = command_buffer.new_compute_command_encoder();

    encoder.set_compute_pipeline_state(pipeline.as_ref());

    encoder.set_buffer(0, Some(a.raw()), 0);

    encoder.set_buffer(1, Some(b.raw()), 0);

    encoder.set_buffer(2, Some(output.raw()), 0);

    let grid = MTLSize::new(a.len() as u64, 1, 1);

    let threads_per_group = MTLSize::new(a.len().min(256) as u64, 1, 1);

    encoder.dispatch_threads(grid, threads_per_group);

    encoder.end_encoding();

    command_buffer.commit();
    command_buffer.wait_until_completed();

    Ok(output)
}
