use std::ffi::c_void;
use std::mem;

use ::metal::MTLSize;

use crate::error::{Result, TinyError};

use crate::metal::{MetalBuffer, MetalContext};

pub fn batched_matmul_f32(
    context: &MetalContext,
    a: &MetalBuffer,
    b: &MetalBuffer,
    batch_count: usize,
    m: usize,
    k: usize,
    n: usize,
) -> Result<MetalBuffer> {
    if batch_count == 0 {
        return Err(TinyError::InvalidShape(
            "batched matmul batch count cannot be zero".to_string(),
        ));
    }

    let expected_a = batch_count * m * k;

    let expected_b = batch_count * k * n;

    if a.len() != expected_a {
        return Err(TinyError::InvalidShape(format!(
            "batched matmul left buffer has {} elements, expected {}",
            a.len(),
            expected_a,
        )));
    }

    if b.len() != expected_b {
        return Err(TinyError::InvalidShape(format!(
            "batched matmul right buffer has {} elements, expected {}",
            b.len(),
            expected_b,
        )));
    }

    let output_len = batch_count * m * n;

    let output = MetalBuffer::empty(context, output_len);

    let shader_source = include_str!("../../kernels/batched_matmul.metal");

    let pipeline = context.pipeline(shader_source, "batched_matmul_f32");

    let command_buffer = context.command_queue.new_command_buffer();

    let encoder = command_buffer.new_compute_command_encoder();

    encoder.set_compute_pipeline_state(pipeline.as_ref());

    encoder.set_buffer(0, Some(a.raw()), 0);

    encoder.set_buffer(1, Some(b.raw()), 0);

    encoder.set_buffer(2, Some(output.raw()), 0);

    let batch_count_u32 = u32::try_from(batch_count).map_err(|_| {
        TinyError::InvalidShape("batched matmul batch count exceeds u32".to_string())
    })?;

    let m_u32 = u32::try_from(m)
        .map_err(|_| TinyError::InvalidShape("batched matmul M exceeds u32".to_string()))?;

    let k_u32 = u32::try_from(k)
        .map_err(|_| TinyError::InvalidShape("batched matmul K exceeds u32".to_string()))?;

    let n_u32 = u32::try_from(n)
        .map_err(|_| TinyError::InvalidShape("batched matmul N exceeds u32".to_string()))?;

    encoder.set_bytes(
        3,
        mem::size_of::<u32>() as u64,
        &batch_count_u32 as *const u32 as *const c_void,
    );

    encoder.set_bytes(
        4,
        mem::size_of::<u32>() as u64,
        &m_u32 as *const u32 as *const c_void,
    );

    encoder.set_bytes(
        5,
        mem::size_of::<u32>() as u64,
        &k_u32 as *const u32 as *const c_void,
    );

    encoder.set_bytes(
        6,
        mem::size_of::<u32>() as u64,
        &n_u32 as *const u32 as *const c_void,
    );

    const TILE_SIZE: u64 = 16;

    let threads_per_group = MTLSize::new(TILE_SIZE, TILE_SIZE, 1);

    let groups_x = (n as u64).div_ceil(TILE_SIZE);

    let groups_y = (m as u64).div_ceil(TILE_SIZE);

    let thread_groups = MTLSize::new(groups_x, groups_y, batch_count as u64);

    encoder.dispatch_thread_groups(thread_groups, threads_per_group);

    encoder.end_encoding();

    command_buffer.commit();

    command_buffer.wait_until_completed();

    Ok(output)
}
