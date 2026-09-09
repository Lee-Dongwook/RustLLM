use std::ffi::c_void;
use std::mem;

use ::metal::MTLSize;

use crate::error::{Result, TinyError};

use crate::metal::{MetalBuffer, MetalContext};

pub fn matmul_f32(
    context: &MetalContext,
    a: &MetalBuffer,
    b: &MetalBuffer,
    m: usize,
    k: usize,
    n: usize,
) -> Result<MetalBuffer> {
    if a.len() != m * k {
        return Err(TinyError::InvalidShape(format!(
            "left buffer has {} elements, expected {} for shape [{m}, {k}]",
            a.len(),
            m * k,
        )));
    }

    if b.len() != k * n {
        return Err(TinyError::InvalidShape(format!(
            "right buffer has {} elements, expected {} for shape [{k}, {n}]",
            b.len(),
            k * n,
        )));
    }

    let result = MetalBuffer::empty(context, m * n);

    let shader_source = include_str!("../../kernels/matmul.metal");

    let pipeline = context.pipeline(shader_source, "matmul_f32");

    let command_buffer = context.command_queue.new_command_buffer();

    let encoder = command_buffer.new_compute_command_encoder();

    encoder.set_compute_pipeline_state(pipeline.as_ref());

    encoder.set_buffer(0, Some(a.raw()), 0);

    encoder.set_buffer(1, Some(b.raw()), 0);

    encoder.set_buffer(2, Some(result.raw()), 0);

    let m_u32 =
        u32::try_from(m).map_err(|_| TinyError::InvalidShape("M exceeds u32".to_string()))?;

    let k_u32 =
        u32::try_from(k).map_err(|_| TinyError::InvalidShape("K exceeds u32".to_string()))?;

    let n_u32 =
        u32::try_from(n).map_err(|_| TinyError::InvalidShape("N exceeds u32".to_string()))?;

    encoder.set_bytes(
        3,
        mem::size_of::<u32>() as u64,
        &m_u32 as *const u32 as *const c_void,
    );

    encoder.set_bytes(
        4,
        mem::size_of::<u32>() as u64,
        &k_u32 as *const u32 as *const c_void,
    );

    encoder.set_bytes(
        5,
        mem::size_of::<u32>() as u64,
        &n_u32 as *const u32 as *const c_void,
    );

    const TILE_SIZE: u64 = 16;

    let threads_per_group = MTLSize::new(TILE_SIZE, TILE_SIZE, 1);

    let thread_groups = MTLSize::new(
        (n as u64).div_ceil(TILE_SIZE),
        (m as u64).div_ceil(TILE_SIZE),
        1,
    );

    encoder.dispatch_thread_groups(thread_groups, threads_per_group);

    encoder.end_encoding();

    command_buffer.commit();

    command_buffer.wait_until_completed();

    Ok(result)
}
