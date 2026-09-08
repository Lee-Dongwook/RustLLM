use std::ffi::c_void;
use std::mem;

use ::metal::MTLSize;

use crate::metal::{
    MetalBuffer,
    MetalContext,
};

pub fn matrix_multiply(
    context: &MetalContext,
    a: &MetalBuffer,
    b: &MetalBuffer,
    m: usize,
    k: usize,
    n: usize,
) -> MetalBuffer {
    assert_eq!(
        a.len(),
        m * k,
        "A Buffer의 크기가 M x K와 일치하지 않습니다."
    );

    assert_eq!(
        b.len(),
        k * n,
        "B Buffer의 크기가 K x N과 일치하지 않습니다."
    );

    let result = MetalBuffer::empty(
        context,
        m * n,
    );

    let shader_source =
        include_str!("../../kernels/matmul.metal");

    let pipeline = context.create_pipeline(
        shader_source,
        "matmul",
    );

    let command_buffer = 
        context.command_queue.new_command_buffer();

    let encoder = 
        command_buffer.new_compute_command_encoder();

    encoder.set_compute_pipeline_state(
        &pipeline,
    );

    encoder.set_buffer(
        0,
        Some(a.raw()),
        0,
    );

    encoder.set_buffer(
        1,
        Some(b.raw()),
        0,
    );

    encoder.set_buffer(
        2,
        Some(result.raw()),
        0,
    );

    let m = m as u32;
    let k = k as u32;
    let n = n as u32;

    encoder.set_bytes(
        3,
        mem::size_of::<u32>() as u64,
        &m as *const u32 as *const c_void,
    );

    encoder.set_bytes(
        4,
        mem::size_of::<u32>() as u64,
        &k as *const u32 as *const c_void,
    );

    encoder.set_bytes(
        5,
        mem::size_of::<u32>() as u64,
        &n as *const u32 as *const c_void,
    );

    let grid_size = MTLSize::new(
        n as u64,
        m as u64,
        1,
    );

    let thread_group_size =
        MTLSize::new(
            16,
            16,
            1,
        );

    encoder.dispatch_threads(
        grid_size,
        thread_group_size,
    );

    encoder.end_encoding();

    command_buffer.commit();
    command_buffer.wait_until_completed();

    result
}
