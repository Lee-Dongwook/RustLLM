use std::ffi::c_void;
use std::mem;

use ::metal::MTLSize;

use crate::metal::{
    MetalBuffer,
    MetalContext,
};

pub fn matrix_multiply_naive(
    context: &MetalContext,
    a: &MetalBuffer,
    b: &MetalBuffer,
    m: usize,
    k: usize,
    n: usize,
) -> MetalBuffer {
    validate_inputs(a, b, m, k, n);

    let result =
        MetalBuffer::empty(context, m * n);

    let shader_source =
        include_str!("../../kernels/matmul_naive.metal");

    let pipeline =
        context.pipeline(
            shader_source,
            "matmul_naive",
        );

    let command_buffer =
        context.command_queue.new_command_buffer();

    let encoder =
        command_buffer.new_compute_command_encoder();

    encoder.set_compute_pipeline_state(
        pipeline.as_ref(),
    );

    bind_arguments(
        encoder,
        a,
        b,
        &result,
        m,
        k,
        n,
    );

    // --------------------------------
    // Naive dispatch
    //
    // 실제 필요한 thread만 실행한다.
    //
    // 2 x 2 결과라면:
    //
    // (0,0) (1,0)
    // (0,1) (1,1)
    // --------------------------------

    let grid_size =
        MTLSize::new(
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

    println!(
        "[naive] command status = {:?}",
        command_buffer.status()
    );

    result
}

pub fn matrix_multiply_tiled(
    context: &MetalContext,
    a: &MetalBuffer,
    b: &MetalBuffer,
    m: usize,
    k: usize,
    n: usize,
) -> MetalBuffer {
    validate_inputs(a, b, m, k, n);

    let result =
        MetalBuffer::empty(context, m * n);

    let shader_source =
        include_str!("../../kernels/matmul_tiled.metal");

    let pipeline =
        context.pipeline(
            shader_source,
            "matmul_tiled",
        );

    let command_buffer =
        context.command_queue.new_command_buffer();

    let encoder =
        command_buffer.new_compute_command_encoder();

    encoder.set_compute_pipeline_state(
        pipeline.as_ref(),
    );

    bind_arguments(
        encoder,
        a,
        b,
        &result,
        m,
        k,
        n,
    );

    // --------------------------------
    // Tiled dispatch
    //
    // 반드시 16 x 16 thread가
    // 전부 존재해야 한다.
    // --------------------------------

    const TILE_SIZE: u64 = 16;

    let thread_group_size =
        MTLSize::new(
            TILE_SIZE,
            TILE_SIZE,
            1,
        );

    let groups_x =
        (n as u64 + TILE_SIZE - 1)
            / TILE_SIZE;

    let groups_y =
        (m as u64 + TILE_SIZE - 1)
            / TILE_SIZE;

    let thread_groups =
        MTLSize::new(
            groups_x,
            groups_y,
            1,
        );

    println!(
        "[tiled] groups = {} x {}, threads/group = {} x {}",
        groups_x,
        groups_y,
        TILE_SIZE,
        TILE_SIZE,
    );

    encoder.dispatch_thread_groups(
        thread_groups,
        thread_group_size,
    );

    encoder.end_encoding();

    command_buffer.commit();
    command_buffer.wait_until_completed();

    println!(
        "[tiled] command status = {:?}",
        command_buffer.status()
    );

    result
}

fn validate_inputs(
    a: &MetalBuffer,
    b: &MetalBuffer,
    m: usize,
    k: usize,
    n: usize,
) {
    assert_eq!(
        a.len(),
        m * k,
        "A Buffer 크기가 M x K와 일치하지 않습니다."
    );

    assert_eq!(
        b.len(),
        k * n,
        "B Buffer 크기가 K x N과 일치하지 않습니다."
    );
}

fn bind_arguments(
    encoder: &::metal::ComputeCommandEncoderRef,
    a: &MetalBuffer,
    b: &MetalBuffer,
    result: &MetalBuffer,
    m: usize,
    k: usize,
    n: usize,
) {
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

    let m =
        u32::try_from(m)
            .expect("M이 u32 범위를 초과했습니다.");

    let k =
        u32::try_from(k)
            .expect("K가 u32 범위를 초과했습니다.");

    let n =
        u32::try_from(n)
            .expect("N이 u32 범위를 초과했습니다.");

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
}
