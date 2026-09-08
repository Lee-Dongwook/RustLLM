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

    result
}

fn matrix_multiply_tiled_impl(
    context: &MetalContext,
    a: &MetalBuffer,
    b: &MetalBuffer,
    m: usize,
    k: usize,
    n: usize,
    tile_size: u64,
    shader_source: &str,
    kernel_name: &str,
) -> MetalBuffer {
    validate_inputs(
        a,
        b,
        m,
        k,
        n,
    );

    let result =
        MetalBuffer::empty(
            context,
            m * n,
        );

    let pipeline =
        context.pipeline(
            shader_source,
            kernel_name,
        );

    // 이 커널에서 Metal이 허용하는
    // 최대 threadgroup thread 수
    let max_threads =
        pipeline
            .max_total_threads_per_threadgroup();

    let execution_width =
        pipeline
            .thread_execution_width();

    let requested_threads =
        tile_size * tile_size;

    assert!(
        requested_threads
            <= max_threads as u64,
        "Tile {}x{}는 이 Pipeline의 최대 thread 수({})를 초과합니다.",
        tile_size,
        tile_size,
        max_threads,
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

    bind_arguments(
        encoder,
        a,
        b,
        &result,
        m,
        k,
        n,
    );

    let threads_per_group =
        MTLSize::new(
            tile_size,
            tile_size,
            1,
        );

    let groups_x =
        (n as u64 + tile_size - 1)
            / tile_size;

    let groups_y =
        (m as u64 + tile_size - 1)
            / tile_size;

    let thread_groups =
        MTLSize::new(
            groups_x,
            groups_y,
            1,
        );

    encoder.dispatch_thread_groups(
        thread_groups,
        threads_per_group,
    );

    encoder.end_encoding();

    command_buffer.commit();

    command_buffer
        .wait_until_completed();

    // 지금은 capability 값이 실제로 어떻게 나오는지
    // 최초 실험에서 보기 위해 남겨둔다.
    let _ = execution_width;

    println!(
    "[{}] execution width = {}, max threads/group = {}",
    kernel_name,
    execution_width,
    max_threads,
);
    result
}

pub fn matrix_multiply_tiled_8(
    context: &MetalContext,
    a: &MetalBuffer,
    b: &MetalBuffer,
    m: usize,
    k: usize,
    n: usize,
) -> MetalBuffer {
    matrix_multiply_tiled_impl(
        context,
        a,
        b,
        m,
        k,
        n,
        8,
        include_str!("../../kernels/matmul_tiled_8.metal"),
        "matmul_tiled_8",
    )
}

pub fn matrix_multiply_tiled_16(
    context: &MetalContext,
    a: &MetalBuffer,
    b: &MetalBuffer,
    m: usize,
    k: usize,
    n: usize,
) -> MetalBuffer {
    matrix_multiply_tiled_impl(
        context,
        a,
        b,
        m,
        k,
        n,
        16,
        include_str!("../../kernels/matmul_tiled_16.metal"),
        "matmul_tiled_16",
    )
}

pub fn matrix_multiply_tiled_32(
    context: &MetalContext,
    a: &MetalBuffer,
    b: &MetalBuffer,
    m: usize,
    k: usize,
    n: usize,
) -> MetalBuffer {
    matrix_multiply_tiled_impl(
        context,
        a,
        b,
        m,
        k,
        n,
        32,
        include_str!("../../kernels/matmul_tiled_32.metal"),
        "matmul_tiled_32",
    )
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
