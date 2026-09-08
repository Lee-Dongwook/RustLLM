use std::time::{
    Duration,
    Instant,
};

use crate::metal::MetalContext;
use crate::tensor::Tensor;

const WARMUP_ITERATIONS: usize = 5;
const BENCHMARK_ITERATIONS: usize = 20;

fn benchmark<F>(
    mut operation: F,
) -> Duration
where
    F: FnMut(),
{
    let start =
        Instant::now();

    for _ in 0..BENCHMARK_ITERATIONS {
        operation();
    }

    start.elapsed()
        / BENCHMARK_ITERATIONS as u32
}

pub fn run_matmul_benchmarks(
    context: &MetalContext,
) {
    println!();
    println!(
        "Apple M2 Tile Size Benchmark"
    );

    println!(
        "========================================================"
    );

    println!(
        "{:<8} {:>9} {:>9} {:>9} {:>9}",
        "Size",
        "Naive",
        "Tile 8",
        "Tile 16",
        "Tile 32",
    );

    println!(
        "--------------------------------------------------------"
    );

    for size in [
        128usize,
        256,
        512,
        1024,
    ] {
        benchmark_size(
            context,
            size,
        );
    }

    println!(
        "========================================================"
    );
}

fn benchmark_size(
    context: &MetalContext,
    size: usize,
) {
    let element_count =
        size * size;

    let a_data =
        vec![0.5f32; element_count];

    let b_data =
        vec![0.25f32; element_count];

    let a = Tensor::from_slice(
        context,
        &a_data,
        &[size, size],
    );

    let b = Tensor::from_slice(
        context,
        &b_data,
        &[size, size],
    );

    // --------------------------
    // Warmup
    // --------------------------

    for _ in 0..WARMUP_ITERATIONS {
        let _ =
            a.matmul_naive(
                context,
                &b,
            );

        let _ =
            a.matmul_tiled_8(
                context,
                &b,
            );

        let _ =
            a.matmul_tiled_16(
                context,
                &b,
            );

        let _ =
            a.matmul_tiled_32(
                context,
                &b,
            );
    }

    // --------------------------
    // Benchmark
    // --------------------------

    let naive =
        benchmark(|| {
            let _ =
                a.matmul_naive(
                    context,
                    &b,
                );
        });

    let tiled_8 =
        benchmark(|| {
            let _ =
                a.matmul_tiled_8(
                    context,
                    &b,
                );
        });

    let tiled_16 =
        benchmark(|| {
            let _ =
                a.matmul_tiled_16(
                    context,
                    &b,
                );
        });

    let tiled_32 =
        benchmark(|| {
            let _ =
                a.matmul_tiled_32(
                    context,
                    &b,
                );
        });

    println!(
        "{:<8} {:>9.3} {:>9.3} {:>9.3} {:>9.3}",
        size,
        duration_ms(naive),
        duration_ms(tiled_8),
        duration_ms(tiled_16),
        duration_ms(tiled_32),
    );
}

fn benchmark_naive(
    context: &MetalContext,
    a: &Tensor,
    b: &Tensor,
) -> Duration {
    let start =
        Instant::now();

    for _ in 0..BENCHMARK_ITERATIONS {
        let _ =
            a.matmul_naive(
                context,
                b,
            );
    }

    let elapsed =
        start.elapsed();

    elapsed
        / BENCHMARK_ITERATIONS as u32
}

fn duration_ms(
    duration: Duration,
) -> f64 {
    duration.as_secs_f64()
        * 1_000.0
}
