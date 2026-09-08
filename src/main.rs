mod benchmark;
mod metal;
mod ops;
mod tensor;

use benchmark::run_matmul_benchmarks;
use metal::MetalContext;

fn main() {
    let context =
        MetalContext::new();

    println!(
        "GPU: {}",
        context.device.name()
    );

    run_matmul_benchmarks(
        &context,
    );
}
