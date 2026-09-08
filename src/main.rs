mod metal;
mod ops;
mod tensor;

use metal::MetalContext;
use tensor::Tensor;

fn main() {
    let context = MetalContext::new();

    println!(
        "GPU: {}",
        context.device.name()
    );

    let a = Tensor::from_slice(
        &context,
        &[
            1.0, 2.0, 3.0,
            4.0, 5.0, 6.0,
        ],
        &[2, 3],
    );

    let b = Tensor::from_slice(
        &context,
        &[
            7.0, 8.0,
            9.0, 10.0,
            11.0, 12.0,
        ],
        &[3, 2],
    );

    let naive_1 =
        a.matmul_naive(
            &context,
            &b,
        );

    let naive_2 =
        a.matmul_naive(
            &context,
            &b,
        );

    let tiled_1 =
        a.matmul_tiled(
            &context,
            &b,
        );

    let tiled_2 =
        a.matmul_tiled(
            &context,
            &b,
        );

    println!(
        "Naive 1 = {:?}",
        naive_1.as_slice()
    );

    println!(
        "Naive 2 = {:?}",
        naive_2.as_slice()
    );

    println!(
        "Tiled 1 = {:?}",
        tiled_1.as_slice()
    );

    println!(
        "Tiled 2 = {:?}",
        tiled_2.as_slice()
    );
}
