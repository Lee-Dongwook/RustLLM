mod metal;
mod ops;
mod tensor;

use metal::MetalContext;
use tensor::Tensor;

fn main() {
    let context =
        MetalContext::new();

    println!(
        "GPU: {}",
        context.device.name()
    );

    // A: 2 x 3
    //
    // [1 2 3]
    // [4 5 6]
    let a = Tensor::from_slice(
        &context,
        &[
            1.0, 2.0, 3.0,
            4.0, 5.0, 6.0,
        ],
        &[2, 3],
    );

    // B: 3 x 2
    //
    // [ 7  8]
    // [ 9 10]
    // [11 12]
    let b = Tensor::from_slice(
        &context,
        &[
            7.0, 8.0,
            9.0, 10.0,
            11.0, 12.0,
        ],
        &[3, 2],
    );

    let result =
        a.matmul(
            &context,
            &b,
        );

    println!(
        "A shape      = {:?}",
        a.shape()
    );

    println!(
        "A            = {:?}",
        a.as_slice()
    );

    println!(
        "B shape      = {:?}",
        b.shape()
    );

    println!(
        "B            = {:?}",
        b.as_slice()
    );

    println!(
        "Result shape = {:?}",
        result.shape()
    );

    println!(
        "Result       = {:?}",
        result.as_slice()
    );
}
