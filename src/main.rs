mod metal;
mod ops;
mod tensor;

use metal::MetalContext;
use tensor::Tensor;

fn main() {
    let context = MetalContext::new();

    println!("GPU: {}", context.device.name());

    let a = Tensor::from_slice(
        &context,
        &[
            1.0,
            2.0,
            3.0,
            4.0,
        ],
        &[2, 2],
    );

    let b = Tensor::from_slice(
        &context,
        &[
            10.0,
            20.0,
            30.0,
            40.0,
        ],
        &[2, 2],
    );

    let result =
        a.add(
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
