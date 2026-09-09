mod error;
mod metal;
mod nn;
mod ops;
mod tensor;

use error::Result;
use metal::MetalContext;
use nn::Linear;
use tensor::Tensor;

fn main() -> Result<()> {
    let context =
        MetalContext::new();

    // X
    //
    // [1 2 3]
    // [4 5 6]
    //
    // shape = [2, 3]
    let x =
        Tensor::from_f32_slice(
            &context,
            &[
                1.0, 2.0, 3.0,
                4.0, 5.0, 6.0,
            ],
            &[2, 3],
        )?;

    // W
    //
    // [1 2]
    // [3 4]
    // [5 6]
    //
    // shape = [3, 2]
    let weight =
        Tensor::from_f32_slice(
            &context,
            &[
                1.0, 2.0,
                3.0, 4.0,
                5.0, 6.0,
            ],
            &[3, 2],
        )?;

    let linear =
        Linear::new(
            weight,
        )?;

    let y =
        linear.forward(
            &context,
            &x,
        )?;

    println!(
        "Linear"
    );

    println!(
        "in_features  = {}",
        linear.in_features(),
    );

    println!(
        "out_features = {}",
        linear.out_features(),
    );

    println!(
        "output shape = {:?}",
        y.shape().dims(),
    );

    println!(
        "output       = {:?}",
        y.as_f32_slice()?,
    );

    Ok(())
}
