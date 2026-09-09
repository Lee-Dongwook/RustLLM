mod error;
mod metal;
mod nn;
mod ops;
mod tensor;

use error::Result;
use metal::MetalContext;

use nn::{
    Linear,
    Mlp,
};

use tensor::Tensor;

fn main() -> Result<()> {
    let context =
        MetalContext::new();

    // input:
    //
    // [1, 2]
    //
    // shape = [1, 2]
    let x =
        Tensor::from_f32_slice(
            &context,
            &[1.0, 2.0],
            &[1, 2],
        )?;

    // gate projection
    // [2, 4]
    let gate_weight =
        Tensor::from_f32_slice(
            &context,
            &[
                1.0, 0.0, 1.0, 0.0,
                0.0, 1.0, 0.0, 1.0,
            ],
            &[2, 4],
        )?;

    // up projection
    // [2, 4]
    let up_weight =
        Tensor::from_f32_slice(
            &context,
            &[
                1.0, 1.0, 1.0, 1.0,
                1.0, 1.0, 1.0, 1.0,
            ],
            &[2, 4],
        )?;

    // down projection
    // [4, 2]
    let down_weight =
        Tensor::from_f32_slice(
            &context,
            &[
                1.0, 0.0,
                0.0, 1.0,
                1.0, 0.0,
                0.0, 1.0,
            ],
            &[4, 2],
        )?;

    let mlp =
        Mlp::new(
            Linear::new(
                gate_weight,
            )?,
            Linear::new(
                up_weight,
            )?,
            Linear::new(
                down_weight,
            )?,
        )?;

    let y =
        mlp.forward(
            &context,
            &x,
        )?;

    println!(
        "input shape  = {:?}",
        x.shape().dims(),
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
