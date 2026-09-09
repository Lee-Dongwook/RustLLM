mod error;
mod metal;
mod nn;
mod ops;
mod tensor;

use error::Result;
use metal::MetalContext;
use nn::RmsNorm;
use tensor::Tensor;

fn main() -> Result<()> {
    let context =
        MetalContext::new();

    let input =
        Tensor::from_f32_slice(
            &context,
            &[
                1.0, 1.0, 1.0, 1.0,

                2.0, 2.0, 2.0, 2.0,
            ],
            &[2, 4],
        )?;

    let weight =
        Tensor::from_f32_slice(
            &context,
            &[
                1.0,
                1.0,
                1.0,
                1.0,
            ],
            &[4],
        )?;

    let norm =
        RmsNorm::new(
            weight,
            1e-5,
        )?;

    let output =
        norm.forward(
            &context,
            &input,
        )?;

    println!(
        "input shape  = {:?}",
        input.shape().dims(),
    );

    println!(
        "hidden size  = {}",
        norm.hidden_size(),
    );

    println!(
        "epsilon      = {}",
        norm.epsilon(),
    );

    println!(
        "output shape = {:?}",
        output.shape().dims(),
    );

    println!(
        "output       = {:?}",
        output.as_f32_slice()?,
    );

    Ok(())
}
