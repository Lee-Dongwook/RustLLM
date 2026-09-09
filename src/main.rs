mod error;
mod metal;
mod nn;
mod ops;
mod tensor;

use error::Result;
use metal::MetalContext;
use tensor::Tensor;

fn main() -> Result<()> {
    let context =
        MetalContext::new();

    let x =
        Tensor::from_f32_slice(
            &context,
            &[
                1.0, 2.0, 3.0,
                0.0, 0.0, 0.0,
            ],
            &[2, 3],
        )?;

    let y =
        x.softmax_last_dim(
            &context,
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

    let output =
        y.as_f32_slice()?;

    let row0_sum: f32 =
        output[0..3]
            .iter()
            .sum();

    let row1_sum: f32 =
        output[3..6]
            .iter()
            .sum();

    println!(
        "row 0 sum    = {}",
        row0_sum,
    );

    println!(
        "row 1 sum    = {}",
        row1_sum,
    );

    Ok(())
}
