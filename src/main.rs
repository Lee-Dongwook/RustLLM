mod error;
mod metal;
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
                4.0, 5.0, 6.0,
            ],
            &[2, 3],
        )?;

    println!(
        "--- Original ---"
    );

    println!(
        "shape       = {:?}",
        x.shape().dims(),
    );

    println!(
        "strides     = {:?}",
        x.strides().values(),
    );

    println!(
        "contiguous  = {}",
        x.is_contiguous(),
    );

    println!(
        "data        = {:?}",
        x.as_f32_slice()?,
    );

    let xt =
        x.transpose(
            0,
            1,
        )?;

    println!();
    println!(
        "--- Transpose View ---"
    );

    println!(
        "shape       = {:?}",
        xt.shape().dims(),
    );

    println!(
        "strides     = {:?}",
        xt.strides().values(),
    );

    println!(
        "contiguous  = {}",
        xt.is_contiguous(),
    );

    let xt =
        xt.contiguous(
            &context,
        )?;

    println!();
    println!(
        "--- Materialized ---"
    );

    println!(
        "shape       = {:?}",
        xt.shape().dims(),
    );

    println!(
        "strides     = {:?}",
        xt.strides().values(),
    );

    println!(
        "contiguous  = {}",
        xt.is_contiguous(),
    );

    println!(
        "data        = {:?}",
        xt.as_f32_slice()?,
    );

    Ok(())
}
