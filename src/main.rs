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

    let a =
        Tensor::from_f32_slice(
            &context,
            &[
                1.0, 2.0, 3.0,
                4.0, 5.0, 6.0,
            ],
            &[2, 3],
        )?;

    println!("--- A ---");

    println!(
        "shape       = {:?}",
        a.shape().dims(),
    );

    println!(
        "strides     = {:?}",
        a.strides().values(),
    );

    println!(
        "rank        = {}",
        a.rank(),
    );

    println!(
        "numel       = {}",
        a.numel(),
    );

    println!(
        "dim(0)      = {}",
        a.dim(0)?,
    );

    println!(
        "dim(1)      = {}",
        a.dim(1)?,
    );

    println!(
        "contiguous  = {}",
        a.is_contiguous(),
    );

    println!(
        "data        = {:?}",
        a.as_f32_slice()?,
    );

    let b =
        a.reshape(&[
            3,
            2,
        ])?;

    println!();
    println!("--- B = reshape(A) ---");

    println!(
        "shape       = {:?}",
        b.shape().dims(),
    );

    println!(
        "strides     = {:?}",
        b.strides().values(),
    );

    println!(
        "contiguous  = {}",
        b.is_contiguous(),
    );

    println!(
        "data        = {:?}",
        b.as_f32_slice()?,
    );

    let zero =
        Tensor::zeros(
            &context,
            &[2, 2],
        )?;

    println!();
    println!("--- Zeros ---");

    println!(
        "shape       = {:?}",
        zero.shape().dims(),
    );

    println!(
        "data        = {:?}",
        zero.as_f32_slice()?,
    );

    Ok(())
}
