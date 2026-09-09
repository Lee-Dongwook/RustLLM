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

    let data: Vec<f32> =
        (1..=24)
            .map(|value| {
                value as f32
            })
            .collect();

    let x =
        Tensor::from_f32_slice(
            &context,
            &data,
            &[2, 3, 4],
        )?;

    println!(
        "--- Original ---"
    );

    println!(
        "shape      = {:?}",
        x.shape().dims(),
    );

    println!(
        "strides    = {:?}",
        x.strides().values(),
    );

    println!(
        "contiguous = {}",
        x.is_contiguous(),
    );

    let y =
        x.permute(
            &[1, 0, 2],
        )?;

    println!();
    println!(
        "--- Permuted ---"
    );

    println!(
        "shape      = {:?}",
        y.shape().dims(),
    );

    println!(
        "strides    = {:?}",
        y.strides().values(),
    );

    println!(
        "contiguous = {}",
        y.is_contiguous(),
    );

    let y =
        y.contiguous(
            &context,
        )?;

    println!();
    println!(
        "--- Materialized ---"
    );

    println!(
        "shape      = {:?}",
        y.shape().dims(),
    );

    println!(
        "strides    = {:?}",
        y.strides().values(),
    );

    println!(
        "contiguous = {}",
        y.is_contiguous(),
    );

    println!(
        "data       = {:?}",
        y.as_f32_slice()?,
    );

    Ok(())
}
