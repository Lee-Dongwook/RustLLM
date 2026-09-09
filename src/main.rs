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

    let b =
        Tensor::from_f32_slice(
            &context,
            &[
                7.0, 8.0,
                9.0, 10.0,
                11.0, 12.0,
            ],
            &[3, 2],
        )?;

    let c =
        a.matmul(
            &context,
            &b,
        )?;

    println!(
        "shape = {:?}",
        c.shape().dims(),
    );

    println!(
        "data  = {:?}",
        c.as_f32_slice()?,
    );

    Ok(())
}
