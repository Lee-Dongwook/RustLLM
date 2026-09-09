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

    // batch 0
    //
    // [1 2]
    // [3 4]
    //
    // batch 1
    //
    // [5 6]
    // [7 8]
    let a =
        Tensor::from_f32_slice(
            &context,
            &[
                1.0, 2.0,
                3.0, 4.0,

                5.0, 6.0,
                7.0, 8.0,
            ],
            &[2, 2, 2],
        )?;

    // batch 0
    //
    // [1 0]
    // [0 1]
    //
    // batch 1
    //
    // [2 0]
    // [0 2]
    let b =
        Tensor::from_f32_slice(
            &context,
            &[
                1.0, 0.0,
                0.0, 1.0,

                2.0, 0.0,
                0.0, 2.0,
            ],
            &[2, 2, 2],
        )?;

    let c =
        a.batched_matmul(
            &context,
            &b,
        )?;

    println!(
        "A shape = {:?}",
        a.shape().dims(),
    );

    println!(
        "B shape = {:?}",
        b.shape().dims(),
    );

    println!(
        "C shape = {:?}",
        c.shape().dims(),
    );

    println!(
        "C data  = {:?}",
        c.as_f32_slice()?,
    );

    Ok(())
}
