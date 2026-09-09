mod error;
mod metal;
mod nn;
mod ops;
mod tensor;

use error::Result;
use metal::MetalContext;
use nn::RotaryEmbedding;
use tensor::Tensor;

fn main() -> Result<()> {
    let context =
        MetalContext::new();

    let rope =
        RotaryEmbedding::new(
            &context,
            4,
            16,
            10_000.0,
        )?;

    // shape:
    //
    // [batch=1, heads=1, seq=2, head_dim=4]
    let x =
        Tensor::from_f32_slice(
            &context,
            &[
                // position 0
                1.0, 2.0, 3.0, 4.0,

                // position 1
                1.0, 2.0, 3.0, 4.0,
            ],
            &[1, 1, 2, 4],
        )?;

    let y =
        rope.forward(
            &context,
            &x,
            0,
        )?;

    println!(
        "shape  = {:?}",
        y.shape().dims(),
    );

    println!(
        "output = {:?}",
        y.as_f32_slice()?,
    );

    Ok(())
}
