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

    let scores =
        Tensor::from_f32_slice(
            &context,
            &[
                1.0, 2.0, 3.0,
                4.0, 5.0, 6.0,
                7.0, 8.0, 9.0,
            ],
            &[1, 1, 3, 3],
        )?;

    let masked =
        scores.attention_scale_mask(
            &context,
            0.5,
            0,
        )?;

    println!(
        "shape  = {:?}",
        masked.shape().dims(),
    );

    println!(
        "masked = {:?}",
        masked.as_f32_slice()?,
    );

    let probs =
        masked.softmax_last_dim(
            &context,
        )?;

    println!(
        "softmax = {:?}",
        probs.as_f32_slice()?,
    );

    Ok(())
}
