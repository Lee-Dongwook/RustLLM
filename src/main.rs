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

    let tensor =
        Tensor::from_f32_slice(
            &context,
            &[
                1.0,
                2.0,
                3.0,
                4.0,
            ],
            &[2, 2],
        )?;

    println!(
        "GPU    = {}",
        context.device.name(),
    );

    println!(
        "shape  = {:?}",
        tensor.shape().dims(),
    );

    println!(
        "rank   = {}",
        tensor.shape().rank(),
    );

    println!(
        "numel  = {}",
        tensor.shape().numel(),
    );

    println!(
        "dtype  = {:?}",
        tensor.dtype(),
    );

    println!(
        "device = {:?}",
        tensor.device(),
    );

    println!(
        "data   = {:?}",
        tensor.as_f32_slice()?,
    );

    Ok(())
}
