mod error;
mod generation;
mod import;
mod metal;
mod model;
mod nn;
mod ops;
mod tensor;
mod tokenizer;

use error::Result;
use import::inspect_safetensors;

fn main() -> Result<()> {
    let tensors =
        inspect_safetensors(
            "models/source/tinystories-llama-15m/model.safetensors",
        )?;

    println!(
        "tensor count = {}",
        tensors.len(),
    );

    for tensor in tensors {
        println!(
            "{} | {:?} | {:?} | {} elements",
            tensor.name,
            tensor.dtype,
            tensor.shape,
            tensor.numel,
        );
    }

    Ok(())
}
