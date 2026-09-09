mod error;
mod generation;
mod import;
mod metal;
mod model;
mod nn;
mod ops;
mod tensor;
mod tokenizer;

use std::fs;

use error::Result;

use import::{
    import_safetensors,
    map_llama_weights,
};

fn main() -> Result<()> {
    let source =
        import_safetensors(
            "models/source/tinystories-llama-15m/model.safetensors",
        )?;

    println!(
        "source tensors = {}",
        source.len(),
    );

    let converted =
        map_llama_weights(
            source,
            6,
        )?;

    println!(
        "converted tensors = {}",
        converted.len(),
    );

    fs::create_dir_all(
        "models/tinystories-llama-15m",
    )?;

    converted.save(
        "models/tinystories-llama-15m/model.bin",
    )?;

    println!(
        "saved converted model"
    );

    Ok(())
}
