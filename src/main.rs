mod error;
mod generation;
mod import;
mod metal;
mod model;
mod nn;
mod ops;
mod tensor;
mod tokenizer;

use std::{
    env,
    fs,
    path::{
        Path,
        PathBuf,
    },
};

use error::{Result, TinyError};

use import::{import_safetensors, load_llama_config, map_llama_weights};

const DEFAULT_SOURCE_DIR: &str = "models/source/tinystories-llama-15m";

const DEFAULT_OUTPUT_DIR: &str = "models/tinystories-llama-15m";

fn main() {
    if let Err(error) = run() {
        eprintln!(
            "error: {error}",
        );
        std::process::exit(1);
    }
}

fn run() -> Result<()> {
    let (source_dir, output_dir) = parse_paths()?;

    ensure_distinct_directories(
        &source_dir,
        &output_dir,
    )?;

    let config = load_llama_config(source_dir.join("config.json"))?;

    let source = import_safetensors(source_dir.join("model.safetensors"))?;

    println!("source tensors = {}", source.len(),);

    let converted = map_llama_weights(source, config.num_layers)?;

    println!("converted tensors = {}", converted.len(),);

    fs::create_dir_all(&output_dir)?;

    config.save_json(output_dir.join("config.json"))?;

    converted.save(output_dir.join("model.bin"))?;

    let saved = model::ModelWeights::load(output_dir.join("model.bin"))?;

    if saved.len() != converted.len() {
        return Err(TinyError::ModelFormat(format!(
            "saved tensor count {} does not match converted count {}",
            saved.len(),
            converted.len(),
        )));
    }

    println!("saved converted model to {}", output_dir.display(),);

    println!("verified {} tensors", saved.len(),);

    Ok(())
}

fn parse_paths() -> Result<(PathBuf, PathBuf)> {
    let args: Vec<String> = env::args().skip(1).collect();

    match args.as_slice() {
        [] => Ok((
            PathBuf::from(DEFAULT_SOURCE_DIR),
            PathBuf::from(DEFAULT_OUTPUT_DIR),
        )),

        [source_dir, output_dir] => Ok((PathBuf::from(source_dir), PathBuf::from(output_dir))),

        _ => Err(TinyError::InvalidArgument(
            "usage: cargo run -- [<source-model-dir> <output-model-dir>]".to_string(),
        )),
    }
}

fn ensure_distinct_directories(
    source_dir: &Path,
    output_dir: &Path,
) -> Result<()> {
    if source_dir == output_dir {
        return Err(TinyError::InvalidArgument(
            "source and output model directories must be different"
                .to_string(),
        ));
    }

    if output_dir.exists()
        && source_dir.canonicalize()? == output_dir.canonicalize()?
    {
        return Err(TinyError::InvalidArgument(
            "source and output model directories resolve to the same directory"
                .to_string(),
        ));
    }

    Ok(())
}
