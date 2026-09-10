use std::{fs, path::Path};

use tiny_metal_llm::{
    error::Result,
    import::{import_safetensors, load_llama_config, map_llama_weights},
    tensor::DType,
};

use crate::cli::{DTypeArg, ImportArgs};

pub fn execute(args: ImportArgs) -> Result<()> {
    let config_path = args.source.join("config.json");

    let weights_path = args.source.join("model.safetensors");

    println!("reading config...");

    let config = load_llama_config(config_path)?;

    println!("reading safetensors...");

    let source_weights = import_safetensors(weights_path)?;

    println!("source tensors: {}", source_weights.len(),);

    println!("converting Llama weights...");

    let weights = map_llama_weights(source_weights, config.num_layers)?;

    println!("converted tensors: {}", weights.len(),);

    fs::create_dir_all(&args.output)?;

    config.save_json(args.output.join("config.json"))?;

    let dtype = match args.dtype {
        DTypeArg::F32 => DType::F32,
        DTypeArg::F16 => DType::F16,
    };
    weights.save_as(args.output.join("model.bin"), dtype)?;

    println!("model imported successfully as {dtype:?}");

    println!("output: {}", args.output.display(),);

    copy_tokenizer_files(&args.source, &args.output)?;

    Ok(())
}

const TOKENIZER_FILES: &[&str] = &[
    "tokenizer.model",
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "vocab.json",
    "merges.txt",
];

fn copy_tokenizer_files(source: &Path, output: &Path) -> Result<()> {
    for file_name in TOKENIZER_FILES {
        let source_path = source.join(file_name);
        if !source_path.exists() {
            continue;
        }

        fs::copy(&source_path, output.join(file_name))?;
        println!("copied {file_name}");
    }

    Ok(())
}
