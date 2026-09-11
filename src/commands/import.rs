use std::{fs, path::Path};

use tiny_metal_llm::{
    error::Result,
    import::{import_safetensors, load_llama_config, map_llama_weights},
    model::quantize_model_i8,
    tensor::DType,
};

use crate::cli::{ImportArgs, WeightFormat};

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

    let model_path = args.output.join("model.bin");

    match args.weight_format {
        WeightFormat::F32 => weights.save_as(&model_path, DType::F32)?,
        WeightFormat::F16 => weights.save_as(&model_path, DType::F16)?,
        WeightFormat::Int8 => {
            println!("quantizing linear weights...");
            let weights = quantize_model_i8(weights, config.num_layers)?;
            // Mixed F16 / INT8 / F32(scale) storage must be preserved.
            weights.save(&model_path)?;
        }
    }

    println!("model imported successfully");
    println!("weight format: {:?}", args.weight_format);

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
