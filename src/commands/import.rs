use std::fs;

use tiny_metal_llm::{
    error::Result,

    import::{
        import_safetensors,
        load_llama_config,
        map_llama_weights,
    },
};

use crate::cli::ImportArgs;

pub fn execute(
    args: ImportArgs,
) -> Result<()> {
    let config_path = 
        args.source
            .join(
                "config.json",
            );
    
    let weights_path =
        args.source
            .join(
                "model.safetensors",
            );
    
    println!(
        "reading config..."
    );

    let config =
        load_llama_config(
            config_path,
        )?;
    
    println!(
        "reading safetensors..."
    );

    let source_weights =
        import_safetensors(
            weights_path,
        )?;
    
    println!(
        "source tensors: {}",
        source_weights.len(),
    );

    println!(
        "converting Llama weights..."
    );

    let weights =
        map_llama_weights(
            source_weights,
            config.num_layers,
        )?;
    
    println!(
        "converted tensors: {}",
        weights.len(),
    );

    fs::create_dir_all(
        &args.output,
    )?;

    config.save_json(
        args.output
            .join(
                "config.json",
            ),
    )?;

    weights.save(
        args.output
            .join(
                "model.bin",
            ),
    )?;

    println!(
        "model imported successfully"
    );

    println!(
        "output: {}",
        args.output.display(),
    );

    for file_name in [
        "tokenizer.model",
        "tokenizer_config.json",
        "special_tokens_map.json",
    ] {
        let source =
            args.source.join(
                file_name,
            );

        if source.exists() {
            let target =
                args.output.join(
                    file_name,
                );

            fs::copy(
                &source,
                &target,
            )?;

            println!(
                "copied {file_name}",
            );
        }
    }

    Ok(())
}
