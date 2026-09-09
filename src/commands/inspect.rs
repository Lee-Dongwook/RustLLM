use tiny_metal_llm::{
    error::Result,
    model::ModelConfig,
};

use crate::cli::InspectArgs;

pub fn execute(
    args: InspectArgs,
) -> Result<()> {
    let config_path =
        args.model
            .join(
                "config.json"
            );
    
    let config = 
        ModelConfig::load_json(
            config_path,
    )?;

    println!(
        "Model"
    );

    println!(
        "  vocab size        : {}",
        config.vocab_size,
    );

    println!(
        "  hidden size       : {}",
        config.hidden_size,
    );

    println!(
        "  intermediate size : {}",
        config.intermediate_size,
    );

    println!(
        "  layers            : {}",
        config.num_layers,
    );

    println!(
        "  attention heads   : {}",
        config.num_heads,
    );

    println!(
        "  head dimension    : {}",
        config.head_dim(),
    );

    println!(
        "  context length    : {}",
        config.max_seq_len,
    );

    println!(
        "  RMSNorm epsilon   : {}",
        config.rms_norm_eps,
    );

    println!(
        "  RoPE theta        : {}",
        config.rope_theta,
    );

    Ok(())
}
