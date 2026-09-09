use tiny_metal_llm::{
    error::{
        Result,
        TinyError,
    },

    metal::MetalContext,

    model::Transformer,
};

use crate::cli::RunArgs;

pub fn execute(
    args: RunArgs,
) -> Result<()> {
    let context = 
        MetalContext::new();
    
    println!(
        "loading model..."
    );

    let model =
        Transformer::load(
            &context,
            &args.model,
        )?;
    
    println!(
        "model loaded"
    );

    println!(
        "  layers: {}",
        model.config()
            .num_layers,
    );

    println!(
        "  hidden: {}",
        model.config()
            .hidden_size,
    );

    println!(
        "prompt: {:?}",
        args.prompt,
    );

    Err(
        TinyError::Tokenizer(
            "SentencePiece tokenizer support is required before text generation".to_string(),
        ),
    )
}
