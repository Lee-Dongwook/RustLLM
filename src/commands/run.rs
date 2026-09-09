use tiny_metal_llm::{
    error::{
        Result,
        TinyError,
    }, generation::generate_greedy, metal::MetalContext, model::Transformer,
    tokenizer::{SentencePieceTokenizer, Tokenizer},
};

use crate::cli::RunArgs;

pub fn execute(
    args: RunArgs,
) -> Result<()> {
    let context =
        MetalContext::new()?;
    
    eprintln!(
        "loading model..."
    );

    let model =
        Transformer::load(
            &context,
            &args.model,
        )?;
    
    let tokenizer =
        SentencePieceTokenizer::from_model_dir(
            &args.model,
        )?;

    if tokenizer.vocab_size()
        != model.config().vocab_size
    {
        return Err(
            TinyError::Tokenizer(
                format!(
                    "tokenizer vocab size {} does not match model vocab size {}",
                    tokenizer.vocab_size(),
                    model.config().vocab_size,
                ),
            ),
        );
    }

    let prompt_tokens =
        tokenizer.encode(
            &args.prompt,
        )?;

    if prompt_tokens.is_empty() {
        return Err(
            TinyError::Tokenizer(
                "prompt produced no tokens"
                    .to_string(),
            ),
        );
    }

    if prompt_tokens.len()
        > model.config().max_seq_len
    {
        return Err(
            TinyError::ModelFormat(
                format!(
                    "prompt is too long: {} tokens; context limit is {}",
                    prompt_tokens.len(),
                    model.config().max_seq_len,
                ),
            ),
        );
    }

    let remaining_context =
        model.config()
            .max_seq_len
            - prompt_tokens.len();

    let max_new_tokens =
        args.max_tokens
            .min(
                remaining_context,
            );

    eprintln!(
        "prompt tokens: {}",
        prompt_tokens.len(),
    );

    eprintln!(
        "generating up to {} tokens...",
        max_new_tokens,
    );

    let generated =
        generate_greedy(
            &context,
            &model,
            &prompt_tokens,
            max_new_tokens,
            tokenizer
                .eos_token_id(),
        )?;

    let text =
        tokenizer.decode(
            &generated,
            true,
        )?;

    println!(
        "{text}"
    );

    Ok(())
}
