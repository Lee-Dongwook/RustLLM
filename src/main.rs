mod error;
mod generation;
mod import;
mod metal;
mod model;
mod nn;
mod ops;
mod tensor;
mod tokenizer;

use error::{
    Result,
    TinyError,
};

use generation::generate_greedy;

use metal::MetalContext;

use model::Transformer;

use tokenizer::Tokenizer;

fn main() -> Result<()> {
    let context =
        MetalContext::new();

    let model =
        Transformer::load(
            &context,
            "models/tiny",
        )?;

    let tokenizer =
        Tokenizer::load_json(
            "models/tiny/tokenizer.json",
        )?;

    if tokenizer.vocab_size()
        != model.config()
            .vocab_size
    {
        return Err(
            TinyError::Tokenizer(
                format!(
                    "tokenizer vocabulary size {} does not match model vocabulary size {}",
                    tokenizer.vocab_size(),
                    model.config().vocab_size,
                ),
            ),
        );
    }

    let prompt =
        "ab";

    let prompt_tokens =
        tokenizer.encode(
            prompt,
        )?;

    println!(
        "prompt        = {:?}",
        prompt,
    );

    println!(
        "prompt tokens = {:?}",
        prompt_tokens,
    );

    let generated =
        generate_greedy(
            &context,
            &model,
            &prompt_tokens,
            5,
            tokenizer.eos_token_id(),
        )?;

    println!(
        "all tokens    = {:?}",
        generated,
    );

    let text =
        tokenizer.decode(
            &generated,
            true,
        )?;

    println!(
        "text          = {:?}",
        text,
    );

    let continuation =
        tokenizer.decode(
            &generated[
                prompt_tokens.len()
                ..
            ],
            true,
        )?;

    println!(
        "continuation  = {:?}",
        continuation,
    );

    Ok(())
}
