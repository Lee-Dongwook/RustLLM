use std::io::{self, Write};

use tiny_metal_llm::{
    error::{Result, TinyError},
    generation::{GenerationConfig, generate_stream},
    metal::MetalContext,
    model::Transformer,
    tokenizer::{SentencePieceTokenizer, StreamingDecoder, Tokenizer},
};

use crate::cli::RunArgs;

pub fn execute(args: RunArgs) -> Result<()> {
    let context = MetalContext::new()?;

    eprintln!("loading model...");

    let model = Transformer::load(&context, &args.model)?;

    let tokenizer = SentencePieceTokenizer::from_model_dir(&args.model)?;

    if tokenizer.vocab_size() != model.config().vocab_size {
        return Err(TinyError::Tokenizer(format!(
            "tokenizer vocab size {} does not match model vocab size {}",
            tokenizer.vocab_size(),
            model.config().vocab_size,
        )));
    }

    let prompt_tokens = tokenizer.encode(&args.prompt)?;

    if prompt_tokens.is_empty() {
        return Err(TinyError::Tokenizer(
            "prompt produced no tokens".to_string(),
        ));
    }

    if prompt_tokens.len() > model.config().max_seq_len {
        return Err(TinyError::ModelFormat(format!(
            "prompt is too long: {} tokens; context limit is {}",
            prompt_tokens.len(),
            model.config().max_seq_len,
        )));
    }

    let remaining_context = model.config().max_seq_len - prompt_tokens.len();

    let max_new_tokens = args.max_tokens.min(remaining_context);

    eprintln!("prompt tokens: {}", prompt_tokens.len(),);

    // ---------------------------------------------
    // 실제 사용자가 보는 출력 시작
    // ---------------------------------------------

    print!("{}", args.prompt,);

    io::stdout().flush()?;

    let mut decoder = StreamingDecoder::new(&tokenizer);

    let mut generated_count = 0usize;

    let generation_config = GenerationConfig {
        max_new_tokens,
        temperature: args.temperature,
        top_k: args.top_k,
        top_p: args.top_p,
        seed: args.seed,
    };
    generation_config.validate()?;
    generate_stream(
        &context,
        &model,
        &prompt_tokens,
        &generation_config,
        tokenizer.eos_token_id(),
        |token_id| {
            generated_count += 1;

            let delta = decoder.push(token_id)?;

            if !delta.is_empty() {
                print!("{delta}");

                io::stdout().flush()?;
            }

            Ok(())
        },
    )?;

    println!();

    eprintln!("generated tokens: {}", generated_count,);

    Ok(())
}
