use std::io::{self, Write};

use tiny_metal_llm::{
    error::{Result, TinyError},
    generation::{GenerationConfig, generate_stream},
    metal::MetalContext,
    model::Transformer,
    tensor::DType,
    tokenizer::{ModelTokenizer, StreamingDecoder, Tokenizer},
};

use crate::cli::{DTypeArg, RunArgs};

pub fn execute(args: RunArgs) -> Result<()> {
    let context = MetalContext::new()?;

    eprintln!("loading model...");

    let model = Transformer::load(&context, &args.model)?;
    let dtype_matches = match args.dtype {
        DTypeArg::F32 => model.dtype() == DType::F32,
        DTypeArg::F16 => model.dtype() == DType::F16 && !model.has_int8_weights(),
        DTypeArg::Int8 => model.has_int8_weights(),
    };
    if !dtype_matches {
        return Err(TinyError::ModelFormat(format!(
            "model package does not match --dtype {:?}; re-import with the requested dtype",
            args.dtype,
        )));
    }

    eprintln!("model dtype: {:?}", model.dtype());

    let tokenizer = ModelTokenizer::from_model_dir(&args.model)?;

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
    let output = generate_stream(
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

    if args.metrics {
        let metrics = &output.metrics;
        eprintln!("\nGeneration metrics");
        eprintln!("  prompt tokens       : {}", metrics.prompt_tokens);
        eprintln!("  generated tokens    : {}", metrics.generated_tokens);
        eprintln!(
            "  prefill             : {:.2} ms",
            metrics.prefill_duration.as_secs_f64() * 1_000.0
        );
        eprintln!(
            "  prefill speed       : {:.2} tok/s",
            metrics.prefill_tokens_per_second()
        );
        eprintln!(
            "  decode forward      : {:.2} ms",
            metrics.decode_forward_duration.as_secs_f64() * 1_000.0
        );
        eprintln!(
            "  decode speed        : {:.2} tok/s",
            metrics.decode_tokens_per_second()
        );
        eprintln!(
            "  generation speed    : {:.2} tok/s",
            metrics.generation_tokens_per_second()
        );
        eprintln!(
            "  total               : {:.2} ms",
            metrics.total_duration.as_secs_f64() * 1_000.0
        );
    }

    Ok(())
}
