use std::{
    io::{self, Write},
    time::Instant,
};

use tiny_metal_llm::{
    benchmark::BenchmarkStats,
    error::{Result, TinyError},
    generation::{GenerationConfig, generate_stream, generate_stream_profiled},
    metal::MetalContext,
    model::Transformer,
    profile::DecodeProfile,
    tensor::DType,
    tokenizer::{ModelTokenizer, StreamingDecoder, Tokenizer},
};

use crate::cli::{DTypeArg, RunArgs};

pub fn execute(args: RunArgs) -> Result<()> {
    let context = MetalContext::new()?;
    let model_path = args.model.join("model.bin");
    let model_size_bytes = std::fs::metadata(&model_path)?.len();

    eprintln!("loading model...");

    // This includes disk loading plus Metal buffer creation, matching the
    // time a user waits before generation can begin.
    let load_started = Instant::now();
    let model = Transformer::load(&context, &args.model)?;
    let load_time = load_started.elapsed();
    let dtype_matches = match args.dtype {
        DTypeArg::F32 => model.dtype() == DType::F32,
        // INT8 weight-only packages still execute with F16 activations.
        DTypeArg::F16 => model.dtype() == DType::F16,
    };
    if !dtype_matches {
        return Err(TinyError::ModelFormat(format!(
            "model activation dtype does not match --dtype {:?}",
            args.dtype,
        )));
    }

    eprintln!("model dtype: {:?}", model.dtype());
    if model.has_int8_weights() {
        eprintln!("weight format: INT8 (F16 activations)");
    }

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

    if !args.benchmark {
        print!("{}", args.prompt,);
        io::stdout().flush()?;
    }

    let mut decoder = StreamingDecoder::new(&tokenizer);

    let generation_config = GenerationConfig {
        max_new_tokens,
        temperature: args.temperature,
        top_k: args.top_k,
        top_p: args.top_p,
        seed: args.seed,
    };
    generation_config.validate()?;
    let mut profile = args.profile_decode.then(DecodeProfile::default);
    let output = if let Some(profile) = profile.as_mut() {
        generate_stream_profiled(
            &context,
            &model,
            &prompt_tokens,
            &generation_config,
            tokenizer.eos_token_id(),
            |token_id| {
                if args.benchmark {
                    return Ok(());
                }
                let delta = decoder.push(token_id)?;
                if !delta.is_empty() {
                    print!("{delta}");
                    io::stdout().flush()?;
                }
                Ok(())
            },
            profile,
        )?
    } else {
        generate_stream(
            &context,
            &model,
            &prompt_tokens,
            &generation_config,
            tokenizer.eos_token_id(),
            |token_id| {
                if args.benchmark {
                    return Ok(());
                }
                let delta = decoder.push(token_id)?;
                if !delta.is_empty() {
                    print!("{delta}");
                    io::stdout().flush()?;
                }
                Ok(())
            },
        )?
    };

    if !args.benchmark {
        println!();
    }

    eprintln!("generated tokens: {}", output.metrics.generated_tokens,);

    if args.benchmark {
        BenchmarkStats {
            model_size_bytes,
            load_time,
            prompt_tokens: output.metrics.prompt_tokens,
            prefill_time: output.metrics.prefill_duration,
            generated_tokens: output.metrics.generated_tokens,
            decode_time: output.metrics.decode_forward_duration,
            total_generation_time: output.metrics.total_duration,
        }
        .print();
    }

    if let Some(profile) = profile.as_ref() {
        profile.print();
        profile.metal.print(profile.sampled_tokens);
    }

    if args.metrics && !args.benchmark {
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
