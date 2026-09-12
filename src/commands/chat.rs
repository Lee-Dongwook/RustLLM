use std::io::{self, Write};

use tiny_metal_llm::{
    chat::{Conversation, generate_chat_stream, templates::SmolLm2Template},
    error::{Result, TinyError},
    generation::GenerationConfig,
    metal::MetalContext,
    model::Transformer,
    tensor::DType,
    tokenizer::{ModelTokenizer, StreamingDecoder, Tokenizer},
};

use crate::cli::{ChatArgs, DTypeArg};

pub fn execute(args: ChatArgs) -> Result<()> {
    let context = MetalContext::new()?;

    eprintln!("loading model....");

    let model = Transformer::load(&context, &args.model)?;

    validate_dtype(&model, args.dtype)?;

    eprintln!("model dtype: {:?}", model.dtype(),);

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

    let template = SmolLm2Template::new();

    let mut conversation = new_conversation(args.system.as_deref());

    println!();
    println!("RustLLM Chat");
    println!("Type /exit to quit, /clear to reset.");
    println!();

    loop {
        print!("> ");
        io::stdout().flush()?;

        let mut input = String::new();

        let bytes_read = io::stdin().read_line(&mut input)?;

        if bytes_read == 0 {
            println!();
            break;
        }

        let input = input.trim_end_matches(&['\r', '\n'][..]);

        if input == "/exit" || input == "/quit" {
            break;
        }

        if input == "/clear" {
            conversation = new_conversation(args.system.as_deref());

            println!("conversation cleared.");
            println!();

            continue;
        }

        if input.trim().is_empty() {
            continue;
        }

        let mut turn = conversation.clone();

        turn.push_user(input);

        let generation_config = GenerationConfig {
            max_new_tokens: args.max_tokens,
            temperature: args.temperature,
            top_k: args.top_k,
            top_p: args.top_p,
            seed: args.seed,
        };

        generation_config.validate()?;

        let mut decoder = StreamingDecoder::new(&tokenizer);

        let mut assistant_text = String::new();

        print!("assistant> ");
        io::stdout().flush()?;

        generate_chat_stream(
            &context,
            &model,
            &tokenizer,
            &template,
            &turn,
            &generation_config,
            |token_id| {
                let delta = decoder.push(token_id)?;
                if !delta.is_empty() {
                    print!("{delta}");
                    io::stdout().flush()?;

                    assistant_text.push_str(&delta);
                }

                Ok(())
            },
        )?;

        println!();
        println!();

        turn.push_assistant(assistant_text);

        conversation = turn;
    }

    Ok(())
}

fn new_conversation(system: Option<&str>) -> Conversation {
    match system {
        Some(system) if !system.trim().is_empty() => Conversation::with_system(system),

        _ => Conversation::new(),
    }
}

fn validate_dtype(model: &Transformer, dtype: DTypeArg) -> Result<()> {
    let matches = match dtype {
        DTypeArg::F32 => model.dtype() == DType::F32,

        DTypeArg::F16 => {
            /*
             * INT8 weight-only 모델도
             * activation dtype은 F16이다.
             */
            model.dtype() == DType::F16
        }
    };

    if !matches {
        return Err(TinyError::ModelFormat(format!(
            "model activation dtype does not match --dtype {dtype:?}",
        )));
    }

    Ok(())
}
