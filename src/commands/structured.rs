use tiny_metal_llm::{
    chat::{Conversation, templates::SmolLm2Template},
    error::{Result, TinyError},
    generation::GenerationConfig,
    metal::MetalContext,
    model::Transformer,
    structured::{JsonSchema, StructuredRetryConfig, generate_structured_with_retry},
    tensor::DType,
    tokenizer::{ModelTokenizer, Tokenizer},
};

use crate::cli::{DTypeArg, StructuredArgs};

pub fn execute(args: StructuredArgs) -> Result<()> {
    let context = MetalContext::new()?;

    eprintln!("loading model...");

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

    let conversation = match args.system.as_deref() {
        Some(system) if !system.trim().is_empty() => Conversation::with_system(system),

        _ => Conversation::new(),
    };

    let schema = JsonSchema::new("cli_output", &args.schema)?;

    let generation_config = GenerationConfig {
        max_new_tokens: args.max_tokens,

        temperature: args.temperature,

        top_k: args.top_k,

        top_p: args.top_p,

        seed: args.seed,
    };

    generation_config.validate()?;

    let retry_config = StructuredRetryConfig::new(args.attempts)?;

    /*
     * CLI에서는 결과 타입을 컴파일 타임에 알 수 없기 때문에
     * serde_json::Value로 받는다.
     *
     * Library API에서는 여전히 사용자가 원하는 Rust struct를
     * T로 사용할 수 있다.
     */
    let output = generate_structured_with_retry::<serde_json::Value>(
        &context,
        &model,
        &tokenizer,
        &template,
        &conversation,
        &args.task,
        &schema,
        &generation_config,
        &retry_config,
    )?;

    let pretty = serde_json::to_string_pretty(output.value()).map_err(|error| {
        TinyError::StructuredOutput(format!("failed to serialize structured result: {error}"))
    })?;

    println!("{pretty}");

    Ok(())
}

fn validate_dtype(model: &Transformer, dtype: DTypeArg) -> Result<()> {
    let matches = match dtype {
        DTypeArg::F32 => model.dtype() == DType::F32,

        DTypeArg::F16 => model.dtype() == DType::F16,
    };

    if !matches {
        return Err(TinyError::ModelFormat(format!(
            "model activation dtype does not match --dtype {dtype:?}",
        )));
    }

    Ok(())
}
