use tiny_metal_llm::{
    chat::{Conversation, templates::SmolLm2Template},
    error::{Result, TinyError},
    generation::GenerationConfig,
    metal::MetalContext,
    model::Transformer,
    structured::StructuredRetryConfig,
    tensor::DType,
    tokenizer::{ModelTokenizer, Tokenizer},
    tool_calling::execute_tool_calling,
    tools::{ToolRegistry, builtin::CalculatorTool},
};

use crate::cli::{DTypeArg, ToolCallArgs};

pub fn execute(args: ToolCallArgs) -> Result<()> {
    /*
     * Runtime
     */
    let context = MetalContext::new()?;

    eprintln!("loading model...");

    let model = Transformer::load(&context, &args.model)?;

    validate_dtype(&model, args.dtype)?;

    /*
     * Tokenizer
     */
    let tokenizer = ModelTokenizer::from_model_dir(&args.model)?;

    if tokenizer.vocab_size() != model.config().vocab_size {
        return Err(TinyError::Tokenizer(format!(
            "tokenizer vocab size {} does not match model vocab size {}",
            tokenizer.vocab_size(),
            model.config().vocab_size,
        )));
    }

    /*
     * Tools
     */
    let mut registry = ToolRegistry::new();

    registry.register(CalculatorTool::new()?)?;

    /*
     * Chat
     */
    let conversation = Conversation::with_system("You are a helpful assistant.");

    let template = SmolLm2Template::new();

    /*
     * Generation
     */
    let generation_config = GenerationConfig {
        max_new_tokens: args.max_tokens,

        temperature: args.temperature,

        top_k: args.top_k,

        top_p: args.top_p,

        seed: args.seed,
    };

    generation_config.validate()?;

    let retry_config = StructuredRetryConfig::new(args.attempts)?;

    eprintln!("generating tool call...");

    let output = execute_tool_calling(
        &context,
        &model,
        &tokenizer,
        &template,
        &conversation,
        &registry,
        &args.request,
        &generation_config,
        &retry_config,
    )?;

    let generated = output.generated_call();

    let call = generated.call();

    eprintln!();
    eprintln!("selected tool: {}", generated.selection_raw(),);

    eprintln!("raw arguments: {}", generated.arguments_raw(),);

    eprintln!();
    eprintln!("tool call:");

    eprintln!("  name: {}", call.name(),);

    eprintln!(
        "  arguments: {}",
        serde_json::to_string_pretty(call.arguments(),).map_err(|error| {
            TinyError::Tool(format!("failed to serialize tool arguments: {error}"))
        })?,
    );

    eprintln!();
    eprintln!("tool result:");

    eprintln!(
        "{}",
        serde_json::to_string_pretty(output.tool_result().output(),).map_err(|error| {
            TinyError::Tool(format!("failed to serialize tool result: {error}"))
        })?,
    );

    eprintln!();
    eprintln!("final answer:");

    println!("{}", output.answer(),);

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
