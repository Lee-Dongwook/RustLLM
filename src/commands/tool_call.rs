use tiny_metal_llm::{
    chat::{Conversation, templates::SmolLm2Template},
    document::{ChunkConfig, chunk_document, load_document},
    error::{Result, TinyError},
    generation::GenerationConfig,
    metal::MetalContext,
    model::Transformer,
    retrieval::LexicalRetriever,
    structured::StructuredRetryConfig,
    tensor::DType,
    tokenizer::{ModelTokenizer, Tokenizer},
    tool_calling::execute_tool_calling,
    tools::{
        ToolRegistry,
        builtin::{CalculatorTool, DocumentSearchTool},
    },
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

    /*
     * Built-in calculator는 항상 사용 가능.
     */
    registry.register(CalculatorTool::new()?)?;

    /*
     * --document가 주어진 경우에만
     * document_search Tool을 활성화한다.
     */
    if let Some(document_path) = args.document.as_ref() {
        eprintln!("loading document: {}", document_path.display());

        let document = load_document(document_path)?;

        let chunk_config = ChunkConfig::new(args.chunk_size, args.overlap)?;

        let chunks = chunk_document(&document, &chunk_config)?;

        eprintln!("document chunks: {}", chunks.len());

        let retriever = LexicalRetriever::new(chunks)?;

        let document_search = DocumentSearchTool::new(retriever, args.retrieve_top_k)?;

        registry.register(document_search)?;
    }

    let mut tool_names = registry.iter().map(|tool| tool.name()).collect::<Vec<_>>();

    tool_names.sort_unstable();

    eprintln!("available tools:");

    for name in tool_names {
        eprintln!("  - {name}");
    }

    eprintln!();

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
