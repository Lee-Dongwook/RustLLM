use tiny_metal_llm::{
    agent::{AgentConfig, run_agent},
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
    tools::{
        ToolRegistry,
        builtin::{CalculatorTool, DocumentSearchTool},
    },
};

use crate::cli::{AgentArgs, DTypeArg};

pub fn execute(args: AgentArgs) -> Result<()> {
    let context = MetalContext::new()?;

    eprintln!("loading model...");

    let model = Transformer::load(&context, &args.model)?;

    validate_dtype(&model, args.dtype)?;

    let tokenizer = ModelTokenizer::from_model_dir(&args.model)?;

    if tokenizer.vocab_size() != model.config().vocab_size {
        return Err(TinyError::Tokenizer(format!(
            "tokenizer vocab size {} does not match model vocab size {}",
            tokenizer.vocab_size(),
            model.config().vocab_size,
        )));
    }

    let mut registry = ToolRegistry::new();

    registry.register(CalculatorTool::new()?)?;

    if let Some(document_path) = args.document.as_ref() {
        eprintln!("loading document: {}", document_path.display(),);

        let document = load_document(document_path)?;

        let chunks = chunk_document(&document, &ChunkConfig::new(args.chunk_size, args.overlap)?)?;

        eprintln!("document chunks: {}", chunks.len(),);

        let retriever = LexicalRetriever::new(chunks)?;

        registry.register(DocumentSearchTool::new(retriever, args.retrieve_top_k)?)?;
    }

    let mut tool_names = registry.iter().map(|tool| tool.name()).collect::<Vec<_>>();

    tool_names.sort_unstable();

    eprintln!("available tools:");

    for name in tool_names {
        eprintln!("  - {name}");
    }

    let conversation =
        Conversation::with_system("You are a helpful assistant that can use tools when needed.");

    let template = SmolLm2Template::new();

    let generation_config = GenerationConfig {
        max_new_tokens: args.max_tokens,

        temperature: args.temperature,

        top_k: args.top_k,

        top_p: args.top_p,

        seed: args.seed,
    };

    generation_config.validate()?;

    let retry_config = StructuredRetryConfig::new(args.attempts)?;

    let agent_config = AgentConfig::new(args.max_steps)?;

    eprintln!();
    eprintln!("running agent...");

    let output = run_agent(
        &context,
        &model,
        &tokenizer,
        &template,
        &conversation,
        &registry,
        &args.request,
        &agent_config,
        &generation_config,
        &retry_config,
    )?;

    eprintln!();
    eprintln!("agent trace:");

    for step in output.trace().steps() {
        eprintln!("  step {}", step.index() + 1,);

        eprintln!("    tool: {}", step.tool_name(),);

        eprintln!("    arguments: {}", step.arguments(),);

        eprintln!("    result: {}", step.result(),);
    }

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
