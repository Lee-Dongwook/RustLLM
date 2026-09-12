use tiny_metal_llm::{
    chat::{Conversation, templates::SmolLm2Template},
    document::{ChunkConfig, chunk_document, load_document},
    error::{Result, TinyError},
    generation::GenerationConfig,
    metal::MetalContext,
    model::Transformer,
    rag::generate_rag,
    retrieval::LexicalRetriever,
    tensor::DType,
    tokenizer::{ModelTokenizer, Tokenizer},
};

use crate::cli::{DTypeArg, RagArgs};

pub fn execute(args: RagArgs) -> Result<()> {
    /*
     * ------------------------------------
     * Runtime / Model
     * ------------------------------------
     */
    let context = MetalContext::new()?;

    eprintln!("loading model...");

    let model = Transformer::load(&context, &args.model)?;

    validate_dtype(&model, args.dtype)?;

    eprintln!("model dtype: {:?}", model.dtype(),);

    if model.has_int8_weights() {
        eprintln!("weight format: INT8 (F16 activations)");
    }

    /*
     * ------------------------------------
     * Tokenizer
     * ------------------------------------
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
     * ------------------------------------
     * Document
     * ------------------------------------
     */
    eprintln!("loading document: {}", args.document.display(),);

    let document = load_document(&args.document)?;

    let chunk_config = ChunkConfig::new(args.chunk_size, args.overlap)?;

    let chunks = chunk_document(&document, &chunk_config)?;

    eprintln!("document chars: {}", document.len(),);

    eprintln!("chunks: {}", chunks.len(),);

    /*
     * ------------------------------------
     * Retriever
     * ------------------------------------
     */
    let retriever = LexicalRetriever::new(chunks)?;

    /*
     * ------------------------------------
     * Conversation
     * ------------------------------------
     */
    let conversation = match args.system.as_deref() {
        Some(system) if !system.trim().is_empty() => Conversation::with_system(system),

        _ => Conversation::with_system("Answer questions using the provided context."),
    };

    let template = SmolLm2Template::new();

    /*
     * ------------------------------------
     * Generation
     * ------------------------------------
     */
    let generation_config = GenerationConfig {
        max_new_tokens: args.max_tokens,

        temperature: args.temperature,

        top_k: args.top_k,

        top_p: args.top_p,

        seed: args.seed,
    };

    generation_config.validate()?;

    eprintln!("retrieving top {} chunks...", args.retrieve_top_k,);

    let result = generate_rag(
        &context,
        &model,
        &tokenizer,
        &template,
        &retriever,
        &conversation,
        &args.question,
        args.retrieve_top_k,
        &generation_config,
    )?;

    /*
     * ------------------------------------
     * Retrieval diagnostics
     *
     * stderr로 보내서 stdout에는
     * 최종 답변만 남길 수 있게 한다.
     * ------------------------------------
     */
    eprintln!();
    eprintln!("retrieved sources:");

    for (rank, retrieved) in result.retrieved().iter().enumerate() {
        let chunk = retrieved.chunk();

        eprintln!(
            "  {}. {} [chunk {}, chars {}..{}] score={:.4}",
            rank + 1,
            chunk.source().display(),
            chunk.index(),
            chunk.start_char(),
            chunk.end_char(),
            retrieved.score(),
        );

        eprintln!("     {}", chunk.text().replace('\n', " "),);
    }

    eprintln!();

    /*
     * stdout = 실제 답변
     */
    println!("{}", result.answer(),);

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
