mod context;
mod generate;

pub use context::build_rag_prompt;

pub use generate::{RagGeneration, generate_rag, prepare_rag_conversation};
