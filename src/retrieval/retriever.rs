use crate::{document::DocumentChunk, error::Result};

#[derive(Debug, Clone)]
pub struct RetrievedChunk {
    chunk: DocumentChunk,
    score: f32,
}

impl RetrievedChunk {
    pub fn new(chunk: DocumentChunk, score: f32) -> Self {
        Self { chunk, score }
    }

    pub fn chunk(&self) -> &DocumentChunk {
        &self.chunk
    }

    pub fn score(&self) -> f32 {
        self.score
    }

    pub fn into_chunk(self) -> DocumentChunk {
        self.chunk
    }
}

pub trait Retriever {
    fn retrieve(&self, query: &str, top_k: usize) -> Result<Vec<RetrievedChunk>>;
}
