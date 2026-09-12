use crate::{
    error::{Result, TinyError},
    retrieval::RetrievedChunk,
};

pub fn build_rag_prompt(question: &str, retrieved: &[RetrievedChunk]) -> Result<String> {
    let question = question.trim();

    if question.is_empty() {
        return Err(TinyError::InvalidArgument(
            "RAG question cannot be empty".to_string(),
        ));
    }

    if retrieved.is_empty() {
        return Err(TinyError::InvalidArgument(
            "RAG requires at least one retrieved chunk".to_string(),
        ));
    }

    let mut prompt = String::new();

    prompt.push_str("Answer the question using only the retrieved context below.\n");

    prompt.push_str(
        "If the retrieved context does not contain enough information, say that the information is not available in the provided context.\n",
    );

    prompt.push_str("Do not invent facts that are not supported by the context.\n\n");

    prompt.push_str("Retrieved context:\n\n");

    for result in retrieved {
        let chunk = result.chunk();

        prompt.push_str(&format!(
            "[Source: {}, chunk {}]\n",
            chunk.source().display(),
            chunk.index(),
        ));

        prompt.push_str(chunk.text().trim());

        prompt.push_str("\n\n");
    }

    prompt.push_str("Question:\n");

    prompt.push_str(question);

    Ok(prompt)
}

#[cfg(test)]
mod tests {
    use super::build_rag_prompt;

    use crate::{document::DocumentChunk, retrieval::RetrievedChunk};

    #[test]
    fn builds_rag_prompt_from_retrieved_chunks() {
        let retrieved = vec![
            RetrievedChunk::new(
                DocumentChunk::new(0, "rust.md", 0, 37, "Rust uses ownership to manage memory."),
                1.42,
            ),
            RetrievedChunk::new(
                DocumentChunk::new(2, "rust.md", 100, 145, "Cargo is Rust's package manager."),
                0.31,
            ),
        ];

        let prompt = build_rag_prompt("How does Rust manage memory?", &retrieved).unwrap();

        assert!(prompt.contains("[Source: rust.md, chunk 0]"));

        assert!(prompt.contains("Rust uses ownership to manage memory."));

        assert!(prompt.contains("How does Rust manage memory?"));

        assert!(prompt.contains("Do not invent facts"));
    }

    #[test]
    fn rejects_empty_question() {
        let retrieved = vec![RetrievedChunk::new(
            DocumentChunk::new(0, "test.txt", 0, 4, "test"),
            1.0,
        )];

        let result = build_rag_prompt("   ", &retrieved);

        assert!(result.is_err());
    }

    #[test]
    fn rejects_empty_retrieval_results() {
        let result = build_rag_prompt("What is Rust?", &[]);

        assert!(result.is_err());
    }
}
