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

    prompt.push_str("Use the context to answer the question.\n\n");

    prompt.push_str("Context:\n");

    for result in retrieved {
        prompt.push_str(result.chunk().text().trim());

        prompt.push_str("\n\n");
    }

    prompt.push_str("Question:\n");

    prompt.push_str(question);

    prompt.push_str("\nAnswer:");

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
