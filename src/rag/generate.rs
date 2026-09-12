use crate::{
    chat::{ChatTemplate, Conversation, generate_chat_stream},
    error::{Result, TinyError},
    generation::{GenerationConfig, GenerationOutput},
    metal::MetalContext,
    model::Transformer,
    retrieval::{RetrievedChunk, Retriever},
    tokenizer::Tokenizer,
};

use super::build_rag_prompt;

#[derive(Debug)]
pub struct RagGeneration {
    answer: String,
    retrieved: Vec<RetrievedChunk>,
    generation: GenerationOutput,
}

impl RagGeneration {
    pub fn answer(&self) -> &str {
        &self.answer
    }

    pub fn retrieved(&self) -> &[RetrievedChunk] {
        &self.retrieved
    }

    pub fn generation(&self) -> &GenerationOutput {
        &self.generation
    }

    pub fn into_parts(self) -> (String, Vec<RetrievedChunk>, GenerationOutput) {
        (self.answer, self.retrieved, self.generation)
    }
}

pub fn prepare_rag_conversation(
    conversation: &Conversation,
    question: &str,
    retrieved: &[RetrievedChunk],
) -> Result<Conversation> {
    let prompt = build_rag_prompt(question, retrieved)?;

    let mut turn = conversation.clone();

    turn.push_user(prompt);

    Ok(turn)
}

pub fn generate_rag(
    context: &MetalContext,
    model: &Transformer,
    tokenizer: &dyn Tokenizer,
    template: &dyn ChatTemplate,
    retriever: &dyn Retriever,
    conversation: &Conversation,
    question: &str,
    top_k: usize,
    config: &GenerationConfig,
) -> Result<RagGeneration> {
    if top_k == 0 {
        return Err(TinyError::InvalidArgument(
            "RAG top_k must be greater than zero".to_string(),
        ));
    }

    /*
     * 1. 질문으로 관련 chunk 검색
     */
    let retrieved = retriever.retrieve(question, top_k)?;

    if retrieved.is_empty() {
        return Err(TinyError::InvalidArgument(
            "no relevant document chunks were retrieved".to_string(),
        ));
    }

    /*
     * 2. retrieval 결과를 Conversation으로 변환
     */
    let turn = prepare_rag_conversation(conversation, question, &retrieved)?;

    /*
     * 3. generated token 수집
     */
    let mut generated_tokens = Vec::new();

    let generation = generate_chat_stream(
        context,
        model,
        tokenizer,
        template,
        &turn,
        config,
        |token_id| {
            generated_tokens.push(token_id);

            Ok(())
        },
    )?;

    /*
     * 4. assistant token → text
     */
    let answer = tokenizer.decode(&generated_tokens, true)?;

    let answer = answer.trim().to_string();

    if answer.is_empty() {
        return Err(TinyError::InvalidArgument(
            "RAG generation produced an empty answer".to_string(),
        ));
    }

    Ok(RagGeneration {
        answer,
        retrieved,
        generation,
    })
}

#[cfg(test)]
mod tests {
    use super::prepare_rag_conversation;

    use crate::{
        chat::{Conversation, Role},
        document::DocumentChunk,
        retrieval::RetrievedChunk,
    };

    #[test]
    fn prepares_rag_question_as_user_message() {
        let conversation = Conversation::with_system("You answer questions about documentation.");

        let retrieved = vec![RetrievedChunk::new(
            DocumentChunk::new(0, "rust.md", 0, 37, "Rust uses ownership to manage memory."),
            1.4,
        )];

        let prepared =
            prepare_rag_conversation(&conversation, "How does Rust manage memory?", &retrieved)
                .unwrap();

        assert_eq!(conversation.len(), 1,);

        assert_eq!(prepared.len(), 2,);

        let message = prepared.last().unwrap();

        assert_eq!(message.role(), Role::User,);

        assert!(message.content().contains("Rust uses ownership"));

        assert!(message.content().contains("How does Rust manage memory?"));
    }

    #[test]
    fn does_not_mutate_original_conversation() {
        let conversation = Conversation::with_system("Be concise.");

        let retrieved = vec![RetrievedChunk::new(
            DocumentChunk::new(0, "test.txt", 0, 4, "test"),
            1.0,
        )];

        let _ = prepare_rag_conversation(&conversation, "What is this?", &retrieved).unwrap();

        assert_eq!(conversation.len(), 1,);
    }
}
