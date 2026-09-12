use crate::{
    error::{Result, TinyError},
    generation::{GenerationConfig, GenerationOutput, generate_stream},
    metal::MetalContext,
    model::Transformer,
    tokenizer::Tokenizer,
};

use super::{ChatTemplate, Conversation};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PreparedChatPrompt {
    text: String,
    token_ids: Vec<u32>,
}

impl PreparedChatPrompt {
    pub fn text(&self) -> &str {
        &self.text
    }

    pub fn token_ids(&self) -> &[u32] {
        &self.token_ids
    }

    pub fn len(&self) -> usize {
        self.token_ids.len()
    }

    pub fn is_empty(&self) -> bool {
        self.token_ids.is_empty()
    }
}

pub fn prepare_chat_prompt(
    tokenizer: &dyn Tokenizer,
    template: &dyn ChatTemplate,
    conversation: &Conversation,
) -> Result<PreparedChatPrompt> {
    let text = template.render(conversation, true)?;

    let token_ids = tokenizer.encode(&text)?;

    if token_ids.is_empty() {
        return Err(TinyError::Tokenizer(
            "chat prompt produced no tokens".to_string(),
        ));
    }

    Ok(PreparedChatPrompt { text, token_ids })
}

pub fn generate_chat_stream<F>(
    context: &MetalContext,
    model: &Transformer,
    tokenizer: &dyn Tokenizer,
    template: &dyn ChatTemplate,
    conversation: &Conversation,
    config: &GenerationConfig,
    on_token: F,
) -> Result<GenerationOutput>
where
    F: FnMut(u32) -> Result<()>,
{
    if tokenizer.vocab_size() != model.config().vocab_size {
        return Err(TinyError::Tokenizer(format!(
            "tokenizer vocab size {} does not match model vocab size {}",
            tokenizer.vocab_size(),
            model.config().vocab_size,
        )));
    }

    let prompt = prepare_chat_prompt(tokenizer, template, conversation)?;

    let max_seq_len = model.config().max_seq_len;

    if prompt.len() >= max_seq_len {
        return Err(TinyError::PositionOutOfRange {
            start_pos: 0,
            seq_len: prompt.len(),
            max_seq_len,
        });
    }

    let remaining_context = max_seq_len - prompt.len();

    let generation_config = GenerationConfig {
        max_new_tokens: config.max_new_tokens.min(remaining_context),
        temperature: config.temperature,
        top_k: config.top_k,
        top_p: config.top_p,
        seed: config.seed,
    };

    generation_config.validate()?;

    generate_stream(
        context,
        model,
        prompt.token_ids(),
        &generation_config,
        tokenizer.eos_token_id(),
        on_token,
    )
}

#[cfg(test)]
mod tests {
    use std::cell::RefCell;

    use super::prepare_chat_prompt;

    use crate::{
        chat::{Conversation, templates::SmolLm2Template},
        error::Result,
        tokenizer::Tokenizer,
    };

    #[derive(Default)]
    struct RecordingTokenizer {
        encoded_text: RefCell<Option<String>>,
    }

    impl Tokenizer for RecordingTokenizer {
        fn encode(&self, text: &str) -> Result<Vec<u32>> {
            *self.encoded_text.borrow_mut() = Some(text.to_string());

            /*
             * 여기서는 실제 tokenizer를
             * 검증하는 게 아니다.
             *
             * Chat → Tokenizer 연결만
             * 검증한다.
             */
            Ok(vec![10, 20, 30])
        }

        fn decode(&self, _token_ids: &[u32], _skip_special_tokens: bool) -> Result<String> {
            Ok(String::new())
        }

        fn vocab_size(&self) -> usize {
            100
        }

        fn bos_token_id(&self) -> Option<u32> {
            None
        }

        fn eos_token_id(&self) -> Option<u32> {
            Some(99)
        }
    }

    #[test]
    fn prepares_chat_prompt_from_conversation() {
        let tokenizer = RecordingTokenizer::default();

        let template = SmolLm2Template::new();

        let mut conversation = Conversation::with_system("Be concise.");

        conversation.push_user("Hello");

        let prepared = prepare_chat_prompt(&tokenizer, &template, &conversation).unwrap();

        assert_eq!(
            prepared.text(),
            concat!(
                "<|im_start|>system\n",
                "Be concise.",
                "<|im_end|>\n",
                "<|im_start|>user\n",
                "Hello",
                "<|im_end|>\n",
                "<|im_start|>assistant\n",
            ),
        );

        assert_eq!(prepared.token_ids(), &[10, 20, 30],);

        /*
         * ChatTemplate이 만든 바로 그 문자열이
         * tokenizer.encode()에 전달됐는지도 검증.
         */
        assert_eq!(
            tokenizer.encoded_text.borrow().as_deref(),
            Some(prepared.text()),
        );
    }

    #[test]
    fn prepared_prompt_reports_token_length() {
        let tokenizer = RecordingTokenizer::default();

        let template = SmolLm2Template::new();

        let mut conversation = Conversation::new();

        conversation.push_user("Hello");

        let prepared = prepare_chat_prompt(&tokenizer, &template, &conversation).unwrap();

        assert_eq!(prepared.len(), 3,);

        assert!(!prepared.is_empty(),);
    }
}
