use crate::error::Result;

use super::super::{ChatTemplate, Conversation};

const IM_START: &str = "<|im_start|>";
const IM_END: &str = "<|im_end|>";

const DEFAULT_SYSTEM_PROMPT: &str =
    "You are a helpful AI assistant named SmolLM, trained by Hugging Face";

#[derive(Debug, Clone, Copy, Default)]
pub struct SmolLm2Template;

impl SmolLm2Template {
    pub fn new() -> Self {
        Self
    }
}

impl ChatTemplate for SmolLm2Template {
    fn render(&self, conversation: &Conversation, add_generation_prompt: bool) -> Result<String> {
        let messages = conversation.messages();

        let mut output = String::new();

        if messages
            .first()
            .is_some_and(|message| message.role().as_str() != "system")
        {
            push_message(&mut output, "system", DEFAULT_SYSTEM_PROMPT);
        }

        for message in messages {
            push_message(&mut output, message.role().as_str(), message.content());
        }

        if add_generation_prompt {
            output.push_str(IM_START);
            output.push_str("assistant\n");
        }

        Ok(output)
    }
}

fn push_message(output: &mut String, role: &str, content: &str) {
    output.push_str(IM_START);
    output.push_str(role);
    output.push('\n');

    output.push_str(content);

    output.push_str(IM_END);
    output.push('\n');
}

#[cfg(test)]
mod tests {
    use super::SmolLm2Template;

    use crate::{
        chat::{ChatTemplate, Conversation},
        tokenizer::{ModelTokenizer, Tokenizer},
    };

    #[test]
    fn renders_explicit_system_message() {
        let mut conversation = Conversation::with_system("You are concise.");

        conversation.push_user("Hello");

        let rendered = SmolLm2Template::new().render(&conversation, true).unwrap();

        assert_eq!(
            rendered,
            concat!(
                "<|im_start|>system\n",
                "You are concise.",
                "<|im_end|>\n",
                "<|im_start|>user\n",
                "Hello",
                "<|im_end|>\n",
                "<|im_start|>assistant\n",
            )
        );
    }

    #[test]
    fn inserts_default_system_prompt_when_missing() {
        let mut conversation = Conversation::new();

        conversation.push_user("Hello");

        let rendered = SmolLm2Template::new().render(&conversation, true).unwrap();

        assert_eq!(
            rendered,
            concat!(
                "<|im_start|>system\n",
                "You are a helpful AI assistant named SmolLM, trained by Hugging Face",
                "<|im_end|>\n",
                "<|im_start|>user\n",
                "Hello",
                "<|im_end|>\n",
                "<|im_start|>assistant\n",
            )
        );
    }

    #[test]
    fn renders_multi_turn_conversation() {
        let mut conversation = Conversation::with_system("Be helpful.");

        conversation.push_user("Hello");

        conversation.push_assistant("Hi!");

        conversation.push_user("What did I say?");

        let rendered = SmolLm2Template::new().render(&conversation, true).unwrap();

        assert_eq!(
            rendered,
            concat!(
                "<|im_start|>system\n",
                "Be helpful.",
                "<|im_end|>\n",
                "<|im_start|>user\n",
                "Hello",
                "<|im_end|>\n",
                "<|im_start|>assistant\n",
                "Hi!",
                "<|im_end|>\n",
                "<|im_start|>user\n",
                "What did I say?",
                "<|im_end|>\n",
                "<|im_start|>assistant\n",
            )
        );
    }

    #[test]
    fn does_not_append_generation_prompt_when_disabled() {
        let mut conversation = Conversation::with_system("Be helpful.");

        conversation.push_user("Hello");

        conversation.push_assistant("Hi!");

        let rendered = SmolLm2Template::new().render(&conversation, false).unwrap();

        assert_eq!(
            rendered,
            concat!(
                "<|im_start|>system\n",
                "Be helpful.",
                "<|im_end|>\n",
                "<|im_start|>user\n",
                "Hello",
                "<|im_end|>\n",
                "<|im_start|>assistant\n",
                "Hi!",
                "<|im_end|>\n",
            )
        );
    }
    fn create_test_tokenizer_dir() -> std::path::PathBuf {
        use std::{
            fs,
            time::{SystemTime, UNIX_EPOCH},
        };

        let unique = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();

        let model_dir = std::env::temp_dir().join(format!(
            "rustllm-chat-template-test-{unique}-{}",
            std::process::id(),
        ));

        fs::create_dir_all(&model_dir).unwrap();

        fs::write(
            model_dir.join("tokenizer.json"),
            r#"{
            "version": "1.0",
            "truncation": null,
            "padding": null,
            "added_tokens": [
                {
                    "id": 0,
                    "content": "<unk>",
                    "single_word": false,
                    "lstrip": false,
                    "rstrip": false,
                    "normalized": false,
                    "special": true
                },
                {
                    "id": 5,
                    "content": "<|im_start|>",
                    "single_word": false,
                    "lstrip": false,
                    "rstrip": false,
                    "normalized": false,
                    "special": true
                },
                {
                    "id": 6,
                    "content": "<|im_end|>",
                    "single_word": false,
                    "lstrip": false,
                    "rstrip": false,
                    "normalized": false,
                    "special": true
                }
            ],
            "normalizer": null,
            "pre_tokenizer": {
                "type": "Whitespace"
            },
            "post_processor": null,
            "decoder": null,
            "model": {
                "type": "WordLevel",
                "vocab": {
                    "<unk>": 0,
                    "system": 1,
                    "user": 2,
                    "assistant": 3,
                    "Hello": 4
                },
                "unk_token": "<unk>"
            }
        }"#,
        )
        .unwrap();

        fs::write(
            model_dir.join("tokenizer_config.json"),
            r#"{
            "bos_token": null,
            "eos_token": "<|im_end|>"
        }"#,
        )
        .unwrap();

        model_dir
    }
    #[test]
    fn rendered_chat_preserves_special_token_boundaries() {
        let model_dir = create_test_tokenizer_dir();

        let tokenizer = ModelTokenizer::from_model_dir(&model_dir).unwrap();

        let mut conversation = Conversation::with_system("Be helpful.");

        conversation.push_user("Hello");

        let prompt = SmolLm2Template::new().render(&conversation, true).unwrap();

        let ids = tokenizer.encode(&prompt).unwrap();

        /*
         * Expected template:
         *
         * <|im_start|>system
         * ...
         * <|im_end|>
         *
         * <|im_start|>user
         * ...
         * <|im_end|>
         *
         * <|im_start|>assistant
         *
         *
         * 따라서:
         *
         * im_start = 3
         * im_end   = 2
         */
        assert_eq!(ids.iter().filter(|&&id| id == 5).count(), 3,);

        assert_eq!(ids.iter().filter(|&&id| id == 6).count(), 2,);

        assert_eq!(tokenizer.eos_token_id(), Some(6),);

        std::fs::remove_dir_all(model_dir).unwrap();
    }
}
