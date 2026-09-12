use crate::error::Result;

use super::super::{
    ChatTemplate,
    Conversation,
};

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
    fn render(
        &self,
        conversation: &Conversation,
        add_generation_prompt: bool,
    ) -> Result<String> {
        let messages = conversation.messages();

        let mut output = String::new();

        if messages.first().is_some_and(|message| {
            message.role().as_str() != "system"
        }) {
            push_message(
                &mut output,
                "system",
                DEFAULT_SYSTEM_PROMPT,
            );
        }

        for message in messages {
            push_message(
                &mut output,
                message.role().as_str(),
                message.content(),
            );
        }

        if add_generation_prompt {
            output.push_str(IM_START);
            output.push_str("assistant\n");
        }

        Ok(output)
    }
}

fn push_message(
    output: &mut String,
    role: &str,
    content: &str,
) {
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

    use crate::chat::{
        ChatTemplate,
        Conversation,
    };

    #[test]
    fn renders_explicit_system_message() {
        let mut conversation =
            Conversation::with_system(
                "You are concise.",
            );

        conversation.push_user(
            "Hello",
        );

        let rendered =
            SmolLm2Template::new()
                .render(
                    &conversation,
                    true,
                )
                .unwrap();

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
        let mut conversation =
            Conversation::new();

        conversation.push_user(
            "Hello",
        );

        let rendered =
            SmolLm2Template::new()
                .render(
                    &conversation,
                    true,
                )
                .unwrap();

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
        let mut conversation =
            Conversation::with_system(
                "Be helpful.",
            );

        conversation.push_user(
            "Hello",
        );

        conversation.push_assistant(
            "Hi!",
        );

        conversation.push_user(
            "What did I say?",
        );

        let rendered =
            SmolLm2Template::new()
                .render(
                    &conversation,
                    true,
                )
                .unwrap();

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
        let mut conversation =
            Conversation::with_system(
                "Be helpful.",
            );

        conversation.push_user(
            "Hello",
        );

        conversation.push_assistant(
            "Hi!",
        );

        let rendered =
            SmolLm2Template::new()
                .render(
                    &conversation,
                    false,
                )
                .unwrap();

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
}
