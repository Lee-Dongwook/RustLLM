use crate::error::Result;

use super::Conversation;

pub trait ChatTemplate {
    fn render(&self, conversation: &Conversation, add_generation_prompt: bool) -> Result<String>;
}

#[cfg(test)]
mod tests {
    use super::ChatTemplate;
    use crate::{chat::Conversation, error::Result};

    struct TestTemplate;

    impl ChatTemplate for TestTemplate {
        fn render(
            &self,
            conversation: &Conversation,
            add_generation_prompt: bool,
        ) -> Result<String> {
            let mut output = String::new();

            for message in conversation.messages() {
                output.push_str(message.role().as_str());
                output.push(':');
                output.push_str(message.content());
                output.push('\n');
            }

            if add_generation_prompt {
                output.push_str("assistant:");
            }

            Ok(output)
        }
    }

    #[test]
    fn renders_conversation_through_template() {
        let mut conversation = Conversation::with_system("Be helpful.");

        conversation.push_user("Hello");

        let rendered = TestTemplate.render(&conversation, true).unwrap();

        assert_eq!(
            rendered,
            concat!("system:Be helpful.\n", "user:Hello\n", "assistant:",)
        );
    }

    #[test]
    fn can_render_without_generation_prompt() {
        let mut conversation = Conversation::new();

        conversation.push_user("Hello");

        let rendered = TestTemplate.render(&conversation, false).unwrap();

        assert_eq!(rendered, "user:Hello\n");
    }
}
