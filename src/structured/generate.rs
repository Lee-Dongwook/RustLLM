use serde::de::DeserializeOwned;

use crate::{
    chat::{ChatTemplate, Conversation, generate_chat_stream},
    error::{Result, TinyError},
    generation::{GenerationConfig, GenerationOutput},
    metal::MetalContext,
    model::Transformer,
    tokenizer::Tokenizer,
};

use super::{JsonSchema, build_json_prompt, parse_json};
#[derive(Debug)]
pub struct RawStructuredGeneration {
    raw_text: String,
    generation: GenerationOutput,
}

impl RawStructuredGeneration {
    pub fn raw_text(&self) -> &str {
        &self.raw_text
    }

    pub fn generation(&self) -> &GenerationOutput {
        &self.generation
    }

    pub fn into_parts(self) -> (String, GenerationOutput) {
        (self.raw_text, self.generation)
    }
}

#[derive(Debug)]
pub struct StructuredGeneration<T> {
    value: T,
    raw_text: String,
    generation: GenerationOutput,
}

impl<T> StructuredGeneration<T> {
    pub(crate) fn new(value: T, raw_text: String, generation: GenerationOutput) -> Self {
        Self {
            value,
            raw_text,
            generation,
        }
    }
    pub fn value(&self) -> &T {
        &self.value
    }

    pub fn raw_text(&self) -> &str {
        &self.raw_text
    }

    pub fn generation(&self) -> &GenerationOutput {
        &self.generation
    }

    pub fn into_value(self) -> T {
        self.value
    }

    pub fn into_parts(self) -> (T, String, GenerationOutput) {
        (self.value, self.raw_text, self.generation)
    }
}

pub fn generate_structured_raw(
    context: &MetalContext,
    model: &Transformer,
    tokenizer: &dyn Tokenizer,
    template: &dyn ChatTemplate,
    conversation: &Conversation,
    task: &str,
    schema: &JsonSchema,
    config: &GenerationConfig,
) -> Result<RawStructuredGeneration> {
    let turn = prepare_structured_conversation(conversation, task, schema)?;

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

    let raw_text = tokenizer.decode(&generated_tokens, true)?;

    if raw_text.trim().is_empty() {
        return Err(TinyError::StructuredOutput(
            "model produced an empty structured response".to_string(),
        ));
    }

    Ok(RawStructuredGeneration {
        raw_text,
        generation,
    })
}

pub fn prepare_structured_conversation(
    conversation: &Conversation,
    task: &str,
    schema: &JsonSchema,
) -> Result<Conversation> {
    let prompt = build_json_prompt(task, schema)?;

    let mut turn = conversation.clone();

    turn.push_user(prompt);

    Ok(turn)
}

pub fn generate_structured<T>(
    context: &MetalContext,
    model: &Transformer,
    tokenizer: &dyn Tokenizer,
    template: &dyn ChatTemplate,
    conversation: &Conversation,
    task: &str,
    schema: &JsonSchema,
    config: &GenerationConfig,
) -> Result<StructuredGeneration<T>>
where
    T: DeserializeOwned,
{
    let raw = generate_structured_raw(
        context,
        model,
        tokenizer,
        template,
        conversation,
        task,
        schema,
        config,
    )?;

    let (raw_text, generation) = raw.into_parts();

    let value = parse_json::<T>(&raw_text).map_err(|error| {
        TinyError::StructuredOutput(format!("{error}; raw model output: {raw_text:?}"))
    })?;

    Ok(StructuredGeneration::new(value, raw_text, generation))
}

#[cfg(test)]
mod tests {
    use super::prepare_structured_conversation;

    use crate::{
        chat::{Conversation, Role},
        structured::JsonSchema,
    };

    #[test]
    fn prepares_structured_task_as_user_message() {
        let conversation = Conversation::with_system("You are a helpful assistant.");

        let schema = JsonSchema::new(
            "classification",
            r#"{
                    "category": "string",
                    "priority": "string"
                }"#,
        )
        .unwrap();

        let prepared = prepare_structured_conversation(
            &conversation,
            "Classify this issue: application crashes after login.",
            &schema,
        )
        .unwrap();

        assert_eq!(prepared.len(), 2,);

        let message = prepared.last().unwrap();

        assert_eq!(message.role(), Role::User,);

        assert!(message.content().contains("Classify this issue"));

        assert!(message.content().contains("Return ONLY valid JSON."));

        assert!(message.content().contains("\"category\": \"string\""));

        assert!(message.content().contains("\"priority\": \"string\""));
    }

    #[test]
    fn does_not_mutate_original_conversation() {
        let conversation = Conversation::with_system("Be concise.");

        let schema = JsonSchema::new(
            "answer",
            r#"{
                    "answer": "string"
                }"#,
        )
        .unwrap();

        let prepared =
            prepare_structured_conversation(&conversation, "Answer the question.", &schema)
                .unwrap();

        assert_eq!(conversation.len(), 1,);

        assert_eq!(prepared.len(), 2,);
    }
}
