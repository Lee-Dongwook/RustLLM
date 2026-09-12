use serde::de::DeserializeOwned;

use crate::{
    chat::{ChatTemplate, Conversation},
    error::{Result, TinyError},
    generation::GenerationConfig,
    metal::MetalContext,
    model::Transformer,
    tokenizer::Tokenizer,
};

use super::{JsonSchema, StructuredGeneration, generate_structured_raw};

use super::generate::parse_structured_value;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct StructuredRetryConfig {
    max_attempts: usize,
}

impl StructuredRetryConfig {
    pub fn new(max_attempts: usize) -> Result<Self> {
        if max_attempts == 0 {
            return Err(TinyError::InvalidArgument(
                "structured output max_attempts must be greater than zero".to_string(),
            ));
        }

        Ok(Self { max_attempts })
    }

    pub fn max_attempts(&self) -> usize {
        self.max_attempts
    }
}

impl Default for StructuredRetryConfig {
    fn default() -> Self {
        Self { max_attempts: 3 }
    }
}

fn build_repair_task(original_task: &str, previous_output: &str, parse_error: &str) -> String {
    format!(
        r#"The previous response did not satisfy the required JSON output contract.

Original task:
{original_task}

Previous invalid response:
{previous_output}

Validation error:
{parse_error}

Correct the previous response so that every field matches the required type.
Use actual values from the original task.
Never use schema type words such as "string", "number", "boolean", "array", or "object" as placeholder values.
Do not explain the error.
Do not include markdown.
Return only the corrected JSON object."#
    )
}

pub fn generate_structured_with_retry<T>(
    context: &MetalContext,
    model: &Transformer,
    tokenizer: &dyn Tokenizer,
    template: &dyn ChatTemplate,
    conversation: &Conversation,
    task: &str,
    schema: &JsonSchema,
    generation_config: &GenerationConfig,
    retry_config: &StructuredRetryConfig,
) -> Result<StructuredGeneration<T>>
where
    T: DeserializeOwned,
{
    let mut current_task = task.to_string();

    for attempt in 1..=retry_config.max_attempts() {
        let raw = generate_structured_raw(
            context,
            model,
            tokenizer,
            template,
            conversation,
            &current_task,
            schema,
            generation_config,
        )?;

        let (raw_text, generation) = raw.into_parts();

        match parse_structured_value::<T>(&raw_text, schema) {
            Ok(value) => {
                return Ok(StructuredGeneration::new(value, raw_text, generation));
            }

            Err(error) => {
                let error_message = error.to_string();

                if attempt == retry_config.max_attempts() {
                    return Err(TinyError::StructuredOutput(format!(
                        "structured generation failed after {attempt} attempts: \
                                 {error_message}; \
                                 last model output: {raw_text:?}"
                    )));
                }

                current_task = build_repair_task(task, &raw_text, &error_message);
            }
        }
    }

    Err(TinyError::StructuredOutput(
        "structured generation failed unexpectedly".to_string(),
    ))
}

#[cfg(test)]
mod tests {
    use super::{StructuredRetryConfig, build_repair_task};

    #[test]
    fn default_retry_config_uses_three_attempts() {
        let config = StructuredRetryConfig::default();

        assert_eq!(config.max_attempts(), 3,);
    }

    #[test]
    fn rejects_zero_attempts() {
        let result = StructuredRetryConfig::new(0);

        assert!(result.is_err(),);
    }

    #[test]
    fn accepts_custom_attempt_count() {
        let config = StructuredRetryConfig::new(5).unwrap();

        assert_eq!(config.max_attempts(), 5,);
    }

    #[test]
    fn repair_task_contains_previous_failure() {
        let task = build_repair_task(
            "Classify this issue.",
            r#"{"priority":"high""#,
            "unexpected end of JSON",
        );

        assert!(task.contains("Classify this issue."));

        assert!(task.contains(r#"{"priority":"high""#));

        assert!(task.contains("unexpected end of JSON"));

        assert!(task.contains("Return only the corrected JSON object."));
    }
}
