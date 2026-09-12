use serde_json::Value;

use crate::{
    chat::{ChatTemplate, Conversation, generate_chat_stream},
    error::{Result, TinyError},
    generation::{GenerationConfig, GenerationOutput},
    metal::MetalContext,
    model::Transformer,
    structured::{StructuredRetryConfig, generate_structured_raw, parse_json},
    tokenizer::Tokenizer,
    tools::{Tool, ToolCall, ToolRegistry},
};

use super::{build_tool_arguments_task, build_tool_selection_prompt};

#[derive(Debug)]
pub struct GeneratedToolArguments {
    arguments: Value,
    raw_text: String,
    generation: GenerationOutput,
}

impl GeneratedToolArguments {
    pub fn arguments(&self) -> &Value {
        &self.arguments
    }

    pub fn raw_text(&self) -> &str {
        &self.raw_text
    }

    pub fn generation(&self) -> &GenerationOutput {
        &self.generation
    }

    pub fn into_parts(self) -> (Value, String, GenerationOutput) {
        (self.arguments, self.raw_text, self.generation)
    }
}

#[derive(Debug)]
pub struct GeneratedToolCall {
    call: ToolCall,

    selection_raw: String,
    arguments_raw: String,

    selection_generation: GenerationOutput,
    arguments_generation: GenerationOutput,
}

impl GeneratedToolCall {
    pub fn call(&self) -> &ToolCall {
        &self.call
    }

    pub fn selection_raw(&self) -> &str {
        &self.selection_raw
    }

    pub fn arguments_raw(&self) -> &str {
        &self.arguments_raw
    }

    pub fn selection_generation(&self) -> &GenerationOutput {
        &self.selection_generation
    }

    pub fn arguments_generation(&self) -> &GenerationOutput {
        &self.arguments_generation
    }
}

pub fn generate_tool_call(
    context: &MetalContext,
    model: &Transformer,
    tokenizer: &dyn Tokenizer,
    template: &dyn ChatTemplate,
    conversation: &Conversation,
    registry: &ToolRegistry,
    user_request: &str,
    generation_config: &GenerationConfig,
    retry_config: &StructuredRetryConfig,
) -> Result<GeneratedToolCall> {
    /*
     * ------------------------------------
     * Stage 1
     * Tool Selection
     * ------------------------------------
     */
    let selection_prompt = build_tool_selection_prompt(user_request, registry)?;

    let mut selection_conversation = conversation.clone();

    selection_conversation.push_user(selection_prompt);

    /*
     * Tool 이름만 필요하므로 길게 생성할 이유가 없다.
     */
    let selection_config = GenerationConfig {
        max_new_tokens: 16,
        temperature: 0.0,
        top_k: None,
        top_p: 1.0,
        seed: generation_config.seed,
    };

    let mut selection_tokens = Vec::new();

    let selection_generation = generate_chat_stream(
        context,
        model,
        tokenizer,
        template,
        &selection_conversation,
        &selection_config,
        |token_id| {
            selection_tokens.push(token_id);

            Ok(())
        },
    )?;

    let selection_raw = tokenizer
        .decode(&selection_tokens, true)?
        .trim()
        .to_string();

    let tool_name = parse_tool_name(&selection_raw)?;

    let tool = registry.get(&tool_name).ok_or_else(|| {
        TinyError::Tool(format!(
            "model selected unknown tool `{tool_name}`; raw output: {selection_raw:?}"
        ))
    })?;

    let generated_arguments = generate_tool_arguments_for_tool(
        context,
        model,
        tokenizer,
        template,
        conversation,
        tool,
        user_request,
        generation_config,
        retry_config,
    )?;

    let (arguments, arguments_raw, arguments_generation) = generated_arguments.into_parts();

    /*
     * ------------------------------------
     * Stage 3
     * Assemble ToolCall
     * ------------------------------------
     */
    let call = ToolCall::new(tool_name, arguments);

    /*
     * 최종 semantic validation.
     */
    registry.validate_call(&call)?;

    Ok(GeneratedToolCall {
        call,
        selection_raw,
        arguments_raw,
        selection_generation,
        arguments_generation,
    })
}

fn parse_tool_name(raw: &str) -> Result<String> {
    let raw = raw.trim();

    if raw.is_empty() {
        return Err(TinyError::Tool(
            "model produced an empty tool selection".to_string(),
        ));
    }

    /*
     * 모델이:
     *
     * calculator
     *
     * 또는:
     *
     * "calculator"
     *
     * 둘 중 어느 형태를 내더라도 허용한다.
     */
    if raw.starts_with('"') && raw.ends_with('"') {
        return serde_json::from_str::<String>(raw).map_err(|error| {
            TinyError::Tool(format!("failed to parse tool name {raw:?}: {error}"))
        });
    }

    Ok(raw.to_string())
}

pub fn generate_tool_arguments_for_tool(
    context: &MetalContext,
    model: &Transformer,
    tokenizer: &dyn Tokenizer,
    template: &dyn ChatTemplate,
    conversation: &Conversation,
    tool: &dyn Tool,
    user_request: &str,
    generation_config: &GenerationConfig,
    retry_config: &StructuredRetryConfig,
) -> Result<GeneratedToolArguments> {
    let (arguments, raw_text, generation) = generate_tool_arguments(
        context,
        model,
        tokenizer,
        template,
        conversation,
        tool,
        user_request,
        generation_config,
        retry_config,
    )?;

    Ok(GeneratedToolArguments {
        arguments,
        raw_text,
        generation,
    })
}

fn generate_tool_arguments(
    context: &MetalContext,
    model: &Transformer,
    tokenizer: &dyn Tokenizer,
    template: &dyn ChatTemplate,
    conversation: &Conversation,
    tool: &dyn Tool,
    user_request: &str,
    generation_config: &GenerationConfig,
    retry_config: &StructuredRetryConfig,
) -> Result<(Value, String, GenerationOutput)> {
    let mut task = build_tool_arguments_task(user_request, tool)?;

    let attempts = retry_config.max_attempts();

    let mut last_error = None;

    for attempt in 0..attempts {
        let raw = generate_structured_raw(
            context,
            model,
            tokenizer,
            template,
            conversation,
            &task,
            tool.input_schema(),
            generation_config,
        )?;

        let (raw_text, generation) = raw.into_parts();

        let parsed = parse_json::<Value>(&raw_text);

        match parsed {
            Ok(value) => match tool.input_schema().coerce(&value) {
                Ok(coerced) => {
                    tool.input_schema().validate(&coerced)?;

                    return Ok((coerced, raw_text, generation));
                }

                Err(error) => {
                    last_error = Some(error.to_string());
                }
            },

            Err(error) => {
                last_error = Some(error.to_string());
            }
        }

        if attempt + 1 < attempts {
            task = format!(
                "The previous tool arguments were invalid.\n\
                     Correct them and return only the JSON arguments object.\n\n\
                     User request:\n{user_request}\n\n\
                     Tool:\n{}\n\n\
                     Previous output:\n{raw_text}\n\n\
                     Error:\n{}",
                tool.name(),
                last_error.as_deref().unwrap_or("unknown validation error"),
            );
        }
    }

    Err(TinyError::Tool(format!(
        "failed to generate valid arguments for tool `{}` after {} attempts: {}",
        tool.name(),
        attempts,
        last_error.unwrap_or_else(|| { "unknown error".to_string() }),
    )))
}

#[cfg(test)]
mod tests {
    use super::parse_tool_name;

    #[test]
    fn parses_plain_tool_name() {
        assert_eq!(parse_tool_name("calculator").unwrap(), "calculator",);
    }

    #[test]
    fn parses_json_string_tool_name() {
        assert_eq!(parse_tool_name("\"calculator\"").unwrap(), "calculator",);
    }

    #[test]
    fn trims_tool_name() {
        assert_eq!(parse_tool_name("  calculator  ").unwrap(), "calculator",);
    }

    #[test]
    fn rejects_empty_tool_name() {
        assert!(parse_tool_name("   ").is_err());
    }
}
