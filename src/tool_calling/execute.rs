use crate::{
    chat::{ChatTemplate, Conversation, generate_chat_stream},
    error::{Result, TinyError},
    generation::{GenerationConfig, GenerationOutput},
    metal::MetalContext,
    model::Transformer,
    structured::StructuredRetryConfig,
    tokenizer::Tokenizer,
    tools::{ToolRegistry, ToolResult},
};

use super::{
    generate::{GeneratedToolCall, generate_tool_call},
    prompt::build_tool_result_prompt,
};

#[derive(Debug)]
pub struct ToolCallingOutput {
    generated_call: GeneratedToolCall,
    tool_result: ToolResult,
    answer: String,
    final_generation: GenerationOutput,
}

impl ToolCallingOutput {
    pub fn generated_call(&self) -> &GeneratedToolCall {
        &self.generated_call
    }

    pub fn tool_result(&self) -> &ToolResult {
        &self.tool_result
    }

    pub fn answer(&self) -> &str {
        &self.answer
    }

    pub fn final_generation(&self) -> &GenerationOutput {
        &self.final_generation
    }
}

pub fn execute_tool_calling(
    context: &MetalContext,
    model: &Transformer,
    tokenizer: &dyn Tokenizer,
    template: &dyn ChatTemplate,
    conversation: &Conversation,
    registry: &ToolRegistry,
    user_request: &str,
    generation_config: &GenerationConfig,
    retry_config: &StructuredRetryConfig,
) -> Result<ToolCallingOutput> {
    /*
     * ------------------------------------
     * Stage 1 + 2
     *
     * Tool selection
     * +
     * Argument generation
     * ------------------------------------
     */
    let generated_call = generate_tool_call(
        context,
        model,
        tokenizer,
        template,
        conversation,
        registry,
        user_request,
        generation_config,
        retry_config,
    )?;

    /*
     * ------------------------------------
     * Stage 3
     *
     * Execute actual Rust Tool
     * ------------------------------------
     */
    let tool_result = registry.execute(generated_call.call())?;

    /*
     * ------------------------------------
     * Stage 4
     *
     * ToolResult -> final answer prompt
     * ------------------------------------
     */
    let final_prompt = build_tool_result_prompt(user_request, generated_call.call(), &tool_result)?;

    let mut final_conversation = conversation.clone();

    final_conversation.push_user(final_prompt);

    /*
     * ------------------------------------
     * Stage 5
     *
     * Generate natural-language answer
     * ------------------------------------
     */
    let mut generated_tokens = Vec::new();

    let final_generation = generate_chat_stream(
        context,
        model,
        tokenizer,
        template,
        &final_conversation,
        generation_config,
        |token_id| {
            generated_tokens.push(token_id);

            Ok(())
        },
    )?;

    let answer = tokenizer
        .decode(&generated_tokens, true)?
        .trim()
        .to_string();

    if answer.is_empty() {
        return Err(TinyError::Tool(
            "tool calling produced an empty final answer".to_string(),
        ));
    }

    Ok(ToolCallingOutput {
        generated_call,
        tool_result,
        answer,
        final_generation,
    })
}
