use crate::{
    chat::{ChatTemplate, Conversation},
    error::Result,
    generation::GenerationConfig,
    metal::MetalContext,
    model::Transformer,
    structured::{
        JsonSchema, StructuredGeneration, StructuredRetryConfig, generate_structured_with_retry,
    },
    tokenizer::Tokenizer,
    tools::{ToolCall, ToolRegistry},
};

use super::build_tool_call_task;

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
) -> Result<StructuredGeneration<ToolCall>> {
    /*
     * 등록된 Tool들의 description/schema를
     * LLM task prompt로 변환.
     */
    let task = build_tool_call_task(user_request, registry)?;

    /*
     * ToolCall 자체의 공통 envelope.
     *
     * 실제 arguments schema는 선택된 tool에 따라
     * 달라지므로 generation 후 Registry에서 한 번 더 검증한다.
     */
    let call_schema = JsonSchema::new(
        "tool_call",
        r#"{
                "name": "string",
                "arguments": "object"
            }"#,
    )?;

    let output = generate_structured_with_retry::<ToolCall>(
        context,
        model,
        tokenizer,
        template,
        conversation,
        &task,
        &call_schema,
        generation_config,
        retry_config,
    )?;

    /*
     * 여기까지 성공했다고 끝이 아님.
     *
     * {
     *   "name": "fake_tool",
     *   "arguments": {}
     * }
     *
     * 역시 outer schema 자체는 만족하니까.
     *
     * 실제 registry를 기준으로 semantic validation.
     */
    registry.validate_call(output.value())?;

    Ok(output)
}
