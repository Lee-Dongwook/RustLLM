use crate::{
    error::{Result, TinyError},
    tools::{Tool, ToolRegistry, ToolResult},
};

pub fn build_tool_selection_prompt(user_request: &str, registry: &ToolRegistry) -> Result<String> {
    let user_request = user_request.trim();

    if user_request.is_empty() {
        return Err(TinyError::InvalidArgument(
            "tool calling request cannot be empty".to_string(),
        ));
    }

    if registry.is_empty() {
        return Err(TinyError::Tool(
            "tool calling requires at least one registered tool".to_string(),
        ));
    }

    let mut tools = registry.iter().collect::<Vec<_>>();

    tools.sort_by_key(|tool| tool.name());

    let mut prompt = String::new();

    prompt.push_str("Select the single best tool for the user request.\n");

    prompt.push_str("Respond with ONLY the tool name.\n");

    prompt.push_str("Do not answer the request.\n\n");

    prompt.push_str("Available tools:\n");

    for tool in tools {
        prompt.push_str("- ");
        prompt.push_str(tool.name());
        prompt.push_str(": ");
        prompt.push_str(tool.description());
        prompt.push('\n');
    }

    prompt.push_str("\nUser request:\n");

    prompt.push_str(user_request);

    prompt.push_str("\n\nTool name:");

    Ok(prompt)
}

pub fn build_tool_arguments_task(user_request: &str, tool: &dyn Tool) -> Result<String> {
    let user_request = user_request.trim();

    if user_request.is_empty() {
        return Err(TinyError::InvalidArgument(
            "tool calling request cannot be empty".to_string(),
        ));
    }

    Ok(format!(
        "Create the arguments needed to call the tool `{}`.\n\
             Use the user's request to determine the actual argument values.\n\
             Return only the arguments object.\n\n\
             Tool description:\n{}\n\n\
             User request:\n{}",
        tool.name(),
        tool.description(),
        user_request,
    ))
}

pub fn build_tool_result_prompt(user_request: &str, result: &ToolResult) -> Result<String> {
    let user_request = user_request.trim();

    if user_request.is_empty() {
        return Err(TinyError::InvalidArgument(
            "tool result prompt requires a non-empty user request".to_string(),
        ));
    }

    Ok(format!(
        "User question: {user_request}\n\
             Tool result: {}\n\
             Give only the final answer to the user.",
        result.output(),
    ))
}
#[cfg(test)]
mod tests {
    use serde_json::{Value, json};

    use super::{build_tool_arguments_task, build_tool_result_prompt, build_tool_selection_prompt};

    use crate::{
        error::Result,
        structured::JsonSchema,
        tools::{Tool, ToolRegistry, ToolResult},
    };
    struct CalculatorTool {
        schema: JsonSchema,
    }

    impl CalculatorTool {
        fn new() -> Self {
            Self {
                schema: JsonSchema::new(
                    "calculator",
                    r#"{
                        "left": "number",
                        "operator": "string",
                        "right": "number"
                    }"#,
                )
                .unwrap(),
            }
        }
    }

    impl Tool for CalculatorTool {
        fn name(&self) -> &str {
            "calculator"
        }

        fn description(&self) -> &str {
            "Performs arithmetic on two numbers."
        }

        fn input_schema(&self) -> &JsonSchema {
            &self.schema
        }

        fn execute(&self, _arguments: &Value) -> Result<Value> {
            unreachable!()
        }
    }

    #[test]
    fn builds_tool_selection_prompt() {
        let mut registry = ToolRegistry::new();

        registry.register(CalculatorTool::new()).unwrap();

        let prompt = build_tool_selection_prompt("What is 123 * 456?", &registry).unwrap();

        assert!(prompt.contains("calculator"));

        assert!(prompt.contains("Performs arithmetic on two numbers."));

        assert!(prompt.contains("What is 123 * 456?"));

        assert!(prompt.contains("Tool name:"));
    }

    #[test]
    fn builds_tool_arguments_task() {
        let tool = CalculatorTool::new();

        let prompt = build_tool_arguments_task("What is 123 multiplied by 456?", &tool).unwrap();

        assert!(prompt.contains("calculator"));

        assert!(prompt.contains("What is 123 multiplied by 456?"));

        assert!(prompt.contains("Return only the arguments object."));
    }

    #[test]
    fn selection_rejects_empty_registry() {
        let registry = ToolRegistry::new();

        let result = build_tool_selection_prompt("Calculate 2 + 2", &registry);

        assert!(result.is_err());
    }

    #[test]
    fn selection_rejects_empty_request() {
        let mut registry = ToolRegistry::new();

        registry.register(CalculatorTool::new()).unwrap();

        let result = build_tool_selection_prompt("   ", &registry);

        assert!(result.is_err());
    }

    #[test]
    fn arguments_reject_empty_request() {
        let tool = CalculatorTool::new();

        let result = build_tool_arguments_task("   ", &tool);

        assert!(result.is_err());
    }

    #[test]
    fn builds_tool_result_prompt() {
        let result = ToolResult::new(
            "calculator",
            json!({
                "value": 56088
            }),
        );

        let prompt = build_tool_result_prompt("What is 123 multiplied by 456?", &result).unwrap();

        assert!(prompt.contains("What is 123 multiplied by 456?"));

        assert!(prompt.contains("56088"));

        assert!(prompt.contains("Give only the final answer to the user."));
    }
}
