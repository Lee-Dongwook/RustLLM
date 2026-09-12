use crate::{
    error::{Result, TinyError},
    tools::ToolRegistry,
};

pub fn build_tool_call_task(user_request: &str, registry: &ToolRegistry) -> Result<String> {
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

    let mut prompt = String::new();

    prompt.push_str("Choose exactly one tool that can best handle the user request.\n");

    prompt.push_str("Do not answer the user request yourself.\n");

    prompt.push_str("Return the selected tool name and its arguments.\n");

    prompt.push_str("Arguments must satisfy the selected tool's input schema.\n\n");

    prompt.push_str("Available tools:\n\n");

    for tool in registry.iter() {
        prompt.push_str("Tool: ");

        prompt.push_str(tool.name());

        prompt.push('\n');

        prompt.push_str("Description: ");

        prompt.push_str(tool.description());

        prompt.push('\n');

        prompt.push_str("Arguments schema:\n");

        prompt.push_str(&tool.input_schema().to_pretty_json()?);

        prompt.push_str("\n\n");
    }

    prompt.push_str("User request:\n");

    prompt.push_str(user_request);

    Ok(prompt)
}

#[cfg(test)]
mod tests {
    use serde_json::Value;

    use super::build_tool_call_task;

    use crate::{
        error::Result,
        structured::JsonSchema,
        tools::{Tool, ToolRegistry},
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
                            "expression": "string"
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
            "Evaluates a mathematical expression."
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

        let prompt = build_tool_call_task("What is 123 * 456?", &registry).unwrap();

        assert!(prompt.contains("calculator"));

        assert!(prompt.contains("Evaluates a mathematical expression."));

        assert!(prompt.contains("\"expression\": \"string\""));

        assert!(prompt.contains("What is 123 * 456?"));
    }

    #[test]
    fn rejects_empty_registry() {
        let registry = ToolRegistry::new();

        let result = build_tool_call_task("Calculate 2 + 2", &registry);

        assert!(result.is_err());
    }
}
