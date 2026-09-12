use serde::{Deserialize, Serialize};
use serde_json::Value;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ToolCall {
    name: String,
    arguments: Value,
}

impl ToolCall {
    pub fn new(name: impl Into<String>, arguments: Value) -> Self {
        Self {
            name: name.into(),
            arguments,
        }
    }

    pub fn name(&self) -> &str {
        &self.name
    }

    pub fn arguments(&self) -> &Value {
        &self.arguments
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ToolResult {
    name: String,
    output: Value,
}

impl ToolResult {
    pub fn new(name: impl Into<String>, output: Value) -> Self {
        Self {
            name: name.into(),
            output,
        }
    }

    pub fn name(&self) -> &str {
        &self.name
    }

    pub fn output(&self) -> &Value {
        &self.output
    }
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::{ToolCall, ToolResult};

    #[test]
    fn creates_tool_call() {
        let call = ToolCall::new(
            "calculator",
            json!({
                "expression": "2 + 2"
            }),
        );

        assert_eq!(call.name(), "calculator",);

        assert_eq!(call.arguments()["expression"], "2 + 2",);
    }

    #[test]
    fn creates_tool_result() {
        let result = ToolResult::new(
            "calculator",
            json!({
                "value": 4
            }),
        );

        assert_eq!(result.name(), "calculator",);

        assert_eq!(result.output()["value"], 4,);
    }
}
