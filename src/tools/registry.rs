use std::collections::HashMap;

use crate::error::{Result, TinyError};

use super::{Tool, ToolCall, ToolResult};

pub struct ToolRegistry {
    tools: HashMap<String, Box<dyn Tool>>,
}

impl ToolRegistry {
    pub fn new() -> Self {
        Self {
            tools: HashMap::new(),
        }
    }

    pub fn register<T>(&mut self, tool: T) -> Result<()>
    where
        T: Tool + 'static,
    {
        let name = tool.name().trim().to_string();

        if name.is_empty() {
            return Err(TinyError::Tool("tool name cannot be empty".to_string()));
        }

        if self.tools.contains_key(&name) {
            return Err(TinyError::Tool(format!(
                "tool `{name}` is already registered"
            )));
        }

        self.tools.insert(name, Box::new(tool));

        Ok(())
    }

    pub fn get(&self, name: &str) -> Option<&dyn Tool> {
        self.tools.get(name).map(|tool| tool.as_ref())
    }

    pub fn contains(&self, name: &str) -> bool {
        self.tools.contains_key(name)
    }

    pub fn len(&self) -> usize {
        self.tools.len()
    }

    pub fn is_empty(&self) -> bool {
        self.tools.is_empty()
    }

    pub fn execute(&self, call: &ToolCall) -> Result<ToolResult> {
        self.validate_call(call)?;

        let tool = self
            .get(call.name())
            .ok_or_else(|| TinyError::Tool(format!("unknown tool `{}`", call.name(),)))?;

        let output = tool.execute(call.arguments())?;

        Ok(ToolResult::new(call.name(), output))
    }

    pub fn iter(&self) -> impl Iterator<Item = &dyn Tool> + '_ {
        self.tools.values().map(|tool| tool.as_ref())
    }

    pub fn validate_call(&self, call: &ToolCall) -> Result<()> {
        let tool = self
            .get(call.name())
            .ok_or_else(|| TinyError::Tool(format!("unknown tool `{}`", call.name(),)))?;

        tool.input_schema()
            .validate(call.arguments())
            .map_err(|error| {
                TinyError::Tool(format!(
                    "invalid arguments for tool `{}`: {error}",
                    call.name(),
                ))
            })?;

        Ok(())
    }
}

impl Default for ToolRegistry {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use serde_json::{Value, json};

    use super::ToolRegistry;

    use crate::{
        error::Result,
        structured::JsonSchema,
        tools::{Tool, ToolCall},
    };

    struct AddTool {
        schema: JsonSchema,
    }

    impl AddTool {
        fn new() -> Self {
            Self {
                schema: JsonSchema::new(
                    "add",
                    r#"{
                            "left": "number",
                            "right": "number"
                        }"#,
                )
                .unwrap(),
            }
        }
    }

    impl Tool for AddTool {
        fn name(&self) -> &str {
            "add"
        }

        fn description(&self) -> &str {
            "Adds two numbers."
        }

        fn input_schema(&self) -> &JsonSchema {
            &self.schema
        }

        fn execute(&self, arguments: &Value) -> Result<Value> {
            let left = arguments["left"].as_f64().unwrap();

            let right = arguments["right"].as_f64().unwrap();

            Ok(json!({
                "value": left + right
            }))
        }
    }

    #[test]
    fn registers_tool() {
        let mut registry = ToolRegistry::new();

        registry.register(AddTool::new()).unwrap();

        assert_eq!(registry.len(), 1,);

        assert!(registry.contains("add"));
    }

    #[test]
    fn executes_registered_tool() {
        let mut registry = ToolRegistry::new();

        registry.register(AddTool::new()).unwrap();

        let call = ToolCall::new(
            "add",
            json!({
                "left": 2,
                "right": 3
            }),
        );

        let result = registry.execute(&call).unwrap();

        assert_eq!(result.output()["value"], 5.0,);
    }

    #[test]
    fn rejects_unknown_tool() {
        let registry = ToolRegistry::new();

        let call = ToolCall::new("missing", json!({}));

        assert!(registry.execute(&call,).is_err());
    }

    #[test]
    fn rejects_invalid_arguments() {
        let mut registry = ToolRegistry::new();

        registry.register(AddTool::new()).unwrap();

        let call = ToolCall::new(
            "add",
            json!({
                "left": "two",
                "right": 3
            }),
        );

        let error = registry.execute(&call).unwrap_err();

        assert!(error.to_string().contains("expected number"));
    }

    #[test]
    fn rejects_duplicate_tool_name() {
        let mut registry = ToolRegistry::new();

        registry.register(AddTool::new()).unwrap();

        let result = registry.register(AddTool::new());

        assert!(result.is_err());
    }

    #[test]
    fn validates_tool_call_without_executing() {
        let mut registry = ToolRegistry::new();

        registry.register(AddTool::new()).unwrap();

        let call = ToolCall::new(
            "add",
            json!({
                "left": 10,
                "right": 20
            }),
        );

        registry.validate_call(&call).unwrap();
    }

    #[test]
    fn validation_rejects_unknown_tool() {
        let registry = ToolRegistry::new();

        let call = ToolCall::new("something_fake", json!({}));

        assert!(registry.validate_call(&call,).is_err());
    }
}
