use crate::{
    agent::AgentTrace,
    error::{Result, TinyError},
    tools::ToolRegistry,
};

pub fn build_agent_decision_prompt(
    user_request: &str,
    trace: &AgentTrace,
    registry: &ToolRegistry,
) -> Result<String> {
    let user_request = user_request.trim();

    if user_request.is_empty() {
        return Err(TinyError::InvalidArgument(
            "agent request cannot be empty".to_string(),
        ));
    }

    if registry.is_empty() {
        return Err(TinyError::Tool(
            "agent requires at least one registered tool".to_string(),
        ));
    }

    let mut tools = registry.iter().collect::<Vec<_>>();

    tools.sort_by_key(|tool| tool.name());

    let mut prompt = String::new();

    prompt.push_str("Choose the next action needed to solve the user request.\n");
    prompt.push_str("Use a tool when information or computation is still needed.\n");
    prompt.push_str("If facts or documents must be looked up first, choose document_search before calculator.\n");
    prompt.push_str("Choose final only when the existing tool results are enough to answer.\n");

    prompt.push_str(
        "Respond with ONLY the action name (e.g. document_search, calculator, or final).\n",
    );
    prompt.push_str("Do not write sentences or bullet points.\n\n");

    prompt.push_str("Available actions:\n");

    for tool in tools {
        prompt.push_str("- ");
        prompt.push_str(tool.name());
        prompt.push_str(": ");
        prompt.push_str(tool.description());
        prompt.push('\n');
    }

    prompt.push_str("- final: Give the final answer when no more tools are needed.\n");

    if !trace.is_empty() {
        prompt.push_str("\nPrevious tool results:\n");

        for step in trace.steps() {
            prompt.push_str(&format!("\nStep {}:\n", step.index() + 1));

            prompt.push_str("Tool: ");

            prompt.push_str(step.tool_name());

            prompt.push_str("\nResult: ");

            prompt.push_str(&step.result().to_string());

            prompt.push('\n');
        }
    }

    prompt.push_str("\nUser request:\n");

    prompt.push_str(user_request);

    prompt.push_str("\n\nNext action:");

    Ok(prompt)
}

pub fn build_agent_final_prompt(user_request: &str, trace: &AgentTrace) -> Result<String> {
    let user_request = user_request.trim();

    if user_request.is_empty() {
        return Err(TinyError::InvalidArgument(
            "agent request cannot be empty".to_string(),
        ));
    }

    let mut prompt = String::new();

    prompt.push_str("Answer the user request using the tool results below.\n\n");

    prompt.push_str("User request:\n");

    prompt.push_str(user_request);

    if !trace.is_empty() {
        prompt.push_str("\n\nTool results:\n");

        for step in trace.steps() {
            prompt.push_str(&format!("\n{}: {}\n", step.tool_name(), step.result()));
        }
    }

    prompt.push_str("\nGive only the final answer to the user.");

    Ok(prompt)
}

#[cfg(test)]
mod tests {
    use serde_json::{Value, json};

    use super::{build_agent_decision_prompt, build_agent_final_prompt};

    use crate::{
        agent::{AgentStep, AgentTrace},
        error::Result,
        structured::JsonSchema,
        tools::{Tool, ToolRegistry},
    };

    struct DummyTool {
        name: &'static str,
        description: &'static str,
        schema: JsonSchema,
    }

    impl DummyTool {
        fn new(name: &'static str, description: &'static str) -> Self {
            Self {
                name,
                description,
                schema: JsonSchema::new(name, r#"{}"#).unwrap(),
            }
        }
    }

    impl Tool for DummyTool {
        fn name(&self) -> &str {
            self.name
        }

        fn description(&self) -> &str {
            self.description
        }

        fn input_schema(&self) -> &JsonSchema {
            &self.schema
        }

        fn execute(&self, _arguments: &Value) -> Result<Value> {
            unreachable!()
        }
    }

    fn registry() -> ToolRegistry {
        let mut registry = ToolRegistry::new();

        registry
            .register(DummyTool::new("calculator", "Performs arithmetic."))
            .unwrap();

        registry
            .register(DummyTool::new("document_search", "Searches documents."))
            .unwrap();

        registry
    }

    #[test]
    fn builds_initial_agent_prompt() {
        let prompt = build_agent_decision_prompt(
            "Find information about KV cache.",
            &AgentTrace::new(),
            &registry(),
        )
        .unwrap();

        assert!(prompt.contains("calculator"));

        assert!(prompt.contains("document_search"));

        assert!(prompt.contains("final"));

        assert!(prompt.contains("Find information about KV cache."));

        assert!(!prompt.contains("Previous tool results:"));
    }

    #[test]
    fn includes_previous_observations() {
        let mut trace = AgentTrace::new();

        trace.push(AgentStep::new(
            0,
            "document_search",
            json!({
                "query": "KV cache"
            }),
            json!({
                "matches": [
                    {
                        "text": "KV cache avoids recomputation."
                    }
                ]
            }),
        ));

        let prompt =
            build_agent_decision_prompt("Find why KV cache is used.", &trace, &registry()).unwrap();

        assert!(prompt.contains("Previous tool results:"));

        assert!(prompt.contains("document_search"));

        assert!(prompt.contains("KV cache avoids recomputation."));
    }

    #[test]
    fn builds_final_prompt_from_trace() {
        let mut trace = AgentTrace::new();

        trace.push(AgentStep::new(
            0,
            "document_search",
            json!({
                "query": "KV cache"
            }),
            json!({
                "matches": [
                    {
                        "text": "The KV cache avoids recomputing keys and values."
                    }
                ]
            }),
        ));

        let prompt = build_agent_final_prompt("Why does RustLLM use a KV cache?", &trace).unwrap();

        assert!(prompt.contains("Why does RustLLM use a KV cache?"));

        assert!(prompt.contains("The KV cache avoids recomputing keys and values."));

        assert!(prompt.contains("Give only the final answer"));
    }

    #[test]
    fn rejects_empty_request() {
        assert!(build_agent_decision_prompt("   ", &AgentTrace::new(), &registry()).is_err());
    }
}
