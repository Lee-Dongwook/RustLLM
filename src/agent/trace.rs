use serde_json::Value;

#[derive(Debug, Clone)]
pub struct AgentStep {
    index: usize,
    tool_name: String,
    arguments: Value,
    result: Value,
}

impl AgentStep {
    pub fn new(
        index: usize,
        tool_name: impl Into<String>,
        arguments: Value,
        result: Value,
    ) -> Self {
        Self {
            index,
            tool_name: tool_name.into(),
            arguments,
            result,
        }
    }

    pub fn index(&self) -> usize {
        self.index
    }

    pub fn tool_name(&self) -> &str {
        &self.tool_name
    }

    pub fn arguments(&self) -> &Value {
        &self.arguments
    }

    pub fn result(&self) -> &Value {
        &self.result
    }
}

#[derive(Debug, Clone, Default)]
pub struct AgentTrace {
    steps: Vec<AgentStep>,
}

impl AgentTrace {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn push(&mut self, step: AgentStep) {
        self.steps.push(step);
    }

    pub fn steps(&self) -> &[AgentStep] {
        &self.steps
    }

    pub fn len(&self) -> usize {
        self.steps.len()
    }

    pub fn is_empty(&self) -> bool {
        self.steps.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::{AgentStep, AgentTrace};

    #[test]
    fn records_agent_steps() {
        let mut trace = AgentTrace::new();

        trace.push(AgentStep::new(
            0,
            "calculator",
            json!({
                "left": 2,
                "operator": "add",
                "right": 3
            }),
            json!({
                "value": 5
            }),
        ));

        assert_eq!(trace.len(), 1);

        assert_eq!(trace.steps()[0].tool_name(), "calculator");

        assert_eq!(trace.steps()[0].result()["value"], 5);
    }
}
