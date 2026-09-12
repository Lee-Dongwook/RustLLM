use crate::{
    error::{Result, TinyError},
    tools::ToolRegistry,
};

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AgentDecision {
    Tool(String),
    Final,
}

impl AgentDecision {
    pub fn tool_name(&self) -> Option<&str> {
        match self {
            Self::Tool(name) => Some(name),
            Self::Final => None,
        }
    }

    pub fn is_final(&self) -> bool {
        matches!(self, Self::Final)
    }
}

pub fn parse_agent_decision(raw: &str, registry: &ToolRegistry) -> Result<AgentDecision> {
    let raw = raw.trim();

    if raw.is_empty() {
        return Err(TinyError::Tool(
            "agent produced an empty decision".to_string(),
        ));
    }

    /*
     * 작은 모델이:
     *
     * calculator
     *
     * 대신:
     *
     * "calculator"
     *
     * 처럼 JSON string을 내는 경우도 허용.
     */
    let text = if raw.starts_with('"') && raw.ends_with('"') {
        serde_json::from_str::<String>(raw).unwrap_or_else(|_| raw.to_string())
    } else {
        raw.to_string()
    };

    let text = text.trim();

    // 1. Direct match on the whole trimmed text
    if text.eq_ignore_ascii_case("final") {
        return Ok(AgentDecision::Final);
    }

    if registry.contains(text) {
        return Ok(AgentDecision::Tool(text.to_string()));
    }

    // 2. Strip leading list markers ('-', '*', '•', '`', '"', etc.)
    let mut cleaned = text;
    while let Some(stripped) =
        cleaned.strip_prefix(|c: char| c == '-' || c == '*' || c == '•' || c == '`')
    {
        cleaned = stripped.trim_start();
    }

    // Strip leading digit numbering like "1." or "1)"
    if let Some(pos) = cleaned.find(|c: char| !c.is_ascii_digit()) {
        if pos > 0 && (cleaned[pos..].starts_with(". ") || cleaned[pos..].starts_with(") ")) {
            cleaned = cleaned[pos + 1..].trim_start();
        }
    }

    // Strip common prefixes like "action:", "next action:", "tool:"
    for prefix in &["next action:", "action:", "tool:"] {
        if cleaned.to_ascii_lowercase().starts_with(prefix) {
            cleaned = cleaned[prefix.len()..].trim_start();
            break;
        }
    }

    let cleaned = cleaned
        .trim_matches(|c: char| c == '`' || c == '"' || c == '\'')
        .trim();

    if cleaned.eq_ignore_ascii_case("final") {
        return Ok(AgentDecision::Final);
    }

    if registry.contains(cleaned) {
        return Ok(AgentDecision::Tool(cleaned.to_string()));
    }

    // 3. Extract the first token from the first line (e.g. "- calculator solve this problem.")
    let first_line = cleaned.lines().next().unwrap_or(cleaned).trim();
    let first_token = first_line
        .split(|c: char| c.is_whitespace() || c == ':' || c == ',' || c == '.' || c == ';')
        .next()
        .unwrap_or("")
        .trim();

    if first_token.eq_ignore_ascii_case("final") {
        return Ok(AgentDecision::Final);
    }

    if registry.contains(first_token) {
        return Ok(AgentDecision::Tool(first_token.to_string()));
    }

    // 4. Scan tokens in the first line in case there is leading noise
    for token in first_line.split_whitespace() {
        let token = token.trim_matches(|c: char| !c.is_alphanumeric() && c != '_');
        if token.eq_ignore_ascii_case("final") {
            return Ok(AgentDecision::Final);
        }
        if registry.contains(token) {
            return Ok(AgentDecision::Tool(token.to_string()));
        }
    }

    Err(TinyError::Tool(format!(
        "agent selected unknown action `{raw}`"
    )))
}

#[cfg(test)]
mod tests {
    use serde_json::Value;

    use super::{AgentDecision, parse_agent_decision};

    use crate::{
        error::Result,
        structured::JsonSchema,
        tools::{Tool, ToolRegistry},
    };

    struct DummyTool {
        schema: JsonSchema,
    }

    impl DummyTool {
        fn new() -> Self {
            Self {
                schema: JsonSchema::new("calculator", r#"{}"#).unwrap(),
            }
        }
    }

    impl Tool for DummyTool {
        fn name(&self) -> &str {
            "calculator"
        }

        fn description(&self) -> &str {
            "Performs calculations."
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

        registry.register(DummyTool::new()).unwrap();

        registry
    }

    #[test]
    fn parses_tool_decision() {
        assert_eq!(
            parse_agent_decision("calculator", &registry()).unwrap(),
            AgentDecision::Tool("calculator".to_string()),
        );
    }

    #[test]
    fn parses_quoted_tool_decision() {
        assert_eq!(
            parse_agent_decision("\"calculator\"", &registry()).unwrap(),
            AgentDecision::Tool("calculator".to_string()),
        );
    }

    #[test]
    fn parses_bullet_tool_decision() {
        assert_eq!(
            parse_agent_decision("- calculator solve this problem.", &registry()).unwrap(),
            AgentDecision::Tool("calculator".to_string()),
        );
    }

    #[test]
    fn parses_prefixed_tool_decision() {
        assert_eq!(
            parse_agent_decision("Action: calculator", &registry()).unwrap(),
            AgentDecision::Tool("calculator".to_string()),
        );
    }

    #[test]
    fn parses_final_decision() {
        assert_eq!(
            parse_agent_decision("final", &registry()).unwrap(),
            AgentDecision::Final,
        );
    }

    #[test]
    fn parses_final_case_insensitively() {
        assert_eq!(
            parse_agent_decision("FINAL", &registry()).unwrap(),
            AgentDecision::Final,
        );
    }

    #[test]
    fn parses_final_with_trailing_period() {
        assert_eq!(
            parse_agent_decision("final.", &registry()).unwrap(),
            AgentDecision::Final,
        );
    }

    #[test]
    fn rejects_unknown_action() {
        assert!(parse_agent_decision("fake_tool", &registry()).is_err());
    }

    #[test]
    fn rejects_empty_decision() {
        assert!(parse_agent_decision("   ", &registry()).is_err());
    }
}
