use crate::error::Result;

use super::JsonSchema;

const JSON_INSTRUCTIONS: &str = "Return ONLY valid JSON.\n\
Do not include markdown code fences or any additional text.";

pub fn build_json_prompt(task: &str, schema: &JsonSchema) -> Result<String> {
    let task = task.trim();

    let schema = schema.to_pretty_json()?;

    let mut prompt = String::new();

    if !task.is_empty() {
        prompt.push_str(task);
        prompt.push_str("\n\n");
    }

    prompt.push_str(JSON_INSTRUCTIONS);

    prompt.push_str("\n\nRequired JSON structure:\n");

    prompt.push_str(&schema);

    Ok(prompt)
}

#[cfg(test)]
mod tests {
    use super::build_json_prompt;
    use crate::structured::JsonSchema;

    #[test]
    fn builds_structured_json_prompt() {
        let schema = JsonSchema::new(
            "classification",
            r#"{
                    "category": "string",
                    "priority": "string"
                }"#,
        )
        .unwrap();

        let prompt = build_json_prompt("Classify this issue.", &schema).unwrap();

        assert!(prompt.starts_with("Classify this issue."));

        assert!(prompt.contains("Return ONLY valid JSON."));

        assert!(prompt.contains("Do not include markdown code fences"));

        assert!(prompt.contains("\"category\": \"string\""));

        assert!(prompt.contains("\"priority\": \"string\""));
    }

    #[test]
    fn allows_empty_task() {
        let schema = JsonSchema::new(
            "answer",
            r#"{
                    "answer": "string"
                }"#,
        )
        .unwrap();

        let prompt = build_json_prompt("", &schema).unwrap();

        assert!(prompt.starts_with("Return ONLY valid JSON."));
    }
}
