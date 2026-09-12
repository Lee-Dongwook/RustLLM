use serde_json::Value;

use crate::error::{Result, TinyError};

#[derive(Debug, Clone)]
pub struct JsonSchema {
    name: String,
    value: Value,
}

impl JsonSchema {
    pub fn new(name: impl Into<String>, schema: &str) -> Result<Self> {
        let name = name.into();

        if name.trim().is_empty() {
            return Err(TinyError::StructuredOutput(
                "JSON schema name cannot be empty".to_string(),
            ));
        }

        let value: Value = serde_json::from_str(schema).map_err(|error| {
            TinyError::StructuredOutput(format!("invalid JSON schema `{name}`: {error}"))
        })?;

        if !value.is_object() {
            return Err(TinyError::StructuredOutput(format!(
                "JSON schema `{name}` must be a JSON object"
            )));
        }

        Ok(Self { name, value })
    }

    pub fn name(&self) -> &str {
        &self.name
    }

    pub fn value(&self) -> &Value {
        &self.value
    }

    pub fn to_pretty_json(&self) -> Result<String> {
        serde_json::to_string_pretty(&self.value).map_err(|error| {
            TinyError::StructuredOutput(format!(
                "failed to serialize JSON schema `{}`: {error}",
                self.name,
            ))
        })
    }
}

#[cfg(test)]
mod tests {
    use super::JsonSchema;

    #[test]
    fn creates_json_schema() {
        let schema = JsonSchema::new(
            "classification",
            r#"{
                    "category": "string",
                    "priority": "string"
                }"#,
        )
        .unwrap();

        assert_eq!(schema.name(), "classification");

        assert!(schema.value().is_object());
    }

    #[test]
    fn rejects_invalid_json() {
        let result = JsonSchema::new(
            "broken",
            r#"{
                    "name":
                }"#,
        );

        assert!(result.is_err());
    }

    #[test]
    fn rejects_non_object_schema() {
        let result = JsonSchema::new("array", r#"["string"]"#);

        assert!(result.is_err());
    }

    #[test]
    fn rejects_empty_name() {
        let result = JsonSchema::new(
            "   ",
            r#"{
                    "name": "string"
                }"#,
        );

        assert!(result.is_err());
    }

    #[test]
    fn formats_schema_as_pretty_json() {
        let schema = JsonSchema::new("person", r#"{"name":"string","age":"number"}"#).unwrap();

        let pretty = schema.to_pretty_json().unwrap();

        assert_eq!(
            pretty,
            concat!(
                "{\n",
                "  \"age\": \"number\",\n",
                "  \"name\": \"string\"\n",
                "}"
            )
        );
    }
}
