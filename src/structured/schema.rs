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

    pub fn validate(&self, value: &Value) -> Result<()> {
        validate_schema_value(&self.value, value, "$")
    }

    pub fn coerce(&self, value: &Value) -> Result<Value> {
        coerce_schema_value(&self.value, value, "$")
    }
}

fn validate_schema_value(schema: &Value, value: &Value, path: &str) -> Result<()> {
    match schema {
        Value::String(kind) => validate_primitive_type(kind, value, path),

        Value::Object(fields) => {
            let object = value.as_object().ok_or_else(|| {
                TinyError::StructuredOutput(format!(
                    "{path}: expected object, got {}",
                    json_type_name(value),
                ))
            })?;

            for (field_name, field_schema) in fields {
                let field_value = object.get(field_name).ok_or_else(|| {
                    TinyError::StructuredOutput(format!(
                        "{path}.{field_name}: required field is missing"
                    ))
                })?;

                let field_path = format!("{path}.{field_name}");

                validate_schema_value(field_schema, field_value, &field_path)?;
            }

            Ok(())
        }

        Value::Array(items) => {
            if items.len() != 1 {
                return Err(TinyError::StructuredOutput(format!(
                    "{path}: array schema must contain exactly one item descriptor"
                )));
            }

            let array = value.as_array().ok_or_else(|| {
                TinyError::StructuredOutput(format!(
                    "{path}: expected array, got {}",
                    json_type_name(value),
                ))
            })?;

            for (index, item) in array.iter().enumerate() {
                validate_schema_value(&items[0], item, &format!("{path}[{index}]"))?;
            }

            Ok(())
        }

        _ => Err(TinyError::StructuredOutput(format!(
            "{path}: invalid schema descriptor {schema}"
        ))),
    }
}

fn validate_primitive_type(expected: &str, value: &Value, path: &str) -> Result<()> {
    let valid = match expected {
        "string" => value.is_string(),

        "number" => value.is_number(),

        "integer" => value.as_i64().is_some() || value.as_u64().is_some(),

        "boolean" => value.is_boolean(),

        "null" => value.is_null(),

        "object" => value.is_object(),

        "array" => value.is_array(),

        "any" => true,

        other => {
            return Err(TinyError::StructuredOutput(format!(
                "{path}: unsupported schema type `{other}`"
            )));
        }
    };

    if valid {
        return Ok(());
    }

    Err(TinyError::StructuredOutput(format!(
        "{path}: expected {expected}, got {}",
        json_type_name(value),
    )))
}

fn json_type_name(value: &Value) -> &'static str {
    match value {
        Value::Null => "null",
        Value::Bool(_) => "boolean",
        Value::Number(_) => "number",
        Value::String(_) => "string",
        Value::Array(_) => "array",
        Value::Object(_) => "object",
    }
}

fn coerce_schema_value(schema: &Value, value: &Value, path: &str) -> Result<Value> {
    match schema {
        Value::String(expected_type) => coerce_primitive(expected_type, value, path),

        Value::Object(schema_object) => {
            let value_object = value.as_object().ok_or_else(|| {
                TinyError::StructuredOutput(format!(
                    "{path}: expected object, got {}",
                    json_type_name(value),
                ))
            })?;

            let mut result = serde_json::Map::new();

            for (key, child_schema) in schema_object {
                let child_value = value_object.get(key).ok_or_else(|| {
                    TinyError::StructuredOutput(format!("{path}.{key}: required field is missing"))
                })?;

                let child_path = format!("{path}.{key}");

                result.insert(
                    key.clone(),
                    coerce_schema_value(child_schema, child_value, &child_path)?,
                );
            }

            /*
             * Schema에 선언되지 않은 값은 그대로 보존.
             *
             * 나중에 strict additionalProperties 정책을
             * 넣고 싶으면 별도로 처리하면 된다.
             */
            for (key, child_value) in value_object {
                if !schema_object.contains_key(key) {
                    result.insert(key.clone(), child_value.clone());
                }
            }

            Ok(Value::Object(result))
        }

        Value::Array(schema_array) => {
            if schema_array.len() != 1 {
                return Err(TinyError::StructuredOutput(format!(
                    "{path}: array schema must contain exactly one item descriptor"
                )));
            }

            let values = value.as_array().ok_or_else(|| {
                TinyError::StructuredOutput(format!(
                    "{path}: expected array, got {}",
                    json_type_name(value),
                ))
            })?;

            let item_schema = &schema_array[0];

            let mut result = Vec::with_capacity(values.len());

            for (index, item) in values.iter().enumerate() {
                result.push(coerce_schema_value(
                    item_schema,
                    item,
                    &format!("{path}[{index}]"),
                )?);
            }

            Ok(Value::Array(result))
        }

        _ => Ok(value.clone()),
    }
}

fn coerce_primitive(expected_type: &str, value: &Value, path: &str) -> Result<Value> {
    match expected_type {
        "number" => {
            if value.is_number() {
                return Ok(value.clone());
            }

            if let Some(text) = value.as_str() {
                if let Ok(number) = text.trim().parse::<f64>() {
                    let number = serde_json::Number::from_f64(number).ok_or_else(|| {
                        TinyError::StructuredOutput(format!("{path}: invalid number `{text}`"))
                    })?;

                    return Ok(Value::Number(number));
                }
            }

            Err(TinyError::StructuredOutput(format!(
                "{path}: cannot coerce {} to number",
                json_type_name(value),
            )))
        }

        "integer" => {
            if value.as_i64().is_some() || value.as_u64().is_some() {
                return Ok(value.clone());
            }

            if let Some(text) = value.as_str() {
                let text = text.trim();

                if let Ok(number) = text.parse::<i64>() {
                    return Ok(Value::Number(number.into()));
                }

                if let Ok(number) = text.parse::<u64>() {
                    return Ok(Value::Number(number.into()));
                }
            }

            Err(TinyError::StructuredOutput(format!(
                "{path}: cannot coerce {} to integer",
                json_type_name(value),
            )))
        }

        "boolean" => {
            if value.is_boolean() {
                return Ok(value.clone());
            }

            if let Some(text) = value.as_str() {
                match text.trim().to_ascii_lowercase().as_str() {
                    "true" => {
                        return Ok(Value::Bool(true));
                    }

                    "false" => {
                        return Ok(Value::Bool(false));
                    }

                    _ => {}
                }
            }

            Err(TinyError::StructuredOutput(format!(
                "{path}: cannot coerce {} to boolean",
                json_type_name(value),
            )))
        }

        /*
         * string은 반대로 숫자를 문자열로
         * 자동 변환하지 않는다.
         *
         * 너무 관대한 coercion을 피하기 위함.
         */
        "string" | "null" | "object" | "array" | "any" => Ok(value.clone()),

        other => Err(TinyError::StructuredOutput(format!(
            "{path}: unknown schema type `{other}`"
        ))),
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

    #[test]
    fn validates_matching_output() {
        let schema = JsonSchema::new(
            "person",
            r#"{
                "name": "string",
                "age": "number"
            }"#,
        )
        .unwrap();

        let value = serde_json::json!({
            "name": "Alice",
            "age": 24
        });

        schema.validate(&value).unwrap();
    }

    #[test]
    fn rejects_type_descriptors_as_values() {
        let schema = JsonSchema::new(
            "person",
            r#"{
                "name": "string",
                "age": "number"
            }"#,
        )
        .unwrap();

        let value = serde_json::json!({
            "name": "string",
            "age": "number"
        });

        let error = schema.validate(&value).unwrap_err();

        assert!(
            error
                .to_string()
                .contains("$.age: expected number, got string")
        );
    }

    #[test]
    fn rejects_missing_required_field() {
        let schema = JsonSchema::new(
            "person",
            r#"{
                "name": "string",
                "age": "number"
            }"#,
        )
        .unwrap();

        let value = serde_json::json!({
            "name": "Alice"
        });

        let error = schema.validate(&value).unwrap_err();

        assert!(
            error
                .to_string()
                .contains("$.age: required field is missing")
        );
    }

    #[test]
    fn coerces_numeric_strings() {
        let schema = JsonSchema::new(
            "calculator",
            r#"{
                "left": "number",
                "operator": "string",
                "right": "number"
            }"#,
        )
        .unwrap();

        let value = serde_json::json!({
            "left": "123",
            "operator": "multiply",
            "right": "456"
        });

        let coerced = schema.coerce(&value).unwrap();

        assert_eq!(coerced["left"], 123.0,);

        assert_eq!(coerced["operator"], "multiply",);

        assert_eq!(coerced["right"], 456.0,);

        schema.validate(&coerced).unwrap();
    }
}
