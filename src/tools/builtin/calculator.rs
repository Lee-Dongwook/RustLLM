use serde_json::{Value, json};

use crate::{
    error::{Result, TinyError},
    structured::JsonSchema,
};

use super::super::Tool;

pub struct CalculatorTool {
    schema: JsonSchema,
}

impl CalculatorTool {
    pub fn new() -> Result<Self> {
        Ok(Self {
            schema: JsonSchema::new(
                "calculator",
                r#"{
                    "left": "number",
                    "operator": "string",
                    "right": "number"
                }"#,
            )?,
        })
    }
}

impl Tool for CalculatorTool {
    fn name(&self) -> &str {
        "calculator"
    }

    fn description(&self) -> &str {
        "Performs arithmetic on two numbers. \
         Supported operators are add, subtract, multiply, and divide."
    }

    fn input_schema(&self) -> &JsonSchema {
        &self.schema
    }

    fn execute(&self, arguments: &Value) -> Result<Value> {
        let left = arguments
            .get("left")
            .and_then(Value::as_f64)
            .ok_or_else(|| {
                TinyError::Tool("calculator argument `left` must be a number".to_string())
            })?;

        let right = arguments
            .get("right")
            .and_then(Value::as_f64)
            .ok_or_else(|| {
                TinyError::Tool("calculator argument `right` must be a number".to_string())
            })?;

        let operator = arguments
            .get("operator")
            .and_then(Value::as_str)
            .ok_or_else(|| {
                TinyError::Tool("calculator argument `operator` must be a string".to_string())
            })?;

        let value = match operator {
            "add" | "+" => left + right,

            "subtract" | "-" => left - right,

            "multiply" | "*" => left * right,

            "divide" | "/" => {
                if right == 0.0 {
                    return Err(TinyError::Tool(
                        "calculator cannot divide by zero".to_string(),
                    ));
                }

                left / right
            }

            _ => {
                return Err(TinyError::Tool(format!(
                    "unsupported calculator operator `{operator}`"
                )));
            }
        };

        Ok(json!({
            "value": value
        }))
    }
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::CalculatorTool;

    use crate::tools::Tool;

    #[test]
    fn multiplies_numbers() {
        let tool = CalculatorTool::new().unwrap();

        let output = tool
            .execute(&json!({
                "left": 123,
                "operator": "multiply",
                "right": 456
            }))
            .unwrap();

        assert_eq!(output["value"], 56088.0,);
    }

    #[test]
    fn adds_numbers() {
        let tool = CalculatorTool::new().unwrap();

        let output = tool
            .execute(&json!({
                "left": 10,
                "operator": "add",
                "right": 20
            }))
            .unwrap();

        assert_eq!(output["value"], 30.0,);
    }

    #[test]
    fn divides_numbers() {
        let tool = CalculatorTool::new().unwrap();

        let output = tool
            .execute(&json!({
                "left": 10,
                "operator": "divide",
                "right": 4
            }))
            .unwrap();

        assert_eq!(output["value"], 2.5,);
    }

    #[test]
    fn rejects_division_by_zero() {
        let tool = CalculatorTool::new().unwrap();

        let result = tool.execute(&json!({
            "left": 10,
            "operator": "divide",
            "right": 0
        }));

        assert!(result.is_err());
    }

    #[test]
    fn rejects_unknown_operator() {
        let tool = CalculatorTool::new().unwrap();

        let result = tool.execute(&json!({
            "left": 10,
            "operator": "power",
            "right": 2
        }));

        assert!(result.is_err());
    }
}
