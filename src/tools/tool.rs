use serde_json::Value;

use crate::{error::Result, structured::JsonSchema};

pub trait Tool {
    fn name(&self) -> &str;

    fn description(&self) -> &str;

    fn input_schema(&self) -> &JsonSchema;

    fn validate_arguments(&self, arguments: &Value) -> Result<()> {
        self.input_schema().validate(arguments)
    }

    fn execute(&self, arguments: &Value) -> Result<Value>;
}
