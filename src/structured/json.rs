use serde::de::DeserializeOwned;

use crate::error::{Result, TinyError};

pub fn parse_json<T>(text: &str) -> Result<T>
where
    T: DeserializeOwned,
{
    let json = normalize_json(text)?;

    serde_json::from_str(json)
        .map_err(|error| TinyError::StructuredOutput(format!("failed to parse JSON: {error}")))
}

fn normalize_json(text: &str) -> Result<&str> {
    let text = text.trim();

    if text.starts_with("```") {
        return strip_code_fence(text);
    }

    Ok(text)
}

fn strip_code_fence(text: &str) -> Result<&str> {
    let Some(opening_end) = text.find('\n') else {
        return Err(TinyError::StructuredOutput(
            "JSON code fence must contain a newline after the opening fence".to_string(),
        ));
    };

    let opening = text[..opening_end].trim();

    if opening != "```" && !opening.eq_ignore_ascii_case("```json") {
        return Err(TinyError::StructuredOutput(format!(
            "unsupported code fence: {opening}"
        )));
    }

    if !text.ends_with("```") {
        return Err(TinyError::StructuredOutput(
            "JSON code fence is not closed".to_string(),
        ));
    }

    /*
     * opening:
     *
     * ```json\n
     *         ^
     *         opening_end
     *
     * 마지막 ``` 3글자도 제거.
     */
    let body = &text[opening_end + 1..text.len() - 3];

    let body = body.trim();

    if body.is_empty() {
        return Err(TinyError::StructuredOutput(
            "JSON code fence is empty".to_string(),
        ));
    }

    Ok(body)
}

#[cfg(test)]
mod tests {
    use serde::Deserialize;

    use super::parse_json;
    use crate::error::TinyError;

    #[derive(Debug, Deserialize, PartialEq, Eq)]
    struct Person {
        name: String,
        age: u32,
    }

    #[test]
    fn parses_plain_json() {
        let value: Person = parse_json(
            r#"{
                    "name": "Dongwook",
                    "age": 27
                }"#,
        )
        .unwrap();

        assert_eq!(
            value,
            Person {
                name: "Dongwook".to_string(),
                age: 27,
            },
        );
    }

    #[test]
    fn parses_json_code_fence() {
        let value: Person = parse_json(
            r#"```json
{
    "name": "Dongwook",
    "age": 27
}
```"#,
        )
        .unwrap();

        assert_eq!(value.name, "Dongwook",);

        assert_eq!(value.age, 27,);
    }

    #[test]
    fn parses_generic_code_fence() {
        let value: Person = parse_json(
            r#"```
{
    "name": "Dongwook",
    "age": 27
}
```"#,
        )
        .unwrap();

        assert_eq!(value.age, 27,);
    }

    #[test]
    fn rejects_surrounding_prose() {
        let result = parse_json::<Person>(
            r#"
Sure, here is the result:

{
    "name": "Dongwook",
    "age": 27
}
"#,
        );

        assert!(matches!(result, Err(TinyError::StructuredOutput(_))));
    }

    #[test]
    fn rejects_prose_after_code_fence() {
        let result = parse_json::<Person>(
            r#"```json
            {
                "name": "Dongwook",
                "age": 27
            }
            Hope this helps!"#,
        );

        assert!(result.is_err(),);
    }

    #[test]
    fn rejects_malformed_json() {
        let result = parse_json::<Person>(
            r#"{
                "name": "Dongwook",
                "age":
            }"#,
        );

        assert!(result.is_err(),);
    }
    #[test]
    fn validates_target_type() {
        let result = parse_json::<Person>(
            r#"{
                "name": "Dongwook",
                "age": "twenty seven"
            }"#,
        );

        assert!(result.is_err(),);
    }

    #[test]
    fn rejects_empty_code_fence() {
        let result = parse_json::<Person>("```json\n```");

        assert!(result.is_err(),);
    }
}
