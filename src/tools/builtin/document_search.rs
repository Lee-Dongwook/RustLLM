use serde_json::{Value, json};

use crate::{
    error::{Result, TinyError},
    retrieval::{LexicalRetriever, Retriever},
    structured::JsonSchema,
};

use super::super::Tool;

pub struct DocumentSearchTool {
    retriever: LexicalRetriever,
    schema: JsonSchema,
    top_k: usize,
}

impl DocumentSearchTool {
    pub fn new(retriever: LexicalRetriever, top_k: usize) -> Result<Self> {
        if top_k == 0 {
            return Err(TinyError::Tool(
                "document search top_k must be greater than zero".to_string(),
            ));
        }

        Ok(Self {
            retriever,
            schema: JsonSchema::new(
                "document_search",
                r#"{
                    "query": "string"
                }"#,
            )?,
            top_k,
        })
    }

    pub fn top_k(&self) -> usize {
        self.top_k
    }
}

impl Tool for DocumentSearchTool {
    fn name(&self) -> &str {
        "document_search"
    }

    fn description(&self) -> &str {
        "Searches the loaded documents for information relevant to a query."
    }

    fn input_schema(&self) -> &JsonSchema {
        &self.schema
    }

    fn validate_arguments(&self, arguments: &Value) -> Result<()> {
        self.input_schema().validate(arguments)?;

        let query = arguments
            .get("query")
            .and_then(Value::as_str)
            .map(str::trim)
            .ok_or_else(|| {
                TinyError::Tool("document_search `query` must be a string".to_string())
            })?;

        if query.is_empty() {
            return Err(TinyError::Tool(
                "document_search `query` cannot be empty".to_string(),
            ));
        }

        if matches!(
            query.to_ascii_lowercase().as_str(),
            "string" | "number" | "boolean" | "object" | "array"
        ) {
            return Err(TinyError::Tool(format!(
                "document_search `query` contains schema placeholder `{query}` instead of an actual search query"
            )));
        }

        Ok(())
    }

    fn execute(&self, arguments: &Value) -> Result<Value> {
        let query = arguments
            .get("query")
            .and_then(Value::as_str)
            .map(str::trim)
            .filter(|query| !query.is_empty())
            .ok_or_else(|| {
                TinyError::Tool(
                    "document_search argument `query` must be a non-empty string".to_string(),
                )
            })?;

        let results = self.retriever.retrieve(query, self.top_k)?;

        let matches = results
            .into_iter()
            .map(|retrieved| {
                let score = retrieved.score();

                let chunk = retrieved.into_chunk();

                json!({
                    "source": chunk.source().display().to_string(),
                    "chunk_index": chunk.index(),
                    "start_char": chunk.start_char(),
                    "end_char": chunk.end_char(),
                    "score": score,
                    "text": chunk.text(),
                })
            })
            .collect::<Vec<_>>();

        Ok(json!({
            "query": query,
            "matches": matches,
        }))
    }
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::DocumentSearchTool;

    use crate::{
        document::{ChunkConfig, Document, chunk_document},
        retrieval::LexicalRetriever,
        tools::Tool,
    };

    fn test_tool() -> DocumentSearchTool {
        let documents = [
            (
                "runtime.md",
                "RustLLM uses a KV cache during autoregressive generation. \
                 The KV cache avoids recomputing keys and values for previous tokens.",
            ),
            (
                "metal.md",
                "RustLLM uses Apple Metal to execute tensor operations on the GPU.",
            ),
            (
                "quantization.md",
                "RustLLM supports INT8 weight-only quantization with F16 activations.",
            ),
        ];

        let chunks = documents
            .into_iter()
            .flat_map(|(source, text)| {
                let document = Document::new(source, text);

                chunk_document(&document, &ChunkConfig::new(512, 0).unwrap()).unwrap()
            })
            .collect();

        let retriever = LexicalRetriever::new(chunks).unwrap();

        DocumentSearchTool::new(retriever, 2).unwrap()
    }

    #[test]
    fn searches_documents() {
        let tool = test_tool();

        let output = tool
            .execute(&json!({
                "query": "KV cache previous tokens"
            }))
            .unwrap();

        let matches = output["matches"].as_array().unwrap();

        assert!(!matches.is_empty());

        assert_eq!(matches[0]["source"], "runtime.md",);

        assert!(matches[0]["text"].as_str().unwrap().contains("KV cache"));
    }

    #[test]
    fn returns_empty_matches_when_nothing_is_found() {
        let tool = test_tool();

        let output = tool
            .execute(&json!({
                "query": "elephant telescope banana"
            }))
            .unwrap();

        assert_eq!(output["matches"].as_array().unwrap().len(), 0,);
    }

    #[test]
    fn rejects_empty_query() {
        let tool = test_tool();

        let result = tool.execute(&json!({
            "query": "   "
        }));

        assert!(result.is_err());
    }

    #[test]
    fn rejects_placeholder_query() {
        let tool = test_tool();

        let result = tool.validate_arguments(&json!({
            "query": "string"
        }));

        assert!(result.is_err());
    }

    #[test]
    fn rejects_zero_top_k() {
        let document = Document::new("test.md", "test document");

        let chunks = chunk_document(&document, &ChunkConfig::new(512, 0).unwrap()).unwrap();

        let retriever = LexicalRetriever::new(chunks).unwrap();

        assert!(DocumentSearchTool::new(retriever, 0,).is_err());
    }
}
