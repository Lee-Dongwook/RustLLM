use std::{fs, path::Path};

use serde_json::Value;
use tokenizers::Tokenizer as HfTokenizer;

use crate::error::{Result, TinyError};

use super::Tokenizer;

pub struct HuggingFaceTokenizer {
    tokenizer: HfTokenizer,
    bos_token_id: Option<u32>,
    eos_token_id: Option<u32>,
}

impl HuggingFaceTokenizer {
    pub fn from_model_dir(model_dir: impl AsRef<Path>) -> Result<Self> {
        let model_dir = model_dir.as_ref();
        let tokenizer_path = model_dir.join("tokenizer.json");

        if !tokenizer_path.exists() {
            return Err(TinyError::Tokenizer(format!(
                "tokenizer file not found: {}",
                tokenizer_path.display(),
            )));
        }
        let mut tokenizer = HfTokenizer::from_file(&tokenizer_path).map_err(|error| {
            TinyError::Tokenizer(format!(
                "failed to load Hugging Face tokenizer {}: {error}",
                tokenizer_path.display(),
            ))
        })?;

        tokenizer.set_encode_special_tokens(false);

        // Encoding lives in tokenizer.json. BOS/EOS are generation policy, so
        // obtain them from Hugging Face's optional sidecar instead of assuming
        // any particular GPT-2 or Llama special-token spelling.
        let (bos_token_id, eos_token_id) = special_token_ids(model_dir, &tokenizer)?;

        Ok(Self {
            tokenizer,
            bos_token_id,
            eos_token_id,
        })
    }
}

impl Tokenizer for HuggingFaceTokenizer {
    fn encode(&self, text: &str) -> Result<Vec<u32>> {
        let encoding = self.tokenizer.encode(text, false).map_err(|error| {
            TinyError::Tokenizer(format!("Hugging Face tokenizer encode failed: {error}"))
        })?;
        Ok(encoding.get_ids().to_vec())
    }

    fn decode(&self, token_ids: &[u32], skip_special_tokens: bool) -> Result<String> {
        self.tokenizer
            .decode(token_ids, skip_special_tokens)
            .map_err(|error| {
                TinyError::Tokenizer(format!("Hugging Face tokenizer decode failed: {error}"))
            })
    }

    fn vocab_size(&self) -> usize {
        self.tokenizer.get_vocab_size(true)
    }

    fn bos_token_id(&self) -> Option<u32> {
        self.bos_token_id
    }

    fn eos_token_id(&self) -> Option<u32> {
        self.eos_token_id
    }
}

fn special_token_ids(
    model_dir: &Path,
    tokenizer: &HfTokenizer,
) -> Result<(Option<u32>, Option<u32>)> {
    let config_path = model_dir.join("tokenizer_config.json");
    if !config_path.exists() {
        return Ok((None, None));
    }

    let text = fs::read_to_string(&config_path)?;
    let config: Value = serde_json::from_str(&text).map_err(|error| {
        TinyError::Tokenizer(format!("invalid {}: {error}", config_path.display()))
    })?;

    Ok((
        config
            .get("bos_token")
            .and_then(token_content)
            .and_then(|token| tokenizer.token_to_id(token)),
        config
            .get("eos_token")
            .and_then(token_content)
            .and_then(|token| tokenizer.token_to_id(token)),
    ))
}

fn token_content(value: &Value) -> Option<&str> {
    value.as_str().or_else(|| value.get("content")?.as_str())
}

#[cfg(test)]
mod tests {
    use std::path::PathBuf;
    use std::{
        fs,
        time::{SystemTime, UNIX_EPOCH},
    };

    use std::sync::atomic::{AtomicU64, Ordering};

    use super::HuggingFaceTokenizer;
    use crate::tokenizer::Tokenizer;

    static TEST_DIR_COUNTER: AtomicU64 = AtomicU64::new(0);

    fn test_model_dir() -> PathBuf {
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("models/SmolLM2-135M-Instruct")
    }

    fn create_test_model_dir() -> std::path::PathBuf {
        let counter = TEST_DIR_COUNTER.fetch_add(1, Ordering::Relaxed);
        let unique = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();

        let model_dir = std::env::temp_dir().join(format!(
            "rustllm-hf-special-token-test-{unique}-{:?}-{}-{counter}",
            std::thread::current().id(),
            std::process::id(),
        ));

        fs::create_dir_all(&model_dir).unwrap();

        /*
         * 일반 vocab:
         *
         * 0 = <unk>
         * 1 = system
         * 2 = user
         * 3 = assistant
         * 4 = Hello
         *
         * added special tokens:
         *
         * 5 = <|im_start|>
         * 6 = <|im_end|>
         */
        fs::write(
            model_dir.join("tokenizer.json"),
            r#"{
                "version": "1.0",
                "truncation": null,
                "padding": null,
                "added_tokens": [
                    {
                        "id": 0,
                        "content": "<unk>",
                        "single_word": false,
                        "lstrip": false,
                        "rstrip": false,
                        "normalized": false,
                        "special": true
                    },
                    {
                        "id": 5,
                        "content": "<|im_start|>",
                        "single_word": false,
                        "lstrip": false,
                        "rstrip": false,
                        "normalized": false,
                        "special": true
                    },
                    {
                        "id": 6,
                        "content": "<|im_end|>",
                        "single_word": false,
                        "lstrip": false,
                        "rstrip": false,
                        "normalized": false,
                        "special": true
                    }
                ],
                "normalizer": null,
                "pre_tokenizer": {
                    "type": "Whitespace"
                },
                "post_processor": null,
                "decoder": null,
                "model": {
                    "type": "WordLevel",
                    "vocab": {
                        "<unk>": 0,
                        "system": 1,
                        "user": 2,
                        "assistant": 3,
                        "Hello": 4
                    },
                    "unk_token": "<unk>"
                }
            }"#,
        )
        .unwrap();

        fs::write(
            model_dir.join("tokenizer_config.json"),
            r#"{
                "bos_token": null,
                "eos_token": "<|im_end|>"
            }"#,
        )
        .unwrap();

        model_dir
    }

    #[test]
    fn encodes_im_start_as_single_special_token() {
        let model_dir = create_test_model_dir();

        let tokenizer = HuggingFaceTokenizer::from_model_dir(&model_dir).unwrap();

        let ids = tokenizer.encode("<|im_start|>").unwrap();

        assert_eq!(
            ids,
            vec![5],
            "<|im_start|> must encode to exactly one token",
        );

        let _ = fs::remove_dir_all(&model_dir);
    }

    #[test]
    fn encodes_im_end_as_single_special_token() {
        let model_dir = create_test_model_dir();

        let tokenizer = HuggingFaceTokenizer::from_model_dir(&model_dir).unwrap();

        let ids = tokenizer.encode("<|im_end|>").unwrap();

        assert_eq!(ids, vec![6], "<|im_end|> must encode to exactly one token",);

        let _ = fs::remove_dir_all(&model_dir);
    }

    #[test]
    fn resolves_im_end_as_eos_token() {
        let model_dir = create_test_model_dir();

        let tokenizer = HuggingFaceTokenizer::from_model_dir(&model_dir).unwrap();

        assert_eq!(tokenizer.eos_token_id(), Some(6),);

        assert_eq!(tokenizer.bos_token_id(), None,);

        let _ = fs::remove_dir_all(&model_dir);
    }

    #[test]
    fn preserves_special_tokens_inside_normal_text() {
        let model_dir = test_model_dir();

        let tokenizer = HuggingFaceTokenizer::from_model_dir(&model_dir).unwrap();

        let im_start_id = tokenizer
            .tokenizer
            .token_to_id("<|im_start|>")
            .expect("<|im_start|> token must exist");

        let im_end_id = tokenizer
            .tokenizer
            .token_to_id("<|im_end|>")
            .expect("<|im_end|> token must exist");

        let token_ids = tokenizer
            .encode("hello <|im_start|>assistant<|im_end|> world")
            .unwrap();

        assert_eq!(token_ids.iter().filter(|&&id| id == im_start_id).count(), 1,);

        assert_eq!(token_ids.iter().filter(|&&id| id == im_end_id).count(), 1,);
    }
}
