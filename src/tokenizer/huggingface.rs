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
        let path = model_dir.join("tokenizer.json");
        let tokenizer = HfTokenizer::from_file(&path).map_err(|error| {
            TinyError::Tokenizer(format!("failed to load {}: {error}", path.display()))
        })?;

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
