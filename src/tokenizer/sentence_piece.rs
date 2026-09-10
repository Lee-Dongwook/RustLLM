use std::fs;
use std::path::Path;

use sentencepiece::SentencePieceProcessor;
use serde::Deserialize;

use crate::error::{Result, TinyError};

use super::Tokenizer;

#[derive(Debug, Deserialize, Default)]
struct TokenizerConfigFile {
    #[serde(default)]
    add_bos_token: bool,

    #[serde(default)]
    add_eos_token: bool,
}

#[derive(Debug)]
pub struct SentencePieceTokenizer {
    processor: SentencePieceProcessor,

    add_bos_token: bool,

    add_eos_token: bool,
}

impl SentencePieceTokenizer {
    pub fn from_model_dir(model_dir: impl AsRef<Path>) -> Result<Self> {
        let model_dir = model_dir.as_ref();

        let model_path = model_dir.join("tokenizer.model");

        if !model_path.exists() {
            return Err(TinyError::Tokenizer(format!(
                "tokenizer model not found: {}",
                model_path.display(),
            )));
        }

        let processor = SentencePieceProcessor::open(&model_path).map_err(|error| {
            TinyError::Tokenizer(format!("failed to open {}: {error}", model_path.display(),))
        })?;

        let config_path = model_dir.join("tokenizer_config.json");

        let config = if config_path.exists() {
            let text = fs::read_to_string(&config_path)?;

            serde_json::from_str::<TokenizerConfigFile>(&text).map_err(|error| {
                TinyError::Tokenizer(format!("invalid tokenizer_config.json: {error}"))
            })?
        } else {
            TokenizerConfigFile::default()
        };

        Ok(Self {
            processor,
            add_bos_token: config.add_bos_token,
            add_eos_token: config.add_eos_token,
        })
    }
}

impl Tokenizer for SentencePieceTokenizer {
    fn encode(&self, text: &str) -> Result<Vec<u32>> {
        let pieces = self.processor.encode(text).map_err(|error| {
            TinyError::Tokenizer(format!("SentencePiece encode failed: {error}"))
        })?;

        let extra = usize::from(self.add_bos_token) + usize::from(self.add_eos_token);

        let mut token_ids = Vec::with_capacity(pieces.len() + extra);

        if self.add_bos_token {
            let bos = self.processor.bos_id().ok_or_else(|| {
                TinyError::Tokenizer(
                    "add_bos_token=true but tokenizer has no BOS token".to_string(),
                )
            })?;
            token_ids.push(bos);
        }

        token_ids.extend(pieces.into_iter().map(|piece| piece.id));

        if self.add_eos_token {
            let eos = self.processor.eos_id().ok_or_else(|| {
                TinyError::Tokenizer(
                    "add_eos_token=true but tokenizer has no EOS token".to_string(),
                )
            })?;

            token_ids.push(eos);
        }

        Ok(token_ids)
    }

    fn decode(&self, token_ids: &[u32], skip_special_tokens: bool) -> Result<String> {
        let filtered;

        let token_ids = if skip_special_tokens {
            filtered = token_ids
                .iter()
                .copied()
                .filter(|token_id| {
                    Some(*token_id) != self.processor.bos_id()
                        && Some(*token_id) != self.processor.eos_id()
                        && Some(*token_id) != self.processor.pad_id()
                        && *token_id != self.processor.unk_id()
                })
                .collect::<Vec<_>>();

            filtered.as_slice()
        } else {
            token_ids
        };

        self.processor
            .decode_piece_ids(token_ids)
            .map_err(|error| TinyError::Tokenizer(format!("SentencePiece decode failed: {error}")))
    }

    fn vocab_size(&self) -> usize {
        self.processor.len()
    }

    fn bos_token_id(&self) -> Option<u32> {
        self.processor.bos_id()
    }

    fn eos_token_id(&self) -> Option<u32> {
        self.processor.eos_id()
    }
}

#[cfg(test)]
mod tests {
    use std::path::PathBuf;

    use super::*;

    fn tiny_stories_model_dir() -> PathBuf {
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("models/source/tinystories-llama-15m")
    }

    #[test]
    #[ignore = "requires a locally downloaded Hugging Face model"]
    fn encodes_and_decodes_a_tiny_stories_prompt() {
        let tokenizer = SentencePieceTokenizer::from_model_dir(tiny_stories_model_dir())
            .expect("locally downloaded TinyStories tokenizer should load");

        let ids = tokenizer
            .encode("Once upon a time")
            .expect("prompt should encode");

        assert_eq!(ids.first().copied(), tokenizer.bos_token_id());
        assert!(ids.len() > 1);
        assert_eq!(
            tokenizer
                .decode(&ids, true)
                .expect("encoded prompt should decode"),
            "Once upon a time",
        );
    }
}
