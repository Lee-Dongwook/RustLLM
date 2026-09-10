mod huggingface;
mod sentence_piece;
mod simple;
mod stream;

use std::path::Path;

use crate::error::{Result, TinyError};

use huggingface::HuggingFaceTokenizer;
pub use sentence_piece::SentencePieceTokenizer;
pub use simple::CharTokenizer;
pub use stream::StreamingDecoder;

/// The common interface used by generation, independent of tokenizer format.
pub trait Tokenizer {
    fn encode(&self, text: &str) -> Result<Vec<u32>>;
    fn decode(&self, token_ids: &[u32], skip_special_tokens: bool) -> Result<String>;
    fn vocab_size(&self) -> usize;
    fn bos_token_id(&self) -> Option<u32>;
    fn eos_token_id(&self) -> Option<u32>;
}

pub enum ModelTokenizer {
    SentencePiece(SentencePieceTokenizer),
    HuggingFace(HuggingFaceTokenizer),
}

impl ModelTokenizer {
    /// Loads the tokenizer shipped with an imported model package.
    /// SentencePiece wins when both formats are present, preserving existing
    /// TinyStories package behaviour.
    pub fn from_model_dir(model_dir: impl AsRef<Path>) -> Result<Self> {
        let model_dir = model_dir.as_ref();

        if model_dir.join("tokenizer.model").exists() {
            return Ok(Self::SentencePiece(SentencePieceTokenizer::from_model_dir(
                model_dir,
            )?));
        }

        if model_dir.join("tokenizer.json").exists() {
            return Ok(Self::HuggingFace(HuggingFaceTokenizer::from_model_dir(
                model_dir,
            )?));
        }

        Err(TinyError::ModelFormat(format!(
            "no supported tokenizer found in {}: expected tokenizer.model or tokenizer.json",
            model_dir.display(),
        )))
    }
}

impl Tokenizer for ModelTokenizer {
    fn encode(&self, text: &str) -> Result<Vec<u32>> {
        match self {
            Self::SentencePiece(tokenizer) => tokenizer.encode(text),
            Self::HuggingFace(tokenizer) => tokenizer.encode(text),
        }
    }

    fn decode(&self, token_ids: &[u32], skip_special_tokens: bool) -> Result<String> {
        match self {
            Self::SentencePiece(tokenizer) => tokenizer.decode(token_ids, skip_special_tokens),
            Self::HuggingFace(tokenizer) => tokenizer.decode(token_ids, skip_special_tokens),
        }
    }

    fn vocab_size(&self) -> usize {
        match self {
            Self::SentencePiece(tokenizer) => tokenizer.vocab_size(),
            Self::HuggingFace(tokenizer) => tokenizer.vocab_size(),
        }
    }

    fn bos_token_id(&self) -> Option<u32> {
        match self {
            Self::SentencePiece(tokenizer) => tokenizer.bos_token_id(),
            Self::HuggingFace(tokenizer) => tokenizer.bos_token_id(),
        }
    }

    fn eos_token_id(&self) -> Option<u32> {
        match self {
            Self::SentencePiece(tokenizer) => tokenizer.eos_token_id(),
            Self::HuggingFace(tokenizer) => tokenizer.eos_token_id(),
        }
    }
}

#[cfg(test)]
mod tests {
    use std::{
        fs,
        time::{SystemTime, UNIX_EPOCH},
    };

    use super::{ModelTokenizer, Tokenizer};

    #[test]
    fn selects_huggingface_tokenizer_json() {
        let unique = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let model_dir = std::env::temp_dir().join(format!(
            "tiny-metal-llm-tokenizer-{unique}-{}",
            std::process::id(),
        ));
        fs::create_dir_all(&model_dir).unwrap();
        fs::write(
            model_dir.join("tokenizer.json"),
            r#"{"version":"1.0","truncation":null,"padding":null,"added_tokens":[{"id":0,"content":"<unk>","single_word":false,"lstrip":false,"rstrip":false,"normalized":false,"special":true}],"normalizer":null,"pre_tokenizer":{"type":"Whitespace"},"post_processor":null,"decoder":null,"model":{"type":"WordLevel","vocab":{"<unk>":0,"Hello":1,"world":2},"unk_token":"<unk>"}}"#,
        )
        .unwrap();

        let tokenizer = ModelTokenizer::from_model_dir(&model_dir).unwrap();
        assert_eq!(tokenizer.vocab_size(), 3);
        assert_eq!(tokenizer.encode("Hello world").unwrap(), vec![1, 2]);
        assert_eq!(tokenizer.decode(&[1, 2], true).unwrap(), "Hello world");

        fs::remove_dir_all(model_dir).unwrap();
    }
}
