use crate::error::{Result, TinyError};

use super::Tokenizer;

pub struct StreamingDecoder<'a, T: Tokenizer + ?Sized> {
    tokenizer: &'a T,
    token_ids: Vec<u32>,
    decoded_text: String,
}

impl<'a, T: Tokenizer + ?Sized> StreamingDecoder<'a, T> {
    pub fn new(tokenizer: &'a T) -> Self {
        Self {
            tokenizer,
            token_ids: Vec::new(),
            decoded_text: String::new(),
        }
    }

    pub fn push(&mut self, token_id: u32) -> Result<String> {
        self.token_ids.push(token_id);

        let next_text = self.tokenizer.decode(&self.token_ids, true)?;

        if !next_text.starts_with(&self.decoded_text) {
            return Err(TinyError::Tokenizer(
                "streaming decode produced a non-prefix result".to_string(),
            ));
        }

        let delta = next_text[self.decoded_text.len()..].to_string();

        self.decoded_text = next_text;

        Ok(delta)
    }

    pub fn text(&self) -> &str {
        &self.decoded_text
    }

    pub fn token_ids(&self) -> &[u32] {
        &self.token_ids
    }
}
