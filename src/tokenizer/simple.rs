use std::collections::HashMap;
use std::fs;
use std::path::Path;

use serde::Deserialize;

use crate::error::{
    Result,
    TinyError,
};

use super::Tokenizer;

#[derive(
    Debug,
    Deserialize,
)]
struct TokenizerFile {
    #[serde(rename = "type")]
    tokenizer_type: String,

    vocab: Vec<String>,

    bos_token_id: Option<u32>,
    eos_token_id: Option<u32>,
    unk_token_id: Option<u32>,
}

#[derive(Debug)]
pub struct CharTokenizer {
    id_to_token: Vec<String>,
    token_to_id:
        HashMap<String, u32>,
    bos_token_id: Option<u32>,
    eos_token_id: Option<u32>,
    unk_token_id: Option<u32>,
}

impl CharTokenizer {
    pub fn load_json(
        path: impl AsRef<Path>,
    ) -> Result<Self> {
        let text =
            fs::read_to_string(
                path,
            )?;

        let file:
            TokenizerFile =
            serde_json::from_str(
                &text,
            )
            .map_err(|error| {
                TinyError::Tokenizer(
                    format!(
                        "invalid tokenizer.json: {error}"
                    ),
                )
            })?;

        if file.tokenizer_type
            != "char"
        {
            return Err(
                TinyError::Tokenizer(
                    format!(
                        "unsupported tokenizer type: {}",
                        file.tokenizer_type,
                    ),
                ),
            );
        }

        if file.vocab.is_empty() {
            return Err(
                TinyError::Tokenizer(
                    "vocabulary cannot be empty"
                        .to_string(),
                ),
            );
        }

        let mut token_to_id =
            HashMap::with_capacity(
                file.vocab.len(),
            );

        for (
            index,
            token,
        ) in file.vocab
            .iter()
            .enumerate()
        {
            if token.is_empty() {
                return Err(
                    TinyError::Tokenizer(
                        format!(
                            "token {index} is empty"
                        ),
                    ),
                );
            }

            if token_to_id
                .contains_key(token)
            {
                return Err(
                    TinyError::Tokenizer(
                        format!(
                            "duplicate token: {token:?}"
                        ),
                    ),
                );
            }

            let token_id =
                u32::try_from(
                    index,
                )
                .map_err(|_| {
                    TinyError::Tokenizer(
                        "vocabulary exceeds u32 token IDs"
                            .to_string(),
                    )
                })?;

            token_to_id.insert(
                token.clone(),
                token_id,
            );
        }

        validate_special_id(
            "bos_token_id",
            file.bos_token_id,
            file.vocab.len(),
        )?;

        validate_special_id(
            "eos_token_id",
            file.eos_token_id,
            file.vocab.len(),
        )?;

        validate_special_id(
            "unk_token_id",
            file.unk_token_id,
            file.vocab.len(),
        )?;

        Ok(Self {
            id_to_token:
                file.vocab,

            token_to_id,

            bos_token_id:
                file.bos_token_id,

            eos_token_id:
                file.eos_token_id,

            unk_token_id:
                file.unk_token_id,
        })
    }

    pub fn vocab_size(
        &self,
    ) -> usize {
        self.id_to_token.len()
    }

    pub fn bos_token_id(
        &self,
    ) -> Option<u32> {
        self.bos_token_id
    }

    pub fn eos_token_id(
        &self,
    ) -> Option<u32> {
        self.eos_token_id
    }

    pub fn unk_token_id(
        &self,
    ) -> Option<u32> {
        self.unk_token_id
    }

    pub fn encode(
        &self,
        text: &str,
    ) -> Result<Vec<u32>> {
        if text.is_empty() {
            return Err(
                TinyError::Tokenizer(
                    "cannot encode an empty string"
                        .to_string(),
                ),
            );
        }

        let mut tokens =
            Vec::new();

        for character in text.chars() {
            let token =
                character.to_string();

            if let Some(
                &token_id
            ) = self.token_to_id
                .get(&token)
            {
                tokens.push(
                    token_id,
                );

                continue;
            }

            if let Some(
                unk_token_id
            ) = self.unk_token_id
            {
                tokens.push(
                    unk_token_id,
                );

                continue;
            }

            return Err(
                TinyError::Tokenizer(
                    format!(
                        "character {character:?} is not in the vocabulary"
                    ),
                ),
            );
        }

        Ok(tokens)
    }

    pub fn decode(
        &self,
        token_ids: &[u32],
        skip_special_tokens: bool,
    ) -> Result<String> {
        let mut output =
            String::new();

        for &token_id
            in token_ids
        {
            let index =
                token_id as usize;

            let token =
                self.id_to_token
                    .get(index)
                    .ok_or_else(|| {
                        TinyError::Tokenizer(
                            format!(
                                "token id {token_id} is outside vocabulary size {}",
                                self.vocab_size(),
                            ),
                        )
                    })?;

            if skip_special_tokens
                && self.is_special_token(
                    token_id,
                )
            {
                continue;
            }

            output.push_str(
                token,
            );
        }

        Ok(output)
    }

    fn is_special_token(
        &self,
        token_id: u32,
    ) -> bool {
        self.bos_token_id
            == Some(token_id)

            || self.eos_token_id
                == Some(token_id)

            || self.unk_token_id
                == Some(token_id)
    }
}

impl Tokenizer for CharTokenizer {
    fn encode(
        &self,
        text: &str,
    ) -> Result<Vec<u32>> {
        CharTokenizer::encode(self, text)
    }

    fn decode(
        &self,
        token_ids: &[u32],
        skip_special_tokens: bool,
    ) -> Result<String> {
        CharTokenizer::decode(
            self,
            token_ids,
            skip_special_tokens,
        )
    }

    fn vocab_size(
        &self,
    ) -> usize {
        CharTokenizer::vocab_size(self)
    }

    fn bos_token_id(
        &self,
    ) -> Option<u32> {
        CharTokenizer::bos_token_id(self)
    }

    fn eos_token_id(
        &self,
    ) -> Option<u32> {
        CharTokenizer::eos_token_id(self)
    }
}

fn validate_special_id(
    name: &str,
    token_id: Option<u32>,
    vocab_size: usize,
) -> Result<()> {
    let Some(token_id) =
        token_id
    else {
        return Ok(());
    };

    if token_id as usize
        >= vocab_size
    {
        return Err(
            TinyError::Tokenizer(
                format!(
                    "{name}={token_id} exceeds vocabulary size {vocab_size}"
                ),
            ),
        );
    }

    Ok(())
}
