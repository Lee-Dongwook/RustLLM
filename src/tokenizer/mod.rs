mod sentence_piece;
mod simple;

use crate::error::Result;

pub use sentence_piece::SentencePieceTokenizer;
pub use simple::CharTokenizer;

pub trait Tokenizer {
    fn encode(
        &self,
        text: &str,
    ) -> Result<Vec<u32>>;

    fn decode(
        &self,
        token_ids: &[u32],
        skip_special_tokens: bool,
    ) -> Result<String>;

    fn vocab_size(
        &self,
    ) -> usize;

    fn bos_token_id(
        &self,
    ) -> Option<u32>;

    fn eos_token_id(
        &self,
    ) -> Option<u32>;
}
