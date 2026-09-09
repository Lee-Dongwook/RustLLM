use crate::error::{
    Result,
    TinyError,
};

use crate::metal::MetalContext;
use crate::model::Transformer;

use super::greedy_next_token;

pub fn generate_greedy(
    context: &MetalContext,
    model: &Transformer,
    prompt_tokens: &[u32],
    max_new_tokens: usize,
    eos_token_id: Option<u32>,
) -> Result<Vec<u32>> {
    if prompt_tokens.is_empty() {
        return Err(
            TinyError::InvalidShape(
                "generation prompt cannot be empty"
                    .to_string(),
            ),
        );
    }

    let max_seq_len = model.config().max_seq_len;

    if prompt_tokens.len() > max_seq_len {
        return Err(
            TinyError::PositionOutOfRange {
                start_pos: 0,
                seq_len:
                    prompt_tokens.len(),
                max_seq_len,
            },
        );
    }

    let mut tokens = prompt_tokens.to_vec();

    for _ in 0..max_new_tokens {
        let logits = 
            model.forward(
                context,
                &tokens,
            )?;
        
        let next_token = 
            greedy_next_token(
                &logits,
            )?;
        
        tokens.push(
            next_token,
        );

        if eos_token_id
            == Some(next_token)
        {
            break;
        }

        if tokens.len()
            >= max_seq_len
        {
            break;
        }
    }

    Ok(tokens)
}
