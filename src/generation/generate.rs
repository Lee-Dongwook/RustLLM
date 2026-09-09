use crate::{
    error::{
        Result,
        TinyError,
    },
    metal::MetalContext,
    model::Transformer,
};

use super::greedy_next_token;

pub fn generate_greedy_stream<F>(
    context: &MetalContext,
    model: &Transformer,
    prompt_tokens: &[u32],
    max_new_tokens: usize,
    eos_token_id: Option<u32>,
    mut on_token: F,
) -> Result<Vec<u32>> where F: FnMut(u32) -> Result<()> {
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
    let mut cache = model.new_kv_cache(context)?;

    let mut logits = model.forward_with_cache(context, prompt_tokens, &mut cache)?;

    for _ in 0..max_new_tokens {
        let next_token = 
            greedy_next_token(
                &logits,
            )?;

        tokens.push(
            next_token,
        );

        on_token(
            next_token,
        )?;

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

        logits =
            model.forward_with_cache(
                context,
                &[next_token],
                &mut cache,
            )?;
    }

    Ok(tokens)
}

pub fn generate_greedy(
    context: &MetalContext,
    model: &Transformer,
    prompt_tokens: &[u32],
    max_new_tokens: usize,
    eos_token_id: Option<u32>,
) -> Result<Vec<u32>> {
    generate_greedy_stream(
        context,
        model,
        prompt_tokens,
        max_new_tokens,
        eos_token_id,
        |_| Ok(()),
    )
}
