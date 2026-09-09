use crate::error::{
    Result,
    TinyError,
};

use crate::tensor::Tensor;

pub fn greedy_next_token(
    logits: &Tensor,
) -> Result<u32> {
    if logits.rank() != 2 {
        return Err(
            TinyError::InvalidDimension(
                format!(
                    "greedy sampler expects [sequence, vocab], got shape {:?}",
                    logits.shape().dims(),
                ),
            ),
        );
    }

    if !logits.is_contiguous() {
        return Err(
            TinyError::Sampling(
                "greedy sampler requires contiguous logits"
                    .to_string(),
            ),
        );
    }

    let seq_len = logits.dim(0)?;
    let vocab_size = logits.dim(1)?;

    if seq_len == 0
        || vocab_size == 0
    {
        return Err(
            TinyError::Sampling(
                "logits cannot have an empty sequence or vocabulary"
                    .to_string(),
            ),
        );
    }

    let data = logits.as_f32_slice()?;

    let start = (seq_len - 1) * vocab_size;

    let last_logits = &data[start .. start+vocab_size];

    let mut best_index: Option<usize> = None;

    let mut best_value = f32::NEG_INFINITY;

    for(
        index,
        &value,
    ) in last_logits
        .iter()
        .enumerate()
    {
        if value.is_nan() {
            continue;
        }

        if best_index.is_none() || value > best_value {
            best_index = Some(index);
            best_value = value;
        }
    }

    let best_index = best_index
        .ok_or_else(|| {
            TinyError::Sampling(
                "all logits are NaN".to_string(),
            )
        })?;

    u32::try_from(
        best_index,
    )
    .map_err(|_| {
        TinyError::Sampling(
            "sampled token id exceeds u32".to_string(),
        )
    })
}
