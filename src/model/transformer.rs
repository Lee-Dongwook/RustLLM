use crate::error::{
    Result,
    TinyError,
};

use crate::metal::MetalContext;

use crate::nn::{
    Embedding,
    Linear,
    RmsNorm,
    TransformerBlock,
};

use crate::tensor::Tensor;

use super::ModelConfig;

pub struct Transformer {
    config: ModelConfig,

    token_embedding: Embedding,

    blocks:
        Vec<TransformerBlock>,

    final_norm: RmsNorm,

    lm_head: Linear,
}

impl Transformer {
    pub fn new(
        config: ModelConfig,
        token_embedding: Embedding,
        blocks: Vec<TransformerBlock>,
        final_norm: RmsNorm,
        lm_head: Linear,
    ) -> Result<Self> {
        config.validate()?;

        if token_embedding.vocab_size()
            != config.vocab_size
        {
            return Err(
                TinyError::InvalidShape(
                    format!(
                        "embedding vocabulary size {} does not match model vocabulary size {}",
                        token_embedding.vocab_size(),
                        config.vocab_size,
                    ),
                ),
            );
        }

        if token_embedding.hidden_size()
            != config.hidden_size
        {
            return Err(
                TinyError::InvalidShape(
                    format!(
                        "embedding hidden size {} does not match model hidden size {}",
                        token_embedding.hidden_size(),
                        config.hidden_size,
                    ),
                ),
            );
        }

        if blocks.len()
            != config.num_layers
        {
            return Err(
                TinyError::InvalidShape(
                    format!(
                        "model expects {} transformer layers, got {}",
                        config.num_layers,
                        blocks.len(),
                    ),
                ),
            );
        }

        for (
            index,
            block,
        ) in blocks
            .iter()
            .enumerate()
        {
            if block.hidden_size()
                != config.hidden_size
            {
                return Err(
                    TinyError::InvalidShape(
                        format!(
                            "transformer block {index} has hidden size {}, expected {}",
                            block.hidden_size(),
                            config.hidden_size,
                        ),
                    ),
                );
            }
        }

        if final_norm.hidden_size()
            != config.hidden_size
        {
            return Err(
                TinyError::InvalidShape(
                    format!(
                        "final RMSNorm hidden size {} does not match model hidden size {}",
                        final_norm.hidden_size(),
                        config.hidden_size,
                    ),
                ),
            );
        }

        if lm_head.in_features()
            != config.hidden_size
        {
            return Err(
                TinyError::InvalidShape(
                    format!(
                        "LM head input size {} does not match model hidden size {}",
                        lm_head.in_features(),
                        config.hidden_size,
                    ),
                ),
            );
        }

        if lm_head.out_features()
            != config.vocab_size
        {
            return Err(
                TinyError::InvalidShape(
                    format!(
                        "LM head output size {} does not match vocabulary size {}",
                        lm_head.out_features(),
                        config.vocab_size,
                    ),
                ),
            );
        }

        Ok(Self {
            config,
            token_embedding,
            blocks,
            final_norm,
            lm_head,
        })
    }

    pub fn config(
        &self,
    ) -> &ModelConfig {
        &self.config
    }

    pub fn forward(
        &self,
        context: &MetalContext,
        token_ids: &[u32],
    ) -> Result<Tensor> {
        if token_ids.is_empty() {
            return Err(
                TinyError::InvalidShape(
                    "Transformer input cannot be empty"
                        .to_string(),
                ),
            );
        }

        if token_ids.len()
            > self.config.max_seq_len
        {
            return Err(
                TinyError::PositionOutOfRange {
                    start_pos: 0,
                    seq_len:
                        token_ids.len(),
                    max_seq_len:
                        self.config
                            .max_seq_len,
                },
            );
        }

        // -----------------------------------------------
        // Token IDs
        // ↓
        // Embedding
        //
        // [sequence, hidden]
        // -----------------------------------------------

        let mut hidden =
            self.token_embedding
                .forward(
                    context,
                    token_ids,
                )?;

        // -----------------------------------------------
        // Transformer layers
        // -----------------------------------------------

        for block in &self.blocks {
            hidden =
                block.forward(
                    context,
                    &hidden,
                )?;
        }

        // -----------------------------------------------
        // Final RMSNorm
        // -----------------------------------------------

        let hidden =
            self.final_norm
                .forward(
                    context,
                    &hidden,
                )?;

        // -----------------------------------------------
        // LM Head
        //
        // [sequence, hidden]
        //
        // ×
        //
        // [hidden, vocab]
        //
        // →
        //
        // [sequence, vocab]
        // -----------------------------------------------

        self.lm_head.forward(
            context,
            &hidden,
        )
    }
}
