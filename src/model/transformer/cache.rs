use crate::{
    error::{Result, TinyError},
    metal::MetalContext,
    tensor::Tensor,
};

use super::{KvCache, Transformer};

impl Transformer {
    /// Creates a cache whose layer count and context limit match this model.
    pub fn new_kv_cache(&self, context: &MetalContext) -> Result<KvCache> {
        KvCache::new(
            context,
            self.config.num_layers,
            self.config.max_seq_len,
            self.config.num_kv_heads,
            self.config.head_dim(),
            self.dtype(),
        )
    }

    /// Appends the given tokens' key/value tensors and returns their logits.
    pub fn forward_with_cache(
        &self,
        context: &MetalContext,
        token_ids: &[u32],
        cache: &mut KvCache,
    ) -> Result<Tensor> {
        if token_ids.is_empty() {
            return Err(TinyError::ModelFormat(
                "token_ids cannot be empty".to_string(),
            ));
        }
        if cache.num_layers() != self.blocks.len() {
            return Err(TinyError::ModelFormat(format!(
                "KV cache has {} layers but model has {}",
                cache.num_layers(),
                self.blocks.len(),
            )));
        }
        if cache.max_seq_len() != self.config.max_seq_len {
            return Err(TinyError::ModelFormat(format!(
                "KV cache context length {} does not match model context length {}",
                cache.max_seq_len(),
                self.config.max_seq_len,
            )));
        }

        let mut hidden = self.token_embedding.forward(context, token_ids)?;
        for (index, block) in self.blocks.iter().enumerate() {
            let layer_cache = cache.layer_mut(index)?;
            if layer_cache.len() + token_ids.len() > self.config.max_seq_len {
                return Err(TinyError::PositionOutOfRange {
                    start_pos: layer_cache.len(),
                    seq_len: token_ids.len(),
                    max_seq_len: self.config.max_seq_len,
                });
            }
            hidden = block.forward_with_cache(context, &hidden, layer_cache)?;
        }

        let hidden = self.final_norm.forward(context, &hidden)?;
        self.lm_head.forward(context, &hidden)
    }
}
