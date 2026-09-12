use crate::{
    error::{Result, TinyError},
    metal::{MetalContext, MetalExecution},
    profile::DecodeProfile,
    tensor::Tensor,
};
use std::time::Instant;

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

        // Embedding, every block, the final norm and the LM head share one
        // command buffer, so a decode step commits and waits exactly once.
        let execution = MetalExecution::new(context);
        let command_buffer = execution.command_buffer();

        let mut hidden = self
            .token_embedding
            .forward_encode(context, command_buffer, token_ids)?;
        for (index, block) in self.blocks.iter().enumerate() {
            let layer_cache = cache.layer_mut(index)?;
            if layer_cache.len() + token_ids.len() > self.config.max_seq_len {
                return Err(TinyError::PositionOutOfRange {
                    start_pos: layer_cache.len(),
                    seq_len: token_ids.len(),
                    max_seq_len: self.config.max_seq_len,
                });
            }
            hidden =
                block.forward_with_cache_encode(context, command_buffer, &hidden, layer_cache)?;
        }

        let hidden = self
            .final_norm
            .forward_encode(context, command_buffer, &hidden)?;
        let logits = self
            .lm_head
            .forward_encode(context, command_buffer, &hidden)?;

        execution.finish();

        Ok(logits)
    }

    /// Profiles one decode forward pass. Callers must only use this with one
    /// token; prefill deliberately remains outside the accumulated profile.
    pub fn forward_with_cache_profiled(
        &self,
        context: &MetalContext,
        token_ids: &[u32],
        cache: &mut KvCache,
        profile: &mut DecodeProfile,
    ) -> Result<Tensor> {
        if token_ids.len() != 1 {
            return self.forward_with_cache(context, token_ids, cache);
        }

        // A decode step is a single command buffer now, so the per-stage CPU
        // timings would only measure encoding. Profile the whole step instead
        // and keep production on exactly the same path.
        let started = Instant::now();
        let logits = self.forward_with_cache(context, token_ids, cache)?;
        profile.decode_step += started.elapsed();
        Ok(logits)
    }
}
