use crate::{
    error::{
        Result,
        TinyError,
    },
    metal::MetalContext,
    ops::concat_sequence_f32,
    tensor::Tensor,
};
#[derive(Default)]
pub struct LayerKvCache {
    key: Option<Tensor>,
    value: Option<Tensor>,
    len: usize,
}

impl LayerKvCache {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn len(&self) -> usize {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    pub fn key(&self) -> Result<&Tensor> {
        self.key
            .as_ref()
            .ok_or_else(|| {
                TinyError::ModelFormat(
                    "KV cache has no key tensor".to_string(),
                )
            })
    }

    pub fn value(&self) -> Result<&Tensor> {
        self.value
            .as_ref()
            .ok_or_else(|| {
                TinyError::ModelFormat(
                    "KV cache has no value tensor".to_string(),
                )
            })
    }

    pub fn append(
        &mut self,
        context: &MetalContext,
        key: Tensor,
        value: Tensor,
    ) -> Result<()> {
        if key.rank() != 4 || value.rank() != 4 {
            return Err(
                TinyError::ModelFormat(
                    format!(
                        "KV cache expects rank-4 tensors, got key {:?}, value {:?}",
                        key.shape().dims(),
                        value.shape().dims(),
                    ),
                ),
            );
        }

        if key.shape().dims()
            != value.shape().dims()
        {
            return Err(
                TinyError::ModelFormat(
                    "key/value shapes must match"
                        .to_string(),
                ),
            );
        }

        let new_seq_len = key.dim(2)?;

        if new_seq_len == 0 {
            return Err(
                TinyError::ModelFormat(
                    "KV cache cannot append an empty sequence".to_string(),
                ),
            );
        }

        if self.is_empty() {
            self.key = Some(key);
            self.value = Some(value);
            self.len = new_seq_len;

            return Ok(());
        }

        let cached_key = self.key()?;

        let cached_value = self.value()?;

        let next_key = concat_sequence_f32(context, cached_key, &key,)?;

        let next_value = concat_sequence_f32(context, cached_value, &value,)?;

        self.key = Some(next_key);
        self.value = Some(next_value);

        self.len += new_seq_len;

        Ok(())
    }
}

pub struct KvCache {
    layers: Vec<LayerKvCache>,
    max_seq_len: usize,
}

impl KvCache {
    pub fn new(
        num_layers: usize,
        max_seq_len: usize,
    ) -> Self {
        let layers = 
            (0..num_layers)
                .map(|_| {
                    LayerKvCache::new()
                })
                .collect();
        
        Self {
            layers,
            max_seq_len,
        }
    }

    pub fn layer_mut(
        &mut self,
        index: usize,
    ) -> Result<&mut LayerKvCache> {
        self.layers
            .get_mut(index)
            .ok_or_else(|| {
                TinyError::ModelFormat(
                    format!(
                        "KV cache layer {index} does not exist"
                    ),
                )
            })
    }

    pub fn num_layers(
        &self,
    ) -> usize {
        self.layers.len()
    }

    pub fn max_seq_len(
        &self,
    ) -> usize {
        self.max_seq_len
    }
}

#[cfg(test)]
mod tests {
    use super::KvCache;

    #[test]
    fn creates_an_empty_cache_for_every_layer() {
        let mut cache = KvCache::new(3, 128);

        assert_eq!(cache.num_layers(), 3);
        assert_eq!(cache.max_seq_len(), 128);
        assert!(cache.layer_mut(0).unwrap().is_empty());
        assert!(cache.layer_mut(1).unwrap().is_empty());
        assert!(cache.layer_mut(2).unwrap().is_empty());
        assert!(cache.layer_mut(3).is_err());
    }
}
