use crate::{
    error::{Result, TinyError},
    metal::{MetalContext, MetalExecution},
    ops::{kv_cache_write_f16_encode, kv_cache_write_f32_encode},
    tensor::{DType, Tensor},
};
use metal::CommandBufferRef;

pub struct LayerKvCache {
    key: Tensor,
    value: Tensor,
    len: usize,
    max_seq_len: usize,
    num_kv_heads: usize,
    head_dim: usize,
}
impl LayerKvCache {
    fn validate_append(&self, key: &Tensor, value: &Tensor) -> Result<()> {
        if key.dtype() != self.key.dtype() || value.dtype() != self.value.dtype() {
            return Err(TinyError::UnsupportedDType(format!(
                "KV cache dtype mismatch: cache={:?}/{:?}, input={:?}/{:?}",
                self.key.dtype(),
                self.value.dtype(),
                key.dtype(),
                value.dtype(),
            )));
        }
        if key.shape().dims() != value.shape().dims()
            || key.rank() != 4
            || key.dim(0)? != 1
            || key.dim(1)? != self.num_kv_heads
            || key.dim(3)? != self.head_dim
        {
            return Err(TinyError::ModelFormat(format!(
                "KV cache write shape does not match this layer: cache has {} KV heads, input has {} heads",
                self.num_kv_heads, key.dim(1)?,
            )));
        }
        let seq = key.dim(2)?;
        if self
            .len
            .checked_add(seq)
            .is_none_or(|end| end > self.max_seq_len)
        {
            return Err(TinyError::PositionOutOfRange {
                start_pos: self.len,
                seq_len: seq,
                max_seq_len: self.max_seq_len,
            });
        }
        Ok(())
    }
    pub fn new(
        context: &MetalContext,
        num_kv_heads: usize,
        max: usize,
        dim: usize,
        dtype: DType,
    ) -> Result<Self> {
        if num_kv_heads == 0 || max == 0 || dim == 0 {
            return Err(TinyError::InvalidShape(
                "KV cache dimensions must be positive".into(),
            ));
        }
        let shape = [1, num_kv_heads, max, dim];
        Ok(Self {
            key: Tensor::empty(context, &shape, dtype)?,
            value: Tensor::empty(context, &shape, dtype)?,
            len: 0,
            max_seq_len: max,
            num_kv_heads,
            head_dim: dim,
        })
    }
    pub fn len(&self) -> usize {
        self.len
    }
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }
    /// Allocated sequence capacity. This does not change as tokens are appended.
    pub fn capacity(&self) -> usize {
        self.max_seq_len
    }
    /// Returns the bytes reserved for both key and value storage.
    pub fn storage_bytes(&self) -> usize {
        2 * self.num_kv_heads * self.max_seq_len * self.head_dim * self.dtype().size_in_bytes()
    }
    /// Starts a new sequence without clearing GPU memory; future writes overwrite
    /// the logical prefix before it can be read.
    pub fn reset(&mut self) {
        self.len = 0;
    }
    pub fn dtype(&self) -> DType {
        self.key.dtype()
    }
    pub fn key(&self) -> Result<Tensor> {
        self.key.narrow(2, 0, self.len)
    }
    pub fn value(&self) -> Result<Tensor> {
        self.value.narrow(2, 0, self.len)
    }
    pub fn append(&mut self, context: &MetalContext, key: Tensor, value: Tensor) -> Result<()> {
        let execution = MetalExecution::new(context);

        self.append_encode(context, execution.command_buffer(), key, value)?;
        execution.finish();
        Ok(())
    }

    pub(crate) fn append_encode(
        &mut self,
        context: &MetalContext,
        command_buffer: &CommandBufferRef,
        key: Tensor,
        value: Tensor,
    ) -> Result<()> {
        self.validate_append(&key, &value)?;
        let seq_len = key.dim(2)?;
        let start_pos = self.len;
        let new_len = start_pos.checked_add(seq_len).ok_or_else(|| {
            TinyError::InvalidShape("KV cache length overflow".into())
        })?;

        let key = key.contiguous_encode(context, command_buffer)?;
        let value = value.contiguous_encode(context, command_buffer)?;

        match self.dtype() {
            DType::F32 => {
                kv_cache_write_f32_encode(context, command_buffer, &self.key, &key, start_pos)?;
                kv_cache_write_f32_encode(context, command_buffer, &self.value, &value, start_pos)?;
            }
            DType::F16 => {
                kv_cache_write_f16_encode(context, command_buffer, &self.key, &key, start_pos)?;
                kv_cache_write_f16_encode(context, command_buffer, &self.value, &value, start_pos)?;
            }
        }

        self.len = new_len;
        Ok(())
    }
}

pub struct KvCache {
    layers: Vec<LayerKvCache>,
    max_seq_len: usize,
}
impl KvCache {
    pub fn new(
        context: &MetalContext,
        layers: usize,
        max: usize,
        num_kv_heads: usize,
        dim: usize,
        dtype: DType,
    ) -> Result<Self> {
        Ok(Self {
            layers: (0..layers)
                .map(|_| LayerKvCache::new(context, num_kv_heads, max, dim, dtype))
                .collect::<Result<Vec<_>>>()?,
            max_seq_len: max,
        })
    }
    pub fn layer_mut(&mut self, index: usize) -> Result<&mut LayerKvCache> {
        self.layers
            .get_mut(index)
            .ok_or_else(|| TinyError::ModelFormat(format!("KV cache layer {index} does not exist")))
    }
    pub fn num_layers(&self) -> usize {
        self.layers.len()
    }
    pub fn max_seq_len(&self) -> usize {
        self.max_seq_len
    }
    pub fn storage_bytes(&self) -> usize {
        self.layers.iter().map(LayerKvCache::storage_bytes).sum()
    }
    pub fn reset(&mut self) {
        for layer in &mut self.layers {
            layer.reset();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::LayerKvCache;
    use crate::{
        error::TinyError,
        metal::MetalContext,
        tensor::{DType, Tensor},
    };

    fn metal_context() -> Option<MetalContext> {
        match MetalContext::new() {
            Ok(context) => Some(context),
            Err(TinyError::Metal(message)) => {
                eprintln!("skipping Metal KV cache test: {message}");
                None
            }
            Err(error) => panic!("failed to create Metal context: {error}"),
        }
    }

    fn f16_tensor(context: &MetalContext, data: &[f32], shape: &[usize]) -> Tensor {
        Tensor::from_f32_slice(context, data, shape)
            .unwrap()
            .to_dtype(context, DType::F16)
            .unwrap()
    }

    #[test]
    fn f16_cache_appends_decode_tokens_and_returns_f16_prefix() {
        let Some(context) = metal_context() else {
            return;
        };
        let mut cache = LayerKvCache::new(&context, 2, 8, 4, DType::F16).unwrap();
        assert_eq!(cache.dtype(), DType::F16);
        assert_eq!(cache.len(), 0);

        let key = f16_tensor(&context, &[1., 2., 3., 4., 5., 6., 7., 8.], &[1, 2, 1, 4]);
        let value = f16_tensor(
            &context,
            &[10., 20., 30., 40., 50., 60., 70., 80.],
            &[1, 2, 1, 4],
        );
        cache.append(&context, key, value).unwrap();

        let key = f16_tensor(
            &context,
            &[9., 10., 11., 12., 13., 14., 15., 16.],
            &[1, 2, 1, 4],
        );
        let value = f16_tensor(
            &context,
            &[90., 100., 110., 120., 130., 140., 150., 160.],
            &[1, 2, 1, 4],
        );
        cache.append(&context, key, value).unwrap();

        assert_eq!(cache.len(), 2);
        let cached_key = cache.key().unwrap();
        let cached_value = cache.value().unwrap();
        assert_eq!(cached_key.shape().dims(), &[1, 2, 2, 4]);
        assert_eq!(cached_key.dtype(), DType::F16);
        assert_eq!(cached_value.dtype(), DType::F16);
        assert_eq!(
            cached_key
                .contiguous(&context)
                .unwrap()
                .to_f32_vec()
                .unwrap(),
            vec![1., 2., 3., 4., 9., 10., 11., 12., 5., 6., 7., 8., 13., 14., 15., 16.,],
        );
    }

    #[test]
    fn f16_cache_appends_prefill_sequence() {
        let Some(context) = metal_context() else {
            return;
        };
        let mut cache = LayerKvCache::new(&context, 2, 8, 4, DType::F16).unwrap();
        let key = f16_tensor(&context, &[1.; 24], &[1, 2, 3, 4]);
        let value = f16_tensor(&context, &[2.; 24], &[1, 2, 3, 4]);

        cache.append(&context, key, value).unwrap();

        assert_eq!(cache.len(), 3);
        assert_eq!(cache.key().unwrap().shape().dims(), &[1, 2, 3, 4]);
    }

    #[test]
    fn append_encode_updates_the_logical_prefix_after_submission() {
        let Some(context) = metal_context() else {
            return;
        };
        let mut cache = LayerKvCache::new(&context, 2, 8, 2, DType::F16).unwrap();
        let key = f16_tensor(&context, &[1.; 4], &[1, 2, 1, 2]);
        let value = f16_tensor(&context, &[2.; 4], &[1, 2, 1, 2]);
        let execution = crate::metal::MetalExecution::new(&context);

        cache
            .append_encode(&context, execution.command_buffer(), key, value)
            .unwrap();
        execution.finish();

        assert_eq!(cache.len(), 1);
        assert_eq!(cache.key().unwrap().shape().dims(), &[1, 2, 1, 2]);
        assert_eq!(cache.value().unwrap().shape().dims(), &[1, 2, 1, 2]);
    }

    #[test]
    fn f16_cache_rejects_f32_inputs_without_casting() {
        let Some(context) = metal_context() else {
            return;
        };
        let mut cache = LayerKvCache::new(&context, 2, 8, 4, DType::F16).unwrap();
        let key = Tensor::from_f32_slice(&context, &[1.; 8], &[1, 2, 1, 4]).unwrap();
        let value = Tensor::from_f32_slice(&context, &[2.; 8], &[1, 2, 1, 4]).unwrap();

        assert!(cache.append(&context, key, value).is_err());
        assert_eq!(cache.len(), 0);
    }
}
