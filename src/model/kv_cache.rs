use crate::{
    error::{Result, TinyError},
    metal::MetalContext,
    ops::{kv_cache_write_f16, kv_cache_write_f32},
    tensor::{DType, Tensor},
};

pub struct LayerKvCache {
    key: Tensor,
    value: Tensor,
    len: usize,
    max_seq_len: usize,
    num_heads: usize,
    head_dim: usize,
}
impl LayerKvCache {
    pub fn new(
        context: &MetalContext,
        heads: usize,
        max: usize,
        dim: usize,
        dtype: DType,
    ) -> Result<Self> {
        if heads == 0 || max == 0 || dim == 0 {
            return Err(TinyError::InvalidShape(
                "KV cache dimensions must be positive".into(),
            ));
        }
        let shape = [1, heads, max, dim];
        Ok(Self {
            key: Tensor::empty(context, &shape, dtype)?,
            value: Tensor::empty(context, &shape, dtype)?,
            len: 0,
            max_seq_len: max,
            num_heads: heads,
            head_dim: dim,
        })
    }
    pub fn len(&self) -> usize {
        self.len
    }
    pub fn is_empty(&self) -> bool {
        self.len == 0
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
            || key.dim(1)? != self.num_heads
            || key.dim(3)? != self.head_dim
        {
            return Err(TinyError::ModelFormat(
                "KV cache write shape does not match this layer".into(),
            ));
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
        append_tensor(context, &self.key, &key, self.len)?;
        append_tensor(context, &self.value, &value, self.len)?;
        self.len += seq;
        Ok(())
    }
}

fn append_tensor(
    context: &MetalContext,
    cache: &Tensor,
    source: &Tensor,
    start: usize,
) -> Result<()> {
    match source.dtype() {
        DType::F32 => kv_cache_write_f32(context, cache, source, start),
        DType::F16 => kv_cache_write_f16(context, cache, source, start),
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
        heads: usize,
        dim: usize,
        dtype: DType,
    ) -> Result<Self> {
        Ok(Self {
            layers: (0..layers)
                .map(|_| LayerKvCache::new(context, heads, max, dim, dtype))
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
            vec![
                1., 2., 3., 4., 9., 10., 11., 12., 5., 6., 7., 8., 13., 14., 15., 16.,
            ],
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
