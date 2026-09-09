use crate::{error::{Result, TinyError}, metal::MetalContext, ops::kv_cache_write_f32, tensor::Tensor};

pub struct LayerKvCache { key: Tensor, value: Tensor, len: usize, max_seq_len: usize, num_heads: usize, head_dim: usize }
impl LayerKvCache {
    pub fn new(context: &MetalContext, heads: usize, max: usize, dim: usize) -> Result<Self> { if heads == 0 || max == 0 || dim == 0 { return Err(TinyError::InvalidShape("KV cache dimensions must be positive".into())); } let shape = [1, heads, max, dim]; Ok(Self { key: Tensor::zeros(context, &shape)?, value: Tensor::zeros(context, &shape)?, len: 0, max_seq_len: max, num_heads: heads, head_dim: dim }) }
    pub fn len(&self) -> usize { self.len }
    pub fn is_empty(&self) -> bool { self.len == 0 }
    pub fn key(&self) -> Result<Tensor> { self.key.narrow(2, 0, self.len) }
    pub fn value(&self) -> Result<Tensor> { self.value.narrow(2, 0, self.len) }
    pub fn append(&mut self, context: &MetalContext, key: Tensor, value: Tensor) -> Result<()> { if key.shape().dims() != value.shape().dims() || key.rank() != 4 || key.dim(0)? != 1 || key.dim(1)? != self.num_heads || key.dim(3)? != self.head_dim { return Err(TinyError::ModelFormat("KV cache write shape does not match this layer".into())); } let seq = key.dim(2)?; if self.len.checked_add(seq).is_none_or(|end| end > self.max_seq_len) { return Err(TinyError::PositionOutOfRange { start_pos: self.len, seq_len: seq, max_seq_len: self.max_seq_len }); } kv_cache_write_f32(context, &self.key, &key, self.len)?; kv_cache_write_f32(context, &self.value, &value, self.len)?; self.len += seq; Ok(()) }
}
pub struct KvCache { layers: Vec<LayerKvCache>, max_seq_len: usize }
impl KvCache {
    pub fn new(context: &MetalContext, layers: usize, max: usize, heads: usize, dim: usize) -> Result<Self> { Ok(Self { layers: (0..layers).map(|_| LayerKvCache::new(context, heads, max, dim)).collect::<Result<Vec<_>>>()?, max_seq_len: max }) }
    pub fn layer_mut(&mut self, index: usize) -> Result<&mut LayerKvCache> { self.layers.get_mut(index).ok_or_else(|| TinyError::ModelFormat(format!("KV cache layer {index} does not exist"))) }
    pub fn num_layers(&self) -> usize { self.layers.len() }
    pub fn max_seq_len(&self) -> usize { self.max_seq_len }
}
