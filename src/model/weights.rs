use crate::error::{Result, TinyError};
use std::{collections::HashMap, path::Path};

pub(super) const MAGIC: &[u8; 8] = b"TMLLWGHT";
pub(super) const VERSION: u32 = 1;
pub(super) const DTYPE_F32: u8 = 1;
pub(super) const MAX_RANK: u32 = 16;
pub(super) const MAX_NAME_LEN: u32 = 1024;

#[derive(Debug, Clone)]
pub struct WeightTensor {
    pub(super) shape: Vec<usize>,
    pub(super) data: Vec<f32>,
}
impl WeightTensor {
    pub fn shape(&self) -> &[usize] {
        &self.shape
    }
    pub fn data(&self) -> &[f32] {
        &self.data
    }
    pub fn into_parts(self) -> (Vec<usize>, Vec<f32>) {
        (self.shape, self.data)
    }
}

#[derive(Debug, Default)]
pub struct ModelWeights {
    pub(super) tensors: HashMap<String, WeightTensor>,
}
impl ModelWeights {
    pub fn new() -> Self {
        Self::default()
    }
    pub fn len(&self) -> usize {
        self.tensors.len()
    }
    pub fn is_empty(&self) -> bool {
        self.tensors.is_empty()
    }
    pub fn contains(&self, name: &str) -> bool {
        self.tensors.contains_key(name)
    }
    pub fn get(&self, name: &str) -> Result<&WeightTensor> {
        self.tensors
            .get(name)
            .ok_or_else(|| TinyError::MissingWeight(name.to_string()))
    }
    pub fn take(&mut self, name: &str) -> Result<WeightTensor> {
        self.tensors
            .remove(name)
            .ok_or_else(|| TinyError::MissingWeight(name.to_string()))
    }
    pub fn insert_f32(
        &mut self,
        name: impl Into<String>,
        shape: &[usize],
        data: Vec<f32>,
    ) -> Result<()> {
        let name = name.into();
        if name.is_empty() || shape.is_empty() || shape.contains(&0) {
            return Err(TinyError::ModelFormat(format!(
                "invalid shape or name for weight {name}"
            )));
        }
        let numel = checked_numel(shape)?;
        if data.len() != numel {
            return Err(TinyError::ModelFormat(format!(
                "weight {name} has {} values, but shape {shape:?} requires {numel}",
                data.len()
            )));
        }
        if self.tensors.contains_key(&name) {
            return Err(TinyError::ModelFormat(format!("duplicate weight: {name}")));
        }
        self.tensors.insert(
            name,
            WeightTensor {
                shape: shape.to_vec(),
                data,
            },
        );
        Ok(())
    }
    pub fn save(&self, path: impl AsRef<Path>) -> Result<()> {
        super::weight_codec::save(self, path)
    }
    pub fn load(path: impl AsRef<Path>) -> Result<Self> {
        super::weight_codec::load(path)
    }
}
pub(super) fn checked_numel(shape: &[usize]) -> Result<usize> {
    shape.iter().try_fold(1usize, |n, &d| {
        n.checked_mul(d).ok_or_else(|| {
            TinyError::ModelFormat(format!("shape element count overflow: {shape:?}"))
        })
    })
}

#[cfg(test)]
mod tests {
    use super::ModelWeights;

    #[test]
    fn saves_and_loads_a_weight_collection() {
        let mut weights = ModelWeights::new();
        weights
            .insert_f32("layer.weight", &[2, 2], vec![1.0, 2.0, 3.0, 4.0])
            .unwrap();

        let path = std::env::temp_dir()
            .join(format!("tiny-metal-llm-weights-{}.bin", std::process::id(),));
        weights.save(&path).unwrap();
        let loaded = ModelWeights::load(&path).unwrap();
        std::fs::remove_file(&path).unwrap();

        assert_eq!(loaded.get("layer.weight").unwrap().shape(), &[2, 2]);
        assert_eq!(
            loaded.get("layer.weight").unwrap().data(),
            &[1.0, 2.0, 3.0, 4.0]
        );
    }
}
