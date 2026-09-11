use crate::{
    error::{Result, TinyError},
    tensor::DType,
};
use std::{collections::HashMap, path::Path};

pub(super) const MAGIC: &[u8; 8] = b"TMLLWGHT";
pub(super) const VERSION: u32 = 1;
pub(super) const DTYPE_F32: u8 = 1;
pub(super) const DTYPE_F16: u8 = 2;
pub(super) const DTYPE_I8: u8 = 3;
pub(super) const MAX_RANK: u32 = 16;
pub(super) const MAX_NAME_LEN: u32 = 1024;

#[derive(Debug, Clone)]
pub struct WeightTensor {
    pub(super) shape: Vec<usize>,
    pub(super) data: WeightData,
}

#[derive(Debug, Clone)]
pub(super) enum WeightData {
    F32(Vec<f32>),
    F16(Vec<half::f16>),
    I8(Vec<i8>),
}
impl WeightTensor {
    pub fn shape(&self) -> &[usize] {
        &self.shape
    }
    pub fn data(&self) -> &[f32] {
        match &self.data {
            WeightData::F32(data) => data,
            WeightData::F16(_) => panic!("F16 weight does not expose F32 data"),
            WeightData::I8(_) => panic!("INT8 weight does not expose F32 data"),
        }
    }
    pub fn into_parts(self) -> (Vec<usize>, Vec<f32>) {
        match self.data {
            WeightData::F32(data) => (self.shape, data),
            WeightData::F16(_) => panic!("F16 weight cannot be converted to F32 parts"),
            WeightData::I8(_) => panic!("INT8 weight cannot be converted to F32 parts"),
        }
    }
    pub(super) fn into_storage_parts(self) -> (Vec<usize>, WeightData) {
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
    pub fn contains_i8(&self) -> bool {
        self.tensors
            .values()
            .any(|tensor| matches!(&tensor.data, WeightData::I8(_)))
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
                data: WeightData::F32(data),
            },
        );
        Ok(())
    }

    pub fn insert_f16(
        &mut self,
        name: impl Into<String>,
        shape: &[usize],
        data: Vec<half::f16>,
    ) -> Result<()> {
        let name = name.into();

        if name.is_empty() || shape.is_empty() || shape.contains(&0) {
            return Err(TinyError::ModelFormat(format!(
                "invalid shape or name for F16 weight {name}"
            )));
        }

        let numel = checked_numel(shape)?;

        if data.len() != numel {
            return Err(TinyError::ModelFormat(format!(
                "weight {name} has {} values, but shape {shape:?} requires {numel}",
                data.len(),
            )));
        }

        if self.tensors.contains_key(&name) {
            return Err(TinyError::ModelFormat(format!("duplicate weight: {name}")));
        }

        self.tensors.insert(
            name,
            WeightTensor {
                shape: shape.to_vec(),
                data: WeightData::F16(data),
            },
        );
        Ok(())
    }

    pub fn insert_i8(
        &mut self,
        name: impl Into<String>,
        shape: &[usize],
        data: Vec<i8>,
    ) -> Result<()> {
        let name = name.into();
        if name.is_empty()
            || shape.is_empty()
            || shape.contains(&0)
            || data.len() != checked_numel(shape)?
            || self.tensors.contains_key(&name)
        {
            return Err(TinyError::ModelFormat(format!(
                "invalid or duplicate INT8 weight: {name}"
            )));
        }
        self.tensors.insert(
            name,
            WeightTensor {
                shape: shape.to_vec(),
                data: WeightData::I8(data),
            },
        );
        Ok(())
    }
    pub fn save(&self, path: impl AsRef<Path>) -> Result<()> {
        super::weight_codec::save(self, path)
    }
    pub fn save_as(&self, path: impl AsRef<Path>, dtype: DType) -> Result<()> {
        super::weight_codec::save_as(self, path, dtype)
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
    use super::{ModelWeights, WeightData};
    use crate::tensor::DType;

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

    #[test]
    fn saves_and_loads_f16_weight_collection() {
        let mut weights = ModelWeights::new();
        weights
            .insert_f32("layer.weight", &[2, 2], vec![1.0, -2.0, 0.5, 3.25])
            .unwrap();

        let path = std::env::temp_dir().join(format!(
            "tiny-metal-llm-weights-f16-{}.bin",
            std::process::id()
        ));
        weights.save_as(&path, DType::F16).unwrap();
        let loaded = ModelWeights::load(&path).unwrap();
        std::fs::remove_file(&path).unwrap();

        assert_eq!(loaded.get("layer.weight").unwrap().shape(), &[2, 2]);
        match &loaded.get("layer.weight").unwrap().data {
            WeightData::F16(data) => assert_eq!(
                data.iter().map(|value| value.to_f32()).collect::<Vec<_>>(),
                vec![1.0, -2.0, 0.5, 3.25]
            ),
            WeightData::F32(_) => panic!("expected F16 weights"),
            WeightData::I8(_) => panic!("expected F16 weights"),
        }
    }
}

#[test]
fn saves_and_loads_mixed_dtype_weights() {
    let mut weights = ModelWeights::new();

    weights
        .insert_f16(
            "embedding.weight",
            &[2],
            vec![half::f16::from_f32(1.0), half::f16::from_f32(2.0)],
        )
        .unwrap();

    weights
        .insert_i8("linear.weight", &[2, 2], vec![127, -127, 64, -64])
        .unwrap();

    weights
        .insert_f32("linear.scale", &[2], vec![0.01, 0.02])
        .unwrap();

    let path =
        std::env::temp_dir().join(format!("tiny-metal-llm-mixed-{}.bin", std::process::id(),));

    weights.save(&path).unwrap();

    let loaded = ModelWeights::load(&path).unwrap();

    std::fs::remove_file(&path).unwrap();

    assert!(matches!(
        &loaded.get("embedding.weight").unwrap().data,
        WeightData::F16(_),
    ));

    assert!(matches!(
        &loaded.get("linear.weight").unwrap().data,
        WeightData::I8(_),
    ));

    assert!(matches!(
        &loaded.get("linear.scale").unwrap().data,
        WeightData::F32(_),
    ));
}
