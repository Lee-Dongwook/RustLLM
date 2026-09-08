use crate::error::{
    Result,
    TinyError,
};

use crate::metal::{
    MetalBuffer,
    MetalContext,
};

use super::{
    DType,
    Device,
    Shape,
    Storage,
};

pub struct Tensor {
    storage: Storage,
    shape: Shape,
    dtype: DType,
}

impl Tensor {
    pub fn from_f32_slice(
        context: &MetalContext,
        data: &[f32],
        dims: &[usize],
    ) -> Result<Self> {
        let shape =
            Shape::new(dims)?;

        if data.len() != shape.numel() {
            return Err(
                TinyError::InvalidShape(
                    format!(
                        "shape {:?} requires {} elements, but {} were provided",
                        shape.dims(),
                        shape.numel(),
                        data.len(),
                    ),
                ),
            );
        }

        let buffer =
            MetalBuffer::from_slice(
                context,
                data,
            );

        Ok(Self {
            storage: Storage::Metal(buffer),
            shape,
            dtype: DType::F32,
        })
    }

    pub fn shape(
        &self,
    ) -> &Shape {
        &self.shape
    }

    pub fn dtype(
        &self,
    ) -> DType {
        self.dtype
    }

    pub fn device(
        &self,
    ) -> Device {
        self.storage.device()
    }

    pub fn len(
        &self,
    ) -> usize {
        self.storage.len()
    }

    pub fn as_f32_slice(
        &self,
    ) -> Result<&[f32]> {
        if self.dtype != DType::F32 {
            return Err(
                TinyError::UnsupportedDType(
                    format!("{:?}", self.dtype),
                ),
            );
        }

        self.storage.as_f32_slice()
    }

    pub(crate) fn metal_buffer(
        &self,
    ) -> Result<&MetalBuffer> {
        self.storage.metal_buffer()
    }
}
