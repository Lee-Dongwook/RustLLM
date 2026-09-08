use std::sync::Arc;

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
    Strides,
};

pub struct Tensor {
    storage: Arc<Storage>,
    shape: Shape,
    strides: Strides,
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

        let strides =
            Strides::contiguous(&shape);

        let buffer =
            MetalBuffer::from_slice(
                context,
                data,
            );

        Ok(Self {
            storage: Arc::new(
                Storage::Metal(buffer),
            ),
            shape,
            strides,
            dtype: DType::F32,
        })
    }

    pub fn zeros(
        context: &MetalContext,
        dims: &[usize],
    ) -> Result<Self> {
        let shape =
            Shape::new(dims)?;

        let data =
            vec![0.0f32; shape.numel()];

        Self::from_f32_slice(
            context,
            &data,
            dims,
        )
    }

    pub fn shape(
        &self,
    ) -> &Shape {
        &self.shape
    }

    pub fn strides(
        &self,
    ) -> &Strides {
        &self.strides
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

    pub fn rank(
        &self,
    ) -> usize {
        self.shape.rank()
    }

    pub fn numel(
        &self,
    ) -> usize {
        self.shape.numel()
    }

    pub fn len(
        &self,
    ) -> usize {
        self.numel()
    }

    pub fn dim(
        &self,
        index: usize,
    ) -> Result<usize> {
        self.shape.dim(index)
    }

    pub fn is_contiguous(
        &self,
    ) -> bool {
        self.strides
            == Strides::contiguous(
                &self.shape,
            )
    }

    pub fn reshape(
        &self,
        dims: &[usize],
    ) -> Result<Self> {
        let new_shape =
            Shape::new(dims)?;

        if new_shape.numel()
            != self.numel()
        {
            return Err(
                TinyError::InvalidShape(
                    format!(
                        "cannot reshape {:?} into {:?}: element count differs",
                        self.shape.dims(),
                        new_shape.dims(),
                    ),
                ),
            );
        }

        if !self.is_contiguous() {
            return Err(
                TinyError::InvalidShape(
                    "cannot reshape a non-contiguous tensor without copying"
                        .to_string(),
                ),
            );
        }

        let new_strides =
            Strides::contiguous(
                &new_shape,
            );

        Ok(Self {
            storage: Arc::clone(
                &self.storage,
            ),
            shape: new_shape,
            strides: new_strides,
            dtype: self.dtype,
        })
    }

    pub fn as_f32_slice(
        &self,
    ) -> Result<&[f32]> {
        if self.dtype != DType::F32 {
            return Err(
                TinyError::UnsupportedDType(
                    format!(
                        "{:?}",
                        self.dtype,
                    ),
                ),
            );
        }

        self.storage
            .as_f32_slice()
    }

    pub(crate) fn metal_buffer(
        &self,
    ) -> Result<&MetalBuffer> {
        self.storage
            .metal_buffer()
    }
}
