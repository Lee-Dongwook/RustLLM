use crate::{
    error::{Result, TinyError},
    metal::MetalContext,
    ops::{add_f16, add_f32, mul_f16, mul_f32, silu_f16, silu_f32},
};

use super::{DType, Tensor};

impl Tensor {
    pub fn add(&self, context: &MetalContext, rhs: &Tensor) -> Result<Self> {
        if self.dtype() != rhs.dtype() {
            return Err(TinyError::UnsupportedDType(format!(
                "add dtype mismatch: {:?} vs {:?}",
                self.dtype(),
                rhs.dtype(),
            )));
        }

        match self.dtype() {
            DType::F32 => self.binary_elementwise(context, rhs, add_f32, DType::F32),
            DType::F16 => self.binary_elementwise(context, rhs, add_f16, DType::F16),
        }
    }

    pub fn mul(&self, context: &MetalContext, rhs: &Tensor) -> Result<Self> {
        if self.dtype() != rhs.dtype() {
            return Err(TinyError::UnsupportedDType(format!(
                "mul dtype mismatch: {:?} vs {:?}",
                self.dtype(),
                rhs.dtype(),
            )));
        }

        match self.dtype() {
            DType::F32 => self.binary_elementwise(context, rhs, mul_f32, DType::F32),
            DType::F16 => self.binary_elementwise(context, rhs, mul_f16, DType::F16),
        }
    }

    pub fn silu(&self, context: &MetalContext) -> Result<Self> {
        let input = self.contiguous(context)?;

        match self.dtype() {
            DType::F32 => Self::from_metal_buffer(
                silu_f32(context, input.metal_buffer()?)?,
                self.shape().dims(),
                DType::F32,
            ),
            DType::F16 => Self::from_metal_buffer(
                silu_f16(context, input.metal_buffer()?)?,
                self.shape().dims(),
                DType::F16,
            ),
        }
    }

    fn binary_elementwise(
        &self,
        context: &MetalContext,
        rhs: &Tensor,
        operation: fn(
            &MetalContext,
            &crate::metal::MetalBuffer,
            &crate::metal::MetalBuffer,
        ) -> Result<crate::metal::MetalBuffer>,
        dtype: DType,
    ) -> Result<Self> {
        if self.shape() != rhs.shape() {
            return Err(TinyError::ShapeMismatch {
                left: self.shape().dims().to_vec(),
                right: rhs.shape().dims().to_vec(),
            });
        }
        let lhs = self.contiguous(context)?;
        let rhs = rhs.contiguous(context)?;
        Self::from_metal_buffer(
            operation(context, lhs.metal_buffer()?, rhs.metal_buffer()?)?,
            self.shape().dims(),
            dtype,
        )
    }
}
