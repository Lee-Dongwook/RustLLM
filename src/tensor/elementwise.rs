use crate::{
    error::{Result, TinyError},
    metal::MetalContext,
    ops::{add_f32, mul_f32, silu_f32},
};

use super::{DType, Tensor};

impl Tensor {
    pub fn add(&self, context: &MetalContext, rhs: &Tensor) -> Result<Self> {
        self.binary_elementwise(context, rhs, add_f32, "add")
    }

    pub fn mul(&self, context: &MetalContext, rhs: &Tensor) -> Result<Self> {
        self.binary_elementwise(context, rhs, mul_f32, "mul")
    }

    pub fn silu(&self, context: &MetalContext) -> Result<Self> {
        self.require_f32("SiLU")?;
        let input = self.contiguous(context)?;
        Self::from_metal_buffer(silu_f32(context, input.metal_buffer()?)?, self.shape().dims(), DType::F32)
    }

    fn binary_elementwise(
        &self,
        context: &MetalContext,
        rhs: &Tensor,
        operation: fn(&MetalContext, &crate::metal::MetalBuffer, &crate::metal::MetalBuffer) -> Result<crate::metal::MetalBuffer>,
        name: &str,
    ) -> Result<Self> {
        self.require_f32(name)?;
        rhs.require_f32(name)?;
        if self.shape() != rhs.shape() {
            return Err(TinyError::ShapeMismatch { left: self.shape().dims().to_vec(), right: rhs.shape().dims().to_vec() });
        }
        let lhs = self.contiguous(context)?;
        let rhs = rhs.contiguous(context)?;
        Self::from_metal_buffer(operation(context, lhs.metal_buffer()?, rhs.metal_buffer()?)?, self.shape().dims(), DType::F32)
    }

    pub(super) fn require_f32(&self, operation: &str) -> Result<()> {
        if self.dtype() != DType::F32 {
            return Err(TinyError::UnsupportedDType(format!("{operation} supports only F32, got {:?}", self.dtype())));
        }
        Ok(())
    }
}
