use crate::ops::{
    attention_scale_mask_f16, attention_scale_mask_f32, batched_matmul_f16, batched_matmul_f32,
    cast, materialize_contiguous_f16, materialize_contiguous_f32, matmul_f16, matmul_f32,
    softmax_f16, softmax_f32,
};
use std::sync::Arc;

use crate::error::{Result, TinyError};

use crate::metal::{MetalBuffer, MetalContext};

use super::{DType, Device, Shape, Storage, Strides};

#[derive(Clone)]
pub struct Tensor {
    storage: Arc<Storage>,
    shape: Shape,
    strides: Strides,
    dtype: DType,
}

impl Tensor {
    pub fn from_f32_slice(context: &MetalContext, data: &[f32], dims: &[usize]) -> Result<Self> {
        let shape = Shape::new(dims)?;

        if data.len() != shape.numel() {
            return Err(TinyError::InvalidShape(format!(
                "shape {:?} requires {} elements, but {} were provided",
                shape.dims(),
                shape.numel(),
                data.len(),
            )));
        }

        let strides = Strides::contiguous(&shape);

        let buffer = MetalBuffer::from_slice(context, data);

        Ok(Self {
            storage: Arc::new(Storage::Metal(buffer)),
            shape,
            strides,
            dtype: DType::F32,
        })
    }

    pub fn from_f16_slice(
        context: &MetalContext,
        data: &[half::f16],
        dims: &[usize],
    ) -> Result<Self> {
        let shape = Shape::new(dims)?;
        if data.len() != shape.numel() {
            return Err(TinyError::InvalidShape(format!(
                "shape {:?} requires {} elements, but {} were provided",
                shape.dims(),
                shape.numel(),
                data.len()
            )));
        }
        Ok(Self {
            storage: Arc::new(Storage::Metal(MetalBuffer::from_f16_slice(context, data))),
            shape: shape.clone(),
            strides: Strides::contiguous(&shape),
            dtype: DType::F16,
        })
    }

    pub fn contiguous(&self, context: &MetalContext) -> Result<Self> {
        if self.is_contiguous() {
            return Ok(self.clone());
        }

        let buffer = match self.dtype {
            DType::F32 => materialize_contiguous_f32(
                context,
                self.metal_buffer()?,
                self.shape.dims(),
                self.strides.values(),
            )?,
            DType::F16 => materialize_contiguous_f16(
                context,
                self.metal_buffer()?,
                self.shape.dims(),
                self.strides.values(),
            )?,
        };

        let shape = self.shape.clone();

        let strides = Strides::contiguous(&shape);

        Ok(Self {
            storage: Arc::new(Storage::Metal(buffer)),
            shape,
            strides,
            dtype: self.dtype,
        })
    }

    pub fn transpose(&self, dim_a: usize, dim_b: usize) -> Result<Self> {
        let rank = self.rank();

        if dim_a >= rank {
            return Err(TinyError::InvalidDimension(format!(
                "dimension {dim_a} does not exist for rank {rank}"
            )));
        }

        if dim_b >= rank {
            return Err(TinyError::InvalidDimension(format!(
                "dimension {dim_b} does not exist for rank {rank}"
            )));
        }

        let mut order: Vec<usize> = (0..rank).collect();

        order.swap(dim_a, dim_b);

        self.permute(&order)
    }

    pub fn permute(&self, order: &[usize]) -> Result<Self> {
        let rank = self.rank();

        if order.len() != rank {
            return Err(TinyError::InvalidDimension(format!(
                "permute order length {} does not match tensor rank {}",
                order.len(),
                rank,
            )));
        }

        let mut seen = vec![false; rank];

        for &dim in order {
            if dim >= rank {
                return Err(TinyError::InvalidDimension(format!(
                    "dimension {dim} does not exist for rank {rank}",
                )));
            }

            if seen[dim] {
                return Err(TinyError::InvalidDimension(format!(
                    "dimension {dim} appears more than once in permutation",
                )));
            }

            seen[dim] = true;
        }

        let old_dims = self.shape.dims();
        let old_strides = self.strides.values();

        let new_dims: Vec<usize> = order.iter().map(|&dim| old_dims[dim]).collect();

        let new_strides: Vec<usize> = order.iter().map(|&dim| old_strides[dim]).collect();

        Ok(Self {
            storage: Arc::clone(&self.storage),
            shape: Shape::new(&new_dims)?,
            strides: Strides::from_values(new_strides),
            dtype: self.dtype,
        })
    }

    pub fn zeros(context: &MetalContext, dims: &[usize]) -> Result<Self> {
        let shape = Shape::new(dims)?;

        let data = vec![0.0f32; shape.numel()];

        Self::from_f32_slice(context, &data, dims)
    }

    pub fn shape(&self) -> &Shape {
        &self.shape
    }

    pub fn strides(&self) -> &Strides {
        &self.strides
    }

    pub fn dtype(&self) -> DType {
        self.dtype
    }

    pub fn device(&self) -> Device {
        self.storage.device()
    }

    pub fn rank(&self) -> usize {
        self.shape.rank()
    }

    pub fn numel(&self) -> usize {
        self.shape.numel()
    }

    pub fn byte_len(&self) -> usize {
        self.numel() * self.dtype.size_in_bytes()
    }

    pub fn len(&self) -> usize {
        self.numel()
    }

    pub fn is_empty(&self) -> bool {
        self.numel() == 0
    }

    pub fn dim(&self, index: usize) -> Result<usize> {
        self.shape.dim(index)
    }

    pub fn is_contiguous(&self) -> bool {
        self.strides == Strides::contiguous(&self.shape)
    }

    pub fn reshape(&self, dims: &[usize]) -> Result<Self> {
        let new_shape = Shape::new(dims)?;

        if new_shape.numel() != self.numel() {
            return Err(TinyError::InvalidShape(format!(
                "cannot reshape {:?} into {:?}: element count differs",
                self.shape.dims(),
                new_shape.dims(),
            )));
        }

        if !self.is_contiguous() {
            return Err(TinyError::InvalidShape(
                "cannot reshape a non-contiguous tensor without copying".to_string(),
            ));
        }

        let new_strides = Strides::contiguous(&new_shape);

        Ok(Self {
            storage: Arc::clone(&self.storage),
            shape: new_shape,
            strides: new_strides,
            dtype: self.dtype,
        })
    }

    /// Returns a zero-copy view over a contiguous range of one dimension.
    pub fn narrow(&self, dim: usize, start: usize, len: usize) -> Result<Self> {
        if dim >= self.rank()
            || start
                .checked_add(len)
                .is_none_or(|end| end > self.shape.dims()[dim])
        {
            return Err(TinyError::InvalidDimension(
                "tensor narrow range is out of bounds".to_string(),
            ));
        }
        if start != 0 {
            return Err(TinyError::InvalidDimension(
                "tensor narrow currently supports ranges starting at zero".to_string(),
            ));
        }
        let mut dims = self.shape.dims().to_vec();
        dims[dim] = len;
        Ok(Self {
            storage: Arc::clone(&self.storage),
            shape: Shape::new(&dims)?,
            strides: self.strides.clone(),
            dtype: self.dtype,
        })
    }

    pub fn as_f32_slice(&self) -> Result<&[f32]> {
        if self.dtype != DType::F32 {
            return Err(TinyError::UnsupportedDType(format!("{:?}", self.dtype,)));
        }

        if !self.is_contiguous() {
            return Err(TinyError::NonContiguousTensor(format!(
                "shape={:?}, strides={:?}",
                self.shape.dims(),
                self.strides.values()
            )));
        }

        self.storage.as_f32_slice()
    }

    pub fn as_f16_slice(&self) -> Result<&[half::f16]> {
        if self.dtype != DType::F16 {
            return Err(TinyError::UnsupportedDType(format!("{:?}", self.dtype)));
        }
        if !self.is_contiguous() {
            return Err(TinyError::NonContiguousTensor(format!(
                "shape={:?}",
                self.shape.dims()
            )));
        }
        self.storage.as_f16_slice()
    }

    pub fn to_f32_vec(&self) -> Result<Vec<f32>> {
        match self.dtype {
            DType::F32 => Ok(self.as_f32_slice()?.to_vec()),
            DType::F16 => Ok(self
                .as_f16_slice()?
                .iter()
                .map(|value| value.to_f32())
                .collect()),
        }
    }

    /// Converts a contiguous tensor between the F32 and F16 storage formats.
    ///
    /// This is the bridge for selectively using F16 kernels; it does not make
    /// all tensor operations F16-capable.
    pub fn to_dtype(&self, context: &MetalContext, dtype: DType) -> Result<Self> {
        cast(context, self, dtype)
    }

    pub(crate) fn metal_buffer(&self) -> Result<&MetalBuffer> {
        self.storage.metal_buffer()
    }

    pub(crate) fn from_metal_buffer(
        buffer: MetalBuffer,
        dims: &[usize],
        dtype: DType,
    ) -> Result<Self> {
        let shape = Shape::new(dims)?;

        if buffer.len() != shape.numel()
            || buffer.byte_len() != shape.numel() * dtype.size_in_bytes()
        {
            return Err(TinyError::InvalidShape(format!(
                "buffer has {} elements, but shape {:?} requires {}",
                buffer.len(),
                shape.dims(),
                shape.numel(),
            )));
        }

        let strides = Strides::contiguous(&shape);

        Ok(Self {
            storage: Arc::new(Storage::Metal(buffer)),

            shape,
            strides,
            dtype,
        })
    }

    pub fn matmul(&self, context: &MetalContext, rhs: &Tensor) -> Result<Self> {
        if self.dtype != rhs.dtype {
            return Err(TinyError::UnsupportedDType(format!(
                "mixed matmul dtypes: {:?} and {:?}",
                self.dtype, rhs.dtype
            )));
        }

        if self.rank() != 2 {
            return Err(TinyError::InvalidDimension(format!(
                "matmul currently requires rank-2 tensors, left rank is {}",
                self.rank(),
            )));
        }

        if rhs.rank() != 2 {
            return Err(TinyError::InvalidDimension(format!(
                "matmul currently requires rank-2 tensors, right rank is {}",
                rhs.rank(),
            )));
        }

        let m = self.dim(0)?;
        let k = self.dim(1)?;
        let rhs_k = rhs.dim(0)?;
        let n = rhs.dim(1)?;

        if k != rhs_k {
            return Err(TinyError::ShapeMismatch {
                left: self.shape.dims().to_vec(),

                right: rhs.shape.dims().to_vec(),
            });
        }

        let lhs = if self.is_contiguous() {
            self.clone()
        } else {
            self.contiguous(context)?
        };

        let rhs = if rhs.is_contiguous() {
            rhs.clone()
        } else {
            rhs.contiguous(context)?
        };

        if self.dtype == DType::F16 && (!lhs.is_contiguous() || !rhs.is_contiguous()) {
            return Err(TinyError::NonContiguousTensor(
                "F16 matmul currently requires contiguous inputs".into(),
            ));
        }
        let buffer = match self.dtype {
            DType::F32 => matmul_f32(context, lhs.metal_buffer()?, rhs.metal_buffer()?, m, k, n)?,
            DType::F16 => matmul_f16(context, lhs.metal_buffer()?, rhs.metal_buffer()?, m, k, n)?,
        };

        let shape = Shape::new(&[m, n])?;

        let strides = Strides::contiguous(&shape);

        Ok(Self {
            storage: Arc::new(Storage::Metal(buffer)),
            shape,
            strides,
            dtype: self.dtype,
        })
    }

    pub fn softmax_last_dim(&self, context: &MetalContext) -> Result<Self> {
        if self.rank() == 0 {
            return Err(TinyError::InvalidDimension(
                "Softmax requires at least one dimension".to_string(),
            ));
        }

        let input = if self.is_contiguous() {
            self.clone()
        } else {
            self.contiguous(context)?
        };

        let width = input.dim(input.rank() - 1)?;

        let rows = input.numel() / width;

        let buffer = match self.dtype {
            DType::F32 => softmax_f32(context, input.metal_buffer()?, rows, width)?,
            DType::F16 => softmax_f16(context, input.metal_buffer()?, rows, width)?,
        };

        Self::from_metal_buffer(buffer, input.shape().dims(), self.dtype)
    }

    pub fn batched_matmul(&self, context: &MetalContext, rhs: &Tensor) -> Result<Self> {
        if self.dtype != rhs.dtype {
            return Err(TinyError::UnsupportedDType(format!(
                "mixed batched matmul dtypes: {:?} and {:?}",
                self.dtype, rhs.dtype
            )));
        }

        if self.rank() < 3 {
            return Err(TinyError::InvalidDimension(format!(
                "batched matmul requires rank >= 3, left rank is {}",
                self.rank(),
            )));
        }

        if rhs.rank() != self.rank() {
            return Err(TinyError::InvalidDimension(format!(
                "batched matmul requires tensors with equal rank, got {} and {}",
                self.rank(),
                rhs.rank(),
            )));
        }

        let rank = self.rank();

        let lhs_dims = self.shape.dims();

        let rhs_dims = rhs.shape.dims();

        // 마지막 두 차원을 제외한
        // 모든 batch dimension이 같아야 한다.
        if lhs_dims[..rank - 2] != rhs_dims[..rank - 2] {
            return Err(TinyError::ShapeMismatch {
                left: lhs_dims.to_vec(),

                right: rhs_dims.to_vec(),
            });
        }

        let m = lhs_dims[rank - 2];

        let k = lhs_dims[rank - 1];

        let rhs_k = rhs_dims[rank - 2];

        let n = rhs_dims[rank - 1];

        if k != rhs_k {
            return Err(TinyError::ShapeMismatch {
                left: lhs_dims.to_vec(),

                right: rhs_dims.to_vec(),
            });
        }

        let batch_count: usize = lhs_dims[..rank - 2].iter().product();

        let lhs = if self.is_contiguous() {
            self.clone()
        } else {
            self.contiguous(context)?
        };

        let rhs = if rhs.is_contiguous() {
            rhs.clone()
        } else {
            rhs.contiguous(context)?
        };

        let buffer = match self.dtype {
            DType::F32 => batched_matmul_f32(
                context,
                lhs.metal_buffer()?,
                rhs.metal_buffer()?,
                batch_count,
                m,
                k,
                n,
            )?,
            DType::F16 => batched_matmul_f16(
                context,
                lhs.metal_buffer()?,
                rhs.metal_buffer()?,
                batch_count,
                m,
                k,
                n,
            )?,
        };

        let mut output_dims = lhs_dims[..rank - 2].to_vec();

        output_dims.push(m);
        output_dims.push(n);

        Self::from_metal_buffer(buffer, &output_dims, self.dtype)
    }

    pub fn attention_scale_mask(
        &self,
        context: &MetalContext,
        scale: f32,
        query_start_pos: usize,
    ) -> Result<Self> {
        if self.rank() < 2 {
            return Err(TinyError::InvalidDimension(format!(
                "attention scores require rank >= 2, got rank {}",
                self.rank(),
            )));
        }

        let rank = self.rank();

        let query_len = self.dim(rank - 2)?;

        let key_len = self.dim(rank - 1)?;

        let input = if self.is_contiguous() {
            self.clone()
        } else {
            self.contiguous(context)?
        };

        let buffer = match self.dtype {
            DType::F32 => attention_scale_mask_f32(
                context,
                input.metal_buffer()?,
                scale,
                query_len,
                key_len,
                query_start_pos,
            )?,
            DType::F16 => attention_scale_mask_f16(
                context,
                input.metal_buffer()?,
                scale,
                query_len,
                key_len,
                query_start_pos,
            )?,
        };

        Self::from_metal_buffer(buffer, input.shape().dims(), self.dtype)
    }

    pub fn empty(context: &MetalContext, dims: &[usize], dtype: DType) -> Result<Self> {
        let shape = Shape::new(dims)?;

        let strides = Strides::contiguous(&shape);

        let buffer =
            MetalBuffer::empty_with_element_size(context, shape.numel(), dtype.size_in_bytes());

        Ok(Self {
            storage: Arc::new(Storage::Metal(buffer)),
            shape,
            strides,
            dtype,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::{DType, Tensor};
    use crate::{error::TinyError, metal::MetalContext};

    fn metal_context() -> Option<MetalContext> {
        match MetalContext::new() {
            Ok(context) => Some(context),
            Err(TinyError::Metal(message)) => {
                eprintln!("skipping Metal batched matmul test: {message}");
                None
            }
            Err(error) => panic!("failed to create Metal context: {error}"),
        }
    }

    fn assert_f16_close_to_f32(expected: &Tensor, actual: &Tensor) {
        assert_eq!(actual.dtype(), DType::F16);

        for (expected, actual) in expected
            .to_f32_vec()
            .unwrap()
            .iter()
            .zip(actual.to_f32_vec().unwrap().iter())
        {
            let error = (expected - actual).abs();
            assert!(
                error < 0.05,
                "F16 batched matmul mismatch: expected={expected}, actual={actual}, error={error}",
            );
        }
    }

    fn assert_scale_mask_close(expected: &Tensor, actual: &Tensor) {
        assert_eq!(actual.dtype(), DType::F16);

        for (expected, actual) in expected
            .to_f32_vec()
            .unwrap()
            .iter()
            .zip(actual.to_f32_vec().unwrap().iter())
        {
            if expected.is_infinite() {
                assert!(actual.is_infinite());
                assert_eq!(expected.is_sign_negative(), actual.is_sign_negative());
                continue;
            }

            let error = (expected - actual).abs();
            assert!(
                error < 0.01,
                "F16 scale/mask mismatch: expected={expected}, actual={actual}, error={error}",
            );
        }
    }

    fn assert_softmax_close(expected: &Tensor, actual: &Tensor) {
        assert_eq!(actual.dtype(), DType::F16);

        for (expected, actual) in expected
            .to_f32_vec()
            .unwrap()
            .iter()
            .zip(actual.to_f32_vec().unwrap().iter())
        {
            let error = (expected - actual).abs();
            assert!(
                error < 0.01,
                "F16 softmax mismatch: expected={expected}, actual={actual}, error={error}",
            );
        }
    }

    #[test]
    fn f16_batched_matmul_stays_close_to_f32_and_returns_f16() {
        let Some(context) = metal_context() else {
            return;
        };
        let a = Tensor::from_f32_slice(
            &context,
            &[1.0, 2.0, 3.0, 4.0, 1.0, 2.0, 3.0, 4.0],
            &[2, 2, 2],
        )
        .unwrap();
        let b = Tensor::from_f32_slice(
            &context,
            &[1.0, 0.0, 0.0, 1.0, 2.0, 2.0, 2.0, 2.0],
            &[2, 2, 2],
        )
        .unwrap();

        let expected = a.batched_matmul(&context, &b).unwrap();
        let actual = a
            .to_dtype(&context, DType::F16)
            .unwrap()
            .batched_matmul(&context, &b.to_dtype(&context, DType::F16).unwrap())
            .unwrap();

        assert_f16_close_to_f32(&expected, &actual);
    }

    #[test]
    fn f16_add_uses_f16_storage_and_matches_expected_values() {
        let Some(context) = metal_context() else {
            return;
        };
        let left = Tensor::from_f32_slice(&context, &[1.0, 2.0, 3.0, 4.0], &[2, 2])
            .unwrap()
            .to_dtype(&context, DType::F16)
            .unwrap();
        let right = Tensor::from_f32_slice(&context, &[0.5, 1.5, -1.0, 2.0], &[2, 2])
            .unwrap()
            .to_dtype(&context, DType::F16)
            .unwrap();

        let output = left.add(&context, &right).unwrap();
        assert_eq!(output.dtype(), DType::F16);
        assert_eq!(output.to_f32_vec().unwrap(), vec![1.5, 3.5, 2.0, 6.0]);
    }

    #[test]
    fn f16_batched_matmul_materializes_transposed_attention_keys() {
        let Some(context) = metal_context() else {
            return;
        };
        let q = Tensor::from_f32_slice(
            &context,
            &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
            &[1, 2, 2, 2],
        )
        .unwrap();
        let k = Tensor::from_f32_slice(
            &context,
            &[1.0, 0.0, 0.0, 1.0, 2.0, 1.0, 1.0, 2.0],
            &[1, 2, 2, 2],
        )
        .unwrap();

        let expected = q
            .batched_matmul(&context, &k.transpose(2, 3).unwrap())
            .unwrap();
        let q_f16 = q.to_dtype(&context, DType::F16).unwrap();
        let k_t_f16 = k
            .to_dtype(&context, DType::F16)
            .unwrap()
            .transpose(2, 3)
            .unwrap();
        assert!(!k_t_f16.is_contiguous());

        let actual = q_f16.batched_matmul(&context, &k_t_f16).unwrap();
        assert_f16_close_to_f32(&expected, &actual);
    }

    #[test]
    fn f16_attention_scale_mask_matches_f32_prefill() {
        let Some(context) = metal_context() else {
            return;
        };
        let scores = Tensor::from_f32_slice(
            &context,
            &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
            &[1, 1, 3, 3],
        )
        .unwrap();

        let expected = scores.attention_scale_mask(&context, 0.5, 0).unwrap();
        let actual = scores
            .to_dtype(&context, DType::F16)
            .unwrap()
            .attention_scale_mask(&context, 0.5, 0)
            .unwrap();

        assert_scale_mask_close(&expected, &actual);
    }

    #[test]
    fn f16_attention_scale_mask_keeps_all_keys_at_decode_position() {
        let Some(context) = metal_context() else {
            return;
        };
        let scores = Tensor::from_f32_slice(&context, &[1.0, 2.0, 3.0, 4.0], &[1, 1, 1, 4])
            .unwrap()
            .to_dtype(&context, DType::F16)
            .unwrap();

        let actual = scores.attention_scale_mask(&context, 0.5, 3).unwrap();
        assert_eq!(actual.dtype(), DType::F16);
        assert_eq!(actual.to_f32_vec().unwrap(), vec![0.5, 1.0, 1.5, 2.0]);
    }

    #[test]
    fn f16_attention_scale_mask_masks_future_key_at_intermediate_position() {
        let Some(context) = metal_context() else {
            return;
        };
        let scores = Tensor::from_f32_slice(&context, &[1.0, 2.0, 3.0, 4.0], &[1, 1, 1, 4])
            .unwrap()
            .to_dtype(&context, DType::F16)
            .unwrap();

        let actual = scores.attention_scale_mask(&context, 0.5, 2).unwrap();
        let values = actual.to_f32_vec().unwrap();
        assert_eq!(&values[..3], &[0.5, 1.0, 1.5]);
        assert!(values[3].is_infinite() && values[3].is_sign_negative());
    }

    #[test]
    fn f16_softmax_last_dim_matches_f32_and_normalizes_each_row() {
        let Some(context) = metal_context() else {
            return;
        };
        let input =
            Tensor::from_f32_slice(&context, &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]).unwrap();

        let expected = input.softmax_last_dim(&context).unwrap();
        let actual = input
            .to_dtype(&context, DType::F16)
            .unwrap()
            .softmax_last_dim(&context)
            .unwrap();

        assert_softmax_close(&expected, &actual);
        for row in actual.to_f32_vec().unwrap().chunks(3) {
            let sum = row.iter().sum::<f32>();
            assert!((sum - 1.0).abs() < 0.01, "softmax row sum = {sum}");
        }
    }

    #[test]
    fn f16_attention_mask_and_softmax_match_f32_and_zero_masked_probabilities() {
        let Some(context) = metal_context() else {
            return;
        };
        let scores = Tensor::from_f32_slice(
            &context,
            &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
            &[1, 1, 3, 3],
        )
        .unwrap();

        let expected = scores
            .attention_scale_mask(&context, 0.5, 0)
            .unwrap()
            .softmax_last_dim(&context)
            .unwrap();
        let actual = scores
            .to_dtype(&context, DType::F16)
            .unwrap()
            .attention_scale_mask(&context, 0.5, 0)
            .unwrap()
            .softmax_last_dim(&context)
            .unwrap();

        assert_softmax_close(&expected, &actual);
        let values = actual.to_f32_vec().unwrap();
        assert_eq!(values[1], 0.0);
        assert_eq!(values[2], 0.0);
        assert_eq!(values[5], 0.0);
    }
}
