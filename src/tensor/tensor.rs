use std::sync::Arc;
use crate::ops::{
    matmul_f32,
    materialize_contiguous_f32,
};

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

#[derive(Clone)]
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

    pub fn contiguous(
        &self,
        context: &MetalContext
    ) -> Result<Self> {
        if self.is_contiguous() {
            return Ok(
                self.clone()
            );
        }

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

        let buffer = materialize_contiguous_f32(
            context, 
            self.metal_buffer()?, 
            self.shape.dims(), 
            self.strides.values(),
        )?;

        let shape = self.shape.clone();

        let strides = Strides::contiguous(&shape,);

        Ok(Self {
            storage:
                Arc::new(
                    Storage::Metal(
                        buffer,
                    ),
                ),
            shape,
            strides,
            dtype:
                self.dtype,
        })
    }

    pub fn transpose(
        &self,
        dim_a: usize,
        dim_b: usize,
    ) -> Result<Self> {
        let rank = self.rank();

       if dim_a >= rank {
         return Err(
            TinyError::InvalidDimension(
                format!(
                    "dimension {dim_a} does not exist for rank {rank}"
                ),
            ),
          );
        }

        if dim_b >= rank {
          return Err(
            TinyError::InvalidDimension(
                format!(
                    "dimension {dim_b} does not exist for rank {rank}"
                ),
            ),
          );
        }
        
        let mut order: Vec<usize> = 
            (0..rank).collect();
        
        order.swap(
            dim_a,
            dim_b,
        );

        self.permute(&order,)
    }

    pub fn permute(
        &self,
        order: &[usize],
    ) -> Result<Self> {
        let rank = self.rank();

        if order.len() != rank {
            return Err(
                TinyError::InvalidDimension(format!(
                    "permute order length {} does not match tensor rank {}",
                    order.len(),
                    rank,
                ),
              ),
            );
        }

        let mut seen = vec![false; rank];

        for &dim in order {
            if dim >= rank {
                return Err(
                    TinyError::InvalidDimension(
                    format!(
                        "dimension {dim} does not exist for rank {rank}",
                    ),
                  ),
                );
            }

            if seen[dim] {
                return Err(
                    TinyError::InvalidDimension(
                    format!(
                        "dimension {dim} appears more than once in permutation",
                    ),
                  ),
                );
            }

            seen[dim] = true;
        }

        let old_dims = self.shape.dims();
        let old_strides = self.strides.values();

        let new_dims: Vec<usize> = 
            order
                .iter()
                .map(|&dim| {
                    old_dims[dim]
                })
                .collect();
        
        let new_strides: Vec<usize> = 
            order
                .iter()
                .map(|&dim| {
                    old_strides[dim]
                })
                .collect();
        
        Ok(Self {
            storage:
                Arc::clone(
                    &self.storage,
                ),
            shape:
                Shape::new(
                    &new_dims,
                )?,
            strides:
                Strides::from_values(
                    new_strides,
                ),
            dtype:
                self.dtype,
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

        if !self.is_contiguous() {
            return Err(
                TinyError::NonContiguousTensor(format!(
                    "shape={:?}, strides={:?}",
                    self.shape.dims(),
                    self.strides.values()
                ),),
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

    pub fn matmul(
        &self,
        context: &MetalContext,
        rhs: &Tensor,
    ) -> Result<Self> {
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

        if rhs.dtype != DType::F32 {
           return Err(
            TinyError::UnsupportedDType(
                format!(
                    "{:?}",
                    rhs.dtype,
                ),
            ),
          );
        }

        if self.rank() != 2 {
          return Err(
            TinyError::InvalidDimension(
                format!(
                    "matmul currently requires rank-2 tensors, left rank is {}",
                    self.rank(),
                ),
            ),
          );
        }

        if rhs.rank() != 2 {
          return Err(
            TinyError::InvalidDimension(
                format!(
                    "matmul currently requires rank-2 tensors, right rank is {}",
                    rhs.rank(),
                ),
            ),
          );
       }

       let m = self.dim(0)?;
       let k = self.dim(1)?;
       let rhs_k = rhs.dim(0)?;
       let n = rhs.dim(1)?;

       if k != rhs_k {
          return Err(
            TinyError::ShapeMismatch {
                left:
                    self.shape
                        .dims()
                        .to_vec(),

                right:
                    rhs.shape
                        .dims()
                        .to_vec(),
             },
          );
       }

       let lhs = 
            if self.is_contiguous() {
                self.clone()
            } else {
                self.contiguous(context,)?
            };
        
        let rhs = 
            if rhs.is_contiguous() {
                rhs.clone()
            } else {
                rhs.contiguous(context,)?
            };
        
        let buffer = matmul_f32(
            context, 
            lhs.metal_buffer()?, 
            rhs.metal_buffer()?, 
            m, 
            k, 
            n
        )?;

        let shape =  
            Shape::new(
                &[m, n],
            )?;
        
        let strides = 
            Strides::contiguous(&shape,);

        
        Ok(Self {
            storage: 
                Arc::new(
                    Storage::Metal(
                        buffer,
                    ),
                ),
            shape,
            strides,
            dtype:
              DType::F32,
        })
    }
}
