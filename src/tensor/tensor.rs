use std::sync::Arc;
use crate::ops::{
    matmul_f32,
    materialize_contiguous_f32,
    silu_f32,
    mul_f32,
    softmax_f32,
    batched_matmul_f32,
    attention_scale_mask_f32,
    add_f32,
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
    pub fn add(
    &self,
    context: &MetalContext,
    rhs: &Tensor,
) -> Result<Self> {
    if self.dtype != DType::F32
        || rhs.dtype != DType::F32
    {
        return Err(
            TinyError::UnsupportedDType(
                "add currently supports only F32"
                    .to_string(),
            ),
        );
    }

    if self.shape != rhs.shape {
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
            self.contiguous(
                context,
            )?
        };

    let rhs =
        if rhs.is_contiguous() {
            rhs.clone()
        } else {
            rhs.contiguous(
                context,
            )?
        };

    let buffer =
        add_f32(
            context,
            lhs.metal_buffer()?,
            rhs.metal_buffer()?,
        )?;

    Self::from_metal_buffer(
        buffer,
        self.shape.dims(),
        DType::F32,
    )
}
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

    pub fn silu(
        &self,
        context: &MetalContext,
    ) -> Result<Self> {
        if self.dtype != DType::F32 {
            return Err(
                TinyError::UnsupportedDType(
                    format!("{:?}", self.dtype),
                ),
            );
        }

        let input =
            if self.is_contiguous() {
                self.clone()
            } else {
                self.contiguous(
                    context,
                )?
            };

        let buffer =
            silu_f32(
                context,
                input.metal_buffer()?,
            )?;

        Self::from_metal_buffer(
            buffer,
            self.shape.dims(),
            DType::F32,
        )
    }
    
    pub fn mul(
        &self,
        context: &MetalContext,
        rhs: &Tensor,
    ) -> Result<Self> {
        if self.dtype != DType::F32
            || rhs.dtype != DType::F32
        {
        return Err(
            TinyError::UnsupportedDType(
                "mul currently supports only F32"
                    .to_string(),
                ),
            );
        }

        if self.shape != rhs.shape {
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
                self.contiguous(
                    context,
                )?
            };

        let rhs =
            if rhs.is_contiguous() {
                rhs.clone()
            } else {
                rhs.contiguous(
                    context,
                )?
            };

        let buffer =
            mul_f32(
                context,
                lhs.metal_buffer()?,
                rhs.metal_buffer()?,
            )?;

        Self::from_metal_buffer(
            buffer,
            self.shape.dims(),
            DType::F32,
        )
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

    pub(crate) fn from_metal_buffer(
        buffer: MetalBuffer,
        dims: &[usize],
        dtype: DType,
    ) -> Result<Self> {
        let shape = Shape::new(dims)?;

        if buffer.len()
            != shape.numel()
        {
            return Err(
                TinyError::InvalidShape(
                    format!(
                        "buffer has {} elements, but shape {:?} requires {}",
                        buffer.len(),
                        shape.dims(),
                        shape.numel(),
                    ),
                ),
            );
        }

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
        dtype,
        })
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

    pub fn softmax_last_dim(
        &self,
        context: &MetalContext,
    ) -> Result<Self> {
    if self.dtype
        != DType::F32
    {
        return Err(
            TinyError::UnsupportedDType(
                format!(
                    "{:?}",
                    self.dtype,
                ),
            ),
        );
    }

    if self.rank() == 0 {
        return Err(
            TinyError::InvalidDimension(
                "Softmax requires at least one dimension"
                    .to_string(),
            ),
        );
    }

    let input =
        if self.is_contiguous() {
            self.clone()
        } else {
            self.contiguous(
                context,
            )?
        };

    let width =
        input.dim(
            input.rank() - 1,
        )?;

    let rows =
        input.numel()
        / width;

    let buffer =
        softmax_f32(
            context,
            input.metal_buffer()?,
            rows,
            width,
        )?;

    Self::from_metal_buffer(
        buffer,
        input.shape().dims(),
        DType::F32,
    )
  }

  pub fn batched_matmul(
    &self,
    context: &MetalContext,
    rhs: &Tensor,
) -> Result<Self> {
    if self.dtype != DType::F32
        || rhs.dtype != DType::F32
    {
        return Err(
            TinyError::UnsupportedDType(
                "batched matmul currently supports only F32"
                    .to_string(),
            ),
        );
    }

    if self.rank() < 3 {
        return Err(
            TinyError::InvalidDimension(
                format!(
                    "batched matmul requires rank >= 3, left rank is {}",
                    self.rank(),
                ),
            ),
        );
    }

    if rhs.rank()
        != self.rank()
    {
        return Err(
            TinyError::InvalidDimension(
                format!(
                    "batched matmul requires tensors with equal rank, got {} and {}",
                    self.rank(),
                    rhs.rank(),
                ),
            ),
        );
    }

    let rank =
        self.rank();

    let lhs_dims =
        self.shape.dims();

    let rhs_dims =
        rhs.shape.dims();

    // 마지막 두 차원을 제외한
    // 모든 batch dimension이 같아야 한다.
    if lhs_dims[..rank - 2]
        != rhs_dims[..rank - 2]
    {
        return Err(
            TinyError::ShapeMismatch {
                left:
                    lhs_dims.to_vec(),

                right:
                    rhs_dims.to_vec(),
            },
        );
    }

    let m =
        lhs_dims[rank - 2];

    let k =
        lhs_dims[rank - 1];

    let rhs_k =
        rhs_dims[rank - 2];

    let n =
        rhs_dims[rank - 1];

    if k != rhs_k {
        return Err(
            TinyError::ShapeMismatch {
                left:
                    lhs_dims.to_vec(),

                right:
                    rhs_dims.to_vec(),
            },
        );
    }

    let batch_count: usize =
        lhs_dims[..rank - 2]
            .iter()
            .product();

    let lhs =
        if self.is_contiguous() {
            self.clone()
        } else {
            self.contiguous(
                context,
            )?
        };

    let rhs =
        if rhs.is_contiguous() {
            rhs.clone()
        } else {
            rhs.contiguous(
                context,
            )?
        };

    let buffer =
        batched_matmul_f32(
            context,
            lhs.metal_buffer()?,
            rhs.metal_buffer()?,
            batch_count,
            m,
            k,
            n,
        )?;

    let mut output_dims =
        lhs_dims[..rank - 2]
            .to_vec();

    output_dims.push(m);
    output_dims.push(n);

    Self::from_metal_buffer(
        buffer,
        &output_dims,
        DType::F32,
    )
}

pub fn attention_scale_mask(
    &self,
    context: &MetalContext,
    scale: f32,
    query_start_pos: usize,
) -> Result<Self> {
    if self.dtype
        != DType::F32
    {
        return Err(
            TinyError::UnsupportedDType(
                format!(
                    "{:?}",
                    self.dtype,
                ),
            ),
        );
    }

    if self.rank() < 2 {
        return Err(
            TinyError::InvalidDimension(
                format!(
                    "attention scores require rank >= 2, got rank {}",
                    self.rank(),
                ),
            ),
        );
    }

    let rank =
        self.rank();

    let query_len =
        self.dim(
            rank - 2,
        )?;

    let key_len =
        self.dim(
            rank - 1,
        )?;

    let input =
        if self.is_contiguous() {
            self.clone()
        } else {
            self.contiguous(
                context,
            )?
        };

    let buffer =
        attention_scale_mask_f32(
            context,
            input.metal_buffer()?,
            scale,
            query_len,
            key_len,
            query_start_pos,
        )?;

    Self::from_metal_buffer(
        buffer,
        input.shape().dims(),
        DType::F32,
    )
}
}
