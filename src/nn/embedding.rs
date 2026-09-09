use crate::error::{
    Result,
    TinyError,
};

use crate::metal::MetalContext;

use crate::ops::embedding_f32;

use crate::tensor::{
    DType,
    Tensor,
};

pub struct Embedding {
    weight: Tensor,
    vocab_size: usize,
    hidden_size: usize,
}

impl Embedding {
    pub fn new(
        weight: Tensor,
    ) -> Result<Self> {
        if weight.rank() != 2 {
            return Err(
                TinyError::InvalidDimension(
                    format!(
                        "Embedding weight must have rank 2, got rank {}",
                        weight.rank(),
                    ),
                ),
            );
        }

        if weight.dtype()
            != DType::F32
        {
            return Err(
                TinyError::UnsupportedDType(
                    format!(
                        "{:?}",
                        weight.dtype(),
                    ),
                ),
            );
        }

        let vocab_size = 
            weight.dim(0)?;

        let hidden_size = 
            weight.dim(1)?;
        
        Ok(Self {
            weight,
            vocab_size,
            hidden_size,
        })
    }

      pub fn vocab_size(
        &self,
    ) -> usize {
        self.vocab_size
    }

    pub fn hidden_size(
        &self,
    ) -> usize {
        self.hidden_size
    }

    pub fn weight(
        &self,
    ) -> &Tensor {
        &self.weight
    }

    pub fn forward(
        &self,
        context: &MetalContext,
        token_ids: &[u32],
    ) -> Result<Tensor> {
        let weight =
            if self.weight
                .is_contiguous()
            {
                self.weight.clone()
            } else {
                self.weight
                    .contiguous(
                        context,
                    )?
            };

        let output =
            embedding_f32(
                context,
                weight.metal_buffer()?,
                token_ids,
                self.vocab_size,
                self.hidden_size,
            )?;
        
        Tensor::from_metal_buffer(
            output,
            &[
                token_ids.len(),
                self.hidden_size,
            ],
            DType::F32,
        )
    }
}
