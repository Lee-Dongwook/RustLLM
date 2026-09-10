use crate::error::{Result, TinyError};

use crate::metal::MetalContext;

use crate::ops::{embedding_f16, embedding_f32};

use crate::tensor::{DType, Tensor};

pub struct Embedding {
    weight: Tensor,
    vocab_size: usize,
    hidden_size: usize,
}

impl Embedding {
    pub fn new(weight: Tensor) -> Result<Self> {
        if weight.rank() != 2 {
            return Err(TinyError::InvalidDimension(format!(
                "Embedding weight must have rank 2, got rank {}",
                weight.rank(),
            )));
        }

        if !matches!(weight.dtype(), DType::F32 | DType::F16) {
            return Err(TinyError::UnsupportedDType(
                format!("{:?}", weight.dtype(),),
            ));
        }

        let vocab_size = weight.dim(0)?;

        let hidden_size = weight.dim(1)?;

        Ok(Self {
            weight,
            vocab_size,
            hidden_size,
        })
    }

    pub fn vocab_size(&self) -> usize {
        self.vocab_size
    }

    pub fn hidden_size(&self) -> usize {
        self.hidden_size
    }

    pub fn weight(&self) -> &Tensor {
        &self.weight
    }

    pub fn dtype(&self) -> DType {
        self.weight.dtype()
    }

    pub fn to_dtype(&self, context: &MetalContext, dtype: DType) -> Result<Self> {
        Self::new(self.weight.to_dtype(context, dtype)?)
    }

    pub fn forward(&self, context: &MetalContext, token_ids: &[u32]) -> Result<Tensor> {
        let weight = if self.weight.is_contiguous() {
            self.weight.clone()
        } else {
            self.weight.contiguous(context)?
        };

        let output = match self.weight.dtype() {
            DType::F32 => embedding_f32(
                context,
                weight.metal_buffer()?,
                token_ids,
                self.vocab_size,
                self.hidden_size,
            )?,
            DType::F16 => embedding_f16(
                context,
                weight.metal_buffer()?,
                token_ids,
                self.vocab_size,
                self.hidden_size,
            )?,
        };

        Tensor::from_metal_buffer(
            output,
            &[token_ids.len(), self.hidden_size],
            self.weight.dtype(),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::Embedding;
    use crate::{
        error::TinyError,
        metal::MetalContext,
        tensor::{DType, Tensor},
    };

    fn metal_context() -> Option<MetalContext> {
        match MetalContext::new() {
            Ok(context) => Some(context),
            Err(TinyError::Metal(message)) => {
                eprintln!("skipping Metal Embedding test: {message}");
                None
            }
            Err(error) => panic!("failed to create Metal context: {error}"),
        }
    }

    #[test]
    fn f16_embedding_selects_rows_and_returns_f16() {
        let Some(context) = metal_context() else {
            return;
        };
        let embedding = Embedding::new(
            Tensor::from_f32_slice(
                &context,
                &[
                    1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
                ],
                &[3, 4],
            )
            .unwrap(),
        )
        .unwrap();
        let embedding_f16 = embedding.to_dtype(&context, DType::F16).unwrap();

        let output = embedding_f16.forward(&context, &[2, 0]).unwrap();
        assert_eq!(embedding_f16.dtype(), DType::F16);
        assert_eq!(output.dtype(), DType::F16);
        assert_eq!(output.shape().dims(), &[2, 4]);
        assert_eq!(
            output.to_f32_vec().unwrap(),
            vec![9.0, 10.0, 11.0, 12.0, 1.0, 2.0, 3.0, 4.0],
        );
    }

    #[test]
    fn f16_embedding_stays_close_to_f32_for_fractional_weights() {
        let Some(context) = metal_context() else {
            return;
        };
        let embedding = Embedding::new(
            Tensor::from_f32_slice(&context, &[0.123456, 1.234567, 2.345678, 3.456789], &[1, 4])
                .unwrap(),
        )
        .unwrap();

        let expected = embedding.forward(&context, &[0]).unwrap();
        let actual = embedding
            .to_dtype(&context, DType::F16)
            .unwrap()
            .forward(&context, &[0])
            .unwrap();

        for (expected, actual) in expected
            .to_f32_vec()
            .unwrap()
            .iter()
            .zip(actual.to_f32_vec().unwrap().iter())
        {
            assert!((expected - actual).abs() < 0.01);
        }
    }
}
