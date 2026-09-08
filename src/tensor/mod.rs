use crate::metal::{
    MetalBuffer,
    MetalContext,
};

use crate::ops::vector_add;

pub struct Tensor {
    buffer: MetalBuffer,
    shape: Vec<usize>,
}

impl Tensor {
    pub fn from_slice(
        context: &MetalContext,
        data: &[f32],
        shape: &[usize],
    ) -> Self {
        assert!(
            !shape.is_empty(),
            "Tensor의 shape는 비어 있을 수 없습니다."
        );

        let expected_len: usize =
            shape.iter().product();

        assert_eq!(
            data.len(),
            expected_len,
            "데이터 개수와 Tensor shape가 일치하지 않습니다."
        );

        let buffer =
            MetalBuffer::from_slice(
                context,
                data,
            );

        Self {
            buffer,
            shape: shape.to_vec(),
        }
    }

    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    pub fn len(&self) -> usize {
        self.buffer.len()
    }

    pub fn as_slice(&self) -> &[f32] {
        self.buffer.as_slice()
    }

    pub fn add(
        &self,
        context: &MetalContext,
        rhs: &Tensor,
    ) -> Tensor {
        assert_eq!(
            self.shape,
            rhs.shape,
            "Tensor의 shape가 서로 다릅니다."
        );

        let result_buffer =
            vector_add(
                context,
                &self.buffer,
                &rhs.buffer,
            );

        Tensor {
            buffer: result_buffer,
            shape: self.shape.clone(),
        }
    }
}
