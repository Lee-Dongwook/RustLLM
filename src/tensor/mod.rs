use crate::metal::{
    MetalBuffer,
    MetalContext,
};

use crate::ops::{
    matrix_multiply_naive,
    matrix_multiply_tiled_8,
    matrix_multiply_tiled_16,
    matrix_multiply_tiled_32,
    vector_add,
};
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

    pub fn matmul_naive(
    &self,
    context: &MetalContext,
    rhs: &Tensor,
) -> Tensor {
    let (m, k, n) =
        self.matmul_dimensions(rhs);

    let result_buffer =
        matrix_multiply_naive(
            context,
            &self.buffer,
            &rhs.buffer,
            m,
            k,
            n,
        );

    Tensor {
        buffer: result_buffer,
        shape: vec![m, n],
    }
}

    pub fn matmul_tiled_8(
    &self,
    context: &MetalContext,
    rhs: &Tensor,
) -> Tensor {
    let (m, k, n) =
        self.matmul_dimensions(rhs);

    let buffer =
        matrix_multiply_tiled_8(
            context,
            &self.buffer,
            &rhs.buffer,
            m,
            k,
            n,
        );

    Tensor {
        buffer,
        shape: vec![m, n],
    }
}

pub fn matmul_tiled_16(
    &self,
    context: &MetalContext,
    rhs: &Tensor,
) -> Tensor {
    let (m, k, n) =
        self.matmul_dimensions(rhs);

    let buffer =
        matrix_multiply_tiled_16(
            context,
            &self.buffer,
            &rhs.buffer,
            m,
            k,
            n,
        );

    Tensor {
        buffer,
        shape: vec![m, n],
    }
}

pub fn matmul_tiled_32(
    &self,
    context: &MetalContext,
    rhs: &Tensor,
) -> Tensor {
    let (m, k, n) =
        self.matmul_dimensions(rhs);

    let buffer =
        matrix_multiply_tiled_32(
            context,
            &self.buffer,
            &rhs.buffer,
            m,
            k,
            n,
        );

    Tensor {
        buffer,
        shape: vec![m, n],
    }
}

    fn matmul_dimensions(
        &self,
        rhs: &Tensor,
    ) -> (usize, usize, usize) {
        assert_eq!(
            self.shape.len(),
            2,
            "현재 matmul은 2차원 Tensor만 지원합니다."
        );

        assert_eq!(
            rhs.shape.len(),
            2,
            "현재 matmul은 2차원 Tensor만 지원합니다."
        );

        let m = self.shape[0];
        let k = self.shape[1];

        let rhs_k = rhs.shape[0];
        let n = rhs.shape[1];

        assert_eq!(
            k,
            rhs_k,
            "행렬 곱셈의 내부 차원이 일치하지 않습니다."
        );

        (m, k, n)
    }
}
