use metal::CommandBufferRef;

use crate::model::quantize_i8_per_output_channel;
use crate::{
    error::{Result, TinyError},
    metal::{MetalBuffer, MetalContext},
    ops::{quantized_linear_i8_f16, quantized_linear_i8_f16_encode},
    tensor::{DType, Tensor},
};

/// Weight-only INT8 linear layer with one symmetric scale per output channel.
/// Activations and outputs use F16 storage; accumulation is performed in F32.
pub struct QuantizedLinear {
    weight: MetalBuffer,
    scales: MetalBuffer,
    in_features: usize,
    out_features: usize,
}

impl QuantizedLinear {
    pub fn from_i8_parts(
        context: &MetalContext,
        in_features: usize,
        out_features: usize,
        weight: &[i8],
        scales: &[f32],
    ) -> Result<Self> {
        if weight.len() != in_features * out_features || scales.len() != out_features {
            return Err(TinyError::InvalidShape(
                "invalid INT8 linear weight or scale shape".into(),
            ));
        }
        Ok(Self {
            weight: MetalBuffer::from_i8_slice(context, weight),
            scales: MetalBuffer::from_slice(context, scales),
            in_features,
            out_features,
        })
    }
    pub fn from_f16_weight(context: &MetalContext, weight: &Tensor) -> Result<Self> {
        if weight.rank() != 2 || weight.dtype() != DType::F16 {
            return Err(TinyError::UnsupportedDType(
                "QuantizedLinear requires a rank-2 F16 weight".into(),
            ));
        }

        let in_features = weight.dim(0)?;
        let out_features = weight.dim(1)?;
        let source = weight.to_f32_vec()?;
        let (quantized, scales) =
            quantize_i8_per_output_channel(&source, in_features, out_features);

        Ok(Self {
            weight: MetalBuffer::from_i8_slice(context, &quantized),
            scales: MetalBuffer::from_slice(context, &scales),
            in_features,
            out_features,
        })
    }

    pub fn in_features(&self) -> usize {
        self.in_features
    }

    pub fn out_features(&self) -> usize {
        self.out_features
    }

    pub fn forward(&self, context: &MetalContext, input: &Tensor) -> Result<Tensor> {
        if input.rank() != 2 || input.dtype() != DType::F16 {
            return Err(TinyError::UnsupportedDType(
                "QuantizedLinear requires a rank-2 F16 input".into(),
            ));
        }
        if input.dim(1)? != self.in_features {
            return Err(TinyError::InvalidShape(format!(
                "QuantizedLinear input feature mismatch: expected {}, got {}",
                self.in_features,
                input.dim(1)?
            )));
        }

        let rows = input.dim(0)?;
        let output = quantized_linear_i8_f16(
            context,
            input.metal_buffer()?,
            &self.weight,
            &self.scales,
            rows,
            self.in_features,
            self.out_features,
        )?;
        Tensor::from_metal_buffer(output, &[rows, self.out_features], DType::F16)
    }

    pub(crate) fn forward_encode(
        &self,
        context: &MetalContext,
        command_buffer: &CommandBufferRef,
        input: &Tensor,
    ) -> Result<Tensor> {
        if input.rank() != 2 || input.dtype() != DType::F16 {
            return Err(TinyError::UnsupportedDType(
                "QuantizedLinear requires a rank-2 F16 input".into(),
            ));
        }

        if input.dim(1)? != self.in_features {
            return Err(TinyError::InvalidShape(format!(
                "QuantizedLinear input feature mismatch: expected {}, got {}",
                self.in_features,
                input.dim(1)?,
            )));
        }

        let rows = input.dim(0)?;
        let output = quantized_linear_i8_f16_encode(
            context,
            command_buffer,
            input.metal_buffer()?,
            &self.weight,
            &self.scales,
            rows,
            self.in_features,
            self.out_features,
        )?;

        Tensor::from_metal_buffer(output, &[rows, self.out_features], DType::F16)
    }
}

#[cfg(test)]
mod tests {
    use super::QuantizedLinear;
    use crate::{
        error::TinyError,
        metal::{MetalContext, MetalExecution},
        model::quantize_i8_per_output_channel,
        ops::{rmsnorm_f16, rmsnorm_f16_encode},
        tensor::{DType, Tensor},
    };

    #[test]
    fn quantizes_each_output_channel_symmetrically() {
        let (values, scales) = quantize_i8_per_output_channel(&[1.0, -2.0, -1.0, 2.0], 2, 2);
        assert_eq!(values, vec![127, -127, -127, 127]);
        assert_eq!(scales, vec![1.0 / 127.0, 2.0 / 127.0]);
    }

    #[test]
    fn int8_weight_only_linear_stays_close_to_f16() {
        let Ok(context) = MetalContext::new() else {
            return;
        };
        let weight = Tensor::from_f16_slice(
            &context,
            &[
                half::f16::from_f32(0.2),
                half::f16::from_f32(-0.5),
                half::f16::from_f32(1.0),
                half::f16::from_f32(0.25),
            ],
            &[2, 2],
        )
        .unwrap();
        let input = Tensor::from_f16_slice(
            &context,
            &[half::f16::from_f32(1.5), half::f16::from_f32(-2.0)],
            &[1, 2],
        )
        .unwrap();
        let expected = input
            .matmul(&context, &weight)
            .unwrap()
            .to_f32_vec()
            .unwrap();
        let actual = QuantizedLinear::from_f16_weight(&context, &weight)
            .unwrap()
            .forward(&context, &input)
            .unwrap();

        assert_eq!(actual.dtype(), DType::F16);
        for (actual, expected) in actual.to_f32_vec().unwrap().iter().zip(expected) {
            assert!((actual - expected).abs() < 0.03);
        }
    }

    #[test]
    fn int8_gemv_handles_partial_tile_and_threadgroup() {
        let Ok(context) = MetalContext::new() else {
            return;
        };
        let source_weight: Vec<half::f16> = (0..70 * 300)
            .map(|index| half::f16::from_f32(((index % 31) as f32 - 15.0) / 20.0))
            .collect();
        let source_input: Vec<half::f16> = (0..70)
            .map(|index| half::f16::from_f32(((index % 13) as f32 - 6.0) / 10.0))
            .collect();
        let weight = Tensor::from_f16_slice(&context, &source_weight, &[70, 300]).unwrap();
        let input = Tensor::from_f16_slice(&context, &source_input, &[1, 70]).unwrap();
        let expected = input
            .matmul(&context, &weight)
            .unwrap()
            .to_f32_vec()
            .unwrap();
        let actual = QuantizedLinear::from_f16_weight(&context, &weight)
            .unwrap()
            .forward(&context, &input)
            .unwrap()
            .to_f32_vec()
            .unwrap();

        assert_eq!(actual.len(), 300);
        for (actual, expected) in actual.iter().zip(expected.iter()) {
            assert!((actual - expected).abs() < 0.15);
        }
    }

    #[test]
    fn rejects_non_f16_weight() {
        let Ok(context) = MetalContext::new() else {
            return;
        };
        let weight = Tensor::from_f32_slice(&context, &[1.0], &[1, 1]).unwrap();
        assert!(matches!(
            QuantizedLinear::from_f16_weight(&context, &weight),
            Err(TinyError::UnsupportedDType(_))
        ));
    }

    #[test]
    fn batches_rmsnorm_then_int8_linear_in_one_command_buffer() {
        let Ok(context) = MetalContext::new() else {
            return;
        };

        /*
         * hidden size = 4
         *
         * 실제 Transformer의 hidden vector처럼
         * [1, hidden] 형태를 사용한다.
         */
        let input_values = [1.0f32, 2.0, 3.0, 4.0];

        let norm_weight_values = [1.0f32, 1.0, 1.0, 1.0];

        let input_f16: Vec<_> = input_values
            .iter()
            .copied()
            .map(half::f16::from_f32)
            .collect();

        let norm_weight_f16: Vec<_> = norm_weight_values
            .iter()
            .copied()
            .map(half::f16::from_f32)
            .collect();

        let input = Tensor::from_f16_slice(&context, &input_f16, &[1, 4]).unwrap();

        let norm_weight = Tensor::from_f16_slice(&context, &norm_weight_f16, &[4]).unwrap();

        /*
         * 4 → 3 Linear.
         *
         * runtime convention:
         * [in_features, out_features]
         *
         * 즉 shape = [4, 3]
         */
        let dense_weight_values = [
            0.2f32, -0.5, 0.7, 1.0, 0.25, -0.3, -0.4, 0.8, 0.1, 0.6, -0.2, 0.9,
        ];

        let dense_weight_f16: Vec<_> = dense_weight_values
            .iter()
            .copied()
            .map(half::f16::from_f32)
            .collect();

        let dense_weight = Tensor::from_f16_slice(&context, &dense_weight_f16, &[4, 3]).unwrap();

        /*
         * 기존 QuantizedLinear constructor가
         * F16 weight를 INT8 + scale로 바꿔준다.
         */
        let linear = QuantizedLinear::from_f16_weight(&context, &dense_weight).unwrap();

        /*
         * 먼저 기존 synchronous path로
         * reference 값을 만든다.
         */

        let normalized_sync_buffer = rmsnorm_f16(
            &context,
            input.metal_buffer().unwrap(),
            norm_weight.metal_buffer().unwrap(),
            1,
            4,
            1e-5,
        )
        .unwrap();

        let normalized_sync =
            Tensor::from_metal_buffer(normalized_sync_buffer, &[1, 4], DType::F16).unwrap();

        let expected = linear
            .forward(&context, &normalized_sync)
            .unwrap()
            .to_f32_vec()
            .unwrap();

        /*
         * =====================================
         * 이제 실제 batching path
         * =====================================
         */

        let execution = MetalExecution::new(&context);

        /*
         * Kernel #1
         *
         * RMSNorm
         *
         * 여기서 기다리지 않는다.
         */
        let normalized_buffer = rmsnorm_f16_encode(
            &context,
            execution.command_buffer(),
            input.metal_buffer().unwrap(),
            norm_weight.metal_buffer().unwrap(),
            1,
            4,
            1e-5,
        )
        .unwrap();

        let normalized = Tensor::from_metal_buffer(normalized_buffer, &[1, 4], DType::F16).unwrap();

        /*
         * Kernel #2
         *
         * INT8 Linear.
         *
         * normalized의 GPU 계산이 아직
         * 완료되지 않았더라도 동일 command buffer이므로
         * 바로 input으로 사용한다.
         */
        let actual_tensor = linear
            .forward_encode(&context, execution.command_buffer(), &normalized)
            .unwrap();

        /*
         * 두 kernel을 모두 encode한 뒤
         * 여기서 단 한 번 실행 / 대기.
         */
        execution.finish();

        let actual = actual_tensor.to_f32_vec().unwrap();

        assert_eq!(actual.len(), expected.len(),);

        for (index, (actual, expected)) in actual.iter().zip(expected.iter()).enumerate() {
            let error = (actual - expected).abs();

            assert!(
                error < 0.02,
                "value {index} differs: actual={actual}, expected={expected}, error={error}",
            );
        }
    }
}
