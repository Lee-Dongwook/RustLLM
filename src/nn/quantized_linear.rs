use crate::{
    error::{Result, TinyError},
    metal::{MetalBuffer, MetalContext},
    ops::quantized_linear_i8_f16,
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
        let (quantized, scales) = quantize_per_output_channel(&source, in_features, out_features);

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
}

fn quantize_per_output_channel(values: &[f32], rows: usize, columns: usize) -> (Vec<i8>, Vec<f32>) {
    let mut quantized = vec![0i8; values.len()];
    let mut scales = vec![1.0f32; columns];

    for column in 0..columns {
        let max_abs = (0..rows)
            .map(|row| values[row * columns + column].abs())
            .fold(0.0f32, f32::max);
        let scale = if max_abs == 0.0 { 1.0 } else { max_abs / 127.0 };
        scales[column] = scale;

        for row in 0..rows {
            quantized[row * columns + column] = (values[row * columns + column] / scale)
                .round()
                .clamp(-127.0, 127.0) as i8;
        }
    }

    (quantized, scales)
}

#[cfg(test)]
mod tests {
    use super::{QuantizedLinear, quantize_per_output_channel};
    use crate::{
        error::TinyError,
        metal::MetalContext,
        tensor::{DType, Tensor},
    };

    #[test]
    fn quantizes_each_output_channel_symmetrically() {
        let (values, scales) = quantize_per_output_channel(&[1.0, -2.0, -1.0, 2.0], 2, 2);
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
}
