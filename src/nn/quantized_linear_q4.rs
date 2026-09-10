use crate::{
    error::{Result, TinyError},
    metal::{MetalBuffer, MetalContext},
    ops::quantized_linear_q4_f16,
    tensor::{DType, Tensor},
};

/// Blockwise symmetric Q4 weight-only linear layer. Each block stores 4-bit
/// signed values packed two per byte and one F32 scale per output channel.
pub struct QuantizedLinearQ4 {
    weight: MetalBuffer,
    scales: MetalBuffer,
    in_features: usize,
    out_features: usize,
    block_size: usize,
}

impl QuantizedLinearQ4 {
    pub const BLOCK_SIZE: usize = 32;

    pub fn from_f16_weight(context: &MetalContext, weight: &Tensor) -> Result<Self> {
        if weight.rank() != 2 || weight.dtype() != DType::F16 {
            return Err(TinyError::UnsupportedDType(
                "QuantizedLinearQ4 requires a rank-2 F16 weight".into(),
            ));
        }
        let in_features = weight.dim(0)?;
        let out_features = weight.dim(1)?;
        let (packed, scales) = quantize_q4(
            &weight.to_f32_vec()?,
            in_features,
            out_features,
            Self::BLOCK_SIZE,
        );
        Ok(Self {
            weight: MetalBuffer::from_u8_slice(context, &packed),
            scales: MetalBuffer::from_slice(context, &scales),
            in_features,
            out_features,
            block_size: Self::BLOCK_SIZE,
        })
    }

    pub fn forward(&self, context: &MetalContext, input: &Tensor) -> Result<Tensor> {
        if input.rank() != 2 || input.dtype() != DType::F16 {
            return Err(TinyError::UnsupportedDType(
                "QuantizedLinearQ4 requires a rank-2 F16 input".into(),
            ));
        }
        if input.dim(1)? != self.in_features {
            return Err(TinyError::InvalidShape("Q4 input feature mismatch".into()));
        }
        let rows = input.dim(0)?;
        let output = quantized_linear_q4_f16(
            context,
            input.metal_buffer()?,
            &self.weight,
            &self.scales,
            rows,
            self.in_features,
            self.out_features,
            self.block_size,
        )?;
        Tensor::from_metal_buffer(output, &[rows, self.out_features], DType::F16)
    }
}

fn quantize_q4(
    values: &[f32],
    rows: usize,
    columns: usize,
    block_size: usize,
) -> (Vec<u8>, Vec<f32>) {
    let mut packed = vec![0u8; rows.div_ceil(2) * columns];
    let mut scales = vec![1.0f32; rows.div_ceil(block_size) * columns];
    for column in 0..columns {
        for block in 0..rows.div_ceil(block_size) {
            let start = block * block_size;
            let end = (start + block_size).min(rows);
            let max_abs = (start..end)
                .map(|row| values[row * columns + column].abs())
                .fold(0.0f32, f32::max);
            let scale = if max_abs == 0.0 { 1.0 } else { max_abs / 7.0 };
            scales[block * columns + column] = scale;
            for row in start..end {
                let code = ((values[row * columns + column] / scale)
                    .round()
                    .clamp(-8.0, 7.0) as i8
                    + 8) as u8;
                let slot = (row / 2) * columns + column;
                if row & 1 == 0 {
                    packed[slot] = code;
                } else {
                    packed[slot] |= code << 4;
                }
            }
        }
    }
    (packed, scales)
}

#[cfg(test)]
mod tests {
    use super::quantize_q4;
    #[test]
    fn packs_two_signed_q4_values_per_byte() {
        let (packed, scales) = quantize_q4(&[-1.0, 1.0], 2, 1, 32);
        assert_eq!(packed, vec![0xF1]);
        assert_eq!(scales, vec![1.0 / 7.0]);
    }
}
