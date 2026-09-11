use super::{ModelWeights, weights::WeightData};
use crate::error::{Result, TinyError};

pub fn linear_scale_name(weight_name: &str) -> String {
    match weight_name.strip_suffix(".weight") {
        Some(prefix) => format!("{prefix}.scale"),

        None => format!("{weight_name}.scale"),
    }
}

pub fn quantize_i8_per_output_channel(
    values: &[f32],
    rows: usize,
    columns: usize,
) -> (Vec<i8>, Vec<f32>) {
    let mut quantized = vec![0i8; values.len()];
    let mut scales = vec![1.0f32; columns];

    for column in 0..columns {
        let max_abs = (0..rows)
            .map(|row| values[row * columns + column].abs())
            .fold(0.0f32, f32::max);

        let scale = if max_abs == 0.0 { 1.0 } else { max_abs / 127.0 };

        scales[column] = scale;
        for row in 0..rows {
            let index = row * columns + column;
            quantized[index] = (values[index] / scale).round().clamp(-127.0, 127.0) as i8;
        }
    }

    (quantized, scales)
}

fn quantize_linear_weight(
    source: &mut ModelWeights,
    target: &mut ModelWeights,
    name: &str,
) -> Result<()> {
    let weight = source.take(name)?;

    let (shape, data) = weight.into_storage_parts();

    if shape.len() != 2 {
        return Err(TinyError::InvalidDimension(format!(
            "INT8 quantization expects rank-2 Linear weight {name}, got {shape:?}"
        )));
    }

    let values = match data {
        WeightData::F32(values) => values,

        _ => {
            return Err(TinyError::ModelFormat(format!(
                "INT8 quantizer expects F32 source weight {name}"
            )));
        }
    };

    let (quantized, scales) = quantize_i8_per_output_channel(&values, shape[0], shape[1]);

    target.insert_i8(name, &shape, quantized)?;

    target.insert_f32(linear_scale_name(name), &[shape[1]], scales)?;

    Ok(())
}

fn move_f16_weight(source: &mut ModelWeights, target: &mut ModelWeights, name: &str) -> Result<()> {
    let weight = source.take(name)?;

    let (shape, data) = weight.into_storage_parts();

    let values = match data {
        WeightData::F32(values) => values,

        _ => {
            return Err(TinyError::ModelFormat(format!(
                "expected F32 source weight {name}"
            )));
        }
    };

    let values = values.into_iter().map(half::f16::from_f32).collect();

    target.insert_f16(name, &shape, values)
}

pub fn quantize_model_i8(mut source: ModelWeights, num_layers: usize) -> Result<ModelWeights> {
    let mut target = ModelWeights::new();

    move_f16_weight(&mut source, &mut target, "token_embedding.weight")?;

    for layer in 0..num_layers {
        let prefix = format!("layers.{layer}");

        move_f16_weight(
            &mut source,
            &mut target,
            &format!("{prefix}.attention_norm.weight"),
        )?;

        for projection in [
            "attention.q_proj.weight",
            "attention.k_proj.weight",
            "attention.v_proj.weight",
            "attention.out_proj.weight",
            "mlp.gate_proj.weight",
            "mlp.up_proj.weight",
            "mlp.down_proj.weight",
        ] {
            quantize_linear_weight(&mut source, &mut target, &format!("{prefix}.{projection}"))?;
        }

        move_f16_weight(
            &mut source,
            &mut target,
            &format!("{prefix}.mlp_norm.weight"),
        )?;
    }

    move_f16_weight(&mut source, &mut target, "final_norm.weight")?;

    quantize_linear_weight(&mut source, &mut target, "lm_head.weight")?;

    if !source.is_empty() {
        return Err(TinyError::ModelFormat(format!(
            "INT8 conversion left {} unhandled tensors",
            source.len(),
        )));
    }

    Ok(target)
}

#[cfg(test)]
mod tests {
    use super::{linear_scale_name, quantize_model_i8};
    use crate::model::{ModelWeights, weights::WeightData};

    #[test]
    fn quantized_model_preserves_mixed_weight_storage() {
        let mut source = ModelWeights::new();
        source
            .insert_f32("token_embedding.weight", &[2, 2], vec![1.0; 4])
            .unwrap();
        source
            .insert_f32("layers.0.attention_norm.weight", &[2], vec![1.0; 2])
            .unwrap();
        for projection in [
            "attention.q_proj.weight",
            "attention.k_proj.weight",
            "attention.v_proj.weight",
            "attention.out_proj.weight",
            "mlp.gate_proj.weight",
            "mlp.up_proj.weight",
            "mlp.down_proj.weight",
        ] {
            source
                .insert_f32(
                    format!("layers.0.{projection}"),
                    &[2, 2],
                    vec![-2.0, 1.0, 2.0, -1.0],
                )
                .unwrap();
        }
        source
            .insert_f32("layers.0.mlp_norm.weight", &[2], vec![1.0; 2])
            .unwrap();
        source
            .insert_f32("final_norm.weight", &[2], vec![1.0; 2])
            .unwrap();
        source
            .insert_f32("lm_head.weight", &[2, 2], vec![-2.0, 1.0, 2.0, -1.0])
            .unwrap();

        let quantized = quantize_model_i8(source, 1).unwrap();

        assert!(matches!(
            &quantized.get("token_embedding.weight").unwrap().data,
            WeightData::F16(_)
        ));
        assert!(matches!(
            &quantized
                .get("layers.0.attention.q_proj.weight")
                .unwrap()
                .data,
            WeightData::I8(_)
        ));
        assert!(matches!(
            &quantized
                .get(&linear_scale_name("layers.0.attention.q_proj.weight"))
                .unwrap()
                .data,
            WeightData::F32(_)
        ));
        assert_eq!(quantized.len(), 20);
    }
}
