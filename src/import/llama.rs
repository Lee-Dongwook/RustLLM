use std::fs;
use std::path::Path;

use serde::Deserialize;

use crate::error::{Result, TinyError};

use crate::model::{ModelConfig, ModelWeights, WeightTensor};

#[derive(Debug, Deserialize)]
struct LlamaConfigFile {
    vocab_size: usize,
    hidden_size: usize,
    intermediate_size: usize,
    num_hidden_layers: usize,
    num_attention_heads: usize,
    num_key_value_heads: Option<usize>,
    attention_bias: Option<bool>,
    mlp_bias: Option<bool>,
    max_position_embeddings: usize,
    rms_norm_eps: f32,
    rope_theta: f32,
}

pub fn load_llama_config(path: impl AsRef<Path>) -> Result<ModelConfig> {
    let text = fs::read_to_string(path)?;

    parse_llama_config(&text)
}

fn parse_llama_config(text: &str) -> Result<ModelConfig> {
    let source: LlamaConfigFile = serde_json::from_str(text)
        .map_err(|error| TinyError::ModelFormat(format!("invalid Llama config.json: {error}")))?;

    let num_key_value_heads = source
        .num_key_value_heads
        .unwrap_or(source.num_attention_heads);

    if num_key_value_heads != source.num_attention_heads {
        return Err(TinyError::ModelFormat(format!(
            "grouped-query attention is not supported: num_key_value_heads={num_key_value_heads}, num_attention_heads={}",
            source.num_attention_heads,
        )));
    }

    if source.attention_bias.unwrap_or(false)
        || source.mlp_bias.unwrap_or(false)
    {
        return Err(TinyError::ModelFormat(
            "Llama bias tensors are not supported".to_string(),
        ));
    }

    let config = ModelConfig {
        vocab_size: source.vocab_size,
        hidden_size: source.hidden_size,
        intermediate_size: source.intermediate_size,
        num_layers: source.num_hidden_layers,
        num_heads: source.num_attention_heads,
        max_seq_len: source.max_position_embeddings,
        rms_norm_eps: source.rms_norm_eps,
        rope_theta: source.rope_theta,
    };

    config.validate()?;

    Ok(config)
}

fn transpose_2d(name: &str, weight: WeightTensor) -> Result<(Vec<usize>, Vec<f32>)> {
    let (shape, data) = weight.into_parts();

    if shape.len() != 2 {
        return Err(TinyError::ModelFormat(format!(
            "cannot transpose {name}: expected rank 2, got shape {shape:?}"
        )));
    }

    let rows = shape[0];

    let cols = shape[1];

    let mut output = vec![0.0f32; rows * cols];

    for row in 0..rows {
        for col in 0..cols {
            output[col * rows + row] = data[row * cols + col];
        }
    }

    Ok((vec![cols, rows], output))
}

fn move_weight(
    source: &mut ModelWeights,
    target: &mut ModelWeights,
    source_name: &str,
    target_name: &str,
) -> Result<()> {
    let weight = source.take(source_name)?;

    let (shape, data) = weight.into_parts();

    target.insert_f32(target_name, &shape, data)
}

fn move_linear_weight(
    source: &mut ModelWeights,
    target: &mut ModelWeights,
    source_name: &str,
    target_name: &str,
) -> Result<()> {
    let weight = source.take(source_name)?;

    let (shape, data) = transpose_2d(source_name, weight)?;

    target.insert_f32(target_name, &shape, data)
}

pub fn map_llama_weights(mut source: ModelWeights, num_layers: usize) -> Result<ModelWeights> {
    let mut target = ModelWeights::new();

    // ------------------------------------------------
    // Token Embedding
    //
    // HF:   [vocab_size, hidden_size]
    // Ours: [vocab_size, hidden_size]
    //
    // lm_head weight tying에 사용할 것이므로 clone해 둔다.
    // ------------------------------------------------

    // 일부 safetensors 파일은 tie_word_embeddings를 사용하면서
    // `model.embed_tokens.weight`를 저장하지 않고 `lm_head.weight`를
    // embedding의 별칭으로 기록한다. 둘 중 실제로 존재하는 텐서를 쓴다.
    let embedding_source_name = if source.contains("model.embed_tokens.weight") {
        "model.embed_tokens.weight"
    } else {
        "lm_head.weight"
    };

    let embedding = source.get(embedding_source_name)?.clone();

    let (embedding_shape, embedding_data) = embedding.clone().into_parts();

    target.insert_f32("token_embedding.weight", &embedding_shape, embedding_data)?;

    // ------------------------------------------------
    // Transformer Layers
    // ------------------------------------------------

    for layer_index in 0..num_layers {
        let hf = format!("model.layers.{layer_index}");

        let ours = format!("layers.{layer_index}");

        // Attention RMSNorm
        move_weight(
            &mut source,
            &mut target,
            &format!("{hf}.input_layernorm.weight"),
            &format!("{ours}.attention_norm.weight"),
        )?;

        // Q projection
        move_linear_weight(
            &mut source,
            &mut target,
            &format!("{hf}.self_attn.q_proj.weight"),
            &format!("{ours}.attention.q_proj.weight"),
        )?;

        // K projection
        move_linear_weight(
            &mut source,
            &mut target,
            &format!("{hf}.self_attn.k_proj.weight"),
            &format!("{ours}.attention.k_proj.weight"),
        )?;

        // V projection
        move_linear_weight(
            &mut source,
            &mut target,
            &format!("{hf}.self_attn.v_proj.weight"),
            &format!("{ours}.attention.v_proj.weight"),
        )?;

        // Output projection
        move_linear_weight(
            &mut source,
            &mut target,
            &format!("{hf}.self_attn.o_proj.weight"),
            &format!("{ours}.attention.out_proj.weight"),
        )?;

        // MLP RMSNorm
        move_weight(
            &mut source,
            &mut target,
            &format!("{hf}.post_attention_layernorm.weight"),
            &format!("{ours}.mlp_norm.weight"),
        )?;

        // SwiGLU gate projection
        move_linear_weight(
            &mut source,
            &mut target,
            &format!("{hf}.mlp.gate_proj.weight"),
            &format!("{ours}.mlp.gate_proj.weight"),
        )?;

        // SwiGLU up projection
        move_linear_weight(
            &mut source,
            &mut target,
            &format!("{hf}.mlp.up_proj.weight"),
            &format!("{ours}.mlp.up_proj.weight"),
        )?;

        // SwiGLU down projection
        move_linear_weight(
            &mut source,
            &mut target,
            &format!("{hf}.mlp.down_proj.weight"),
            &format!("{ours}.mlp.down_proj.weight"),
        )?;
    }

    // ------------------------------------------------
    // Final RMSNorm
    // ------------------------------------------------

    move_weight(
        &mut source,
        &mut target,
        "model.norm.weight",
        "final_norm.weight",
    )?;

    // ------------------------------------------------
    // LM Head
    //
    // 별도 lm_head.weight가 존재하면 그것을 사용.
    //
    // 없으면 embedding weight tying 모델이므로
    // embedding [vocab, hidden]을 transpose하여
    // [hidden, vocab]으로 만든다.
    // ------------------------------------------------

    if source.contains("lm_head.weight") {
        move_linear_weight(&mut source, &mut target, "lm_head.weight", "lm_head.weight")?;
    } else {
        let (lm_head_shape, lm_head_data) = transpose_2d(embedding_source_name, embedding)?;

        target.insert_f32("lm_head.weight", &lm_head_shape, lm_head_data)?;
    }

    Ok(target)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn insert(weights: &mut ModelWeights, name: &str, shape: &[usize], data: Vec<f32>) {
        weights.insert_f32(name, shape, data).unwrap();
    }

    fn one_layer_source() -> ModelWeights {
        let mut weights = ModelWeights::new();
        let matrix = vec![1.0, 2.0, 3.0, 4.0];

        // 이 checkpoint는 tie_word_embeddings를 사용해 embedding을 lm_head에만 저장한다.
        insert(&mut weights, "lm_head.weight", &[2, 2], matrix.clone());
        insert(&mut weights, "model.norm.weight", &[2], vec![1.0, 1.0]);

        let layer = "model.layers.0";
        insert(
            &mut weights,
            &format!("{layer}.input_layernorm.weight"),
            &[2],
            vec![1.0, 1.0],
        );
        insert(
            &mut weights,
            &format!("{layer}.post_attention_layernorm.weight"),
            &[2],
            vec![1.0, 1.0],
        );

        for name in ["q_proj", "k_proj", "v_proj", "o_proj"] {
            insert(
                &mut weights,
                &format!("{layer}.self_attn.{name}.weight"),
                &[2, 2],
                matrix.clone(),
            );
        }

        for name in ["gate_proj", "up_proj", "down_proj"] {
            insert(
                &mut weights,
                &format!("{layer}.mlp.{name}.weight"),
                &[2, 2],
                matrix.clone(),
            );
        }

        weights
    }

    #[test]
    fn maps_tied_lm_head_as_token_embedding() {
        let converted = map_llama_weights(one_layer_source(), 1).unwrap();

        let embedding = converted.get("token_embedding.weight").unwrap();
        assert_eq!(embedding.shape(), &[2, 2]);
        assert_eq!(embedding.data(), &[1.0, 2.0, 3.0, 4.0]);

        let lm_head = converted.get("lm_head.weight").unwrap();
        assert_eq!(lm_head.shape(), &[2, 2]);
        assert_eq!(lm_head.data(), &[1.0, 3.0, 2.0, 4.0]);

        let q_proj = converted.get("layers.0.attention.q_proj.weight").unwrap();
        assert_eq!(q_proj.data(), &[1.0, 3.0, 2.0, 4.0]);
        assert_eq!(converted.len(), 12);
    }

    #[test]
    fn rejects_grouped_query_attention() {
        let error = parse_llama_config(
            r#"{
                "vocab_size": 8,
                "hidden_size": 4,
                "intermediate_size": 8,
                "num_hidden_layers": 1,
                "num_attention_heads": 2,
                "num_key_value_heads": 1,
                "max_position_embeddings": 16,
                "rms_norm_eps": 0.00001,
                "rope_theta": 10000.0
            }"#,
        )
        .unwrap_err();

        assert!(error.to_string().contains("grouped-query attention"));
    }

    #[test]
    fn rejects_models_with_bias_tensors() {
        let error = parse_llama_config(
            r#"{
                "vocab_size": 8,
                "hidden_size": 4,
                "intermediate_size": 8,
                "num_hidden_layers": 1,
                "num_attention_heads": 2,
                "max_position_embeddings": 16,
                "rms_norm_eps": 0.00001,
                "rope_theta": 10000.0,
                "attention_bias": true
            }"#,
        )
        .unwrap_err();

        assert!(error.to_string().contains("bias tensors"));
    }
}
