use crate::error::{Result, TinyError};

use crate::model::{ModelWeights, WeightTensor};

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
