use std::fs;
use std::path::Path;

use serde::Deserialize;

use crate::error::{
    Result,
    TinyError,
};

use crate::model::{
    ModelConfig,
    ModelWeights,
    WeightTensor,
};

#[derive(Debug, Deserialize)]
struct LlamaConfigFile {
    #[serde(default)]
    model_type: Option<String>,

    vocab_size: usize,
    hidden_size: usize,
    intermediate_size: usize,

    num_hidden_layers: usize,
    num_attention_heads: usize,

    #[serde(default)]
    num_key_value_heads: Option<usize>,

    #[serde(default)]
    attention_bias: Option<bool>,

    #[serde(default)]
    mlp_bias: Option<bool>,

    max_position_embeddings: usize,
    rms_norm_eps: f32,

    #[serde(default = "default_rope_theta")]
    rope_theta: f32,

    #[serde(default)]
    hidden_act: Option<String>,
}

fn default_rope_theta() -> f32 {
    10_000.0
}

pub fn load_llama_config(
    path: impl AsRef<Path>,
) -> Result<ModelConfig> {
    let text =
        fs::read_to_string(path)?;

    parse_llama_config(
        &text,
    )
}

pub fn map_llama_weights(
    mut source: ModelWeights,
    num_layers: usize,
) -> Result<ModelWeights> {
    let mut target = ModelWeights::new();

    // ------------------------------------------------
    // Token Embedding
    //
    // 일반적인 HF checkpoint:
    //   model.embed_tokens.weight
    //
    // 일부 tied checkpoint:
    //   lm_head.weight 만 존재할 수도 있음
    //
    // shape:
    //   [vocab_size, hidden_size]
    // ------------------------------------------------

    let embedding_source_name =
        if source.contains("model.embed_tokens.weight") {
            "model.embed_tokens.weight"
        } else if source.contains("lm_head.weight") {
            "lm_head.weight"
        } else {
            return Err(
                TinyError::MissingWeight(
                    "model.embed_tokens.weight or lm_head.weight"
                        .to_string(),
                ),
            );
        };
    
    let embedding =
        source
            .get(embedding_source_name)?
            .clone();
    
    let (
        embedding_shape,
        embedding_data,
    ) = embedding
        .clone()
        .into_parts();

    target.insert_f32(
        "token_embedding.weight", 
        &embedding_shape, 
        embedding_data
    )?;

    for layer_index in 0..num_layers {
        let hf = format!("model.layers.{layer_index}");
        let ours = format!("layers.{layer_index}");

        move_weight(
            &mut source,
            &mut target,
            &format!(
                "{hf}.input_layernorm.weight"
            ),
            &format!(
                "{ours}.attention_norm.weight"
            ),
        )?;

        move_linear_weight(
            &mut source,
            &mut target,
            &format!(
                "{hf}.self_attn.q_proj.weight"
            ),
            &format!(
                "{ours}.attention.q_proj.weight"
            ),
        )?;

        move_linear_weight(
            &mut source,
            &mut target,
            &format!(
                "{hf}.self_attn.k_proj.weight"
            ),
            &format!(
                "{ours}.attention.k_proj.weight"
            ),
        )?;

        move_linear_weight(
            &mut source,
            &mut target,
            &format!(
                "{hf}.self_attn.v_proj.weight"
            ),
            &format!(
                "{ours}.attention.v_proj.weight"
            ),
        )?;

        move_linear_weight(
            &mut source,
            &mut target,
            &format!(
                "{hf}.self_attn.o_proj.weight"
            ),
            &format!(
                "{ours}.attention.out_proj.weight"
            ),
        )?;

        move_weight(
            &mut source,
            &mut target,
            &format!(
                "{hf}.post_attention_layernorm.weight"
            ),
            &format!(
                "{ours}.mlp_norm.weight"
            ),
        )?;

        move_linear_weight(
            &mut source,
            &mut target,
            &format!(
                "{hf}.mlp.gate_proj.weight"
            ),
            &format!(
                "{ours}.mlp.gate_proj.weight"
            ),
        )?;

        move_linear_weight(
            &mut source,
            &mut target,
            &format!(
                "{hf}.mlp.up_proj.weight"
            ),
            &format!(
                "{ours}.mlp.up_proj.weight"
            ),
        )?;

        move_linear_weight(
            &mut source,
            &mut target,
            &format!(
                "{hf}.mlp.down_proj.weight"
            ),
            &format!(
                "{ours}.mlp.down_proj.weight"
            ),
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
        // HF Linear:
        //   [vocab_size, hidden_size]
        //
        // 우리 Linear:
        //   [hidden_size, vocab_size]
        //
        // 따라서 transpose 필요
        // ------------------------------------------------

        if source.contains("lm_head.weight") {
            move_linear_weight(
                &mut source,
                &mut target,
                "lm_head.weight",
                "lm_head.weight",
            )?;
        } else {
           let (
             lm_head_shape,
             lm_head_data,
           ) = transpose_2d(
            embedding_source_name,
            embedding,
           )?;

           target.insert_f32(
            "lm_head.weight", 
            &lm_head_shape, 
            lm_head_data
            )?;
        }
        Ok(target)
}

fn transpose_2d(
    name: &str,
    weight: WeightTensor,
) -> Result<(Vec<usize>, Vec<f32>)> {
    let (shape, data) =
        weight.into_parts();

    if shape.len() != 2 {
        return Err(
            TinyError::ModelFormat(
                format!(
                    "cannot transpose {name}: expected rank 2, got {shape:?}"
                ),
            ),
        );
    }

    let rows = shape[0];
    let cols = shape[1];

    let mut output =
        vec![0.0f32; rows * cols];

    for row in 0..rows {
        for col in 0..cols {
            output[
                col * rows + row
            ] =
                data[
                    row * cols + col
                ];
        }
    }

    Ok((
        vec![cols, rows],
        output,
    ))
}

fn move_weight(
    source: &mut ModelWeights,
    target: &mut ModelWeights,
    source_name: &str,
    target_name: &str,
) -> Result<()> {
    let weight =
        source.take(source_name)?;

    let (shape, data) =
        weight.into_parts();

    target.insert_f32(
        target_name,
        &shape,
        data,
    )
}

fn move_linear_weight(
    source: &mut ModelWeights,
    target: &mut ModelWeights,
    source_name: &str,
    target_name: &str,
) -> Result<()> {
    let weight =
        source.take(source_name)?;

    let (shape, data) =
        transpose_2d(
            source_name,
            weight,
        )?;

    target.insert_f32(
        target_name,
        &shape,
        data,
    )
}

fn parse_llama_config(
    text: &str,
) -> Result<ModelConfig> {
    let source: LlamaConfigFile =
        serde_json::from_str(text)
            .map_err(|error| {
                TinyError::ModelFormat(
                    format!(
                        "invalid Llama config.json: {error}"
                    ),
                )
            })?;

    // ------------------------------------------------
    // Architecture
    // ------------------------------------------------

    if let Some(
        model_type
    ) = source.model_type.as_deref()
    {
        if model_type != "llama" {
            return Err(
                TinyError::ModelFormat(
                    format!(
                        "unsupported model_type: {model_type}; expected llama"
                    ),
                ),
            );
        }
    }

    // ------------------------------------------------
    // GQA / MQA
    //
    // 현재 우리 SelfAttention은
    // num_heads == num_key_value_heads 만 지원
    // ------------------------------------------------

    let num_key_value_heads =
        source
            .num_key_value_heads
            .unwrap_or(
                source.num_attention_heads,
            );

    if num_key_value_heads
        != source.num_attention_heads
    {
        return Err(
            TinyError::ModelFormat(
                format!(
                    "grouped-query attention is not supported: \
                     num_key_value_heads={}, \
                     num_attention_heads={}",
                    num_key_value_heads,
                    source.num_attention_heads,
                ),
            ),
        );
    }

    // ------------------------------------------------
    // Head dimension
    // ------------------------------------------------

    if source.hidden_size
        % source.num_attention_heads
        != 0
    {
        return Err(
            TinyError::ModelFormat(
                format!(
                    "hidden_size {} is not divisible by num_attention_heads {}",
                    source.hidden_size,
                    source.num_attention_heads,
                ),
            ),
        );
    }

    // ------------------------------------------------
    // Bias
    // ------------------------------------------------

    if source
        .attention_bias
        .unwrap_or(false)
        || source
            .mlp_bias
            .unwrap_or(false)
    {
        return Err(
            TinyError::ModelFormat(
                "Llama bias tensors are not supported"
                    .to_string(),
            ),
        );
    }

    // ------------------------------------------------
    // Activation
    // ------------------------------------------------

    if let Some(
        hidden_act
    ) = source.hidden_act.as_deref()
    {
        if hidden_act != "silu" {
            return Err(
                TinyError::ModelFormat(
                    format!(
                        "unsupported hidden_act: {hidden_act}; expected silu"
                    ),
                ),
            );
        }
    }

    // ------------------------------------------------
    // HF Config → Our Config
    // ------------------------------------------------

    let config =
        ModelConfig {
            vocab_size:
                source.vocab_size,

            hidden_size:
                source.hidden_size,

            intermediate_size:
                source.intermediate_size,

            num_layers:
                source.num_hidden_layers,

            num_heads:
                source.num_attention_heads,

            max_seq_len:
                source.max_position_embeddings,

            rms_norm_eps:
                source.rms_norm_eps,

            rope_theta:
                source.rope_theta,
        };

    config.validate()?;

    Ok(config)
}
