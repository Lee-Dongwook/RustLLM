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
