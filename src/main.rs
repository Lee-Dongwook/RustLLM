mod error;
mod generation;
mod metal;
mod model;
mod nn;
mod ops;
mod tensor;

use std::fs;

use error::Result;

use generation::generate_greedy;

use metal::MetalContext;

use model::{
    ModelConfig,
    ModelWeights,
    Transformer,
};

fn main() -> Result<()> {
    let context =
        MetalContext::new();

    fs::create_dir_all(
        "models/tiny",
    )?;

    let config =
        ModelConfig {
            vocab_size: 4,
            hidden_size: 4,
            intermediate_size: 8,

            num_layers: 1,
            num_heads: 2,

            max_seq_len: 128,

            rms_norm_eps: 1e-5,
            rope_theta: 10_000.0,
        };

    config.save_json(
        "models/tiny/config.json",
    )?;

    let identity =
        vec![
            1.0, 0.0, 0.0, 0.0,
            0.0, 1.0, 0.0, 0.0,
            0.0, 0.0, 1.0, 0.0,
            0.0, 0.0, 0.0, 1.0,
        ];

    let mut weights =
        ModelWeights::new();

    weights.insert_f32(
        "token_embedding.weight",
        &[4, 4],
        vec![
            1.0, 1.0, 1.0, 1.0,

            1.0, 0.0, 0.0, 0.0,

            0.0, 1.0, 0.0, 0.0,

            0.0, 0.0, 1.0, 0.0,
        ],
    )?;

    weights.insert_f32(
        "layers.0.attention_norm.weight",
        &[4],
        vec![
            1.0,
            1.0,
            1.0,
            1.0,
        ],
    )?;

    weights.insert_f32(
        "layers.0.attention.q_proj.weight",
        &[4, 4],
        identity.clone(),
    )?;

    weights.insert_f32(
        "layers.0.attention.k_proj.weight",
        &[4, 4],
        identity.clone(),
    )?;

    weights.insert_f32(
        "layers.0.attention.v_proj.weight",
        &[4, 4],
        identity.clone(),
    )?;

    weights.insert_f32(
        "layers.0.attention.out_proj.weight",
        &[4, 4],
        identity.clone(),
    )?;

    weights.insert_f32(
        "layers.0.mlp_norm.weight",
        &[4],
        vec![
            1.0,
            1.0,
            1.0,
            1.0,
        ],
    )?;

    weights.insert_f32(
        "layers.0.mlp.gate_proj.weight",
        &[4, 8],
        vec![
            0.0;
            4 * 8
        ],
    )?;

    weights.insert_f32(
        "layers.0.mlp.up_proj.weight",
        &[4, 8],
        vec![
            0.0;
            4 * 8
        ],
    )?;

    weights.insert_f32(
        "layers.0.mlp.down_proj.weight",
        &[8, 4],
        vec![
            0.0;
            8 * 4
        ],
    )?;

    weights.insert_f32(
        "final_norm.weight",
        &[4],
        vec![
            1.0,
            1.0,
            1.0,
            1.0,
        ],
    )?;

    weights.insert_f32(
        "lm_head.weight",
        &[4, 4],
        identity,
    )?;

    weights.save(
        "models/tiny/model.bin",
    )?;

    println!(
        "saved {} tensors",
        weights.len(),
    );

    // --------------------------------------
    // 여기부터 실제 로드
    // --------------------------------------

    let model =
        Transformer::load(
            &context,
            "models/tiny",
        )?;

    println!(
        "loaded model: {:?}",
        model.config(),
    );

    let generated =
        generate_greedy(
            &context,
            &model,
            &[0u32],
            5,
            None,
        )?;

    println!(
        "generated = {:?}",
        generated,
    );

    Ok(())
}
