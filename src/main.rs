mod error;
mod generation;
mod metal;
mod model;
mod nn;
mod ops;
mod tensor;

use error::Result;
use metal::MetalContext;

use model::{
    ModelConfig,
    Transformer,
};

use generation::generate_greedy;

use nn::{
    Embedding,
    Linear,
    Mlp,
    RmsNorm,
    RotaryEmbedding,
    SelfAttention,
    TransformerBlock,
};

use tensor::Tensor;

fn main() -> Result<()> {
    let context =
        MetalContext::new();

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

    // -----------------------------------------------
    // Embedding
    // -----------------------------------------------

    let embedding_weight =
        Tensor::from_f32_slice(
            &context,
            &[
                // token 0
                1.0, 1.0, 1.0, 1.0,

                // token 1
                1.0, 0.0, 0.0, 0.0,

                // token 2
                0.0, 1.0, 0.0, 0.0,

                // token 3
                0.0, 0.0, 1.0, 0.0,
            ],
            &[4, 4],
        )?;

    let embedding =
        Embedding::new(
            embedding_weight,
        )?;

    // -----------------------------------------------
    // Identity projection
    // -----------------------------------------------

    let identity = [
        1.0, 0.0, 0.0, 0.0,
        0.0, 1.0, 0.0, 0.0,
        0.0, 0.0, 1.0, 0.0,
        0.0, 0.0, 0.0, 1.0,
    ];

    let q_proj =
        Linear::new(
            Tensor::from_f32_slice(
                &context,
                &identity,
                &[4, 4],
            )?,
        )?;

    let k_proj =
        Linear::new(
            Tensor::from_f32_slice(
                &context,
                &identity,
                &[4, 4],
            )?,
        )?;

    let v_proj =
        Linear::new(
            Tensor::from_f32_slice(
                &context,
                &identity,
                &[4, 4],
            )?,
        )?;

    let out_proj =
        Linear::new(
            Tensor::from_f32_slice(
                &context,
                &identity,
                &[4, 4],
            )?,
        )?;

    let rope =
        RotaryEmbedding::new(
            &context,
            config.head_dim(),
            config.max_seq_len,
            config.rope_theta,
        )?;

    let attention =
        SelfAttention::new(
            q_proj,
            k_proj,
            v_proj,
            out_proj,
            rope,
            config.num_heads,
        )?;

    // -----------------------------------------------
    // Block RMSNorm
    // -----------------------------------------------

    let attention_norm =
        RmsNorm::new(
            Tensor::from_f32_slice(
                &context,
                &[1.0, 1.0, 1.0, 1.0],
                &[4],
            )?,
            config.rms_norm_eps,
        )?;

    let mlp_norm =
        RmsNorm::new(
            Tensor::from_f32_slice(
                &context,
                &[1.0, 1.0, 1.0, 1.0],
                &[4],
            )?,
            config.rms_norm_eps,
        )?;

    // 테스트에서는 MLP = 0
    let mlp =
        Mlp::new(
            Linear::new(
                Tensor::zeros(
                    &context,
                    &[4, 8],
                )?,
            )?,
            Linear::new(
                Tensor::zeros(
                    &context,
                    &[4, 8],
                )?,
            )?,
            Linear::new(
                Tensor::zeros(
                    &context,
                    &[8, 4],
                )?,
            )?,
        )?;

    let block =
        TransformerBlock::new(
            attention_norm,
            attention,
            mlp_norm,
            mlp,
        )?;

    // -----------------------------------------------
    // Final RMSNorm
    // -----------------------------------------------

    let final_norm =
        RmsNorm::new(
            Tensor::from_f32_slice(
                &context,
                &[1.0, 1.0, 1.0, 1.0],
                &[4],
            )?,
            config.rms_norm_eps,
        )?;

    // -----------------------------------------------
    // LM Head
    //
    // hidden=4 → vocab=4
    //
    // 테스트에서는 identity
    // -----------------------------------------------

    let lm_head =
        Linear::new(
            Tensor::from_f32_slice(
                &context,
                &identity,
                &[4, 4],
            )?,
        )?;

    // -----------------------------------------------
    // Full Transformer
    // -----------------------------------------------

    let model =
        Transformer::new(
            config,
            embedding,
            vec![
                block,
            ],
            final_norm,
            lm_head,
        )?;

    // token 0
    let tokens =
        [0u32];

    let logits =
        model.forward(
            &context,
            &tokens,
        )?;

    let generated = 
        generate_greedy(
            &context, 
            &model, 
            &[0u32], 
            5, 
            None
    )?;

    println!(
        "tokens       = {:?}",
        tokens,
    );

    println!(
        "logits shape = {:?}",
        logits.shape().dims(),
    );

    println!(
        "logits       = {:?}",
        logits.as_f32_slice()?,
    );

    println!(
        "generated = {:?}",
        generated,
    );

    Ok(())
}
