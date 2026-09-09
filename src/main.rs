mod error;
mod metal;
mod nn;
mod ops;
mod tensor;

use error::Result;
use metal::MetalContext;

use nn::{
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

    // --------------------------------------------------
    // 공통 Identity 4x4
    // --------------------------------------------------

    let identity_data = [
        1.0, 0.0, 0.0, 0.0,
        0.0, 1.0, 0.0, 0.0,
        0.0, 0.0, 1.0, 0.0,
        0.0, 0.0, 0.0, 1.0,
    ];

    let q_proj =
        Linear::new(
            Tensor::from_f32_slice(
                &context,
                &identity_data,
                &[4, 4],
            )?,
        )?;

    let k_proj =
        Linear::new(
            Tensor::from_f32_slice(
                &context,
                &identity_data,
                &[4, 4],
            )?,
        )?;

    let v_proj =
        Linear::new(
            Tensor::from_f32_slice(
                &context,
                &identity_data,
                &[4, 4],
            )?,
        )?;

    let out_proj =
        Linear::new(
            Tensor::from_f32_slice(
                &context,
                &identity_data,
                &[4, 4],
            )?,
        )?;

    let rope =
        RotaryEmbedding::new(
            &context,
            2,
            128,
            10_000.0,
        )?;

    let attention =
        SelfAttention::new(
            q_proj,
            k_proj,
            v_proj,
            out_proj,
            rope,
            2,
        )?;

    // --------------------------------------------------
    // RMSNorm weights
    // --------------------------------------------------

    let attention_norm =
        RmsNorm::new(
            Tensor::from_f32_slice(
                &context,
                &[1.0, 1.0, 1.0, 1.0],
                &[4],
            )?,
            1e-5,
        )?;

    let mlp_norm =
        RmsNorm::new(
            Tensor::from_f32_slice(
                &context,
                &[1.0, 1.0, 1.0, 1.0],
                &[4],
            )?,
            1e-5,
        )?;

    // --------------------------------------------------
    // MLP는 출력이 무조건 0이 되도록
    // 모든 weight를 0으로 둔다.
    //
    // hidden = 4
    // intermediate = 8
    // --------------------------------------------------

    let gate_proj =
        Linear::new(
            Tensor::zeros(
                &context,
                &[4, 8],
            )?,
        )?;

    let up_proj =
        Linear::new(
            Tensor::zeros(
                &context,
                &[4, 8],
            )?,
        )?;

    let down_proj =
        Linear::new(
            Tensor::zeros(
                &context,
                &[8, 4],
            )?,
        )?;

    let mlp =
        Mlp::new(
            gate_proj,
            up_proj,
            down_proj,
        )?;

    let block =
        TransformerBlock::new(
            attention_norm,
            attention,
            mlp_norm,
            mlp,
        );

    // --------------------------------------------------
    // sequence = 1
    // hidden = 4
    // --------------------------------------------------

    let input =
        Tensor::from_f32_slice(
            &context,
            &[
                1.0,
                2.0,
                3.0,
                4.0,
            ],
            &[1, 4],
        )?;

    let output =
        block.forward(
            &context,
            &input,
        )?;

    println!(
        "input shape  = {:?}",
        input.shape().dims(),
    );

    println!(
        "output shape = {:?}",
        output.shape().dims(),
    );

    println!(
        "input  = {:?}",
        input.as_f32_slice()?,
    );

    println!(
        "output = {:?}",
        output.as_f32_slice()?,
    );

    Ok(())
}
