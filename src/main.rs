mod error;
mod metal;
mod nn;
mod ops;
mod tensor;

use error::Result;
use metal::MetalContext;

use nn::{
    Linear,
    RotaryEmbedding,
    SelfAttention,
};

use tensor::Tensor;

fn main() -> Result<()> {
    let context =
        MetalContext::new();

    // Identity 4x4
    let identity =
        Tensor::from_f32_slice(
            &context,
            &[
                1.0, 0.0, 0.0, 0.0,
                0.0, 1.0, 0.0, 0.0,
                0.0, 0.0, 1.0, 0.0,
                0.0, 0.0, 0.0, 1.0,
            ],
            &[4, 4],
        )?;

    let q_proj =
        Linear::new(
            identity.clone(),
        )?;

    let k_proj =
        Linear::new(
            identity.clone(),
        )?;

    let v_proj =
        Linear::new(
            identity.clone(),
        )?;

    let out_proj =
        Linear::new(
            identity,
        )?;

    let rope =
        RotaryEmbedding::new(
            &context,
            2,          // head_dim
            128,        // max_seq_len
            10_000.0,
        )?;

    let attention =
        SelfAttention::new(
            q_proj,
            k_proj,
            v_proj,
            out_proj,
            rope,
            2,          // num_heads
        )?;

    // sequence = 1
    // hidden = 4
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
        attention.forward(
            &context,
            &input,
        )?;

    println!(
        "hidden_size = {}",
        attention.hidden_size(),
    );

    println!(
        "num_heads   = {}",
        attention.num_heads(),
    );

    println!(
        "head_dim    = {}",
        attention.head_dim(),
    );

    println!(
        "input shape = {:?}",
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
