mod error;
mod metal;
mod nn;
mod ops;
mod tensor;

use error::Result;
use metal::MetalContext;
use nn::Embedding;
use tensor::Tensor;

fn main() -> Result<()> {
    let context =
        MetalContext::new();

    // vocab_size = 5
    // hidden_size = 3
    //
    // token 0 -> [ 0,  1,  2]
    // token 1 -> [10, 11, 12]
    // token 2 -> [20, 21, 22]
    // token 3 -> [30, 31, 32]
    // token 4 -> [40, 41, 42]
    let weight =
        Tensor::from_f32_slice(
            &context,
            &[
                0.0, 1.0, 2.0,
                10.0, 11.0, 12.0,
                20.0, 21.0, 22.0,
                30.0, 31.0, 32.0,
                40.0, 41.0, 42.0,
            ],
            &[5, 3],
        )?;

    let embedding =
        Embedding::new(
            weight,
        )?;

    let token_ids =
        [
            3u32,
            1u32,
            4u32,
        ];

    let output =
        embedding.forward(
            &context,
            &token_ids,
        )?;

    println!(
        "vocab_size  = {}",
        embedding.vocab_size(),
    );

    println!(
        "hidden_size = {}",
        embedding.hidden_size(),
    );

    println!(
        "tokens      = {:?}",
        token_ids,
    );

    println!(
        "shape       = {:?}",
        output.shape().dims(),
    );

    println!(
        "output      = {:?}",
        output.as_f32_slice()?,
    );

    Ok(())
}
