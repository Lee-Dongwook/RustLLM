mod error;
mod generation;
mod import;
mod metal;
mod model;
mod nn;
mod ops;
mod tensor;
mod tokenizer;

use error::Result;
use metal::MetalContext;
use model::Transformer;

use crate::error::TinyError;

fn main() -> Result<()> {
    let context =
        MetalContext::new();

    println!(
        "loading real model..."
    );

    let model =
        Transformer::load(
            &context,
            "models/tinystories-llama-15m",
        )?;

    println!(
        "model loaded successfully"
    );

    println!(
        "vocab_size  = {}",
        model.config().vocab_size,
    );

    println!(
        "hidden_size = {}",
        model.config().hidden_size,
    );

    println!(
        "num_layers  = {}",
        model.config().num_layers,
    );

    println!(
        "num_heads   = {}",
        model.config().num_heads,
    );

    println!(
        "head_dim    = {}",
        model.config().head_dim(),
    );

    let token_ids =
        vec![
            1u32,
            42u32,
        ];

    println!(
        "running forward with token ids = {:?}",
        token_ids,
    );

    let logits =
        model.forward(
            &context,
            &token_ids,
        )?;

    println!("forward completed");

    println!(
        "logits shape = {:?}",
        logits.shape().dims(),
    );

    let values =
        logits.as_f32_slice()?;

    let vocab_size =
        model.config().vocab_size;

    println!(
        "logic count = {}",
        values.len(),
    );

    let non_finite_count = 
        values.iter().filter(|value| !value.is_finite()).count();

    println!(
        "non-finite logits = {}",
        non_finite_count,
    );

    let min_logit = values.iter().copied().fold(f32::INFINITY, f32::min);
    let max_logit = values.iter().copied().fold(f32::NEG_INFINITY, f32::max,);

    println!(
        "logit range = [{min_logit}, {max_logit}]",
    );

    let last_token_logits = &values[values.len() - vocab_size..];

    let (next_token_id, next_token_logit) = last_token_logits
        .iter()
        .copied()
        .enumerate()
        .max_by(
            |left, right| {
                left.1.total_cmp(
                    &right.1,
                )
            },
        )
        .ok_or_else(|| {
            TinyError::Sampling(
                "empty logits".to_string(),
            )
        })?;

        println!(
            "argmax token = {}",
            next_token_id,
        );
        
        println!(
            "argmax logit = {}",
            next_token_logit,
        );

        let mut top_logits:
            Vec<(usize, f32)> =
            last_token_logits
                .iter()
                .copied()
                .enumerate()
                .collect();

        top_logits.sort_by(
            |left, right| {
                right.1
                    .total_cmp(
                        &left.1,
                    )
            },
        );

        println!("top 5 logits:");

        for(token_id, logit,) in top_logits.iter().take(5) {
            println!("token {:5} => {}", token_id, logit,);
        }
        
    Ok(())
}
