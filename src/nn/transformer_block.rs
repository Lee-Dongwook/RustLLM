use crate::error::Result;
use crate::metal::MetalContext;
use crate::tensor::Tensor;

use super::{
    Mlp,
    RmsNorm,
    SelfAttention,
};

pub struct TransformerBlock {
    attention_norm: RmsNorm,
    attention: SelfAttention,

    mlp_norm: RmsNorm,
    mlp: Mlp,
}

impl TransformerBlock {
    pub fn new(
        attention_norm: RmsNorm,
        attention: SelfAttention,
        mlp_norm: RmsNorm,
        mlp: Mlp,
    ) -> Self {
        Self {
            attention_norm,
            attention,
            mlp_norm,
            mlp,
        }
    }

    pub fn forward(
        &self,
        context: &MetalContext,
        input: &Tensor,
    ) -> Result<Tensor> {
        // -----------------------------------------------
        // 1. Attention pre-norm
        // -----------------------------------------------

        let normalized =
            self.attention_norm
                .forward(
                    context,
                    input,
                )?;

        // -----------------------------------------------
        // 2. Self Attention
        // -----------------------------------------------

        let attention_output =
            self.attention
                .forward(
                    context,
                    &normalized,
                )?;

        // -----------------------------------------------
        // 3. First residual
        //
        // x = x + attention(norm(x))
        // -----------------------------------------------

        let hidden =
            input.add(
                context,
                &attention_output,
            )?;

        // -----------------------------------------------
        // 4. MLP pre-norm
        // -----------------------------------------------

        let normalized =
            self.mlp_norm
                .forward(
                    context,
                    &hidden,
                )?;

        // -----------------------------------------------
        // 5. MLP
        // -----------------------------------------------

        let mlp_output =
            self.mlp
                .forward(
                    context,
                    &normalized,
                )?;

        // -----------------------------------------------
        // 6. Second residual
        //
        // y = hidden + mlp(norm(hidden))
        // -----------------------------------------------

        hidden.add(
            context,
            &mlp_output,
        )
    }
}
