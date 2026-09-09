use crate::error::{
    Result,
    TinyError,
};

use crate::metal::MetalContext;
use crate::tensor::Tensor;
use crate::model::LayerKvCache;

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
    ) -> Result<Self> {
        let hidden_size =
            attention.hidden_size();

        if attention_norm.hidden_size()
            != hidden_size
        {
            return Err(
                TinyError::InvalidShape(
                    format!(
                        "attention RMSNorm hidden size {} does not match attention hidden size {hidden_size}",
                        attention_norm.hidden_size(),
                    ),
                ),
            );
        }

        if mlp_norm.hidden_size()
            != hidden_size
        {
            return Err(
                TinyError::InvalidShape(
                    format!(
                        "MLP RMSNorm hidden size {} does not match hidden size {hidden_size}",
                        mlp_norm.hidden_size(),
                    ),
                ),
            );
        }

        if mlp.input_size()
            != hidden_size
        {
            return Err(
                TinyError::InvalidShape(
                    format!(
                        "MLP input size {} does not match hidden size {hidden_size}",
                        mlp.input_size(),
                    ),
                ),
            );
        }

        if mlp.output_size()
            != hidden_size
        {
            return Err(
                TinyError::InvalidShape(
                    format!(
                        "MLP output size {} does not match hidden size {hidden_size}",
                        mlp.output_size(),
                    ),
                ),
            );
        }

        Ok(Self {
            attention_norm,
            attention,
            mlp_norm,
            mlp,
        })
    }

    pub fn hidden_size(
        &self,
    ) -> usize {
        self.attention
            .hidden_size()
    }

    pub fn forward(
        &self,
        context: &MetalContext,
        input: &Tensor,
    ) -> Result<Tensor> {
        let normalized =
            self.attention_norm
                .forward(
                    context,
                    input,
                )?;

        let attention_output =
            self.attention
                .forward(
                    context,
                    &normalized,
                )?;

        let hidden =
            input.add(
                context,
                &attention_output,
            )?;

        let normalized =
            self.mlp_norm
                .forward(
                    context,
                    &hidden,
                )?;

        let mlp_output =
            self.mlp
                .forward(
                    context,
                    &normalized,
                )?;

        hidden.add(
            context,
            &mlp_output,
        )
    }
    pub fn forward_with_cache(
    &self,
    context: &MetalContext,
    x: &Tensor,
    cache: &mut LayerKvCache,
) -> Result<Tensor> {
    let normalized =
        self.attention_norm
            .forward(
                context,
                x,
            )?;

    let attention =
        self.attention
            .forward_with_cache(
                context,
                &normalized,
                cache,
            )?;

    let hidden =
        x.add(
            context,
            &attention,
        )?;

    let normalized =
        self.mlp_norm
            .forward(
                context,
                &hidden,
            )?;

    let mlp =
        self.mlp
            .forward(
                context,
                &normalized,
            )?;

    hidden.add(
        context,
        &mlp,
    )
}
}
