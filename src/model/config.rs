use crate::error::{
    Result,
    TinyError,
};

#[derive(
    Debug,
    Clone,
)]
pub struct ModelConfig {
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub intermediate_size: usize,

    pub num_layers: usize,
    pub num_heads: usize,

    pub max_seq_len: usize,

    pub rms_norm_eps: f32,
    pub rope_theta: f32,
}

impl ModelConfig {
    pub fn validate(
        &self,
    ) -> Result<()> {
        if self.vocab_size == 0 {
            return Err(
                TinyError::InvalidShape(
                    "vocab size cannot be zero"
                        .to_string(),
                ),
            );
        }

        if self.hidden_size == 0 {
            return Err(
                TinyError::InvalidShape(
                    "hidden size cannot be zero"
                        .to_string(),
                ),
            );
        }

        if self.intermediate_size == 0 {
            return Err(
                TinyError::InvalidShape(
                    "intermediate size cannot be zero"
                        .to_string(),
                ),
            );
        }

        if self.num_layers == 0 {
            return Err(
                TinyError::InvalidShape(
                    "number of layers cannot be zero"
                        .to_string(),
                ),
            );
        }

        if self.num_heads == 0 {
            return Err(
                TinyError::InvalidShape(
                    "number of attention heads cannot be zero"
                        .to_string(),
                ),
            );
        }

        if self.hidden_size
            % self.num_heads
            != 0
        {
            return Err(
                TinyError::InvalidShape(
                    format!(
                        "hidden size {} must be divisible by num_heads {}",
                        self.hidden_size,
                        self.num_heads,
                    ),
                ),
            );
        }

        let head_dim =
            self.hidden_size
                / self.num_heads;

        if head_dim % 2 != 0 {
            return Err(
                TinyError::InvalidShape(
                    format!(
                        "RoPE requires an even head dimension, got {head_dim}"
                    ),
                ),
            );
        }

        if self.max_seq_len == 0 {
            return Err(
                TinyError::InvalidShape(
                    "max sequence length cannot be zero"
                        .to_string(),
                ),
            );
        }

        if !self.rms_norm_eps.is_finite()
            || self.rms_norm_eps <= 0.0
        {
            return Err(
                TinyError::InvalidShape(
                    format!(
                        "RMSNorm epsilon must be positive and finite, got {}",
                        self.rms_norm_eps,
                    ),
                ),
            );
        }

        if !self.rope_theta.is_finite()
            || self.rope_theta <= 0.0
        {
            return Err(
                TinyError::InvalidShape(
                    format!(
                        "RoPE theta must be positive and finite, got {}",
                        self.rope_theta,
                    ),
                ),
            );
        }

        Ok(())
    }

    pub fn head_dim(
        &self,
    ) -> usize {
        self.hidden_size
            / self.num_heads
    }
}
