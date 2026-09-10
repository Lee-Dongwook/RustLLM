use std::fs;
use std::path::Path;

use serde::{Deserialize, Serialize};

use crate::error::{Result, TinyError};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelConfig {
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub intermediate_size: usize,

    pub num_layers: usize,
    pub num_heads: usize,
    pub num_kv_heads: usize,

    pub max_seq_len: usize,

    pub rms_norm_eps: f32,
    pub rope_theta: f32,
}

impl ModelConfig {
    pub fn validate(&self) -> Result<()> {
        if self.vocab_size == 0 {
            return Err(TinyError::InvalidShape(
                "vocab size cannot be zero".to_string(),
            ));
        }

        if self.hidden_size == 0 {
            return Err(TinyError::InvalidShape(
                "hidden size cannot be zero".to_string(),
            ));
        }

        if self.intermediate_size == 0 {
            return Err(TinyError::InvalidShape(
                "intermediate size cannot be zero".to_string(),
            ));
        }

        if self.num_layers == 0 {
            return Err(TinyError::InvalidShape(
                "number of layers cannot be zero".to_string(),
            ));
        }

        if self.num_heads == 0 {
            return Err(TinyError::InvalidShape(
                "number of attention heads cannot be zero".to_string(),
            ));
        }

        if self.num_kv_heads == 0 {
            return Err(TinyError::InvalidShape(
                "number of key/value attention heads cannot be zero".to_string(),
            ));
        }

        if !self.hidden_size.is_multiple_of(self.num_heads) {
            return Err(TinyError::InvalidShape(format!(
                "hidden size {} must be divisible by num_heads {}",
                self.hidden_size, self.num_heads,
            )));
        }

        let head_dim = self.hidden_size / self.num_heads;

        if !head_dim.is_multiple_of(2) {
            return Err(TinyError::InvalidShape(format!(
                "RoPE requires an even head dimension, got {head_dim}"
            )));
        }

        if self.max_seq_len == 0 {
            return Err(TinyError::InvalidShape(
                "max sequence length cannot be zero".to_string(),
            ));
        }

        if !self.rms_norm_eps.is_finite() || self.rms_norm_eps <= 0.0 {
            return Err(TinyError::InvalidShape(format!(
                "RMSNorm epsilon must be positive and finite, got {}",
                self.rms_norm_eps,
            )));
        }

        if !self.rope_theta.is_finite() || self.rope_theta <= 0.0 {
            return Err(TinyError::InvalidShape(format!(
                "RoPE theta must be positive and finite, got {}",
                self.rope_theta,
            )));
        }

        if !self.num_heads.is_multiple_of(self.num_kv_heads) {
            return Err(TinyError::ModelFormat(format!(
                "num_heads {} must be divisible by num_kv_heads {}",
                self.num_heads, self.num_kv_heads,
            )));
        }

        Ok(())
    }

    pub fn head_dim(&self) -> usize {
        self.hidden_size / self.num_heads
    }

    pub fn q_proj_size(&self) -> usize {
        self.num_heads * self.head_dim()
    }

    pub fn kv_proj_size(&self) -> usize {
        self.num_kv_heads * self.head_dim()
    }

    pub fn num_kv_groups(&self) -> usize {
        self.num_heads / self.num_kv_heads
    }

    pub fn load_json(path: impl AsRef<Path>) -> Result<Self> {
        let text = fs::read_to_string(path)?;

        let config: Self = serde_json::from_str(&text)
            .map_err(|error| TinyError::ModelFormat(format!("invalid config.json: {error}")))?;

        config.validate()?;

        Ok(config)
    }

    pub fn save_json(&self, path: impl AsRef<Path>) -> Result<()> {
        self.validate()?;

        let text = serde_json::to_string_pretty(self).map_err(|error| {
            TinyError::ModelFormat(format!("failed to serialize config: {error}"))
        })?;

        fs::write(path, text)?;

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::ModelConfig;

    fn gqa_config(num_heads: usize, num_kv_heads: usize) -> ModelConfig {
        ModelConfig {
            vocab_size: 100,
            hidden_size: 576,
            intermediate_size: 1536,
            num_layers: 4,
            num_heads,
            num_kv_heads,
            max_seq_len: 1024,
            rms_norm_eps: 1e-5,
            rope_theta: 10_000.0,
        }
    }

    #[test]
    fn validates_gqa_config_and_reports_group_sizes() {
        let config = gqa_config(9, 3);

        config.validate().unwrap();
        assert_eq!(config.head_dim(), 64);
        assert_eq!(config.q_proj_size(), 576);
        assert_eq!(config.kv_proj_size(), 192);
        assert_eq!(config.num_kv_groups(), 3);
    }

    #[test]
    fn rejects_non_divisible_kv_head_count() {
        let mut config = gqa_config(10, 3);
        config.hidden_size = 640;

        let error = config.validate().unwrap_err();

        assert!(
            error
                .to_string()
                .contains("num_heads 10 must be divisible by num_kv_heads 3")
        );
    }
}
