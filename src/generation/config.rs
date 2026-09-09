use crate::error::{Result, TinyError};

#[derive(Debug, Clone)]
pub struct GenerationConfig {
    pub max_new_tokens: usize,
    pub temperature: f32,
    pub top_k: Option<usize>,
    pub top_p: f32,
    pub seed: u64,
}
impl Default for GenerationConfig {
    fn default() -> Self {
        Self {
            max_new_tokens: 64,
            temperature: 0.0,
            top_k: None,
            top_p: 1.0,
            seed: 42,
        }
    }
}
impl GenerationConfig {
    pub fn validate(&self) -> Result<()> {
        if !self.temperature.is_finite() || self.temperature < 0.0 {
            return Err(TinyError::Sampling(
                "temperature must be finite and >= 0".into(),
            ));
        }
        if !self.top_p.is_finite() || self.top_p <= 0.0 || self.top_p > 1.0 {
            return Err(TinyError::Sampling(
                "top_p must satisfy 0 < top_p <= 1".into(),
            ));
        }
        if self.top_k == Some(0) {
            return Err(TinyError::Sampling("top_k must be greater than 0".into()));
        }
        Ok(())
    }
    pub fn is_greedy(&self) -> bool {
        self.temperature == 0.0
    }
}
