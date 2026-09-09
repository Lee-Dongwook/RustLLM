use std::time::Duration;

#[derive(Debug, Clone)]
pub struct GenerationMetrics {
    pub prompt_tokens: usize,
    pub generated_tokens: usize,
    pub prefill_duration: Duration,
    pub decode_forward_duration: Duration,
    pub decode_forward_calls: usize,
    pub generation_duration: Duration,
    pub total_duration: Duration,
}
impl GenerationMetrics {
    pub fn prefill_tokens_per_second(&self) -> f64 {
        rate(self.prompt_tokens, self.prefill_duration)
    }
    pub fn decode_tokens_per_second(&self) -> f64 {
        rate(self.decode_forward_calls, self.decode_forward_duration)
    }
    pub fn generation_tokens_per_second(&self) -> f64 {
        rate(self.generated_tokens, self.generation_duration)
    }
}
#[derive(Debug)]
pub struct GenerationOutput {
    pub token_ids: Vec<u32>,
    pub metrics: GenerationMetrics,
}
fn rate(tokens: usize, duration: Duration) -> f64 {
    let seconds = duration.as_secs_f64();
    if seconds == 0.0 {
        0.0
    } else {
        tokens as f64 / seconds
    }
}
