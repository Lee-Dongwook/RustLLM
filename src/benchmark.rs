use std::time::Duration;

#[derive(Debug, Clone)]
pub struct BenchmarkStats {
    pub model_size_bytes: u64,

    pub load_time: Duration,

    pub prompt_tokens: usize,
    pub prefill_time: Duration,

    pub generated_tokens: usize,
    pub decode_time: Duration,

    pub total_generation_time: Duration,
}

impl BenchmarkStats {
    pub fn prefill_tokens_per_second(&self) -> f64 {
        if self.prefill_time.is_zero() {
            return 0.0;
        }

        self.prompt_tokens as f64 / self.prefill_time.as_secs_f64()
    }

    pub fn decode_tokens_per_second(&self) -> f64 {
        if self.decode_time.is_zero() {
            return 0.0;
        }

        self.generated_tokens as f64 / self.decode_time.as_secs_f64()
    }

    pub fn model_size_mb(&self) -> f64 {
        self.model_size_bytes as f64 / 1024.0 / 1024.0
    }

    pub fn print(&self) {
        println!();
        println!("=== RustLLM Benchmark ===");
        println!("Model size:       {:.2} MB", self.model_size_mb());
        println!(
            "Load time:        {:.2} ms",
            self.load_time.as_secs_f64() * 1_000.0
        );
        println!("Prompt tokens:    {}", self.prompt_tokens);
        println!(
            "Prefill time:     {:.2} ms",
            self.prefill_time.as_secs_f64() * 1_000.0
        );
        println!(
            "Prefill speed:    {:.2} tok/s",
            self.prefill_tokens_per_second()
        );
        println!("Generated tokens: {}", self.generated_tokens);
        println!(
            "Decode time:      {:.2} ms",
            self.decode_time.as_secs_f64() * 1_000.0
        );
        println!(
            "Decode speed:     {:.2} tok/s",
            self.decode_tokens_per_second()
        );
        println!(
            "Generation time:  {:.2} ms",
            self.total_generation_time.as_secs_f64() * 1_000.0
        );
    }
}
