use std::{collections::BTreeMap, time::Duration};

/// Decode-phase Metal submission counters. Every currently supported tensor
/// operation creates one command buffer, one compute encoder, and performs one
/// dispatch followed by commit/wait.
#[derive(Default, Debug, Clone)]
pub struct MetalDecodeProfile {
    pub command_buffers: usize,
    pub compute_encoders: usize,
    pub kernel_dispatches: usize,
    pub commits: usize,
    pub waits: usize,
    pub kernel_dispatches_by_name: BTreeMap<String, usize>,
}

impl MetalDecodeProfile {
    pub fn record_kernel_dispatch(&mut self, kernel_name: &str) {
        self.kernel_dispatches += 1;
        *self
            .kernel_dispatches_by_name
            .entry(kernel_name.to_owned())
            .or_default() += 1;
    }

    pub fn record_command_buffer(&mut self) {
        self.command_buffers += 1;
    }
    pub fn record_encoder(&mut self) {
        self.compute_encoders += 1;
    }
    pub fn record_commit(&mut self) {
        self.commits += 1;
    }
    pub fn record_wait(&mut self) {
        self.waits += 1;
    }

    pub fn print(&self, profiled_tokens: usize) {
        let per_token = |value: usize| -> f64 {
            if profiled_tokens == 0 {
                0.0
            } else {
                value as f64 / profiled_tokens as f64
            }
        };
        eprintln!("\n=== Metal Decode Profile ===");
        eprintln!(
            "Command buffers:       {} total / {:.2} token",
            self.command_buffers,
            per_token(self.command_buffers)
        );
        eprintln!(
            "Compute encoders:      {} total / {:.2} token",
            self.compute_encoders,
            per_token(self.compute_encoders)
        );
        eprintln!(
            "Kernel dispatches:     {} total / {:.2} token",
            self.kernel_dispatches,
            per_token(self.kernel_dispatches)
        );
        eprintln!(
            "Commits:               {} total / {:.2} token",
            self.commits,
            per_token(self.commits)
        );
        eprintln!(
            "Waits:                 {} total / {:.2} token",
            self.waits,
            per_token(self.waits)
        );
        eprintln!("\nTop dispatched kernels:");
        let mut kernels: Vec<_> = self.kernel_dispatches_by_name.iter().collect();
        kernels.sort_unstable_by(|(left_name, left_count), (right_name, right_count)| {
            right_count
                .cmp(left_count)
                .then_with(|| left_name.cmp(right_name))
        });
        for (name, count) in kernels {
            eprintln!("{name:<30} {count}");
        }
    }
}

#[derive(Default, Debug, Clone)]
pub struct DecodeProfile {
    pub norm: Duration,
    pub qkv_projection: Duration,
    pub attention: Duration,
    pub output_projection: Duration,
    pub mlp_gate_up: Duration,
    pub mlp_activation: Duration,
    pub mlp_down: Duration,
    pub residual: Duration,
    pub final_norm: Duration,
    pub lm_head: Duration,
    pub decode_wall_time: Duration,
    pub sampled_tokens: usize,
    pub metal: MetalDecodeProfile,
}

impl DecodeProfile {
    pub fn profiled_operations(&self) -> Duration {
        self.norm
            + self.qkv_projection
            + self.attention
            + self.output_projection
            + self.mlp_gate_up
            + self.mlp_activation
            + self.mlp_down
            + self.residual
            + self.final_norm
            + self.lm_head
    }

    pub fn record_decode_wall_time(&mut self, elapsed: Duration) {
        self.decode_wall_time += elapsed;
        self.sampled_tokens += 1;
    }

    pub fn unaccounted(&self) -> Duration {
        self.decode_wall_time
            .saturating_sub(self.profiled_operations())
    }

    pub fn percentage(&self, duration: Duration) -> f64 {
        let total = self.profiled_operations().as_secs_f64();
        if total == 0.0 {
            0.0
        } else {
            duration.as_secs_f64() / total * 100.0
        }
    }

    pub fn average_ms(&self, duration: Duration) -> f64 {
        if self.sampled_tokens == 0 {
            0.0
        } else {
            duration.as_secs_f64() * 1_000.0 / self.sampled_tokens as f64
        }
    }

    pub fn print(&self) {
        let total = self.profiled_operations();
        eprintln!("\n=== Decode Profile ===");
        eprintln!("Profiled tokens:      {}", self.sampled_tokens);
        eprintln!();
        for (name, duration) in [
            ("Norm", self.norm),
            ("QKV projection", self.qkv_projection),
            ("Attention", self.attention),
            ("O projection", self.output_projection),
            ("MLP gate/up", self.mlp_gate_up),
            ("MLP activation", self.mlp_activation),
            ("MLP down", self.mlp_down),
            ("Residual", self.residual),
            ("Final norm", self.final_norm),
            ("LM head", self.lm_head),
        ] {
            eprintln!(
                "{name:<20} {:>9.2} ms {:>6.1}% {:>8.2} ms/token",
                duration.as_secs_f64() * 1_000.0,
                self.percentage(duration),
                self.average_ms(duration)
            );
        }
        eprintln!("-------------------------------------------------------");
        eprintln!(
            "Profiled operations:  {:.2} ms",
            total.as_secs_f64() * 1_000.0
        );
        eprintln!(
            "Decode wall time:      {:.2} ms",
            self.decode_wall_time.as_secs_f64() * 1_000.0
        );
        eprintln!(
            "Unaccounted:           {:.2} ms",
            self.unaccounted().as_secs_f64() * 1_000.0
        );
        eprintln!(
            "Average wall time:     {:.2} ms/token",
            self.average_ms(self.decode_wall_time)
        );
    }
}

#[cfg(test)]
mod tests {
    use super::{DecodeProfile, MetalDecodeProfile};
    use std::time::Duration;

    #[test]
    fn accumulates_and_reports_unaccounted_time() {
        let mut profile = DecodeProfile::default();
        profile.qkv_projection += Duration::from_millis(10);
        profile.qkv_projection += Duration::from_millis(20);
        profile.record_decode_wall_time(Duration::from_millis(40));
        assert_eq!(profile.profiled_operations(), Duration::from_millis(30));
        assert_eq!(profile.unaccounted(), Duration::from_millis(10));
        assert_eq!(profile.average_ms(profile.qkv_projection), 30.0);
        assert_eq!(profile.percentage(profile.qkv_projection), 100.0);
    }

    #[test]
    fn zero_tokens_do_not_divide_by_zero() {
        let profile = DecodeProfile::default();
        assert_eq!(profile.average_ms(Duration::from_secs(1)), 0.0);
        assert_eq!(profile.percentage(Duration::from_secs(1)), 0.0);
    }

    #[test]
    fn metal_submission_counter_accumulates_by_kernel() {
        let mut profile = MetalDecodeProfile::default();
        profile.record_command_buffer();
        profile.record_encoder();
        profile.record_kernel_dispatch("linear");
        profile.record_kernel_dispatch("linear");
        profile.record_kernel_dispatch("softmax");
        profile.record_commit();
        profile.record_wait();
        assert_eq!(profile.command_buffers, 1);
        assert_eq!(profile.compute_encoders, 1);
        assert_eq!(profile.kernel_dispatches, 3);
        assert_eq!(profile.commits, 1);
        assert_eq!(profile.waits, 1);
        assert_eq!(profile.kernel_dispatches_by_name["linear"], 2);
    }
}
