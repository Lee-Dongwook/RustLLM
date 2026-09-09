use rand::{rngs::StdRng, Rng, SeedableRng};
use crate::{error::{Result, TinyError}, tensor::Tensor};
use super::GenerationConfig;

pub struct Sampler { rng: StdRng }
impl Sampler {
    pub fn new(seed: u64) -> Self { Self { rng: StdRng::seed_from_u64(seed) } }
    pub fn sample(&mut self, logits: &Tensor, config: &GenerationConfig) -> Result<u32> {
        if logits.rank() != 2 || !logits.is_contiguous() { return Err(TinyError::Sampling("sampler requires contiguous rank-2 logits".into())); }
        let vocab = logits.dim(1)?; let data = logits.as_f32_slice()?; self.sample_slice(&data[data.len() - vocab..], config)
    }
    fn sample_slice(&mut self, logits: &[f32], config: &GenerationConfig) -> Result<u32> {
        if config.is_greedy() { return greedy_from_slice(logits); }
        let mut c: Vec<_> = logits.iter().enumerate().filter_map(|(i, &v)| v.is_finite().then_some((i, v / config.temperature))).collect();
        if c.is_empty() { return Err(TinyError::Sampling("no finite logits available".into())); }
        c.sort_by(|a,b| b.1.total_cmp(&a.1)); if let Some(k) = config.top_k { c.truncate(k.min(c.len())); }
        let max = c[0].1; let mut p: Vec<_> = c.into_iter().map(|(i,v)| (i, (v-max).exp())).collect(); let sum: f32 = p.iter().map(|x|x.1).sum(); if !sum.is_finite() || sum <= 0.0 { return Err(TinyError::Sampling("invalid sampling probability sum".into())); } for x in &mut p { x.1 /= sum; }
        if config.top_p < 1.0 { let mut total=0.0; let mut keep=1; for (i, x) in p.iter().enumerate() { total += x.1; if total >= config.top_p { keep=i+1; break; } } p.truncate(keep); }
        let mut draw = self.rng.r#gen::<f32>() * p.iter().map(|x|x.1).sum::<f32>(); for (id, prob) in p { if draw <= prob { return u32::try_from(id).map_err(|_| TinyError::Sampling("token id exceeds u32".into())); } draw -= prob; } Err(TinyError::Sampling("failed to sample a token".into()))
    }
}
fn greedy_from_slice(logits: &[f32]) -> Result<u32> { logits.iter().enumerate().filter(|(_,v)| v.is_finite()).max_by(|a,b| a.1.total_cmp(b.1)).map(|(i,_)| u32::try_from(i).map_err(|_| TinyError::Sampling("token id exceeds u32".into()))).unwrap_or_else(|| Err(TinyError::Sampling("no finite logits available".into()))) }
pub fn greedy_next_token(logits: &Tensor) -> Result<u32> { Sampler::new(0).sample(logits, &GenerationConfig::default()) }
