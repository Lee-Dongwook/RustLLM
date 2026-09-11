use std::path::PathBuf;

use clap::{Args, Parser, Subcommand, ValueEnum};

#[derive(Clone, Copy, Debug, ValueEnum)]
pub enum DTypeArg {
    F32,
    F16,
}

#[derive(Debug, Parser)]
#[command(
    name = "tiny-metal-llm",
    version,
    about = "A tiny LLM runtime powered by Rust and Apple Metal"
)]
pub struct Cli {
    #[command(subcommand)]
    pub command: Command,
}

#[derive(Debug, Subcommand)]
pub enum Command {
    Run(RunArgs),

    Import(ImportArgs),

    Inspect(InspectArgs),
}

#[derive(Debug, Args)]
pub struct RunArgs {
    #[arg(long, short = 'm')]
    pub model: PathBuf,

    #[arg(long, short = 'p')]
    pub prompt: String,

    #[arg(long, value_enum, default_value_t = DTypeArg::F32)]
    pub dtype: DTypeArg,

    #[arg(long, default_value_t = 64)]
    pub max_tokens: usize,

    #[arg(long, default_value_t = 0.0)]
    pub temperature: f32,
    #[arg(long)]
    pub top_k: Option<usize>,
    #[arg(long, default_value_t = 1.0)]
    pub top_p: f32,
    #[arg(long, default_value_t = 42)]
    pub seed: u64,
    #[arg(long, default_value_t = false)]
    pub metrics: bool,

    /// Suppress streamed output and print load, prefill, and decode timings.
    #[arg(long)]
    pub benchmark: bool,

    /// Emit benchmark metrics as one JSON object (also enables benchmark mode).
    #[arg(long)]
    pub benchmark_json: bool,

    /// Accumulate coarse timings for decode-only Transformer operations.
    #[arg(long)]
    pub profile_decode: bool,
}

#[derive(Debug, Args)]
pub struct ImportArgs {
    #[arg(long, short = 's')]
    pub source: PathBuf,

    #[arg(long, short = 'o')]
    pub output: PathBuf,

    #[arg(long, value_enum, default_value_t = WeightFormat::F16)]
    pub weight_format: WeightFormat,
}

#[derive(Debug, Args)]
pub struct InspectArgs {
    #[arg(long, short = 'm')]
    pub model: PathBuf,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum)]
pub enum WeightFormat {
    F32,
    F16,
    Int8,
}
