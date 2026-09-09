use std::path::PathBuf;

use clap::{
    Args,
    Parser,
    Subcommand,
};

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

#[derive(
    Debug,
    Subcommand,
)]
pub enum Command {
    Run(RunArgs),

    Import(ImportArgs),

    Inspect(InspectArgs),
}

#[derive(Debug, Args)]
pub struct RunArgs {
    #[arg(
        long,
        short = 'm',
    )]
    pub model: PathBuf,

    #[arg(
        long,
        short = 'p',
    )]
    pub prompt: String,

    #[arg(
        long,
        default_value_t = 64,
    )]
    pub max_tokens: usize,

    #[arg(long, default_value_t = 0.0)]
    pub temperature: f32,
    #[arg(long)]
    pub top_k: Option<usize>,
    #[arg(long, default_value_t = 1.0)]
    pub top_p: f32,
    #[arg(long, default_value_t = 42)]
    pub seed: u64,
}

#[derive(Debug, Args)]
pub struct ImportArgs {
    #[arg(
        long,
        short = 's',
    )]
    pub source: PathBuf,

    #[arg(
        long,
        short = 'o',
    )]
    pub output: PathBuf,
}

#[derive(Debug, Args)]
pub struct InspectArgs {
    #[arg(
        long,
        short = 'm',
    )]
    pub model: PathBuf,
}
