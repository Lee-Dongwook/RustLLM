mod chat;
mod import;
mod inspect;
mod run;

use tiny_metal_llm::error::Result;

use crate::cli::{Cli, Command};

pub fn execute(cli: Cli) -> Result<()> {
    match cli.command {
        Command::Run(args) => run::execute(args),

        Command::Chat(args) => chat::execute(args),

        Command::Import(args) => import::execute(args),

        Command::Inspect(args) => inspect::execute(args),
    }
}
