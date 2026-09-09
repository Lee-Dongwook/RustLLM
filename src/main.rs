mod cli;
mod commands;

use clap::Parser;

use cli::Cli;

fn main() {
    let cli = 
        Cli::parse();
    
    if let Err(error) =
        commands::execute(cli) {
            eprintln!(
                "error: {error}"
            );
        std::process::exit(1);
    }
}
