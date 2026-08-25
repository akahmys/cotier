#![forbid(unsafe_code)]

use clap::{Parser, Subcommand};
use cotier_server::Result;
use std::path::PathBuf;

#[derive(Parser, Debug)]
#[command(name = "cotier")]
#[command(author = "Cotier Authors")]
#[command(version = "0.1.0")]
#[command(about = "Cortical-Tier Recurrent Reasoning Engine (Apple Silicon & Metal Native)", long_about = None)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand, Debug)]
enum Commands {
    /// Start OpenAI/MCP compatible API server
    Serve {
        /// Path to model directory
        #[arg(short, long, default_value = "./models/cotier-0.5b")]
        model: PathBuf,

        /// Server port
        #[arg(short, long, default_value_t = 8080)]
        port: u16,

        /// Enable Apple Silicon Metal GPU acceleration
        #[arg(long, default_value_t = true)]
        metal: bool,
    },
    /// Interactive terminal chat mode with cortical load metrics
    Chat {
        /// Path to model directory
        #[arg(short, long, default_value = "./models/cotier-0.5b")]
        model: PathBuf,
    },
    /// Display memory buffer and consolidation status
    Memory {
        /// Path to model directory
        #[arg(short, long, default_value = "./models/cotier-0.5b")]
        model: PathBuf,
    },
    /// Rollback dynamic LoRA adapter to clean base or previous checkpoint
    Rollback {
        /// Path to model directory
        #[arg(short, long, default_value = "./models/cotier-0.5b")]
        model: PathBuf,

        /// Target snapshot ('base' or version ID)
        #[arg(long, default_value = "base")]
        to: String,
    },
}

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt::init();
    let cli = Cli::parse();

    match cli.command {
        Commands::Serve { model, port, metal } => {
            println!(
                "🚀 Starting Cotier server on port {} (Metal: {})...",
                port, metal
            );
            println!("📂 Model path: {:?}", model);
            // Server initialization logic
        }
        Commands::Chat { model } => {
            println!("💬 Starting interactive chat with model at {:?}...", model);
        }
        Commands::Memory { model } => {
            println!("🧠 Memory status for model at {:?}", model);
        }
        Commands::Rollback { model, to } => {
            println!("⏪ Rolling back model adapter at {:?} to '{}'", model, to);
        }
    }

    Ok(())
}
