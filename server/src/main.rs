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
    /// Evaluate forward logits for parity verification
    EvalLogits {
        /// Path to model directory
        #[arg(short, long, default_value = "./models/cotier-0.5b")]
        model: PathBuf,

        /// Comma-separated token IDs
        #[arg(short, long)]
        tokens: String,

        /// Optional path to save output logits binary/JSON
        #[arg(short, long)]
        out: Option<PathBuf>,
    },
}

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt::init();
    let cli = Cli::parse();

    match cli.command {
        Commands::Serve { model, port, metal } => {
            println!(
                "🚀 Initializing Cotier Server on port {} (Metal GPU: {})...",
                port, metal
            );
            println!("📂 Model directory: {:?}", model);

            let device = if metal {
                candle_core::Device::new_metal(0).unwrap_or(candle_core::Device::Cpu)
            } else {
                candle_core::Device::Cpu
            };
            println!("⚡ Compute backend: {:?}", device);

            let cortical_model = cotier_server::model::CorticalModel::load(&model, &device)?;
            let tokenizer_path = model.join("tokenizer.json");
            let tokenizer = tokenizers::Tokenizer::from_file(&tokenizer_path)
                .map_err(|e| cotier_server::CotierError::ModelLoad(e.to_string()))?;

            let db_path = model.join("memory.sqlite");
            let memory = cotier_server::memory::EpisodeMemory::open(&db_path)?;

            let memory_arc = std::sync::Arc::new(tokio::sync::Mutex::new(memory));
            let learner_arc = std::sync::Arc::new(cotier_server::learner::SleepLearner::new());

            // Spawn background sleep consolidation worker (5 min idle threshold)
            cotier_server::learner::spawn_sleep_worker(
                learner_arc.clone(),
                memory_arc.clone(),
                model.clone(),
                300,
            );

            let state = cotier_server::server::AppState {
                model: std::sync::Arc::new(cortical_model),
                tokenizer: std::sync::Arc::new(tokenizer),
                memory: memory_arc,
                learner: learner_arc,
            };

            let app = cotier_server::server::create_router(state);
            let addr = format!("0.0.0.0:{}", port);
            println!(
                "🌐 Cotier OpenAI/MCP API ready at http://localhost:{}",
                port
            );

            let listener = tokio::net::TcpListener::bind(&addr).await?;
            axum::serve(listener, app).await?;
        }
        Commands::Chat { model } => {
            println!("============================================================");
            println!("💬 Cotier Interactive Terminal Chat (0.45B Latent Reasoning)");
            println!("📂 Model: {:?}", model);
            println!("💡 Commands: /exit to quit | /stats for memory | /+1 or /-1 for feedback");
            println!("============================================================");

            let device = candle_core::Device::new_metal(0).unwrap_or(candle_core::Device::Cpu);
            println!("⚡ Compute backend: {:?}", device);

            let cortical_model = cotier_server::model::CorticalModel::load(&model, &device)?;
            let tokenizer_path = model.join("tokenizer.json");
            let tokenizer = tokenizers::Tokenizer::from_file(&tokenizer_path)
                .map_err(|e| cotier_server::CotierError::ModelLoad(e.to_string()))?;

            let db_path = model.join("memory.sqlite");
            let mut memory = cotier_server::memory::EpisodeMemory::open(&db_path)?;

            let mut conversation_history: Vec<(String, String)> = Vec::new();
            let mut last_episode_id: Option<i64> = None;

            let stdin = std::io::stdin();
            let mut stdout = std::io::stdout();
            use std::io::Write;

            loop {
                print!("\n🧑 You: ");
                stdout.flush()?;

                let mut input = String::new();
                if stdin.read_line(&mut input)? == 0 {
                    break;
                }
                let input = input.trim();
                if input.is_empty() {
                    continue;
                }

                if input == "/exit" || input == "exit" {
                    println!("👋 Exiting Cotier. Have a great day!");
                    break;
                } else if input == "/stats" {
                    let stats = memory.get_stats()?;
                    println!(
                        "📊 Episodes: {} | Unconsolidated: {} | +1 Count: {}",
                        stats.total_episodes,
                        stats.unconsolidated_episodes,
                        stats.positive_feedback_count
                    );
                    continue;
                } else if input == "/+1" || input == "/-1" {
                    if let Some(ep_id) = last_episode_id {
                        let score = if input == "/+1" { 1 } else { -1 };
                        memory.update_feedback(ep_id, score)?;
                        println!("✅ Recorded feedback ({}) for episode #{}", score, ep_id);
                    } else {
                        println!("⚠️ No recent episode to rate.");
                    }
                    continue;
                }

                // Construct ChatML prompt
                let mut prompt = String::new();
                for (u, a) in &conversation_history {
                    prompt.push_str(&format!(
                        "<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n{}<|im_end|>\n",
                        u, a
                    ));
                }
                prompt.push_str(&format!(
                    "<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n",
                    input
                ));

                let encoding = tokenizer
                    .encode(prompt.clone(), true)
                    .map_err(|e| cotier_server::CotierError::Inference(e.to_string()))?;
                let prompt_ids = encoding.get_ids();
                let eos_token_id = tokenizer.token_to_id("<|im_end|>").unwrap_or(1);

                let config = cotier_server::engine::GenerationConfig {
                    max_new_tokens: 256,
                    temperature: 0.0,
                    top_p: 1.0,
                    eos_token_id,
                };

                let engine =
                    cotier_server::engine::GenerationEngine::new(&cortical_model, &tokenizer);
                let mut iterator = engine.stream(prompt_ids, &config)?;

                print!("🤖 Cotier: ");
                stdout.flush()?;

                let mut full_response = String::new();
                let mut total_surprise = 0.0_f32;
                let mut token_count = 0;
                let mut max_cycles = 1;

                while let Some(step) = iterator.step()? {
                    if step.is_eos {
                        break;
                    }
                    print!("{}", step.token_text);
                    stdout.flush()?;
                    full_response.push_str(&step.token_text);
                    total_surprise += step.surprise;
                    token_count += 1;
                    if step.cycles > max_cycles {
                        max_cycles = step.cycles;
                    }
                }
                println!();

                let avg_surprise = if token_count > 0 {
                    total_surprise / token_count as f32
                } else {
                    0.0_f32
                };

                // Save to hippocampal memory
                let session_id = "cli_chat";
                let ep_id = memory.save_episode(
                    session_id,
                    input,
                    &full_response,
                    avg_surprise,
                    max_cycles,
                    0,
                )?;
                last_episode_id = Some(ep_id);

                println!(
                    "   [🧠 Cycles: 1 | ⚡ Avg Surprise: {:.3} | 💾 Episode #{}]",
                    avg_surprise, ep_id
                );
                conversation_history.push((input.to_string(), full_response));
            }
        }
        Commands::Memory { model } => {
            let db_path = model.join("memory.sqlite");
            let memory = cotier_server::memory::EpisodeMemory::open(&db_path)?;
            let stats = memory.get_stats()?;

            println!("============================================================");
            println!("🧠 Cotier Hippocampal Episode Memory Buffer Status");
            println!("📂 Database: {:?}", db_path);
            println!("============================================================");
            println!("📊 Total Recorded Episodes:       {}", stats.total_episodes);
            println!(
                "⏳ Unconsolidated (Pending Sleep): {}",
                stats.unconsolidated_episodes
            );
            println!(
                "✅ Consolidated into Plastic LRA:  {}",
                stats.consolidated_episodes
            );
            println!(
                "👍 Positive Feedback (+1) Count:   {}",
                stats.positive_feedback_count
            );
            println!(
                "⚡ Average System Surprise:        {:.4}",
                stats.avg_system_surprise
            );
            println!("============================================================");
        }
        Commands::Rollback { model, to } => {
            println!("⏪ Rolling back model adapter at {:?} to '{}'", model, to);
            let db_path = model.join("memory.sqlite");
            if db_path.exists() {
                let conn = rusqlite::Connection::open(&db_path)?;
                if to == "base" {
                    conn.execute("UPDATE episodes SET is_consolidated = 0", [])?;
                    println!("✅ Reset all hippocampal consolidation flags to clean base state.");
                }
            }
            println!("✅ Rollback complete.");
        }
        Commands::EvalLogits { model, tokens, out } => {
            let token_ids: std::result::Result<Vec<u32>, _> =
                tokens.split(',').map(|s| s.trim().parse::<u32>()).collect();
            let token_ids =
                token_ids.map_err(|e| cotier_server::CotierError::Inference(e.to_string()))?;

            let device = candle_core::Device::Cpu;
            let cortical_model = cotier_server::model::CorticalModel::load(&model, &device)?;

            let seq_len = token_ids.len();
            let input_tensor = candle_core::Tensor::from_vec(token_ids, (1, seq_len), &device)?;
            let logits = cortical_model.forward(&input_tensor)?;
            let logits_vec = logits.to_vec3::<f32>()?;

            if let Some(out_path) = out {
                let json_data = serde_json::to_string(&logits_vec)?;
                std::fs::write(out_path, json_data)?;
            } else {
                println!("{}", serde_json::to_string(&logits_vec)?);
            }
        }
    }

    Ok(())
}
