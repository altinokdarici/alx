//! alx dev CLI: a line-based REPL.
//!
//! This is just one client of the core. The agent itself is wired up in
//! `alx_core::agent_from_settings`; this crate only reads input and renders core
//! events in `CliRenderer`.

use alx_core::protocol::{Event, Message, StopReason};
use alx_core::{
    AgentLoop, EventSink, InMemorySessionStore, ProviderKind, Session, SessionStore, Settings,
};
use clap::Parser;
use std::io::Write;
use std::path::PathBuf;
use std::process::ExitCode;
use tokio::io::{AsyncBufReadExt, BufReader};

/// Command-line flags: the highest-precedence settings layer.
#[derive(Parser)]
#[command(version, about)]
struct Cli {
    /// LLM backend: `anthropic` or `copilot`.
    #[arg(long)]
    provider: Option<ProviderKind>,
    /// Model id for the selected provider.
    #[arg(long)]
    model: Option<String>,
    /// Use this config file instead of `./alx.toml`.
    #[arg(long, value_name = "PATH")]
    config: Option<PathBuf>,
}

/// Plain stdout/stderr rendering of core events.
struct CliRenderer;

impl EventSink for CliRenderer {
    fn emit(&self, event: Event) {
        match event {
            Event::ModelRequest => println!("Making request to the LLM provider"),
            Event::AssistantText { text } => println!("\nAssistant: {text}\n"),
            Event::ToolCallStarted { name, input, .. } => {
                match input.get("input").and_then(|v| v.as_str()) {
                    Some(script) if name == "bash" => println!("$ {script}"),
                    _ => println!("[{name}] {input}"),
                }
            }
            Event::ToolCallFinished {
                output, is_error, ..
            } => {
                if is_error {
                    eprintln!("{output}");
                } else {
                    println!("{output}");
                }
            }
            Event::TurnFinished {
                model: Some(model), ..
            } => println!("[model: {model}]\n"),
            Event::TurnStarted | Event::TurnFinished { .. } => {}
        }
    }
}

/// Process exit codes. `end_turn` is the only "successful" way for a turn to finish.
fn exit_code_for(reason: &StopReason) -> u8 {
    match reason {
        StopReason::EndTurn => 0,
        StopReason::MaxTokens => 2,
        StopReason::StopSequence => 3,
        StopReason::Refusal => 4,
        StopReason::PauseTurn => 5,
        StopReason::ToolUse | StopReason::Other(_) => 1,
    }
}

fn main() -> ExitCode {
    alx_core::load_dotenv();
    let runtime = tokio::runtime::Runtime::new().expect("failed to start tokio runtime");
    match runtime.block_on(run()) {
        Ok(code) => ExitCode::from(code),
        Err(e) => {
            eprintln!("error: {e:#}");
            ExitCode::FAILURE
        }
    }
}

async fn run() -> anyhow::Result<u8> {
    let cli = Cli::parse();
    let settings = Settings::load(cli.config.as_deref())?.merge(Settings {
        provider: cli.provider,
        model: cli.model,
        ..Default::default()
    });
    let agent = alx_core::agent_from_settings(&settings)?;
    let info = agent.provider.info();
    println!(
        "alx · provider: {} · model: {}\n",
        info.name,
        info.model.as_deref().unwrap_or("provider default")
    );
    let store = InMemorySessionStore::default();
    let mut session = Session::new("cli");
    let mut lines = BufReader::new(tokio::io::stdin()).lines();

    loop {
        print!("It's your turn (type 'exit' to quit): ");
        std::io::stdout().flush()?;
        let Some(line) = lines.next_line().await? else {
            break;
        }; // EOF
        let input = line.trim();
        if input == "exit" || input == "quit" {
            break;
        }
        if input.is_empty() {
            continue;
        }

        let stop = match agent
            .run_turn(&mut session, Message::user_text(input), &CliRenderer)
            .await
        {
            Ok(stop) => stop,
            Err(e) => {
                // The session was rolled back, so the user can just try again.
                eprintln!("error: {e:#}");
                continue;
            }
        };
        store.save(&session).await?;

        if stop != StopReason::EndTurn {
            let code = exit_code_for(&stop);
            eprintln!("Unexpected stop_reason \"{stop}\", exiting with code {code}");
            return Ok(code);
        }
    }
    Ok(0)
}
