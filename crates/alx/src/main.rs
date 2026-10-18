//! alx dev CLI: a line-based REPL.
//!
//! This is just one client of the core. It wires the pluggable parts together
//! in `build_agent` and renders core events in `CliRenderer`.

use alx_core::protocol::{Event, Message, StopReason};
use alx_core::{
    AgentConfig, AgentLoop, AnthropicConfig, AnthropicProvider, BashTool, EventSink,
    InMemorySessionStore, LocalExecutor, Session, SessionStore, SingleAgent, ToolContext,
    ToolRegistry,
};
use anyhow::Context;
use std::io::Write;
use std::process::ExitCode;
use std::sync::Arc;
use tokio::io::{AsyncBufReadExt, BufReader};

/// Swap any part here: another `Provider`, `Executor` (Docker, remote, ...),
/// more tools, or a different `AgentLoop` entirely.
fn build_agent() -> anyhow::Result<impl AgentLoop> {
    let api_key = std::env::var("CLAUDE_API_KEY")
        .context("CLAUDE_API_KEY environment variable is not set")?;
    let mut provider_config = AnthropicConfig::new(api_key);
    if let Ok(model) = std::env::var("ALX_MODEL") {
        provider_config.model = model;
    }
    Ok(SingleAgent {
        config: AgentConfig::default(),
        provider: Arc::new(AnthropicProvider::new(provider_config)),
        tools: ToolRegistry::new().with(BashTool),
        tool_ctx: ToolContext {
            executor: Arc::new(LocalExecutor),
        },
    })
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

/// Minimal `.env` loader: `KEY=VALUE` lines, without overriding variables
/// that are already set.
fn load_dotenv() {
    let Ok(contents) = std::fs::read_to_string(".env") else {
        return;
    };
    for line in contents.lines() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        if let Some((key, value)) = line.split_once('=') {
            let key = key.trim().trim_start_matches("export ").trim();
            let value = value.trim().trim_matches(|c| c == '"' || c == '\'');
            if std::env::var_os(key).is_none() {
                // SAFETY: called at startup before any other threads exist.
                unsafe { std::env::set_var(key, value) };
            }
        }
    }
}

fn main() -> ExitCode {
    load_dotenv();
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
    let agent = build_agent()?;
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
