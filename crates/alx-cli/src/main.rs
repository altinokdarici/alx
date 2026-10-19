//! alx dev CLI: a line-based REPL.
//!
//! This is just one client of the core. The agent itself is wired up in
//! `alx_core::agent_from_settings`; this crate only reads input and renders core
//! events in `CliRenderer`.

mod render;

use alx_core::protocol::{Message, StopReason};
use alx_core::{
    AgentLoop, FileSessionStore, ProviderKind, Session, SessionStore, SessionSummary, Settings,
};
use clap::Parser;
use render::{CliRenderer, Style};
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
    /// Directory to store sessions in.
    #[arg(long, value_name = "DIR")]
    session_dir: Option<PathBuf>,
    /// Resume the most recent session started in this directory.
    #[arg(short = 'c', long = "continue", conflicts_with_all = ["resume", "list_sessions"])]
    continue_last: bool,
    /// Resume the session with this id (see --list-sessions).
    #[arg(long, value_name = "ID", conflicts_with = "list_sessions")]
    resume: Option<String>,
    /// Show the full output of tool calls, not just a one-line summary.
    #[arg(short, long)]
    verbose: bool,
    /// List stored sessions and exit.
    #[arg(long)]
    list_sessions: bool,
}

/// Formats a Unix timestamp as UTC `YYYY-MM-DD HH:MM`.
fn format_time(secs: u64) -> String {
    let days = (secs / 86_400) as i64;
    let rem = secs % 86_400;
    // Civil-from-days (Howard Hinnant's algorithm).
    let z = days + 719_468;
    let era = z.div_euclid(146_097);
    let doe = z.rem_euclid(146_097);
    let yoe = (doe - doe / 1_460 + doe / 36_524 - doe / 146_096) / 365;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let m = if mp < 10 { mp + 3 } else { mp - 9 };
    let y = yoe + era * 400 + i64::from(m <= 2);
    format!(
        "{y:04}-{m:02}-{d:02} {:02}:{:02}",
        rem / 3_600,
        rem % 3_600 / 60
    )
}

fn session_line(s: &SessionSummary) -> String {
    format!(
        "{}  {}  {:>3} msgs  {}  {}",
        s.id,
        format_time(s.updated_at),
        s.message_count,
        s.cwd.as_deref().unwrap_or("-"),
        s.preview
    )
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
        session_dir: cli.session_dir,
        ..Default::default()
    });
    let store = FileSessionStore::new(settings.session_dir()?);
    let cwd = std::env::current_dir()
        .ok()
        .map(|p| p.to_string_lossy().into_owned());

    if cli.list_sessions {
        for s in store.list().await? {
            println!("{}", session_line(&s));
        }
        return Ok(0);
    }
    let resumed = if let Some(id) = &cli.resume {
        match store.load(id).await? {
            Some(s) => Some(s),
            None => anyhow::bail!("no session with id `{id}` (see --list-sessions)"),
        }
    } else if cli.continue_last {
        let latest = store.list().await?.into_iter().find(|s| s.cwd == cwd);
        match latest {
            Some(s) => store.load(&s.id).await?,
            None => anyhow::bail!("no previous session for this directory"),
        }
    } else {
        None
    };
    let agent = alx_core::agent_from_settings(&settings)?;
    let info = agent.provider.info();
    let style = Style::detect();
    println!(
        "{}",
        render::banner(&style, &info.name, info.model.as_deref())
    );
    let renderer = CliRenderer::new(style, cli.verbose);
    let mut session = match resumed {
        Some(s) => {
            println!("Resumed session {} ({} messages)", s.id, s.messages.len());
            s
        }
        None => {
            let mut s = Session::new(Session::generate_id());
            s.cwd = cwd;
            s
        }
    };
    let mut lines = BufReader::new(tokio::io::stdin()).lines();

    loop {
        print!("{}", render::prompt(&style));
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
            .run_turn(&mut session, Message::user_text(input), &renderer)
            .await
        {
            Ok(stop) => stop,
            Err(e) => {
                // The session was rolled back, so the user can just try again.
                eprintln!("{}", render::error_line(&style, &format!("{e:#}")));
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
