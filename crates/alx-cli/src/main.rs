//! alx dev CLI: a line-based REPL.
//!
//! This is just one client of the core. The agent itself is wired up in
//! `alx_core::agent_from_settings` and served over ACP by `alx-acp` on its
//! in-process (native) transport; this crate is an ACP client that reads
//! input and renders the updates in `CliRenderer`.

mod client;
mod markdown;
mod render;

use agent_client_protocol::schema::ProtocolVersion;
use agent_client_protocol::schema::v1::{
    CancelNotification, ContentBlock, InitializeRequest, LoadSessionRequest, NewSessionRequest,
    PromptRequest, RequestPermissionOutcome, RequestPermissionRequest, RequestPermissionResponse,
    SelectedPermissionOutcome, SessionId, SessionNotification, TextContent,
};
use agent_client_protocol::{Agent, Client, ConnectTo, ConnectionTo};
use alx_core::protocol::{Event, StopReason};
use alx_core::{
    EventSink, FileSessionStore, ProviderKind, Session, SessionStore, SessionSummary, Settings,
};
use clap::Parser;
use client::{CompactOutcome, Translator};
use render::{CliRenderer, Style};
use rustyline::DefaultEditor;
use rustyline::error::ReadlineError;
use std::path::PathBuf;
use std::process::ExitCode;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};

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
    /// Serve the Agent Client Protocol on stdin/stdout instead of starting
    /// the REPL, so an editor can drive the agent.
    #[arg(long, conflicts_with_all = ["continue_last", "resume", "list_sessions"])]
    acp: bool,
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
        StopReason::Cancelled => 130,
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
    let store: Arc<dyn SessionStore> = Arc::new(store);
    if cli.acp {
        // stdout carries the protocol, so nothing else may print to it.
        alx_acp::serve_stdio(Arc::new(agent), store)
            .await
            .map_err(|e| anyhow::anyhow!("{e}"))?;
        return Ok(0);
    }
    let info = agent.provider.info();
    let style = Style::detect();
    println!(
        "{}",
        render::banner(&style, &info.name, info.model.as_deref())
    );
    let transport = alx_acp::native(Arc::new(agent), store);
    let repl = Repl {
        style,
        renderer: Arc::new(CliRenderer::new(style, cli.verbose)),
        translator: Arc::default(),
        replaying: Arc::default(),
        compacted: Arc::default(),
    };
    repl.run(transport, resumed, cwd).await
}

/// The REPL, an ACP client of the agent.
struct Repl {
    style: Style,
    renderer: Arc<CliRenderer>,
    translator: Arc<Translator>,
    /// `session/load` is replaying history, which is already on screen.
    replaying: Arc<AtomicBool>,
    /// The final word on the running `/compact`.
    compacted: Arc<Mutex<Option<CompactOutcome>>>,
}

impl Repl {
    async fn run(
        self,
        transport: impl ConnectTo<Client> + 'static,
        resumed: Option<Session>,
        cwd: Option<String>,
    ) -> anyhow::Result<u8> {
        let (renderer, translator, replaying, compacted) = (
            self.renderer.clone(),
            self.translator.clone(),
            self.replaying.clone(),
            self.compacted.clone(),
        );
        Client
            .builder()
            .on_receive_notification(
                async move |n: SessionNotification, _cx| {
                    if replaying.load(Ordering::Relaxed) {
                        return Ok(());
                    }
                    if let Some(outcome) = client::compact_outcome(&n.update) {
                        *compacted.lock().unwrap() = Some(outcome);
                    }
                    for event in translator.events(n.update) {
                        renderer.emit(event);
                    }
                    Ok(())
                },
                agent_client_protocol::on_receive_notification!(),
            )
            .on_receive_request(
                // The CLI does not ask before running tools (yet).
                async move |_r: RequestPermissionRequest, responder, _cx| {
                    responder.respond(RequestPermissionResponse::new(
                        RequestPermissionOutcome::Selected(SelectedPermissionOutcome::new(
                            "allow_always",
                        )),
                    ))
                },
                agent_client_protocol::on_receive_request!(),
            )
            .connect_with(transport, async move |cx: ConnectionTo<Agent>| {
                Ok(self.session(cx, resumed, cwd).await)
            })
            .await
            .map_err(|e| anyhow::anyhow!("{e}"))?
    }

    async fn session(
        &self,
        cx: ConnectionTo<Agent>,
        resumed: Option<Session>,
        cwd: Option<String>,
    ) -> anyhow::Result<u8> {
        let acp = |e: agent_client_protocol::Error| anyhow::anyhow!("{e}");
        let style = &self.style;
        cx.send_request(InitializeRequest::new(ProtocolVersion::V1))
            .block_task()
            .await
            .map_err(acp)?;
        let dir = cwd.map_or_else(|| PathBuf::from("."), PathBuf::from);
        let session = match resumed {
            Some(s) => {
                self.replaying.store(true, Ordering::Relaxed);
                let loaded = cx
                    .send_request(LoadSessionRequest::new(s.id.clone(), dir))
                    .block_task()
                    .await;
                self.replaying.store(false, Ordering::Relaxed);
                loaded.map_err(acp)?;
                println!("Resumed session {} ({} messages)", s.id, s.messages.len());
                SessionId::new(s.id)
            }
            None => {
                cx.send_request(NewSessionRequest::new(dir))
                    .block_task()
                    .await
                    .map_err(acp)?
                    .session_id
            }
        };
        let prompt = |text: &str| {
            cx.send_request(PromptRequest::new(
                session.clone(),
                vec![ContentBlock::Text(TextContent::new(text))],
            ))
            .block_task()
        };
        // Line editing (arrows, Home/End, Up/Down history). It reads stdin in
        // the terminal's raw mode only while waiting at the prompt, so Ctrl-C
        // during a turn still arrives as a signal. Piped input is read plainly.
        let mut editor = Some(DefaultEditor::new()?);

        loop {
            let prompt_text = render::prompt(style);
            let mut rl = editor.take().expect("editor is put back after each read");
            let (rl, read) = tokio::task::spawn_blocking(move || {
                let read = rl.readline(&prompt_text);
                (rl, read)
            })
            .await?;
            editor = Some(rl);
            let line = match read {
                Ok(line) => line,
                Err(ReadlineError::Eof) => break,
                // Ctrl-C at the prompt exits.
                Err(ReadlineError::Interrupted) => return Ok(130),
                Err(e) => return Err(e.into()),
            };
            let input = line.trim();
            if !input.is_empty()
                && let Some(rl) = editor.as_mut()
            {
                let _ = rl.add_history_entry(input);
            }
            if input == "exit" || input == "quit" {
                break;
            }
            if input.is_empty() {
                continue;
            }
            if input == "/compact" {
                *self.compacted.lock().unwrap() = None;
                if let Err(e) = prompt(input).await {
                    eprintln!("{}", render::error_line(style, &e.to_string()));
                }
                match self.compacted.lock().unwrap().take() {
                    Some(CompactOutcome::NothingToDo) => {
                        println!("{}", style.dim("Nothing to compact yet."))
                    }
                    Some(CompactOutcome::Failed(e)) => {
                        eprintln!("{}", render::error_line(style, &e))
                    }
                    Some(CompactOutcome::Compacted) | None => {}
                }
                continue;
            }

            self.renderer.emit(Event::TurnStarted);
            self.renderer.emit(Event::ModelRequest);
            // Ctrl-C cancels the running turn; a second one gives up waiting
            // for it to wind down and exits.
            let mut cancelled = false;
            let result = {
                let turn = prompt(input);
                tokio::pin!(turn);
                loop {
                    tokio::select! {
                        result = &mut turn => break result,
                        _ = tokio::signal::ctrl_c() => {
                            if cancelled {
                                println!();
                                return Ok(130);
                            }
                            cancelled = true;
                            cx.send_notification(CancelNotification::new(session.clone()))
                                .map_err(acp)?;
                        }
                    }
                }
            };
            let reply = match result {
                Ok(reply) => reply,
                Err(e) => {
                    // The session was rolled back, so the user can just try again.
                    eprintln!("{}", render::error_line(style, &e.to_string()));
                    continue;
                }
            };
            let finished = client::turn_finished(&reply);
            self.renderer.emit(finished);

            let stop = client::stop_reason(reply.stop_reason);
            // The turn was cancelled, which is not a reason to leave the REPL.
            if stop == StopReason::Cancelled {
                continue;
            }
            if stop != StopReason::EndTurn {
                let code = exit_code_for(&stop);
                eprintln!("Unexpected stop_reason \"{stop}\", exiting with code {code}");
                return Ok(code);
            }
        }
        Ok(0)
    }
}
