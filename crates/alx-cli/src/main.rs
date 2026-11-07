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
    CancelNotification, ContentBlock, ErrorCode, InitializeRequest, ListSessionsRequest,
    NewSessionRequest, PromptRequest, RequestPermissionOutcome, RequestPermissionRequest,
    RequestPermissionResponse, ResumeSessionRequest, SelectedPermissionOutcome, SessionId,
    SessionInfo, SessionNotification, TextContent,
};
use agent_client_protocol::{Agent, Client, ConnectTo, ConnectionTo};
use alx_core::protocol::{Event, StopReason};
use alx_core::{EventSink, FileSessionStore, ProviderKind, SessionStore, Settings};
use clap::Parser;
use client::{CompactOutcome, Translator};
use render::{CliRenderer, Style};
use rustyline::DefaultEditor;
use rustyline::error::ReadlineError;
use std::path::PathBuf;
use std::process::ExitCode;
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

/// `2024-02-29T12:34:56Z` as `2024-02-29 12:34`.
fn short_time(iso: &str) -> String {
    iso.get(..16).unwrap_or(iso).replacen('T', " ", 1)
}

fn session_line(s: &SessionInfo) -> String {
    let count = s
        .meta
        .as_ref()
        .and_then(|m| m.get(alx_acp::MESSAGE_COUNT_META))
        .and_then(|c| c.as_u64())
        .map_or_else(|| "?".to_owned(), |c| c.to_string());
    let cwd = s.cwd.to_string_lossy();
    format!(
        "{}  {}  {count:>3} msgs  {}  {}",
        s.session_id.0,
        s.updated_at.as_deref().map(short_time).unwrap_or_default(),
        if cwd.is_empty() { "-".into() } else { cwd },
        s.title.as_deref().unwrap_or("")
    )
}

/// Which session the REPL works in, or `List` to only list them.
enum Start {
    New,
    Continue,
    Resume(String),
    List,
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
    let store: Arc<dyn SessionStore> = Arc::new(FileSessionStore::new(settings.session_dir()?));
    let agent = alx_core::agent_from_settings(&settings)?;
    if cli.acp {
        // stdout carries the protocol, so nothing else may print to it.
        alx_acp::serve_stdio(Arc::new(agent), store)
            .await
            .map_err(|e| anyhow::anyhow!("{e}"))?;
        return Ok(0);
    }
    let start = if cli.list_sessions {
        Start::List
    } else if let Some(id) = cli.resume {
        Start::Resume(id)
    } else if cli.continue_last {
        Start::Continue
    } else {
        Start::New
    };
    let style = Style::detect();
    let info = agent.provider.info();
    let banner = render::banner(&style, &info.name, info.model.as_deref());
    // The REPL is an ACP client like any editor; this is only the transport.
    let transport = alx_acp::native(Arc::new(agent), store);
    let repl = Repl {
        style,
        banner,
        renderer: Arc::new(CliRenderer::new(style, cli.verbose)),
        translator: Arc::default(),
        compacted: Arc::default(),
    };
    repl.run(transport, start).await
}

/// The REPL, an ACP client of the agent.
struct Repl {
    style: Style,
    /// Shown once the session is open.
    banner: String,
    renderer: Arc<CliRenderer>,
    translator: Arc<Translator>,
    /// The final word on the running `/compact`.
    compacted: Arc<Mutex<Option<CompactOutcome>>>,
}

impl Repl {
    async fn run(
        self,
        transport: impl ConnectTo<Client> + 'static,
        start: Start,
    ) -> anyhow::Result<u8> {
        let (renderer, translator, compacted) = (
            self.renderer.clone(),
            self.translator.clone(),
            self.compacted.clone(),
        );
        Client
            .builder()
            .on_receive_notification(
                async move |n: SessionNotification, _cx| {
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
                Ok(self.session(cx, start).await)
            })
            .await
            .map_err(|e| anyhow::anyhow!("{e}"))?
    }

    async fn session(&self, cx: ConnectionTo<Agent>, start: Start) -> anyhow::Result<u8> {
        let style = &self.style;
        cx.send_request(InitializeRequest::new(ProtocolVersion::V1))
            .block_task()
            .await
            .map_err(acp)?;
        let dir = std::env::current_dir()?;
        let (session, resumed) = match start {
            Start::List => {
                for s in list_sessions(&cx, None).await? {
                    println!("{}", session_line(&s));
                }
                return Ok(0);
            }
            Start::New => {
                let id = cx
                    .send_request(NewSessionRequest::new(dir))
                    .block_task()
                    .await
                    .map_err(acp)?
                    .session_id;
                (id, None)
            }
            Start::Continue => {
                let latest = list_sessions(&cx, Some(dir.clone()))
                    .await?
                    .into_iter()
                    .next();
                let Some(latest) = latest else {
                    anyhow::bail!("no previous session for this directory");
                };
                let (id, count) = resume(&cx, latest.session_id, dir).await?;
                (id, Some(count))
            }
            Start::Resume(id) => {
                let (id, count) = resume(&cx, SessionId::new(id), dir).await?;
                (id, Some(count))
            }
        };
        println!("{}", self.banner);
        if let Some(count) = resumed {
            println!("Resumed session {} ({count} messages)", session.0);
        }
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

fn acp(e: agent_client_protocol::Error) -> anyhow::Error {
    anyhow::anyhow!("{e}")
}

/// Every stored session (only those started in `cwd`, if given), most
/// recent first, following `session/list`'s cursor.
async fn list_sessions(
    cx: &ConnectionTo<Agent>,
    cwd: Option<PathBuf>,
) -> anyhow::Result<Vec<SessionInfo>> {
    let mut all = Vec::new();
    let mut cursor = None;
    loop {
        let page = cx
            .send_request(ListSessionsRequest::new().cwd(cwd.clone()).cursor(cursor))
            .block_task()
            .await
            .map_err(acp)?;
        all.extend(page.sessions);
        match page.next_cursor {
            Some(next) => cursor = Some(next),
            None => return Ok(all),
        }
    }
}

/// Reopens a stored session with `session/resume`, which does not replay
/// its history. Also returns its message count.
async fn resume(
    cx: &ConnectionTo<Agent>,
    id: SessionId,
    cwd: PathBuf,
) -> anyhow::Result<(SessionId, u64)> {
    let reply = cx
        .send_request(ResumeSessionRequest::new(id.clone(), cwd))
        .block_task()
        .await
        .map_err(|e| match e.code {
            ErrorCode::InvalidParams => {
                anyhow::anyhow!("no session with id `{}` (see --list-sessions)", id.0)
            }
            _ => acp(e),
        })?;
    let count = reply
        .meta
        .as_ref()
        .and_then(|m| m.get(alx_acp::MESSAGE_COUNT_META))
        .and_then(|c| c.as_u64())
        .unwrap_or(0);
    Ok((id, count))
}
