//! alx dev CLI: a line-based REPL.
//!
//! This is just one client of the core. Agents are wired up by
//! `alx_core::SettingsAgentFactory` and served over ACP by `alx-acp` on its
//! in-process (native) transport; this crate is an ACP client that reads
//! input and renders the updates in `CliRenderer`.

mod client;
mod commands;
mod markdown;
mod picker;
mod render;

use agent_client_protocol::schema::ProtocolVersion;
use agent_client_protocol::schema::v1::{
    CancelNotification, ContentBlock, ErrorCode, InitializeRequest, ListSessionsRequest, Meta,
    NewSessionRequest, PromptRequest, RequestPermissionOutcome, RequestPermissionRequest,
    RequestPermissionResponse, ResumeSessionRequest, SelectedPermissionOutcome, SessionConfigKind,
    SessionConfigOption, SessionConfigSelectOption, SessionConfigSelectOptions, SessionId,
    SessionInfo, SessionNotification, SessionUpdate, SetSessionConfigOptionRequest, TextContent,
};
use agent_client_protocol::{Agent, Client, ConnectTo, ConnectionTo};
use alx_core::protocol::{Event, StopReason};
use alx_core::{
    AgentFactory, EventSink, FileSessionStore, ProviderKind, SessionStore, Settings,
    SettingsAgentFactory,
};
use clap::Parser;
use client::{CompactOutcome, Translator};
use render::{CliRenderer, Style};
use rustyline::Editor;
use rustyline::error::ReadlineError;
use rustyline::history::DefaultHistory;
use std::path::PathBuf;
use std::process::ExitCode;
use std::sync::{Arc, Mutex};

/// Command-line flags: the highest-precedence settings layer.
#[derive(Parser)]
#[command(version, about)]
struct Cli {
    /// LLM backend for new sessions (`anthropic` or `copilot`). With
    /// --continue or --resume, switches the resumed session to it.
    #[arg(long)]
    provider: Option<ProviderKind>,
    /// Model id for the default provider (see --provider). With --continue
    /// or --resume, switches the resumed session to it.
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

/// Which session the REPL works in, or `List` to only list them. A resumed
/// session can be switched to another provider.
enum Start {
    New,
    Continue(Switch),
    Resume(String, Switch),
    List,
}

/// What to switch a resumed session to, from --provider and --model.
#[derive(Default)]
struct Switch {
    provider: Option<ProviderKind>,
    model: Option<String>,
}

/// One choice of a select config option: its value and display name.
struct Choice {
    value: String,
    name: String,
}

/// The select config option `id`: its current value and the choices.
fn select_option(options: &[SessionConfigOption], id: &str) -> Option<(String, Vec<Choice>)> {
    let option = options.iter().find(|o| &*o.id.0 == id)?;
    let SessionConfigKind::Select(select) = &option.kind else {
        return None;
    };
    let choice = |o: &SessionConfigSelectOption| Choice {
        value: o.value.0.to_string(),
        name: o.name.clone(),
    };
    let choices = match &select.options {
        SessionConfigSelectOptions::Ungrouped(o) => o.iter().map(choice).collect(),
        SessionConfigSelectOptions::Grouped(g) => {
            g.iter().flat_map(|g| &g.options).map(choice).collect()
        }
        _ => Vec::new(),
    };
    Some((select.current_value.0.to_string(), choices))
}

fn model_of(meta: Option<&Meta>) -> Option<String> {
    meta?.get(alx_acp::MODEL_META)?.as_str().map(str::to_owned)
}

/// What the REPL shows about the session's provider.
/// Shows the arrow-key picker when stdin and stdout are terminals.
/// `None`: not interactive (caller falls back to a listing);
/// `Some(None)`: cancelled.
async fn choose(
    title: &str,
    items: Vec<commands::Item>,
    current: Option<&str>,
    style: &Style,
) -> Option<Option<String>> {
    use std::io::IsTerminal;
    if items.is_empty() || !std::io::stdin().is_terminal() || !std::io::stdout().is_terminal() {
        return None;
    }
    let (title, current, color) = (title.to_string(), current.map(str::to_string), style.color);
    tokio::task::spawn_blocking(move || picker::pick(&title, items, current.as_deref(), color))
        .await
        .ok()?
        .ok()
}

struct OnProvider {
    current: String,
    choices: Vec<String>,
    model: Option<String>,
    /// The models the provider offers (may be empty).
    models: Vec<Choice>,
}

impl OnProvider {
    fn new(options: Option<&[SessionConfigOption]>, meta: Option<&Meta>) -> Self {
        let options = options.unwrap_or_default();
        let (current, choices) = select_option(options, alx_acp::PROVIDER_OPTION)
            .unwrap_or_else(|| ("?".into(), Vec::new()));
        let models = select_option(options, alx_acp::MODEL_OPTION)
            .map(|(_, m)| m)
            .unwrap_or_default();
        Self {
            current,
            choices: choices.into_iter().map(|c| c.value).collect(),
            model: model_of(meta),
            models,
        }
    }

    /// The `/model` listing: the current model, then one line per choice.
    fn model_listing(&self) -> String {
        let current = self.model.as_deref().unwrap_or(alx_acp::DEFAULT_MODEL);
        let mut out = format!("model: {current}. Switch with /model <id>.");
        let listed: Vec<_> = self
            .models
            .iter()
            .filter(|m| m.value != alx_acp::DEFAULT_MODEL)
            .collect();
        if listed.is_empty() {
            out.push_str(&format!(" ({} lists no models.)", self.current));
        }
        for m in listed {
            let mark = if m.value == current { '*' } else { ' ' };
            out.push_str(&format!("\n {mark} {}", m.value));
            if m.name != m.value {
                out.push_str(&format!("  {}", m.name));
            }
        }
        out
    }

    /// The models as `(id, name)`, for completion.
    fn model_items(&self) -> Vec<commands::Item> {
        self.models
            .iter()
            .map(|m| (m.value.clone(), m.name.clone()))
            .collect()
    }

    fn provider_items(&self) -> Vec<commands::Item> {
        self.choices
            .iter()
            .map(|p| (p.clone(), p.clone()))
            .collect()
    }

    fn switched_line(&self) -> String {
        format!(
            "Switched to {} · model: {}",
            self.current,
            self.model.as_deref().unwrap_or("provider default")
        )
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
    let switch_to = Switch {
        provider: cli.provider,
        model: cli.model.clone(),
    };
    let settings = Settings::load(cli.config.as_deref())?.merge(Settings {
        provider: cli.provider,
        model: cli.model,
        session_dir: cli.session_dir,
        ..Default::default()
    });
    let store: Arc<dyn SessionStore> = Arc::new(FileSessionStore::new(settings.session_dir()?));
    let agents: Arc<dyn AgentFactory> = Arc::new(SettingsAgentFactory(settings));
    if cli.acp {
        // stdout carries the protocol, so nothing else may print to it.
        alx_acp::serve_stdio(agents, store)
            .await
            .map_err(|e| anyhow::anyhow!("{e}"))?;
        return Ok(0);
    }
    let start = if cli.list_sessions {
        Start::List
    } else if let Some(id) = cli.resume {
        Start::Resume(id, switch_to)
    } else if cli.continue_last {
        Start::Continue(switch_to)
    } else {
        Start::New
    };
    let style = Style::detect();
    // The REPL is an ACP client like any editor; this is only the transport.
    let transport = alx_acp::native(agents, store);
    let repl = Repl {
        style,
        renderer: Arc::new(CliRenderer::new(style, cli.verbose)),
        translator: Arc::default(),
        compacted: Arc::default(),
        known: Arc::default(),
    };
    repl.run(transport, start).await
}

/// The REPL, an ACP client of the agent.
struct Repl {
    style: Style,
    renderer: Arc<CliRenderer>,
    translator: Arc<Translator>,
    /// The final word on the running `/compact`.
    compacted: Arc<Mutex<Option<CompactOutcome>>>,
    /// What the prompt's hints offer (agent commands, models, providers).
    known: commands::SharedKnown,
}

impl Repl {
    async fn run(
        self,
        transport: impl ConnectTo<Client> + 'static,
        start: Start,
    ) -> anyhow::Result<u8> {
        let known = self.known.clone();
        let (renderer, translator, compacted) = (
            self.renderer.clone(),
            self.translator.clone(),
            self.compacted.clone(),
        );
        Client
            .builder()
            .on_receive_notification(
                async move |n: SessionNotification, _cx| {
                    if let SessionUpdate::AvailableCommandsUpdate(u) = &n.update {
                        known.lock().unwrap().agent_commands = u
                            .available_commands
                            .iter()
                            .map(|c| (format!("/{}", c.name), c.description.clone()))
                            .collect();
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
        let opened = match start {
            Start::List => {
                for s in list_sessions(&cx, None).await? {
                    println!("{}", session_line(&s));
                }
                return Ok(0);
            }
            Start::New => {
                let new = cx
                    .send_request(NewSessionRequest::new(dir))
                    .block_task()
                    .await
                    .map_err(acp)?;
                let on = OnProvider::new(new.config_options.as_deref(), new.meta.as_ref());
                (new.session_id, None, on, Switch::default())
            }
            Start::Continue(switch_to) => {
                let latest = list_sessions(&cx, Some(dir.clone()))
                    .await?
                    .into_iter()
                    .next();
                let Some(latest) = latest else {
                    anyhow::bail!("no previous session for this directory");
                };
                let (id, count, on) = resume(&cx, latest.session_id, dir).await?;
                (id, Some(count), on, switch_to)
            }
            Start::Resume(id, switch_to) => {
                let (id, count, on) = resume(&cx, SessionId::new(id), dir).await?;
                (id, Some(count), on, switch_to)
            }
        };
        let (mut session, resumed, mut on, switch_to) = opened;
        if let Some(to) = switch_to.provider.map(|k| k.to_string())
            && to != on.current
        {
            on = set_option(&cx, &session, alx_acp::PROVIDER_OPTION, &to).await?;
        }
        if let Some(to) = switch_to.model
            && on.model.as_deref() != Some(&to)
        {
            on = set_option(&cx, &session, alx_acp::MODEL_OPTION, &to).await?;
        }
        println!(
            "{}",
            render::banner(style, &on.current, on.model.as_deref())
        );
        if let Some(count) = resumed {
            println!("Resumed session {} ({count} messages)", session.0);
        }
        let prompt = |session: &SessionId, text: &str| {
            cx.send_request(PromptRequest::new(
                session.clone(),
                vec![ContentBlock::Text(TextContent::new(text))],
            ))
            .block_task()
        };
        // Line editing (arrows, Home/End, Up/Down history). It reads stdin in
        // the terminal's raw mode only while waiting at the prompt, so Ctrl-C
        // during a turn still arrives as a signal. Piped input is read plainly.
        let mut rl = Editor::<commands::CommandHelper, DefaultHistory>::new()?;
        rl.set_helper(Some(commands::CommandHelper {
            color: style.color,
            known: self.known.clone(),
        }));
        let mut editor = Some(rl);

        loop {
            // The values `/model` and `/provider` complete to, as of now.
            {
                let mut known = self.known.lock().unwrap();
                known.values.insert("/model", on.model_items());
                known.values.insert("/provider", on.provider_items());
            }
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
            if input == "/provider" || input.starts_with("/provider ") {
                let arg = input["/provider".len()..].trim();
                let arg = match arg {
                    "" => match choose("Provider", on.provider_items(), Some(&on.current), style)
                        .await
                    {
                        Some(Some(p)) => p,
                        Some(None) => continue,
                        None => String::new(),
                    },
                    a => a.to_string(),
                };
                match arg.as_str() {
                    "" => println!(
                        "{}",
                        style.dim(&format!(
                            "provider: {} (available: {}). Switch with /provider <name>.",
                            on.current,
                            on.choices.join(", ")
                        ))
                    ),
                    to => match async {
                        let to = commands::resolve(to, &on.provider_items())
                            .map_err(anyhow::Error::msg)?;
                        set_option(&cx, &session, alx_acp::PROVIDER_OPTION, &to).await
                    }
                    .await
                    {
                        Ok(now) => {
                            on = now;
                            println!("{}", style.dim(&on.switched_line()));
                        }
                        Err(e) => eprintln!("{}", render::error_line(style, &format!("{e:#}"))),
                    },
                }
                continue;
            }
            if matches!(input, "/clear" | "/reset" | "/new") {
                match reset(&cx, &on).await {
                    Ok((id, now)) => {
                        session = id;
                        on = now;
                        println!(
                            "{}",
                            style.dim(&format!("Started a new session {}.", session.0))
                        );
                    }
                    Err(e) => eprintln!("{}", render::error_line(style, &format!("{e:#}"))),
                }
                continue;
            }
            if input == "/model" || input.starts_with("/model ") {
                let arg = input["/model".len()..].trim();
                let listed: Vec<_> = on
                    .model_items()
                    .into_iter()
                    .filter(|(v, _)| v != alx_acp::DEFAULT_MODEL)
                    .collect();
                let arg = match arg {
                    "" if !listed.is_empty() => {
                        match choose("Model", listed, on.model.as_deref(), style).await {
                            Some(Some(m)) => m,
                            Some(None) => continue,
                            None => String::new(),
                        }
                    }
                    a => a.to_string(),
                };
                match arg.as_str() {
                    "" => println!("{}", style.dim(&on.model_listing())),
                    to => match async {
                        let to =
                            commands::resolve(to, &on.model_items()).map_err(anyhow::Error::msg)?;
                        set_option(&cx, &session, alx_acp::MODEL_OPTION, &to).await
                    }
                    .await
                    {
                        Ok(now) => {
                            on = now;
                            println!("{}", style.dim(&on.switched_line()));
                        }
                        Err(e) => eprintln!("{}", render::error_line(style, &format!("{e:#}"))),
                    },
                }
                continue;
            }
            if input == "/compact" {
                *self.compacted.lock().unwrap() = None;
                if let Err(e) = prompt(&session, input).await {
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
                let turn = prompt(&session, input);
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

/// Starts a new session in the current directory on the same provider and
/// model as `on`. The old one stays stored and can be resumed.
async fn reset(
    cx: &ConnectionTo<Agent>,
    on: &OnProvider,
) -> anyhow::Result<(SessionId, OnProvider)> {
    let new = cx
        .send_request(NewSessionRequest::new(std::env::current_dir()?))
        .block_task()
        .await
        .map_err(acp)?;
    let id = new.session_id;
    let mut now = OnProvider::new(new.config_options.as_deref(), new.meta.as_ref());
    if now.current != on.current {
        now = set_option(cx, &id, alx_acp::PROVIDER_OPTION, &on.current).await?;
    }
    if now.model != on.model {
        let model = on.model.as_deref().unwrap_or(alx_acp::DEFAULT_MODEL);
        now = set_option(cx, &id, alx_acp::MODEL_OPTION, model).await?;
    }
    Ok((id, now))
}

/// Sets the session's config option `id` (provider or model) to `value`
/// with `session/set_config_option`.
async fn set_option(
    cx: &ConnectionTo<Agent>,
    session: &SessionId,
    id: &'static str,
    value: &str,
) -> anyhow::Result<OnProvider> {
    let reply = cx
        .send_request(SetSessionConfigOptionRequest::new(
            session.clone(),
            id,
            value,
        ))
        .block_task()
        .await
        .map_err(acp)?;
    Ok(OnProvider::new(
        Some(&reply.config_options),
        reply.meta.as_ref(),
    ))
}

/// Reopens a stored session with `session/resume`, which does not replay
/// its history. Also returns its message count and provider.
async fn resume(
    cx: &ConnectionTo<Agent>,
    id: SessionId,
    cwd: PathBuf,
) -> anyhow::Result<(SessionId, u64, OnProvider)> {
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
    let on = OnProvider::new(reply.config_options.as_deref(), reply.meta.as_ref());
    Ok((id, count, on))
}
