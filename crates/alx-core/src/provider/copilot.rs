//! GitHub Copilot, through the Copilot SDK and the user's `copilot` CLI.
//!
//! Copilot is an agent runtime, not a stateless completion API. It keeps the
//! conversation itself and runs its own tool loop. To keep alx in charge of
//! the loop, the tools and the executor, this provider runs Copilot as a bare
//! model:
//!
//! - `ClientMode::Empty`, with alx's system prompt replacing Copilot's, and
//!   none of Copilot's built-in tools.
//! - alx's tools are registered *declaration-only*. When the model calls one,
//!   Copilot emits `external_tool.requested` and waits. [`Provider::complete`]
//!   returns that call as a `ToolUse`, alx runs it, and the next `complete`
//!   (carrying the tool results) resolves the pending call so Copilot resumes.
//!
//! Because Copilot holds the history, the provider remembers which alx
//! messages the live Copilot session already reflects. A request that extends
//! them by exactly one user message continues the session. Anything else (a
//! new conversation, a rolled-back turn, a changed system prompt or tool set)
//! starts a fresh Copilot session and replays the prior history as a
//! transcript.

use super::{CompletionRequest, CompletionResponse, Provider};
use alx_protocol::{ContentBlock, Message, ProviderInfo, Role, StopReason, ToolSpec};
use anyhow::{Context, anyhow, bail};
use async_trait::async_trait;
use github_copilot_sdk::rpc::HandlePendingToolCallRequest;
use github_copilot_sdk::session::Session;
use github_copilot_sdk::session_events::{
    AssistantMessageData, ExternalToolRequestedData, SessionErrorData, SessionEventType,
};
use github_copilot_sdk::{
    CliProgram, Client, ClientMode, ClientOptions, EventSubscription, RequestId, SessionConfig,
    SessionEvent, SystemMessageConfig, Tool, ToolSet,
};
use serde_json::Value;
use std::collections::HashMap;
use std::path::PathBuf;
use std::time::Duration;
use tokio::sync::Mutex;
use tokio::time::{Instant, timeout_at};

#[derive(Debug, Clone)]
pub struct CopilotConfig {
    /// Copilot model id (e.g. `gpt-5`, `claude-sonnet-4.5`). `None` uses
    /// Copilot's default.
    pub model: Option<String>,
    /// The `copilot` CLI to run. `None` uses `COPILOT_CLI_PATH`, then
    /// `copilot` on `PATH`.
    pub cli_path: Option<PathBuf>,
    /// The CLI's state directory (`COPILOT_HOME`): sessions and its login.
    /// `None` uses `$COPILOT_HOME`, then `~/.copilot`, so the user's existing
    /// `copilot` login works. alx's sessions then show up in that history.
    pub copilot_home: Option<PathBuf>,
    /// GitHub token to authenticate with. `None` uses the CLI's own login
    /// (`copilot` then `/login`, or `COPILOT_GITHUB_TOKEN` / `GH_TOKEN`).
    pub github_token: Option<String>,
    /// Once one tool call has arrived, how long to wait for the rest of a
    /// parallel batch before handing what we have to the agent loop.
    pub tool_batch_window: Duration,
    /// Upper bound on a single `complete` call.
    pub response_timeout: Duration,
}

impl Default for CopilotConfig {
    fn default() -> Self {
        Self {
            model: None,
            cli_path: None,
            copilot_home: None,
            github_token: None,
            tool_batch_window: Duration::from_millis(250),
            response_timeout: Duration::from_secs(600),
        }
    }
}

pub struct CopilotProvider {
    config: CopilotConfig,
    state: Mutex<State>,
}

#[derive(Default)]
struct State {
    client: Option<Client>,
    live: Option<Live>,
}

/// A Copilot session plus what alx knows about it.
struct Live {
    session: Session,
    events: EventSubscription,
    system: String,
    tools: Vec<ToolSpec>,
    /// The alx messages this Copilot session already reflects.
    seen: Vec<Message>,
    /// Tool calls Copilot is blocked on: alx tool-use id -> Copilot request id.
    pending: HashMap<String, RequestId>,
}

impl CopilotProvider {
    pub fn new(config: CopilotConfig) -> Self {
        Self {
            config,
            state: Mutex::default(),
        }
    }

    async fn client(&self, state: &mut State) -> anyhow::Result<Client> {
        if let Some(client) = &state.client {
            return Ok(client.clone());
        }
        let mut options = ClientOptions::default();
        options.mode = ClientMode::Empty;
        options.program = CliProgram::Path(self.cli_path()?);
        options.base_directory = Some(self.copilot_home()?);
        options.github_token = self.config.github_token.clone();
        let client = Client::start(options)
            .await
            .context("starting the Copilot CLI")?;
        state.client = Some(client.clone());
        Ok(client)
    }

    fn cli_path(&self) -> anyhow::Result<PathBuf> {
        if let Some(path) = &self.config.cli_path {
            return Ok(path.clone());
        }
        if let Some(path) = std::env::var_os("COPILOT_CLI_PATH") {
            return Ok(path.into());
        }
        let path = std::env::var_os("PATH").unwrap_or_default();
        std::env::split_paths(&path)
            .map(|dir| dir.join("copilot"))
            .find(|candidate| candidate.is_file())
            .context(
                "Copilot CLI not found: install it (`npm install -g @github/copilot`) \
                 or set COPILOT_CLI_PATH",
            )
    }

    fn copilot_home(&self) -> anyhow::Result<PathBuf> {
        if let Some(dir) = &self.config.copilot_home {
            return Ok(dir.clone());
        }
        if let Some(dir) = std::env::var_os("COPILOT_HOME") {
            return Ok(dir.into());
        }
        let home = std::env::var_os("HOME").context("HOME is not set; set COPILOT_HOME")?;
        Ok(PathBuf::from(home).join(".copilot"))
    }

    async fn new_session(&self, client: &Client, req: &CompletionRequest) -> anyhow::Result<Live> {
        let tools = req.tools.iter().map(|spec| {
            // No handler: the SDK leaves the call pending for us to resolve.
            Tool::new(&spec.name)
                .with_description(&spec.description)
                .with_parameters(spec.input_schema.clone())
                // Built-ins are off anyway; alx's `bash` etc. take the name.
                .with_overrides_built_in_tool(true)
        });
        let mut config = SessionConfig::default()
            .with_system_message(
                SystemMessageConfig::new()
                    .with_mode("replace")
                    .with_content(&req.system),
            )
            .with_tools(tools)
            // Only alx's tools, none of Copilot's built-ins.
            .with_available_tools(ToolSet::new().add_custom("*")?.into_vec())
            // alx decides what its tools may do; Copilot shouldn't ask.
            .approve_all_permissions();
        if let Some(model) = &self.config.model {
            config = config.with_model(model);
        }
        let session = client
            .create_session(config)
            .await
            .context("creating a Copilot session")?;
        let events = session.subscribe();
        Ok(Live {
            session,
            events,
            system: req.system.clone(),
            tools: req.tools.clone(),
            seen: Vec::new(),
            pending: HashMap::new(),
        })
    }

    async fn complete_inner(
        &self,
        state: &mut State,
        req: CompletionRequest,
    ) -> anyhow::Result<CompletionResponse> {
        let continuable = state
            .live
            .as_ref()
            .filter(|live| live.system == req.system && live.tools == req.tools)
            .map(|live| (live.seen.as_slice(), &live.pending));
        match plan(continuable, &req.messages) {
            Step::Continue { results, text } => {
                let live = state.live.as_mut().expect("plan continues a live session");
                for result in results {
                    let request_id = live
                        .pending
                        .remove(&result.tool_use_id)
                        .expect("plan checked the id is pending");
                    let (result, error) = if result.is_error {
                        (None, Some(result.content))
                    } else {
                        (Some(Value::String(result.content)), None)
                    };
                    live.session
                        .rpc()
                        .tools()
                        .handle_pending_tool_call(HandlePendingToolCallRequest {
                            request_id,
                            result,
                            error,
                        })
                        .await
                        .context("returning a tool result to Copilot")?;
                }
                if let Some(text) = text {
                    live.session
                        .send(text)
                        .await
                        .context("sending to Copilot")?;
                }
            }
            Step::Fresh { prompt } => {
                if let Some(old) = state.live.take() {
                    // Best effort: the old session is abandoned either way.
                    let _ = old.session.disconnect().await;
                }
                let client = self.client(state).await?;
                let live = self.new_session(&client, &req).await?;
                live.session
                    .send(prompt)
                    .await
                    .context("sending to Copilot")?;
                state.live = Some(live);
            }
        }

        let live = state.live.as_mut().expect("a live session after planning");
        let deadline = Instant::now() + self.config.response_timeout;
        let mut collector = Collector::default();
        let stop_reason = loop {
            // Once a tool call is in hand, only wait briefly for the rest of
            // its batch; Copilot may also be waiting on us.
            let wait_until = if collector.has_tool_calls() {
                deadline.min(Instant::now() + self.config.tool_batch_window)
            } else {
                deadline
            };
            let event = match timeout_at(wait_until, live.events.recv()).await {
                Ok(event) => event.context("reading Copilot events")?,
                Err(_) if collector.has_tool_calls() => break StopReason::ToolUse,
                Err(_) => bail!(
                    "Copilot did not respond within {:?}",
                    self.config.response_timeout
                ),
            };
            if let Some(stop) = collector.feed(Incoming::from_event(&event))? {
                break stop;
            }
        };

        let (content, requests, model) = collector.finish();
        live.pending.extend(requests);
        live.seen = req.messages;
        live.seen.push(Message {
            role: Role::Assistant,
            content: content.clone(),
        });
        Ok(CompletionResponse {
            content,
            stop_reason,
            model,
            usage: None,
        })
    }
}

#[async_trait]
impl Provider for CopilotProvider {
    fn info(&self) -> ProviderInfo {
        ProviderInfo {
            name: "copilot".into(),
            model: self.config.model.clone(),
        }
    }

    async fn complete(&self, req: CompletionRequest) -> anyhow::Result<CompletionResponse> {
        let mut state = self.state.lock().await;
        let result = self.complete_inner(&mut state, req).await;
        if result.is_err() {
            // We no longer know what the Copilot side looks like; start over
            // (including the CLI process) next time.
            *state = State::default();
        }
        result
    }
}

// ---------------------------------------------------------------------------
// Pure parts, unit tested without a Copilot CLI.

#[derive(Debug, PartialEq)]
struct PendingResult {
    tool_use_id: String,
    content: String,
    is_error: bool,
}

#[derive(Debug, PartialEq)]
enum Step {
    /// The live session reflects all but the last message; feed it that one.
    Continue {
        results: Vec<PendingResult>,
        text: Option<String>,
    },
    /// Start a new Copilot session with this first message.
    Fresh { prompt: String },
}

/// Decide how to get Copilot to where `messages` say the conversation is.
/// `live` is what the current session reflects and which tool calls it is
/// blocked on, if it can be reused at all.
fn plan(live: Option<(&[Message], &HashMap<String, RequestId>)>, messages: &[Message]) -> Step {
    if let Some((seen, pending)) = live
        && let Some((last, prefix)) = messages.split_last()
        && prefix == seen
        && last.role == Role::User
    {
        let mut results = Vec::new();
        let mut text = Vec::new();
        for block in &last.content {
            match block {
                ContentBlock::ToolResult {
                    tool_use_id,
                    content,
                    is_error,
                } => results.push(PendingResult {
                    tool_use_id: tool_use_id.clone(),
                    content: content.clone(),
                    is_error: *is_error,
                }),
                ContentBlock::Text { text: t } => text.push(t.as_str()),
                ContentBlock::ToolUse { .. } => {}
            }
        }
        // Copilot is blocked until every pending call gets a result, and it
        // can only take results for calls it is actually waiting on.
        let answers_pending = results.len() == pending.len()
            && results.iter().all(|r| pending.contains_key(&r.tool_use_id));
        if answers_pending && (!results.is_empty() || !text.is_empty()) {
            return Step::Continue {
                results,
                text: (!text.is_empty()).then(|| text.join("\n\n")),
            };
        }
    }
    Step::Fresh {
        prompt: replay(messages),
    }
}

/// The first message of a fresh Copilot session. A plain first user message
/// goes as is; any earlier history is replayed as a transcript.
fn replay(messages: &[Message]) -> String {
    if let [only] = messages
        && only.role == Role::User
        && let Some(text) = plain_text(only)
    {
        return text;
    }
    let mut out = String::from(
        "This conversation is continuing from an earlier session. \
         The transcript so far:\n\n",
    );
    for message in messages {
        let who = match message.role {
            Role::User => "User",
            Role::Assistant => "Assistant",
        };
        for block in &message.content {
            match block {
                ContentBlock::Text { text } => out.push_str(&format!("{who}: {text}\n\n")),
                ContentBlock::ToolUse { id, name, input } => out.push_str(&format!(
                    "Assistant called tool `{name}` ({id}): {input}\n\n"
                )),
                ContentBlock::ToolResult {
                    tool_use_id,
                    content,
                    is_error,
                } => {
                    let kind = if *is_error { "error" } else { "result" };
                    out.push_str(&format!("Tool {kind} ({tool_use_id}): {content}\n\n"))
                }
            }
        }
    }
    out.push_str("Continue from here.");
    out
}

fn plain_text(message: &Message) -> Option<String> {
    let texts: Option<Vec<&str>> = message
        .content
        .iter()
        .map(|block| match block {
            ContentBlock::Text { text } => Some(text.as_str()),
            _ => None,
        })
        .collect();
    texts.map(|t| t.join("\n\n"))
}

/// The Copilot events the provider cares about.
#[derive(Debug, Clone, PartialEq)]
enum Incoming {
    /// A finished assistant message, announcing `tool_calls` tool calls.
    Message {
        text: String,
        tool_calls: usize,
        model: Option<String>,
    },
    ToolRequested {
        tool_use_id: String,
        name: String,
        input: Value,
        request_id: RequestId,
    },
    Idle,
    Error(String),
    Other,
}

impl Incoming {
    fn from_event(event: &SessionEvent) -> Self {
        match event.parsed_type() {
            // Sub-agent output is not part of alx's conversation.
            SessionEventType::AssistantMessage if event.agent_id.is_none() => {
                match event.typed_data::<AssistantMessageData>() {
                    Some(data) => Self::Message {
                        text: data.content,
                        tool_calls: data.tool_requests.map_or(0, |t| t.len()),
                        model: data.model,
                    },
                    _ => Self::Other,
                }
            }
            SessionEventType::ExternalToolRequested => {
                match event.typed_data::<ExternalToolRequestedData>() {
                    Some(data) => Self::ToolRequested {
                        tool_use_id: data.tool_call_id,
                        name: data.tool_name,
                        input: data.arguments.unwrap_or(Value::Object(Default::default())),
                        request_id: data.request_id,
                    },
                    None => Self::Error("malformed external_tool.requested event".into()),
                }
            }
            SessionEventType::SessionIdle => Self::Idle,
            SessionEventType::SessionError => Self::Error(
                event
                    .typed_data::<SessionErrorData>()
                    .map_or_else(|| event.data.to_string(), |d| d.message),
            ),
            _ => Self::Other,
        }
    }
}

/// Folds Copilot events into one alx assistant message.
#[derive(Default)]
struct Collector {
    content: Vec<ContentBlock>,
    announced: usize,
    requests: Vec<(String, RequestId)>,
    model: Option<String>,
}

impl Collector {
    fn has_tool_calls(&self) -> bool {
        !self.requests.is_empty()
    }

    /// Returns the stop reason once the response is complete.
    fn feed(&mut self, event: Incoming) -> anyhow::Result<Option<StopReason>> {
        match event {
            Incoming::Message {
                text,
                tool_calls,
                model,
            } => {
                if model.is_some() {
                    self.model = model;
                }
                if !text.is_empty() {
                    self.content.push(ContentBlock::Text { text });
                }
                self.announced += tool_calls;
            }
            Incoming::ToolRequested {
                tool_use_id,
                name,
                input,
                request_id,
            } => {
                self.content.push(ContentBlock::ToolUse {
                    id: tool_use_id.clone(),
                    name,
                    input,
                });
                self.requests.push((tool_use_id, request_id));
                if self.requests.len() >= self.announced {
                    return Ok(Some(StopReason::ToolUse));
                }
            }
            Incoming::Idle if self.has_tool_calls() => return Ok(Some(StopReason::ToolUse)),
            Incoming::Idle => return Ok(Some(StopReason::EndTurn)),
            Incoming::Error(message) => return Err(anyhow!("Copilot error: {message}")),
            Incoming::Other => {}
        }
        Ok(None)
    }

    fn finish(self) -> (Vec<ContentBlock>, Vec<(String, RequestId)>, Option<String>) {
        (self.content, self.requests, self.model)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn user(text: &str) -> Message {
        Message::user_text(text)
    }

    fn assistant(content: Vec<ContentBlock>) -> Message {
        Message {
            role: Role::Assistant,
            content,
        }
    }

    fn tool_use(id: &str) -> ContentBlock {
        ContentBlock::ToolUse {
            id: id.into(),
            name: "bash".into(),
            input: json!({"input": "ls"}),
        }
    }

    fn tool_result(id: &str, is_error: bool) -> ContentBlock {
        ContentBlock::ToolResult {
            tool_use_id: id.into(),
            content: "out".into(),
            is_error,
        }
    }

    fn pending(ids: &[&str]) -> HashMap<String, RequestId> {
        ids.iter()
            .map(|id| (id.to_string(), RequestId::new(format!("req-{id}"))))
            .collect()
    }

    #[test]
    fn first_message_starts_fresh_as_is() {
        assert_eq!(
            plan(None, &[user("hi")]),
            Step::Fresh {
                prompt: "hi".into()
            }
        );
    }

    #[test]
    fn next_user_message_continues() {
        let seen = [
            user("hi"),
            assistant(vec![ContentBlock::Text { text: "yo".into() }]),
        ];
        let mut messages = seen.to_vec();
        messages.push(user("again"));
        assert_eq!(
            plan(Some((&seen, &pending(&[]))), &messages),
            Step::Continue {
                results: vec![],
                text: Some("again".into())
            }
        );
    }

    #[test]
    fn tool_results_resolve_pending_calls() {
        let seen = [user("hi"), assistant(vec![tool_use("a"), tool_use("b")])];
        let mut messages = seen.to_vec();
        messages.push(Message {
            role: Role::User,
            content: vec![tool_result("a", false), tool_result("b", true)],
        });
        let Step::Continue { results, text } =
            plan(Some((&seen, &pending(&["a", "b"]))), &messages)
        else {
            panic!("expected continue");
        };
        assert_eq!(text, None);
        assert_eq!(results.len(), 2);
        assert!(results[1].is_error);
    }

    #[test]
    fn diverged_history_replays_transcript() {
        let seen = [
            user("hi"),
            assistant(vec![ContentBlock::Text { text: "yo".into() }]),
        ];
        // e.g. the agent loop rolled back and retried with different input.
        let messages = [user("hello"), assistant(vec![tool_use("a")]), user("next")];
        let Step::Fresh { prompt } = plan(Some((&seen, &pending(&[]))), &messages) else {
            panic!("expected fresh");
        };
        assert!(prompt.contains("User: hello"));
        assert!(prompt.contains("Assistant called tool `bash` (a)"));
        assert!(prompt.ends_with("User: next\n\nContinue from here."));
    }

    #[test]
    fn unanswered_pending_call_starts_fresh() {
        let seen = [user("hi"), assistant(vec![tool_use("a"), tool_use("b")])];
        let mut messages = seen.to_vec();
        messages.push(Message {
            role: Role::User,
            content: vec![tool_result("a", false)],
        });
        assert!(matches!(
            plan(Some((&seen, &pending(&["a", "b"]))), &messages),
            Step::Fresh { .. }
        ));
    }

    fn requested(id: &str) -> Incoming {
        Incoming::ToolRequested {
            tool_use_id: id.into(),
            name: "bash".into(),
            input: json!({"input": "ls"}),
            request_id: RequestId::new(format!("req-{id}")),
        }
    }

    #[test]
    fn collects_text_until_idle() {
        let mut c = Collector::default();
        let msg = Incoming::Message {
            text: "done".into(),
            tool_calls: 0,
            model: Some("gpt-x".into()),
        };
        assert_eq!(c.feed(msg).unwrap(), None);
        assert_eq!(c.feed(Incoming::Other).unwrap(), None);
        assert_eq!(c.feed(Incoming::Idle).unwrap(), Some(StopReason::EndTurn));
        let (content, requests, model) = c.finish();
        assert_eq!(
            content,
            vec![ContentBlock::Text {
                text: "done".into()
            }]
        );
        assert!(requests.is_empty());
        assert_eq!(model.as_deref(), Some("gpt-x"));
    }

    #[test]
    fn waits_for_every_announced_tool_call() {
        let mut c = Collector::default();
        let msg = Incoming::Message {
            text: "let me look".into(),
            tool_calls: 2,
            model: None,
        };
        c.feed(msg).unwrap();
        assert_eq!(c.feed(requested("a")).unwrap(), None);
        assert!(c.has_tool_calls());
        assert_eq!(c.feed(requested("b")).unwrap(), Some(StopReason::ToolUse));
        let (content, requests, _) = c.finish();
        assert_eq!(content.len(), 3);
        assert_eq!(requests[1].0, "b");
    }

    #[test]
    fn session_error_fails_the_request() {
        let mut c = Collector::default();
        assert!(c.feed(Incoming::Error("quota".into())).is_err());
    }
}
