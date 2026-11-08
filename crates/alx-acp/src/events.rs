//! alx `Event`s to ACP `session/update` notifications.

use agent_client_protocol::schema::v1::{
    CompactionStatus, CompactionUpdate, ContentBlock, ContentChunk, Meta, SessionId,
    SessionNotification, SessionUpdate, TextContent, ToolCall, ToolCallLocation, ToolCallStatus,
    ToolCallUpdate, ToolCallUpdateFields, ToolKind, UsageUpdate,
};
use agent_client_protocol::{Client, ConnectionTo, Error};
use alx_core::EventSink;
use alx_core::protocol::{Compaction, ContentBlock as AlxBlock, Event, Message, Role, Usage};
use std::sync::Mutex;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};

pub struct AcpSink {
    cx: ConnectionTo<Client>,
    session: SessionId,
    /// Whether this turn already streamed text, so the whole `AssistantText`
    /// that follows would repeat it.
    streamed: AtomicBool,
    /// Token usage of the last finished turn, for the `session/prompt` reply.
    usage: Mutex<Option<Usage>>,
    /// The model that answered the last finished turn.
    model: Mutex<Option<String>>,
}

impl AcpSink {
    pub fn new(cx: ConnectionTo<Client>, session: SessionId) -> Self {
        Self {
            cx,
            session,
            streamed: AtomicBool::new(false),
            usage: Mutex::new(None),
            model: Mutex::new(None),
        }
    }

    pub fn usage(&self) -> Option<Usage> {
        *self.usage.lock().unwrap()
    }

    pub fn model(&self) -> Option<String> {
        self.model.lock().unwrap().clone()
    }

    pub fn send(&self, update: SessionUpdate) {
        // A gone client is not the agent's problem.
        let _ = self
            .cx
            .send_notification(SessionNotification::new(self.session.clone(), update));
    }
}

fn chunk(text: &str) -> ContentChunk {
    ContentChunk::new(ContentBlock::Text(TextContent::new(text)))
}

/// The one-line title shown for a tool call.
pub fn tool_title(name: &str, input: &serde_json::Value) -> String {
    let arg = |key: &str| input.get(key).and_then(|v| v.as_str());
    match (name, arg("input"), arg("path")) {
        ("bash", Some(cmd), _) => cmd.lines().next().unwrap_or("").to_string(),
        ("read_file", _, Some(path)) => format!("Read {path}"),
        ("write_file", _, Some(path)) => format!("Write {path}"),
        ("edit_file", _, Some(path)) => format!("Edit {path}"),
        ("grep" | "glob", _, path) => match arg("pattern") {
            Some(p) => format!("Search {p} in {}", path.unwrap_or(".")),
            None => name.to_string(),
        },
        ("list_dir", _, path) => format!("List {}", path.unwrap_or(".")),
        _ => name.to_string(),
    }
}

pub fn tool_kind(name: &str) -> ToolKind {
    match name {
        "bash" => ToolKind::Execute,
        "read_file" => ToolKind::Read,
        "write_file" | "edit_file" => ToolKind::Edit,
        "grep" | "glob" | "list_dir" => ToolKind::Search,
        _ => ToolKind::Other,
    }
}

/// The files a tool call touches, so a client can follow along.
fn tool_locations(input: &serde_json::Value) -> Vec<ToolCallLocation> {
    input
        .get("path")
        .and_then(|p| p.as_str())
        .map(|p| vec![ToolCallLocation::new(p)])
        .unwrap_or_default()
}

/// A finished compaction, as a `compaction_update`. The counts alx knows go
/// in `_meta`, since ACP has no fields for them.
pub fn compaction_update(id: String, c: &Compaction) -> SessionUpdate {
    let mut meta = Meta::new();
    meta.insert("messagesRemoved".into(), c.messages_removed.into());
    if let Some(t) = c.tokens_before {
        meta.insert("tokensBefore".into(), t.into());
    }
    if let Some(t) = c.tokens_after {
        meta.insert("tokensAfter".into(), t.into());
    }
    SessionUpdate::CompactionUpdate(
        CompactionUpdate::new(id, CompactionStatus::Completed).meta(meta),
    )
}

/// A retry notice, as a thought chunk (ACP has nothing closer). The numbers
/// go in `_meta` under `alx/retrying`, so alx's own client can read them back.
pub fn retrying_update(
    attempt: u32,
    max_attempts: u32,
    delay_ms: u64,
    error: &str,
) -> SessionUpdate {
    let mut meta = Meta::new();
    meta.insert(
        "alx/retrying".into(),
        serde_json::json!({
            "attempt": attempt,
            "maxAttempts": max_attempts,
            "delayMs": delay_ms,
            "error": error,
        }),
    );
    let text = format!(
        "Retrying in {:.0}s (attempt {attempt}/{max_attempts}): {error}",
        delay_ms as f64 / 1000.0
    );
    SessionUpdate::AgentThoughtChunk(chunk(&text).meta(meta))
}

impl EventSink for AcpSink {
    fn emit(&self, event: Event) {
        match event {
            Event::TurnStarted => self.streamed.store(false, Ordering::Relaxed),
            Event::AssistantTextDelta { text } => {
                self.streamed.store(true, Ordering::Relaxed);
                self.send(SessionUpdate::AgentMessageChunk(chunk(&text)));
            }
            Event::AssistantText { text } => {
                // Deltas arrive per model response; only repeat-free text is new.
                if !self.streamed.swap(false, Ordering::Relaxed) {
                    self.send(SessionUpdate::AgentMessageChunk(chunk(&text)));
                }
            }
            Event::ToolCallStarted { id, name, input } => {
                let call = ToolCall::new(id, tool_title(&name, &input))
                    .kind(tool_kind(&name))
                    .locations(tool_locations(&input))
                    .name(name)
                    .status(ToolCallStatus::InProgress)
                    .raw_input(input);
                self.send(SessionUpdate::ToolCall(call));
            }
            Event::ToolCallFinished {
                id,
                output,
                is_error,
                ..
            } => {
                let status = if is_error {
                    ToolCallStatus::Failed
                } else {
                    ToolCallStatus::Completed
                };
                let fields = ToolCallUpdateFields::new()
                    .status(status)
                    .content(vec![output.into()]);
                self.send(SessionUpdate::ToolCallUpdate(ToolCallUpdate::new(
                    id, fields,
                )));
            }
            Event::Retrying {
                attempt,
                max_attempts,
                delay_ms,
                error,
            } => self.send(retrying_update(attempt, max_attempts, delay_ms, &error)),
            Event::ContextUsage(u) => {
                self.send(SessionUpdate::UsageUpdate(UsageUpdate::new(
                    u.used_tokens,
                    u.limit_tokens,
                )));
            }
            Event::Compacted(c) => {
                static NEXT: AtomicU64 = AtomicU64::new(1);
                let id = format!("compaction-{}", NEXT.fetch_add(1, Ordering::Relaxed));
                self.send(compaction_update(id, &c));
            }
            Event::TurnFinished { usage, model, .. } => {
                *self.usage.lock().unwrap() = usage;
                *self.model.lock().unwrap() = model;
            }
            _ => {}
        }
    }
}

/// Sends a stored conversation to the client as updates, for `session/load`.
pub fn replay(
    cx: &ConnectionTo<Client>,
    session: &SessionId,
    messages: &[Message],
) -> Result<(), Error> {
    for m in messages {
        for block in &m.content {
            let AlxBlock::Text { text } = block else {
                continue;
            };
            let update = match m.role {
                Role::User => SessionUpdate::UserMessageChunk(chunk(text)),
                Role::Assistant => SessionUpdate::AgentMessageChunk(chunk(text)),
            };
            cx.send_notification(SessionNotification::new(session.clone(), update))?;
        }
    }
    Ok(())
}
