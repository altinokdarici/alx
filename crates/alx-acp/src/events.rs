//! alx `Event`s to ACP `session/update` notifications.

use agent_client_protocol::schema::v1::{
    ContentBlock, ContentChunk, SessionId, SessionNotification, SessionUpdate, TextContent,
    ToolCall, ToolCallStatus, ToolCallUpdate, ToolCallUpdateFields, ToolKind, UsageUpdate,
};
use agent_client_protocol::{Client, ConnectionTo, Error};
use alx_core::EventSink;
use alx_core::protocol::{ContentBlock as AlxBlock, Event, Message, Role};
use std::sync::atomic::{AtomicBool, Ordering};

pub struct AcpSink {
    cx: ConnectionTo<Client>,
    session: SessionId,
    /// Whether this turn already streamed text, so the whole `AssistantText`
    /// that follows would repeat it.
    streamed: AtomicBool,
}

impl AcpSink {
    pub fn new(cx: ConnectionTo<Client>, session: SessionId) -> Self {
        Self {
            cx,
            session,
            streamed: AtomicBool::new(false),
        }
    }

    fn send(&self, update: SessionUpdate) {
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
fn title(name: &str, input: &serde_json::Value) -> String {
    match input.get("input").and_then(|c| c.as_str()) {
        Some(cmd) if name == "bash" => cmd.lines().next().unwrap_or("").to_string(),
        _ => name.to_string(),
    }
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
                let call = ToolCall::new(id, title(&name, &input))
                    .name(name)
                    .kind(ToolKind::Execute)
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
            Event::ContextUsage(u) => {
                self.send(SessionUpdate::UsageUpdate(UsageUpdate::new(
                    u.used_tokens,
                    u.limit_tokens,
                )));
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
