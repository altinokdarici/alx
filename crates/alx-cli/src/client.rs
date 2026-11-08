//! The CLI as an ACP client: it talks to the agent through `alx-acp` and turns
//! the `session/update`s it gets back into the core `Event`s the renderer
//! draws.

use agent_client_protocol::schema::v1::{
    CompactionStatus, CompactionUpdate, ContentBlock, ContentChunk, PromptResponse, SessionUpdate,
    StopReason as AcpStop, ToolCallContent, ToolCallStatus,
};
use alx_core::protocol::{Compaction, ContextUsage, Event, StopReason, Usage};
use std::collections::HashMap;
use std::sync::Mutex;

/// What a `/compact` request came to, from its final `compaction_update`.
#[derive(Debug, PartialEq)]
pub enum CompactOutcome {
    Compacted,
    NothingToDo,
    Failed(String),
}

/// Maps ACP updates to `Event`s. Keeps the tool names by call id, since an
/// update for a finished call carries only the id.
#[derive(Default)]
pub struct Translator {
    tools: Mutex<HashMap<String, String>>,
}

fn text_of(blocks: impl IntoIterator<Item = ContentBlock>) -> String {
    blocks
        .into_iter()
        .filter_map(|b| match b {
            ContentBlock::Text(t) => Some(t.text),
            _ => None,
        })
        .collect::<Vec<_>>()
        .join("\n")
}

fn meta_u64(c: &CompactionUpdate, key: &str) -> Option<u64> {
    serde_json::to_value(&c.meta).ok()?.get(key)?.as_u64()
}

impl Translator {
    /// The events for one update. After a tool call finishes the model is
    /// asked again, so that also yields a `ModelRequest`.
    pub fn events(&self, update: SessionUpdate) -> Vec<Event> {
        match update {
            SessionUpdate::AgentMessageChunk(c) => vec![Event::AssistantTextDelta {
                text: text_of([c.content]),
            }],
            SessionUpdate::ToolCall(c) => {
                let id = c.tool_call_id.0.to_string();
                let name = c.name.unwrap_or(c.title);
                self.tools.lock().unwrap().insert(id.clone(), name.clone());
                vec![Event::ToolCallStarted {
                    id,
                    name,
                    input: c.raw_input.unwrap_or_default(),
                }]
            }
            SessionUpdate::ToolCallUpdate(u) => {
                let is_error = match u.fields.status {
                    Some(ToolCallStatus::Failed) => true,
                    Some(ToolCallStatus::Completed) => false,
                    _ => return vec![],
                };
                let id = u.tool_call_id.0.to_string();
                let name = self.tools.lock().unwrap().remove(&id).unwrap_or_default();
                let output = text_of(u.fields.content.unwrap_or_default().into_iter().filter_map(
                    |c| match c {
                        ToolCallContent::Content(c) => Some(c.content),
                        _ => None,
                    },
                ));
                vec![
                    Event::ToolCallFinished {
                        id,
                        name,
                        output,
                        is_error,
                    },
                    Event::ModelRequest,
                ]
            }
            SessionUpdate::AgentThoughtChunk(c) => retrying(&c).into_iter().collect(),
            SessionUpdate::UsageUpdate(u) => vec![Event::ContextUsage(ContextUsage {
                used_tokens: u.used,
                limit_tokens: u.size,
            })],
            SessionUpdate::CompactionUpdate(c) => match compaction(&c) {
                Some(c) => vec![Event::Compacted(c)],
                None => vec![],
            },
            _ => vec![],
        }
    }
}

/// A retry notice sent by alx's server (see `alx_acp::events::retrying_update`).
fn retrying(c: &ContentChunk) -> Option<Event> {
    let v = serde_json::to_value(&c.meta).ok()?;
    let r = v.get("alx/retrying")?;
    Some(Event::Retrying {
        attempt: r.get("attempt")?.as_u64()? as u32,
        max_attempts: r.get("maxAttempts")?.as_u64()? as u32,
        delay_ms: r.get("delayMs")?.as_u64()?,
        error: r.get("error")?.as_str()?.to_owned(),
    })
}

/// A finished compaction with its counts (sent by the server in `_meta`).
pub fn compaction(c: &CompactionUpdate) -> Option<Compaction> {
    if c.status != CompactionStatus::Completed {
        return None;
    }
    Some(Compaction {
        messages_removed: meta_u64(c, "messagesRemoved")?,
        tokens_before: meta_u64(c, "tokensBefore"),
        tokens_after: meta_u64(c, "tokensAfter"),
    })
}

/// The outcome of a `/compact`, if `update` is its final word.
pub fn compact_outcome(update: &SessionUpdate) -> Option<CompactOutcome> {
    let SessionUpdate::CompactionUpdate(c) = update else {
        return None;
    };
    match c.status {
        CompactionStatus::Completed if compaction(c).is_some() => Some(CompactOutcome::Compacted),
        CompactionStatus::Completed => Some(CompactOutcome::NothingToDo),
        CompactionStatus::Failed => {
            let error = serde_json::to_value(&c.error)
                .ok()
                .and_then(|v| v.as_str().map(str::to_owned))
                .unwrap_or_else(|| "compaction failed".into());
            Some(CompactOutcome::Failed(error))
        }
        _ => None,
    }
}

pub fn stop_reason(r: AcpStop) -> StopReason {
    match r {
        AcpStop::EndTurn => StopReason::EndTurn,
        AcpStop::MaxTokens => StopReason::MaxTokens,
        AcpStop::Refusal => StopReason::Refusal,
        AcpStop::Cancelled => StopReason::Cancelled,
        other => StopReason::Other(format!("{other:?}")),
    }
}

/// The `TurnFinished` event for a `session/prompt` reply.
pub fn turn_finished(reply: &PromptResponse) -> Event {
    let usage = reply.usage.as_ref().map(|u| {
        let read = u.cached_read_tokens.unwrap_or(0);
        let write = u.cached_write_tokens.unwrap_or(0);
        Usage {
            input_tokens: u.input_tokens.saturating_sub(read + write),
            output_tokens: u.output_tokens,
            cache_read_tokens: read,
            cache_write_tokens: write,
        }
    });
    let model = reply
        .meta
        .as_ref()
        .and_then(|m| m.get(alx_acp::MODEL_META))
        .and_then(|m| m.as_str())
        .map(str::to_owned);
    Event::TurnFinished {
        stop_reason: stop_reason(reply.stop_reason),
        model,
        usage,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use agent_client_protocol::schema::v1::{
        Meta, ToolCall, ToolCallUpdate, ToolCallUpdateFields, Usage as AcpUsage,
    };
    use serde_json::json;

    #[test]
    fn a_finished_tool_call_keeps_its_name_and_asks_the_model_again() {
        let t = Translator::default();
        let started = t.events(SessionUpdate::ToolCall(
            ToolCall::new("t1", "echo hi")
                .name("bash")
                .raw_input(json!({"input": "echo hi"})),
        ));
        assert!(matches!(&started[..], [Event::ToolCallStarted { name, .. }] if name == "bash"));
        let fields = ToolCallUpdateFields::new()
            .status(ToolCallStatus::Failed)
            .content(vec!["boom".to_string().into()]);
        let done = t.events(SessionUpdate::ToolCallUpdate(ToolCallUpdate::new(
            "t1", fields,
        )));
        assert!(matches!(
            &done[..],
            [Event::ToolCallFinished { name, output, is_error: true, .. }, Event::ModelRequest]
                if name == "bash" && output == "boom"
        ));
    }

    #[test]
    fn compaction_counts_come_from_meta() {
        let mut meta = Meta::new();
        meta.insert("messagesRemoved".into(), 4.into());
        meta.insert("tokensAfter".into(), 100.into());
        let done = SessionUpdate::CompactionUpdate(
            CompactionUpdate::new("c", CompactionStatus::Completed).meta(meta),
        );
        assert_eq!(compact_outcome(&done), Some(CompactOutcome::Compacted));
        assert!(matches!(
            &Translator::default().events(done)[..],
            [Event::Compacted(Compaction {
                messages_removed: 4,
                tokens_before: None,
                tokens_after: Some(100)
            })]
        ));
        let empty = SessionUpdate::CompactionUpdate(CompactionUpdate::new(
            "c",
            CompactionStatus::Completed,
        ));
        assert_eq!(compact_outcome(&empty), Some(CompactOutcome::NothingToDo));
        let failed = SessionUpdate::CompactionUpdate(
            CompactionUpdate::new("c", CompactionStatus::Failed).error("nope"),
        );
        assert_eq!(
            compact_outcome(&failed),
            Some(CompactOutcome::Failed("nope".into()))
        );
    }

    #[test]
    fn reply_usage_splits_cached_tokens_back_out_and_names_the_model() {
        let mut meta = Meta::new();
        meta.insert(alx_acp::MODEL_META.into(), "m1".into());
        let reply = PromptResponse::new(AcpStop::EndTurn)
            .usage(
                AcpUsage::new(116, 111, 5)
                    .cached_read_tokens(100)
                    .cached_write_tokens(1),
            )
            .meta(meta);
        let Event::TurnFinished {
            stop_reason,
            model,
            usage: Some(u),
        } = turn_finished(&reply)
        else {
            panic!("no usage");
        };
        assert_eq!(stop_reason, StopReason::EndTurn);
        assert_eq!(model.as_deref(), Some("m1"));
        assert_eq!(
            (
                u.input_tokens,
                u.cache_read_tokens,
                u.cache_write_tokens,
                u.output_tokens
            ),
            (10, 100, 1, 5)
        );
    }
}
