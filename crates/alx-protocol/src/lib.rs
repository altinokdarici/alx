//! Wire types shared by the alx core and its clients.
//!
//! These types are provider-neutral: they describe a conversation and the
//! events the core emits while running it. Provider-specific wire formats
//! (Anthropic, OpenAI, ...) are mapped to and from these inside each
//! `Provider` implementation in `alx-core`.

use serde::{Deserialize, Serialize};
use serde_json::Value;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Role {
    User,
    Assistant,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ContentBlock {
    Text {
        text: String,
    },
    ToolUse {
        id: String,
        name: String,
        input: Value,
    },
    ToolResult {
        tool_use_id: String,
        content: String,
        #[serde(default, skip_serializing_if = "std::ops::Not::not")]
        is_error: bool,
    },
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Message {
    pub role: Role,
    pub content: Vec<ContentBlock>,
}

impl Message {
    pub fn user_text(text: impl Into<String>) -> Self {
        Self {
            role: Role::User,
            content: vec![ContentBlock::Text { text: text.into() }],
        }
    }
}

/// Why the model stopped producing output.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(from = "String", into = "String")]
pub enum StopReason {
    EndTurn,
    ToolUse,
    MaxTokens,
    StopSequence,
    Refusal,
    PauseTurn,
    /// The user cancelled the turn. Never sent by a provider.
    Cancelled,
    Other(String),
}

impl From<String> for StopReason {
    fn from(s: String) -> Self {
        match s.as_str() {
            "end_turn" => Self::EndTurn,
            "tool_use" => Self::ToolUse,
            "max_tokens" => Self::MaxTokens,
            "stop_sequence" => Self::StopSequence,
            "refusal" => Self::Refusal,
            "pause_turn" => Self::PauseTurn,
            "cancelled" => Self::Cancelled,
            _ => Self::Other(s),
        }
    }
}

impl From<StopReason> for String {
    fn from(r: StopReason) -> Self {
        r.as_str().to_owned()
    }
}

impl StopReason {
    pub fn as_str(&self) -> &str {
        match self {
            Self::EndTurn => "end_turn",
            Self::ToolUse => "tool_use",
            Self::MaxTokens => "max_tokens",
            Self::StopSequence => "stop_sequence",
            Self::Refusal => "refusal",
            Self::PauseTurn => "pause_turn",
            Self::Cancelled => "cancelled",
            Self::Other(s) => s,
        }
    }
}

impl std::fmt::Display for StopReason {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

/// Description of a tool, as advertised to the model.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ToolSpec {
    pub name: String,
    pub description: String,
    /// JSON Schema for the tool's input object.
    pub input_schema: Value,
}

/// Which LLM backend an agent talks to, for clients to display.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ProviderInfo {
    /// Short provider id, e.g. `anthropic` or `copilot`.
    pub name: String,
    /// The configured model, or `None` when the provider picks its default.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub model: Option<String>,
}

/// Token counts for one or more model requests. Anthropic reports
/// `input_tokens` separately from cached tokens, so total prompt size is
/// `input_tokens + cache_read_tokens + cache_write_tokens`.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct Usage {
    /// Prompt tokens processed at the full input price.
    pub input_tokens: u64,
    pub output_tokens: u64,
    /// Prompt tokens served from the prompt cache.
    #[serde(default)]
    pub cache_read_tokens: u64,
    /// Prompt tokens written to the prompt cache.
    #[serde(default)]
    pub cache_write_tokens: u64,
}

impl Usage {
    pub fn add(&mut self, other: &Usage) {
        self.input_tokens += other.input_tokens;
        self.output_tokens += other.output_tokens;
        self.cache_read_tokens += other.cache_read_tokens;
        self.cache_write_tokens += other.cache_write_tokens;
    }
}

/// How full the model's context window is, as the provider reports it.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct ContextUsage {
    pub used_tokens: u64,
    /// The model's context window size.
    pub limit_tokens: u64,
}

/// A finished compaction: older conversation replaced by a summary.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct Compaction {
    pub messages_removed: u64,
    /// Context size before and after, when known.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tokens_before: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tokens_after: Option<u64>,
}

/// Events emitted by the core while running a turn. UIs (CLI, TUI, web, ...)
/// render these; they never reach into the core directly.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum Event {
    TurnStarted,
    /// A request to the model provider is about to be made.
    ModelRequest,
    /// A piece of assistant text, sent live while the model is still
    /// writing. A client that shows these can ignore the `AssistantText`
    /// that follows (it repeats the same text, whole); one that doesn't can
    /// ignore these. Not every provider sends deltas.
    AssistantTextDelta {
        text: String,
    },
    /// A complete block of assistant text.
    AssistantText {
        text: String,
    },
    ToolCallStarted {
        id: String,
        name: String,
        input: Value,
    },
    ToolCallFinished {
        id: String,
        name: String,
        output: String,
        is_error: bool,
    },
    /// The context window fill level, after a model response.
    ContextUsage(ContextUsage),
    /// The provider compacted the conversation (automatically or on request).
    Compacted(Compaction),
    TurnFinished {
        stop_reason: StopReason,
        /// The model that produced the turn's last response, if the provider
        /// reported it. It can differ from the configured one (e.g. Copilot's
        /// default or automatic model selection).
        #[serde(default, skip_serializing_if = "Option::is_none")]
        model: Option<String>,
        /// Tokens used by all requests in the turn, if the provider reports them.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        usage: Option<Usage>,
    },
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn stop_reason_round_trips_including_unknown() {
        for s in [
            "end_turn",
            "tool_use",
            "pause_turn",
            "cancelled",
            "something_new",
        ] {
            let r: StopReason = serde_json::from_value(json!(s)).unwrap();
            assert_eq!(serde_json::to_value(&r).unwrap(), json!(s));
        }
        assert_eq!(
            StopReason::from("x".to_string()),
            StopReason::Other("x".into())
        );
    }

    #[test]
    fn content_block_wire_shape() {
        let b = ContentBlock::ToolResult {
            tool_use_id: "t1".into(),
            content: "ok".into(),
            is_error: false,
        };
        assert_eq!(
            serde_json::to_value(&b).unwrap(),
            json!({"type": "tool_result", "tool_use_id": "t1", "content": "ok"})
        );
    }

    #[test]
    fn delta_event_wire_shape() {
        let e = Event::AssistantTextDelta { text: "hi".into() };
        assert_eq!(
            serde_json::to_value(&e).unwrap(),
            json!({"type": "assistant_text_delta", "text": "hi"})
        );
    }

    #[test]
    fn event_wire_shape() {
        let e = Event::TurnFinished {
            stop_reason: StopReason::EndTurn,
            model: None,
            usage: None,
        };
        assert_eq!(
            serde_json::to_value(&e).unwrap(),
            json!({"type": "turn_finished", "stop_reason": "end_turn"})
        );
    }
}
