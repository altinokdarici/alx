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

/// Events emitted by the core while running a turn. UIs (CLI, TUI, web, ...)
/// render these; they never reach into the core directly.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum Event {
    TurnStarted,
    /// A request to the model provider is about to be made.
    ModelRequest,
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
    TurnFinished {
        stop_reason: StopReason,
        /// The model that produced the turn's last response, if the provider
        /// reported it. It can differ from the configured one (e.g. Copilot's
        /// default or automatic model selection).
        #[serde(default, skip_serializing_if = "Option::is_none")]
        model: Option<String>,
    },
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn stop_reason_round_trips_including_unknown() {
        for s in ["end_turn", "tool_use", "pause_turn", "something_new"] {
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
    fn event_wire_shape() {
        let e = Event::TurnFinished {
            stop_reason: StopReason::EndTurn,
            model: None,
        };
        assert_eq!(
            serde_json::to_value(&e).unwrap(),
            json!({"type": "turn_finished", "stop_reason": "end_turn"})
        );
    }
}
