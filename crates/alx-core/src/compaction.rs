//! alx's own compaction, for providers that don't compact by themselves:
//! the older part of the conversation is replaced (in requests only) by a
//! model-written summary. The constants follow pi's.

use alx_protocol::{ContentBlock, Message, Role};

/// Compact when the context is within this many tokens of the window.
pub const RESERVE_TOKENS: u64 = 16_384;
/// About this many tokens of recent messages are kept as they are.
pub const KEEP_RECENT_TOKENS: u64 = 20_000;
/// Don't compact less than this many (estimated) tokens.
pub const MIN_COMPACT_TOKENS: u64 = 2_000;

/// A rough token count: about four characters a token.
pub fn estimate_tokens(m: &Message) -> u64 {
    let chars: usize = m
        .content
        .iter()
        .map(|b| match b {
            ContentBlock::Text { text } => text.len(),
            ContentBlock::ToolUse { name, input, .. } => name.len() + input.to_string().len(),
            ContentBlock::ToolResult { content, .. } => content.len(),
            // Roughly what a typical image costs (~1600 tokens).
            ContentBlock::Image { .. } => 6_400,
        })
        .sum();
    (chars as u64).div_ceil(4)
}

/// Whether the conversation may start at `messages[i]`: an assistant
/// message, or a user message that isn't answering tool calls (its results
/// would lose their calls).
fn is_boundary(m: &Message) -> bool {
    m.role == Role::Assistant
        || !m
            .content
            .iter()
            .any(|b| matches!(b, ContentBlock::ToolResult { .. }))
}

/// Where to cut `messages` (searching from `from`): the latest boundary that
/// still keeps at least `keep` estimated tokens after it, preferring user
/// messages. `None` if no cut would remove anything.
pub fn find_cut(messages: &[Message], from: usize, keep: u64) -> Option<usize> {
    let mut tail = 0;
    let mut fallback = None;
    for i in (from + 1..messages.len()).rev() {
        tail += estimate_tokens(&messages[i]);
        if tail < keep || !is_boundary(&messages[i]) {
            continue;
        }
        if messages[i].role == Role::User {
            return Some(i);
        }
        fallback.get_or_insert(i);
    }
    fallback
}

pub const SUMMARY_SYSTEM_PROMPT: &str = "You summarize a conversation between a user and an AI \
agent so the agent can continue the work with only your summary and the most recent messages. \
Write the summary only, no preamble.";

/// The request that asks for a summary of `messages` (which may start with
/// an earlier summary, so summaries build on each other).
pub fn summary_request(messages: &[Message], instructions: Option<&str>) -> Message {
    let mut transcript = String::new();
    for m in messages {
        for b in &m.content {
            let line = match (m.role, b) {
                (Role::User, ContentBlock::Text { text }) => format!("[User]: {text}"),
                (Role::Assistant, ContentBlock::Text { text }) => format!("[Assistant]: {text}"),
                (_, ContentBlock::ToolUse { name, input, .. }) => {
                    format!("[Tool call {name}]: {input}")
                }
                (
                    _,
                    ContentBlock::ToolResult {
                        content, is_error, ..
                    },
                ) => {
                    let content = clip(content, 2_000);
                    let kind = if *is_error {
                        "Tool error"
                    } else {
                        "Tool result"
                    };
                    format!("[{kind}]: {content}")
                }
                (_, ContentBlock::Image { .. }) => "[Image]".to_owned(),
            };
            transcript.push_str(&line);
            transcript.push_str("\n\n");
        }
    }
    let extra = instructions
        .map(|i| format!("\nAlso follow these instructions: {i}\n"))
        .unwrap_or_default();
    Message::user_text(format!(
        "<conversation>\n{transcript}</conversation>\n\n\
Summarize the conversation above. If it starts with an earlier summary, fold it in. Use these \
sections:\n\n\
## Goal\nWhat the user wants overall.\n\n\
## Progress\nWhat has been done, with key results.\n\n\
## Decisions\nChoices made and constraints learned (user preferences included).\n\n\
## Files\nFiles read or changed, and how.\n\n\
## Next steps\nWhat remains, in order.\n{extra}"
    ))
}

/// The text that stands in for the compacted messages.
pub fn summary_message(summary: &str) -> String {
    format!(
        "<conversation_summary>\nThe earlier part of this conversation was compacted into this summary:\n\n{}\n</conversation_summary>",
        summary.trim()
    )
}

/// The first `max` bytes of `s` (on a char boundary), marked if cut.
fn clip(s: &str, max: usize) -> String {
    if s.len() <= max {
        return s.to_owned();
    }
    let mut end = max;
    while !s.is_char_boundary(end) {
        end -= 1;
    }
    format!("{}[... {} bytes truncated ...]", &s[..end], s.len() - end)
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn big(role: Role, tokens: usize) -> Message {
        Message {
            role,
            content: vec![ContentBlock::Text {
                text: "x".repeat(tokens * 4),
            }],
        }
    }

    fn results() -> Message {
        Message {
            role: Role::User,
            content: vec![ContentBlock::ToolResult {
                tool_use_id: "t".into(),
                content: "x".repeat(400),
                is_error: false,
            }],
        }
    }

    #[test]
    fn cuts_at_the_latest_user_message_that_keeps_enough() {
        let m = vec![
            big(Role::User, 100),
            big(Role::Assistant, 100),
            big(Role::User, 100),
            big(Role::Assistant, 100),
            big(Role::User, 100),
            big(Role::Assistant, 100),
        ];
        assert_eq!(find_cut(&m, 0, 150), Some(4));
        assert_eq!(find_cut(&m, 0, 250), Some(2));
        assert_eq!(find_cut(&m, 0, 10_000), None);
        assert_eq!(find_cut(&m, 2, 250), Some(3));
        assert_eq!(find_cut(&m, 3, 250), None);
    }

    #[test]
    fn never_cuts_at_tool_results() {
        let call = Message {
            role: Role::Assistant,
            content: vec![ContentBlock::ToolUse {
                id: "t".into(),
                name: "bash".into(),
                input: json!({}),
            }],
        };
        let m = vec![
            big(Role::User, 10),
            call.clone(),
            results(),
            call,
            results(),
            big(Role::Assistant, 10),
        ];
        // Only one user message: the cut falls on an assistant message.
        assert_eq!(find_cut(&m, 0, 150), Some(1));
    }
}
