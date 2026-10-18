//! Anthropic Messages API (non-streaming for now).

use super::{CompletionRequest, CompletionResponse, Provider};
use alx_protocol::{ContentBlock, Message, ProviderInfo, Role};
use anyhow::{Context, bail};
use async_trait::async_trait;
use serde_json::{Value, json};

#[derive(Debug, Clone)]
pub struct AnthropicConfig {
    pub api_key: String,
    pub model: String,
    pub base_url: String,
    pub api_version: String,
}

impl AnthropicConfig {
    pub const DEFAULT_MODEL: &str = "claude-opus-5-5";

    pub fn new(api_key: impl Into<String>) -> Self {
        Self {
            api_key: api_key.into(),
            model: Self::DEFAULT_MODEL.into(),
            base_url: "https://api.anthropic.com".into(),
            api_version: "2023-06-01".into(),
        }
    }
}

pub struct AnthropicProvider {
    config: AnthropicConfig,
    http: reqwest::Client,
}

impl AnthropicProvider {
    pub fn new(config: AnthropicConfig) -> Self {
        Self {
            config,
            http: reqwest::Client::new(),
        }
    }

    fn body(&self, req: &CompletionRequest) -> Value {
        let tools: Vec<Value> = req
            .tools
            .iter()
            .map(|t| {
                json!({
                    "name": t.name,
                    "description": t.description,
                    "input_schema": t.input_schema,
                })
            })
            .collect();
        json!({
            "model": self.config.model,
            "stream": false,
            "max_tokens": req.max_tokens,
            "system": [{ "type": "text", "text": req.system }],
            "messages": req.messages.iter().map(message_to_wire).collect::<Vec<_>>(),
            "tools": tools,
        })
    }
}

fn message_to_wire(m: &Message) -> Value {
    let role = match m.role {
        Role::User => "user",
        Role::Assistant => "assistant",
    };
    let content: Vec<Value> = m
        .content
        .iter()
        .map(|b| match b {
            ContentBlock::Text { text } => json!({ "type": "text", "text": text }),
            ContentBlock::ToolUse { id, name, input } => {
                json!({ "type": "tool_use", "id": id, "name": name, "input": input })
            }
            ContentBlock::ToolResult {
                tool_use_id,
                content,
                is_error,
            } => json!({
                "type": "tool_result",
                "tool_use_id": tool_use_id,
                "content": content,
                "is_error": is_error,
            }),
        })
        .collect();
    json!({ "role": role, "content": content })
}

fn response_from_wire(data: Value) -> anyhow::Result<CompletionResponse> {
    let stop_reason = data
        .get("stop_reason")
        .and_then(Value::as_str)
        .context("response has no stop_reason")?
        .to_owned()
        .into();
    let blocks = data
        .get("content")
        .and_then(Value::as_array)
        .context("response has no content")?;
    let mut content = Vec::with_capacity(blocks.len());
    for b in blocks {
        match b.get("type").and_then(Value::as_str) {
            Some("text") => content.push(ContentBlock::Text {
                text: b["text"].as_str().unwrap_or_default().to_owned(),
            }),
            Some("tool_use") => content.push(ContentBlock::ToolUse {
                id: b["id"].as_str().context("tool_use without id")?.to_owned(),
                name: b["name"]
                    .as_str()
                    .context("tool_use without name")?
                    .to_owned(),
                input: b.get("input").cloned().unwrap_or(Value::Null),
            }),
            // Block types we don't model yet (thinking, server tools, ...) are dropped.
            _ => {}
        }
    }
    Ok(CompletionResponse {
        content,
        stop_reason,
        model: data.get("model").and_then(Value::as_str).map(str::to_owned),
    })
}

#[async_trait]
impl Provider for AnthropicProvider {
    fn info(&self) -> ProviderInfo {
        ProviderInfo {
            name: "anthropic".into(),
            model: Some(self.config.model.clone()),
        }
    }

    async fn complete(&self, req: CompletionRequest) -> anyhow::Result<CompletionResponse> {
        let url = format!("{}/v1/messages", self.config.base_url.trim_end_matches('/'));
        let resp = self
            .http
            .post(url)
            .header("x-api-key", &self.config.api_key)
            .header("anthropic-version", &self.config.api_version)
            .json(&self.body(&req))
            .send()
            .await
            .context("LLM request failed to send")?;
        let status = resp.status();
        let data: Value = resp.json().await.context("LLM response was not JSON")?;
        if !status.is_success() {
            bail!("LLM request failed ({status}): {data}");
        }
        response_from_wire(data)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use alx_protocol::{StopReason, ToolSpec};

    #[test]
    fn builds_request_body() {
        let p = AnthropicProvider::new(AnthropicConfig::new("k"));
        let body = p.body(&CompletionRequest {
            system: "sys".into(),
            messages: vec![Message::user_text("hi")],
            tools: vec![ToolSpec {
                name: "bash".into(),
                description: "d".into(),
                input_schema: json!({"type": "object"}),
            }],
            max_tokens: 10,
        });
        assert_eq!(body["model"], AnthropicConfig::DEFAULT_MODEL);
        assert_eq!(body["system"][0]["text"], "sys");
        assert_eq!(
            body["messages"][0],
            json!({"role": "user", "content": [{"type": "text", "text": "hi"}]})
        );
        assert_eq!(body["tools"][0]["name"], "bash");
    }

    #[test]
    fn parses_response_and_skips_unknown_blocks() {
        let r = response_from_wire(json!({
            "id": "msg_1",
            "model": "claude-x",
            "stop_reason": "tool_use",
            "content": [
                {"type": "thinking", "thinking": "..."},
                {"type": "text", "text": "running"},
                {"type": "tool_use", "id": "t1", "name": "bash", "input": {"input": "ls"}}
            ]
        }))
        .unwrap();
        assert_eq!(r.stop_reason, StopReason::ToolUse);
        assert_eq!(r.content.len(), 2);
        assert_eq!(r.model.as_deref(), Some("claude-x"));
    }
}
