//! Anthropic Messages API, streamed (server-sent events) and reassembled
//! into one response; the agent loop still sees a single `complete` result.

use super::{CompletionRequest, CompletionResponse, Provider, TransientError};
use alx_protocol::{ContentBlock, Message, ModelInfo, ProviderInfo, Role, StopReason, Usage};
use anyhow::{Context, bail};
use async_trait::async_trait;
use serde_json::{Value, json};
use std::collections::BTreeMap;
use std::time::Duration;

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
            "stream": true,
            // Automatic prompt caching: the breakpoint follows the end of the
            // conversation, so each request reads the previous one's prefix.
            // Tools and system are rendered in a stable order to keep the
            // prefix byte-identical between requests.
            "cache_control": { "type": "ephemeral" },
            "max_tokens": req.max_tokens,
            "system": [{ "type": "text", "text": req.system }],
            "messages": req.messages.iter().map(message_to_wire).collect::<Vec<_>>(),
            "tools": tools,
        })
    }
}

/// The `data` of a `GET /v1/models` reply.
fn models_from_wire(body: &Value) -> Vec<ModelInfo> {
    body["data"]
        .as_array()
        .into_iter()
        .flatten()
        .filter_map(|m| {
            let id = m["id"].as_str()?.to_owned();
            let name = m["display_name"].as_str().unwrap_or(&id).to_owned();
            Some(ModelInfo { id, name })
        })
        .collect()
}

fn message_to_wire(m: &Message) -> Value {
    let role = match m.role {
        Role::User => "user",
        Role::Assistant => "assistant",
    };
    let mut content: Vec<Value> = Vec::new();
    // Index of the last block if it is a tool_result, so images that follow
    // it go inside it (tool_result blocks must come first in the message).
    let mut open_result: Option<usize> = None;
    for b in &m.content {
        let block = match b {
            ContentBlock::Text { text } => json!({ "type": "text", "text": text }),
            ContentBlock::ToolUse { id, name, input } => {
                json!({ "type": "tool_use", "id": id, "name": name, "input": input })
            }
            ContentBlock::ToolResult {
                tool_use_id,
                content: text,
                is_error,
            } => {
                content.push(json!({
                    "type": "tool_result",
                    "tool_use_id": tool_use_id,
                    "content": [{ "type": "text", "text": text }],
                    "is_error": is_error,
                }));
                open_result = Some(content.len() - 1);
                continue;
            }
            ContentBlock::Image { media_type, data } => {
                let image = json!({
                    "type": "image",
                    "source": { "type": "base64", "media_type": media_type, "data": data },
                });
                if let Some(i) = open_result
                    && let Some(inner) = content[i]["content"].as_array_mut()
                {
                    inner.push(image);
                    continue;
                }
                image
            }
        };
        open_result = None;
        content.push(block);
    }
    json!({ "role": role, "content": content })
}

/// Splits a server-sent-events byte stream into event payloads (`data:`).
#[derive(Default)]
struct SseParser {
    buf: Vec<u8>,
    data: Vec<String>,
}

impl SseParser {
    /// Feed more bytes; returns the payload of every event completed by them.
    fn push(&mut self, bytes: &[u8]) -> Vec<String> {
        self.buf.extend_from_slice(bytes);
        let mut events = Vec::new();
        while let Some(nl) = self.buf.iter().position(|&b| b == b'\n') {
            let raw: Vec<u8> = self.buf.drain(..=nl).collect();
            let line = String::from_utf8_lossy(&raw);
            let line = line.trim_end_matches(['\n', '\r']);
            if line.is_empty() {
                if !self.data.is_empty() {
                    events.push(self.data.join("\n"));
                    self.data.clear();
                }
            } else if let Some(d) = line.strip_prefix("data:") {
                self.data.push(d.strip_prefix(' ').unwrap_or(d).to_owned());
            }
            // `event:`, `id:` and comment lines carry nothing we need: the
            // payload has its own `type`.
        }
        events
    }
}

enum Partial {
    Text(String),
    ToolUse {
        id: String,
        name: String,
        json: String,
    },
    /// Thinking, server tools, ... are not modelled yet and are dropped.
    Other,
}

/// Rebuilds a [`CompletionResponse`] from streaming events.
#[derive(Default)]
struct StreamAccumulator {
    blocks: BTreeMap<u64, Partial>,
    model: Option<String>,
    stop_reason: Option<String>,
    usage: Option<Usage>,
}

impl StreamAccumulator {
    fn merge_usage(&mut self, u: &Value) {
        // Later events only carry the fields that changed (output_tokens is
        // cumulative), so merge field by field.
        let cur = self.usage.get_or_insert_with(Usage::default);
        let new = usage_from_wire(u);
        for (key, dst, src) in [
            ("input_tokens", &mut cur.input_tokens, new.input_tokens),
            ("output_tokens", &mut cur.output_tokens, new.output_tokens),
            (
                "cache_read_input_tokens",
                &mut cur.cache_read_tokens,
                new.cache_read_tokens,
            ),
            (
                "cache_creation_input_tokens",
                &mut cur.cache_write_tokens,
                new.cache_write_tokens,
            ),
        ] {
            if u.get(key).is_some() {
                *dst = src;
            }
        }
    }

    /// Applies one event; returns the text it added, if it was a text delta.
    fn event(&mut self, ev: &Value) -> anyhow::Result<Option<String>> {
        let mut added = None;
        let index = ev.get("index").and_then(Value::as_u64).unwrap_or(0);
        match ev.get("type").and_then(Value::as_str) {
            Some("message_start") => {
                let m = &ev["message"];
                self.model = m.get("model").and_then(Value::as_str).map(str::to_owned);
                if let Some(u) = m.get("usage") {
                    self.merge_usage(u);
                }
            }
            Some("content_block_start") => {
                let b = &ev["content_block"];
                let partial = match b.get("type").and_then(Value::as_str) {
                    Some("text") => {
                        Partial::Text(b["text"].as_str().unwrap_or_default().to_owned())
                    }
                    Some("tool_use") => Partial::ToolUse {
                        id: b["id"].as_str().context("tool_use without id")?.to_owned(),
                        name: b["name"]
                            .as_str()
                            .context("tool_use without name")?
                            .to_owned(),
                        json: String::new(),
                    },
                    _ => Partial::Other,
                };
                self.blocks.insert(index, partial);
            }
            Some("content_block_delta") => {
                let d = &ev["delta"];
                match (
                    self.blocks.get_mut(&index),
                    d.get("type").and_then(Value::as_str),
                ) {
                    (Some(Partial::Text(t)), Some("text_delta")) => {
                        let text = d["text"].as_str().unwrap_or_default();
                        t.push_str(text);
                        added = Some(text.to_owned());
                    }
                    (Some(Partial::ToolUse { json, .. }), Some("input_json_delta")) => {
                        json.push_str(d["partial_json"].as_str().unwrap_or_default());
                    }
                    _ => {}
                }
            }
            Some("message_delta") => {
                if let Some(r) = ev["delta"].get("stop_reason").and_then(Value::as_str) {
                    self.stop_reason = Some(r.to_owned());
                }
                if let Some(u) = ev.get("usage") {
                    self.merge_usage(u);
                }
            }
            Some("error") => bail!("LLM stream error: {}", ev["error"]),
            // ping, content_block_stop, message_stop, unknown future events.
            _ => {}
        }
        Ok(added)
    }

    fn finish(self) -> anyhow::Result<CompletionResponse> {
        let stop_reason: StopReason = self
            .stop_reason
            .context("LLM stream ended before the message finished")?
            .into();
        let mut content = Vec::new();
        for partial in self.blocks.into_values() {
            match partial {
                Partial::Text(text) => content.push(ContentBlock::Text { text }),
                Partial::ToolUse { id, name, json } => {
                    // A response cut off by max_tokens can end mid-JSON; the
                    // agent loop doesn't run its tool calls anyway.
                    let input = if json.trim().is_empty() {
                        json!({})
                    } else if stop_reason == StopReason::MaxTokens {
                        serde_json::from_str(&json).unwrap_or_else(|_| json!({}))
                    } else {
                        serde_json::from_str(&json)
                            .with_context(|| format!("tool_use `{name}` had invalid JSON input"))?
                    };
                    content.push(ContentBlock::ToolUse { id, name, input });
                }
                Partial::Other => {}
            }
        }
        Ok(CompletionResponse {
            context: None,
            compactions: Vec::new(),
            content,
            stop_reason,
            model: self.model,
            usage: self.usage,
        })
    }
}

/// What the response headers say about retrying: `Some(false)` from
/// `x-should-retry: false`, otherwise whether to retry is up to the status;
/// the delay from `retry-after-ms` or `retry-after` (seconds).
struct RetryHint {
    should_retry: Option<bool>,
    after: Option<Duration>,
}

fn retry_hint(headers: &reqwest::header::HeaderMap) -> RetryHint {
    let get = |k: &str| headers.get(k).and_then(|v| v.to_str().ok());
    let after = get("retry-after-ms")
        .and_then(|v| v.trim().parse::<f64>().ok())
        .map(|ms| Duration::from_secs_f64(ms.max(0.0) / 1000.0))
        .or_else(|| {
            get("retry-after")
                .and_then(|v| v.trim().parse::<f64>().ok())
                .map(|s| Duration::from_secs_f64(s.max(0.0)))
        });
    let should_retry = get("x-should-retry").and_then(|v| match v.trim() {
        "true" => Some(true),
        "false" => Some(false),
        _ => None,
    });
    RetryHint {
        should_retry,
        after,
    }
}

/// The error for a failed `/v1/messages` request: transient for timeouts,
/// conflicts, rate limits, overload and server errors, unless the server says
/// not to retry or the body is about quota or billing (retrying won't help).
fn request_error(status: u16, hint: RetryHint, body: &str) -> anyhow::Error {
    let message = format!("LLM request failed ({status}): {body}");
    let lower = body.to_ascii_lowercase();
    let hopeless = ["quota", "billing", "credit balance"]
        .iter()
        .any(|w| lower.contains(w));
    let retryable = match hint.should_retry {
        Some(r) => r,
        None => matches!(status, 408 | 409 | 429 | 500..=599) && !hopeless,
    };
    if retryable {
        anyhow::Error::new(TransientError {
            message,
            retry_after: hint.after,
        })
    } else {
        anyhow::anyhow!(message)
    }
}

fn usage_from_wire(u: &Value) -> Usage {
    let n = |k: &str| u.get(k).and_then(Value::as_u64).unwrap_or(0);
    Usage {
        input_tokens: n("input_tokens"),
        output_tokens: n("output_tokens"),
        cache_read_tokens: n("cache_read_input_tokens"),
        cache_write_tokens: n("cache_creation_input_tokens"),
    }
}

#[async_trait]
impl Provider for AnthropicProvider {
    fn context_window(&self) -> Option<u64> {
        // Every current Claude model; the 1M-token window is opt-in (beta).
        Some(200_000)
    }

    fn info(&self) -> ProviderInfo {
        ProviderInfo {
            name: "anthropic".into(),
            model: Some(self.config.model.clone()),
        }
    }

    async fn list_models(&self) -> anyhow::Result<Vec<ModelInfo>> {
        if self.config.api_key.is_empty() {
            bail!("CLAUDE_API_KEY environment variable is not set");
        }
        let url = format!("{}/v1/models", self.config.base_url.trim_end_matches('/'));
        let resp = self
            .http
            .get(url)
            .query(&[("limit", "1000")])
            .header("x-api-key", &self.config.api_key)
            .header("anthropic-version", &self.config.api_version)
            .send()
            .await
            .context("listing Anthropic models")?;
        let status = resp.status();
        if !status.is_success() {
            let body = resp.text().await.unwrap_or_default();
            bail!("listing Anthropic models failed ({status}): {body}");
        }
        let body: Value = resp.json().await.context("bad model list")?;
        Ok(models_from_wire(&body))
    }

    async fn complete(&self, req: CompletionRequest) -> anyhow::Result<CompletionResponse> {
        self.complete_streaming(req, &|_| {}).await
    }

    async fn complete_streaming(
        &self,
        req: CompletionRequest,
        on_text: &(dyn for<'a> Fn(&'a str) + Send + Sync),
    ) -> anyhow::Result<CompletionResponse> {
        // Checked here rather than at construction, so an agent can be built
        // (e.g. to list sessions) without a key.
        if self.config.api_key.is_empty() {
            bail!("CLAUDE_API_KEY environment variable is not set");
        }
        let url = format!("{}/v1/messages", self.config.base_url.trim_end_matches('/'));
        let resp = self
            .http
            .post(url)
            .header("x-api-key", &self.config.api_key)
            .header("anthropic-version", &self.config.api_version)
            .json(&self.body(&req))
            .send()
            .await
            .map_err(|e| {
                anyhow::Error::new(TransientError {
                    message: format!("LLM request failed to send: {e}"),
                    retry_after: None,
                })
            })?;
        let status = resp.status();
        if !status.is_success() {
            let retry = retry_hint(resp.headers());
            let body = resp.text().await.unwrap_or_default();
            return Err(request_error(status.as_u16(), retry, &body));
        }
        // Dropping this future drops the connection, which stops the
        // response mid-stream (that is how cancellation works).
        let mut resp = resp;
        let mut sse = SseParser::default();
        let mut acc = StreamAccumulator::default();
        while let Some(chunk) = resp.chunk().await.context("LLM stream was interrupted")? {
            for data in sse.push(&chunk) {
                let ev: Value = serde_json::from_str(&data).context("bad event in LLM stream")?;
                if let Some(text) = acc.event(&ev)?
                    && !text.is_empty()
                {
                    on_text(&text);
                }
            }
        }
        acc.finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use alx_protocol::ToolSpec;

    #[test]
    fn images_after_a_tool_result_go_inside_it() {
        let image = || ContentBlock::Image {
            media_type: "image/png".into(),
            data: "AAAA".into(),
        };
        let wire = message_to_wire(&Message {
            role: Role::User,
            content: vec![
                ContentBlock::ToolResult {
                    tool_use_id: "t1".into(),
                    content: "read".into(),
                    is_error: false,
                },
                image(),
                ContentBlock::Text {
                    text: "look".into(),
                },
                image(),
            ],
        });
        let content = wire["content"].as_array().unwrap();
        assert_eq!(content.len(), 3);
        assert_eq!(content[0]["type"], "tool_result");
        assert_eq!(content[0]["content"][1]["type"], "image");
        assert_eq!(content[0]["content"][1]["source"]["data"], "AAAA");
        assert_eq!(content[1]["type"], "text");
        assert_eq!(content[2]["type"], "image");
    }

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
        assert_eq!(body["cache_control"], json!({"type": "ephemeral"}));
        assert_eq!(body["stream"], true);
    }

    fn sse(events: &[Value]) -> String {
        events
            .iter()
            .map(|e| format!("event: {}\ndata: {e}\n\n", e["type"].as_str().unwrap()))
            .collect()
    }

    fn assemble(stream: &str, chunk: usize) -> anyhow::Result<CompletionResponse> {
        let mut sse = SseParser::default();
        let mut acc = StreamAccumulator::default();
        for part in stream.as_bytes().chunks(chunk) {
            for data in sse.push(part) {
                acc.event(&serde_json::from_str(&data)?)?;
            }
        }
        acc.finish()
    }

    fn sample_stream() -> String {
        sse(&[
            json!({"type": "message_start", "message": {"model": "claude-x", "usage": {
                "input_tokens": 10, "output_tokens": 1,
                "cache_read_input_tokens": 300, "cache_creation_input_tokens": 40}}}),
            json!({"type": "ping"}),
            json!({"type": "content_block_start", "index": 0, "content_block": {"type": "thinking", "thinking": ""}}),
            json!({"type": "content_block_delta", "index": 0, "delta": {"type": "thinking_delta", "thinking": "hm"}}),
            json!({"type": "content_block_stop", "index": 0}),
            json!({"type": "content_block_start", "index": 1, "content_block": {"type": "text", "text": ""}}),
            json!({"type": "content_block_delta", "index": 1, "delta": {"type": "text_delta", "text": "run"}}),
            json!({"type": "content_block_delta", "index": 1, "delta": {"type": "text_delta", "text": "ning é"}}),
            json!({"type": "content_block_stop", "index": 1}),
            json!({"type": "content_block_start", "index": 2, "content_block": {"type": "tool_use", "id": "t1", "name": "bash", "input": {}}}),
            json!({"type": "content_block_delta", "index": 2, "delta": {"type": "input_json_delta", "partial_json": "{\"input\": "}}),
            json!({"type": "content_block_delta", "index": 2, "delta": {"type": "input_json_delta", "partial_json": "\"ls\"}"}}),
            json!({"type": "content_block_stop", "index": 2}),
            json!({"type": "message_delta", "delta": {"stop_reason": "tool_use"}, "usage": {"output_tokens": 5}}),
            json!({"type": "message_stop"}),
        ])
    }

    #[test]
    fn assembles_text_tool_use_and_usage_from_a_stream() {
        // Whole, and split into awkward chunks (mid-line, mid-character).
        for chunk in [usize::MAX, 7, 1] {
            let r = assemble(&sample_stream(), chunk).unwrap();
            assert_eq!(r.stop_reason, StopReason::ToolUse);
            assert_eq!(r.model.as_deref(), Some("claude-x"));
            assert_eq!(
                r.content,
                vec![
                    ContentBlock::Text {
                        text: "running é".into()
                    },
                    ContentBlock::ToolUse {
                        id: "t1".into(),
                        name: "bash".into(),
                        input: json!({"input": "ls"}),
                    },
                ]
            );
            assert_eq!(
                r.usage,
                Some(Usage {
                    input_tokens: 10,
                    output_tokens: 5,
                    cache_read_tokens: 300,
                    cache_write_tokens: 40,
                })
            );
        }
    }

    #[test]
    fn text_deltas_are_reported_as_they_arrive() {
        let mut sse = SseParser::default();
        let mut acc = StreamAccumulator::default();
        let mut seen = Vec::new();
        for data in sse.push(sample_stream().as_bytes()) {
            seen.extend(acc.event(&serde_json::from_str(&data).unwrap()).unwrap());
        }
        assert_eq!(seen, ["run", "ning é"]);
    }

    #[test]
    fn tool_use_without_input_gets_an_empty_object() {
        let r = assemble(
            &sse(&[
                json!({"type": "message_start", "message": {}}),
                json!({"type": "content_block_start", "index": 0, "content_block": {"type": "tool_use", "id": "t", "name": "n", "input": {}}}),
                json!({"type": "message_delta", "delta": {"stop_reason": "tool_use"}}),
            ]),
            usize::MAX,
        )
        .unwrap();
        assert_eq!(
            r.content,
            vec![ContentBlock::ToolUse {
                id: "t".into(),
                name: "n".into(),
                input: json!({})
            }]
        );
        assert_eq!(r.usage, None);
    }

    #[test]
    fn truncated_stream_and_error_events_are_errors() {
        let cut = sse(&[
            json!({"type": "message_start", "message": {}}),
            json!({"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}}),
        ]);
        assert!(assemble(&cut, usize::MAX).is_err());
        let err = sse(&[
            json!({"type": "error", "error": {"type": "overloaded_error", "message": "busy"}}),
        ]);
        let e = assemble(&err, usize::MAX).unwrap_err();
        assert!(e.to_string().contains("overloaded_error"), "{e}");
    }

    #[test]
    fn sse_handles_crlf_and_ignores_other_fields() {
        let mut p = SseParser::default();
        let evs = p.push(b": comment\r\nevent: x\r\ndata: {\"a\":1}\r\n\r\ndata: 2\n\n");
        assert_eq!(evs, vec!["{\"a\":1}".to_owned(), "2".to_owned()]);
    }

    #[test]
    fn which_request_errors_are_transient() {
        let hint = |should_retry, after| RetryHint {
            should_retry,
            after,
        };
        let transient = |e: anyhow::Error| e.downcast_ref::<TransientError>().cloned();
        let e = transient(request_error(
            529,
            hint(None, Some(Duration::from_secs(3))),
            "overloaded",
        ));
        assert_eq!(e.unwrap().retry_after, Some(Duration::from_secs(3)));
        assert!(transient(request_error(429, hint(None, None), "rate limited")).is_some());
        assert!(transient(request_error(400, hint(None, None), "bad request")).is_none());
        assert!(transient(request_error(429, hint(None, None), "quota exceeded")).is_none());
        assert!(transient(request_error(503, hint(Some(false), None), "")).is_none());
        assert!(transient(request_error(400, hint(Some(true), None), "")).is_some());
    }
}
