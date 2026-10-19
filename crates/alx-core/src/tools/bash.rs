use crate::executor::ExecRequest;
use crate::tool::{Tool, ToolContext, ToolOutput};
use alx_protocol::ToolSpec;
use async_trait::async_trait;
use serde_json::{Value, json};
use std::time::Duration;

/// Runs a bash script through the context's executor.
#[derive(Debug, Default, Clone)]
pub struct BashTool;

impl BashTool {
    /// How long a command may run unless the model asks for more.
    pub const DEFAULT_TIMEOUT: Duration = Duration::from_secs(120);
    /// The longest a command may run, whatever the model asks for.
    pub const MAX_TIMEOUT: Duration = Duration::from_secs(1800);
    /// Longest stdout or stderr (in bytes) passed back to the model. The
    /// result stays in the history and is re-sent on every request, so one
    /// noisy command would otherwise bloat the rest of the session.
    pub const MAX_OUTPUT_BYTES: usize = 30_000;
}

/// Shortens `s` to about `max` bytes by cutting out the middle, keeping the
/// start (what ran) and, more of, the end (where errors and summaries are).
fn truncate_middle(s: &str, max: usize) -> String {
    if s.len() <= max {
        return s.to_owned();
    }
    let mut head = max / 4;
    while !s.is_char_boundary(head) {
        head -= 1;
    }
    let mut tail = s.len() - (max - max / 4);
    while !s.is_char_boundary(tail) {
        tail += 1;
    }
    format!(
        "{}\n[... {} bytes truncated ...]\n{}",
        &s[..head],
        tail - head,
        &s[tail..]
    )
}

#[async_trait]
impl Tool for BashTool {
    fn spec(&self) -> ToolSpec {
        ToolSpec {
            name: "bash".into(),
            description: "Run a bash script. The `input` string is piped to bash via stdin. \
                          Returns stdout, stderr and exit code; \
                          output over 30000 bytes each is cut in the middle. The command is killed after `timeout_seconds` (default 120, max 1800)."
                .into(),
            input_schema: json!({
                "type": "object",
                "properties": {
                    "input": { "type": "string", "description": "The bash script to execute" },
                    "timeout_seconds": {
                        "type": "integer",
                        "description": "Kill the command after this many seconds (default 120, max 1800)"
                    }
                },
                "required": ["input"]
            }),
        }
    }

    async fn call(&self, input: Value, ctx: &ToolContext) -> ToolOutput {
        let Some(script) = input.get("input").and_then(Value::as_str) else {
            return ToolOutput::error("Invalid tool call: bash requires a string `input`");
        };
        let timeout = input
            .get("timeout_seconds")
            .and_then(Value::as_u64)
            .map_or(Self::DEFAULT_TIMEOUT, Duration::from_secs)
            .clamp(Duration::from_secs(1), Self::MAX_TIMEOUT);
        let req = ExecRequest {
            timeout: Some(timeout),
            program: "bash".into(),
            stdin: Some(script.to_owned()),
            ..Default::default()
        };
        // Same result shape as the TypeScript agent. A non-zero exit is
        // reported in the result, not flagged as a tool error.
        let result = match ctx.executor.run(req).await {
            Ok(out) if out.timed_out => json!({
                "stdout": truncate_middle(&out.stdout, Self::MAX_OUTPUT_BYTES),
                "stderr": format!(
                    "{}\n[alx] timed out after {}s and was killed",
                    truncate_middle(&out.stderr, Self::MAX_OUTPUT_BYTES),
                    timeout.as_secs()
                ),
                "exitCode": out.exit_code,
                "timedOut": true,
            }),
            Ok(out) => json!({
                "stdout": truncate_middle(&out.stdout, Self::MAX_OUTPUT_BYTES),
                "stderr": truncate_middle(&out.stderr, Self::MAX_OUTPUT_BYTES),
                "exitCode": out.exit_code,
            }),
            Err(err) => json!({ "stdout": "", "stderr": err.to_string(), "exitCode": -1 }),
        };
        ToolOutput::ok(result.to_string())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::executor::LocalExecutor;
    use std::sync::Arc;

    #[tokio::test]
    async fn reports_a_timeout_with_partial_output() {
        let ctx = ToolContext {
            executor: Arc::new(LocalExecutor),
        };
        let out = BashTool
            .call(
                json!({"input": "echo started; sleep 30", "timeout_seconds": 1}),
                &ctx,
            )
            .await;
        assert!(!out.is_error);
        let v: Value = serde_json::from_str(&out.content).unwrap();
        assert_eq!(v["timedOut"], true);
        assert_eq!(v["stdout"], "started\n");
        assert!(v["stderr"].as_str().unwrap().contains("timed out after 1s"));
    }

    #[test]
    fn short_output_is_untouched() {
        assert_eq!(truncate_middle("hello", 100), "hello");
        assert_eq!(truncate_middle("hello", 5), "hello");
    }

    #[test]
    fn long_output_keeps_start_and_more_of_the_end() {
        let s = format!("{}{}", "a".repeat(1000), "z".repeat(1000));
        let t = truncate_middle(&s, 400);
        assert!(t.starts_with(&"a".repeat(100)));
        assert!(t.ends_with(&"z".repeat(300)));
        assert!(t.contains("[... 1600 bytes truncated ...]"));
        assert!(t.len() < 500);
    }

    #[test]
    fn truncation_respects_char_boundaries() {
        let s = "é".repeat(500); // 2 bytes each
        let t = truncate_middle(&s, 101);
        assert!(t.contains("truncated"));
    }

    #[tokio::test]
    async fn huge_output_is_truncated_in_the_result() {
        let ctx = ToolContext {
            executor: Arc::new(LocalExecutor),
        };
        let out = BashTool
            .call(
                json!({"input": "head -c 200000 /dev/zero | tr '\\0' x"}),
                &ctx,
            )
            .await;
        let v: Value = serde_json::from_str(&out.content).unwrap();
        let stdout = v["stdout"].as_str().unwrap();
        assert!(stdout.len() < BashTool::MAX_OUTPUT_BYTES + 100);
        assert!(stdout.contains("bytes truncated"));
    }
}
