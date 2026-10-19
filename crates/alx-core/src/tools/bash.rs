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
}

#[async_trait]
impl Tool for BashTool {
    fn spec(&self) -> ToolSpec {
        ToolSpec {
            name: "bash".into(),
            description: "Run a bash script. The `input` string is piped to bash via stdin. \
                          Returns stdout, stderr and exit code. \
                          The command is killed after `timeout_seconds` (default 120, max 1800)."
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
                "stdout": out.stdout,
                "stderr": format!(
                    "{}\n[alx] timed out after {}s and was killed",
                    out.stderr,
                    timeout.as_secs()
                ),
                "exitCode": out.exit_code,
                "timedOut": true,
            }),
            Ok(out) => json!({
                "stdout": out.stdout,
                "stderr": out.stderr,
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
}
