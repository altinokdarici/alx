use crate::executor::ExecRequest;
use crate::tool::{Tool, ToolContext, ToolOutput};
use alx_protocol::ToolSpec;
use async_trait::async_trait;
use serde_json::{Value, json};

/// Runs a bash script through the context's executor.
#[derive(Debug, Default, Clone)]
pub struct BashTool;

#[async_trait]
impl Tool for BashTool {
    fn spec(&self) -> ToolSpec {
        ToolSpec {
            name: "bash".into(),
            description: "Run a bash script. The `input` string is piped to bash via stdin. \
                          Returns stdout, stderr and exit code."
                .into(),
            input_schema: json!({
                "type": "object",
                "properties": {
                    "input": { "type": "string", "description": "The bash script to execute" }
                },
                "required": ["input"]
            }),
        }
    }

    async fn call(&self, input: Value, ctx: &ToolContext) -> ToolOutput {
        let Some(script) = input.get("input").and_then(Value::as_str) else {
            return ToolOutput::error("Invalid tool call: bash requires a string `input`");
        };
        let req = ExecRequest {
            program: "bash".into(),
            stdin: Some(script.to_owned()),
            ..Default::default()
        };
        // Same result shape as the TypeScript agent. A non-zero exit is
        // reported in the result, not flagged as a tool error.
        let result = match ctx.executor.run(req).await {
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
