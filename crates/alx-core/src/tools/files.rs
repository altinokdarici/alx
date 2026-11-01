//! Reading and writing text files through the context's executor, so a
//! client that owns the files (an editor over ACP) can serve them.

use crate::tool::{Tool, ToolContext, ToolOutput};
use alx_protocol::ToolSpec;
use async_trait::async_trait;
use serde_json::{Value, json};
use std::path::Path;

/// Reads a text file, a window of lines at a time.
#[derive(Debug, Default, Clone)]
pub struct ReadFileTool;

impl ReadFileTool {
    /// Lines returned when the model doesn't pass `limit`.
    pub const DEFAULT_LIMIT: usize = 2000;
    /// Longest result in bytes; the rest is cut with a note to page on.
    pub const MAX_OUTPUT_BYTES: usize = 50_000;
}

fn path_arg(input: &Value, tool: &str) -> Result<String, ToolOutput> {
    match input.get("path").and_then(Value::as_str) {
        Some(p) if !p.is_empty() => Ok(p.to_owned()),
        _ => Err(ToolOutput::error(format!(
            "Invalid tool call: {tool} requires a string `path`"
        ))),
    }
}

/// Lines `offset..offset + limit` (1-based) of `text`, cut to `max_bytes`,
/// with a note saying where to continue when there is more.
fn window(text: &str, offset: usize, limit: usize, max_bytes: usize) -> String {
    let lines: Vec<&str> = text.lines().collect();
    let start = offset.saturating_sub(1).min(lines.len());
    let end = start.saturating_add(limit).min(lines.len());
    let mut out = String::new();
    let mut last = start;
    for line in &lines[start..end] {
        if out.len() + line.len() + 1 > max_bytes && last > start {
            break;
        }
        if line.len() > max_bytes {
            let mut cut = max_bytes;
            while !line.is_char_boundary(cut) {
                cut -= 1;
            }
            out.push_str(&line[..cut]);
            out.push_str(" [... line cut ...]\n");
        } else {
            out.push_str(line);
            out.push('\n');
        }
        last += 1;
    }
    if last < lines.len() {
        out.push_str(&format!(
            "[... {} more lines; read on with offset {} ...]\n",
            lines.len() - last,
            last + 1
        ));
    }
    out
}

#[async_trait]
impl Tool for ReadFileTool {
    fn spec(&self) -> ToolSpec {
        ToolSpec {
            name: "read_file".into(),
            description: "Read a UTF-8 text file. Returns up to `limit` lines (default 2000) \
                          starting at line `offset` (1-based, default 1), cut at 50000 bytes, \
                          with a note on where to continue if there is more."
                .into(),
            input_schema: json!({
                "type": "object",
                "properties": {
                    "path": { "type": "string", "description": "Path of the file to read" },
                    "offset": { "type": "integer", "description": "First line to return, 1-based" },
                    "limit": { "type": "integer", "description": "Most lines to return" }
                },
                "required": ["path"]
            }),
        }
    }

    fn read_only(&self) -> bool {
        true
    }

    async fn call(&self, input: Value, ctx: &ToolContext) -> ToolOutput {
        let path = match path_arg(&input, "read_file") {
            Ok(p) => p,
            Err(e) => return e,
        };
        let offset = input.get("offset").and_then(Value::as_u64).unwrap_or(1) as usize;
        let limit = input
            .get("limit")
            .and_then(Value::as_u64)
            .map_or(Self::DEFAULT_LIMIT, |l| l as usize);
        match ctx.executor.read_file(Path::new(&path)).await {
            Ok(text) => ToolOutput::ok(window(&text, offset, limit, Self::MAX_OUTPUT_BYTES)),
            Err(e) => ToolOutput::error(format!("Could not read {path}: {e}")),
        }
    }
}

/// Writes a whole text file, creating it if needed.
#[derive(Debug, Default, Clone)]
pub struct WriteFileTool;

#[async_trait]
impl Tool for WriteFileTool {
    fn spec(&self) -> ToolSpec {
        ToolSpec {
            name: "write_file".into(),
            description: "Write `content` to a text file, replacing it whole. Creates the file \
                          and its parent directories if needed."
                .into(),
            input_schema: json!({
                "type": "object",
                "properties": {
                    "path": { "type": "string", "description": "Path of the file to write" },
                    "content": { "type": "string", "description": "The full new file content" }
                },
                "required": ["path", "content"]
            }),
        }
    }

    async fn call(&self, input: Value, ctx: &ToolContext) -> ToolOutput {
        let path = match path_arg(&input, "write_file") {
            Ok(p) => p,
            Err(e) => return e,
        };
        let Some(content) = input.get("content").and_then(Value::as_str) else {
            return ToolOutput::error("Invalid tool call: write_file requires a string `content`");
        };
        match ctx.executor.write_file(Path::new(&path), content).await {
            Ok(()) => ToolOutput::ok(format!("Wrote {} bytes to {path}", content.len())),
            Err(e) => ToolOutput::error(format!("Could not write {path}: {e}")),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::executor::LocalExecutor;
    use std::sync::Arc;

    fn ctx() -> ToolContext {
        ToolContext {
            executor: Arc::new(LocalExecutor),
        }
    }

    #[test]
    fn window_pages_and_says_where_to_continue() {
        let text = "a\nb\nc\nd\n";
        assert_eq!(window(text, 1, 10, 1000), "a\nb\nc\nd\n");
        assert_eq!(
            window(text, 2, 2, 1000),
            "b\nc\n[... 1 more lines; read on with offset 4 ...]\n"
        );
        assert_eq!(
            window(text, 1, 10, 3),
            "a\n[... 3 more lines; read on with offset 2 ...]\n"
        );
        assert_eq!(window(text, 9, 10, 1000), "");
    }

    #[tokio::test]
    async fn writes_then_reads_a_file() {
        let dir = std::env::temp_dir().join(format!("alx-files-{}", std::process::id()));
        let path = dir.join("sub/x.txt");
        let path_s = path.to_string_lossy().into_owned();
        let out = WriteFileTool
            .call(json!({"path": path_s, "content": "one\ntwo\n"}), &ctx())
            .await;
        assert!(!out.is_error, "{}", out.content);
        let out = ReadFileTool
            .call(json!({"path": path_s, "offset": 2}), &ctx())
            .await;
        assert_eq!(out, ToolOutput::ok("two\n"));
        let _ = std::fs::remove_dir_all(dir);
    }

    #[tokio::test]
    async fn missing_file_and_bad_input_are_tool_errors() {
        let out = ReadFileTool
            .call(json!({"path": "/definitely/not/here"}), &ctx())
            .await;
        assert!(out.is_error);
        assert!(
            WriteFileTool
                .call(json!({"path": "x"}), &ctx())
                .await
                .is_error
        );
        assert!(ReadFileTool.call(json!({}), &ctx()).await.is_error);
    }
}
