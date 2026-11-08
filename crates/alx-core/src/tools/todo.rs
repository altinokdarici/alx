//! `todo_write`: the model's task list for the current piece of work.

use crate::tool::{Tool, ToolContext, ToolOutput};
use alx_protocol::ToolSpec;
use async_trait::async_trait;
use serde::Deserialize;
use serde_json::{Value, json};

/// One entry of a `todo_write` list.
#[derive(Debug, Clone, PartialEq, Eq, Deserialize)]
pub struct Todo {
    pub content: String,
    pub status: TodoStatus,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TodoStatus {
    Pending,
    InProgress,
    Completed,
}

/// Parses a `todo_write` input into its list, for clients that show it
/// (e.g. as an ACP plan).
pub fn parse_todos(input: &Value) -> Result<Vec<Todo>, String> {
    let todos = input
        .get("todos")
        .ok_or("todo_write needs a `todos` array")?;
    serde_json::from_value(todos.clone()).map_err(|e| format!("invalid `todos`: {e}"))
}

/// Replaces the whole task list each call. The list lives in the
/// conversation (the call's input), so the tool keeps no state.
#[derive(Debug, Default, Clone)]
pub struct TodoWriteTool;

#[async_trait]
impl Tool for TodoWriteTool {
    fn spec(&self) -> ToolSpec {
        ToolSpec {
            name: "todo_write".into(),
            description: "Write your task list for multi-step work, replacing the previous \
                          one. Send every item each time with its status; keep exactly one \
                          `in_progress` while working, and mark items `completed` as soon as \
                          they are done. Skip it for simple one-step requests."
                .into(),
            input_schema: json!({
                "type": "object",
                "properties": {
                    "todos": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "content": { "type": "string" },
                                "status": {
                                    "type": "string",
                                    "enum": ["pending", "in_progress", "completed"]
                                }
                            },
                            "required": ["content", "status"]
                        }
                    }
                },
                "required": ["todos"]
            }),
        }
    }

    fn read_only(&self) -> bool {
        true
    }

    async fn call(&self, input: Value, _ctx: &ToolContext) -> ToolOutput {
        match parse_todos(&input) {
            Ok(todos) if todos.is_empty() => ToolOutput::ok("Task list cleared."),
            Ok(todos) => {
                let lines: Vec<String> = todos
                    .iter()
                    .map(|t| {
                        let mark = match t.status {
                            TodoStatus::Pending => "[ ]",
                            TodoStatus::InProgress => "[~]",
                            TodoStatus::Completed => "[x]",
                        };
                        format!("{mark} {}", t.content)
                    })
                    .collect();
                ToolOutput::ok(lines.join("\n"))
            }
            Err(e) => ToolOutput::error(e),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn renders_the_list_and_rejects_bad_input() {
        let ctx = ToolContext {
            executor: std::sync::Arc::new(crate::LocalExecutor),
        };
        let out = TodoWriteTool
            .call(
                json!({"todos": [
                    {"content": "a", "status": "completed"},
                    {"content": "b", "status": "in_progress"},
                    {"content": "c", "status": "pending"}
                ]}),
                &ctx,
            )
            .await;
        assert_eq!(out, ToolOutput::ok("[x] a\n[~] b\n[ ] c"));
        let out = TodoWriteTool
            .call(json!({"todos": [{"content": "a", "status": "done"}]}), &ctx)
            .await;
        assert!(out.is_error);
    }
}
