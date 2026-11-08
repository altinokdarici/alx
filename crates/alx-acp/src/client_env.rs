//! Tool environment backed by the ACP client: commands run in the client's
//! terminals, files are read and written through the client (so an editor
//! sees unsaved buffers and records the edits), and tool calls are approved
//! by the user through `session/request_permission`.
//!
//! Each capability falls back to the local machine when the client doesn't
//! offer it.

use crate::events::{tool_kind, tool_title};
use agent_client_protocol::schema::v1::{
    ClientCapabilities, CreateTerminalRequest, KillTerminalRequest, PermissionOption,
    PermissionOptionKind, ReadTextFileRequest, ReleaseTerminalRequest, RequestPermissionOutcome,
    RequestPermissionRequest, SessionId, TerminalId, TerminalOutputRequest, ToolCallUpdate,
    ToolCallUpdateFields, WaitForTerminalExitRequest, WriteTextFileRequest,
};
use agent_client_protocol::{Client, ConnectionTo};
use alx_core::{Approval, ExecOutput, ExecRequest, Executor, LocalExecutor, ToolApprover};
use async_trait::async_trait;
use serde_json::Value;
use std::collections::HashMap;
use std::io;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

/// Most terminal output the client keeps for one command. `BashTool` cuts it
/// further before it reaches the model.
const TERMINAL_OUTPUT_LIMIT: u64 = 1_000_000;

fn rpc_err(e: agent_client_protocol::Error) -> io::Error {
    io::Error::other(format!("ACP client: {e}"))
}

/// Runs commands and file operations through the client where it can.
pub struct ClientExecutor {
    pub cx: ConnectionTo<Client>,
    pub session: SessionId,
    /// The session's working directory: where commands run and relative
    /// paths start from.
    pub cwd: PathBuf,
    pub caps: ClientCapabilities,
}

impl ClientExecutor {
    fn resolve(&self, path: &Path) -> PathBuf {
        if path.is_absolute() {
            path.to_owned()
        } else {
            self.cwd.join(path)
        }
    }

    async fn run_in_terminal(
        &self,
        command: String,
        args: Vec<String>,
        cwd: PathBuf,
        timeout: Option<std::time::Duration>,
    ) -> io::Result<ExecOutput> {
        let created = self
            .cx
            .send_request(
                CreateTerminalRequest::new(self.session.clone(), command)
                    .args(args)
                    .cwd(cwd)
                    .output_byte_limit(TERMINAL_OUTPUT_LIMIT),
            )
            .block_task()
            .await
            .map_err(rpc_err)?;
        let mut terminal = TerminalGuard {
            cx: self.cx.clone(),
            session: self.session.clone(),
            id: Some(created.terminal_id),
        };
        let id = terminal.id.clone().expect("just set");

        let wait = self
            .cx
            .send_request(WaitForTerminalExitRequest::new(
                self.session.clone(),
                id.clone(),
            ))
            .block_task();
        let (exit, timed_out) = match timeout {
            Some(limit) => match tokio::time::timeout(limit, wait).await {
                Ok(done) => (Some(done.map_err(rpc_err)?.exit_status), false),
                Err(_) => {
                    let _ = self
                        .cx
                        .send_request(KillTerminalRequest::new(self.session.clone(), id.clone()))
                        .block_task()
                        .await;
                    (None, true)
                }
            },
            None => (Some(wait.await.map_err(rpc_err)?.exit_status), false),
        };
        let output = self
            .cx
            .send_request(TerminalOutputRequest::new(self.session.clone(), id))
            .block_task()
            .await
            .map_err(rpc_err)?;
        terminal.release().await;

        let mut stdout = output.output;
        if output.truncated {
            stdout.insert_str(0, "[... earlier output dropped by the client ...]\n");
        }
        Ok(ExecOutput {
            stdout,
            // A terminal interleaves both streams into one.
            stderr: String::new(),
            exit_code: exit.and_then(|e| e.exit_code).map(|c| c as i32),
            timed_out,
        })
    }
}

/// Kills and releases a client terminal unless released normally, so a
/// cancelled command doesn't keep running in the editor.
struct TerminalGuard {
    cx: ConnectionTo<Client>,
    session: SessionId,
    id: Option<TerminalId>,
}

impl TerminalGuard {
    async fn release(&mut self) {
        if let Some(id) = self.id.take() {
            let _ = self
                .cx
                .send_request(ReleaseTerminalRequest::new(self.session.clone(), id))
                .block_task()
                .await;
        }
    }
}

impl Drop for TerminalGuard {
    fn drop(&mut self) {
        let Some(id) = self.id.take() else { return };
        let (cx, session) = (self.cx.clone(), self.session.clone());
        tokio::spawn(async move {
            let _ = cx
                .send_request(KillTerminalRequest::new(session.clone(), id.clone()))
                .block_task()
                .await;
            let _ = cx
                .send_request(ReleaseTerminalRequest::new(session, id))
                .block_task()
                .await;
        });
    }
}

fn is_shell(program: &str) -> bool {
    matches!(
        Path::new(program).file_name().and_then(|n| n.to_str()),
        Some("bash" | "sh" | "zsh" | "dash")
    )
}

#[async_trait]
impl Executor for ClientExecutor {
    async fn run(&self, mut req: ExecRequest) -> io::Result<ExecOutput> {
        let cwd = req.cwd.take().unwrap_or_else(|| self.cwd.clone());
        // A terminal has no stdin, so only a shell script can move there
        // (as `-c`); anything else that needs stdin runs locally.
        let in_terminal = self.caps.terminal
            && match &req.stdin {
                None => true,
                Some(_) => req.args.is_empty() && is_shell(&req.program),
            };
        if !in_terminal {
            req.cwd = Some(cwd);
            return LocalExecutor.run(req).await;
        }
        let args = match req.stdin.take() {
            Some(script) => vec!["-c".to_owned(), script],
            None => std::mem::take(&mut req.args),
        };
        self.run_in_terminal(req.program, args, cwd, req.timeout)
            .await
    }

    async fn read_file(&self, path: &Path) -> io::Result<String> {
        let path = self.resolve(path);
        if !self.caps.fs.read_text_file {
            return LocalExecutor.read_file(&path).await;
        }
        let resp = self
            .cx
            .send_request(ReadTextFileRequest::new(self.session.clone(), path))
            .block_task()
            .await
            .map_err(rpc_err)?;
        Ok(resp.content)
    }

    async fn read_bytes(&self, path: &Path) -> io::Result<Vec<u8>> {
        LocalExecutor.read_bytes(&self.resolve(path)).await
    }

    async fn write_file(&self, path: &Path, content: &str) -> io::Result<()> {
        let path = self.resolve(path);
        if !self.caps.fs.write_text_file {
            return LocalExecutor.write_file(&path, content).await;
        }
        self.cx
            .send_request(WriteTextFileRequest::new(
                self.session.clone(),
                path,
                content,
            ))
            .block_task()
            .await
            .map_err(rpc_err)?;
        Ok(())
    }
}

const ALLOW_ONCE: &str = "allow_once";
const ALLOW_ALWAYS: &str = "allow_always";
const REJECT_ONCE: &str = "reject_once";
const REJECT_ALWAYS: &str = "reject_always";

/// Asks the user, through the client, before a tool call runs. "Always"
/// answers are remembered per tool for the rest of the session.
pub struct ClientApprover {
    pub cx: ConnectionTo<Client>,
    pub session: SessionId,
    pub always: Arc<Mutex<HashMap<String, Approval>>>,
}

#[async_trait]
impl ToolApprover for ClientApprover {
    async fn approve(&self, id: &str, name: &str, input: &Value) -> Approval {
        if let Some(remembered) = self.always.lock().unwrap().get(name) {
            return *remembered;
        }
        let call = ToolCallUpdate::new(
            id.to_owned(),
            ToolCallUpdateFields::new()
                .title(tool_title(name, input))
                .kind(tool_kind(name))
                .raw_input(input.clone()),
        );
        let options = vec![
            PermissionOption::new(ALLOW_ONCE, "Allow", PermissionOptionKind::AllowOnce),
            PermissionOption::new(
                ALLOW_ALWAYS,
                format!("Always allow {name}"),
                PermissionOptionKind::AllowAlways,
            ),
            PermissionOption::new(REJECT_ONCE, "Reject", PermissionOptionKind::RejectOnce),
            PermissionOption::new(
                REJECT_ALWAYS,
                format!("Always reject {name}"),
                PermissionOptionKind::RejectAlways,
            ),
        ];
        let answer = self
            .cx
            .send_request(RequestPermissionRequest::new(
                self.session.clone(),
                call,
                options,
            ))
            .block_task()
            .await;
        let Ok(answer) = answer else {
            return Approval::Deny;
        };
        let RequestPermissionOutcome::Selected(selected) = answer.outcome else {
            return Approval::Deny;
        };
        let (approval, remember) = match &*selected.option_id.0 {
            ALLOW_ONCE => (Approval::Allow, false),
            ALLOW_ALWAYS => (Approval::Allow, true),
            REJECT_ALWAYS => (Approval::Deny, true),
            _ => (Approval::Deny, false),
        };
        if remember {
            self.always
                .lock()
                .unwrap()
                .insert(name.to_owned(), approval);
        }
        approval
    }
}
