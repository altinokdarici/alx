//! Where commands run. Local for now; Docker or remote later.

use async_trait::async_trait;
use std::path::PathBuf;
use std::process::Stdio;
use tokio::io::AsyncWriteExt;
use tokio::process::Command;

#[derive(Debug, Clone, Default)]
pub struct ExecRequest {
    pub program: String,
    pub args: Vec<String>,
    pub stdin: Option<String>,
    pub cwd: Option<PathBuf>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ExecOutput {
    pub stdout: String,
    pub stderr: String,
    /// `None` if the process was killed by a signal.
    pub exit_code: Option<i32>,
}

#[async_trait]
pub trait Executor: Send + Sync {
    /// Run a process to completion. `Err` means it could not be run at all
    /// (e.g. spawn failed). A non-zero exit is still `Ok`.
    async fn run(&self, req: ExecRequest) -> std::io::Result<ExecOutput>;
}

/// Runs processes on the host machine.
#[derive(Debug, Default, Clone)]
pub struct LocalExecutor;

#[async_trait]
impl Executor for LocalExecutor {
    async fn run(&self, req: ExecRequest) -> std::io::Result<ExecOutput> {
        let mut cmd = Command::new(&req.program);
        cmd.args(&req.args)
            .stdin(if req.stdin.is_some() {
                Stdio::piped()
            } else {
                Stdio::null()
            })
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .kill_on_drop(true);
        if let Some(cwd) = &req.cwd {
            cmd.current_dir(cwd);
        }
        let mut child = cmd.spawn()?;

        // Write stdin on its own task so a large input can't deadlock against
        // the child filling its stdout/stderr pipes.
        let writer = match (req.stdin, child.stdin.take()) {
            (Some(input), Some(mut pipe)) => Some(tokio::spawn(async move {
                // The child may exit without reading all input; ignore EPIPE.
                let _ = pipe.write_all(input.as_bytes()).await;
            })),
            _ => None,
        };

        let out = child.wait_with_output().await?;
        if let Some(w) = writer {
            let _ = w.await;
        }
        Ok(ExecOutput {
            stdout: String::from_utf8_lossy(&out.stdout).into_owned(),
            stderr: String::from_utf8_lossy(&out.stderr).into_owned(),
            exit_code: out.status.code(),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn runs_bash_with_stdin() {
        let out = LocalExecutor
            .run(ExecRequest {
                program: "bash".into(),
                stdin: Some("echo hi; echo err >&2; exit 3".into()),
                ..Default::default()
            })
            .await
            .unwrap();
        assert_eq!(out.stdout, "hi\n");
        assert_eq!(out.stderr, "err\n");
        assert_eq!(out.exit_code, Some(3));
    }

    #[tokio::test]
    async fn spawn_failure_is_err() {
        let res = LocalExecutor
            .run(ExecRequest {
                program: "/definitely/not/here".into(),
                ..Default::default()
            })
            .await;
        assert!(res.is_err());
    }
}
