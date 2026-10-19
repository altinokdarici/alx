//! Where commands run. Local for now; Docker or remote later.

use async_trait::async_trait;
use std::path::PathBuf;
use std::process::Stdio;
use std::time::Duration;
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::process::Command;
use tokio::task::JoinHandle;

#[derive(Debug, Clone, Default)]
pub struct ExecRequest {
    pub program: String,
    pub args: Vec<String>,
    pub stdin: Option<String>,
    pub cwd: Option<PathBuf>,
    /// Kill the process (and its children) after this long.
    pub timeout: Option<Duration>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ExecOutput {
    pub stdout: String,
    pub stderr: String,
    /// `None` if the process was killed by a signal (including on timeout).
    pub exit_code: Option<i32>,
    /// The process hit `ExecRequest::timeout` and was killed. Output is
    /// whatever it wrote before that.
    pub timed_out: bool,
}

#[async_trait]
pub trait Executor: Send + Sync {
    /// Run a process to completion. `Err` means it could not be run at all
    /// (e.g. spawn failed). A non-zero exit is still `Ok`.
    ///
    /// Dropping the returned future (e.g. on cancellation) must stop the
    /// process.
    async fn run(&self, req: ExecRequest) -> std::io::Result<ExecOutput>;
}

/// Runs processes on the host machine.
#[derive(Debug, Default, Clone)]
pub struct LocalExecutor;

/// Kills the child's whole process group when dropped, so a cancelled or
/// timed-out `bash` doesn't leave its children (`sleep`, servers, ...)
/// running. Disarmed once the command finishes on its own.
struct KillGroupOnDrop(Option<u32>);

impl KillGroupOnDrop {
    fn kill(&mut self) {
        #[cfg(unix)]
        if let Some(pid) = self.0.take() {
            // SAFETY: plain kill(2) on the process group we created with
            // `process_group(0)`; the pid is our child's.
            unsafe {
                libc::kill(-(pid as i32), libc::SIGKILL);
            }
        }
    }
    fn disarm(&mut self) {
        self.0 = None;
    }
}

impl Drop for KillGroupOnDrop {
    fn drop(&mut self) {
        self.kill();
    }
}

fn read_all<R: tokio::io::AsyncRead + Unpin + Send + 'static>(mut pipe: R) -> JoinHandle<Vec<u8>> {
    tokio::spawn(async move {
        let mut buf = Vec::new();
        let _ = pipe.read_to_end(&mut buf).await;
        buf
    })
}

/// After a kill the pipes normally close at once. A grandchild that escaped
/// the process group could keep them open, so don't wait for it for long.
async fn collect(task: JoinHandle<Vec<u8>>, bounded: bool) -> Vec<u8> {
    if !bounded {
        return task.await.unwrap_or_default();
    }
    let abort = task.abort_handle();
    match tokio::time::timeout(Duration::from_secs(1), task).await {
        Ok(res) => res.unwrap_or_default(),
        Err(_) => {
            abort.abort();
            Vec::new()
        }
    }
}

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
        #[cfg(unix)]
        cmd.process_group(0);
        if let Some(cwd) = &req.cwd {
            cmd.current_dir(cwd);
        }
        let mut child = cmd.spawn()?;
        let mut guard = KillGroupOnDrop(child.id());

        // Write stdin on its own task so a large input can't deadlock against
        // the child filling its stdout/stderr pipes.
        let writer = match (req.stdin, child.stdin.take()) {
            (Some(input), Some(mut pipe)) => Some(tokio::spawn(async move {
                // The child may exit without reading all input; ignore EPIPE.
                let _ = pipe.write_all(input.as_bytes()).await;
            })),
            _ => None,
        };
        let stdout = read_all(child.stdout.take().expect("stdout is piped"));
        let stderr = read_all(child.stderr.take().expect("stderr is piped"));

        let (status, timed_out) = match req.timeout {
            Some(limit) => match tokio::time::timeout(limit, child.wait()).await {
                Ok(status) => (Some(status?), false),
                Err(_) => {
                    guard.kill();
                    let _ = child.kill().await;
                    (None, true)
                }
            },
            None => (Some(child.wait().await?), false),
        };
        let stdout = collect(stdout, timed_out).await;
        let stderr = collect(stderr, timed_out).await;
        guard.disarm();
        if let Some(w) = writer {
            let _ = w.await;
        }
        Ok(ExecOutput {
            stdout: String::from_utf8_lossy(&stdout).into_owned(),
            stderr: String::from_utf8_lossy(&stderr).into_owned(),
            exit_code: status.and_then(|s| s.code()),
            timed_out,
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

    #[tokio::test]
    async fn timeout_kills_the_process_and_keeps_partial_output() {
        let started = std::time::Instant::now();
        let out = LocalExecutor
            .run(ExecRequest {
                program: "bash".into(),
                stdin: Some("echo before; sleep 30; echo after".into()),
                timeout: Some(Duration::from_millis(300)),
                ..Default::default()
            })
            .await
            .unwrap();
        assert!(out.timed_out);
        assert_eq!(out.exit_code, None);
        assert_eq!(out.stdout, "before\n");
        assert!(started.elapsed() < Duration::from_secs(10));
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn dropping_the_future_kills_the_process_group() {
        let marker = std::env::temp_dir().join(format!("alx-drop-{}", std::process::id()));
        let _ = std::fs::remove_file(&marker);
        let script = format!("(sleep 1; touch {}) & wait", marker.display());
        let run = LocalExecutor.run(ExecRequest {
            program: "bash".into(),
            stdin: Some(script),
            ..Default::default()
        });
        // Cancel after the shell has had time to start its child.
        let _ = tokio::time::timeout(Duration::from_millis(300), run).await;
        tokio::time::sleep(Duration::from_millis(1500)).await;
        assert!(!marker.exists(), "background child survived the drop");
    }
}
