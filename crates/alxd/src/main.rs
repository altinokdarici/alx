//! alxd: the headless alx agent. No prompts, no human-readable output.
//!
//! For now it talks over stdio in JSON lines and drives the core directly;
//! the long-term transport (socket, WebSocket, ...) is still open.
//!
//! - stdin: one request per line, `{"text": "..."}`.
//! - stdout: one [`Event`] per line, in the `alx-protocol` wire format, plus
//!   `{"type": "error", "message": "..."}` when a request fails.
//!
//! The session lives for the whole process. EOF on stdin ends it.

use alx_core::protocol::{Event, Message};
use alx_core::{AgentLoop, EventSink, InMemorySessionStore, Session, SessionStore};
use serde::{Deserialize, Serialize};
use std::io::Write;
use std::process::ExitCode;
use tokio::io::{AsyncBufReadExt, BufReader};

/// One line of input.
#[derive(Debug, Deserialize)]
struct Request {
    text: String,
}

/// Output lines that are not core events.
#[derive(Debug, Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
enum Notice {
    Error { message: String },
}

/// Writes each value as one JSON line on stdout.
struct JsonLines;

impl JsonLines {
    fn write(&self, value: &impl Serialize) {
        let mut out = std::io::stdout().lock();
        // If stdout is gone there is nobody to report to, so ignore errors.
        if serde_json::to_writer(&mut out, value).is_ok() {
            let _ = out.write_all(b"\n");
            let _ = out.flush();
        }
    }

    fn error(&self, message: impl Into<String>) {
        self.write(&Notice::Error {
            message: message.into(),
        });
    }
}

impl EventSink for JsonLines {
    fn emit(&self, event: Event) {
        self.write(&event);
    }
}

fn parse_request(line: &str) -> Result<Request, String> {
    serde_json::from_str(line).map_err(|e| format!("invalid request: {e}"))
}

fn main() -> ExitCode {
    alx_core::load_dotenv();
    let runtime = tokio::runtime::Runtime::new().expect("failed to start tokio runtime");
    match runtime.block_on(run()) {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            JsonLines.error(format!("{e:#}"));
            ExitCode::FAILURE
        }
    }
}

async fn run() -> anyhow::Result<()> {
    let agent = alx_core::agent_from_env()?;
    let store = InMemorySessionStore::default();
    let mut session = Session::new("alxd");
    let out = JsonLines;
    let mut lines = BufReader::new(tokio::io::stdin()).lines();

    while let Some(line) = lines.next_line().await? {
        if line.trim().is_empty() {
            continue;
        }
        let request = match parse_request(&line) {
            Ok(r) => r,
            Err(e) => {
                out.error(e);
                continue;
            }
        };
        // The stop reason reaches the client as a `turn_finished` event.
        // On error the session is rolled back, so the client can retry.
        match agent
            .run_turn(&mut session, Message::user_text(request.text), &out)
            .await
        {
            Ok(_) => store.save(&session).await?,
            Err(e) => out.error(format!("{e:#}")),
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_requests() {
        assert_eq!(parse_request(r#"{"text":"hi"}"#).unwrap().text, "hi");
        assert!(parse_request("hi").is_err());
        assert!(parse_request(r#"{"prompt":"hi"}"#).is_err());
    }

    #[test]
    fn error_wire_shape() {
        let v = serde_json::to_value(Notice::Error {
            message: "boom".into(),
        })
        .unwrap();
        assert_eq!(v, serde_json::json!({"type": "error", "message": "boom"}));
    }
}
