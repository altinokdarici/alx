//! A real ACP client talking to the server over the native transport.

use agent_client_protocol::schema::ProtocolVersion;
use agent_client_protocol::schema::v1::{
    ClientCapabilities, CompactionStatus, ContentBlock, CreateTerminalRequest,
    CreateTerminalResponse, FileSystemCapabilities, InitializeRequest, NewSessionRequest,
    PromptRequest, PromptResponse, ReadTextFileRequest, ReadTextFileResponse,
    ReleaseTerminalRequest, ReleaseTerminalResponse, RequestPermissionOutcome,
    RequestPermissionRequest, RequestPermissionResponse, SelectedPermissionOutcome,
    SessionNotification, SessionUpdate, StopReason, TerminalExitStatus, TerminalOutputRequest,
    TerminalOutputResponse, TextContent, ToolCallStatus, ToolKind, WaitForTerminalExitRequest,
    WaitForTerminalExitResponse, WriteTextFileRequest, WriteTextFileResponse,
};
use agent_client_protocol::{Agent, Client, ConnectionTo};
use alx_core::protocol::{
    Compaction, ContentBlock as Block, ProviderInfo, StopReason as Stop, Usage,
};
use alx_core::{
    AgentConfig, BashTool, CompletionRequest, CompletionResponse, InMemorySessionStore,
    LocalExecutor, Provider, ReadFileTool, SingleAgent, ToolContext, ToolRegistry, WriteFileTool,
};
use async_trait::async_trait;
use serde_json::json;
use std::collections::VecDeque;
use std::sync::{Arc, Mutex};

struct Scripted(Mutex<VecDeque<CompletionResponse>>);

fn reply(content: Vec<Block>) -> CompletionResponse {
    let stop_reason = if content.iter().any(|b| matches!(b, Block::ToolUse { .. })) {
        Stop::ToolUse
    } else {
        Stop::EndTurn
    };
    CompletionResponse {
        content,
        stop_reason,
        model: None,
        usage: None,
        context: None,
        compactions: vec![],
    }
}

fn call(id: &str, name: &str, input: serde_json::Value) -> Block {
    Block::ToolUse {
        id: id.into(),
        name: name.into(),
        input,
    }
}

fn text(t: &str) -> Block {
    Block::Text { text: t.into() }
}

#[async_trait]
impl Provider for Scripted {
    fn info(&self) -> ProviderInfo {
        ProviderInfo {
            name: "scripted".into(),
            model: None,
        }
    }
    async fn complete(&self, _req: CompletionRequest) -> anyhow::Result<CompletionResponse> {
        Ok(self.0.lock().unwrap().pop_front().expect("no more replies"))
    }
}

/// What the fake client was asked to do, in order.
#[derive(Default)]
struct Log {
    updates: Vec<SessionUpdate>,
    permissions: Vec<String>,
    requests: Vec<String>,
}

struct Run {
    log: Log,
    replies: Vec<PromptResponse>,
}

/// Serves an agent with `replies`, then connects a client with `caps` that
/// answers permission requests with `permission` and sends `prompts`.
async fn run(
    replies: Vec<CompletionResponse>,
    caps: ClientCapabilities,
    permission: &'static str,
    prompts: &[&str],
    cwd: &'static str,
) -> Run {
    let agent = Arc::new(SingleAgent {
        config: AgentConfig::default(),
        provider: Arc::new(Scripted(Mutex::new(replies.into()))),
        tools: ToolRegistry::new()
            .with(BashTool)
            .with(ReadFileTool)
            .with(WriteFileTool),
        tool_ctx: ToolContext {
            executor: Arc::new(LocalExecutor),
        },
    });
    let client_end = alx_acp::native(agent, Arc::new(InMemorySessionStore::default()));

    let log = Arc::new(Mutex::new(Log::default()));
    let (l1, l2, l3, l4, l5, l6, l7, l8) = (
        log.clone(),
        log.clone(),
        log.clone(),
        log.clone(),
        log.clone(),
        log.clone(),
        log.clone(),
        log.clone(),
    );
    let prompts: Vec<String> = prompts.iter().map(|p| p.to_string()).collect();
    let replies = Client
        .builder()
        .on_receive_notification(
            async move |n: SessionNotification, _cx| {
                l1.lock().unwrap().updates.push(n.update);
                Ok(())
            },
            agent_client_protocol::on_receive_notification!(),
        )
        .on_receive_request(
            async move |r: RequestPermissionRequest, responder, _cx| {
                l2.lock()
                    .unwrap()
                    .permissions
                    .push(r.tool_call.tool_call_id.0.to_string());
                responder.respond(RequestPermissionResponse::new(
                    RequestPermissionOutcome::Selected(SelectedPermissionOutcome::new(permission)),
                ))
            },
            agent_client_protocol::on_receive_request!(),
        )
        .on_receive_request(
            async move |r: CreateTerminalRequest, responder, _cx| {
                l3.lock().unwrap().requests.push(format!(
                    "create {} {:?} {}",
                    r.command,
                    r.args,
                    r.cwd.unwrap().display()
                ));
                responder.respond(CreateTerminalResponse::new("term-1"))
            },
            agent_client_protocol::on_receive_request!(),
        )
        .on_receive_request(
            async move |_r: WaitForTerminalExitRequest, responder, _cx| {
                l4.lock().unwrap().requests.push("wait".into());
                responder.respond(WaitForTerminalExitResponse::new(
                    TerminalExitStatus::new().exit_code(3),
                ))
            },
            agent_client_protocol::on_receive_request!(),
        )
        .on_receive_request(
            async move |_r: TerminalOutputRequest, responder, _cx| {
                l5.lock().unwrap().requests.push("output".into());
                responder.respond(TerminalOutputResponse::new("from terminal\n", false))
            },
            agent_client_protocol::on_receive_request!(),
        )
        .on_receive_request(
            async move |_r: ReleaseTerminalRequest, responder, _cx| {
                l6.lock().unwrap().requests.push("release".into());
                responder.respond(ReleaseTerminalResponse::new())
            },
            agent_client_protocol::on_receive_request!(),
        )
        .on_receive_request(
            async move |r: ReadTextFileRequest, responder, _cx| {
                l7.lock()
                    .unwrap()
                    .requests
                    .push(format!("read {}", r.path.display()));
                responder.respond(ReadTextFileResponse::new("buffer text\n"))
            },
            agent_client_protocol::on_receive_request!(),
        )
        .on_receive_request(
            async move |r: WriteTextFileRequest, responder, _cx| {
                l8.lock().unwrap().requests.push(format!(
                    "write {} {:?}",
                    r.path.display(),
                    r.content
                ));
                responder.respond(WriteTextFileResponse::new())
            },
            agent_client_protocol::on_receive_request!(),
        )
        .connect_with(client_end, async move |cx: ConnectionTo<Agent>| {
            cx.send_request(InitializeRequest::new(ProtocolVersion::V1).client_capabilities(caps))
                .block_task()
                .await?;
            let session = cx
                .send_request(NewSessionRequest::new(cwd))
                .block_task()
                .await?
                .session_id;
            let mut replies = Vec::new();
            for p in prompts {
                replies.push(
                    cx.send_request(PromptRequest::new(
                        session.clone(),
                        vec![ContentBlock::Text(TextContent::new(p))],
                    ))
                    .block_task()
                    .await?,
                );
            }
            Ok(replies)
        })
        .await
        .unwrap();
    let log = std::mem::take(&mut *log.lock().unwrap());
    Run { log, replies }
}

fn tool_results(updates: &[SessionUpdate]) -> Vec<(ToolCallStatus, String)> {
    updates
        .iter()
        .filter_map(|u| match u {
            SessionUpdate::ToolCallUpdate(u) => {
                let out = serde_json::to_value(&u.fields.content).unwrap();
                Some((u.fields.status.unwrap(), out.to_string()))
            }
            _ => None,
        })
        .collect()
}

#[tokio::test]
async fn prompt_streams_updates_and_reports_usage() {
    let mut last = reply(vec![text("done")]);
    last.usage = Some(Usage {
        input_tokens: 10,
        output_tokens: 5,
        cache_read_tokens: 100,
        cache_write_tokens: 1,
    });
    let run = run(
        vec![
            reply(vec![call("t1", "bash", json!({"input": "echo hello"}))]),
            last,
        ],
        ClientCapabilities::new(),
        "allow_once",
        &["hi"],
        "/tmp",
    )
    .await;

    let reply = &run.replies[0];
    assert_eq!(reply.stop_reason, StopReason::EndTurn);
    let usage = reply.usage.as_ref().expect("usage");
    assert_eq!(
        (usage.total_tokens, usage.input_tokens, usage.output_tokens),
        (116, 111, 5)
    );
    assert_eq!(usage.cached_read_tokens, Some(100));

    let kinds: Vec<&str> = run
        .log
        .updates
        .iter()
        .map(|u| match u {
            SessionUpdate::AvailableCommandsUpdate(c) => {
                assert_eq!(c.available_commands[0].name, "compact");
                "commands"
            }
            SessionUpdate::ToolCall(c) => {
                assert_eq!(c.kind, ToolKind::Execute);
                "tool_call"
            }
            SessionUpdate::ToolCallUpdate(_) => "tool_update",
            SessionUpdate::AgentMessageChunk(_) => "text",
            _ => "other",
        })
        .collect();
    assert_eq!(kinds, ["commands", "tool_call", "tool_update", "text"]);
    assert_eq!(run.log.permissions, ["t1"]);
    // No terminal capability: bash ran on this machine.
    assert!(run.log.requests.is_empty());
    let results = tool_results(&run.log.updates);
    assert_eq!(results[0].0, ToolCallStatus::Completed);
    assert!(results[0].1.contains("hello"), "{}", results[0].1);
}

#[tokio::test]
async fn tools_use_the_clients_terminal_and_files() {
    let caps = ClientCapabilities::new()
        .terminal(true)
        .fs(FileSystemCapabilities::new()
            .read_text_file(true)
            .write_text_file(true));
    let run = run(
        vec![
            reply(vec![
                call("t1", "bash", json!({"input": "make"})),
                call(
                    "t2",
                    "write_file",
                    json!({"path": "a.txt", "content": "new"}),
                ),
                call("t3", "read_file", json!({"path": "/abs/b.txt"})),
                call("t4", "bash", json!({"input": "make again"})),
            ]),
            reply(vec![text("done")]),
        ],
        caps,
        "allow_always",
        &["hi"],
        "/work",
    )
    .await;

    assert_eq!(
        run.log.requests,
        [
            r#"create bash ["-c", "make"] /work"#,
            "wait",
            "output",
            "release",
            r#"write /work/a.txt "new""#,
            "read /abs/b.txt",
            r#"create bash ["-c", "make again"] /work"#,
            "wait",
            "output",
            "release",
        ]
    );
    // "Always" is remembered per tool; read_file never asks.
    assert_eq!(run.log.permissions, ["t1", "t2"]);
    let results = tool_results(&run.log.updates);
    assert!(results[0].1.contains("from terminal"), "{}", results[0].1);
    assert!(
        results[0].1.contains(r#"\"exitCode\":3"#),
        "{}",
        results[0].1
    );
    assert!(results[2].1.contains("buffer text"), "{}", results[2].1);
    let kinds: Vec<ToolKind> = run
        .log
        .updates
        .iter()
        .filter_map(|u| match u {
            SessionUpdate::ToolCall(c) => Some(c.kind),
            _ => None,
        })
        .collect();
    assert_eq!(
        kinds,
        [
            ToolKind::Execute,
            ToolKind::Edit,
            ToolKind::Read,
            ToolKind::Execute
        ]
    );
}

#[tokio::test]
async fn a_rejected_tool_call_does_not_run() {
    let caps = ClientCapabilities::new().terminal(true);
    let run = run(
        vec![
            reply(vec![call("t1", "bash", json!({"input": "rm -rf /"}))]),
            reply(vec![text("ok, I won't")]),
        ],
        caps,
        "reject_once",
        &["hi"],
        "/work",
    )
    .await;
    assert_eq!(run.log.permissions, ["t1"]);
    assert!(run.log.requests.is_empty(), "{:?}", run.log.requests);
    let results = tool_results(&run.log.updates);
    assert_eq!(results[0].0, ToolCallStatus::Failed);
    assert!(results[0].1.contains("denied"), "{}", results[0].1);
}

#[tokio::test]
async fn compactions_become_compaction_updates() {
    let mut compacted = reply(vec![text("ok")]);
    compacted.compactions = vec![Compaction {
        messages_removed: 4,
        tokens_before: Some(1000),
        tokens_after: Some(100),
    }];
    // The scripted provider can't compact on request, so `/compact` fails.
    let run = run(
        vec![compacted],
        ClientCapabilities::new(),
        "allow_once",
        &["hi", "/compact"],
        "/tmp",
    )
    .await;
    assert_eq!(run.replies[1].stop_reason, StopReason::EndTurn);
    let compactions: Vec<_> = run
        .log
        .updates
        .iter()
        .filter_map(|u| match u {
            SessionUpdate::CompactionUpdate(c) => Some(c),
            _ => None,
        })
        .collect();
    assert_eq!(compactions.len(), 3);
    assert_eq!(compactions[0].status, CompactionStatus::Completed);
    let meta = serde_json::to_value(&compactions[0].meta).unwrap();
    assert_eq!(meta["messagesRemoved"], 4);
    assert_eq!(meta["tokensAfter"], 100);
    assert_eq!(compactions[1].status, CompactionStatus::InProgress);
    assert_eq!(compactions[2].status, CompactionStatus::Failed);
    assert_eq!(compactions[1].compaction_id, compactions[2].compaction_id);
}

#[tokio::test]
async fn sessions_are_listed_and_resumed_without_replay() {
    use agent_client_protocol::schema::v1::{ErrorCode, ListSessionsRequest, ResumeSessionRequest};
    let agent = Arc::new(SingleAgent {
        config: AgentConfig::default(),
        provider: Arc::new(Scripted(Mutex::new(
            vec![
                reply(vec![text("one")]),
                reply(vec![text("two")]),
                reply(vec![text("three")]),
            ]
            .into(),
        ))),
        tools: ToolRegistry::new(),
        tool_ctx: ToolContext {
            executor: Arc::new(LocalExecutor),
        },
    });
    let store = Arc::new(InMemorySessionStore::default());
    let updates = Arc::new(Mutex::new(Vec::new()));
    let seen = updates.clone();
    let prompt =
        |id, t: &str| PromptRequest::new(id, vec![ContentBlock::Text(TextContent::new(t))]);
    Client
        .builder()
        .on_receive_notification(
            async move |n: SessionNotification, _cx| {
                seen.lock().unwrap().push(n.update);
                Ok(())
            },
            agent_client_protocol::on_receive_notification!(),
        )
        .connect_with(
            alx_acp::native(agent, store),
            async move |cx: ConnectionTo<Agent>| {
                let init = cx
                    .send_request(InitializeRequest::new(ProtocolVersion::V1))
                    .block_task()
                    .await?;
                let caps = init.agent_capabilities.session_capabilities;
                assert!(caps.list.is_some() && caps.resume.is_some());

                let a = cx
                    .send_request(NewSessionRequest::new("/a"))
                    .block_task()
                    .await?
                    .session_id;
                cx.send_request(prompt(a.clone(), "first"))
                    .block_task()
                    .await?;
                let b = cx
                    .send_request(NewSessionRequest::new("/b"))
                    .block_task()
                    .await?
                    .session_id;
                cx.send_request(prompt(b.clone(), "other"))
                    .block_task()
                    .await?;

                let all = cx
                    .send_request(ListSessionsRequest::new())
                    .block_task()
                    .await?;
                assert_eq!(all.sessions.len(), 2);
                assert_eq!(all.next_cursor, None);
                let in_a = cx
                    .send_request(ListSessionsRequest::new().cwd(std::path::PathBuf::from("/a")))
                    .block_task()
                    .await?;
                assert_eq!(in_a.sessions.len(), 1);
                let info = &in_a.sessions[0];
                assert_eq!(info.session_id, a);
                assert_eq!(info.title.as_deref(), Some("first"));
                assert!(info.updated_at.as_deref().unwrap().ends_with('Z'));
                assert_eq!(info.meta.as_ref().unwrap()[alx_acp::MESSAGE_COUNT_META], 2);

                updates.lock().unwrap().clear();
                let resumed = cx
                    .send_request(ResumeSessionRequest::new(a.clone(), "/a"))
                    .block_task()
                    .await?;
                assert_eq!(resumed.meta.unwrap()[alx_acp::MESSAGE_COUNT_META], 2);
                // No history replay, only the command list.
                assert!(
                    updates
                        .lock()
                        .unwrap()
                        .iter()
                        .all(|u| matches!(u, SessionUpdate::AvailableCommandsUpdate(_)))
                );
                cx.send_request(prompt(a.clone(), "again"))
                    .block_task()
                    .await?;
                let in_a = cx
                    .send_request(ListSessionsRequest::new().cwd(std::path::PathBuf::from("/a")))
                    .block_task()
                    .await?;
                assert_eq!(
                    in_a.sessions[0].meta.as_ref().unwrap()[alx_acp::MESSAGE_COUNT_META],
                    4
                );

                let missing = cx
                    .send_request(ResumeSessionRequest::new("nope", "/a"))
                    .block_task()
                    .await
                    .unwrap_err();
                assert_eq!(missing.code, ErrorCode::InvalidParams);
                Ok(())
            },
        )
        .await
        .unwrap();
}
