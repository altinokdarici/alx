//! A real ACP client talking to `serve` over an in-process channel.

use agent_client_protocol::schema::ProtocolVersion;
use agent_client_protocol::schema::v1::{
    ContentBlock, InitializeRequest, NewSessionRequest, PromptRequest, SessionNotification,
    SessionUpdate, StopReason, TextContent, ToolCallStatus,
};
use agent_client_protocol::{Agent, Channel, Client, ConnectionTo};
use alx_core::protocol::{ContentBlock as Block, ProviderInfo, StopReason as Stop};
use alx_core::{
    AgentConfig, BashTool, CompletionRequest, CompletionResponse, ExecOutput, ExecRequest,
    Executor, InMemorySessionStore, Provider, SingleAgent, ToolContext, ToolRegistry,
};
use async_trait::async_trait;
use std::collections::VecDeque;
use std::sync::{Arc, Mutex};

struct Scripted(Mutex<VecDeque<Vec<Block>>>);

#[async_trait]
impl Provider for Scripted {
    fn info(&self) -> ProviderInfo {
        ProviderInfo {
            name: "scripted".into(),
            model: None,
        }
    }
    async fn complete(&self, _req: CompletionRequest) -> anyhow::Result<CompletionResponse> {
        let content = self.0.lock().unwrap().pop_front().expect("no more replies");
        let stop_reason = if content.iter().any(|b| matches!(b, Block::ToolUse { .. })) {
            Stop::ToolUse
        } else {
            Stop::EndTurn
        };
        Ok(CompletionResponse {
            content,
            stop_reason,
            model: None,
            usage: None,
            context: None,
            compactions: vec![],
        })
    }
}

struct FakeExec;

#[async_trait]
impl Executor for FakeExec {
    async fn run(&self, _req: ExecRequest) -> std::io::Result<ExecOutput> {
        Ok(ExecOutput {
            stdout: "hello\n".into(),
            stderr: String::new(),
            exit_code: Some(0),
            timed_out: false,
        })
    }
}

#[tokio::test]
async fn prompt_streams_updates_and_ends_the_turn() {
    let provider = Scripted(Mutex::new(
        [
            vec![Block::ToolUse {
                id: "t1".into(),
                name: "bash".into(),
                input: serde_json::json!({"input": "echo hello"}),
            }],
            vec![Block::Text {
                text: "done".into(),
            }],
        ]
        .into(),
    ));
    let agent = Arc::new(SingleAgent {
        config: AgentConfig::default(),
        provider: Arc::new(provider),
        tools: ToolRegistry::new().with(BashTool),
        tool_ctx: ToolContext {
            executor: Arc::new(FakeExec),
        },
    });
    let (client_end, agent_end) = Channel::duplex();
    let server = tokio::spawn(alx_acp::serve(
        agent,
        Arc::new(InMemorySessionStore::default()),
        agent_end,
    ));

    let updates = Arc::new(Mutex::new(Vec::new()));
    let seen = updates.clone();
    let stop = Client
        .builder()
        .on_receive_notification(
            async move |n: SessionNotification, _cx| {
                seen.lock().unwrap().push(n.update);
                Ok(())
            },
            agent_client_protocol::on_receive_notification!(),
        )
        .connect_with(client_end, async |cx: ConnectionTo<Agent>| {
            cx.send_request(InitializeRequest::new(ProtocolVersion::V1))
                .block_task()
                .await?;
            let session = cx
                .send_request(NewSessionRequest::new("/tmp"))
                .block_task()
                .await?
                .session_id;
            let reply = cx
                .send_request(PromptRequest::new(
                    session,
                    vec![ContentBlock::Text(TextContent::new("hi"))],
                ))
                .block_task()
                .await?;
            Ok(reply.stop_reason)
        })
        .await
        .unwrap();
    assert_eq!(stop, StopReason::EndTurn);

    let updates = updates.lock().unwrap();
    let kinds: Vec<&str> = updates
        .iter()
        .map(|u| match u {
            SessionUpdate::ToolCall(_) => "tool_call",
            SessionUpdate::ToolCallUpdate(u) => {
                assert_eq!(u.fields.status, Some(ToolCallStatus::Completed));
                "tool_update"
            }
            SessionUpdate::AgentMessageChunk(_) => "text",
            _ => "other",
        })
        .collect();
    assert_eq!(kinds, ["tool_call", "tool_update", "text"]);
    server.abort();
}
