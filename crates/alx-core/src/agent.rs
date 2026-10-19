//! The agent loop: drives provider <-> tools until the model ends its turn.

use crate::event::EventSink;
use crate::provider::{CompletionRequest, Provider};
use crate::session::Session;
use crate::tool::{ToolContext, ToolRegistry};
use alx_protocol::{ContentBlock, Event, Message, Role, StopReason, Usage};
use async_trait::async_trait;
use std::sync::Arc;

#[async_trait]
pub trait AgentLoop: Send + Sync {
    /// Append `input` to the session and run until the model stops asking
    /// for tools. Returns the final stop reason.
    ///
    /// On error the session is rolled back to how it was before the call,
    /// so the caller can retry or carry on.
    async fn run_turn(
        &self,
        session: &mut Session,
        input: Message,
        events: &dyn EventSink,
    ) -> anyhow::Result<StopReason>;
}

#[derive(Debug, Clone)]
pub struct AgentConfig {
    pub system_prompt: String,
    pub max_tokens: u32,
}

impl Default for AgentConfig {
    fn default() -> Self {
        Self {
            system_prompt: "You are a coding agent, you use bash tool to write code, read files, \
                            write files, execute code"
                .into(),
            max_tokens: 10240,
        }
    }
}

/// A single agent: one provider, one tool set, one executor.
pub struct SingleAgent {
    pub config: AgentConfig,
    pub provider: Arc<dyn Provider>,
    pub tools: ToolRegistry,
    pub tool_ctx: ToolContext,
}

impl SingleAgent {
    async fn turn_inner(
        &self,
        session: &mut Session,
        events: &dyn EventSink,
        model: &mut Option<String>,
        usage: &mut Option<Usage>,
    ) -> anyhow::Result<StopReason> {
        loop {
            events.emit(Event::ModelRequest);
            let resp = self
                .provider
                .complete(CompletionRequest {
                    system: self.config.system_prompt.clone(),
                    messages: session.messages.clone(),
                    tools: self.tools.specs(),
                    max_tokens: self.config.max_tokens,
                })
                .await?;
            if resp.model.is_some() {
                model.clone_from(&resp.model);
            }
            if let Some(u) = &resp.usage {
                usage.get_or_insert_with(Usage::default).add(u);
            }
            session.messages.push(Message {
                role: Role::Assistant,
                content: resp.content.clone(),
            });

            let mut tool_calls = Vec::new();
            for block in resp.content {
                match block {
                    ContentBlock::Text { text } => events.emit(Event::AssistantText { text }),
                    ContentBlock::ToolUse { id, name, input } => tool_calls.push((id, name, input)),
                    ContentBlock::ToolResult { .. } => {}
                }
            }

            if resp.stop_reason != StopReason::ToolUse || tool_calls.is_empty() {
                return Ok(resp.stop_reason);
            }

            // All tool results go in a single user message after the assistant turn.
            let mut results = Vec::with_capacity(tool_calls.len());
            for (id, name, input) in tool_calls {
                events.emit(Event::ToolCallStarted {
                    id: id.clone(),
                    name: name.clone(),
                    input: input.clone(),
                });
                let out = self.tools.call(&name, input, &self.tool_ctx).await;
                events.emit(Event::ToolCallFinished {
                    id: id.clone(),
                    name,
                    output: out.content.clone(),
                    is_error: out.is_error,
                });
                results.push(ContentBlock::ToolResult {
                    tool_use_id: id,
                    content: out.content,
                    is_error: out.is_error,
                });
            }
            session.messages.push(Message {
                role: Role::User,
                content: results,
            });
        }
    }
}

#[async_trait]
impl AgentLoop for SingleAgent {
    async fn run_turn(
        &self,
        session: &mut Session,
        input: Message,
        events: &dyn EventSink,
    ) -> anyhow::Result<StopReason> {
        let checkpoint = session.messages.len();
        session.messages.push(input);
        events.emit(Event::TurnStarted);
        let mut model = None;
        let mut usage = None;
        match self
            .turn_inner(session, events, &mut model, &mut usage)
            .await
        {
            Ok(stop_reason) => {
                events.emit(Event::TurnFinished {
                    stop_reason: stop_reason.clone(),
                    model,
                    usage,
                });
                Ok(stop_reason)
            }
            Err(e) => {
                session.messages.truncate(checkpoint);
                Err(e)
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::executor::{ExecOutput, ExecRequest, Executor};
    use crate::provider::CompletionResponse;
    use crate::tools::bash::BashTool;
    use serde_json::json;
    use std::collections::VecDeque;
    use std::sync::Mutex;

    /// Replays canned responses and records the requests it saw.
    struct ScriptedProvider {
        responses: Mutex<VecDeque<anyhow::Result<CompletionResponse>>>,
        seen: Mutex<Vec<CompletionRequest>>,
    }

    impl ScriptedProvider {
        fn new(responses: Vec<anyhow::Result<CompletionResponse>>) -> Self {
            Self {
                responses: Mutex::new(responses.into()),
                seen: Mutex::default(),
            }
        }
    }

    #[async_trait]
    impl Provider for ScriptedProvider {
        fn info(&self) -> alx_protocol::ProviderInfo {
            alx_protocol::ProviderInfo {
                name: "scripted".into(),
                model: None,
            }
        }

        async fn complete(&self, req: CompletionRequest) -> anyhow::Result<CompletionResponse> {
            self.seen.lock().unwrap().push(req);
            self.responses
                .lock()
                .unwrap()
                .pop_front()
                .expect("unexpected provider call")
        }
    }

    /// Records commands instead of running them.
    #[derive(Default)]
    struct FakeExecutor {
        ran: Mutex<Vec<String>>,
    }

    #[async_trait]
    impl Executor for FakeExecutor {
        async fn run(&self, req: ExecRequest) -> std::io::Result<ExecOutput> {
            self.ran.lock().unwrap().push(req.stdin.unwrap_or_default());
            Ok(ExecOutput {
                stdout: "fake\n".into(),
                stderr: String::new(),
                exit_code: Some(0),
            })
        }
    }

    #[derive(Default)]
    struct VecSink(Mutex<Vec<Event>>);
    impl EventSink for VecSink {
        fn emit(&self, e: Event) {
            self.0.lock().unwrap().push(e);
        }
    }

    fn agent(provider: Arc<ScriptedProvider>, exec: Arc<FakeExecutor>) -> SingleAgent {
        SingleAgent {
            config: AgentConfig::default(),
            provider,
            tools: ToolRegistry::new().with(BashTool),
            tool_ctx: ToolContext { executor: exec },
        }
    }

    fn text(t: &str) -> ContentBlock {
        ContentBlock::Text { text: t.into() }
    }

    #[tokio::test]
    async fn runs_tools_until_end_turn() {
        let provider = Arc::new(ScriptedProvider::new(vec![
            Ok(CompletionResponse {
                content: vec![
                    text("let me look"),
                    ContentBlock::ToolUse {
                        id: "t1".into(),
                        name: "bash".into(),
                        input: json!({"input": "ls"}),
                    },
                    ContentBlock::ToolUse {
                        id: "t2".into(),
                        name: "nope".into(),
                        input: json!({}),
                    },
                ],
                stop_reason: StopReason::ToolUse,
                model: None,
                usage: Some(alx_protocol::Usage {
                    input_tokens: 10,
                    output_tokens: 5,
                    cache_read_tokens: 0,
                    cache_write_tokens: 100,
                }),
            }),
            Ok(CompletionResponse {
                content: vec![text("done")],
                stop_reason: StopReason::EndTurn,
                model: Some("m-2".into()),
                usage: Some(alx_protocol::Usage {
                    input_tokens: 2,
                    output_tokens: 7,
                    cache_read_tokens: 100,
                    cache_write_tokens: 20,
                }),
            }),
        ]));
        let exec = Arc::new(FakeExecutor::default());
        let agent = agent(provider.clone(), exec.clone());
        let sink = VecSink::default();
        let mut session = Session::new("s");

        let stop = agent
            .run_turn(&mut session, Message::user_text("hi"), &sink)
            .await
            .unwrap();

        assert_eq!(stop, StopReason::EndTurn);
        assert_eq!(*exec.ran.lock().unwrap(), vec!["ls".to_string()]);
        // user, assistant(tool_use), user(tool_results), assistant(done)
        assert_eq!(session.messages.len(), 4);
        let ContentBlock::ToolResult {
            is_error, content, ..
        } = &session.messages[2].content[1]
        else {
            panic!("expected tool result");
        };
        assert!(*is_error);
        assert_eq!(content, "Invalid tool call: nope");
        // The second request carried the tool results.
        assert_eq!(provider.seen.lock().unwrap()[1].messages.len(), 3);

        let events = sink.0.lock().unwrap();
        assert_eq!(events.first(), Some(&Event::TurnStarted));
        assert_eq!(
            events.last(),
            Some(&Event::TurnFinished {
                stop_reason: StopReason::EndTurn,
                model: Some("m-2".into()),
                usage: Some(alx_protocol::Usage {
                    input_tokens: 12,
                    output_tokens: 12,
                    cache_read_tokens: 100,
                    cache_write_tokens: 120
                }),
            })
        );
        assert!(events.contains(&Event::AssistantText {
            text: "done".into()
        }));
    }

    #[tokio::test]
    async fn rolls_back_session_on_provider_error() {
        let provider = Arc::new(ScriptedProvider::new(vec![Err(anyhow::anyhow!("boom"))]));
        let agent = agent(provider, Arc::default());
        let mut session = Session::new("s");
        session.messages.push(Message::user_text("earlier"));

        let res = agent
            .run_turn(
                &mut session,
                Message::user_text("hi"),
                &crate::event::NullSink,
            )
            .await;

        assert!(res.is_err());
        assert_eq!(session.messages, vec![Message::user_text("earlier")]);
    }
}
