//! The agent loop: drives provider <-> tools until the model ends its turn.

use crate::event::EventSink;
use crate::provider::{CompletionRequest, Provider};
use crate::session::Session;
use crate::tool::{ToolContext, ToolOutput, ToolRegistry};
use alx_protocol::{ContentBlock, Event, Message, Role, StopReason, Usage};
use async_trait::async_trait;
use std::sync::Arc;
use tokio_util::sync::CancellationToken;

#[async_trait]
pub trait AgentLoop: Send + Sync {
    /// Append `input` to the session and run until the model stops asking
    /// for tools. Returns the final stop reason.
    ///
    /// On error the session is rolled back to how it was before the call,
    /// so the caller can retry or carry on.
    ///
    /// Cancelling `cancel` stops the turn as soon as possible: a running
    /// model request or tool call is dropped (killing its process), and the
    /// result is `Ok(StopReason::Cancelled)`. Work done so far stays in the
    /// session, which is left in a valid state to continue from.
    async fn run_turn(
        &self,
        session: &mut Session,
        input: Message,
        events: &dyn EventSink,
        cancel: &CancellationToken,
    ) -> anyhow::Result<StopReason>;

    /// Compact the provider's view of the conversation now. Emits
    /// [`Event::Compacted`] when something was compacted and returns whether
    /// it was. Errors if the provider can't compact.
    async fn compact(&self, events: &dyn EventSink) -> anyhow::Result<bool>;
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
        cancel: &CancellationToken,
    ) -> anyhow::Result<StopReason> {
        loop {
            events.emit(Event::ModelRequest);
            let on_text = |text: &str| events.emit(Event::AssistantTextDelta { text: text.into() });
            let request = self.provider.complete_streaming(
                CompletionRequest {
                    system: self.config.system_prompt.clone(),
                    messages: session.messages.clone(),
                    tools: self.tools.specs(),
                    max_tokens: self.config.max_tokens,
                },
                &on_text,
            );
            let resp = tokio::select! {
                biased;
                () = cancel.cancelled() => return Ok(mark_cancelled(session)),
                resp = request => resp?,
            };
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

            for c in &resp.compactions {
                events.emit(Event::Compacted(*c));
            }
            if let Some(c) = resp.context {
                events.emit(Event::ContextUsage(c));
            }

            if resp.stop_reason != StopReason::ToolUse || tool_calls.is_empty() {
                return Ok(resp.stop_reason);
            }

            // All tool results go in a single user message after the assistant turn.
            let mut results = Vec::with_capacity(tool_calls.len());
            let mut cancelled = false;
            for (id, name, input) in tool_calls {
                if cancelled {
                    // Every tool_use still needs a tool_result.
                    results.push(ContentBlock::ToolResult {
                        tool_use_id: id,
                        content: "Not executed: cancelled by user.".into(),
                        is_error: true,
                    });
                    continue;
                }
                events.emit(Event::ToolCallStarted {
                    id: id.clone(),
                    name: name.clone(),
                    input: input.clone(),
                });
                let call = self.tools.call(&name, input, &self.tool_ctx);
                let out = tokio::select! {
                    biased;
                    () = cancel.cancelled() => {
                        cancelled = true;
                        ToolOutput::error("Cancelled by user.")
                    }
                    out = call => out,
                };
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
            if cancelled {
                return Ok(mark_cancelled(session));
            }
        }
    }
}

/// Ends a cancelled turn with an assistant message, so the history keeps
/// alternating user/assistant and the next user message follows cleanly.
fn mark_cancelled(session: &mut Session) -> StopReason {
    session.messages.push(Message {
        role: Role::Assistant,
        content: vec![ContentBlock::Text {
            text: "[Cancelled by user]".into(),
        }],
    });
    StopReason::Cancelled
}

#[async_trait]
impl AgentLoop for SingleAgent {
    async fn compact(&self, events: &dyn EventSink) -> anyhow::Result<bool> {
        let compaction = self.provider.compact().await?;
        if let Some(c) = compaction {
            events.emit(Event::Compacted(c));
        }
        Ok(compaction.is_some())
    }

    async fn run_turn(
        &self,
        session: &mut Session,
        input: Message,
        events: &dyn EventSink,
        cancel: &CancellationToken,
    ) -> anyhow::Result<StopReason> {
        let checkpoint = session.messages.len();
        session.messages.push(input);
        events.emit(Event::TurnStarted);
        let mut model = None;
        let mut usage = None;
        match self
            .turn_inner(session, events, &mut model, &mut usage, cancel)
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
                timed_out: false,
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

    fn agent(provider: Arc<dyn Provider>, exec: Arc<dyn Executor>) -> SingleAgent {
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
    async fn context_and_compaction_reports_become_events() {
        use alx_protocol::{Compaction, ContextUsage};
        let compaction = Compaction {
            messages_removed: 4,
            tokens_before: Some(160_000),
            tokens_after: Some(9_000),
        };
        let context = ContextUsage {
            used_tokens: 9_500,
            limit_tokens: 200_000,
        };
        let provider = Arc::new(ScriptedProvider::new(vec![Ok(CompletionResponse {
            content: vec![text("ok")],
            stop_reason: StopReason::EndTurn,
            model: None,
            usage: None,
            context: Some(context),
            compactions: vec![compaction],
        })]));
        let agent = agent(provider, Arc::new(FakeExecutor::default()));
        let sink = VecSink::default();
        let mut session = Session::new("s");
        agent
            .run_turn(
                &mut session,
                Message::user_text("hi"),
                &sink,
                &CancellationToken::new(),
            )
            .await
            .unwrap();
        {
            let events = sink.0.lock().unwrap();
            assert!(events.contains(&Event::Compacted(compaction)));
            assert!(events.contains(&Event::ContextUsage(context)));
        }

        // The scripted provider can't compact: the error reaches the caller.
        let err = agent.compact(&sink).await.unwrap_err();
        assert!(err.to_string().contains("cannot compact"), "{err}");
    }

    #[tokio::test]
    async fn runs_tools_until_end_turn() {
        let provider = Arc::new(ScriptedProvider::new(vec![
            Ok(CompletionResponse {
                context: None,
                compactions: Vec::new(),
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
                context: None,
                compactions: Vec::new(),
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
            .run_turn(
                &mut session,
                Message::user_text("hi"),
                &sink,
                &CancellationToken::new(),
            )
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
        let agent = agent(provider, Arc::new(FakeExecutor::default()));
        let mut session = Session::new("s");
        session.messages.push(Message::user_text("earlier"));

        let res = agent
            .run_turn(
                &mut session,
                Message::user_text("hi"),
                &crate::event::NullSink,
                &CancellationToken::new(),
            )
            .await;

        assert!(res.is_err());
        assert_eq!(session.messages, vec![Message::user_text("earlier")]);
    }

    /// Never finishes, so a turn can only end by being cancelled.
    struct HangingExecutor;

    #[async_trait]
    impl Executor for HangingExecutor {
        async fn run(&self, _req: ExecRequest) -> std::io::Result<ExecOutput> {
            std::future::pending().await
        }
    }

    struct HangingProvider;

    #[async_trait]
    impl Provider for HangingProvider {
        fn info(&self) -> alx_protocol::ProviderInfo {
            alx_protocol::ProviderInfo {
                name: "hanging".into(),
                model: None,
            }
        }

        async fn complete(&self, _req: CompletionRequest) -> anyhow::Result<CompletionResponse> {
            std::future::pending().await
        }
    }

    fn cancel_soon(token: &CancellationToken) {
        let token = token.clone();
        tokio::spawn(async move {
            tokio::time::sleep(std::time::Duration::from_millis(50)).await;
            token.cancel();
        });
    }

    #[tokio::test]
    async fn cancelling_a_tool_call_keeps_history_valid() {
        let call = |id: &str| ContentBlock::ToolUse {
            id: id.into(),
            name: "bash".into(),
            input: json!({"input": "sleep 100"}),
        };
        let provider = Arc::new(ScriptedProvider::new(vec![Ok(CompletionResponse {
            context: None,
            compactions: Vec::new(),
            content: vec![call("t1"), call("t2")],
            stop_reason: StopReason::ToolUse,
            model: None,
            usage: None,
        })]));
        let mut agent = agent(provider, Arc::new(FakeExecutor::default()));
        agent.tool_ctx.executor = Arc::new(HangingExecutor);
        let sink = VecSink::default();
        let mut session = Session::new("s");
        let cancel = CancellationToken::new();
        cancel_soon(&cancel);

        let stop = agent
            .run_turn(&mut session, Message::user_text("hi"), &sink, &cancel)
            .await
            .unwrap();

        assert_eq!(stop, StopReason::Cancelled);
        // user, assistant(tool_use x2), user(tool_results x2), assistant(cancelled)
        assert_eq!(session.messages.len(), 4);
        assert_eq!(session.messages[3].role, Role::Assistant);
        let results = &session.messages[2].content;
        assert_eq!(results.len(), 2);
        for (r, expected) in results
            .iter()
            .zip(["Cancelled by user.", "Not executed: cancelled by user."])
        {
            let ContentBlock::ToolResult {
                content, is_error, ..
            } = r
            else {
                panic!("expected tool result");
            };
            assert!(*is_error);
            assert_eq!(content, expected);
        }
        // Only the call that actually started is reported as started.
        let events = sink.0.lock().unwrap();
        let started = events
            .iter()
            .filter(|e| matches!(e, Event::ToolCallStarted { .. }))
            .count();
        let finished = events
            .iter()
            .filter(|e| matches!(e, Event::ToolCallFinished { .. }))
            .count();
        assert_eq!((started, finished), (1, 1));
        assert!(matches!(
            events.last(),
            Some(Event::TurnFinished {
                stop_reason: StopReason::Cancelled,
                ..
            })
        ));
    }

    /// Reports its text in two pieces before returning it whole.
    struct StreamingProvider;

    #[async_trait]
    impl Provider for StreamingProvider {
        fn info(&self) -> alx_protocol::ProviderInfo {
            alx_protocol::ProviderInfo {
                name: "streaming".into(),
                model: None,
            }
        }

        async fn complete(&self, req: CompletionRequest) -> anyhow::Result<CompletionResponse> {
            self.complete_streaming(req, &|_| {}).await
        }

        async fn complete_streaming(
            &self,
            _req: CompletionRequest,
            on_text: &(dyn for<'a> Fn(&'a str) + Send + Sync),
        ) -> anyhow::Result<CompletionResponse> {
            on_text("hel");
            on_text("lo");
            Ok(CompletionResponse {
                context: None,
                compactions: Vec::new(),
                content: vec![text("hello")],
                stop_reason: StopReason::EndTurn,
                model: None,
                usage: None,
            })
        }
    }

    #[tokio::test]
    async fn streamed_text_becomes_delta_events_before_the_whole_text() {
        let agent = agent(
            Arc::new(StreamingProvider),
            Arc::new(FakeExecutor::default()),
        );
        let sink = VecSink::default();
        let mut session = Session::new("s");
        agent
            .run_turn(
                &mut session,
                Message::user_text("hi"),
                &sink,
                &CancellationToken::new(),
            )
            .await
            .unwrap();
        let events = sink.0.lock().unwrap();
        let kinds: Vec<_> = events
            .iter()
            .filter_map(|e| match e {
                Event::AssistantTextDelta { text } => Some(format!("delta:{text}")),
                Event::AssistantText { text } => Some(format!("text:{text}")),
                _ => None,
            })
            .collect();
        assert_eq!(kinds, ["delta:hel", "delta:lo", "text:hello"]);
    }

    #[tokio::test]
    async fn cancelling_a_model_request_ends_the_turn() {
        let agent = agent(Arc::new(HangingProvider), Arc::new(FakeExecutor::default()));
        let mut session = Session::new("s");
        let cancel = CancellationToken::new();
        cancel_soon(&cancel);

        let stop = agent
            .run_turn(
                &mut session,
                Message::user_text("hi"),
                &crate::event::NullSink,
                &cancel,
            )
            .await
            .unwrap();

        assert_eq!(stop, StopReason::Cancelled);
        assert_eq!(session.messages.len(), 2);
        assert_eq!(session.messages[1].role, Role::Assistant);
    }
}
