//! The agent loop: drives provider <-> tools until the model ends its turn.

use crate::compaction;
use crate::event::EventSink;
use crate::executor::Executor;
use crate::provider::{CompletionRequest, CompletionResponse, Provider, TransientError};
use crate::session::{CompactionState, Session};
use crate::tool::{Approval, ToolApprover, ToolContext, ToolOutput, ToolRegistry};
use alx_protocol::{
    Compaction, ContentBlock, ContextUsage, Event, Message, ModelInfo, ProviderInfo, Role,
    StopReason, Usage,
};
use anyhow::Context;
use async_trait::async_trait;
use std::sync::Arc;
use std::time::Duration;
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
    ) -> anyhow::Result<StopReason> {
        self.run_turn_with(session, input, events, cancel, &TurnOptions::default())
            .await
    }

    /// [`run_turn`](Self::run_turn) with per-turn overrides, e.g. from a
    /// client that runs commands itself or approves tool calls.
    async fn run_turn_with(
        &self,
        session: &mut Session,
        input: Message,
        events: &dyn EventSink,
        cancel: &CancellationToken,
        options: &TurnOptions,
    ) -> anyhow::Result<StopReason>;

    /// Compact the conversation now: the provider's own copy if it keeps one,
    /// else the session's (see [`crate::compaction`]). `instructions` steer
    /// the summary. Emits [`Event::Compacted`] when something was compacted
    /// and returns whether it was.
    async fn compact(
        &self,
        session: &mut Session,
        instructions: Option<&str>,
        events: &dyn EventSink,
    ) -> anyhow::Result<bool>;

    /// The provider (and model) this agent talks to, for display.
    fn provider_info(&self) -> Option<ProviderInfo> {
        None
    }

    /// The models the provider offers. Empty (the default) if it can't say.
    async fn list_models(&self) -> anyhow::Result<Vec<ModelInfo>> {
        Ok(Vec::new())
    }
}

/// Builds agents on demand, one per provider name, so a server can give
/// every session its own agent on the provider the session picked.
pub trait AgentFactory: Send + Sync {
    /// The provider names [`build`](Self::build) accepts, the default first.
    fn providers(&self) -> Vec<String>;

    /// A fresh agent on `provider`, on `model` if given, else the
    /// configured default. Cheap: providers connect lazily.
    fn build(&self, provider: &str, model: Option<&str>) -> anyhow::Result<Arc<dyn AgentLoop>>;
}

/// Per-turn overrides for [`AgentLoop::run_turn_with`].
#[derive(Clone, Default)]
pub struct TurnOptions {
    /// Run this turn's tools here instead of the agent's own executor.
    pub executor: Option<Arc<dyn Executor>>,
    /// Ask this before each call to a tool that is not read-only. `None`
    /// runs every call.
    pub approver: Option<Arc<dyn ToolApprover>>,
}

#[derive(Debug, Clone)]
pub struct AgentConfig {
    pub system_prompt: String,
    pub max_tokens: u32,
    /// Retries of a model request that failed with a
    /// [`TransientError`](crate::provider::TransientError).
    pub max_retries: u32,
}

impl Default for AgentConfig {
    fn default() -> Self {
        Self {
            system_prompt: "You are a coding agent, you use bash tool to write code, read files, \
                            write files, execute code"
                .into(),
            max_tokens: 10240,
            max_retries: 4,
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
        options: &TurnOptions,
    ) -> anyhow::Result<StopReason> {
        let tool_ctx = match &options.executor {
            Some(executor) => ToolContext {
                executor: executor.clone(),
            },
            None => self.tool_ctx.clone(),
        };
        let window = self
            .provider
            .context_window()
            .filter(|_| !self.provider.compacts_itself());
        // Tokens the next request will carry: estimated at first, then taken
        // from each response's usage.
        let mut context_tokens: u64 = session
            .request_messages()
            .iter()
            .map(compaction::estimate_tokens)
            .sum();
        // The context size at the last automatic attempt: don't try again
        // until it has grown, in case the attempt couldn't free much.
        let mut attempted_at: Option<u64> = None;
        loop {
            if let Some(window) = window
                && context_tokens > window.saturating_sub(compaction::RESERVE_TOKENS)
                && attempted_at.is_none_or(|t| context_tokens > t + compaction::KEEP_RECENT_TOKENS)
            {
                attempted_at = Some(context_tokens);
                let compacted = tokio::select! {
                    biased;
                    () = cancel.cancelled() => return Ok(mark_cancelled(session)),
                    r = self.compact_session(session, None, events) => r?,
                };
                if compacted {
                    attempted_at = None;
                    context_tokens = session
                        .request_messages()
                        .iter()
                        .map(compaction::estimate_tokens)
                        .sum();
                }
            }
            let Some(resp) = self.request_with_retry(session, events, cancel).await? else {
                return Ok(mark_cancelled(session));
            };
            if let Some(u) = &resp.usage {
                context_tokens =
                    u.input_tokens + u.cache_read_tokens + u.cache_write_tokens + u.output_tokens;
            }
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
            } else if let (Some(limit), Some(_)) = (window, &resp.usage) {
                events.emit(Event::ContextUsage(ContextUsage {
                    used_tokens: context_tokens,
                    limit_tokens: limit,
                }));
            }

            if resp.stop_reason == StopReason::MaxTokens && !tool_calls.is_empty() {
                // The calls were cut off, so their arguments may be incomplete.
                // Don't run them; tell the model and let it try again.
                let results = tool_calls
                    .into_iter()
                    .map(|(id, name, input)| {
                        events.emit(Event::ToolCallStarted {
                            id: id.clone(),
                            name: name.clone(),
                            input,
                        });
                        events.emit(Event::ToolCallFinished {
                            id: id.clone(),
                            name,
                            output: TRUNCATED_CALL.into(),
                            is_error: true,
                        });
                        ContentBlock::ToolResult {
                            tool_use_id: id,
                            content: TRUNCATED_CALL.into(),
                            is_error: true,
                        }
                    })
                    .collect();
                session.messages.push(Message {
                    role: Role::User,
                    content: results,
                });
                continue;
            }
            if resp.stop_reason != StopReason::ToolUse || tool_calls.is_empty() {
                return Ok(resp.stop_reason);
            }

            // Approvals are asked in order, one at a time. The approved calls
            // then run concurrently (or in order, if any tool asks for that),
            // and the results keep the order of the calls.
            let mut cancelled = false;
            let mut prepared = Vec::with_capacity(tool_calls.len());
            for (id, name, input) in tool_calls {
                if cancelled || cancel.is_cancelled() {
                    cancelled = true;
                    // Every tool_use still needs a tool_result.
                    prepared.push((
                        id,
                        name,
                        input,
                        Some(ToolOutput::error("Not executed: cancelled by user.")),
                    ));
                    continue;
                }
                events.emit(Event::ToolCallStarted {
                    id: id.clone(),
                    name: name.clone(),
                    input: input.clone(),
                });
                let approver = options
                    .approver
                    .as_ref()
                    .filter(|_| !self.tools.is_read_only(&name));
                let denied = match approver {
                    None => None,
                    Some(approver) => tokio::select! {
                        biased;
                        () = cancel.cancelled() => {
                            cancelled = true;
                            Some(ToolOutput::error("Cancelled by user."))
                        }
                        a = approver.approve(&id, &name, &input) => (a == Approval::Deny)
                            .then(|| ToolOutput::error("The user denied permission to run this tool.")),
                    },
                };
                if let Some(out) = &denied {
                    events.emit(Event::ToolCallFinished {
                        id: id.clone(),
                        name: name.clone(),
                        output: out.content.clone(),
                        is_error: true,
                    });
                }
                prepared.push((id, name, input, denied));
            }

            let sequential = prepared
                .iter()
                .any(|(_, name, _, done)| done.is_none() && self.tools.is_sequential(name));
            let run_one =
                |id: String, name: String, input: serde_json::Value, done: Option<ToolOutput>| {
                    let tool_ctx = &tool_ctx;
                    async move {
                        let out = match done {
                            Some(out) => return (id, out),
                            None => tokio::select! {
                                biased;
                                () = cancel.cancelled() => ToolOutput::error("Cancelled by user."),
                                out = self.tools.call(&name, input, tool_ctx) => out,
                            },
                        };
                        events.emit(Event::ToolCallFinished {
                            id: id.clone(),
                            name,
                            output: out.content.clone(),
                            is_error: out.is_error,
                        });
                        (id, out)
                    }
                };
            let outputs = if sequential {
                let mut outputs = Vec::with_capacity(prepared.len());
                for (id, name, input, done) in prepared {
                    let done = done.or_else(|| {
                        cancel
                            .is_cancelled()
                            .then(|| ToolOutput::error("Not executed: cancelled by user."))
                    });
                    outputs.push(run_one(id, name, input, done).await);
                }
                outputs
            } else {
                futures::future::join_all(
                    prepared
                        .into_iter()
                        .map(|(id, name, input, done)| run_one(id, name, input, done)),
                )
                .await
            };
            cancelled |= cancel.is_cancelled();
            let results = outputs
                .into_iter()
                .map(|(id, out)| ContentBlock::ToolResult {
                    tool_use_id: id,
                    content: out.content,
                    is_error: out.is_error,
                })
                .collect();
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

const TRUNCATED_CALL: &str = "Not executed: your response hit the output token limit, so \
     this tool call was cut off and its arguments may be incomplete. Re-issue it with complete \
     arguments; split large content into several smaller calls.";

/// The backoff before retry `attempt` (1-based): 2s, 4s, 8s, ... capped at 60s.
fn backoff(attempt: u32) -> Duration {
    Duration::from_secs(2u64.saturating_pow(attempt).min(60))
}

/// Server-requested waits longer than this aren't worth sitting through.
const MAX_RETRY_DELAY: Duration = Duration::from_secs(60);

impl SingleAgent {
    /// Summarizes the older part of the session (see [`crate::compaction`]).
    /// `Ok(false)` if there is too little to compact.
    async fn compact_session(
        &self,
        session: &mut Session,
        instructions: Option<&str>,
        events: &dyn EventSink,
    ) -> anyhow::Result<bool> {
        let from = session.compaction.as_ref().map_or(0, |c| c.first_kept);
        let Some(cut) =
            compaction::find_cut(&session.messages, from, compaction::KEEP_RECENT_TOKENS)
        else {
            return Ok(false);
        };
        let before = session.request_messages();
        let tokens_before: u64 = before.iter().map(compaction::estimate_tokens).sum();
        // `before` starts at `from`, with the old summary merged or prepended.
        let older = &before[..before.len() - (session.messages.len() - cut)];
        let older_tokens: u64 = older.iter().map(compaction::estimate_tokens).sum();
        if older_tokens < compaction::MIN_COMPACT_TOKENS {
            // A summary would be about as long as what it replaces.
            return Ok(false);
        }
        let resp = self
            .provider
            .complete(CompletionRequest {
                system: compaction::SUMMARY_SYSTEM_PROMPT.into(),
                messages: vec![compaction::summary_request(older, instructions)],
                tools: Vec::new(),
                max_tokens: self.config.max_tokens,
            })
            .await
            .context("summarizing the conversation")?;
        let summary: String = resp
            .content
            .iter()
            .filter_map(|b| match b {
                ContentBlock::Text { text } => Some(text.as_str()),
                _ => None,
            })
            .collect::<Vec<_>>()
            .join("\n");
        if summary.trim().is_empty() {
            anyhow::bail!("the model returned an empty summary");
        }
        session.compaction = Some(CompactionState {
            first_kept: cut,
            summary,
        });
        let tokens_after = session
            .request_messages()
            .iter()
            .map(compaction::estimate_tokens)
            .sum();
        events.emit(Event::Compacted(Compaction {
            messages_removed: (cut - from) as u64,
            tokens_before: Some(tokens_before),
            tokens_after: Some(tokens_after),
        }));
        Ok(true)
    }

    /// One model request, retried on transient errors. `None` if cancelled.
    async fn request_with_retry(
        &self,
        session: &Session,
        events: &dyn EventSink,
        cancel: &CancellationToken,
    ) -> anyhow::Result<Option<CompletionResponse>> {
        let on_text = |text: &str| events.emit(Event::AssistantTextDelta { text: text.into() });
        let mut attempt = 0;
        loop {
            events.emit(Event::ModelRequest);
            let request = self.provider.complete_streaming(
                CompletionRequest {
                    system: self.config.system_prompt.clone(),
                    messages: session.request_messages(),
                    tools: self.tools.specs(),
                    max_tokens: self.config.max_tokens,
                },
                &on_text,
            );
            let err = tokio::select! {
                biased;
                () = cancel.cancelled() => return Ok(None),
                resp = request => match resp {
                    Ok(resp) => return Ok(Some(resp)),
                    Err(e) => e,
                },
            };
            let Some(transient) = err.downcast_ref::<TransientError>() else {
                return Err(err);
            };
            attempt += 1;
            let delay = transient.retry_after.unwrap_or_else(|| backoff(attempt));
            if attempt > self.config.max_retries || delay > MAX_RETRY_DELAY {
                return Err(err);
            }
            events.emit(Event::Retrying {
                attempt,
                max_attempts: self.config.max_retries,
                delay_ms: delay.as_millis() as u64,
                error: transient.message.clone(),
            });
            tokio::select! {
                biased;
                () = cancel.cancelled() => return Ok(None),
                () = tokio::time::sleep(delay) => {}
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
    fn provider_info(&self) -> Option<ProviderInfo> {
        Some(self.provider.info())
    }

    async fn list_models(&self) -> anyhow::Result<Vec<ModelInfo>> {
        self.provider.list_models().await
    }

    async fn compact(
        &self,
        session: &mut Session,
        instructions: Option<&str>,
        events: &dyn EventSink,
    ) -> anyhow::Result<bool> {
        if !self.provider.compacts_itself() {
            return self.compact_session(session, instructions, events).await;
        }
        let compaction = self.provider.compact().await?;
        if let Some(c) = compaction {
            events.emit(Event::Compacted(c));
        }
        Ok(compaction.is_some())
    }

    async fn run_turn_with(
        &self,
        session: &mut Session,
        input: Message,
        events: &dyn EventSink,
        cancel: &CancellationToken,
        options: &TurnOptions,
    ) -> anyhow::Result<StopReason> {
        let checkpoint = session.messages.len();
        let compaction_before = session.compaction.clone();
        session.messages.push(input);
        events.emit(Event::TurnStarted);
        let mut model = None;
        let mut usage = None;
        match self
            .turn_inner(session, events, &mut model, &mut usage, cancel, options)
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
                // A compaction made in this turn may point past the rollback.
                if session
                    .compaction
                    .as_ref()
                    .is_some_and(|c| c.first_kept > checkpoint)
                {
                    session.compaction = compaction_before;
                }
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
        window: Option<u64>,
    }

    impl ScriptedProvider {
        fn new(responses: Vec<anyhow::Result<CompletionResponse>>) -> Self {
            Self {
                responses: Mutex::new(responses.into()),
                seen: Mutex::default(),
                window: None,
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

        fn context_window(&self) -> Option<u64> {
            self.window
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

        // Too short to compact by alx's own compaction.
        assert!(!agent.compact(&mut session, None, &sink).await.unwrap());
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

    fn transient() -> anyhow::Error {
        anyhow::Error::new(TransientError {
            message: "overloaded".into(),
            retry_after: Some(Duration::from_millis(1)),
        })
    }

    fn end_turn(t: &str) -> anyhow::Result<CompletionResponse> {
        Ok(CompletionResponse {
            content: vec![text(t)],
            stop_reason: StopReason::EndTurn,
            model: None,
            usage: None,
            context: None,
            compactions: vec![],
        })
    }

    #[tokio::test]
    async fn a_full_context_is_compacted_before_the_request() {
        let mut provider = ScriptedProvider::new(vec![end_turn("SUM"), end_turn("ok")]);
        provider.window = Some(30_000);
        let provider = Arc::new(provider);
        let agent = agent(provider.clone(), Arc::new(FakeExecutor::default()));
        let sink = VecSink::default();
        let mut session = Session::new("s");
        for i in 0..6 {
            session.messages.push(Message {
                role: if i % 2 == 0 {
                    Role::User
                } else {
                    Role::Assistant
                },
                content: vec![text(&format!("{i}{}", "x".repeat(40_000)))],
            });
        }
        agent
            .run_turn(
                &mut session,
                Message::user_text("hi"),
                &sink,
                &CancellationToken::new(),
            )
            .await
            .unwrap();

        let seen = provider.seen.lock().unwrap();
        assert_eq!(seen[0].system, compaction::SUMMARY_SYSTEM_PROMPT);
        let ContentBlock::Text { text: asked } = &seen[0].messages[0].content[0] else {
            panic!()
        };
        assert!(asked.contains("[User]: 2x") && !asked.contains("[User]: 4x"));
        // The real request: summary, then the kept messages (4, 5, "hi", ...).
        let first = &seen[1].messages[0];
        assert!(matches!(&first.content[0], ContentBlock::Text { text } if text.contains("SUM")));
        assert!(matches!(&first.content[1], ContentBlock::Text { text } if text.starts_with('4')));
        assert_eq!(seen[1].messages.len(), 3);
        assert_eq!(session.messages.len(), 8);
        assert_eq!(session.compaction.as_ref().unwrap().first_kept, 4);
        assert!(sink.0.lock().unwrap().iter().any(|e| matches!(
            e,
            Event::Compacted(Compaction {
                messages_removed: 4,
                ..
            })
        )));
    }

    #[tokio::test]
    async fn transient_errors_are_retried() {
        let provider = Arc::new(ScriptedProvider::new(vec![
            Err(transient()),
            Err(transient()),
            end_turn("ok"),
        ]));
        let agent = agent(provider, Arc::new(FakeExecutor::default()));
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
        let retries: Vec<u32> = sink
            .0
            .lock()
            .unwrap()
            .iter()
            .filter_map(|e| match e {
                Event::Retrying { attempt, .. } => Some(*attempt),
                _ => None,
            })
            .collect();
        assert_eq!(retries, vec![1, 2]);
    }

    #[tokio::test]
    async fn retries_give_up_after_max_retries() {
        let provider = Arc::new(ScriptedProvider::new(vec![
            Err(transient()),
            Err(transient()),
        ]));
        let mut agent = agent(provider, Arc::new(FakeExecutor::default()));
        agent.config.max_retries = 1;
        let res = agent
            .run_turn(
                &mut Session::new("s"),
                Message::user_text("hi"),
                &crate::event::NullSink,
                &CancellationToken::new(),
            )
            .await;
        assert!(res.unwrap_err().downcast_ref::<TransientError>().is_some());
    }

    #[test]
    fn backoff_doubles_up_to_a_minute() {
        assert_eq!(backoff(1), Duration::from_secs(2));
        assert_eq!(backoff(3), Duration::from_secs(8));
        assert_eq!(backoff(10), Duration::from_secs(60));
    }

    #[tokio::test]
    async fn tool_calls_cut_off_by_max_tokens_are_not_run() {
        let provider = Arc::new(ScriptedProvider::new(vec![
            Ok(CompletionResponse {
                content: vec![ContentBlock::ToolUse {
                    id: "t1".into(),
                    name: "bash".into(),
                    input: json!({"input": "echo hi"}),
                }],
                stop_reason: StopReason::MaxTokens,
                model: None,
                usage: None,
                context: None,
                compactions: vec![],
            }),
            end_turn("done"),
        ]));
        let exec = Arc::new(FakeExecutor::default());
        let agent = agent(provider.clone(), exec.clone());
        let mut session = Session::new("s");
        let stop = agent
            .run_turn(
                &mut session,
                Message::user_text("hi"),
                &crate::event::NullSink,
                &CancellationToken::new(),
            )
            .await
            .unwrap();
        assert_eq!(stop, StopReason::EndTurn);
        assert!(exec.ran.lock().unwrap().is_empty());
        assert_eq!(
            session.messages[2].content,
            vec![ContentBlock::ToolResult {
                tool_use_id: "t1".into(),
                content: TRUNCATED_CALL.into(),
                is_error: true,
            }]
        );
        assert_eq!(provider.seen.lock().unwrap().len(), 2);
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
            .zip(["Cancelled by user.", "Cancelled by user."])
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
        // Both calls ran concurrently, so both started and both were cancelled.
        let events = sink.0.lock().unwrap();
        let started = events
            .iter()
            .filter(|e| matches!(e, Event::ToolCallStarted { .. }))
            .count();
        let finished = events
            .iter()
            .filter(|e| matches!(e, Event::ToolCallFinished { .. }))
            .count();
        assert_eq!((started, finished), (2, 2));
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

    fn reply(
        content: Vec<ContentBlock>,
        stop_reason: StopReason,
    ) -> anyhow::Result<CompletionResponse> {
        Ok(CompletionResponse {
            content,
            stop_reason,
            model: None,
            usage: None,
            context: None,
            compactions: Vec::new(),
        })
    }

    /// Approves calls whose input says `"ok": true`, and records what it was asked.
    #[derive(Default)]
    struct InputApprover(Mutex<Vec<String>>);

    #[async_trait]
    impl ToolApprover for InputApprover {
        async fn approve(&self, id: &str, _name: &str, input: &serde_json::Value) -> Approval {
            self.0.lock().unwrap().push(id.into());
            if input["ok"] == true {
                Approval::Allow
            } else {
                Approval::Deny
            }
        }
    }

    #[tokio::test]
    async fn approver_gates_tools_that_are_not_read_only() {
        let call = |id: &str, name: &str, input: serde_json::Value| ContentBlock::ToolUse {
            id: id.into(),
            name: name.into(),
            input,
        };
        let provider = Arc::new(ScriptedProvider::new(vec![
            reply(
                vec![
                    call("t1", "bash", json!({"input": "yes", "ok": true})),
                    call("t2", "bash", json!({"input": "no"})),
                    call("t3", "read_file", json!({"path": "/definitely/not/here"})),
                ],
                StopReason::ToolUse,
            ),
            reply(vec![text("done")], StopReason::EndTurn),
        ]));
        let default_exec = Arc::new(FakeExecutor::default());
        let mut agent = agent(provider, default_exec.clone());
        agent
            .tools
            .register(Arc::new(crate::tools::files::ReadFileTool));
        let turn_exec = Arc::new(FakeExecutor::default());
        let approver = Arc::new(InputApprover::default());
        let options = TurnOptions {
            executor: Some(turn_exec.clone()),
            approver: Some(approver.clone()),
        };
        let mut session = Session::new("s");

        agent
            .run_turn_with(
                &mut session,
                Message::user_text("hi"),
                &VecSink::default(),
                &CancellationToken::new(),
                &options,
            )
            .await
            .unwrap();

        // read_file is read-only, so only the bash calls were asked about.
        assert_eq!(*approver.0.lock().unwrap(), ["t1", "t2"]);
        // Only the approved call ran, on the turn's executor.
        assert_eq!(*turn_exec.ran.lock().unwrap(), ["yes"]);
        assert!(default_exec.ran.lock().unwrap().is_empty());
        let results = &session.messages[2].content;
        let ContentBlock::ToolResult {
            is_error, content, ..
        } = &results[1]
        else {
            panic!("expected tool result");
        };
        assert!(*is_error);
        assert!(content.contains("denied"), "{content}");
    }

    /// Sleeps briefly and records how many calls were running at once.
    struct CountingTool {
        name: &'static str,
        sequential: bool,
        running: Arc<std::sync::atomic::AtomicUsize>,
        peak: Arc<std::sync::atomic::AtomicUsize>,
    }

    #[async_trait]
    impl crate::tool::Tool for CountingTool {
        fn spec(&self) -> alx_protocol::ToolSpec {
            alx_protocol::ToolSpec {
                name: self.name.into(),
                description: String::new(),
                input_schema: json!({"type": "object"}),
            }
        }

        async fn call(&self, input: serde_json::Value, _ctx: &ToolContext) -> ToolOutput {
            use std::sync::atomic::Ordering::SeqCst;
            let now = self.running.fetch_add(1, SeqCst) + 1;
            self.peak.fetch_max(now, SeqCst);
            // Later calls finish first, so ordered results prove reordering.
            let ms = 60 - 20 * input["n"].as_u64().unwrap();
            tokio::time::sleep(std::time::Duration::from_millis(ms)).await;
            self.running.fetch_sub(1, SeqCst);
            ToolOutput::ok(format!("done {}", input["n"]))
        }

        fn sequential(&self) -> bool {
            self.sequential
        }
    }

    /// Runs three calls (the last to `last_tool`) and returns the peak
    /// concurrency and the result texts in order.
    async fn run_batch(sequential: bool, last_tool: &str) -> (usize, Vec<String>) {
        let running = Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let peak = Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let tool = |name, sequential| CountingTool {
            name,
            sequential,
            running: running.clone(),
            peak: peak.clone(),
        };
        let call = |n: u64, name: &str| ContentBlock::ToolUse {
            id: format!("t{n}"),
            name: name.into(),
            input: json!({"n": n}),
        };
        let provider = Arc::new(ScriptedProvider::new(vec![
            Ok(CompletionResponse {
                content: vec![call(0, "par"), call(1, "par"), call(2, last_tool)],
                stop_reason: StopReason::ToolUse,
                model: None,
                usage: None,
                context: None,
                compactions: Vec::new(),
            }),
            Ok(CompletionResponse {
                content: vec![text("ok")],
                stop_reason: StopReason::EndTurn,
                model: None,
                usage: None,
                context: None,
                compactions: Vec::new(),
            }),
        ]));
        let mut agent = agent(provider, Arc::new(FakeExecutor::default()));
        agent.tools = ToolRegistry::new()
            .with(tool("par", false))
            .with(tool("seq", sequential));
        let mut session = Session::new("s");
        agent
            .run_turn(
                &mut session,
                Message::user_text("hi"),
                &VecSink::default(),
                &CancellationToken::new(),
            )
            .await
            .unwrap();
        let results = session.messages[2]
            .content
            .iter()
            .map(|b| match b {
                ContentBlock::ToolResult { content, .. } => content.clone(),
                _ => panic!("expected tool result"),
            })
            .collect();
        (peak.load(std::sync::atomic::Ordering::SeqCst), results)
    }

    #[tokio::test]
    async fn tool_calls_run_concurrently_and_keep_their_order() {
        let (peak, results) = run_batch(true, "par").await;
        assert_eq!(peak, 3);
        assert_eq!(results, ["done 0", "done 1", "done 2"]);
    }

    #[tokio::test]
    async fn a_sequential_tool_makes_the_batch_run_in_order() {
        let (peak, results) = run_batch(true, "seq").await;
        assert_eq!(peak, 1);
        assert_eq!(results, ["done 0", "done 1", "done 2"]);
    }
}
