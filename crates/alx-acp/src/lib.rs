//! Serves an alx agent over the Agent Client Protocol (ACP).
//!
//! [`serve`] runs the agent side of an ACP connection on any transport the
//! SDK can connect to; [`transport`] has the ready-made ones (native
//! in-process, stdio). The agent and
//! the session store are the ones from `alx-core`; this crate only translates
//! between ACP messages and alx's `Event`s.
//!
//! Tools run in the client's environment where the client offers it
//! (terminals, file reads and writes; see `client_env`), and every tool call
//! that is not read-only is approved by the user through
//! `session/request_permission`.

mod client_env;
mod events;
pub mod transport;

pub use transport::{native, serve_stdio};

/// `_meta` key of a `session/prompt` reply that names the model that answered.
pub const MODEL_META: &str = "model";

use agent_client_protocol::schema::v1::{
    AgentCapabilities, AvailableCommand, AvailableCommandsUpdate, CancelNotification,
    ClientCapabilities, CompactionStatus, CompactionUpdate, ContentBlock, Implementation,
    InitializeRequest, InitializeResponse, LoadSessionRequest, LoadSessionResponse, Meta,
    NewSessionRequest, NewSessionResponse, PromptRequest, PromptResponse, SessionId,
    SessionNotification, SessionUpdate, StopReason, Usage as AcpUsage,
};
use agent_client_protocol::{Agent, Client, ConnectTo, ConnectionTo, Error};
use alx_core::protocol::{Compaction, Event, Message, Usage};
use alx_core::{
    AgentLoop, Approval, CancellationToken, EventSink, Session, SessionStore, TurnOptions,
};
use client_env::{ClientApprover, ClientExecutor};
use events::AcpSink;
use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::{Arc, Mutex};

/// One open session: its history, the token that cancels its running turn,
/// and the user's "always" answers to permission requests.
struct Live {
    session: tokio::sync::Mutex<Session>,
    cancel: Mutex<Option<CancellationToken>>,
    always: Arc<Mutex<HashMap<String, Approval>>>,
}

struct Server {
    agent: Arc<dyn AgentLoop>,
    store: Arc<dyn SessionStore>,
    live: Mutex<HashMap<String, Arc<Live>>>,
    /// What the client said it can do, from `initialize`.
    client_caps: Mutex<ClientCapabilities>,
}

impl Server {
    fn open(&self, session: Session) -> SessionId {
        let id = session.id.clone();
        let live = Arc::new(Live {
            session: tokio::sync::Mutex::new(session),
            cancel: Mutex::new(None),
            always: Arc::default(),
        });
        self.live.lock().unwrap().insert(id.clone(), live);
        SessionId::new(id)
    }

    fn get(&self, id: &SessionId) -> Result<Arc<Live>, Error> {
        self.live
            .lock()
            .unwrap()
            .get(&*id.0)
            .cloned()
            .ok_or_else(|| Error::invalid_params().data(format!("unknown session `{}`", id.0)))
    }
}

fn prompt_text(blocks: &[ContentBlock]) -> String {
    blocks
        .iter()
        .filter_map(|b| match b {
            ContentBlock::Text(t) => Some(t.text.as_str()),
            _ => None,
        })
        .collect::<Vec<_>>()
        .join("\n")
}

/// The slash commands alx handles itself instead of sending to the model.
fn commands_update() -> SessionUpdate {
    SessionUpdate::AvailableCommandsUpdate(AvailableCommandsUpdate::new(vec![
        AvailableCommand::new("compact", "Compact the conversation to free up context"),
    ]))
}

fn acp_usage(u: &Usage) -> AcpUsage {
    let input = u.input_tokens + u.cache_read_tokens + u.cache_write_tokens;
    AcpUsage::new(input + u.output_tokens, input, u.output_tokens)
        .cached_read_tokens(u.cache_read_tokens)
        .cached_write_tokens(u.cache_write_tokens)
}

/// Runs `/compact`: reports progress as `compaction_update`s and ends the
/// turn. A provider that can't compact fails the compaction, not the turn.
async fn compact(agent: &dyn AgentLoop, sink: &AcpSink) {
    let id = "manual-compaction";
    sink.send(SessionUpdate::CompactionUpdate(CompactionUpdate::new(
        id,
        CompactionStatus::InProgress,
    )));
    // Catch the agent's `Compacted` event, to report it under this id.
    struct Catch(Mutex<Option<Compaction>>);
    impl EventSink for Catch {
        fn emit(&self, event: Event) {
            if let Event::Compacted(c) = event {
                *self.0.lock().unwrap() = Some(c);
            }
        }
    }
    let caught = Catch(Mutex::new(None));
    let done = match agent.compact(&caught).await {
        Ok(_) => match caught.0.lock().unwrap().take() {
            Some(c) => events::compaction_update(id.into(), &c),
            None => SessionUpdate::CompactionUpdate(CompactionUpdate::new(
                id,
                CompactionStatus::Completed,
            )),
        },
        Err(e) => SessionUpdate::CompactionUpdate(
            CompactionUpdate::new(id, CompactionStatus::Failed).error(format!("{e:#}")),
        ),
    };
    sink.send(done);
}

fn stop_reason(r: &alx_core::protocol::StopReason) -> StopReason {
    use alx_core::protocol::StopReason as S;
    match r {
        S::MaxTokens => StopReason::MaxTokens,
        S::Refusal => StopReason::Refusal,
        S::Cancelled => StopReason::Cancelled,
        _ => StopReason::EndTurn,
    }
}

/// Runs the agent side of an ACP connection until the client disconnects.
pub async fn serve(
    agent: Arc<dyn AgentLoop>,
    store: Arc<dyn SessionStore>,
    transport: impl ConnectTo<Agent>,
) -> Result<(), Error> {
    let server = Arc::new(Server {
        agent,
        store,
        live: Mutex::new(HashMap::new()),
        client_caps: Mutex::default(),
    });

    Agent
        .builder()
        .name("alx")
        .on_receive_request(
            {
                let server = server.clone();
                async move |init: InitializeRequest, responder, _cx: ConnectionTo<Client>| {
                    *server.client_caps.lock().unwrap() = init.client_capabilities;
                    responder.respond(
                        InitializeResponse::new(init.protocol_version)
                            .agent_capabilities(AgentCapabilities::new().load_session(true))
                            .agent_info(Implementation::new("alx", env!("CARGO_PKG_VERSION"))),
                    )
                }
            },
            agent_client_protocol::on_receive_request!(),
        )
        .on_receive_request(
            {
                let server = server.clone();
                async move |req: NewSessionRequest, responder, cx: ConnectionTo<Client>| {
                    let mut session = Session::new(Session::generate_id());
                    session.cwd = Some(req.cwd.to_string_lossy().into_owned());
                    let id = server.open(session);
                    responder.respond(NewSessionResponse::new(id.clone()))?;
                    cx.send_notification(SessionNotification::new(id, commands_update()))
                }
            },
            agent_client_protocol::on_receive_request!(),
        )
        .on_receive_request(
            {
                let server = server.clone();
                async move |req: LoadSessionRequest, responder, cx: ConnectionTo<Client>| {
                    let server = server.clone();
                    cx.clone().spawn(async move {
                        let loaded = server
                            .store
                            .load(&req.session_id.0)
                            .await
                            .map_err(|e| Error::internal_error().data(format!("{e:#}")))?;
                        let Some(session) = loaded else {
                            return responder.respond_with_error(
                                Error::invalid_params()
                                    .data(format!("unknown session `{}`", req.session_id.0)),
                            );
                        };
                        // ACP wants the history replayed as updates before the reply.
                        events::replay(&cx, &req.session_id, &session.messages)?;
                        server.open(session);
                        responder.respond(LoadSessionResponse::new())?;
                        cx.send_notification(SessionNotification::new(
                            req.session_id,
                            commands_update(),
                        ))
                    })
                }
            },
            agent_client_protocol::on_receive_request!(),
        )
        .on_receive_request(
            {
                let server = server.clone();
                async move |req: PromptRequest, responder, cx: ConnectionTo<Client>| {
                    let live = server.get(&req.session_id)?;
                    let server = server.clone();
                    // A turn takes a long time; running it here would block the
                    // dispatch loop, and with it `session/cancel`.
                    cx.clone().spawn(async move {
                        let cancel = CancellationToken::new();
                        *live.cancel.lock().unwrap() = Some(cancel.clone());
                        let sink = AcpSink::new(cx.clone(), req.session_id.clone());
                        let mut session = live.session.lock().await;
                        let text = prompt_text(&req.prompt);
                        if text.trim() == "/compact" {
                            compact(&*server.agent, &sink).await;
                            *live.cancel.lock().unwrap() = None;
                            return responder.respond(PromptResponse::new(StopReason::EndTurn));
                        }
                        let cwd = session
                            .cwd
                            .as_ref()
                            .map_or_else(|| PathBuf::from("."), PathBuf::from);
                        let options = TurnOptions {
                            executor: Some(Arc::new(ClientExecutor {
                                cx: cx.clone(),
                                session: req.session_id.clone(),
                                cwd,
                                caps: server.client_caps.lock().unwrap().clone(),
                            })),
                            approver: Some(Arc::new(ClientApprover {
                                cx,
                                session: req.session_id.clone(),
                                always: live.always.clone(),
                            })),
                        };
                        let result = server
                            .agent
                            .run_turn_with(
                                &mut session,
                                Message::user_text(text),
                                &sink,
                                &cancel,
                                &options,
                            )
                            .await;
                        *live.cancel.lock().unwrap() = None;
                        match result {
                            Ok(reason) => {
                                if let Err(e) = server.store.save(&session).await {
                                    return responder.respond_with_error(
                                        Error::internal_error().data(format!("{e:#}")),
                                    );
                                }
                                let mut reply = PromptResponse::new(stop_reason(&reason));
                                if let Some(u) = sink.usage() {
                                    reply = reply.usage(acp_usage(&u));
                                }
                                if let Some(model) = sink.model() {
                                    let mut meta = Meta::new();
                                    meta.insert(MODEL_META.into(), model.into());
                                    reply = reply.meta(meta);
                                }
                                responder.respond(reply)
                            }
                            Err(e) => responder
                                .respond_with_error(Error::internal_error().data(format!("{e:#}"))),
                        }
                    })
                }
            },
            agent_client_protocol::on_receive_request!(),
        )
        .on_receive_notification(
            {
                let server = server.clone();
                async move |note: CancelNotification, _cx: ConnectionTo<Client>| {
                    if let Ok(live) = server.get(&note.session_id)
                        && let Some(token) = live.cancel.lock().unwrap().as_ref()
                    {
                        token.cancel();
                    }
                    Ok(())
                }
            },
            agent_client_protocol::on_receive_notification!(),
        )
        .connect_to(transport)
        .await
}
