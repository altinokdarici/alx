//! Serves an alx agent over the Agent Client Protocol (ACP).
//!
//! [`serve`] runs the agent side of an ACP connection on any transport the
//! SDK can connect to: stdio for an editor, or an in-process
//! `Channel::duplex()` pair for a client in the same process. The agent and
//! the session store are the ones from `alx-core`; this crate only translates
//! between ACP messages and alx's `Event`s.

mod events;

use agent_client_protocol::schema::v1::{
    AgentCapabilities, CancelNotification, ContentBlock, Implementation, InitializeRequest,
    InitializeResponse, LoadSessionRequest, LoadSessionResponse, NewSessionRequest,
    NewSessionResponse, PromptRequest, PromptResponse, SessionId, StopReason,
};
use agent_client_protocol::{Agent, Client, ConnectTo, ConnectionTo, Error};
use alx_core::protocol::Message;
use alx_core::{AgentLoop, CancellationToken, Session, SessionStore};
use events::AcpSink;
use std::collections::HashMap;
use std::sync::{Arc, Mutex};

/// One open session: its history, and the token that cancels its running turn.
struct Live {
    session: tokio::sync::Mutex<Session>,
    cancel: Mutex<Option<CancellationToken>>,
}

struct Server {
    agent: Arc<dyn AgentLoop>,
    store: Arc<dyn SessionStore>,
    live: Mutex<HashMap<String, Arc<Live>>>,
}

impl Server {
    fn open(&self, session: Session) -> SessionId {
        let id = session.id.clone();
        let live = Arc::new(Live {
            session: tokio::sync::Mutex::new(session),
            cancel: Mutex::new(None),
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
    });

    Agent
        .builder()
        .name("alx")
        .on_receive_request(
            async move |init: InitializeRequest, responder, _cx: ConnectionTo<Client>| {
                responder.respond(
                    InitializeResponse::new(init.protocol_version)
                        .agent_capabilities(AgentCapabilities::new().load_session(true))
                        .agent_info(Implementation::new("alx", env!("CARGO_PKG_VERSION"))),
                )
            },
            agent_client_protocol::on_receive_request!(),
        )
        .on_receive_request(
            {
                let server = server.clone();
                async move |req: NewSessionRequest, responder, _cx: ConnectionTo<Client>| {
                    let mut session = Session::new(Session::generate_id());
                    session.cwd = Some(req.cwd.to_string_lossy().into_owned());
                    responder.respond(NewSessionResponse::new(server.open(session)))
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
                        responder.respond(LoadSessionResponse::new())
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
                        let sink = AcpSink::new(cx, req.session_id.clone());
                        let mut session = live.session.lock().await;
                        let input = Message::user_text(prompt_text(&req.prompt));
                        let result = server
                            .agent
                            .run_turn(&mut session, input, &sink, &cancel)
                            .await;
                        *live.cancel.lock().unwrap() = None;
                        match result {
                            Ok(reason) => {
                                if let Err(e) = server.store.save(&session).await {
                                    return responder.respond_with_error(
                                        Error::internal_error().data(format!("{e:#}")),
                                    );
                                }
                                responder.respond(PromptResponse::new(stop_reason(&reason)))
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
