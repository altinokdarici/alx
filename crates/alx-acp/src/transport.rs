//! The ways a client can reach the ACP server.
//!
//! The server itself ([`crate::serve`]) runs on any transport; these are the
//! ready-made ones:
//!
//! - [`native`]: in the same process, over an in-memory channel. No
//!   serialization to bytes; the CLI uses it.
//! - [`serve_stdio`]: JSON-RPC on stdin/stdout, for editors (`alx --acp`).
//!
//! Remote transports (HTTP, WebSocket) are still to come.

use agent_client_protocol::{Channel, Error, Stdio};
use alx_core::{AgentFactory, SessionStore};
use std::sync::Arc;

/// Starts the server in this process and returns the client's end of the
/// connection, for `Client.builder().connect_with(..)`. The server stops when
/// the client end is dropped.
pub fn native(agents: Arc<dyn AgentFactory>, store: Arc<dyn SessionStore>) -> Channel {
    let (client_end, agent_end) = Channel::duplex();
    tokio::spawn(crate::serve(agents, store, agent_end));
    client_end
}

/// Serves on stdin/stdout until the client disconnects. Nothing else may
/// write to stdout meanwhile.
pub async fn serve_stdio(
    agents: Arc<dyn AgentFactory>,
    store: Arc<dyn SessionStore>,
) -> Result<(), Error> {
    crate::serve(agents, store, Stdio::new()).await
}
