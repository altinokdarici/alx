//! Conversation state and its persistence.

use alx_protocol::Message;
use async_trait::async_trait;
use std::collections::HashMap;
use tokio::sync::Mutex;

#[derive(Debug, Clone, Default, PartialEq)]
pub struct Session {
    pub id: String,
    pub messages: Vec<Message>,
}

impl Session {
    pub fn new(id: impl Into<String>) -> Self {
        Self {
            id: id.into(),
            messages: Vec::new(),
        }
    }
}

#[async_trait]
pub trait SessionStore: Send + Sync {
    async fn load(&self, id: &str) -> anyhow::Result<Option<Session>>;
    async fn save(&self, session: &Session) -> anyhow::Result<()>;
}

/// Keeps sessions in memory only, so they are lost on exit.
#[derive(Default)]
pub struct InMemorySessionStore {
    sessions: Mutex<HashMap<String, Session>>,
}

#[async_trait]
impl SessionStore for InMemorySessionStore {
    async fn load(&self, id: &str) -> anyhow::Result<Option<Session>> {
        Ok(self.sessions.lock().await.get(id).cloned())
    }

    async fn save(&self, session: &Session) -> anyhow::Result<()> {
        self.sessions
            .lock()
            .await
            .insert(session.id.clone(), session.clone());
        Ok(())
    }
}
