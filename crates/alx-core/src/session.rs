//! Conversation state and its persistence.

use alx_protocol::{ContentBlock, Message, Role};
use anyhow::{Context, bail};
use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::PathBuf;
use std::time::{SystemTime, UNIX_EPOCH};
use tokio::sync::Mutex;

/// Seconds since the Unix epoch.
pub fn unix_now() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |d| d.as_secs())
}

#[derive(Debug, Clone, Default, PartialEq)]
pub struct Session {
    pub id: String,
    pub messages: Vec<Message>,
    /// Unix seconds when the session was created.
    pub created_at: u64,
    /// Unix seconds of the last save. Stamped by the store.
    pub updated_at: u64,
    /// Working directory the session was started in, so clients can offer
    /// "continue the last session in this project".
    pub cwd: Option<String>,
}

impl Session {
    pub fn new(id: impl Into<String>) -> Self {
        let now = unix_now();
        Self {
            id: id.into(),
            messages: Vec::new(),
            created_at: now,
            updated_at: now,
            cwd: None,
        }
    }

    /// A fresh, sortable, unique-enough session id.
    pub fn generate_id() -> String {
        let millis = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map_or(0, |d| d.as_millis());
        format!("{millis:x}-{:x}", std::process::id())
    }

    /// The first thing the user said, for listings.
    fn preview(&self) -> String {
        preview_of(&self.messages)
    }

    fn summary(&self) -> SessionSummary {
        SessionSummary {
            id: self.id.clone(),
            created_at: self.created_at,
            updated_at: self.updated_at,
            cwd: self.cwd.clone(),
            message_count: self.messages.len(),
            preview: self.preview(),
        }
    }
}

fn preview_of(messages: &[Message]) -> String {
    let text = messages
        .iter()
        .filter(|m| m.role == Role::User)
        .flat_map(|m| &m.content)
        .find_map(|b| match b {
            ContentBlock::Text { text } => Some(text.as_str()),
            _ => None,
        })
        .unwrap_or("");
    let line = text.lines().next().unwrap_or("").trim();
    if line.chars().count() > 80 {
        let cut: String = line.chars().take(79).collect();
        format!("{cut}…")
    } else {
        line.to_owned()
    }
}

/// What a listing shows about a stored session.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SessionSummary {
    pub id: String,
    pub created_at: u64,
    pub updated_at: u64,
    pub cwd: Option<String>,
    pub message_count: usize,
    pub preview: String,
}

#[async_trait]
pub trait SessionStore: Send + Sync {
    async fn load(&self, id: &str) -> anyhow::Result<Option<Session>>;
    async fn save(&self, session: &Session) -> anyhow::Result<()>;
    /// All stored sessions, most recently updated first.
    async fn list(&self) -> anyhow::Result<Vec<SessionSummary>>;
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
        let mut saved = session.clone();
        saved.updated_at = unix_now();
        self.sessions.lock().await.insert(saved.id.clone(), saved);
        Ok(())
    }

    async fn list(&self) -> anyhow::Result<Vec<SessionSummary>> {
        let mut all: Vec<_> = self
            .sessions
            .lock()
            .await
            .values()
            .map(Session::summary)
            .collect();
        all.sort_by(|a, b| b.updated_at.cmp(&a.updated_at).then(b.id.cmp(&a.id)));
        Ok(all)
    }
}

/// First line of a session file.
#[derive(Serialize, Deserialize)]
struct Header {
    id: String,
    created_at: u64,
    updated_at: u64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    cwd: Option<String>,
}

/// One JSONL file per session in a directory: a header line, then one
/// message per line. Writes go to a temp file that is renamed into place, so
/// a crash never leaves a half-written session.
pub struct FileSessionStore {
    dir: PathBuf,
}

impl FileSessionStore {
    pub fn new(dir: impl Into<PathBuf>) -> Self {
        Self { dir: dir.into() }
    }

    fn path(&self, id: &str) -> anyhow::Result<PathBuf> {
        let ok = !id.is_empty()
            && id != "."
            && id != ".."
            && id
                .chars()
                .all(|c| c.is_ascii_alphanumeric() || matches!(c, '-' | '_' | '.'));
        if !ok {
            bail!("invalid session id `{id}`");
        }
        Ok(self.dir.join(format!("{id}.jsonl")))
    }
}

fn parse(text: &str) -> anyhow::Result<Session> {
    let mut lines = text.lines().filter(|l| !l.trim().is_empty());
    let header: Header = serde_json::from_str(lines.next().context("empty session file")?)
        .context("bad session header")?;
    let messages = lines
        .map(|l| serde_json::from_str(l).context("bad message line"))
        .collect::<anyhow::Result<Vec<Message>>>()?;
    Ok(Session {
        id: header.id,
        messages,
        created_at: header.created_at,
        updated_at: header.updated_at,
        cwd: header.cwd,
    })
}

fn render(session: &Session, updated_at: u64) -> anyhow::Result<String> {
    let mut out = serde_json::to_string(&Header {
        id: session.id.clone(),
        created_at: session.created_at,
        updated_at,
        cwd: session.cwd.clone(),
    })?;
    out.push('\n');
    for m in &session.messages {
        out.push_str(&serde_json::to_string(m)?);
        out.push('\n');
    }
    Ok(out)
}

#[async_trait]
impl SessionStore for FileSessionStore {
    async fn load(&self, id: &str) -> anyhow::Result<Option<Session>> {
        let path = self.path(id)?;
        match tokio::fs::read_to_string(&path).await {
            Ok(text) => Ok(Some(
                parse(&text).with_context(|| format!("reading {}", path.display()))?,
            )),
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(None),
            Err(e) => Err(e).with_context(|| format!("reading {}", path.display())),
        }
    }

    async fn save(&self, session: &Session) -> anyhow::Result<()> {
        let path = self.path(&session.id)?;
        tokio::fs::create_dir_all(&self.dir)
            .await
            .with_context(|| format!("creating {}", self.dir.display()))?;
        let body = render(session, unix_now())?;
        let tmp = path.with_extension("jsonl.tmp");
        tokio::fs::write(&tmp, body)
            .await
            .with_context(|| format!("writing {}", tmp.display()))?;
        tokio::fs::rename(&tmp, &path)
            .await
            .with_context(|| format!("writing {}", path.display()))?;
        Ok(())
    }

    async fn list(&self) -> anyhow::Result<Vec<SessionSummary>> {
        let mut entries = match tokio::fs::read_dir(&self.dir).await {
            Ok(e) => e,
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(Vec::new()),
            Err(e) => return Err(e).with_context(|| format!("listing {}", self.dir.display())),
        };
        let mut all = Vec::new();
        while let Some(entry) = entries.next_entry().await? {
            let path = entry.path();
            if path.extension().and_then(|e| e.to_str()) != Some("jsonl") {
                continue;
            }
            // A corrupt file must not hide the others.
            let Ok(text) = tokio::fs::read_to_string(&path).await else {
                continue;
            };
            if let Ok(session) = parse(&text) {
                all.push(session.summary());
            }
        }
        all.sort_by(|a, b| b.updated_at.cmp(&a.updated_at).then(b.id.cmp(&a.id)));
        Ok(all)
    }
}

/// Where sessions live by default: `$XDG_DATA_HOME/alx/sessions`, or
/// `~/.local/share/alx/sessions`.
pub fn default_session_dir() -> Option<PathBuf> {
    let base = match std::env::var_os("XDG_DATA_HOME").filter(|v| !v.is_empty()) {
        Some(dir) => PathBuf::from(dir),
        None => PathBuf::from(std::env::var_os("HOME")?).join(".local/share"),
    };
    Some(base.join("alx").join("sessions"))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn temp_dir(name: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!(
            "alx-test-{name}-{}-{}",
            std::process::id(),
            Session::generate_id()
        ));
        let _ = std::fs::remove_dir_all(&dir);
        dir
    }

    fn session(id: &str, text: &str) -> Session {
        let mut s = Session::new(id);
        s.cwd = Some("/proj".into());
        s.messages.push(Message::user_text(text));
        s.messages.push(Message {
            role: Role::Assistant,
            content: vec![
                ContentBlock::Text { text: "ok".into() },
                ContentBlock::ToolUse {
                    id: "t1".into(),
                    name: "bash".into(),
                    input: serde_json::json!({"command": "ls"}),
                },
            ],
        });
        s
    }

    #[tokio::test]
    async fn file_store_round_trips() {
        let dir = temp_dir("roundtrip");
        let store = FileSessionStore::new(&dir);
        let s = session("a", "hello\nworld");
        store.save(&s).await.unwrap();
        let loaded = store.load("a").await.unwrap().unwrap();
        assert_eq!(loaded.messages, s.messages);
        assert_eq!(loaded.cwd, s.cwd);
        assert_eq!(loaded.created_at, s.created_at);
        assert!(store.load("missing").await.unwrap().is_none());
        // Saving again overwrites rather than duplicating.
        store.save(&loaded).await.unwrap();
        assert_eq!(store.load("a").await.unwrap().unwrap().messages.len(), 2);
        let _ = std::fs::remove_dir_all(dir);
    }

    #[tokio::test]
    async fn file_store_lists_newest_first_and_skips_corrupt_files() {
        let dir = temp_dir("list");
        let store = FileSessionStore::new(&dir);
        assert!(store.list().await.unwrap().is_empty()); // dir does not exist yet
        store.save(&session("old", "first")).await.unwrap();
        // Make "old" older than "new" without sleeping.
        let text = std::fs::read_to_string(dir.join("old.jsonl")).unwrap();
        let (head, rest) = text.split_once('\n').unwrap();
        let mut head: serde_json::Value = serde_json::from_str(head).unwrap();
        head["updated_at"] = 1.into();
        let text = format!("{head}\n{rest}");
        std::fs::write(dir.join("old.jsonl"), text).unwrap();
        store.save(&session("new", "second")).await.unwrap();
        std::fs::write(dir.join("bad.jsonl"), "not json").unwrap();
        std::fs::write(dir.join("notes.txt"), "ignored").unwrap();

        let all = store.list().await.unwrap();
        let ids: Vec<_> = all.iter().map(|s| s.id.as_str()).collect();
        assert_eq!(ids, ["new", "old"]);
        assert_eq!(all[0].preview, "second");
        assert_eq!(all[0].message_count, 2);
        let _ = std::fs::remove_dir_all(dir);
    }

    #[tokio::test]
    async fn file_store_rejects_path_traversal_ids() {
        let store = FileSessionStore::new(temp_dir("ids"));
        for id in ["../x", "a/b", "", "..", "a b"] {
            assert!(store.load(id).await.is_err(), "{id:?} should be rejected");
        }
    }

    #[tokio::test]
    async fn in_memory_store_lists() {
        let store = InMemorySessionStore::default();
        store.save(&session("a", "hi")).await.unwrap();
        let all = store.list().await.unwrap();
        assert_eq!(all.len(), 1);
        assert_eq!(all[0].preview, "hi");
    }

    #[test]
    fn preview_truncates_long_lines() {
        let long = "x".repeat(200);
        let p = preview_of(&[Message::user_text(long)]);
        assert_eq!(p.chars().count(), 80);
        assert!(p.ends_with('…'));
    }
}
