//! alx core: the agent loop plus the pluggable layers it is built from.
//!
//! Every layer is a trait with at least one built-in implementation:
//!
//! | Layer          | Trait            | Built-in                       |
//! |----------------|------------------|--------------------------------|
//! | LLM backend    | [`Provider`]     | [`AnthropicProvider`], `CopilotProvider` |
//! | Tools          | [`Tool`]         | [`BashTool`]                   |
//! | Command runner | [`Executor`]     | [`LocalExecutor`]              |
//! | Agent loop     | [`AgentLoop`]    | [`SingleAgent`]                |
//! | Persistence    | [`SessionStore`] | [`InMemorySessionStore`], [`FileSessionStore`] |
//! | UI / output    | [`EventSink`]    | any client (see `alx-cli`)     |
//!
//! The core never prints. It reports progress through an [`EventSink`], so
//! any UI can be plugged in on top.

pub mod agent;
pub mod env;
pub mod event;
pub mod executor;
pub mod provider;
pub mod session;
pub mod settings;
pub mod tool;
pub mod tools;

pub use agent::{AgentConfig, AgentLoop, SingleAgent};
pub use env::{agent_from_settings, load_dotenv, provider_from_settings};
pub use event::EventSink;
pub use executor::{ExecOutput, ExecRequest, Executor, LocalExecutor};
pub use provider::anthropic::{AnthropicConfig, AnthropicProvider};
#[cfg(feature = "copilot")]
pub use provider::copilot::{CopilotConfig, CopilotProvider};
pub use provider::{CompletionRequest, CompletionResponse, Provider};
pub use session::{
    FileSessionStore, InMemorySessionStore, Session, SessionStore, SessionSummary,
    default_session_dir,
};
pub use settings::{AnthropicSettings, CopilotSettings, ProviderKind, Settings};
pub use tool::{Tool, ToolContext, ToolOutput, ToolRegistry};
pub use tools::bash::BashTool;

pub use alx_protocol as protocol;
