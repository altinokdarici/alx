//! alx core: the agent loop plus the pluggable layers it is built from.
//!
//! Every layer is a trait with at least one built-in implementation:
//!
//! | Layer          | Trait            | Built-in                       |
//! |----------------|------------------|--------------------------------|
//! | LLM backend    | [`Provider`]     | [`AnthropicProvider`]          |
//! | Tools          | [`Tool`]         | [`BashTool`]                   |
//! | Command runner | [`Executor`]     | [`LocalExecutor`]              |
//! | Agent loop     | [`AgentLoop`]    | [`SingleAgent`]                |
//! | Persistence    | [`SessionStore`] | [`InMemorySessionStore`]       |
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
pub mod tool;
pub mod tools;

pub use agent::{AgentConfig, AgentLoop, SingleAgent};
pub use env::{agent_from_env, load_dotenv};
pub use event::EventSink;
pub use executor::{ExecOutput, ExecRequest, Executor, LocalExecutor};
pub use provider::anthropic::{AnthropicConfig, AnthropicProvider};
pub use provider::{CompletionRequest, CompletionResponse, Provider};
pub use session::{InMemorySessionStore, Session, SessionStore};
pub use tool::{Tool, ToolContext, ToolOutput, ToolRegistry};
pub use tools::bash::BashTool;

pub use alx_protocol as protocol;
