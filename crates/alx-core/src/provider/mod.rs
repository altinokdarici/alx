//! LLM backends. Each one maps the provider-neutral protocol types to its
//! own wire format; the agent loop only ever sees [`Provider`].

pub mod anthropic;
#[cfg(feature = "copilot")]
pub mod copilot;

use alx_protocol::{ContentBlock, Message, ProviderInfo, StopReason, ToolSpec};
use async_trait::async_trait;

#[derive(Debug, Clone)]
pub struct CompletionRequest {
    pub system: String,
    pub messages: Vec<Message>,
    pub tools: Vec<ToolSpec>,
    pub max_tokens: u32,
}

#[derive(Debug, Clone, PartialEq)]
pub struct CompletionResponse {
    pub content: Vec<ContentBlock>,
    pub stop_reason: StopReason,
    /// The model that actually produced this response, when known.
    pub model: Option<String>,
}

#[async_trait]
pub trait Provider: Send + Sync {
    /// What this provider is, for clients to display.
    fn info(&self) -> ProviderInfo;

    async fn complete(&self, req: CompletionRequest) -> anyhow::Result<CompletionResponse>;
}
