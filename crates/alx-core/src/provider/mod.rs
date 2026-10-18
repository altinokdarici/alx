//! LLM backends.

pub mod anthropic;

use alx_protocol::{ContentBlock, Message, StopReason, ToolSpec};
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
}

#[async_trait]
pub trait Provider: Send + Sync {
    async fn complete(&self, req: CompletionRequest) -> anyhow::Result<CompletionResponse>;
}
