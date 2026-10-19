//! LLM backends. Each one maps the provider-neutral protocol types to its
//! own wire format; the agent loop only ever sees [`Provider`].

pub mod anthropic;
#[cfg(feature = "copilot")]
pub mod copilot;

use alx_protocol::{
    Compaction, ContentBlock, ContextUsage, Message, ProviderInfo, StopReason, ToolSpec, Usage,
};
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
    /// Token usage for this request, when the provider reports it.
    pub usage: Option<Usage>,
    /// How full the context window is after this response, when known.
    pub context: Option<ContextUsage>,
    /// Compactions the provider ran by itself while producing this response.
    pub compactions: Vec<Compaction>,
}

#[async_trait]
pub trait Provider: Send + Sync {
    /// What this provider is, for clients to display.
    fn info(&self) -> ProviderInfo;

    async fn complete(&self, req: CompletionRequest) -> anyhow::Result<CompletionResponse>;

    /// Compact the conversation now, if the provider keeps one it can
    /// compact. `Ok(None)` means there was nothing to compact. Providers that
    /// can't compact return an error (the default).
    async fn compact(&self) -> anyhow::Result<Option<Compaction>> {
        anyhow::bail!("the {} provider cannot compact", self.info().name)
    }

    /// Like [`complete`](Self::complete), but calls `on_text` with each piece
    /// of text as the model writes it. The returned response is the same
    /// whole one. Providers that can't stream just don't call `on_text`.
    async fn complete_streaming(
        &self,
        req: CompletionRequest,
        on_text: &(dyn for<'a> Fn(&'a str) + Send + Sync),
    ) -> anyhow::Result<CompletionResponse> {
        let _ = on_text;
        self.complete(req).await
    }
}
