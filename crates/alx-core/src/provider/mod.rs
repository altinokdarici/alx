//! LLM backends. Each one maps the provider-neutral protocol types to its
//! own wire format; the agent loop only ever sees [`Provider`].

pub mod anthropic;
#[cfg(feature = "copilot")]
pub mod copilot;

use alx_protocol::{
    Compaction, ContentBlock, ContextUsage, Message, ModelInfo, ProviderInfo, StopReason, ToolSpec,
    Usage,
};
use async_trait::async_trait;
use std::time::Duration;

/// A provider error worth retrying: overload, rate limit, a 5xx, a dropped
/// connection. The agent loop retries these with backoff; any other error
/// ends the turn. Return it (as the `anyhow` error) only when nothing of the
/// response has been reported yet, since a retry starts the response over.
#[derive(Debug, Clone, PartialEq)]
pub struct TransientError {
    pub message: String,
    /// How long the server asked us to wait, if it said.
    pub retry_after: Option<Duration>,
}

impl std::fmt::Display for TransientError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.message)
    }
}

impl std::error::Error for TransientError {}

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

    /// Whether the provider keeps and compacts the conversation by itself
    /// (Copilot). If not, the agent loop compacts the session it sends.
    fn compacts_itself(&self) -> bool {
        false
    }

    /// The model's context window in tokens, if known. Needed for automatic
    /// compaction by the agent loop.
    fn context_window(&self) -> Option<u64> {
        None
    }

    /// Compact the conversation now, if the provider keeps one it can
    /// compact. `Ok(None)` means there was nothing to compact. Providers that
    /// can't compact return an error (the default).
    async fn compact(&self) -> anyhow::Result<Option<Compaction>> {
        anyhow::bail!("the {} provider cannot compact", self.info().name)
    }

    /// The models this provider offers. Empty (the default) if it can't say.
    async fn list_models(&self) -> anyhow::Result<Vec<ModelInfo>> {
        Ok(Vec::new())
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
