use crate::executor::Executor;
use alx_protocol::ToolSpec;
use async_trait::async_trait;
use serde_json::Value;
use std::collections::BTreeMap;
use std::sync::Arc;

/// What a tool gets from the environment. Tools don't own an executor, so
/// swapping the agent's executor (e.g. to Docker) moves every tool with it.
#[derive(Clone)]
pub struct ToolContext {
    pub executor: Arc<dyn Executor>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ToolOutput {
    pub content: String,
    pub is_error: bool,
}

impl ToolOutput {
    pub fn ok(content: impl Into<String>) -> Self {
        Self {
            content: content.into(),
            is_error: false,
        }
    }
    pub fn error(content: impl Into<String>) -> Self {
        Self {
            content: content.into(),
            is_error: true,
        }
    }
}

#[async_trait]
pub trait Tool: Send + Sync {
    fn spec(&self) -> ToolSpec;

    /// Invalid input should come back as an error `ToolOutput` (for the
    /// model to see), not as a panic.
    async fn call(&self, input: Value, ctx: &ToolContext) -> ToolOutput;
}

/// The set of tools offered to the model, keyed by name.
#[derive(Default, Clone)]
pub struct ToolRegistry {
    tools: BTreeMap<String, Arc<dyn Tool>>,
}

impl ToolRegistry {
    pub fn new() -> Self {
        Self::default()
    }

    /// Add a tool, replacing any existing tool with the same name.
    pub fn with(mut self, tool: impl Tool + 'static) -> Self {
        self.register(Arc::new(tool));
        self
    }

    pub fn register(&mut self, tool: Arc<dyn Tool>) {
        self.tools.insert(tool.spec().name, tool);
    }

    pub fn specs(&self) -> Vec<ToolSpec> {
        self.tools.values().map(|t| t.spec()).collect()
    }

    pub async fn call(&self, name: &str, input: Value, ctx: &ToolContext) -> ToolOutput {
        match self.tools.get(name) {
            Some(tool) => tool.call(input, ctx).await,
            None => ToolOutput::error(format!("Invalid tool call: {name}")),
        }
    }
}
