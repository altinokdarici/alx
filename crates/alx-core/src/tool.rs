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
    /// Images attached to the result, as `(media_type, base64 data)`.
    pub images: Vec<(String, String)>,
}

impl ToolOutput {
    pub fn ok(content: impl Into<String>) -> Self {
        Self {
            content: content.into(),
            is_error: false,
            images: Vec::new(),
        }
    }
    pub fn with_image(mut self, media_type: impl Into<String>, data: impl Into<String>) -> Self {
        self.images.push((media_type.into(), data.into()));
        self
    }
    pub fn error(content: impl Into<String>) -> Self {
        Self {
            content: content.into(),
            is_error: true,
            images: Vec::new(),
        }
    }
}

#[async_trait]
pub trait Tool: Send + Sync {
    fn spec(&self) -> ToolSpec;

    /// Invalid input should come back as an error `ToolOutput` (for the
    /// model to see), not as a panic.
    async fn call(&self, input: Value, ctx: &ToolContext) -> ToolOutput;

    /// Whether the tool only reads. Read-only tools run without asking a
    /// [`ToolApprover`].
    fn read_only(&self) -> bool {
        false
    }

    /// Whether a batch of tool calls that includes this tool must run one
    /// call at a time. By default, calls in a batch run concurrently.
    fn sequential(&self) -> bool {
        false
    }
}

/// The answer to a request to run a tool call.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Approval {
    Allow,
    Deny,
}

/// Decides whether a tool call may run, e.g. by asking the user. The agent
/// asks it before every call to a tool that is not [`Tool::read_only`].
#[async_trait]
pub trait ToolApprover: Send + Sync {
    async fn approve(&self, id: &str, name: &str, input: &Value) -> Approval;
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

    /// Whether `name` is a known read-only tool.
    pub fn is_sequential(&self, name: &str) -> bool {
        self.tools.get(name).is_some_and(|t| t.sequential())
    }

    pub fn is_read_only(&self, name: &str) -> bool {
        self.tools.get(name).is_some_and(|t| t.read_only())
    }

    pub async fn call(&self, name: &str, input: Value, ctx: &ToolContext) -> ToolOutput {
        match self.tools.get(name) {
            Some(tool) => tool.call(input, ctx).await,
            None => ToolOutput::error(format!("Invalid tool call: {name}")),
        }
    }
}
