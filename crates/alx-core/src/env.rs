//! Building an agent from the process environment. Shared by every binary
//! (the `alx` CLI and the headless `alxd`) so they are wired identically.

use crate::{
    AgentConfig, AnthropicConfig, AnthropicProvider, BashTool, LocalExecutor, SingleAgent,
    ToolContext, ToolRegistry,
};
use anyhow::Context;
use std::sync::Arc;

/// Default agent: Anthropic provider, bash tool, local executor.
///
/// Reads `CLAUDE_API_KEY` (required) and `ALX_MODEL` (optional). Swap any
/// part here: another `Provider`, `Executor` (Docker, remote, ...), more
/// tools, or a different `AgentLoop` entirely.
pub fn agent_from_env() -> anyhow::Result<SingleAgent> {
    let api_key = std::env::var("CLAUDE_API_KEY")
        .context("CLAUDE_API_KEY environment variable is not set")?;
    let mut provider_config = AnthropicConfig::new(api_key);
    if let Ok(model) = std::env::var("ALX_MODEL") {
        provider_config.model = model;
    }
    Ok(SingleAgent {
        config: AgentConfig::default(),
        provider: Arc::new(AnthropicProvider::new(provider_config)),
        tools: ToolRegistry::new().with(BashTool),
        tool_ctx: ToolContext {
            executor: Arc::new(LocalExecutor),
        },
    })
}

/// Minimal `.env` loader: `KEY=VALUE` lines from `./.env`, without
/// overriding variables that are already set.
///
/// Call it at startup, before any other threads exist (it uses
/// `std::env::set_var`).
pub fn load_dotenv() {
    let Ok(contents) = std::fs::read_to_string(".env") else {
        return;
    };
    for (key, value) in parse_dotenv(&contents) {
        if std::env::var_os(key).is_none() {
            // SAFETY: documented to be called at startup before any other
            // threads exist.
            unsafe { std::env::set_var(key, value) };
        }
    }
}

fn parse_dotenv(contents: &str) -> impl Iterator<Item = (&str, &str)> {
    contents.lines().filter_map(|line| {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            return None;
        }
        let (key, value) = line.split_once('=')?;
        let key = key.trim().trim_start_matches("export ").trim();
        let value = value.trim().trim_matches(|c| c == '"' || c == '\'');
        Some((key, value))
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_dotenv_lines() {
        let parsed: Vec<_> =
            parse_dotenv("# comment\n\nA=1\nexport B=\"two\"\n C = 'three' \nnoequals\n").collect();
        assert_eq!(parsed, [("A", "1"), ("B", "two"), ("C", "three")]);
    }
}
