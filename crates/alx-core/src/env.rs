//! Building an agent from the process environment. Kept in the core rather
//! than in a binary so every client is wired the same way.

use crate::{
    AgentConfig, AnthropicConfig, AnthropicProvider, BashTool, LocalExecutor, Provider,
    SingleAgent, ToolContext, ToolRegistry,
};
use anyhow::{Context, bail};
use std::sync::Arc;

/// Default agent: the provider from [`provider_from_env`], bash tool, local
/// executor.
///
/// Swap any part here: another `Provider`, `Executor` (Docker, remote, ...),
/// more tools, or a different `AgentLoop` entirely.
pub fn agent_from_env() -> anyhow::Result<SingleAgent> {
    Ok(SingleAgent {
        config: AgentConfig::default(),
        provider: provider_from_env()?,
        tools: ToolRegistry::new().with(BashTool),
        tool_ctx: ToolContext {
            executor: Arc::new(LocalExecutor),
        },
    })
}

/// The LLM backend named by `ALX_PROVIDER`:
///
/// - `anthropic` (default): the Anthropic API. Needs `CLAUDE_API_KEY`.
/// - `copilot`: GitHub Copilot, through the `copilot` CLI and its login.
///
/// `ALX_MODEL` overrides the model for either.
pub fn provider_from_env() -> anyhow::Result<Arc<dyn Provider>> {
    let model = std::env::var("ALX_MODEL").ok();
    let name = std::env::var("ALX_PROVIDER").unwrap_or_else(|_| "anthropic".into());
    match name.as_str() {
        "anthropic" => {
            let api_key = std::env::var("CLAUDE_API_KEY")
                .context("CLAUDE_API_KEY environment variable is not set")?;
            let mut config = AnthropicConfig::new(api_key);
            if let Some(model) = model {
                config.model = model;
            }
            Ok(Arc::new(AnthropicProvider::new(config)))
        }
        #[cfg(feature = "copilot")]
        "copilot" => Ok(Arc::new(crate::CopilotProvider::new(
            crate::CopilotConfig {
                model,
                ..Default::default()
            },
        ))),
        #[cfg(not(feature = "copilot"))]
        "copilot" => bail!("alx was built without the `copilot` feature"),
        other => bail!("unknown ALX_PROVIDER `{other}` (expected `anthropic` or `copilot`)"),
    }
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
