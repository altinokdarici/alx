//! Building an agent from [`Settings`], plus the `.env` loader. Kept in the core rather
//! than in a binary so every client is wired the same way.

use crate::{
    AgentConfig, AnthropicConfig, AnthropicProvider, BashTool, LocalExecutor, Provider,
    ProviderKind, Settings, SingleAgent, ToolContext, ToolRegistry,
};
use anyhow::Context;
use std::sync::Arc;

/// Default agent: the provider from [`provider_from_settings`], bash tool,
/// local executor.
///
/// Swap any part here: another `Provider`, `Executor` (Docker, remote, ...),
/// more tools, or a different `AgentLoop` entirely.
pub fn agent_from_settings(settings: &Settings) -> anyhow::Result<SingleAgent> {
    let mut config = AgentConfig::default();
    if let Ok(instructions) = std::fs::read_to_string("AGENTS.md") {
        config.system_prompt = with_instructions(&config.system_prompt, &instructions);
    }
    Ok(SingleAgent {
        config,
        provider: provider_from_settings(settings)?,
        tools: ToolRegistry::new().with(BashTool),
        tool_ctx: ToolContext {
            executor: Arc::new(LocalExecutor),
        },
    })
}

/// Appends project instructions (the contents of `./AGENTS.md`) to the system
/// prompt, fenced so the model can tell them from alx's own prompt.
fn with_instructions(system_prompt: &str, instructions: &str) -> String {
    format!(
        "{system_prompt}\n\n<project_instructions source=\"AGENTS.md\">\n{}\n</project_instructions>",
        instructions.trim()
    )
}

/// The LLM backend named by `settings.provider` (default `anthropic`).
pub fn provider_from_settings(settings: &Settings) -> anyhow::Result<Arc<dyn Provider>> {
    match settings.provider_kind() {
        ProviderKind::Anthropic => {
            let api_key = settings
                .anthropic
                .api_key
                .clone()
                .context("CLAUDE_API_KEY environment variable is not set")?;
            let mut config = AnthropicConfig::new(api_key);
            if let Some(model) = &settings.model {
                config.model = model.clone();
            }
            if let Some(url) = &settings.anthropic.base_url {
                config.base_url = url.clone();
            }
            Ok(Arc::new(AnthropicProvider::new(config)))
        }
        #[cfg(feature = "copilot")]
        ProviderKind::Copilot => Ok(Arc::new(crate::CopilotProvider::new(
            crate::CopilotConfig {
                model: settings.model.clone(),
                cli_path: settings.copilot.cli_path.clone(),
                copilot_home: settings.copilot.home.clone(),
                ..Default::default()
            },
        ))),
        #[cfg(not(feature = "copilot"))]
        ProviderKind::Copilot => anyhow::bail!("alx was built without the `copilot` feature"),
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
    fn appends_instructions_to_the_system_prompt() {
        let prompt = with_instructions("base", "\nbe terse\n");
        assert!(prompt.starts_with("base\n\n<project_instructions"));
        assert!(prompt.contains("\nbe terse\n</project_instructions>"));
    }

    #[test]
    fn parses_dotenv_lines() {
        let parsed: Vec<_> =
            parse_dotenv("# comment\n\nA=1\nexport B=\"two\"\n C = 'three' \nnoequals\n").collect();
        assert_eq!(parsed, [("A", "1"), ("B", "two"), ("C", "three")]);
    }
}
