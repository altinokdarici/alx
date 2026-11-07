//! Building an agent from [`Settings`], plus the `.env` loader. Kept in the core rather
//! than in a binary so every client is wired the same way.

use crate::{
    AgentConfig, AgentFactory, AgentLoop, AnthropicConfig, AnthropicProvider, BashTool,
    EditFileTool, GlobTool, GrepTool, ListDirTool, LocalExecutor, Provider, ProviderKind,
    ReadFileTool, Settings, SingleAgent, ToolContext, ToolRegistry, WriteFileTool,
};
use std::sync::Arc;

/// Default agent: the provider from [`provider_from_settings`], the bash,
/// read_file and write_file tools, local executor.
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
        tools: ToolRegistry::new()
            .with(BashTool)
            .with(ReadFileTool)
            .with(WriteFileTool)
            .with(EditFileTool)
            .with(GrepTool)
            .with(GlobTool)
            .with(ListDirTool),
        tool_ctx: ToolContext {
            executor: Arc::new(LocalExecutor),
        },
    })
}

/// Builds [`agent_from_settings`] agents for any provider alx was built
/// with. `settings.provider` is the default; `settings.model` applies to it
/// only, since a model id rarely means anything to another provider.
pub struct SettingsAgentFactory(pub Settings);

impl AgentFactory for SettingsAgentFactory {
    fn providers(&self) -> Vec<String> {
        let default = self.0.provider_kind();
        let mut all = vec![default.to_string()];
        all.extend(
            ProviderKind::ALL
                .iter()
                .filter(|&&k| k != default)
                .map(ToString::to_string),
        );
        all
    }

    fn build(&self, provider: &str, model: Option<&str>) -> anyhow::Result<Arc<dyn AgentLoop>> {
        let kind: ProviderKind = provider.parse()?;
        let mut settings = self.0.clone();
        if let Some(model) = model {
            settings.model = Some(model.to_owned());
        } else if kind != settings.provider_kind() {
            settings.model = None;
        }
        settings.provider = Some(kind);
        Ok(Arc::new(agent_from_settings(&settings)?))
    }
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
            // A missing key fails the first request, not construction.
            let api_key = settings.anthropic.api_key.clone().unwrap_or_default();
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
    fn factory_puts_the_default_provider_first_and_keeps_its_model() {
        let factory = SettingsAgentFactory(Settings {
            provider: Some(ProviderKind::Copilot),
            model: Some("m".into()),
            ..Default::default()
        });
        assert_eq!(factory.providers()[0], "copilot");
        assert!(factory.providers().contains(&"anthropic".to_owned()));
        let info = |p| factory.build(p, None).unwrap().provider_info().unwrap();
        assert_eq!(info("copilot").model.as_deref(), Some("m"));
        let picked = factory.build("anthropic", Some("x")).unwrap();
        assert_eq!(picked.provider_info().unwrap().model.as_deref(), Some("x"));
        // No key needed to build it; the model is the provider's own default.
        assert_eq!(
            info("anthropic").model.as_deref(),
            Some(AnthropicConfig::DEFAULT_MODEL)
        );
        assert!(factory.build("nope", None).is_err());
    }

    #[test]
    fn parses_dotenv_lines() {
        let parsed: Vec<_> =
            parse_dotenv("# comment\n\nA=1\nexport B=\"two\"\n C = 'three' \nnoequals\n").collect();
        assert_eq!(parsed, [("A", "1"), ("B", "two"), ("C", "three")]);
    }
}
