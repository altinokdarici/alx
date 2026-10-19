//! Settings: one typed schema, filled from several layers.
//!
//! Lowest to highest precedence (a later layer overrides an earlier one, field
//! by field):
//!
//! 1. user file: `$XDG_CONFIG_HOME/alx/config.toml` (or `~/.config/alx/config.toml`)
//! 2. project file: `./alx.toml`
//! 3. environment variables (see [`Settings::from_env_with`])
//! 4. command-line flags, built by the client and merged last
//!
//! Every field is optional, so a layer only says what it wants to change.
//! Secrets (the Anthropic API key) come from the environment only; they are
//! rejected in files.

use anyhow::{Context, bail};
use serde::Deserialize;
use std::fmt;
use std::path::{Path, PathBuf};
use std::str::FromStr;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ProviderKind {
    Anthropic,
    Copilot,
}

impl FromStr for ProviderKind {
    type Err = anyhow::Error;

    fn from_str(s: &str) -> anyhow::Result<Self> {
        match s {
            "anthropic" => Ok(Self::Anthropic),
            "copilot" => Ok(Self::Copilot),
            other => bail!("unknown provider `{other}` (expected `anthropic` or `copilot`)"),
        }
    }
}

impl fmt::Display for ProviderKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::Anthropic => "anthropic",
            Self::Copilot => "copilot",
        })
    }
}

#[derive(Debug, Clone, Default, PartialEq, Eq, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Settings {
    /// Which LLM backend to use. Defaults to `anthropic`.
    pub provider: Option<ProviderKind>,
    /// Model id, for whichever provider is selected.
    pub model: Option<String>,
    /// Where sessions are stored. Defaults to `~/.local/share/alx/sessions`.
    pub session_dir: Option<PathBuf>,
    #[serde(default)]
    pub anthropic: AnthropicSettings,
    #[serde(default)]
    pub copilot: CopilotSettings,
}

#[derive(Clone, Default, PartialEq, Eq, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AnthropicSettings {
    /// API base URL, for proxies or gateways.
    pub base_url: Option<String>,
    /// From `CLAUDE_API_KEY` only; never read from a file.
    #[serde(skip)]
    pub api_key: Option<String>,
}

// Not derived, so the key can never end up in logs.
impl fmt::Debug for AnthropicSettings {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("AnthropicSettings")
            .field("base_url", &self.base_url)
            .field("api_key", &self.api_key.as_ref().map(|_| "<redacted>"))
            .finish()
    }
}

#[derive(Debug, Clone, Default, PartialEq, Eq, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CopilotSettings {
    /// The `copilot` CLI to run. Defaults to `copilot` on `PATH`.
    pub cli_path: Option<PathBuf>,
    /// The CLI's state directory. Defaults to `~/.copilot`.
    pub home: Option<PathBuf>,
}

impl Settings {
    /// Resolves the file and environment layers. Command-line flags are not
    /// included: build a `Settings` from them and [`merge`](Self::merge) it
    /// on top.
    ///
    /// `config` replaces the project file when given, and must exist.
    pub fn load(config: Option<&Path>) -> anyhow::Result<Self> {
        let mut settings = Settings::default();
        if let Some(path) = user_config_path() {
            settings = settings.merge(Self::from_file(&path)?.unwrap_or_default());
        }
        settings = match config {
            Some(path) => settings.merge(
                Self::from_file(path)?
                    .with_context(|| format!("config file {} not found", path.display()))?,
            ),
            None => settings.merge(Self::from_file(Path::new("alx.toml"))?.unwrap_or_default()),
        };
        Ok(settings.merge(Self::from_env_with(|k| std::env::var(k).ok())))
    }

    /// Reads one file. `Ok(None)` if it does not exist.
    pub fn from_file(path: &Path) -> anyhow::Result<Option<Self>> {
        let text = match std::fs::read_to_string(path) {
            Ok(text) => text,
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(None),
            Err(e) => return Err(e).with_context(|| format!("reading {}", path.display())),
        };
        Self::parse(&text)
            .map(Some)
            .with_context(|| format!("parsing {}", path.display()))
    }

    pub fn parse(text: &str) -> anyhow::Result<Self> {
        Ok(toml::from_str(text)?)
    }

    /// The environment layer, with the lookup injected so tests need no real
    /// environment.
    ///
    /// | Variable                 | Setting              |
    /// |--------------------------|----------------------|
    /// | `ALX_PROVIDER`           | `provider`           |
    /// | `ALX_MODEL`              | `model`              |
    /// | `ALX_SESSION_DIR`        | `session_dir`        |
    /// | `ALX_ANTHROPIC_BASE_URL` | `anthropic.base_url` |
    /// | `CLAUDE_API_KEY`         | `anthropic.api_key`  |
    /// | `COPILOT_CLI_PATH`       | `copilot.cli_path`   |
    /// | `COPILOT_HOME`           | `copilot.home`       |
    ///
    /// An invalid `ALX_PROVIDER` is ignored here and reported when the
    /// provider is built, so one bad variable doesn't hide a good flag.
    pub fn from_env_with(get: impl Fn(&str) -> Option<String>) -> Self {
        Self {
            provider: get("ALX_PROVIDER").and_then(|v| v.parse().ok()),
            model: get("ALX_MODEL"),
            session_dir: get("ALX_SESSION_DIR").map(PathBuf::from),
            anthropic: AnthropicSettings {
                base_url: get("ALX_ANTHROPIC_BASE_URL"),
                api_key: get("CLAUDE_API_KEY"),
            },
            copilot: CopilotSettings {
                cli_path: get("COPILOT_CLI_PATH").map(PathBuf::from),
                home: get("COPILOT_HOME").map(PathBuf::from),
            },
        }
    }

    /// Layers `other` on top of `self`: every field `other` sets wins.
    #[must_use]
    pub fn merge(self, other: Self) -> Self {
        Self {
            provider: other.provider.or(self.provider),
            model: other.model.or(self.model),
            session_dir: other.session_dir.or(self.session_dir),
            anthropic: AnthropicSettings {
                base_url: other.anthropic.base_url.or(self.anthropic.base_url),
                api_key: other.anthropic.api_key.or(self.anthropic.api_key),
            },
            copilot: CopilotSettings {
                cli_path: other.copilot.cli_path.or(self.copilot.cli_path),
                home: other.copilot.home.or(self.copilot.home),
            },
        }
    }

    /// The directory sessions are stored in, with the default applied.
    pub fn session_dir(&self) -> anyhow::Result<PathBuf> {
        self.session_dir
            .clone()
            .or_else(crate::session::default_session_dir)
            .context("cannot locate a session directory: set `session_dir` or HOME")
    }

    /// The provider to use, with the default applied.
    pub fn provider_kind(&self) -> ProviderKind {
        self.provider.unwrap_or(ProviderKind::Anthropic)
    }
}

fn user_config_path() -> Option<PathBuf> {
    let base = match std::env::var_os("XDG_CONFIG_HOME").filter(|v| !v.is_empty()) {
        Some(dir) => PathBuf::from(dir),
        None => PathBuf::from(std::env::var_os("HOME")?).join(".config"),
    };
    Some(base.join("alx").join("config.toml"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_a_full_file() {
        let s = Settings::parse(
            "provider = \"copilot\"\nmodel = \"gpt-5\"\n[anthropic]\nbase_url = \"http://x\"\n[copilot]\ncli_path = \"/bin/copilot\"\nhome = \"/h\"\n",
        )
        .unwrap();
        assert_eq!(s.provider, Some(ProviderKind::Copilot));
        assert_eq!(s.model.as_deref(), Some("gpt-5"));
        assert_eq!(s.anthropic.base_url.as_deref(), Some("http://x"));
        assert_eq!(s.copilot.cli_path, Some("/bin/copilot".into()));
        assert_eq!(s.copilot.home, Some("/h".into()));
    }

    #[test]
    fn rejects_unknown_keys_and_file_secrets() {
        assert!(Settings::parse("modle = \"x\"").is_err());
        assert!(Settings::parse("[anthropic]\napi_key = \"sk-secret\"").is_err());
        assert!(Settings::parse("provider = \"nope\"").is_err());
    }

    #[test]
    fn later_layers_override_field_by_field() {
        let user = Settings::parse("provider = \"anthropic\"\nmodel = \"user\"").unwrap();
        let project = Settings::parse("model = \"project\"\n[copilot]\nhome = \"/p\"").unwrap();
        let env = Settings::from_env_with(|k| (k == "ALX_PROVIDER").then(|| "copilot".into()));
        let cli = Settings {
            model: Some("cli".into()),
            ..Default::default()
        };
        let merged = user.merge(project).merge(env).merge(cli);
        assert_eq!(merged.provider, Some(ProviderKind::Copilot)); // env beat user
        assert_eq!(merged.model.as_deref(), Some("cli")); // cli beat project and user
        assert_eq!(merged.copilot.home, Some("/p".into())); // untouched lower layer survives
    }

    #[test]
    fn env_layer_maps_variables() {
        let s = Settings::from_env_with(|k| match k {
            "ALX_MODEL" => Some("m".into()),
            "ALX_SESSION_DIR" => Some("/s".into()),
            "CLAUDE_API_KEY" => Some("key".into()),
            "COPILOT_HOME" => Some("/c".into()),
            _ => None,
        });
        assert_eq!(s.model.as_deref(), Some("m"));
        assert_eq!(s.session_dir, Some("/s".into()));
        assert_eq!(s.anthropic.api_key.as_deref(), Some("key"));
        assert_eq!(s.copilot.home, Some("/c".into()));
        assert_eq!(s.provider_kind(), ProviderKind::Anthropic);
        assert!(!format!("{s:?}").contains("key\""));
    }

    #[test]
    fn missing_file_is_none() {
        assert!(
            Settings::from_file(Path::new("/definitely/not/here.toml"))
                .unwrap()
                .is_none()
        );
    }
}
