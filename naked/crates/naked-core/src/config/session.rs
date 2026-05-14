//! Session configuration.

use super::AgentError;
use super::Config;
use super::McpServerConfig;
use super::dirs_home;
use crate::error::Result;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::{Path, PathBuf};

/// Per-session config override. All fields are optional — missing fields
/// fall back to the global `Config`. Placed in `sessions/{id}/config.json`.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct SessionConfig {
    #[serde(default)]
    pub default_provider: Option<String>,
    #[serde(default)]
    pub default_model: Option<String>,
    #[serde(default)]
    pub max_tokens: Option<u32>,
    #[serde(default)]
    pub temperature: Option<f32>,
    #[serde(default)]
    pub context_window: Option<u32>,
    #[serde(default)]
    pub max_iterations: Option<usize>,
    /// Reasoning/thinking level: "off", "low", "medium", "high"
    #[serde(default)]
    pub reasoning: Option<String>,
    #[serde(default, rename = "mcpServers")]
    pub mcp_servers: Option<HashMap<String, McpServerConfig>>,
    #[serde(default)]
    pub skill_roots: Option<Vec<PathBuf>>,
    #[serde(default)]
    pub system_prompt_path: Option<PathBuf>,
    /// Unix timestamp (secs) when yolo was enabled. Expires after 72h.
    #[serde(default)]
    pub yolo_enabled_at: Option<i64>,
    /// Per-session tool allow-list (persisted across restarts).
    #[serde(default)]
    pub allow_list: Option<Vec<String>>,
}

impl SessionConfig {
    pub fn from_file(path: &Path) -> Result<Self> {
        let data = std::fs::read_to_string(path)?;
        serde_json::from_str(&data)
            .map_err(|e| AgentError::Config(format!("session config parse error: {e}")))
    }
}

/// Effective config for a single session: global merged with per-session overrides.
#[derive(Debug, Clone)]
pub struct EffectiveSessionConfig {
    pub provider: String,
    pub model: String,
    pub max_tokens: u32,
    pub temperature: Option<f32>,
    /// Context window in tokens (None = default 100k).
    pub context_window: Option<u32>,
    pub max_iterations: usize,
    /// Reasoning/thinking level: "off", "low", "medium", "high"
    pub reasoning: Option<String>,
    pub mcp_servers: HashMap<String, McpServerConfig>,
    pub skill_roots: Vec<PathBuf>,
    pub system_prompt_path: Option<PathBuf>,
}

impl Config {
    /// Merge global config with per-session overrides.
    /// Session values win; missing session fields fall back to global.
    /// MCP servers are additive: session servers are merged on top of global
    /// (session overrides global if same name).
    pub fn merge_session(&self, session: &SessionConfig) -> EffectiveSessionConfig {
        let mut mcp_servers = self.mcp_servers.clone();
        if let Some(extra) = &session.mcp_servers {
            for (name, cfg) in extra {
                mcp_servers.insert(name.clone(), cfg.clone());
            }
        }

        let skill_roots = session
            .skill_roots
            .clone()
            .unwrap_or_else(|| self.skill_roots.clone());

        EffectiveSessionConfig {
            provider: session
                .default_provider
                .clone()
                .unwrap_or_else(|| self.default_provider.clone()),
            model: session
                .default_model
                .clone()
                .unwrap_or_else(|| self.default_model.clone()),
            max_tokens: session.max_tokens.unwrap_or(self.max_tokens),
            temperature: session.temperature.or(self.temperature),
            // B50: do NOT auto-merge global config.context_window here.
            // The global value would silently override per-provider and
            // per-model `capabilities.<model>.context_window` (1M for qwen,
            // deepseek, claude-4-6 etc.) when no session override exists.
            // Resolution now happens in `session_ops::turn::pre_compact`
            // with proper precedence: session > capability > provider > global.
            context_window: session.context_window,
            max_iterations: session.max_iterations.unwrap_or(self.max_iterations),
            // Session reasoning wins; otherwise inherit `Config.default_reasoning`
            // so e.g. the bot launches every chat at the configured global
            // thinking level without the user re-typing `/reasoning medium`.
            reasoning: session
                .reasoning
                .clone()
                .or_else(|| self.default_reasoning.clone()),
            mcp_servers,
            skill_roots,
            system_prompt_path: session
                .system_prompt_path
                .clone()
                .or_else(|| self.system_prompt_path.clone()),
        }
    }

    /// Produce an effective config with no overrides (all global values).
    pub fn default_effective(&self) -> EffectiveSessionConfig {
        self.merge_session(&SessionConfig::default())
    }
}

/// Expand `$VAR` or `${VAR}` references in a string from env.
pub fn expand_env(s: &str) -> Result<String> {
    if let Some(var_name) = s.strip_prefix('$') {
        let var_name = var_name.trim_start_matches('{').trim_end_matches('}');
        std::env::var(var_name).map_err(|_| {
            AgentError::Config(format!(
                "env var '{var_name}' not set (referenced in config)"
            ))
        })
    } else {
        Ok(s.to_string())
    }
}

/// Expand leading `~` or `~/` to the user's home directory.
pub fn expand_tilde(p: &Path) -> PathBuf {
    let s = p.to_string_lossy();
    if s == "~" {
        dirs_home()
    } else if let Some(rest) = s.strip_prefix("~/") {
        dirs_home().join(rest)
    } else {
        p.to_path_buf()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// B50 regression guard: `merge_session` must NOT auto-merge the
    /// global `Config.context_window` into `EffectiveSessionConfig`.
    /// Doing so silently overrides per-provider / per-capability values
    /// that are typically much larger (1M for qwen / deepseek / claude-4-6).
    /// Resolution is delegated to `session_ops::turn::pre_compact` which
    /// applies the proper precedence: session > capability > provider > global.
    #[test]
    fn b50_merge_session_does_not_inherit_global_context_window() {
        use super::SessionConfig;
        let global = Config {
            context_window: Some(200_000),
            ..Config::default()
        };
        let session = SessionConfig::default(); // no override
        let eff = global.merge_session(&session);
        assert_eq!(
            eff.context_window, None,
            "global Config.context_window must NOT leak into EffectiveSessionConfig \
             (B50 — was causing 1M-capable models to compact at 200K)"
        );
    }

    /// Sanity: explicit session override DOES make it through.
    #[test]
    fn b50_session_override_context_window_is_preserved() {
        use super::SessionConfig;
        let global = Config {
            context_window: Some(200_000),
            ..Config::default()
        };
        let session = SessionConfig {
            context_window: Some(500_000),
            ..SessionConfig::default()
        };
        let eff = global.merge_session(&session);
        assert_eq!(eff.context_window, Some(500_000));
    }

    #[test]
    fn expand_tilde_home() {
        let home = super::super::dirs_home();
        assert_eq!(expand_tilde(Path::new("~")), home);
        assert_eq!(expand_tilde(Path::new("~/foo")), home.join("foo"));
    }

    #[test]
    fn expand_tilde_absolute_unchanged() {
        assert_eq!(
            expand_tilde(Path::new("/abs/path")),
            PathBuf::from("/abs/path")
        );
    }

    #[test]
    fn expand_env_plain_string() {
        assert_eq!(expand_env("plain-key").unwrap(), "plain-key");
    }

    #[test]
    fn expand_env_dollar_home() {
        // $HOME is always set in UNIX environments
        let result = expand_env("$HOME");
        assert!(result.is_ok());
        assert!(!result.unwrap().is_empty());
    }

    #[test]
    fn expand_env_missing_is_error() {
        assert!(expand_env("$DEFINITELY_NOT_SET_XYZ_12345").is_err());
    }

    #[test]
    fn session_config_default() {
        let sc = SessionConfig::default();
        assert!(sc.default_provider.is_none());
        assert!(sc.default_model.is_none());
        assert!(sc.reasoning.is_none());
    }
}
