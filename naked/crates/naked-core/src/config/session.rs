//! Session configuration.

use super::AgentError;
use super::Config;
use super::McpServerConfig;
use super::dirs_home;
use crate::error::Result;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::{Path, PathBuf};

/// YOLO auto-approve grant window (30 days, in seconds). Single source of
/// truth in naked-core; `set_session_yolo` stamps this into the persisted
/// `SessionConfig.yolo_ttl_secs` so restarts judge the grant against the
/// window it was created under (not whatever the current default happens to
/// be). naked-tg's `channel_map` keeps its own constant (separate crate).
pub const YOLO_TTL_SECS: i64 = 30 * 24 * 3600;

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
    /// Unix timestamp (secs) when yolo was enabled. Expiration is governed by
    /// yolo_ttl_secs (new grants default 30 days; legacy grants without a TTL
    /// marker use the 72h cutoff on restore).
    #[serde(default)]
    pub yolo_enabled_at: Option<i64>,
    /// TTL window (secs) for this grant; `None` = legacy pre-30d grant judged
    /// against the 72h cutoff on restore. Absent in old configs (serde default
    /// = None) so they keep the conservative legacy admission.
    #[serde(default)]
    pub yolo_ttl_secs: Option<i64>,
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
    /// B105: resolve the session's provider/model pin, rewriting a retired
    /// provider through `provider_aliases`.
    ///
    /// Split out of `merge_session` so that function stays a plain field-merge:
    /// "which provider/model is this session really on" is its own decision,
    /// with its own alias/logging rules, and is what the B105 tests target.
    fn effective_provider_model(&self, session: &SessionConfig) -> (String, String) {
        let pinned_provider = session
            .default_provider
            .clone()
            .unwrap_or_else(|| self.default_provider.clone());
        let pinned_model = session
            .default_model
            .clone()
            .unwrap_or_else(|| self.default_model.clone());

        let (provider, alias_model) = self.resolve_provider_alias(&pinned_provider);
        if provider == pinned_provider {
            return (provider, pinned_model);
        }

        // An alias-pinned model replaces the session's own only when the alias
        // actually redirected — a retired model must not survive the hop.
        let model = alias_model.unwrap_or(pinned_model);
        tracing::info!(
            from = %pinned_provider,
            to = %provider,
            %model,
            "provider alias applied (B105)"
        );
        (provider, model)
    }

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

        // B105: a session's provider pin outlives the removal of that
        // provider from `providers`. Rewrite retired names through
        // `provider_aliases` HERE, at the single point where a pin becomes an
        // effective pair, so provider resolution, health records, tracing
        // spans and `/model` all agree instead of a silent default-fallback
        // that kept reporting the dead pair.
        let (provider, model) = self.effective_provider_model(session);

        EffectiveSessionConfig {
            provider,
            model,
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

    // ── D-INV-PROVIDER-ALIAS (B105) ──────────────────────────────────
    //
    // A session pin outlives the removal of its provider. Aliases must
    // rewrite the retired name at merge time so provider resolution, health
    // records, spans and `/model` all report the SAME live pair. Removing
    // the alias application in `merge_session` fails these tests.

    fn alias_cfg(aliases: &[(&str, &str)], live: &[&str]) -> Config {
        let mut c = Config {
            default_provider: "anthropic".into(),
            default_model: "claude-sonnet-4-6".into(),
            ..Config::default()
        };
        for (from, to) in aliases {
            c.provider_aliases
                .insert((*from).to_string(), (*to).to_string());
        }
        for name in live {
            c.providers.insert(
                (*name).to_string(),
                crate::config::ProviderConfig {
                    api_key: "k".into(),
                    models: vec!["m1".into()],
                    ..Default::default()
                },
            );
        }
        c
    }

    fn pinned(provider: &str, model: &str) -> SessionConfig {
        SessionConfig {
            default_provider: Some(provider.to_string()),
            default_model: Some(model.to_string()),
            ..Default::default()
        }
    }

    /// The production case: 229 sessions pinned to the retired `qwen`.
    #[test]
    fn b105_retired_provider_pin_is_rewritten_to_live_pair() {
        let cfg = alias_cfg(
            &[("qwen", "anthropic/claude-sonnet-4-6")],
            &["anthropic", "deepseek"],
        );
        let eff = cfg.merge_session(&pinned("qwen", "qwen3.6-plus"));
        assert_eq!(eff.provider, "anthropic");
        assert_eq!(
            eff.model, "claude-sonnet-4-6",
            "the dead model must not survive the redirect"
        );
    }

    /// Restoring a provider must beat its own alias with no code change.
    #[test]
    fn b105_live_provider_beats_its_alias() {
        let cfg = alias_cfg(
            &[("qwen", "anthropic/claude-sonnet-4-6")],
            &["anthropic", "qwen"], // qwen restored
        );
        let eff = cfg.merge_session(&pinned("qwen", "qwen3.6-plus"));
        assert_eq!(eff.provider, "qwen");
        assert_eq!(
            eff.model, "qwen3.6-plus",
            "restored provider keeps its model"
        );
    }

    /// Bare `provider` form redirects but preserves the session's model.
    #[test]
    fn b105_bare_alias_keeps_session_model() {
        let cfg = alias_cfg(
            &[("deepseek-direct", "deepseek")],
            &["anthropic", "deepseek"],
        );
        let eff = cfg.merge_session(&pinned("deepseek-direct", "deepseek-v4-pro"));
        assert_eq!(eff.provider, "deepseek");
        assert_eq!(eff.model, "deepseek-v4-pro");
    }

    /// Chains are followed; the FIRST pinned model wins (closest to intent).
    #[test]
    fn b105_alias_chain_is_followed_first_model_wins() {
        let cfg = alias_cfg(
            &[
                ("ali_cp", "qwen/qwen-plus"),
                ("qwen", "anthropic/claude-sonnet-4-6"),
            ],
            &["anthropic"],
        );
        let eff = cfg.merge_session(&pinned("ali_cp", "qwen3.6-plus"));
        assert_eq!(eff.provider, "anthropic");
        assert_eq!(eff.model, "qwen-plus");
    }

    /// A cycle must terminate, not hang or overflow.
    #[test]
    fn b105_alias_cycle_terminates() {
        let cfg = alias_cfg(&[("a", "b"), ("b", "c"), ("c", "a")], &["anthropic"]);
        let (p, _) = cfg.resolve_provider_alias("a");
        assert!(!p.is_empty(), "cycle must return a name, not loop");
    }

    /// An alias pointing at a still-missing provider is a bounded no-op:
    /// the caller keeps its default-fallback behaviour and the WARN names
    /// the end of the broken chain.
    #[test]
    fn b105_alias_to_missing_provider_is_bounded_noop() {
        let cfg = alias_cfg(&[("qwen", "also_gone")], &["anthropic"]);
        let (p, _) = cfg.resolve_provider_alias("qwen");
        assert_eq!(p, "also_gone");
        assert!(!cfg.providers.contains_key(&p));
    }

    /// No aliases configured = byte-identical legacy behaviour.
    #[test]
    fn b105_no_aliases_is_passthrough() {
        let cfg = alias_cfg(&[], &["anthropic", "deepseek"]);
        let eff = cfg.merge_session(&pinned("qwen", "qwen3.6-plus"));
        assert_eq!(eff.provider, "qwen", "absent aliases must not rewrite");
        assert_eq!(eff.model, "qwen3.6-plus");
    }
    /// Malformed alias targets must be bounded no-ops, not hangs or jumps to a
    /// blank provider. Extracted as its own function during the DRY/KISS pass,
    /// so pin the edge shapes directly.
    #[test]
    fn b105_alias_target_parsing_rejects_provider_less_shapes() {
        use crate::config::parse_alias_target;
        assert_eq!(
            parse_alias_target("live/model"),
            Some(("live".into(), Some("model".into())))
        );
        assert_eq!(parse_alias_target("live"), Some(("live".into(), None)));
        assert_eq!(
            parse_alias_target("live/"),
            Some(("live".into(), None)),
            "trailing slash keeps the provider, drops the empty model"
        );
        assert_eq!(parse_alias_target("/model"), None, "no provider named");
        assert_eq!(parse_alias_target(""), None);
    }
}
