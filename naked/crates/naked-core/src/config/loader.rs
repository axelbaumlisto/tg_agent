//! Config loading — discover, parse, env-override.

use std::path::{Path, PathBuf};

use crate::error::{AgentError, Result};

use super::session::expand_tilde;
use super::{Config, dirs_home};

pub type ConfigSourceBytes = Option<(PathBuf, Vec<u8>)>;

impl Config {
    pub fn load() -> Result<Self> {
        Ok(Self::load_with_source_bytes()?.0)
    }

    /// Load config plus the exact JSON file bytes that were parsed, when a
    /// file exists. The byte snapshot is for boot observability only: callers
    /// can hash the deterministic raw file bytes without re-serializing
    /// `Config` (which contains maps).
    pub fn load_with_source_bytes() -> Result<(Self, ConfigSourceBytes)> {
        Self::load_dotenv();

        let (mut cfg, source) = if let Ok(path) = std::env::var("NAKED_CONFIG") {
            let path = PathBuf::from(path);
            let (cfg, bytes) = Self::from_json_file_with_bytes(&path)?;
            (cfg, Some((path, bytes)))
        } else if let Some(path) = Self::discover_json_path() {
            let (cfg, bytes) = Self::from_json_file_with_bytes(&path)?;
            (cfg, Some((path, bytes)))
        } else {
            (Self::default(), None)
        };

        cfg.apply_env_overrides();
        cfg.resolve_mcp_names();

        cfg.skill_roots = cfg
            .skill_roots
            .into_iter()
            .map(|p| expand_tilde(&p))
            .collect();

        if cfg.skill_roots.is_empty() {
            let home = dirs_home();
            cfg.skill_roots = vec![home.join(".naked/skills"), home.join(".agents/skills")];
        }

        cfg.agent_dirs = cfg
            .agent_dirs
            .into_iter()
            .map(|p| expand_tilde(&p))
            .collect();

        if cfg.agent_dirs.is_empty() {
            let home = dirs_home();
            cfg.agent_dirs = vec![PathBuf::from("./agents"), home.join(".naked/agents")];
        }

        for persona in cfg.chat_personas.values_mut() {
            persona.workspace = expand_tilde(&persona.workspace);
        }

        cfg.validate_and_warn();

        Ok((cfg, source))
    }

    /// Load from a specific JSON file.
    pub fn from_json_file(path: &Path) -> Result<Self> {
        Ok(Self::from_json_file_with_bytes(path)?.0)
    }

    fn from_json_file_with_bytes(path: &Path) -> Result<(Self, Vec<u8>)> {
        if !path.exists() {
            return Err(AgentError::Config(format!(
                "config file not found: {}",
                path.display()
            )));
        }
        let bytes = std::fs::read(path)?;
        let data = std::str::from_utf8(&bytes).map_err(|e| {
            AgentError::ConfigParse(format!("{} is not valid UTF-8 JSON: {e}", path.display()))
        })?;
        Ok((Self::from_json_str(data)?, bytes))
    }

    /// Parse from JSON string.
    ///
    /// B46 / PLAN_PROVIDER_HEALTH_v1: before deserializing, prune each
    /// `providers.*.api_keys[]` array of entries present in any
    /// `_dead_api_keys_*` / `_low_balance_keys_*` sibling array. This
    /// runs unconditionally (no env gate) — if an operator-curated dead
    /// list says a key is dead, we trust it.
    pub fn from_json_str(json: &str) -> Result<Self> {
        let filtered =
            filter_dead_keys_from_json(json).map_err(|e| AgentError::ConfigParse(e.to_string()))?;
        serde_json::from_str(&filtered).map_err(|e| AgentError::ConfigParse(e.to_string()))
    }

    /// Search standard locations for config JSON.
    /// Walks up parent directories (like git) to find config.
    fn discover_json_path() -> Option<PathBuf> {
        let cwd = std::env::current_dir().unwrap_or_else(|_| PathBuf::from("."));

        let mut dir = cwd.as_path();
        loop {
            let names = [".naked/config.json", "naked.json"];
            for name in &names {
                let path = dir.join(name);
                if path.exists() {
                    tracing::info!("Loading config from {}", path.display());
                    return Some(path);
                }
            }
            match dir.parent() {
                Some(p) => dir = p,
                None => break,
            }
        }

        let home = dirs_home().join(".naked/config.json");
        if home.exists() {
            tracing::info!("Loading config from {}", home.display());
            return Some(home);
        }

        None
    }

    /// Walk up from cwd to find `.env`, like `discover_json`.
    fn load_dotenv() {
        let cwd = std::env::current_dir().unwrap_or_else(|_| PathBuf::from("."));
        let mut dir = cwd.as_path();
        loop {
            let path = dir.join(".env");
            if path.exists() {
                let _ = dotenvy::from_path(&path);
                return;
            }
            match dir.parent() {
                Some(p) => dir = p,
                None => break,
            }
        }
        let _ = dotenvy::dotenv();
    }

    /// Env vars override JSON values (for CI, Docker, etc).
    fn apply_env_overrides(&mut self) {
        if let Ok(v) = std::env::var("NAKED_PROVIDER") {
            self.default_provider = v;
        }
        if let Ok(v) = std::env::var("NAKED_MODEL") {
            self.default_model = v;
        }
        if let Ok(v) = std::env::var("NAKED_WORKSPACE") {
            self.workspace = PathBuf::from(v);
        }
        if let Ok(v) = std::env::var("NAKED_MAX_ITERATIONS")
            && let Ok(n) = v.parse()
        {
            self.max_iterations = n;
        }
        if let Ok(v) = std::env::var("NAKED_MAX_TOKENS")
            && let Ok(n) = v.parse()
        {
            self.max_tokens = n;
        }
        if let Ok(v) = std::env::var("NAKED_TEMPERATURE")
            && let Ok(n) = v.parse()
        {
            self.temperature = Some(n);
        }
        if let Ok(v) = std::env::var("NAKED_TOOL_TIMEOUT")
            && let Ok(n) = v.parse()
        {
            self.tool_timeout_secs = n;
        }
        if let Ok(v) = std::env::var("NAKED_SESSION_DIR") {
            self.session_dir = PathBuf::from(v);
        }
        if let Ok(v) = std::env::var("NAKED_TELEGRAM_TOKEN") {
            self.telegram.telegram_bot_token = Some(v);
        }
        if let Ok(v) = std::env::var("NAKED_ALLOWED_CHAT_IDS") {
            self.telegram.allowed_chat_ids = v
                .split(',')
                .filter_map(|s| s.trim().parse::<i64>().ok())
                .collect();
        }
        if self.exa_api_keys.is_empty() {
            if let Ok(v) = std::env::var("EXA_API_KEYS") {
                self.exa_api_keys = v
                    .split(',')
                    .map(|s| s.trim().to_string())
                    .filter(|s| !s.is_empty())
                    .collect();
            } else if let Ok(v) = std::env::var("EXA_API_KEY")
                && !v.is_empty()
            {
                self.exa_api_keys = vec![v];
            }
        }
    }
}

/// B46 boot-time filter: walk `providers.<name>`, gather all `_dead_api_keys_*`
/// and `_low_balance_keys_*` sibling arrays, then remove any matching entries
/// from `api_keys[]`. Primary `api_key` is left untouched (operator's call).
///
/// Returns the modified JSON (pretty-printed for stable diff) or the original
/// string unchanged if the input has no `providers` object.
pub fn filter_dead_keys_from_json(json: &str) -> std::result::Result<String, String> {
    let mut value: serde_json::Value =
        serde_json::from_str(json).map_err(|e| format!("parse: {e}"))?;

    let providers = match value.get_mut("providers").and_then(|v| v.as_object_mut()) {
        Some(p) => p,
        None => return Ok(json.to_string()),
    };

    let mut total_removed = 0_usize;
    for (provider_name, pcfg) in providers.iter_mut() {
        let Some(obj) = pcfg.as_object_mut() else {
            continue;
        };

        // 1. Collect dead-keys from all `_dead_api_keys*` / `_low_balance_keys*`
        //    sibling arrays.
        let mut dead: std::collections::HashSet<String> = std::collections::HashSet::new();
        for (key, val) in obj.iter() {
            if (key.starts_with("_dead_api_keys") || key.starts_with("_low_balance_keys"))
                && let Some(arr) = val.as_array()
            {
                for v in arr {
                    if let Some(s) = v.as_str() {
                        dead.insert(s.to_string());
                    }
                }
            }
        }
        if dead.is_empty() {
            continue;
        }

        // 2. Prune api_keys[] of any entry present in `dead`.
        if let Some(rotation) = obj.get_mut("api_keys").and_then(|v| v.as_array_mut()) {
            let before = rotation.len();
            rotation.retain(|v| match v.as_str() {
                Some(s) => !dead.contains(s),
                None => true,
            });
            let removed = before - rotation.len();
            if removed > 0 {
                total_removed += removed;
                tracing::info!(
                    target: "naked_core::config::loader",
                    provider = %provider_name,
                    removed,
                    dead_buckets = dead.len(),
                    "B46: dropped dead keys from rotation during config load"
                );
            }
        }
    }

    if total_removed == 0 {
        // Skip re-serialise to keep byte-identical output for non-dead configs.
        return Ok(json.to_string());
    }

    serde_json::to_string_pretty(&value).map_err(|e| format!("serialise: {e}"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn from_json_str_minimal() {
        let config = Config::from_json_str("{}").unwrap();
        assert!(config.providers.is_empty());
        assert!(config.default_provider.is_empty());
    }

    #[test]
    fn from_json_str_with_provider() {
        let json = r#"{
            "providers": {
                "test": {
                    "type": "openai",
                    "api_key": "sk-test",
                    "models": ["model-1"]
                }
            },
            "default_provider": "test"
        }"#;
        let config = Config::from_json_str(json).unwrap();
        assert_eq!(config.default_provider, "test");
        assert!(config.providers.contains_key("test"));
    }

    #[test]
    fn from_json_str_invalid_json_is_error() {
        assert!(Config::from_json_str("not json").is_err());
    }

    // ─── B46 filter_dead_keys_from_json tests ──────────────────

    #[test]
    fn b46_filter_drops_keys_present_in_dead_bucket() {
        let json = r#"{
            "providers": {
                "deepseek": {
                    "type": "openai_compat",
                    "api_key": "sk-alive",
                    "api_keys": ["sk-alive", "sk-dead-1", "sk-dead-2", "sk-other"],
                    "_dead_api_keys_auto_2026-05-13": ["sk-dead-1", "sk-dead-2"]
                }
            }
        }"#;
        let cfg = Config::from_json_str(json).unwrap();
        let ds = cfg.providers.get("deepseek").unwrap();
        assert_eq!(ds.api_keys, vec!["sk-alive", "sk-other"]);
        assert_eq!(ds.api_key, "sk-alive");
    }

    #[test]
    fn b46_filter_handles_low_balance_bucket_too() {
        let json = r#"{
            "providers": {
                "deepseek": {
                    "type": "openai_compat",
                    "api_key": "sk-primary",
                    "api_keys": ["sk-paused-1", "sk-active"],
                    "_low_balance_keys_2026_05_13": ["sk-paused-1"]
                }
            }
        }"#;
        let cfg = Config::from_json_str(json).unwrap();
        let ds = cfg.providers.get("deepseek").unwrap();
        assert_eq!(ds.api_keys, vec!["sk-active"]);
    }

    #[test]
    fn b46_filter_keeps_primary_intact_even_if_in_dead_list() {
        // Primary marked dead in the bucket but we still keep it in api_key
        // (operator's call to rotate primary; auto-persist never touches it).
        let json = r#"{
            "providers": {
                "deepseek": {
                    "type": "openai_compat",
                    "api_key": "sk-PRIMARY-dead",
                    "api_keys": ["sk-alive"],
                    "_dead_api_keys_auto_2026-05-13": ["sk-PRIMARY-dead"]
                }
            }
        }"#;
        let cfg = Config::from_json_str(json).unwrap();
        let ds = cfg.providers.get("deepseek").unwrap();
        assert_eq!(ds.api_key, "sk-PRIMARY-dead");
        assert_eq!(ds.api_keys, vec!["sk-alive"]);
    }

    #[test]
    fn b46_filter_no_op_when_no_dead_buckets() {
        let json =
            r#"{"providers":{"foo":{"type":"openai_compat","api_key":"sk","api_keys":["a","b"]}}}"#;
        let out = filter_dead_keys_from_json(json).unwrap();
        assert_eq!(out, json, "no-dead configs returned byte-identical");
    }

    #[test]
    fn b46_filter_handles_missing_providers_object() {
        let json = r#"{"unrelated": "value"}"#;
        let out = filter_dead_keys_from_json(json).unwrap();
        assert_eq!(out, json);
    }

    #[test]
    fn from_json_file_nonexistent_is_error() {
        let result =
            Config::from_json_file(std::path::Path::new("/tmp/nonexistent_naked_test.json"));
        assert!(result.is_err());
    }
}
