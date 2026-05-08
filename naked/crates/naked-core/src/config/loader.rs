//! Config loading — discover, parse, env-override.

use std::path::{Path, PathBuf};

use crate::error::{AgentError, Result};

use super::session::expand_tilde;
use super::{Config, dirs_home};

impl Config {
    pub fn load() -> Result<Self> {
        Self::load_dotenv();

        let mut cfg = if let Ok(path) = std::env::var("NAKED_CONFIG") {
            Self::from_json_file(Path::new(&path))?
        } else {
            Self::discover_json()?
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

        Ok(cfg)
    }

    /// Load from a specific JSON file.
    pub fn from_json_file(path: &Path) -> Result<Self> {
        if !path.exists() {
            return Err(AgentError::Config(format!(
                "config file not found: {}",
                path.display()
            )));
        }
        let data = std::fs::read_to_string(path)?;
        Self::from_json_str(&data)
    }

    /// Parse from JSON string.
    pub fn from_json_str(json: &str) -> Result<Self> {
        serde_json::from_str(json).map_err(|e| AgentError::ConfigParse(e.to_string()))
    }

    /// Search standard locations for config JSON.
    /// Walks up parent directories (like git) to find config.
    fn discover_json() -> Result<Self> {
        let cwd = std::env::current_dir().unwrap_or_else(|_| PathBuf::from("."));

        let mut dir = cwd.as_path();
        loop {
            let names = [".naked/config.json", "naked.json"];
            for name in &names {
                let path = dir.join(name);
                if path.exists() {
                    tracing::info!("Loading config from {}", path.display());
                    return Self::from_json_file(&path);
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
            return Self::from_json_file(&home);
        }

        Ok(Self::default())
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

    #[test]
    fn from_json_file_nonexistent_is_error() {
        let result =
            Config::from_json_file(std::path::Path::new("/tmp/nonexistent_naked_test.json"));
        assert!(result.is_err());
    }
}
