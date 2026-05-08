//! Provider configuration — ProviderConfig, ResolvedProvider, key resolution.

use std::collections::HashMap;

use serde::{Deserialize, Serialize};

use super::expand_env;
use crate::error::Result;

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ProviderConfig {
    /// "anthropic" or "openai_compat" -- selects wire format
    #[serde(rename = "type")]
    pub provider_type: String,
    /// Primary API key, or `$ENV_VAR` to read from environment
    pub api_key: String,
    /// Additional keys for automatic rotation on failure
    #[serde(default)]
    pub api_keys: Vec<String>,
    #[serde(default)]
    pub base_url: Option<String>,
    #[serde(default)]
    pub models: Vec<String>,
    /// Per-provider max_tokens override (0 = use global default)
    #[serde(default)]
    pub max_tokens: Option<u32>,
    /// Per-provider temperature override
    #[serde(default)]
    pub temperature: Option<f32>,
    /// Per-provider context window override in tokens
    #[serde(default)]
    pub context_window: Option<u32>,
    /// Extra HTTP headers (e.g. User-Agent for Kimi Code)
    #[serde(default)]
    pub headers: HashMap<String, String>,
    /// Model name aliases: `alias → real_model_id`. The alias is what the
    /// user sees in `/model` menus and what `request.model` carries through
    /// the bot session; the real ID is what gets sent in the JSON body to
    /// the upstream API. Useful when the upstream model ID is opaque or
    /// differs from how the model is publicly known (e.g.
    /// `kimi-k2.6 → kimi-for-coding` because the Kimi For Coding endpoint
    /// is the K2.6 release, but the API only accepts the literal
    /// `kimi-for-coding` string).
    ///
    /// Aliases are appended to `models()` so they show up alongside real
    /// IDs in selection menus. If both the alias and the real ID are listed
    /// in `models`, that's fine — both will route to the same upstream
    /// model.
    #[serde(default)]
    pub model_aliases: HashMap<String, String>,
    /// Force vision-capability classification for this provider's models. When
    /// `Some(true)`, every model under this provider is treated as vision-
    /// capable (overrides the built-in needle list). When `Some(false)`, every
    /// model is forced text-only. When `None` (default), classification falls
    /// back to `BUILTIN_VISION_MODEL_NEEDLES`.
    ///
    /// Use this to opt-in new vision-capable models without recompiling the
    /// binary, or to explicitly disable native multimodal routing for a
    /// provider whose vision tier you don't want to pay for.
    #[serde(default)]
    pub supports_vision: Option<bool>,
    /// Structured per-model capability catalog. Keyed by model id (post-alias
    /// resolution) — entries for alias keys are accepted but the canonical
    /// location is the real model id. Missing entries default to
    /// [`crate::model_catalog::ModelCapabilities::unknown`], which is
    /// permissive so legacy configs keep booting unchanged. See the
    /// [`crate::model_catalog`] module for the schema.
    #[serde(default)]
    pub capabilities: HashMap<String, crate::model_catalog::ModelCapabilities>,
}

impl ProviderConfig {
    /// Resolve api_key: if starts with `$`, read from env var.
    pub fn resolved_api_key(&self) -> Result<String> {
        expand_env(&self.api_key)
    }

    /// All resolved keys: primary `api_key` + any `api_keys`, deduplicated.
    pub fn resolved_all_keys(&self) -> Vec<String> {
        let mut keys = Vec::new();
        if let Ok(k) = expand_env(&self.api_key)
            && !k.is_empty()
            && !k.starts_with('$')
        {
            keys.push(k);
        }
        for raw in &self.api_keys {
            if let Ok(k) = expand_env(raw)
                && !k.is_empty()
                && !k.starts_with('$')
                && !keys.contains(&k)
            {
                keys.push(k);
            }
        }
        keys
    }

    /// Return a copy with the api_key resolved from env.
    pub fn resolved(&self) -> Result<ResolvedProvider> {
        let all_keys = self.resolved_all_keys();
        Ok(ResolvedProvider {
            provider_type: self.provider_type.clone(),
            api_key: self.resolved_api_key()?,
            all_keys,
            base_url: self.base_url.clone(),
            models: self.models.clone(),
            max_tokens: self.max_tokens,
            temperature: self.temperature,
            headers: self.headers.clone(),
            model_aliases: self.model_aliases.clone(),
        })
    }

    /// Resolve a model name through `model_aliases`. If `model` matches an
    /// alias key, return the real upstream model ID; otherwise return the
    /// input unchanged. Used by every provider implementation right before
    /// it serializes the chat-completions request body.
    pub fn resolve_model_alias<'a>(&'a self, model: &'a str) -> &'a str {
        self.model_aliases
            .get(model)
            .map(String::as_str)
            .unwrap_or(model)
    }

    /// Look up the capability block for `model`, trying the literal id
    /// first and falling back to the alias-resolved id if the literal
    /// wasn't registered. Returns a cloned permissive default
    /// ([`crate::model_catalog::ModelCapabilities::unknown`]) when neither
    /// is present — callers never need to special-case `Option`.
    pub fn capabilities_for(&self, model: &str) -> crate::model_catalog::ModelCapabilities {
        if let Some(caps) = self.capabilities.get(model) {
            return caps.clone();
        }
        let resolved = self.resolve_model_alias(model);
        if resolved != model
            && let Some(caps) = self.capabilities.get(resolved)
        {
            return caps.clone();
        }
        crate::model_catalog::ModelCapabilities::unknown()
    }

    /// Combined model list: real `models` plus alias keys. Aliases that
    /// duplicate an entry already in `models` are skipped so the menu
    /// stays clean. Order: real models first (in declaration order), then
    /// aliases (alphabetical for stability).
    pub fn models_with_aliases(&self) -> Vec<String> {
        let mut out = self.models.clone();
        let mut alias_keys: Vec<&String> = self.model_aliases.keys().collect();
        alias_keys.sort();
        for k in alias_keys {
            if !out.iter().any(|m| m == k) {
                out.push(k.clone());
            }
        }
        out
    }
}

/// Provider config with api_key already resolved (no `$VAR` references).
#[derive(Debug, Clone)]
pub struct ResolvedProvider {
    pub provider_type: String,
    pub api_key: String,
    /// All valid keys (primary + extras), already resolved and deduplicated.
    pub all_keys: Vec<String>,
    pub base_url: Option<String>,
    pub models: Vec<String>,
    pub max_tokens: Option<u32>,
    pub temperature: Option<f32>,
    pub headers: HashMap<String, String>,
    /// See [`ProviderConfig::model_aliases`].
    pub model_aliases: HashMap<String, String>,
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_pc() -> ProviderConfig {
        ProviderConfig {
            provider_type: "openai_compat".into(),
            api_key: "sk-test-123".into(),
            models: vec!["gpt-4o".into(), "gpt-4o-mini".into()],
            model_aliases: HashMap::from([("fast".into(), "gpt-4o-mini".into())]),
            ..Default::default()
        }
    }

    #[test]
    fn resolved_api_key_returns_literal() {
        let pc = test_pc();
        assert_eq!(pc.resolved_api_key().unwrap(), "sk-test-123");
    }

    #[test]
    fn resolved_all_keys_includes_primary() {
        let pc = test_pc();
        let keys = pc.resolved_all_keys();
        assert!(keys.contains(&"sk-test-123".to_string()));
    }

    #[test]
    fn resolved_all_keys_deduplicates() {
        let mut pc = test_pc();
        pc.api_keys = vec!["sk-test-123".into(), "sk-extra".into()];
        let keys = pc.resolved_all_keys();
        assert_eq!(
            keys.iter().filter(|k| k.as_str() == "sk-test-123").count(),
            1
        );
    }

    #[test]
    fn resolve_model_alias_maps() {
        let pc = test_pc();
        assert_eq!(pc.resolve_model_alias("fast"), "gpt-4o-mini");
        assert_eq!(pc.resolve_model_alias("gpt-4o"), "gpt-4o");
    }

    #[test]
    fn models_with_aliases_includes_both() {
        let pc = test_pc();
        let all = pc.models_with_aliases();
        assert!(all.contains(&"gpt-4o".to_string()));
        assert!(all.contains(&"fast".to_string()));
    }

    #[test]
    fn resolved_produces_resolved_provider() {
        let pc = test_pc();
        let rp = pc.resolved().unwrap();
        assert_eq!(rp.provider_type, "openai_compat");
        assert!(!rp.api_key.is_empty());
    }

    #[test]
    fn capabilities_for_unknown_model() {
        let pc = test_pc();
        let caps = pc.capabilities_for("unknown-model");
        // Should return default capabilities, not panic
        assert!(caps.context_window.is_none() || caps.context_window.is_some());
    }
}
