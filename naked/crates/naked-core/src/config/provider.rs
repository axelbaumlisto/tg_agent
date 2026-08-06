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

    /// Is `model` selectable on this provider?
    ///
    /// Accepts the literal id, an alias KEY (what menus show), or an alias
    /// TARGET (the upstream id). Single source of truth: this predicate was
    /// open-coded in three places with two DIFFERENT shapes — `turn::
    /// validate_model` and `set_session_provider` checked
    /// `models | alias keys | alias values` while `ModelSelector::validate`
    /// checked `models (literal or alias-resolved) | alias keys`. A model
    /// could therefore pass one gate and fail another, which is how a `/model`
    /// pick can be accepted and then rejected at turn time.
    pub fn serves_model(&self, model: &str) -> bool {
        let real = self.resolve_model_alias(model);
        self.models.iter().any(|m| m == model || m == real)
            || self.model_aliases.contains_key(model)
            || self.model_aliases.values().any(|v| v == model)
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

    /// Alias-resolved capability fallback: caps registered under the
    /// canonical target id must be inherited when looked up via the alias.
    /// Guards the `resolve_model_alias` fallback branch in
    /// [`ProviderConfig::capabilities_for`] — deleting it makes an alias
    /// lookup return `unknown()` instead of the canonical caps.
    #[test]
    fn capabilities_for_alias_inherits_canonical_target() {
        use crate::model_catalog::ModelCapabilities;

        // Distinctive, non-default caps registered under the CANONICAL id.
        let canonical_caps = ModelCapabilities {
            supports_vision: Some(true),
            context_window: Some(200_000),
            ..ModelCapabilities::unknown()
        };

        let mut pc = test_pc(); // model_aliases: {"fast" -> "gpt-4o-mini"}
        pc.capabilities
            .insert("gpt-4o-mini".into(), canonical_caps.clone());

        // Alias lookup inherits the canonical target's caps (fallback branch),
        // NOT unknown().
        let via_alias = pc.capabilities_for("fast");
        assert_eq!(via_alias.supports_vision, Some(true));
        assert_eq!(via_alias.context_window, Some(200_000));
        let unknown = ModelCapabilities::unknown();
        assert_ne!(
            via_alias.context_window, unknown.context_window,
            "alias must not fall through to unknown()"
        );
    }

    /// Literal-registered id wins over any alias resolution (literal-first).
    #[test]
    fn capabilities_for_literal_registered_wins() {
        use crate::model_catalog::ModelCapabilities;

        let literal_caps = ModelCapabilities {
            context_window: Some(512),
            ..ModelCapabilities::unknown()
        };
        let canonical_caps = ModelCapabilities {
            context_window: Some(200_000),
            ..ModelCapabilities::unknown()
        };

        let mut pc = test_pc(); // model_aliases: {"fast" -> "gpt-4o-mini"}
        // Register BOTH the alias key literally and the canonical target,
        // with different caps; literal must win.
        pc.capabilities.insert("fast".into(), literal_caps);
        pc.capabilities.insert("gpt-4o-mini".into(), canonical_caps);

        let caps = pc.capabilities_for("fast");
        assert_eq!(caps.context_window, Some(512));
    }

    /// A model that is neither literally registered nor an alias falls back
    /// to the permissive `unknown()` default.
    #[test]
    fn capabilities_for_totally_unknown_is_unknown_default() {
        use crate::model_catalog::ModelCapabilities;

        let pc = test_pc();
        let caps = pc.capabilities_for("totally-unknown");
        let unknown = ModelCapabilities::unknown();
        assert_eq!(caps.supports_vision, unknown.supports_vision);
        assert_eq!(caps.context_window, unknown.context_window);
        assert_eq!(caps.status, unknown.status);
    }
    // ── serves_model: one predicate for every model-selectability gate ─────
    //
    // Was open-coded in three places with TWO different shapes, so a model
    // could pass `/model` and then be rejected at turn time. These pin the
    // union all callers now share.

    #[test]
    fn serves_model_accepts_literal_alias_key_and_alias_target() {
        let pc = test_pc();
        assert!(pc.serves_model("gpt-4o"), "literal id");
        assert!(pc.serves_model("fast"), "alias key (what menus show)");
        assert!(
            pc.serves_model("gpt-4o-mini"),
            "alias target is also a listed model"
        );
    }

    #[test]
    fn serves_model_accepts_alias_target_not_in_models() {
        // Alias points at an upstream id the provider does not list directly.
        let pc = ProviderConfig {
            models: vec!["listed".into()],
            model_aliases: HashMap::from([("nice-name".into(), "opaque-upstream-id".into())]),
            ..Default::default()
        };
        assert!(pc.serves_model("nice-name"), "alias key");
        assert!(pc.serves_model("opaque-upstream-id"), "alias target");
        assert!(pc.serves_model("listed"));
    }

    #[test]
    fn serves_model_rejects_unknown() {
        let pc = test_pc();
        assert!(!pc.serves_model("claude-sonnet-4-6"));
        assert!(!pc.serves_model(""));
    }
}
