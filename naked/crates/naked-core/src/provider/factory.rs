//! Provider factory machinery.

use std::collections::HashMap;

use crate::config::{Config, ResolvedProvider};
use crate::error::Result;
use crate::provider::Provider;
use crate::provider::anthropic::AnthropicProvider;
use crate::provider::copilot::CopilotProvider;
use crate::provider::openai_compat::OpenAiCompatProvider;
use crate::provider::resilient::ResilientProvider;
use crate::provider::timeout::{
    DEFAULT_CONNECT_TIMEOUT, DEFAULT_INTER_CHUNK_TIMEOUT, TimeoutProvider,
};

/// Registry of provider factories — add new provider types here.
///
/// Open/Closed: adding a new provider type = one line in this array,
/// no changes to `create_single_provider`.
type ProviderFactory = fn(String, crate::config::ProviderConfig) -> Box<dyn Provider>;
pub(super) static PROVIDER_FACTORIES: &[(&str, ProviderFactory)] = &[
    ("anthropic", |name, cfg| {
        Box::new(AnthropicProvider::new(name, cfg))
    }),
    ("copilot", |name, cfg| {
        Box::new(CopilotProvider::new(name, cfg))
    }),
];

/// Create a single Provider instance for one key.
/// Looks up `cfg.provider_type` in [`PROVIDER_FACTORIES`]; falls back
/// to OpenAI-compatible if no match (covers openai, deepseek, kimi, etc.).
pub(super) fn create_single_provider(
    name: &str,
    cfg: crate::config::ProviderConfig,
) -> Box<dyn Provider> {
    for &(type_name, factory) in PROVIDER_FACTORIES {
        if cfg.provider_type == type_name {
            return factory(name.to_string(), cfg);
        }
    }
    Box::new(OpenAiCompatProvider::new(name.to_string(), cfg))
}

/// Create a Provider from a resolved config.
/// If multiple keys are available, wraps them in ResilientProvider for
/// automatic key rotation on failure.
pub fn create_provider(name: &str, resolved: ResolvedProvider) -> Box<dyn Provider> {
    if resolved.all_keys.len() <= 1 {
        let cfg = crate::config::ProviderConfig {
            provider_type: resolved.provider_type,
            api_key: resolved.api_key,
            api_keys: Vec::new(),
            base_url: resolved.base_url,
            models: resolved.models,
            max_tokens: resolved.max_tokens,
            temperature: resolved.temperature,
            context_window: None,
            headers: resolved.headers,
            supports_vision: None,
            model_aliases: resolved.model_aliases,
            capabilities: HashMap::new(),
        };
        return wrap_with_timeout(create_single_provider(name, cfg));
    }

    let providers: Vec<Box<dyn Provider>> = resolved
        .all_keys
        .iter()
        .enumerate()
        .map(|(i, key)| {
            let tag = format!("{name}[key-{i}]");
            let cfg = crate::config::ProviderConfig {
                provider_type: resolved.provider_type.clone(),
                api_key: key.clone(),
                api_keys: Vec::new(),
                base_url: resolved.base_url.clone(),
                models: resolved.models.clone(),
                max_tokens: resolved.max_tokens,
                temperature: resolved.temperature,
                context_window: None,
                headers: resolved.headers.clone(),
                supports_vision: None,
                model_aliases: resolved.model_aliases.clone(),
                capabilities: HashMap::new(),
            };
            create_single_provider(&tag, cfg)
        })
        .collect();

    tracing::info!(
        "Provider '{name}': {} keys configured for rotation",
        providers.len()
    );
    wrap_with_timeout(Box::new(ResilientProvider::new(providers)))
}

/// Wrap any provider in [`TimeoutProvider`] for connect + inter-chunk
/// deadlock protection. Every provider MUST go through this before
/// being returned to the caller.
fn wrap_with_timeout(inner: Box<dyn Provider>) -> Box<dyn Provider> {
    Box::new(TimeoutProvider::new(
        inner,
        DEFAULT_CONNECT_TIMEOUT,
        DEFAULT_INTER_CHUNK_TIMEOUT,
    ))
}

/// Build a provider (possibly resilient with fallbacks) from config.
pub fn build_provider_from_config(config: &Config) -> Result<Box<dyn Provider>> {
    let (primary_name, primary_resolved) = config.resolve_default_provider()?;
    let mut providers: Vec<Box<dyn Provider>> =
        vec![create_provider(&primary_name, primary_resolved)];

    for (fb_provider, _fb_model) in config.fallback_providers() {
        if fb_provider == primary_name {
            continue;
        }
        if let Some(pc) = config.provider_config(&fb_provider)
            && let Ok(resolved) = pc.resolved()
        {
            providers.push(create_provider(&fb_provider, resolved));
        }
    }

    if providers.len() == 1 {
        Ok(wrap_with_timeout(providers.remove(0)))
    } else {
        Ok(wrap_with_timeout(Box::new(ResilientProvider::new(
            providers,
        ))))
    }
}

#[cfg(test)]
#[path = "factory_tests.rs"]
mod tests;
