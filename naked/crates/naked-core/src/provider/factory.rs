//! Provider factory machinery.

use std::collections::HashMap;
use std::pin::Pin;

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
use tokio_stream::Stream;

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

/// Provider decorator used for fallback entries in `config.fallback[]`.
///
/// A provider fallback is a `(provider, model)` pair. The failed request's
/// original model id usually belongs to the primary provider (for example
/// `accounts/fireworks/models/deepseek-v4-pro`) and must not be sent to the
/// fallback provider. This wrapper keeps the normal request body but rewrites
/// only `ChatRequest.model` to the fallback entry's model before dispatch.
struct ModelOverrideProvider {
    inner: Box<dyn Provider>,
    model: String,
}

#[async_trait::async_trait]
impl Provider for ModelOverrideProvider {
    fn name(&self) -> &str {
        self.inner.name()
    }

    fn models(&self) -> Vec<crate::types::ModelInfo> {
        self.inner.models()
    }

    fn blacklisted_key_count(&self) -> usize {
        self.inner.blacklisted_key_count()
    }

    fn total_key_count(&self) -> usize {
        self.inner.total_key_count()
    }

    // B106: delegate so a fallback recorded by an inner chain stays visible.
    fn last_fallback(&self) -> Option<super::FallbackInfo> {
        self.inner.last_fallback()
    }
    fn key_hint(&self) -> Option<String> {
        self.inner.key_hint()
    }

    async fn audit_keys_on_boot(&self) {
        self.inner.audit_keys_on_boot().await;
    }

    async fn stream_chat(
        &self,
        mut request: crate::provider::ChatRequest,
    ) -> Result<Pin<Box<dyn Stream<Item = crate::types::StreamChunk> + Send>>> {
        request.model = self.model.clone();
        self.inner.stream_chat(request).await
    }
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
            wrap_with_timeout(create_single_provider(&tag, cfg))
        })
        .collect();

    tracing::info!(
        "Provider '{name}': {} keys configured for rotation",
        providers.len()
    );
    Box::new(ResilientProvider::new(providers))
}

/// Wrap a concrete network provider path in [`TimeoutProvider`] for
/// stream-open/connect + inter-chunk deadlock protection.
///
/// Invariant: every concrete network provider path returned by
/// `create_provider` is timeout-guarded: the single-key path is wrapped
/// directly, and each key of a multi-key provider is wrapped before entering
/// the key-rotation aggregate (B78). Aggregate fallback chains — including the
/// multi-key [`ResilientProvider`] and the outer global chain in
/// [`create_provider_chain`] — are NOT additionally connect-timeout wrapped;
/// doing so cancels the chain mid-iteration and defeats fallback when one
/// entry is slow to fail (B66).
fn wrap_with_timeout(inner: Box<dyn Provider>) -> Box<dyn Provider> {
    Box::new(TimeoutProvider::new(
        inner,
        DEFAULT_CONNECT_TIMEOUT,
        DEFAULT_INTER_CHUNK_TIMEOUT,
    ))
}

/// Create a provider chain whose first entry is `primary_name`, followed by
/// the global chat fallback chain from `config.fallback[]`.
///
/// Used both for the configured default provider and for a session-selected
/// non-default provider. Without this, `/provider fireworks` bypasses the
/// shared fallback chain entirely (B58).
pub fn create_provider_chain(
    config: &Config,
    primary_name: &str,
    primary_resolved: ResolvedProvider,
) -> Box<dyn Provider> {
    let mut providers: Vec<Box<dyn Provider>> =
        vec![create_provider(primary_name, primary_resolved)];

    for (fb_provider, fb_model) in config.fallback_providers() {
        if fb_provider == primary_name {
            continue;
        }
        if let Some(pc) = config.provider_config(&fb_provider)
            && let Ok(resolved) = pc.resolved()
        {
            providers.push(Box::new(ModelOverrideProvider {
                inner: create_provider(&fb_provider, resolved),
                model: fb_model,
            }));
        }
    }

    if providers.len() == 1 {
        providers.remove(0)
    } else {
        Box::new(ResilientProvider::new(providers))
    }
}

/// Build a provider (possibly resilient with fallbacks) from config.
pub fn build_provider_from_config(config: &Config) -> Result<Box<dyn Provider>> {
    let (primary_name, primary_resolved) = config.resolve_default_provider()?;
    Ok(create_provider_chain(
        config,
        &primary_name,
        primary_resolved,
    ))
}

#[cfg(test)]
#[path = "factory_tests.rs"]
mod tests;
