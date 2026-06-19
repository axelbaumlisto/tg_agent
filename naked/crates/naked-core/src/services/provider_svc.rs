//! ProviderService — owns all provider-related state.
//!
//! Before this existed, `provider`, `provider_cache`, and `model_health` were
//! fields on `AgentCore`, and any `impl AgentCore` block could touch them.
//! Now they live behind a single `Arc<ProviderService>` and only this module
//! can mutate the cache.

use std::collections::HashMap;
use std::sync::Arc;

use tokio::sync::RwLock;

use crate::ProviderInfo;
use crate::config::Config;
use crate::model_catalog::ModelHealth;
use crate::provider::Provider;
use crate::types::ModelInfo;

/// Owns provider resolution, caching, and health tracking.
pub struct ProviderService {
    /// Default provider (used when name is empty or matches default).
    default: Arc<dyn Provider>,
    /// Cache of non-default providers, keyed by name.
    cache: RwLock<HashMap<String, Arc<dyn Provider>>>,
    /// Model health tracker (token tracking, error rates).
    health: Arc<ModelHealth>,
    /// Reference to config for provider catalog lookup.
    config: Arc<Config>,
}

impl ProviderService {
    pub fn new(default: Arc<dyn Provider>, health: Arc<ModelHealth>, config: Arc<Config>) -> Self {
        Self {
            default,
            cache: RwLock::new(HashMap::new()),
            health,
            config,
        }
    }

    /// Resolve a provider by name. Returns default if name is empty or not configured.
    pub async fn resolve(&self, name: &str) -> Arc<dyn Provider> {
        if name.is_empty() || name == self.config.default_provider {
            return self.default.clone();
        }

        // Check cache first.
        if let Some(cached) = self.cache.read().await.get(name) {
            return cached.clone();
        }

        // Build from config.
        let built: Arc<dyn Provider> = if let Some(pc) = self.config.providers.get(name)
            && let Ok(resolved) = pc.resolved()
        {
            Arc::from(crate::create_provider_chain(&self.config, name, resolved))
        } else {
            tracing::warn!("session requests provider '{name}' not in catalog, using default");
            return self.default.clone();
        };

        self.cache
            .write()
            .await
            .insert(name.to_string(), built.clone());
        built
    }

    /// The default provider instance.
    pub fn default_provider(&self) -> Arc<dyn Provider> {
        self.default.clone()
    }

    /// Model health tracker.
    pub fn health(&self) -> Arc<ModelHealth> {
        self.health.clone()
    }

    /// List models from the default provider.
    pub fn list_models(&self) -> Vec<ModelInfo> {
        self.default.models()
    }

    /// List models for a specific provider.
    pub fn provider_models(&self, provider_name: &str) -> Vec<(String, String)> {
        if let Some(pc) = self.config.providers.get(provider_name) {
            pc.models
                .iter()
                .map(|m| (provider_name.to_string(), m.clone()))
                .collect()
        } else {
            Vec::new()
        }
    }

    /// List all configured providers.
    pub fn list_providers(&self) -> Vec<ProviderInfo> {
        let mut result = Vec::new();
        for (name, pc) in &self.config.providers {
            let active = name == &self.config.default_provider;
            result.push(ProviderInfo {
                name: name.clone(),
                models: pc.models.clone(),
                active,
            });
        }
        result.sort_by(|a, b| a.name.cmp(&b.name));
        result
    }

    /// Default provider + model.
    pub fn default_provider_model(&self) -> (String, String) {
        let effective = self
            .config
            .merge_session(&crate::config::SessionConfig::default());
        (effective.provider, effective.model)
    }

    /// Invalidate a cached provider (e.g. after config change).
    pub async fn invalidate(&self, name: &str) {
        self.cache.write().await.remove(name);
    }
}

// ---------------------------------------------------------------------------
// Trait impl
// ---------------------------------------------------------------------------

#[async_trait::async_trait]
impl super::ProviderResolver for ProviderService {
    async fn resolve_provider(&self, name: &str) -> Arc<dyn Provider> {
        self.resolve(name).await
    }
    fn default_provider_model(&self) -> (String, String) {
        self.default_provider_model()
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::services::ProviderResolver;

    struct MockProvider {
        name: String,
    }

    #[async_trait::async_trait]
    impl Provider for MockProvider {
        fn name(&self) -> &str {
            &self.name
        }
        fn models(&self) -> Vec<ModelInfo> {
            vec![]
        }
        async fn stream_chat(
            &self,
            _request: crate::provider::ChatRequest,
        ) -> crate::error::Result<
            std::pin::Pin<Box<dyn tokio_stream::Stream<Item = crate::types::StreamChunk> + Send>>,
        > {
            Err(crate::error::AgentError::Config("mock".into()))
        }
    }

    fn test_config() -> Arc<Config> {
        Arc::new(Config::default())
    }

    #[tokio::test]
    async fn resolve_empty_returns_default() {
        let svc = ProviderService::new(
            Arc::new(MockProvider {
                name: "default".into(),
            }),
            Arc::new(ModelHealth::new(Default::default())),
            test_config(),
        );
        let p = svc.resolve("").await;
        assert_eq!(p.name(), "default");
    }

    #[tokio::test]
    async fn resolve_unknown_returns_default() {
        let svc = ProviderService::new(
            Arc::new(MockProvider {
                name: "default".into(),
            }),
            Arc::new(ModelHealth::new(Default::default())),
            test_config(),
        );
        let p = svc.resolve("nonexistent").await;
        assert_eq!(p.name(), "default");
    }

    #[tokio::test]
    async fn trait_impl_works() {
        let svc = ProviderService::new(
            Arc::new(MockProvider {
                name: "default".into(),
            }),
            Arc::new(ModelHealth::new(Default::default())),
            test_config(),
        );
        let resolver: &dyn ProviderResolver = &svc;
        let p = resolver.resolve_provider("").await;
        assert_eq!(p.name(), "default");
    }

    #[tokio::test]
    async fn resolve_non_default_inherits_global_fallback_chain() {
        let mut cfg = Config {
            default_provider: "qwen".into(),
            fallback: vec!["deepseek-direct/deepseek-v4-flash".into()],
            ..Default::default()
        };
        cfg.providers.insert(
            "fireworks".into(),
            crate::config::ProviderConfig {
                api_key: "fw-test".into(),
                models: vec!["accounts/fireworks/models/deepseek-v4-pro".into()],
                ..Default::default()
            },
        );
        cfg.providers.insert(
            "deepseek-direct".into(),
            crate::config::ProviderConfig {
                api_key: "ds-test".into(),
                models: vec!["deepseek-v4-flash".into()],
                ..Default::default()
            },
        );
        let svc = ProviderService::new(
            Arc::new(MockProvider {
                name: "qwen".into(),
            }),
            Arc::new(ModelHealth::new(Default::default())),
            Arc::new(cfg),
        );

        let p = svc.resolve("fireworks").await;

        assert_eq!(p.name(), "fireworks");
        assert_eq!(p.total_key_count(), 2);
    }
}
