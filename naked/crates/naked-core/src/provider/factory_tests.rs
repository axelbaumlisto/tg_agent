//! Unit tests for provider::factory.
//!
//! These tests used to live in `lib.rs` next to the inline factory machinery.
//! Moved here together with the factory in T4 of PLAN_CORE_HARDENING_v2.

use std::pin::Pin;
use std::sync::{Arc, Mutex};

use super::{
    ModelOverrideProvider, PROVIDER_FACTORIES, create_provider_chain, create_single_provider,
};
use crate::config;
use crate::provider::{ChatRequest, Provider};
use crate::types::{ModelInfo, StreamChunk};

#[test]
fn provider_factory_anthropic_registered() {
    assert!(
        PROVIDER_FACTORIES
            .iter()
            .any(|(name, _)| *name == "anthropic")
    );
}

#[test]
fn provider_factory_copilot_registered() {
    assert!(
        PROVIDER_FACTORIES
            .iter()
            .any(|(name, _)| *name == "copilot")
    );
}

#[test]
fn create_single_provider_anthropic() {
    let cfg = config::ProviderConfig {
        provider_type: "anthropic".into(),
        api_key: "test-key".into(),
        ..Default::default()
    };
    let p = create_single_provider("test", cfg);
    assert!(p.name().contains("test"));
}

#[test]
fn create_single_provider_unknown_falls_back_to_openai() {
    let cfg = config::ProviderConfig {
        provider_type: "unknown_provider".into(),
        api_key: "test-key".into(),
        ..Default::default()
    };
    let p = create_single_provider("test", cfg);
    // OpenAiCompatProvider is the fallback for unknown types.
    assert!(p.name().contains("test"));
}

#[test]
fn create_provider_single_key_no_resilient_wrapper() {
    use crate::config::ResolvedProvider;

    let resolved = ResolvedProvider {
        provider_type: "anthropic".into(),
        api_key: "key1".into(),
        all_keys: vec!["key1".into()],
        base_url: None,
        models: vec!["claude-test".into()],
        max_tokens: None,
        temperature: None,
        headers: Default::default(),
        model_aliases: Default::default(),
    };
    let p = super::create_provider("solo", resolved);
    // Single-key path returns a plain provider, not a ResilientProvider.
    assert_eq!(p.name(), "solo");
}

#[test]
fn create_provider_chain_for_non_default_inherits_global_fallbacks() {
    let mut cfg = config::Config {
        default_provider: "qwen".into(),
        fallback: vec![
            "deepseek-direct/deepseek-v4-flash".into(),
            "anthropic/claude-sonnet-4-6".into(),
        ],
        ..Default::default()
    };
    cfg.providers.insert(
        "fireworks".into(),
        config::ProviderConfig {
            api_key: "fw-test".into(),
            models: vec!["accounts/fireworks/models/deepseek-v4-pro".into()],
            ..Default::default()
        },
    );
    cfg.providers.insert(
        "deepseek-direct".into(),
        config::ProviderConfig {
            api_key: "ds-test".into(),
            models: vec!["deepseek-v4-flash".into()],
            ..Default::default()
        },
    );
    cfg.providers.insert(
        "anthropic".into(),
        config::ProviderConfig {
            api_key: "ant-test".into(),
            models: vec!["claude-sonnet-4-6".into()],
            ..Default::default()
        },
    );

    let resolved = cfg
        .provider_config("fireworks")
        .expect("provider exists")
        .resolved()
        .expect("resolved");
    let provider = create_provider_chain(&cfg, "fireworks", resolved);

    assert_eq!(provider.name(), "fireworks");
    assert_eq!(provider.total_key_count(), 3);
}

struct CapturingProvider {
    name: String,
    seen_model: Arc<Mutex<Option<String>>>,
}

#[async_trait::async_trait]
impl Provider for CapturingProvider {
    fn name(&self) -> &str {
        &self.name
    }

    fn models(&self) -> Vec<ModelInfo> {
        Vec::new()
    }

    async fn stream_chat(
        &self,
        request: ChatRequest,
    ) -> crate::error::Result<Pin<Box<dyn tokio_stream::Stream<Item = StreamChunk> + Send>>> {
        *self.seen_model.lock().expect("lock") = Some(request.model);
        Ok(Box::pin(tokio_stream::empty()))
    }
}

#[tokio::test]
async fn fallback_model_override_rewrites_request_model() {
    let seen_model = Arc::new(Mutex::new(None));
    let provider = ModelOverrideProvider {
        inner: Box::new(CapturingProvider {
            name: "fallback".into(),
            seen_model: seen_model.clone(),
        }),
        model: "fallback-model".into(),
    };

    let _stream = provider
        .stream_chat(ChatRequest {
            model: "primary-model".into(),
            system: String::new(),
            messages: Vec::new(),
            tools: Vec::new(),
            max_tokens: 1,
            temperature: None,
            reasoning: None,
        })
        .await
        .expect("stream");

    assert_eq!(
        seen_model.lock().expect("lock").as_deref(),
        Some("fallback-model")
    );
}
