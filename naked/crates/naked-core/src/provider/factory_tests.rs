//! Unit tests for provider::factory.
//!
//! These tests used to live in `lib.rs` next to the inline factory machinery.
//! Moved here together with the factory in T4 of PLAN_CORE_HARDENING_v2.

use std::pin::Pin;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use super::{
    ModelOverrideProvider, PROVIDER_FACTORIES, create_provider, create_provider_chain,
    create_single_provider,
};
use crate::config;
use crate::provider::resilient::ResilientProvider;
use crate::provider::timeout::{DEFAULT_CONNECT_TIMEOUT, TimeoutProvider};
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

struct FastFailProvider {
    name: String,
}

#[async_trait::async_trait]
impl Provider for FastFailProvider {
    fn name(&self) -> &str {
        &self.name
    }

    fn models(&self) -> Vec<ModelInfo> {
        Vec::new()
    }

    async fn stream_chat(
        &self,
        _request: ChatRequest,
    ) -> crate::error::Result<Pin<Box<dyn tokio_stream::Stream<Item = StreamChunk> + Send>>> {
        Err(crate::error::AgentError::Provider(format!(
            "{} failed fast",
            self.name
        )))
    }
}

struct HangingProvider {
    name: String,
}

#[async_trait::async_trait]
impl Provider for HangingProvider {
    fn name(&self) -> &str {
        &self.name
    }

    fn models(&self) -> Vec<ModelInfo> {
        Vec::new()
    }

    async fn stream_chat(
        &self,
        _request: ChatRequest,
    ) -> crate::error::Result<Pin<Box<dyn tokio_stream::Stream<Item = StreamChunk> + Send>>> {
        std::future::pending::<()>().await;
        unreachable!("pending future never resolves")
    }
}

struct OkProvider {
    name: String,
}

#[async_trait::async_trait]
impl Provider for OkProvider {
    fn name(&self) -> &str {
        &self.name
    }

    fn models(&self) -> Vec<ModelInfo> {
        Vec::new()
    }

    async fn stream_chat(
        &self,
        _request: ChatRequest,
    ) -> crate::error::Result<Pin<Box<dyn tokio_stream::Stream<Item = StreamChunk> + Send>>> {
        Ok(Box::pin(tokio_stream::empty()))
    }
}

fn short_timeout(inner: Box<dyn Provider>) -> Box<dyn Provider> {
    Box::new(TimeoutProvider::new(
        inner,
        Duration::from_millis(50),
        Duration::from_secs(1),
    ))
}

fn make_chat_request(model: &str) -> ChatRequest {
    // ChatRequest does not derive Default; all provider tests build it
    // exhaustively (see timeout.rs, resilient.rs). Mirrors that convention.
    // REGISTRY-WAIVE: B16 — ChatRequest has no Default derive.
    ChatRequest {
        model: model.into(),
        system: String::new(),
        messages: vec![serde_json::json!({"role": "user", "content": "hi"})],
        tools: Vec::new(),
        max_tokens: 1,
        temperature: None,
        reasoning: None,
    }
}

fn openai_sse_response() -> wiremock::ResponseTemplate {
    wiremock::ResponseTemplate::new(200)
        .append_header("content-type", "text/event-stream")
        .set_body_string("data: [DONE]\n\n")
}

async fn mount_chat_completion(
    server: &wiremock::MockServer,
    response: wiremock::ResponseTemplate,
) {
    use wiremock::Mock;
    use wiremock::matchers::{method, path};

    Mock::given(method("POST"))
        .and(path("/chat/completions"))
        .respond_with(response)
        .mount(server)
        .await;
}

fn provider_config(base_url: String, model: &str) -> config::ProviderConfig {
    config::ProviderConfig {
        provider_type: "openai_compat".into(),
        api_key: "test-key".into(),
        base_url: Some(base_url),
        models: vec![model.into()],
        ..Default::default()
    }
}

#[tokio::test]
async fn multi_key_hanging_last_key_falls_through_to_fallback() {
    let qwen_keys: Vec<Box<dyn Provider>> = vec![
        short_timeout(Box::new(FastFailProvider {
            name: "qwen[key-0]".into(),
        })),
        short_timeout(Box::new(FastFailProvider {
            name: "qwen[key-1]".into(),
        })),
        short_timeout(Box::new(FastFailProvider {
            name: "qwen[key-2]".into(),
        })),
        short_timeout(Box::new(FastFailProvider {
            name: "qwen[key-3]".into(),
        })),
        short_timeout(Box::new(HangingProvider {
            name: "qwen[key-4]".into(),
        })),
    ];
    let qwen = Box::new(ResilientProvider::new(qwen_keys));
    let deepseek = Box::new(OkProvider {
        name: "deepseek".into(),
    });
    let outer = ResilientProvider::new(vec![qwen, deepseek]);

    let started = Instant::now();
    let _stream = outer
        .stream_chat(make_chat_request("m"))
        .await
        .expect("per-key timeout should let fallback reach deepseek");

    assert!(
        started.elapsed() < Duration::from_millis(500),
        "hanging final qwen key should time out at the per-key budget and fall through quickly; elapsed={:?}",
        started.elapsed()
    );
}

#[tokio::test]
#[ignore = "B66 fidelity test: waits real DEFAULT_CONNECT_TIMEOUT (~20s); run deliberately: cargo test -p naked-core --lib provider::factory -- --ignored"]
async fn create_provider_chain_slow_primary_reaches_healthy_secondary() {
    let primary_server = wiremock::MockServer::start().await;
    let secondary_server = wiremock::MockServer::start().await;
    mount_chat_completion(
        &primary_server,
        openai_sse_response().set_delay(DEFAULT_CONNECT_TIMEOUT + Duration::from_secs(2)),
    )
    .await;
    mount_chat_completion(&secondary_server, openai_sse_response()).await;

    let mut cfg = config::Config {
        default_provider: "primary".into(),
        fallback: vec!["secondary/secondary-model".into()],
        ..Default::default()
    };
    cfg.providers.insert(
        "primary".into(),
        provider_config(primary_server.uri(), "primary-model"),
    );
    cfg.providers.insert(
        "secondary".into(),
        provider_config(secondary_server.uri(), "secondary-model"),
    );

    let resolved = cfg
        .provider_config("primary")
        .expect("provider exists")
        .resolved()
        .expect("resolved");
    let provider = create_provider_chain(&cfg, "primary", resolved);

    let started = Instant::now();
    let _stream = provider
        .stream_chat(make_chat_request("primary-model"))
        .await
        .expect("slow primary should time out and healthy secondary should be reached");

    assert!(
        started.elapsed() < DEFAULT_CONNECT_TIMEOUT + Duration::from_secs(2),
        "outer timeout should not wait for the delayed primary response"
    );
    assert_eq!(
        primary_server
            .received_requests()
            .await
            .expect("request recording enabled")
            .len(),
        1,
        "primary should be attempted first"
    );
    assert_eq!(
        secondary_server
            .received_requests()
            .await
            .expect("request recording enabled")
            .len(),
        1,
        "secondary fallback should be reached after primary timeout"
    );
}

#[tokio::test]
#[ignore = "B66 fidelity test: waits real DEFAULT_CONNECT_TIMEOUT (~20s); run deliberately: cargo test -p naked-core --lib provider::factory -- --ignored"]
async fn create_provider_single_provider_path_remains_timeout_guarded() {
    let server = wiremock::MockServer::start().await;
    mount_chat_completion(
        &server,
        openai_sse_response().set_delay(DEFAULT_CONNECT_TIMEOUT + Duration::from_secs(2)),
    )
    .await;

    let resolved = provider_config(server.uri(), "solo-model")
        .resolved()
        .expect("resolved");
    let provider = create_provider("solo", resolved);

    let started = Instant::now();
    let err = match provider.stream_chat(make_chat_request("solo-model")).await {
        Ok(_) => panic!("single provider path should still be timeout guarded"),
        Err(err) => err,
    };

    assert!(err.to_string().contains("connect timeout"));
    assert!(
        started.elapsed() < DEFAULT_CONNECT_TIMEOUT + Duration::from_secs(2),
        "single provider should time out before the delayed response returns"
    );
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

/// D-INV-PROVIDER-DECORATOR-DELEGATES (B110), fourth wrapper.
///
/// `ModelOverrideProvider` is private to `factory.rs`, so it cannot join the
/// shared wrapper sweep in `provider/mod.rs` — same contract, checked here:
/// a forgotten forward compiles fine and silently reverts the method to its
/// trait default (that is exactly how B106 and B110 shipped).
#[tokio::test]
async fn b110_model_override_provider_delegates_optional_methods() {
    struct Probe;

    #[async_trait::async_trait]
    impl Provider for Probe {
        fn name(&self) -> &str {
            "probe"
        }
        fn models(&self) -> Vec<crate::types::ModelInfo> {
            vec![]
        }
        fn blacklisted_key_count(&self) -> usize {
            3
        }
        fn total_key_count(&self) -> usize {
            5
        }
        fn key_hint(&self) -> Option<String> {
            Some("probe-key".into())
        }
        fn last_fallback(&self) -> Option<crate::provider::FallbackInfo> {
            Some(crate::provider::FallbackInfo {
                requested: "req".into(),
                served_by: "srv".into(),
                reason: "why".into(),
            })
        }
        async fn stream_chat(
            &self,
            _r: crate::provider::ChatRequest,
        ) -> crate::error::Result<
            Pin<Box<dyn futures_util::Stream<Item = crate::types::StreamChunk> + Send>>,
        > {
            Ok(Box::pin(tokio_stream::iter(vec![
                crate::types::StreamChunk::Done,
            ])))
        }
    }

    let w = ModelOverrideProvider {
        inner: Box::new(Probe),
        model: "whatever".into(),
    };

    assert_eq!(w.blacklisted_key_count(), 3);
    assert_eq!(w.total_key_count(), 5);
    assert_eq!(
        w.key_hint().as_deref(),
        Some("probe-key"),
        "key_hint must delegate (B46 dead-key persistence)"
    );
    assert_eq!(
        w.last_fallback().map(|f| f.served_by).as_deref(),
        Some("srv"),
        "last_fallback must delegate (B106 banner)"
    );
}
