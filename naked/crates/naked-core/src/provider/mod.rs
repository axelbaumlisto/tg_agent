pub mod anthropic;
pub mod audit_targets;
pub mod copilot;
pub mod dead_key_persist; // B46 PLAN_PROVIDER_HEALTH_v1
pub mod error;
pub mod factory;
pub mod openai_compat;
pub mod resilient;
pub mod timeout;

pub use timeout::{DEFAULT_CONNECT_TIMEOUT, DEFAULT_INTER_CHUNK_TIMEOUT, TimeoutProvider};

pub use audit_targets::{
    DedupAuditPlan, DedupAuditTarget, KeyFingerprint, dedup_audit_targets, key_fingerprint,
};
pub use factory::{build_provider_from_config, create_provider, create_provider_chain};

use std::pin::Pin;

use async_trait::async_trait;
use tokio_stream::Stream;

use crate::types::{ModelInfo, StreamChunk, ToolSpec};

#[derive(Debug, Clone)]
pub struct ChatRequest {
    pub model: String,
    pub system: String,
    pub messages: Vec<serde_json::Value>,
    pub tools: Vec<serde_json::Value>,
    pub max_tokens: u32,
    pub temperature: Option<f32>,
    /// Reasoning/thinking level: "off", "low", "medium", "high"
    pub reasoning: Option<String>,
}

#[async_trait]
pub trait Provider: Send + Sync {
    fn name(&self) -> &str;
    fn models(&self) -> Vec<ModelInfo>;

    async fn stream_chat(
        &self,
        request: ChatRequest,
    ) -> crate::error::Result<Pin<Box<dyn Stream<Item = StreamChunk> + Send>>>;

    /// Number of currently blacklisted keys. Default: 0 (single-key providers).
    /// Override in `ResilientProvider` to report actual blacklist state.
    fn blacklisted_key_count(&self) -> usize {
        0
    }

    /// Total number of keys this provider was configured with.
    fn total_key_count(&self) -> usize {
        1
    }

    /// R2 of PLAN_RESILIENCE_v1: optional boot-time key audit.
    /// Implementations that hold multiple keys (`ResilientProvider`)
    /// override this to probe each key and permanent-blacklist any
    /// that return 401/402. Default: no-op (single-key providers
    /// have nothing to audit).
    ///
    /// Decorators (`TimeoutProvider`, `Box<dyn Provider>`) MUST
    /// delegate to the inner provider so the audit reaches the
    /// `ResilientProvider` at the bottom of the chain.
    async fn audit_keys_on_boot(&self) {}

    /// B46 / PLAN_PROVIDER_HEALTH_v1: optional accessor for the literal
    /// `api_key` value this provider holds. Used by
    /// [`super::dead_key_persist::persist_dead_key`] when a key is
    /// permanent-blacklisted at runtime, so we can write it back to
    /// `state/naked.json::_dead_api_keys_auto_<date>`.
    ///
    /// Default: `None` (single-key auth-less providers — copilot via OAuth,
    /// decorators, mocks). Override in concrete provider implementations
    /// that actually keep an `api_key: String` field.
    fn key_hint(&self) -> Option<String> {
        None
    }
}

/// Strip `[key-N]` suffix from a tagged provider name to recover the
/// logical provider key used in `state/naked.json::providers.<X>`.
///
/// `qwen[key-3]` → `qwen`; `qwen` (no suffix) → `qwen`.
pub fn logical_provider_name(tagged: &str) -> &str {
    match tagged.find("[key-") {
        Some(i) => &tagged[..i],
        None => tagged,
    }
}

/// Pass-through impl so `Box<dyn Provider>` can stand in wherever
/// `P: Provider` is required (notably as the inner of
/// [`timeout::TimeoutProvider`]). R3 of `PLAN_NEXT_SESSION.md`.
#[async_trait]
impl Provider for Box<dyn Provider> {
    fn name(&self) -> &str {
        (**self).name()
    }
    fn models(&self) -> Vec<ModelInfo> {
        (**self).models()
    }
    fn blacklisted_key_count(&self) -> usize {
        (**self).blacklisted_key_count()
    }
    fn total_key_count(&self) -> usize {
        (**self).total_key_count()
    }
    fn key_hint(&self) -> Option<String> {
        (**self).key_hint()
    }
    async fn stream_chat(
        &self,
        request: ChatRequest,
    ) -> crate::error::Result<Pin<Box<dyn Stream<Item = StreamChunk> + Send>>> {
        (**self).stream_chat(request).await
    }
    async fn audit_keys_on_boot(&self) {
        (**self).audit_keys_on_boot().await
    }
}

pub fn tool_spec_to_anthropic_json(spec: &ToolSpec) -> serde_json::Value {
    serde_json::json!({
        "name": spec.name,
        "description": spec.description,
        "input_schema": spec.parameters,
    })
}

/// Adapter: wraps `Arc<dyn Provider>` into `Box<dyn Provider>`.
struct ArcProvider(std::sync::Arc<dyn Provider>);

#[async_trait]
impl Provider for ArcProvider {
    fn name(&self) -> &str {
        self.0.name()
    }
    fn models(&self) -> Vec<crate::types::ModelInfo> {
        self.0.models()
    }
    async fn stream_chat(
        &self,
        request: ChatRequest,
    ) -> crate::error::Result<Pin<Box<dyn Stream<Item = crate::types::StreamChunk> + Send>>> {
        self.0.stream_chat(request).await
    }
}

/// Convert `Arc<dyn Provider>` to `Box<dyn Provider>` for APIs that need ownership.
pub(crate) fn provider_to_box(provider: &std::sync::Arc<dyn Provider>) -> Box<dyn Provider> {
    Box::new(ArcProvider(provider.clone()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::Permission;

    #[test]
    fn tool_spec_to_json_format() {
        let spec = ToolSpec {
            name: "bash".into(),
            description: "Run a command".into(),
            parameters: serde_json::json!({"type": "object", "properties": {"command": {"type": "string"}}}),
            permission: Permission::Dangerous,
        };
        let json = tool_spec_to_anthropic_json(&spec);
        assert_eq!(json["name"], "bash");
        assert_eq!(json["description"], "Run a command");
        assert!(json["input_schema"]["properties"]["command"].is_object());
    }

    #[test]
    fn chat_request_debug() {
        let req = ChatRequest {
            model: "claude-sonnet-4".into(),
            system: "You are helpful".into(),
            messages: vec![],
            tools: vec![],
            max_tokens: 1024,
            temperature: None,
            reasoning: None,
        };
        let debug = format!("{req:?}");
        assert!(debug.contains("claude-sonnet-4"));
    }
}
