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

    /// B106: identity of the provider that actually served the LAST
    /// `stream_chat` call, when it differs from the one that was requested.
    ///
    /// `ResilientProvider` silently falls back across providers/keys on
    /// 429/413/5xx, but the caller keeps labelling the turn with the name the
    /// user picked — so the tracing span, `model_health.jsonl` and the chat
    /// UI all reported e.g. `groq` success for a turn groq rejected with 413
    /// and deepseek actually answered. This accessor lets the turn layer
    /// re-label itself with the truth.
    ///
    /// Returns `None` when no fallback happened (the common case) or when the
    /// implementation does not track it. Decorators MUST delegate so the
    /// answer reaches the `ResilientProvider` at the bottom of the chain.
    fn last_fallback(&self) -> Option<FallbackInfo> {
        None
    }
}

/// B106: who actually answered, versus who was asked.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FallbackInfo {
    /// Provider name the caller requested (front of the order at call time).
    pub requested: String,
    /// Provider name that actually produced the stream.
    pub served_by: String,
    /// Short reason from the first failure (already redacted/truncated).
    pub reason: String,
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

/// Shared reqwest client builder for streaming LLM providers.
///
/// - `connect_timeout(10s)`: bound TCP+TLS establishment.
/// - `read_timeout(DEFAULT_INTER_CHUNK_TIMEOUT)`: B78 — bound idle gaps
///   between body reads (header-wait / first-byte / inter-chunk) at the
///   transport layer, below the StreamExt inter-chunk guard. Aligns with
///   the 60s inter-chunk policy so legitimate slow thinking streams that
///   already pass the StreamExt guard are not newly killed.
/// - `timeout(300s)`: total request deadline (unchanged).
pub(crate) fn build_streaming_http_client() -> reqwest::Client {
    reqwest::Client::builder()
        .connect_timeout(std::time::Duration::from_secs(10))
        .read_timeout(crate::provider::timeout::DEFAULT_INTER_CHUNK_TIMEOUT)
        .timeout(std::time::Duration::from_secs(300))
        .build()
        .unwrap_or_else(|e| {
            // B112: `unwrap_or_default()` here silently substituted a client
            // with NO connect timeout, NO read timeout and NO total deadline —
            // the exact opposite of this function's purpose, and invisible.
            // Subsequent hangs would be blamed on the provider.
            tracing::error!(
                error = %e,
                "streaming HTTP client build failed; falling back to a client WITHOUT timeouts"
            );
            reqwest::Client::default()
        })
}

/// B71 + B81: thinking-class models (Claude via airpx, native Anthropic
/// extended-thinking) reject an explicit `temperature` when thinking is
/// enabled. B81 (2026-06-21 live): airpx `claude-sonnet-4-6` runs in
/// *adaptive mode* where thinking is on by default server-side, so it
/// rejects ANY explicit temperature ≠ 1 even when the request carries
/// `reasoning: off` / no reasoning (error: "temperature may only be set
/// to 1 when thinking is enabled or in adaptive mode"). The earlier B71
/// rule — "thinking model + reasoning off + 0.0 <= temp < 1.0 → send" —
/// was therefore unsafe and broke the research fallback chain. KISS-safe
/// fix: for a thinking-capable model, NEVER serialize an explicit
/// temperature; let the provider's adaptive default apply.
/// Returns true only when it is safe to serialize `temperature`.
pub(crate) fn should_send_temperature(
    temp: Option<f32>,
    reasoning_on: bool,
    model_supports_thinking: bool,
) -> bool {
    match temp {
        None => false,
        Some(t) => {
            if reasoning_on {
                return false;
            }
            if model_supports_thinking {
                // B81: adaptive-mode thinking proxies reject any explicit
                // temperature regardless of reasoning flag.
                return false;
            }
            t.is_finite() && (0.0..=1.0).contains(&t)
        }
    }
}

/// B112: forward every OPTIONAL `Provider` method to an inner provider.
///
/// Four wrappers must forward all of them (`Box<dyn Provider>`, `ArcProvider`,
/// `TimeoutProvider`, `ModelOverrideProvider`). Each forward was hand-written,
/// and because the trait supplies defaults a forgotten one COMPILES SILENTLY
/// and reverts that method to its default — twice already the cause of a live
/// issue. A test pins the behaviour, but the duplication itself is the hazard:
/// adding a method still meant N remembered edits, and one wrapper lives in
/// another module where the shared test cannot reach it.
///
/// Adding a SYNC optional method is now a single line here. `async` methods
/// stay hand-written: `#[async_trait]` rewrites their signatures before it can
/// see through a macro expansion, so `audit_keys_on_boot` is excluded and each
/// wrapper still forwards it explicitly (the shared probe test covers that).
macro_rules! delegate_optional_provider_methods {
    // `$me` must be the caller's own `self` identifier: macro hygiene forbids
    // the macro body from naming `self` itself.
    ($me:ident => $($inner:tt)+) => {
        fn blacklisted_key_count(&$me) -> usize {
            $($inner)+.blacklisted_key_count()
        }
        fn total_key_count(&$me) -> usize {
            $($inner)+.total_key_count()
        }
        fn key_hint(&$me) -> Option<String> {
            $($inner)+.key_hint()
        }
        fn last_fallback(&$me) -> Option<$crate::provider::FallbackInfo> {
            $($inner)+.last_fallback()
        }
    };
}
pub(crate) use delegate_optional_provider_methods;

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
    delegate_optional_provider_methods!(self => (**self));
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
    // B106: this wrapper previously forwarded ONLY name/models/stream_chat, so
    // every optional trait method silently fell back to its default once a
    // provider was boxed for `AgentLoop` — `last_fallback()` always answered
    // None in production even though `ResilientProvider` had recorded a real
    // fallback, and key-health accessors under-reported the same way.
    delegate_optional_provider_methods!(self => self.0);
    async fn audit_keys_on_boot(&self) {
        self.0.audit_keys_on_boot().await
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
    fn should_send_temperature_matches_b71_b81_truth_table() {
        // D-INV-THINKING-TEMP: FULL truth table over
        // (temp) x (reasoning_on) x (model_supports_thinking).
        // Expected values are literal (not derived from the impl) per the
        // documented B71 + B81 contract:
        //   - temp None                      -> never send
        //   - reasoning_on                   -> never send (any model)
        //   - model_supports_thinking (B81)  -> never send (adaptive mode
        //     rejects any explicit temperature != 1, even with reasoning off)
        //   - plain model, reasoning off     -> send iff finite and in [0,1]
        let cells: &[(Option<f32>, bool, bool, bool)] = &[
            // (temp, reasoning_on, model_supports_thinking, expected)
            // -- reasoning off, plain (non-thinking) model: the ONLY rows
            //    where an explicit temperature may be serialized.
            (None, false, false, false),
            (Some(0.0), false, false, true),
            (Some(0.5), false, false, true),
            (Some(1.0), false, false, true),
            (Some(-0.1), false, false, false),
            (Some(1.5), false, false, false),
            (Some(f32::NAN), false, false, false),
            (Some(f32::INFINITY), false, false, false),
            (Some(f32::NEG_INFINITY), false, false, false),
            // -- reasoning off, thinking-capable model (B81): never send.
            (None, false, true, false),
            (Some(0.0), false, true, false),
            (Some(0.5), false, true, false),
            (Some(1.0), false, true, false),
            (Some(-0.1), false, true, false),
            (Some(1.5), false, true, false),
            (Some(f32::NAN), false, true, false),
            (Some(f32::INFINITY), false, true, false),
            (Some(f32::NEG_INFINITY), false, true, false),
            // -- reasoning on, plain model: never send.
            (None, true, false, false),
            (Some(0.0), true, false, false),
            (Some(0.5), true, false, false),
            (Some(1.0), true, false, false),
            (Some(-0.1), true, false, false),
            (Some(1.5), true, false, false),
            (Some(f32::NAN), true, false, false),
            (Some(f32::INFINITY), true, false, false),
            (Some(f32::NEG_INFINITY), true, false, false),
            // -- reasoning on, thinking-capable model: never send.
            (None, true, true, false),
            (Some(0.0), true, true, false),
            (Some(0.5), true, true, false),
            (Some(1.0), true, true, false),
            (Some(-0.1), true, true, false),
            (Some(1.5), true, true, false),
            (Some(f32::NAN), true, true, false),
            (Some(f32::INFINITY), true, true, false),
            (Some(f32::NEG_INFINITY), true, true, false),
        ];
        for &(temp, reasoning_on, thinking, expected) in cells {
            assert_eq!(
                should_send_temperature(temp, reasoning_on, thinking),
                expected,
                "cell (temp={temp:?}, reasoning_on={reasoning_on}, \
                 model_supports_thinking={thinking}) must be {expected}"
            );
        }
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
    // ── D-INV-PROVIDER-DECORATOR-DELEGATES (B110) ──────────────────────────
    //
    // `Provider` has five OPTIONAL methods with defaults. Four wrappers must
    // forward all of them: `Box<dyn Provider>`, `ArcProvider`,
    // `TimeoutProvider`, `ModelOverrideProvider`. A forgotten forward compiles
    // silently and reverts that method to its default — which is how B106
    // (`last_fallback` lost through `ArcProvider`) and B110 (`key_hint` lost
    // through `TimeoutProvider`, disabling B46 dead-key persistence) both
    // happened. This test drives every wrapper through the SAME probe so a new
    // optional method only has to be added here once.

    struct ProbeProvider;

    #[async_trait]
    impl Provider for ProbeProvider {
        fn name(&self) -> &str {
            "probe"
        }
        fn models(&self) -> Vec<ModelInfo> {
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
        fn last_fallback(&self) -> Option<FallbackInfo> {
            Some(FallbackInfo {
                requested: "req".into(),
                served_by: "srv".into(),
                reason: "why".into(),
            })
        }
        async fn stream_chat(
            &self,
            _request: ChatRequest,
        ) -> crate::error::Result<Pin<Box<dyn Stream<Item = crate::types::StreamChunk> + Send>>>
        {
            Ok(Box::pin(tokio_stream::iter(vec![
                crate::types::StreamChunk::Done,
            ])))
        }
    }

    /// Assert a wrapper forwards every optional method to `ProbeProvider`.
    fn assert_delegates(label: &str, w: &dyn Provider) {
        assert_eq!(
            w.blacklisted_key_count(),
            3,
            "{label}: blacklisted_key_count"
        );
        assert_eq!(w.total_key_count(), 5, "{label}: total_key_count");
        assert_eq!(
            w.key_hint().as_deref(),
            Some("probe-key"),
            "{label}: key_hint (B46 dead-key persistence depends on it)"
        );
        assert_eq!(
            w.last_fallback().map(|f| f.served_by).as_deref(),
            Some("srv"),
            "{label}: last_fallback (B106 fallback banner depends on it)"
        );
    }

    #[test]
    fn b110_every_provider_wrapper_delegates_optional_methods() {
        let boxed: Box<dyn Provider> = Box::new(ProbeProvider);
        assert_delegates("Box<dyn Provider>", &boxed);

        let arc: std::sync::Arc<dyn Provider> = std::sync::Arc::new(ProbeProvider);
        let via_arc = provider_to_box(&arc);
        assert_delegates("ArcProvider (provider_to_box)", &via_arc);

        let timed = super::timeout::TimeoutProvider::with_defaults(
            Box::new(ProbeProvider) as Box<dyn Provider>
        );
        assert_delegates("TimeoutProvider", &timed);
    }
}
