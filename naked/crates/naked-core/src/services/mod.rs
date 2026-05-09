//! Service traits — compile-time contracts between subsystems.
//!
//! These traits define boundaries: session code depends on `dyn ProviderResolver`,
//! not on concrete provider fields. The compiler enforces separation.

use std::path::Path;
use std::sync::Arc;

pub mod provider_svc;
pub mod research_adapter;
pub mod session_config;
pub mod session_state;

use async_trait::async_trait;
use tokio::sync::mpsc;

use crate::error::Result;
use crate::provider::Provider;
use crate::session::SessionSummary;
use crate::types::{AgentEvent, AgentHandle, TurnUsage};

pub use provider_svc::ProviderService;
pub use session_config::SessionConfigService;

// ---------------------------------------------------------------------------
// ProviderResolver — resolving a provider by name
// ---------------------------------------------------------------------------

/// Resolve a named provider to a concrete `Arc<dyn Provider>`.
///
/// Session code uses this instead of touching `provider_cache` directly.
#[async_trait]
pub trait ProviderResolver: Send + Sync {
    /// Return the provider for `name`, or the default if `name` is empty / unknown.
    async fn resolve_provider(&self, name: &str) -> Arc<dyn Provider>;

    /// Default provider + model as `(provider_name, model_id)`.
    fn default_provider_model(&self) -> (String, String);
}

// ---------------------------------------------------------------------------
// SessionManager — session lifecycle (create / query / send prompt)
// ---------------------------------------------------------------------------

/// High-level session operations.
///
/// Provider code uses this for usage queries instead of touching `sessions` HashMap.
#[async_trait]
/// Core session lifecycle: create, send, query.
#[async_trait]
pub trait SessionLifecycle: Send + Sync {
    async fn create_session(&self, workspace: &Path) -> String;
    async fn send_prompt(&self, session_id: &str, text: &str) -> Result<AgentHandle>;
    async fn is_session_active(&self, session_id: &str) -> bool;
}

/// Session control: abort, list.
#[async_trait]
pub trait SessionControl: Send + Sync {
    async fn abort(&self, session_id: &str);
    async fn list_sessions(&self) -> Vec<SessionSummary>;
}

/// Session diagnostics: usage, provider info.
#[async_trait]
pub trait SessionDiagnostics: Send + Sync {
    async fn session_total_usage(&self, session_id: &str) -> TurnUsage;
    async fn session_provider_model(&self, session_id: &str) -> (String, String);
}

/// Combined session manager — all session operations.
/// Implementors get this automatically when they implement all three sub-traits.
pub trait SessionManager: SessionLifecycle + SessionControl + SessionDiagnostics {}

// ---------------------------------------------------------------------------
// EventSink — receiving agent events (tool results, text, errors)
// ---------------------------------------------------------------------------

/// A destination for agent events during a turn.
///
/// Decouples the agent loop from the transport (Telegram, CLI, WebSocket).
pub trait EventSink: Send + Sync {
    fn send_event(&self, event: AgentEvent) -> anyhow::Result<()>;
}

/// Adapter: wrap an `mpsc::UnboundedSender<AgentEvent>` as an `EventSink`.
pub struct ChannelEventSink {
    tx: mpsc::UnboundedSender<AgentEvent>,
}

impl ChannelEventSink {
    pub fn new(tx: mpsc::UnboundedSender<AgentEvent>) -> Self {
        Self { tx }
    }
}

impl EventSink for ChannelEventSink {
    fn send_event(&self, event: AgentEvent) -> anyhow::Result<()> {
        self.tx
            .send(event)
            .map_err(|_| anyhow::anyhow!("event channel closed"))
    }
}

// ---------------------------------------------------------------------------
// ToolBuilder — construct per-session tool registries
// ---------------------------------------------------------------------------

/// Build a tool registry for a specific session.
///
/// Decouples session_ops from the concrete tool construction pipeline
/// (core tools, research tools, skills, MCP, extras).
#[async_trait]
pub trait ToolBuilder: Send + Sync {
    async fn build_registry(
        &self,
        session_id: &str,
        effective: &crate::EffectiveSessionConfig,
        provider: &Arc<dyn Provider>,
        model: &str,
        workspace: &std::path::Path,
    ) -> crate::tool::registry::ToolRegistry;
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use crate::services::*;
    use crate::test_support::TestCore;
    use crate::types::AgentEvent;
    use tokio::sync::mpsc;

    /// Verify ChannelEventSink delivers events.
    #[tokio::test]
    async fn channel_sink_delivers() {
        let (tx, mut rx) = mpsc::unbounded_channel();
        let sink = ChannelEventSink::new(tx);
        sink.send_event(AgentEvent::Heartbeat).unwrap();
        let ev = rx.recv().await.unwrap();
        assert!(matches!(ev, AgentEvent::Heartbeat));
    }

    /// Verify ChannelEventSink returns error when receiver is dropped.
    #[tokio::test]
    async fn channel_sink_closed() {
        let (tx, rx) = mpsc::unbounded_channel();
        drop(rx);
        let sink = ChannelEventSink::new(tx);
        assert!(sink.send_event(AgentEvent::Heartbeat).is_err());
    }

    /// Compile-time check: AgentCore implements ToolBuilder.
    #[test]
    fn agent_core_implements_tool_builder() {
        fn _assert<T: super::ToolBuilder>() {}
        _assert::<crate::AgentCore>();
    }

    #[tokio::test]
    async fn provider_resolver_returns_default_for_empty_name() {
        let tc = TestCore::build();
        let p = <crate::AgentCore as ProviderResolver>::resolve_provider(&tc.core, "").await;
        assert_eq!(p.name(), "noop");
    }

    #[tokio::test]
    async fn provider_resolver_returns_default_model() {
        let tc = TestCore::build();
        let (prov, model) =
            <crate::AgentCore as ProviderResolver>::default_provider_model(&tc.core);
        assert!(!prov.is_empty() || !model.is_empty());
    }

    #[tokio::test]
    async fn session_lifecycle_create_and_query() {
        let tc = TestCore::build();
        let ws = tc.workspace();
        let sid = <crate::AgentCore as SessionLifecycle>::create_session(&tc.core, &ws).await;
        assert!(!sid.is_empty());
        let active =
            <crate::AgentCore as SessionLifecycle>::is_session_active(&tc.core, &sid).await;
        assert!(!active, "new session should be idle");
    }

    #[tokio::test]
    async fn session_control_list_includes_created() {
        let tc = TestCore::build();
        let ws = tc.workspace();
        let sid = tc.core.create_session(&ws).await;
        let sessions = <crate::AgentCore as SessionControl>::list_sessions(&tc.core).await;
        assert!(
            sessions.iter().any(|s| s.id == sid),
            "created session should appear in list"
        );
    }

    #[tokio::test]
    async fn session_diagnostics_usage_starts_zero() {
        let tc = TestCore::build();
        let ws = tc.workspace();
        let sid = tc.core.create_session(&ws).await;
        let usage =
            <crate::AgentCore as SessionDiagnostics>::session_total_usage(&tc.core, &sid).await;
        assert_eq!(usage.input_tokens, 0);
        assert_eq!(usage.output_tokens, 0);
    }
}
pub mod research;
pub mod search;

#[cfg(test)]
mod boundary_tests;
