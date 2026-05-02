//! Service traits — compile-time contracts between subsystems.
//!
//! These traits define boundaries: session code depends on `dyn ProviderResolver`,
//! not on concrete provider fields. The compiler enforces separation.

use std::path::Path;
use std::sync::Arc;

pub mod provider_svc;
pub mod session_state;

use async_trait::async_trait;
use tokio::sync::mpsc;

use crate::error::Result;
use crate::provider::Provider;
use crate::session::SessionSummary;
use crate::types::{AgentEvent, AgentHandle, TurnUsage};

pub use provider_svc::ProviderService;

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
pub trait SessionManager: Send + Sync {
    async fn create_session(&self, workspace: &Path) -> String;
    async fn send_prompt(&self, session_id: &str, text: &str) -> Result<AgentHandle>;
    async fn is_session_active(&self, session_id: &str) -> bool;
    async fn abort(&self, session_id: &str);
    async fn list_sessions(&self) -> Vec<SessionSummary>;
    async fn session_total_usage(&self, session_id: &str) -> TurnUsage;
    async fn session_provider_model(&self, session_id: &str) -> (String, String);
}

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
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

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
}
