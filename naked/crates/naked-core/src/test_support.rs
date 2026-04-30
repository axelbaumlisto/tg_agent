//! Test harness: minimal AgentCore for unit tests.
//!
//! `TestCore::build()` creates an AgentCore with:
//! - In-memory config (no disk)
//! - NoopProvider (returns empty stream)  
//! - Temp directory for sessions + research
//! - No MCP servers
//!
//! Usage:
//! ```ignore
//! let tc = TestCore::build();
//! let sid = tc.core.create_session(tc.workspace()).await;
//! ```

use std::path::Path;
use std::pin::Pin;
use std::sync::Arc;

use async_trait::async_trait;

use crate::AgentCore;
use crate::config::Config;
use crate::error::Result;
use crate::provider::{ChatRequest, Provider};
use crate::types::{ModelInfo, StreamChunk};

/// Minimal provider that returns an empty stream.
pub struct NoopProvider;

#[async_trait]
impl Provider for NoopProvider {
    fn name(&self) -> &str {
        "noop"
    }
    fn models(&self) -> Vec<ModelInfo> {
        vec![ModelInfo {
            provider: "noop".into(),
            model_id: "noop-model".into(),
            display_name: "Noop".into(),
        }]
    }
    async fn stream_chat(
        &self,
        _request: ChatRequest,
    ) -> Result<Pin<Box<dyn futures_util::Stream<Item = StreamChunk> + Send>>> {
        Ok(Box::pin(futures_util::stream::once(async {
            StreamChunk::Done
        })))
    }
}

/// Test harness wrapping AgentCore with temp directories.
pub struct TestCore {
    pub core: Arc<AgentCore>,
    _tmp: tempfile::TempDir,
}

impl TestCore {
    /// Build a minimal AgentCore for testing.
    #[allow(clippy::field_reassign_with_default)]
    pub fn build() -> Self {
        let tmp = tempfile::tempdir().expect("tempdir");
        let mut config = Config::default();
        config.workspace = tmp.path().join("workspace");
        config.session_dir = tmp.path().join("sessions");
        config.research.storage_dir = Some(tmp.path().join("research"));
        config.default_provider = "noop".into();
        config.default_model = "noop-model".into();

        let core = Arc::new(AgentCore::new(config, Box::new(NoopProvider)));
        core.init_self_ref();

        Self { core, _tmp: tmp }
    }

    pub fn workspace(&self) -> &Path {
        self.core.config().workspace.as_path()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_core_creates_session() {
        let tc = TestCore::build();
        let sid = tc.core.create_session(tc.workspace()).await;
        assert!(!sid.is_empty());
    }

    #[tokio::test]
    async fn test_core_list_sessions() {
        let tc = TestCore::build();
        let sid = tc.core.create_session(tc.workspace()).await;
        let sessions = tc.core.list_sessions().await;
        assert!(sessions.iter().any(|s| s.id == sid));
    }

    #[tokio::test]
    async fn test_core_abort_session() {
        let tc = TestCore::build();
        let sid = tc.core.create_session(tc.workspace()).await;
        tc.core.abort(&sid).await;
        assert!(!tc.core.is_session_active(&sid).await);
    }

    #[tokio::test]
    async fn test_core_compact_empty_session() {
        let tc = TestCore::build();
        let sid = tc.core.create_session(tc.workspace()).await;
        let result = tc.core.compact_session(&sid).await;
        assert!(result.is_none(), "empty session should not compact");
    }

    #[tokio::test]
    async fn test_core_fork_session() {
        let tc = TestCore::build();
        let sid = tc.core.create_session(tc.workspace()).await;
        let forked = tc.core.fork_session(&sid, Some("test-branch".into())).await;
        assert!(forked.is_ok());
        let fid = forked.unwrap();
        assert_ne!(sid, fid);
    }

    #[tokio::test]
    async fn test_core_research_create() {
        let tc = TestCore::build();
        let spec = tc
            .core
            .create_research("test topic", vec![], None, None, None)
            .await
            .unwrap();
        assert!(spec.id.contains("test-topic"));
        assert_eq!(spec.topic, "test topic");
    }

    #[tokio::test]
    async fn test_core_research_list() {
        let tc = TestCore::build();
        tc.core
            .create_research("topic a", vec![], None, None, None)
            .await
            .unwrap();
        tc.core
            .create_research("topic b", vec![], None, None, None)
            .await
            .unwrap();
        let list = tc.core.list_research().await.unwrap();
        assert_eq!(list.len(), 2);
    }

    #[tokio::test]
    async fn test_core_provider_model() {
        let tc = TestCore::build();
        let (prov, model) = tc.core.default_provider_model();
        assert_eq!(prov, "noop");
        assert_eq!(model, "noop-model");
    }
}
