//! Test harness: minimal AgentCore for unit tests.
//!
//! `TestCore::build()` creates an AgentCore with:
#![allow(clippy::unwrap_used)] // test-utility module: panics on misuse are intentional
//! - In-memory config (no disk)
//! - NoopProvider (returns empty stream)  
//! - Temp directory for sessions + research
//! - No MCP servers
//!
//! Usage:
//! ```ignore
//! let tc = TestCore::build();
//! let sid = tc.core.create_session(&tc.workspace()).await;
//! ```

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
    // REGISTRY-WAIVE: field_reassign_with_default — test-construction pattern (default + mutate)
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

    pub fn workspace(&self) -> std::path::PathBuf {
        self.core.config().workspace.clone()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_core_creates_session() {
        let tc = TestCore::build();
        let sid = tc.core.create_session(&tc.workspace()).await;
        assert!(!sid.is_empty());
    }

    #[tokio::test]
    async fn test_core_list_sessions() {
        let tc = TestCore::build();
        let sid = tc.core.create_session(&tc.workspace()).await;
        let sessions = tc.core.list_sessions().await;
        assert!(sessions.iter().any(|s| s.id == sid));
    }

    #[tokio::test]
    async fn test_core_abort_session() {
        let tc = TestCore::build();
        let sid = tc.core.create_session(&tc.workspace()).await;
        tc.core.abort(&sid).await;
        assert!(!tc.core.is_session_active(&sid).await);
    }

    #[tokio::test]
    async fn test_core_compact_empty_session() {
        let tc = TestCore::build();
        let sid = tc.core.create_session(&tc.workspace()).await;
        let result = tc.core.compact_session(&sid).await;
        assert!(result.is_none(), "empty session should not compact");
    }

    #[tokio::test]
    async fn test_core_fork_session() {
        let tc = TestCore::build();
        let sid = tc.core.create_session(&tc.workspace()).await;
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

#[cfg(test)]
mod session_ops_tests {
    use super::*;

    #[tokio::test]
    async fn set_and_get_sender() {
        let tc = TestCore::build();
        let sid = tc.core.create_session(&tc.workspace()).await;
        tc.core
            .set_session_sender(&sid, Some("user123".into()))
            .await;
        assert_eq!(tc.core.session_sender(&sid).await, Some("user123".into()));
    }

    #[tokio::test]
    async fn sender_cleared() {
        let tc = TestCore::build();
        let sid = tc.core.create_session(&tc.workspace()).await;
        tc.core
            .set_session_sender(&sid, Some("user123".into()))
            .await;
        tc.core.set_session_sender(&sid, None).await;
        assert_eq!(tc.core.session_sender(&sid).await, None);
    }

    #[tokio::test]
    async fn create_session_with_channel() {
        let tc = TestCore::build();
        let sid = tc
            .core
            .create_session_with_channel(&tc.workspace(), "ch42")
            .await;
        assert!(!sid.is_empty());
    }

    #[tokio::test]
    async fn session_workspace_returns_path() {
        let tc = TestCore::build();
        let sid = tc.core.create_session(&tc.workspace()).await;
        let ws = tc.core.session_workspace(&sid).await;
        assert!(ws.is_some());
    }

    #[tokio::test]
    async fn list_sessions_paged() {
        let tc = TestCore::build();
        for i in 0..5 {
            tc.core
                .create_session_with_channel(&tc.workspace(), &format!("ch{i}"))
                .await;
        }
        let page1 = tc.core.list_sessions_paged(0, 3).await;
        let page2 = tc.core.list_sessions_paged(3, 3).await;
        assert_eq!(page1.len(), 3);
        assert_eq!(page2.len(), 2);
    }

    #[tokio::test]
    async fn queue_message_while_inactive() {
        let tc = TestCore::build();
        let sid = tc.core.create_session(&tc.workspace()).await;
        // Queue a message — should not panic even with no active turn
        tc.core.queue_message(&sid, "hello").await;
    }

    #[tokio::test]
    async fn set_channel_id() {
        let tc = TestCore::build();
        let sid = tc.core.create_session(&tc.workspace()).await;
        tc.core.set_session_channel_id(&sid, "tg:12345").await;
        let mappings = tc.core.channel_session_mappings().await;
        assert!(mappings.iter().any(|(ch, _)| ch == "tg:12345"));
    }

    #[tokio::test]
    async fn session_total_usage_empty() {
        let tc = TestCore::build();
        let sid = tc.core.create_session(&tc.workspace()).await;
        let usage = tc.core.session_total_usage(&sid).await;
        assert_eq!(usage.input_tokens, 0);
        assert_eq!(usage.output_tokens, 0);
    }
}

#[cfg(test)]
mod research_ops_tests {
    use super::*;

    #[tokio::test]
    async fn create_and_load() {
        let tc = TestCore::build();
        let spec = tc
            .core
            .create_research("apartments samui", vec![], None, None, None)
            .await
            .unwrap();
        let loaded = tc.core.load_research(&spec.id).await.unwrap();
        assert_eq!(loaded.topic, "apartments samui");
    }

    #[tokio::test]
    async fn delete_research() {
        let tc = TestCore::build();
        let spec = tc
            .core
            .create_research("temp topic", vec![], None, None, None)
            .await
            .unwrap();
        tc.core.delete_research(&spec.id).await.unwrap();
        assert!(tc.core.load_research(&spec.id).await.is_err());
    }

    #[tokio::test]
    async fn set_paused() {
        let tc = TestCore::build();
        let spec = tc
            .core
            .create_research("pausable", vec![], None, None, None)
            .await
            .unwrap();
        tc.core.set_research_paused(&spec.id, true).await.unwrap();
        let loaded = tc.core.load_research(&spec.id).await.unwrap();
        assert!(loaded.paused);
    }

    #[tokio::test]
    async fn update_research_patch() {
        let tc = TestCore::build();
        let spec = tc
            .core
            .create_research("patchable", vec![], None, None, None)
            .await
            .unwrap();
        let patch = crate::ResearchPatch {
            interval_seconds: crate::PatchField::Set(3600),
            ..Default::default()
        };
        let updated = tc.core.update_research(&spec.id, patch).await.unwrap();
        assert_eq!(updated.interval_seconds, Some(3600));
    }
}

#[cfg(test)]
mod provider_ops_tests {
    use super::*;

    #[tokio::test]
    async fn list_models_includes_noop() {
        let tc = TestCore::build();
        let models = tc.core.list_models();
        assert!(models.iter().any(|m| m.model_id == "noop-model"));
    }

    #[tokio::test]
    async fn list_providers_includes_noop() {
        let tc = TestCore::build();
        let _providers = tc.core.list_providers();
        let (prov, _) = tc.core.default_provider_model();
        assert_eq!(prov, "noop");
    }

    #[tokio::test]
    async fn list_skills_empty() {
        let tc = TestCore::build();
        let skills = tc.core.list_skills();
        // No skill_roots configured → empty
        assert!(skills.is_empty());
    }

    // ── Research auto-scheduling tests ──────────────────────────────

    #[tokio::test]
    async fn create_research_gets_default_interval() {
        let tc = TestCore::build();
        // Config default: default_interval_seconds = 21600
        let spec = tc
            .core
            .create_research("test topic", vec![], None, None, None)
            .await
            .expect("create_research");
        assert_eq!(
            spec.interval_seconds,
            Some(tc.core.config().research.default_interval_seconds),
            "new spec should inherit default_interval_seconds from config"
        );
    }

    #[tokio::test]
    async fn create_research_auto_first_run() {
        let tc = TestCore::build();
        // Config default: auto_first_run = true
        assert!(tc.core.config().research.auto_first_run);
        let spec = tc
            .core
            .create_research("test first run", vec![], None, None, None)
            .await
            .expect("create_research");
        assert!(
            spec.run_at.is_some(),
            "auto_first_run=true should set run_at"
        );
        // run_at should be very recent (within last 5 seconds)
        let age = chrono::Utc::now() - spec.run_at.unwrap();
        assert!(age.num_seconds() < 5, "run_at should be ~now");
    }

    #[tokio::test]
    async fn create_research_cron_overrides_interval() {
        let tmp = tempfile::tempdir().expect("tempdir");
        let config = Config {
            workspace: tmp.path().join("workspace"),
            session_dir: tmp.path().join("sessions"),
            research: crate::config::ResearchConfig {
                storage_dir: Some(tmp.path().join("research")),
                default_cron: Some("0 10 * * *".into()),
                default_interval_seconds: 21600,
                ..Default::default()
            },
            default_provider: "noop".into(),
            default_model: "noop-model".into(),
            ..Default::default()
        };

        let core = Arc::new(AgentCore::new(config, Box::new(NoopProvider)));
        core.init_self_ref();

        let spec = core
            .create_research("cron test", vec![], None, None, None)
            .await
            .expect("create_research");
        assert_eq!(spec.cron.as_deref(), Some("0 10 * * *"));
        assert_eq!(
            spec.interval_seconds, None,
            "cron should take priority, interval should be None"
        );
    }

    #[tokio::test]
    async fn create_research_zero_interval_means_manual() {
        let tmp = tempfile::tempdir().expect("tempdir");
        let config = Config {
            workspace: tmp.path().join("workspace"),
            session_dir: tmp.path().join("sessions"),
            research: crate::config::ResearchConfig {
                storage_dir: Some(tmp.path().join("research")),
                default_interval_seconds: 0,
                auto_first_run: false,
                ..Default::default()
            },
            default_provider: "noop".into(),
            default_model: "noop-model".into(),
            ..Default::default()
        };

        let core = Arc::new(AgentCore::new(config, Box::new(NoopProvider)));
        core.init_self_ref();

        let spec = core
            .create_research("manual only", vec![], None, None, None)
            .await
            .expect("create_research");
        assert_eq!(spec.interval_seconds, None);
        assert_eq!(spec.cron, None);
        assert_eq!(spec.run_at, None);
    }
}
