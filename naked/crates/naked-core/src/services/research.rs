//! Research subsystem state.
//!
//! Owns the research store, context, semaphore, and scheduler hook.
//! Construction extracted from AgentCore::new().

use std::collections::HashMap;
use std::sync::Arc;

use tokio::sync::RwLock;
use tokio_util::sync::CancellationToken;

use crate::research::{self, ResearchContext, ResearchStore, RunEventRegistry};

pub(crate) struct ResearchState {
    pub store: Arc<dyn ResearchStore>,
    pub context: ResearchContext,
    pub run_semaphore: Arc<tokio::sync::Semaphore>,
    pub run_events: research::RunEventRegistry,
    pub cancels: Arc<RwLock<HashMap<String, CancellationToken>>>,
    pub scheduler_hook: std::sync::RwLock<Arc<dyn research::SchedulerHook>>,
}

impl ResearchState {
    /// Build from config parts. Called by AgentCore::new().
    pub fn new(store: Arc<dyn ResearchStore>, max_concurrent: usize) -> Self {
        Self {
            store,
            context: ResearchContext::new(),
            run_semaphore: Arc::new(tokio::sync::Semaphore::new(max_concurrent)),
            run_events: RunEventRegistry::new(),
            cancels: Arc::new(RwLock::new(HashMap::new())),
            scheduler_hook: std::sync::RwLock::new(research::noop_hook()),
        }
    }
    // ── Pure store delegations (used via AgentCore delegation) ────

    /// List all research specs.
    pub async fn list_specs(&self) -> crate::error::Result<Vec<crate::research::ResearchSpec>> {
        self.store.list_specs().await
    }

    /// Load a single spec by id.
    pub async fn load_spec(&self, id: &str) -> crate::error::Result<crate::research::ResearchSpec> {
        self.store.load_spec(id).await
    }

    /// Delete a spec and all its data.
    /// Reserved for ResearchFacade migration (plan-solid-core-v4).
    #[allow(dead_code)]
    pub async fn delete_spec(&self, id: &str) -> crate::error::Result<()> {
        self.store.delete_spec(id).await
    }

    /// Pause/resume a spec.
    /// Reserved for ResearchFacade migration (plan-solid-core-v4).
    #[allow(dead_code)]
    pub async fn set_paused(&self, id: &str, paused: bool) -> crate::error::Result<()> {
        let mut spec = self.store.load_spec(id).await?;
        spec.paused = paused;
        if !paused {
            spec.pause_reason = None;
        }
        self.store.save_spec(&spec).await
    }

    /// Run events registry.
    pub fn run_events(&self) -> &crate::research::RunEventRegistry {
        &self.run_events
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::research::FsResearchStore;

    #[test]
    fn research_state_constructs() {
        let tmp = tempfile::tempdir().unwrap();
        let store: Arc<dyn ResearchStore> =
            Arc::new(FsResearchStore::new(tmp.path().to_path_buf()));
        let state = ResearchState::new(store, 2);
        assert_eq!(state.run_semaphore.available_permits(), 2);
    }
}
