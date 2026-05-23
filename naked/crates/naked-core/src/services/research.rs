//! Research subsystem state and service.
//!
//! `ResearchState` owns the research store, context, semaphore, scheduler
//! hook, run-events registry, and cancellation registry. Construction
//! extracted from `AgentCore::new()`.
//!
//! `ResearchService` exposes pure store/registry operations on top of
//! `ResearchState`. AgentCore keeps it in `Arc` and delegates the 15
//! "bucket A" methods (CRUD on specs, scheduler hooks, run-event
//! snapshots, cancel signalling). The 5 coordinator-bound methods
//! (`run_research*`, `ask_research`) stay on `AgentCore` because they
//! need `Arc<Self>` for the runner adapter and a Provider; they reach
//! into the service for the parts that ARE pure (`acquire_permit`,
//! `install_cancel_guard`, `write_research_memory_link`).
//!
//! Plan: naked/docs/PLAN_CORE_HARDENING_v2.md §4 T1.

use std::collections::HashMap;
use std::sync::Arc;

use tokio::sync::RwLock;
use tokio_util::sync::CancellationToken;

use crate::error::{AgentError, Result};
use crate::research::{
    self, ResearchContext, ResearchPatch, ResearchSpec, ResearchStore, RunEventRegistry,
    VerifiedRunReport, apply_research_patch, new_research_id,
};

pub struct ResearchState {
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
    #[allow(dead_code)] // exposed for downstream access via service.state()
    pub fn run_events(&self) -> &crate::research::RunEventRegistry {
        &self.run_events
    }
}

// ---------------------------------------------------------------------------
// ResearchService — pure store / registry operations on top of ResearchState
// ---------------------------------------------------------------------------

/// Service facade over `ResearchState`. Lives behind `Arc` on AgentCore.
///
/// Holds the same `Arc<arc_swap::ArcSwap<Config>>` AgentCore uses, so
/// `reload_config` is observed live (no stale snapshots).
pub struct ResearchService {
    state: Arc<ResearchState>,
    config: Arc<arc_swap::ArcSwap<crate::config::Config>>,
}

impl ResearchService {
    pub fn new(
        state: Arc<ResearchState>,
        config: Arc<arc_swap::ArcSwap<crate::config::Config>>,
    ) -> Self {
        Self { state, config }
    }

    #[allow(dead_code)] // public escape hatch for non-delegate-friendly callers
    pub(crate) fn state(&self) -> &Arc<ResearchState> {
        &self.state
    }

    fn cfg(&self) -> Arc<crate::config::Config> {
        self.config.load_full()
    }

    fn scheduler_hook(&self) -> Arc<dyn research::SchedulerHook> {
        crate::read_or_recover(&self.state.scheduler_hook).clone()
    }

    // ── Bucket A: pure CRUD on specs + scheduler notifications ─────────────────

    pub async fn create_research(
        &self,
        topic: &str,
        sources: Vec<String>,
        session_id: Option<String>,
        chat_id: Option<i64>,
        thread_id: Option<i32>,
    ) -> Result<ResearchSpec> {
        let cfg = self.cfg();
        if !cfg.research.enabled {
            return Err(AgentError::Config("research subsystem is disabled".into()));
        }
        let mut seeds = sources;
        if seeds.is_empty() {
            seeds = cfg.research.default_sources.clone();
        }
        let rcfg = &cfg.research;
        let cron = rcfg.default_cron.clone();
        let interval = if cron.is_some() {
            None
        } else if rcfg.default_interval_seconds > 0 {
            Some(rcfg.default_interval_seconds)
        } else {
            None
        };
        let run_at = if rcfg.auto_first_run {
            Some(chrono::Utc::now())
        } else {
            None
        };

        let spec = ResearchSpec {
            id: new_research_id(topic),
            topic: topic.trim().to_string(),
            sources: seeds,
            interval_seconds: interval,
            run_at,
            cron,
            task_timeout_seconds: None,
            session_id,
            chat_id,
            thread_id,
            provider: rcfg.provider.clone(),
            model: rcfg.model.clone(),
            max_iterations: Some(rcfg.max_iterations),
            max_wall_seconds: Some(rcfg.max_wall_seconds),
            created_at: chrono::Utc::now(),
            paused: false,
            pause_reason: None,
        };
        self.state.store.create_spec(&spec).await?;
        self.scheduler_hook()
            .notify(research::SchedulerEvent::SpecCreated {
                spec_id: spec.id.clone(),
            })
            .await;
        Ok(spec)
    }

    pub async fn list_research(&self) -> Result<Vec<ResearchSpec>> {
        self.state.list_specs().await
    }

    pub async fn load_research(&self, id: &str) -> Result<ResearchSpec> {
        self.state.load_spec(id).await
    }

    pub async fn delete_research(&self, id: &str) -> Result<()> {
        self.state.store.delete_spec(id).await?;
        self.scheduler_hook()
            .notify(research::SchedulerEvent::SpecRemoved {
                spec_id: id.to_string(),
            })
            .await;
        Ok(())
    }

    pub async fn set_research_paused(&self, id: &str, paused: bool) -> Result<()> {
        self.set_research_paused_with_reason(id, paused, None).await
    }

    pub async fn set_research_paused_with_reason(
        &self,
        id: &str,
        paused: bool,
        reason: Option<String>,
    ) -> Result<()> {
        let mut spec = self.state.store.load_spec(id).await?;
        spec.paused = paused;
        spec.pause_reason = if paused { reason } else { None };
        self.state.store.save_spec(&spec).await?;
        self.scheduler_hook()
            .notify(research::SchedulerEvent::SpecUpdated {
                spec_id: id.to_string(),
            })
            .await;
        Ok(())
    }

    pub async fn update_research(&self, id: &str, patch: ResearchPatch) -> Result<ResearchSpec> {
        let mut spec = self.state.store.load_spec(id).await?;
        apply_research_patch(&mut spec, patch);
        self.state.store.save_spec(&spec).await?;
        tracing::info!(
            spec_id = %spec.id,
            topic = %spec.topic,
            interval_seconds = ?spec.interval_seconds,
            sources = spec.sources.len(),
            "research spec updated"
        );
        self.scheduler_hook()
            .notify(research::SchedulerEvent::SpecUpdated {
                spec_id: spec.id.clone(),
            })
            .await;
        Ok(spec)
    }

    // ── Run events / cancellation registry ─────────────────────────────────

    pub fn research_run_events(&self) -> research::RunEventRegistry {
        self.state.run_events.clone()
    }

    pub async fn research_run_events_snapshot(
        &self,
        run_id: &str,
        limit: usize,
    ) -> Vec<research::RunEvent> {
        self.state.run_events.snapshot(run_id, limit).await
    }

    // B4 (PLAN_RESEARCH_FLOW_CLOSURE_v1): cancel_research_run removed.
    // Was deprecated T3.4; all callers use AgentCore::abort(session_id)
    // via ChannelSessionMap now (INV-CANCEL-1, B56).

    pub fn research_run_permits(&self) -> Arc<tokio::sync::Semaphore> {
        self.state.run_semaphore.clone()
    }

    // ── Scheduler hook ─────────────────────────────────────────────────

    pub fn set_scheduler_hook(&self, hook: Arc<dyn research::SchedulerHook>) {
        *crate::write_or_recover(&self.state.scheduler_hook) = hook;
    }

    pub async fn scheduler_failure_snapshot(&self, spec_id: &str) -> Option<(u32, bool)> {
        self.scheduler_hook().failure_snapshot(spec_id).await
    }

    pub async fn reset_research_failures(&self, id: &str) -> Result<()> {
        self.scheduler_hook().reset_failures(id).await;
        let mut spec = self.state.store.load_spec(id).await?;
        let needs_save = spec.paused || spec.pause_reason.is_some();
        spec.paused = false;
        spec.pause_reason = None;
        if needs_save {
            self.state.store.save_spec(&spec).await?;
        }
        self.scheduler_hook()
            .notify(research::SchedulerEvent::SpecUpdated {
                spec_id: id.to_string(),
            })
            .await;
        Ok(())
    }

    pub fn research_store(&self) -> Arc<dyn ResearchStore> {
        self.state.store.clone()
    }

    /// Append a memory-link breadcrumb after a finished research run.
    /// Pure: only needs the store + workspace from config.
    pub async fn write_research_memory_link(
        &self,
        spec_id: &str,
        run_id: &str,
        verified: Option<&VerifiedRunReport>,
    ) -> Result<()> {
        crate::research::runlog::write_research_memory_link_for(
            self.state.store.as_ref(),
            &self.cfg().workspace,
            spec_id,
            run_id,
            verified,
            None,
        )
        .await
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

    fn make_service(tmp: &std::path::Path, enabled: bool) -> ResearchService {
        let store: Arc<dyn ResearchStore> = Arc::new(FsResearchStore::new(tmp.to_path_buf()));
        let state = Arc::new(ResearchState::new(store, 2));
        let mut cfg = crate::config::Config::default();
        cfg.research.enabled = enabled;
        cfg.research.default_sources = vec!["https://seed.example".into()];
        let arc_cfg = Arc::new(arc_swap::ArcSwap::from_pointee(cfg));
        ResearchService::new(state, arc_cfg)
    }

    #[tokio::test]
    async fn create_then_list_returns_spec() {
        let tmp = tempfile::tempdir().unwrap();
        let svc = make_service(tmp.path(), true);
        let spec = svc
            .create_research("da nang rentals", vec![], None, None, None)
            .await
            .unwrap();
        let all = svc.list_research().await.unwrap();
        assert!(all.iter().any(|s| s.id == spec.id));
        assert_eq!(spec.sources, vec!["https://seed.example"]);
    }

    #[tokio::test]
    async fn create_when_disabled_returns_config_error() {
        let tmp = tempfile::tempdir().unwrap();
        let svc = make_service(tmp.path(), false);
        let err = svc
            .create_research("x", vec![], None, None, None)
            .await
            .unwrap_err();
        assert!(matches!(err, AgentError::Config(_)));
    }

    #[tokio::test]
    async fn pause_resume_round_trip_clears_reason() {
        let tmp = tempfile::tempdir().unwrap();
        let svc = make_service(tmp.path(), true);
        let spec = svc
            .create_research("topic", vec!["u".into()], None, None, None)
            .await
            .unwrap();
        svc.set_research_paused_with_reason(&spec.id, true, Some("flake".into()))
            .await
            .unwrap();
        let after_pause = svc.load_research(&spec.id).await.unwrap();
        assert!(after_pause.paused);
        assert_eq!(after_pause.pause_reason.as_deref(), Some("flake"));

        svc.set_research_paused(&spec.id, false).await.unwrap();
        let after_resume = svc.load_research(&spec.id).await.unwrap();
        assert!(!after_resume.paused);
        assert!(
            after_resume.pause_reason.is_none(),
            "reason cleared on resume"
        );
    }

    #[tokio::test]
    async fn delete_makes_subsequent_load_fail() {
        let tmp = tempfile::tempdir().unwrap();
        let svc = make_service(tmp.path(), true);
        let spec = svc
            .create_research("transient", vec!["x".into()], None, None, None)
            .await
            .unwrap();
        svc.delete_research(&spec.id).await.unwrap();
        assert!(svc.load_research(&spec.id).await.is_err());
    }

    // B4 (PLAN_RESEARCH_FLOW_CLOSURE_v1): cancel_run_returns_false_for_unknown
    // and cancel_research_run_carries_deprecated_attribute removed together
    // with the function itself.  INV-CANCEL-1 coverage is now in
    // callbacks.rs::stop_callback_has_no_legacy_cancel_fallback.
}
