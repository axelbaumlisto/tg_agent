// REGISTRY-WAIVE: B45 (dead-code revealed by Phase D' pub→pub(crate) flip).
// PLAN_SKILL_VS_CORE_v1 audit found 55 items in research/ never used inside
// the crate; they were hidden by `pub` visibility (dead_code lint exempts
// pub items). Audit + delete is queued as separate B45 cleanup task.
// Until then, this allow keeps the clippy gate green.
#![allow(dead_code)]

//! Research context — ambient state for research runs.

use std::sync::Arc;
use std::sync::RwLock;

#[derive(Clone)]
pub struct ResearchContext {
    inner: Arc<RwLock<Option<String>>>,
    run_id: Arc<RwLock<Option<String>>>,
    /// Optional waterfall sink (installed by `AgentCore` when the
    /// run-event registry is wired up). `None` in unit tests and CLI
    /// runs that don't care about the live TG progress stream.
    run_events: Arc<RwLock<Option<super::run_events::RunEventRegistry>>>,
    /// Successful `research_save` count for the current run. Reset to 0
    /// when the context is (re)bound to a fresh `(spec_id, run_id)` and
    /// inspected by `research_set_target` to refuse a "clear without
    /// saving anything" — the most common idle-failure mode where the
    /// agent visits 30+ pages but never persists a single finding.
    saves: Arc<RwLock<u32>>,
}

impl ResearchContext {
    pub fn new() -> Self {
        Self::default()
    }

    pub(crate) fn set_id(&self, id: Option<String>) {
        if let Ok(mut g) = self.inner.write() {
            *g = id;
        }
    }

    pub fn id(&self) -> Option<String> {
        self.inner.read().ok().and_then(|g| g.clone())
    }

    pub(crate) fn set_run_id(&self, id: Option<String>) {
        if let Ok(mut g) = self.run_id.write() {
            *g = id;
        }
    }

    pub fn run_id(&self) -> Option<String> {
        self.run_id.read().ok().and_then(|g| g.clone())
    }

    /// Attach (or replace) the run-event registry used by tools to
    /// push waterfall events. Cheap — the registry is `Arc`-wrapped
    /// internally.
    pub(crate) fn set_run_events(&self, reg: Option<super::run_events::RunEventRegistry>) {
        if let Ok(mut g) = self.run_events.write() {
            *g = reg;
        }
    }

    pub(crate) fn run_events(&self) -> Option<super::run_events::RunEventRegistry> {
        self.run_events.read().ok().and_then(|g| g.clone())
    }

    /// Increment the per-run save counter. Called by `ResearchSaveTool`
    /// after a successful append (skipped saves do not count). Saturates
    /// at u32::MAX — we only ever check `> 0`.
    pub(crate) fn note_save(&self) {
        if let Ok(mut g) = self.saves.write() {
            *g = g.saturating_add(1);
        }
    }

    /// Number of successful saves recorded for the current bound run.
    pub(crate) fn save_count(&self) -> u32 {
        self.saves.read().map(|g| *g).unwrap_or(0)
    }

    /// Reset the per-run save counter to zero. Called from
    /// `set_id`/`set_run_id`-like rebind paths so a freshly-acquired
    /// context starts at 0 saves regardless of what the previous run
    /// observed.
    pub(crate) fn reset_saves(&self) {
        if let Ok(mut g) = self.saves.write() {
            *g = 0;
        }
    }
}

impl Default for ResearchContext {
    fn default() -> Self {
        Self {
            inner: Arc::new(RwLock::new(None)),
            run_id: Arc::new(RwLock::new(None)),
            run_events: Arc::new(RwLock::new(None)),
            saves: Arc::new(RwLock::new(0)),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn context_new_has_no_id() {
        let ctx = ResearchContext::new();
        assert!(ctx.id().is_none());
        assert!(ctx.run_id().is_none());
    }

    #[test]
    fn context_set_and_get_id() {
        let ctx = ResearchContext::new();
        ctx.set_id(Some("test-123".into()));
        assert_eq!(ctx.id(), Some("test-123".to_string()));
    }

    #[test]
    fn context_save_count_tracks() {
        let ctx = ResearchContext::new();
        assert_eq!(ctx.save_count(), 0);
        ctx.note_save();
        ctx.note_save();
        assert_eq!(ctx.save_count(), 2);
    }

    #[test]
    fn context_reset_saves() {
        let ctx = ResearchContext::new();
        ctx.note_save();
        ctx.reset_saves();
        assert_eq!(ctx.save_count(), 0);
    }

    #[test]
    fn context_clone_shares_state() {
        let ctx = ResearchContext::new();
        let clone = ctx.clone();
        ctx.set_id(Some("shared".into()));
        assert_eq!(clone.id(), Some("shared".to_string()));
    }
}
