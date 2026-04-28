//! Trait surface a [`crate::AgentCore`] uses to notify a scheduler that
//! research specs were created or mutated. Lives in `naked-core` so the core
//! doesn't depend on the bot crate, but the actual scheduler implementation
//! lives in `naked-tg` (see `naked_tg::research_scheduler`).
//!
//! The [`NoopSchedulerHook`] is the default — used by tests and by the CLI
//! where there is no in-process scheduler.

use std::sync::Arc;

use async_trait::async_trait;

/// Notification kinds the scheduler cares about.
#[derive(Debug, Clone)]
pub enum SchedulerEvent {
    /// A new spec was just created. Scheduler should plan future runs if a
    /// schedule is set.
    SpecCreated { spec_id: String },
    /// An existing spec was mutated (schedule change, pause/resume, source
    /// edits). Scheduler should re-evaluate the next tick for this spec.
    SpecUpdated { spec_id: String },
    /// A spec was deleted. Scheduler should drop any pending tick.
    SpecRemoved { spec_id: String },
}

#[async_trait]
pub trait SchedulerHook: Send + Sync {
    /// Notify the scheduler about a research spec lifecycle event.
    /// Implementations must NOT block; treat this as fire-and-forget.
    async fn notify(&self, event: SchedulerEvent);

    /// Snapshot of the in-memory failure tracker for `spec_id`.
    /// Returns `Some((consecutive_failures, alert_already_fired))`
    /// when the scheduler has any record for the spec, `None` when
    /// the scheduler is not wired (CLI / tests / a freshly-restarted
    /// process that hasn't seen this spec yet).
    ///
    /// Default impl returns `None` so existing hooks (NoopSchedulerHook,
    /// any third-party plug-ins) keep compiling unchanged
    /// (Open/Closed). Production overrides this in
    /// `naked_tg::research_scheduler::ResearchSchedulerHook` to
    /// surface live counters into `/research state <id>`.
    async fn failure_snapshot(&self, _spec_id: &str) -> Option<(u32, bool)> {
        None
    }

    /// Forget the in-memory failure streak / alert flag for `spec_id`.
    /// Called by `/research reset <id>` so an operator can manually
    /// rearm a spec after fixing whatever was breaking it (network,
    /// LLM creds, an upstream site's HTML changing) without waiting
    /// for the next successful run to clear the counters
    /// automatically. Default impl is a no-op so non-scheduler hooks
    /// (CLI, tests) keep compiling.
    async fn reset_failures(&self, _spec_id: &str) {}
}

/// Default no-op hook. Used when no scheduler is wired (CLI, tests).
pub struct NoopSchedulerHook;

#[async_trait]
impl SchedulerHook for NoopSchedulerHook {
    async fn notify(&self, _event: SchedulerEvent) {}
}

/// Convenience constructor for an [`Arc<dyn SchedulerHook>`].
pub fn noop_hook() -> Arc<dyn SchedulerHook> {
    Arc::new(NoopSchedulerHook)
}
