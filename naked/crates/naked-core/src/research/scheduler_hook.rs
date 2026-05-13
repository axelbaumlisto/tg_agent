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
pub(crate) fn noop_hook() -> Arc<dyn SchedulerHook> {
    Arc::new(NoopSchedulerHook)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn noop_notify_completes() {
        let hook = NoopSchedulerHook;
        let event = SchedulerEvent::SpecCreated {
            spec_id: "test-spec".to_string(),
        };

        // This should not panic and should complete successfully
        hook.notify(event).await;
        // notify() returns (), so we just verify it doesn't panic
    }

    #[tokio::test]
    async fn noop_notify_all_event_types() {
        let hook = NoopSchedulerHook;

        // Test all event types
        hook.notify(SchedulerEvent::SpecCreated {
            spec_id: "test-1".to_string(),
        })
        .await;

        hook.notify(SchedulerEvent::SpecUpdated {
            spec_id: "test-2".to_string(),
        })
        .await;

        hook.notify(SchedulerEvent::SpecRemoved {
            spec_id: "test-3".to_string(),
        })
        .await;

        // All should complete without panic
    }

    #[tokio::test]
    async fn noop_failure_snapshot_returns_none() {
        let hook = NoopSchedulerHook;

        let result = hook.failure_snapshot("some-spec-id").await;
        assert_eq!(result, None);
    }

    #[tokio::test]
    async fn noop_reset_completes() {
        let hook = NoopSchedulerHook;

        // This should not panic
        hook.reset_failures("test-spec-id").await;

        // Call it multiple times to ensure it's idempotent
        hook.reset_failures("test-spec-id").await;
        hook.reset_failures("another-spec").await;
    }

    #[test]
    fn noop_hook_constructor() {
        let hook = noop_hook();

        // Verify it's actually a SchedulerHook
        let _: Arc<dyn SchedulerHook> = hook;
    }

    #[test]
    fn scheduler_event_debug() {
        // Test that SchedulerEvent implements Debug properly
        let event = SchedulerEvent::SpecCreated {
            spec_id: "debug-test".to_string(),
        };
        let debug_str = format!("{:?}", event);
        assert!(debug_str.contains("SpecCreated"));
        assert!(debug_str.contains("debug-test"));
    }

    #[test]
    fn scheduler_event_clone() {
        // Test that SchedulerEvent implements Clone properly
        let event1 = SchedulerEvent::SpecUpdated {
            spec_id: "clone-test".to_string(),
        };
        let event2 = event1.clone();

        match (event1, event2) {
            (
                SchedulerEvent::SpecUpdated { spec_id: id1 },
                SchedulerEvent::SpecUpdated { spec_id: id2 },
            ) => assert_eq!(id1, id2),
            _ => panic!("Events should be identical after clone"),
        }
    }
}
