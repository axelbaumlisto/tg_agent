//! Phase 4/5 e2e for the in-process scheduler hook contract.
//!
//! These tests don't run a real scheduler — they just confirm the
//! [`AgentCore`] correctly fires [`SchedulerEvent`]s on every spec mutation
//! path the LLM tools touch (`create_research`, `update_research`,
//! `set_research_paused`, `delete_research`). If any of these stop firing,
//! the bot's scheduler loses sight of changes and reverts to "wait for the
//! next 30s tick" behaviour.

use std::pin::Pin;
use std::sync::{Arc, Mutex};

use async_trait::async_trait;
use naked_core::AgentCore;
use naked_core::CreateSchedule;
use naked_core::PatchField;
use naked_core::ResearchPatch;
use naked_core::config::Config;
use naked_core::provider::{ChatRequest, Provider};
use naked_core::research::{SchedulerEvent, SchedulerHook};
use naked_core::types::{ModelInfo, StreamChunk};
use tempfile::tempdir;
use tokio_stream::Stream;

/// No-op provider — needed because [`AgentCore::new`] requires one even
/// though these tests never trigger an LLM call.
struct DummyProvider;

#[async_trait]
impl Provider for DummyProvider {
    fn name(&self) -> &str {
        "dummy"
    }
    fn models(&self) -> Vec<ModelInfo> {
        vec![]
    }
    async fn stream_chat(
        &self,
        _request: ChatRequest,
    ) -> naked_core::error::Result<Pin<Box<dyn Stream<Item = StreamChunk> + Send>>> {
        Err(naked_core::error::AgentError::Provider(
            "dummy provider invoked in test".into(),
        ))
    }
}

#[derive(Default, Clone)]
struct RecordingHook {
    events: Arc<Mutex<Vec<SchedulerEvent>>>,
}

#[async_trait]
impl SchedulerHook for RecordingHook {
    async fn notify(&self, event: SchedulerEvent) {
        self.events.lock().unwrap().push(event);
    }
}

impl RecordingHook {
    fn snapshot(&self) -> Vec<String> {
        self.events
            .lock()
            .unwrap()
            .iter()
            .map(|e| match e {
                SchedulerEvent::SpecCreated { spec_id } => format!("created:{spec_id}"),
                SchedulerEvent::SpecUpdated { spec_id } => format!("updated:{spec_id}"),
                SchedulerEvent::SpecRemoved { spec_id } => format!("removed:{spec_id}"),
            })
            .collect()
    }
}

fn make_core() -> Arc<AgentCore> {
    let tmp = tempdir().unwrap();
    let cfg = Config {
        workspace: tmp.path().to_path_buf(),
        session_dir: tmp.path().join("sessions"),
        research: naked_core::config::ResearchConfig {
            enabled: true,
            storage_dir: Some(tmp.path().join("research")),
            ..Default::default()
        },
        ..Default::default()
    };
    let agent = Arc::new(AgentCore::new(cfg, Box::new(DummyProvider)));
    agent.init_self_ref();
    // Leak the tempdir for the lifetime of the test — agent owns it
    // logically.
    std::mem::forget(tmp);
    agent
}

#[tokio::test]
async fn scheduler_hook_fires_on_create_update_pause_delete() {
    let agent = make_core();
    let hook = RecordingHook::default();
    agent.set_scheduler_hook(Arc::new(hook.clone()));

    // 1. create
    let spec = agent
        .create_research(
            "hook-test",
            vec!["https://example.com".into()],
            None,
            None,
            None,
            CreateSchedule::OneShotNow,
        )
        .await
        .expect("create");
    let id = spec.id.clone();

    let patch = ResearchPatch {
        sources_add: Some(vec!["https://added.example".into()]),
        interval_seconds: PatchField::Set(3_600),
        ..Default::default()
    };
    agent.update_research(&id, patch).await.expect("update");

    // 3. pause
    agent.set_research_paused(&id, true).await.expect("pause");

    // 4. resume
    agent.set_research_paused(&id, false).await.expect("resume");

    // 5. delete
    agent.delete_research(&id).await.expect("delete");

    let events = hook.snapshot();
    assert_eq!(
        events,
        vec![
            format!("created:{id}"),
            format!("updated:{id}"),
            format!("updated:{id}"), // pause
            format!("updated:{id}"), // resume
            format!("removed:{id}"),
        ],
        "every spec mutation path must call SchedulerHook::notify"
    );
}

#[tokio::test]
async fn set_paused_with_reason_persists_reason_on_pause() {
    let agent = make_core();
    let spec = agent
        .create_research(
            "reason-test",
            vec![],
            None,
            None,
            None,
            CreateSchedule::OneShotNow,
        )
        .await
        .expect("create");
    let id = spec.id.clone();

    agent
        .set_research_paused_with_reason(
            &id,
            true,
            Some("auto: 5 consecutive failures — last error: stream closed".into()),
        )
        .await
        .expect("pause");

    let loaded = agent.load_research(&id).await.expect("load");
    assert!(loaded.paused);
    let reason = loaded.pause_reason.as_deref().unwrap_or("");
    assert!(
        reason.contains("auto: 5 consecutive failures"),
        "pause_reason must round-trip to disk, got {reason:?}"
    );
}

#[tokio::test]
async fn set_paused_with_reason_clears_reason_on_resume() {
    let agent = make_core();
    let spec = agent
        .create_research(
            "clear-reason-test",
            vec![],
            None,
            None,
            None,
            CreateSchedule::OneShotNow,
        )
        .await
        .expect("create");
    let id = spec.id.clone();
    agent
        .set_research_paused_with_reason(&id, true, Some("manual operator pause".into()))
        .await
        .expect("pause");
    // Resume — reason MUST be cleared so a subsequent pause without a
    // reason doesn't inherit the previous one silently.
    agent.set_research_paused(&id, false).await.expect("resume");
    let loaded = agent.load_research(&id).await.expect("load");
    assert!(!loaded.paused);
    assert!(
        loaded.pause_reason.is_none(),
        "resume MUST clear pause_reason"
    );
}

#[tokio::test]
async fn legacy_set_paused_does_not_set_reason() {
    let agent = make_core();
    let spec = agent
        .create_research(
            "legacy-pause",
            vec![],
            None,
            None,
            None,
            CreateSchedule::OneShotNow,
        )
        .await
        .expect("create");
    let id = spec.id.clone();
    agent.set_research_paused(&id, true).await.expect("pause");
    let loaded = agent.load_research(&id).await.expect("load");
    assert!(loaded.paused);
    assert!(
        loaded.pause_reason.is_none(),
        "legacy set_research_paused must leave pause_reason as None"
    );
}

#[tokio::test]
async fn reset_research_failures_clears_pause_and_reason() {
    let agent = make_core();
    let spec = agent
        .create_research(
            "reset-test",
            vec![],
            None,
            None,
            None,
            CreateSchedule::OneShotNow,
        )
        .await
        .expect("create");
    let id = spec.id.clone();
    agent
        .set_research_paused_with_reason(&id, true, Some("auto: 5 failures".into()))
        .await
        .expect("auto-pause");

    agent
        .reset_research_failures(&id)
        .await
        .expect("reset must succeed");

    let loaded = agent.load_research(&id).await.expect("load");
    assert!(!loaded.paused, "reset must resume the spec");
    assert!(
        loaded.pause_reason.is_none(),
        "reset must clear pause_reason"
    );
}

#[tokio::test]
async fn reset_research_failures_invokes_hook_reset_and_notifies_update() {
    use std::sync::atomic::{AtomicUsize, Ordering};

    #[derive(Default)]
    struct CountingHook {
        resets: AtomicUsize,
        updates: AtomicUsize,
    }
    #[async_trait]
    impl SchedulerHook for CountingHook {
        async fn notify(&self, event: SchedulerEvent) {
            if matches!(event, SchedulerEvent::SpecUpdated { .. }) {
                self.updates.fetch_add(1, Ordering::SeqCst);
            }
        }
        async fn reset_failures(&self, _spec_id: &str) {
            self.resets.fetch_add(1, Ordering::SeqCst);
        }
    }

    let agent = make_core();
    let hook = Arc::new(CountingHook::default());
    agent.set_scheduler_hook(hook.clone());

    let spec = agent
        .create_research(
            "reset-hook",
            vec![],
            None,
            None,
            None,
            CreateSchedule::OneShotNow,
        )
        .await
        .expect("create");
    agent
        .reset_research_failures(&spec.id)
        .await
        .expect("reset");

    assert_eq!(hook.resets.load(Ordering::SeqCst), 1);
    assert!(hook.updates.load(Ordering::SeqCst) >= 1);
}

#[tokio::test]
async fn reset_research_failures_is_idempotent_on_healthy_spec() {
    let agent = make_core();
    let spec = agent
        .create_research(
            "healthy",
            vec![],
            None,
            None,
            None,
            CreateSchedule::OneShotNow,
        )
        .await
        .expect("create");
    let id = spec.id.clone();
    agent
        .reset_research_failures(&id)
        .await
        .expect("reset on healthy spec must succeed");
    let loaded = agent.load_research(&id).await.expect("load");
    assert!(!loaded.paused);
    assert!(loaded.pause_reason.is_none());
}

#[tokio::test]
async fn default_scheduler_hook_is_noop_and_safe() {
    // No hook installed = NoopSchedulerHook. Mutations must succeed without
    // panicking and without crashing on the noop notify().
    let agent = make_core();
    let spec = agent
        .create_research(
            "noop-test",
            vec!["https://x".into()],
            None,
            None,
            None,
            CreateSchedule::OneShotNow,
        )
        .await
        .expect("create");
    let id = spec.id.clone();
    agent.set_research_paused(&id, true).await.expect("pause");
    agent.delete_research(&id).await.expect("delete");
}
