//! Deterministic integration tests for the in-process research scheduler.
//!
//! These tests exercise the pure scheduling decision (`plan_dispatches`) end
//! to end with a real `FsResearchStore` on disk and the `is_due` predicate
//! that the scheduler loop consults on every tick. They cover the four
//! observable behaviours required by Phase 4 of the research LLM control
//! plane:
//!
//! 1. `t_scheduler_runs_on_interval` — a spec with `interval_seconds=N` and
//!    no prior run is dispatched on the very next scan.
//! 2. `t_scheduler_skips_paused`     — a paused spec is never dispatched,
//!    even past its interval.
//! 3. `t_reschedule_replaces_interval` — shortening the interval via
//!    `update_research` immediately makes a previously-not-yet-due spec due.
//! 4. `t_global_semaphore_serializes_runs` — when more specs are due than
//!    permits available, only `available_permits` of them are dispatched
//!    on a single tick (the rest pick up on the next tick).
//!
//! These tests deliberately do NOT spin up a real LLM-driven research run —
//! the goal is to verify the scheduling decision the loop makes,
//! deterministically and in milliseconds. The dispatch glue itself
//! (semaphore acquire + `tokio::spawn` + write-back to last_runs) is trivial
//! Tokio plumbing exercised by the bot in production.

use std::collections::{HashMap, HashSet};
use std::pin::Pin;
use std::sync::Arc;

use async_trait::async_trait;
use chrono::{Duration as ChronoDuration, Utc};
use naked_core::AgentCore;
use naked_core::PatchField;
use naked_core::ResearchPatch;
use naked_core::config::{Config, ResearchConfig};
use naked_core::provider::{ChatRequest, Provider};
use naked_core::research::{FsResearchStore, ResearchSpec, ResearchStore};
use naked_core::types::{ModelInfo, StreamChunk};
use naked_tg::research_scheduler::{is_due, plan_dispatches};
use tempfile::TempDir;
use tokio_stream::Stream;

fn make_store() -> (TempDir, Arc<dyn ResearchStore>) {
    let tmp = TempDir::new().expect("tempdir");
    let store: Arc<dyn ResearchStore> = Arc::new(FsResearchStore::new(tmp.path().to_path_buf()));
    (tmp, store)
}

fn make_spec(id: &str, interval: Option<u64>, paused: bool) -> ResearchSpec {
    ResearchSpec {
        id: id.into(),
        topic: format!("topic for {id}"),
        sources: vec![],
        interval_seconds: interval,
        run_at: None,
        cron: None,
        task_timeout_seconds: None,
        session_id: None,
        chat_id: None,
        thread_id: None,
        provider: None,
        model: None,
        max_iterations: None,
        max_wall_seconds: None,
        created_at: Utc::now(),
        paused,
        pause_reason: None,
    }
}

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

fn make_core(tmp: &TempDir) -> Arc<AgentCore> {
    let cfg = Config {
        workspace: tmp.path().to_path_buf(),
        session_dir: tmp.path().join("sessions"),
        research: ResearchConfig {
            enabled: true,
            storage_dir: Some(tmp.path().join("research")),
            ..Default::default()
        },
        ..Default::default()
    };
    let agent = Arc::new(AgentCore::new(cfg, Box::new(DummyProvider)));
    agent.init_self_ref();
    agent
}

#[tokio::test]
async fn t_scheduler_runs_on_interval() {
    let (_tmp, store) = make_store();
    let s = make_spec("alpha", Some(60), false);
    store.create_spec(&s).await.expect("create_spec");

    let specs = store.list_specs().await.expect("list_specs");
    let last_runs: HashMap<String, _> = HashMap::new();
    let running: HashSet<String> = HashSet::new();
    let plan = plan_dispatches(&specs, &last_runs, &running, 1, Utc::now());
    assert_eq!(plan, vec!["alpha".to_string()]);

    let now = Utc::now();
    let mut last_runs = HashMap::new();
    last_runs.insert("alpha".to_string(), now - ChronoDuration::seconds(30));
    let plan = plan_dispatches(&specs, &last_runs, &running, 1, now);
    assert!(plan.is_empty(), "30s elapsed < 60s interval, must not fire");

    last_runs.insert("alpha".to_string(), now - ChronoDuration::seconds(61));
    let plan = plan_dispatches(&specs, &last_runs, &running, 1, now);
    assert_eq!(plan, vec!["alpha".to_string()]);
}

#[tokio::test]
async fn t_scheduler_skips_paused() {
    let (_tmp, store) = make_store();
    let s = make_spec("paused-one", Some(30), true);
    store.create_spec(&s).await.expect("create_spec");

    let specs = store.list_specs().await.expect("list_specs");
    let now = Utc::now();

    let last_runs: HashMap<String, _> = HashMap::new();
    let running: HashSet<String> = HashSet::new();
    let plan = plan_dispatches(&specs, &last_runs, &running, 4, now);
    assert!(plan.is_empty(), "paused spec must not be dispatched");

    let mut last_runs = HashMap::new();
    last_runs.insert(
        "paused-one".to_string(),
        now - ChronoDuration::seconds(3_600),
    );
    let plan = plan_dispatches(&specs, &last_runs, &running, 4, now);
    assert!(plan.is_empty(), "paused spec stays out even past interval");

    let paused_spec = make_spec("paused-pred", Some(30), true);
    assert!(!is_due(&paused_spec, None, now));
    assert!(!is_due(
        &paused_spec,
        Some(now - ChronoDuration::seconds(3_600)),
        now
    ));
}

#[tokio::test]
async fn t_reschedule_replaces_interval() {
    let tmp = TempDir::new().expect("tempdir");
    let core = make_core(&tmp);

    // Start: long interval, recent run → not due.
    let spec = core
        .create_research("rolling daily refresh", vec![], None, None, None)
        .await
        .expect("create_research");
    let id = spec.id.clone();
    let patch = ResearchPatch {
        interval_seconds: PatchField::Set(86_400),
        ..Default::default()
    };
    core.update_research(&id, patch)
        .await
        .expect("set initial interval");

    let now = Utc::now();
    let mut last_runs = HashMap::new();
    last_runs.insert(id.clone(), now - ChronoDuration::seconds(60));

    let specs_initial = core
        .research_store()
        .list_specs()
        .await
        .expect("list_specs");
    let running: HashSet<String> = HashSet::new();
    let plan = plan_dispatches(&specs_initial, &last_runs, &running, 1, now);
    assert!(plan.is_empty(), "60s elapsed < 86400s, must not be due");

    // User shortens the interval to 30s via the LLM-facing patch surface.
    let patch = ResearchPatch {
        interval_seconds: PatchField::Set(30),
        ..Default::default()
    };
    core.update_research(&id, patch)
        .await
        .expect("update_research");

    let specs_after = core
        .research_store()
        .list_specs()
        .await
        .expect("list_specs after");
    let plan = plan_dispatches(&specs_after, &last_runs, &running, 1, now);
    assert_eq!(
        plan,
        vec![id.clone()],
        "after rescheduling to 30s, the 60s-old run is past interval"
    );

    // And clearing the schedule entirely (interval_seconds=null) makes it
    // never due again, even with no prior run.
    let patch = ResearchPatch {
        interval_seconds: PatchField::Clear,
        ..Default::default()
    };
    core.update_research(&id, patch)
        .await
        .expect("update_research clear");
    let specs_cleared = core
        .research_store()
        .list_specs()
        .await
        .expect("list_specs cleared");
    let empty: HashMap<String, _> = HashMap::new();
    let plan = plan_dispatches(&specs_cleared, &empty, &running, 1, now);
    assert!(plan.is_empty(), "cleared schedule means never due");
}

#[tokio::test]
async fn t_global_semaphore_serializes_runs() {
    let (_tmp, store) = make_store();
    // Three specs all due simultaneously: same interval, no prior runs.
    for id in ["a", "b", "c"] {
        store
            .create_spec(&make_spec(id, Some(60), false))
            .await
            .expect("create_spec");
    }
    let specs = store.list_specs().await.expect("list_specs");
    let now = Utc::now();
    let last_runs: HashMap<String, _> = HashMap::new();
    let running: HashSet<String> = HashSet::new();

    let plan = plan_dispatches(&specs, &last_runs, &running, 1, now);
    assert_eq!(plan.len(), 1, "one permit → one dispatch");

    let plan = plan_dispatches(&specs, &last_runs, &running, 2, now);
    assert_eq!(plan.len(), 2, "two permits → two dispatches");

    let plan = plan_dispatches(&specs, &last_runs, &running, 0, now);
    assert!(plan.is_empty(), "zero permits → zero dispatches");

    let plan = plan_dispatches(&specs, &last_runs, &running, 99, now);
    assert_eq!(plan.len(), 3, "all due specs fire when permits > demand");
}
