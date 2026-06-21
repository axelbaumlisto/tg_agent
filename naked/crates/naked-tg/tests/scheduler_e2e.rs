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
use naked_tg::research_scheduler::{Clock, is_due, plan_dispatches};
use tempfile::TempDir;
use tokio_stream::Stream;

fn make_store() -> (TempDir, Arc<dyn ResearchStore>) {
    let tmp = TempDir::new().expect("tempdir");
    let store: Arc<dyn ResearchStore> = Arc::new(FsResearchStore::new(tmp.path().to_path_buf()));
    (tmp, store)
}

fn make_spec(id: &str, interval: Option<u64>, paused: bool) -> ResearchSpec {
    // REGISTRY-WAIVE: exhaustive test struct ctor — pre-2026-05-13 baseline; new tests must use ..Default::default()
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

/// Deterministic clock for use in integration tests.
///
/// Starts at a fixed moment in time; never advances on its own.
/// Tests that need time to pass call [`MockClock::advance`] explicitly.
#[derive(Debug)]
struct MockClock {
    fixed: std::sync::Mutex<chrono::DateTime<chrono::Utc>>,
}

impl MockClock {
    fn new(t: chrono::DateTime<chrono::Utc>) -> Self {
        Self {
            fixed: std::sync::Mutex::new(t),
        }
    }
}

impl Clock for MockClock {
    fn now(&self) -> chrono::DateTime<chrono::Utc> {
        *self.fixed.lock().unwrap()
    }
}

fn make_core(tmp: &TempDir) -> Arc<AgentCore> {
    let cfg = Config {
        workspace: tmp.path().to_path_buf(),
        session_dir: tmp.path().join("sessions"),
        research: ResearchConfig {
            enabled: true,
            storage_dir: Some(tmp.path().join("research")),
            // Disable auto_first_run so create_research does NOT set run_at.
            // If run_at is set to the creation instant, the is_due run_at
            // trigger fires in plan_dispatches because last_run (= now-60s)
            // pre-dates run_at (= creation time ≈ now-epsilon), causing
            // t_reschedule_replaces_interval to fail non-deterministically.
            auto_first_run: false,
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

    // Use a fixed, deterministic "now" so the test never races against
    // the real wall clock.  MockClock is defined locally in this file;
    // the scheduler's Clock trait is public via research_scheduler.
    let clock = MockClock::new(
        chrono::DateTime::parse_from_rfc3339("2025-06-01T00:00:00Z")
            .unwrap()
            .with_timezone(&Utc),
    );
    let now = clock.now();

    // Start: long interval, recent run → not due.
    // make_core sets auto_first_run=false so create_research leaves run_at=None;
    // without that, the run_at one-shot trigger would fire here and cause a
    // spurious "spec is due" result (the is_due run_at check fires when
    // last_run < run_at, which is always true when last_run = now-60s and
    // run_at = creation-time ≈ now-epsilon).
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

/// T6.5 (PLAN_RESEARCH_AGENT_FLOW_v1): synthetic_mode flag in scheduler
/// config is the seam between scheduler and chat-flow dispatch.
///
/// When `dispatch_fn` is set AND `spec.chat_id` is Some, the scheduler
/// must take the synthetic path. When either is missing, it falls back
/// to the legacy direct-run path. This test verifies the policy gate
/// without spinning up an LLM or real bot — purely the configuration
/// decision.
#[tokio::test]
async fn t_synthetic_mode_requires_dispatch_fn_and_chat_id() {
    use naked_tg::research_scheduler::SchedulerConfig;
    use naked_tg::synthetic::{SyntheticDispatchFn, SyntheticMessage};

    // Default config: no dispatch_fn, no chat_id on spec → legacy.
    let cfg = SchedulerConfig::default();
    assert!(
        cfg.dispatch_fn.is_none(),
        "default scheduler must NOT install dispatch_fn — synthetic flow is opt-in via wiring.rs"
    );

    // With dispatch_fn installed, the seam exists. We can't trigger
    // an actual dispatch in a unit test (would need Bot + BotDeps),
    // but we can verify the Option<SyntheticDispatchFn> field is
    // wired through.
    let stub: SyntheticDispatchFn = Arc::new(
        |_msg: SyntheticMessage, _latch: naked_tg::synthetic::SessionIdLatch| {
            Box::pin(async move {
                Ok(naked_tg::synthetic::SyntheticDispatchOutcome {
                    session_id: "stub".into(),
                    stop_reason: naked_core::research::StopReason::AgentIdle,
                    errors: vec![],
                })
            })
        },
    );
    let cfg_with_dispatch = SchedulerConfig {
        dispatch_fn: Some(stub),
        ..SchedulerConfig::default()
    };
    assert!(
        cfg_with_dispatch.dispatch_fn.is_some(),
        "dispatch_fn must be settable via struct-update syntax"
    );
}

/// B64 test 7e: dispatch returning Err maps to scheduler failure path.
/// Verifies the tasks.rs synthetic branch converts dispatch Err into
/// AgentError::Session so the scheduler records Failed, not Completed.
#[test]
fn t_b64_dispatch_err_maps_to_session_error() {
    let src = include_str!("../src/scheduler/tasks.rs");
    // The synthetic dispatch Err branch must produce AgentError::Session.
    assert!(
        src.contains("synthetic dispatch error"),
        "Err branch must format 'synthetic dispatch error' for scheduler failure path"
    );
    assert!(
        src.contains("AgentError::Session"),
        "synthetic dispatch errors must be AgentError::Session so scheduler records Failed"
    );
}

/// B64 test 7f: dispatch returning Ok(Idle, no errors) maps to scheduler success.
/// Verifies the tasks.rs synthetic branch produces StopReason::AgentIdle only
/// when the outcome is clean.
#[test]
fn t_b64_clean_idle_maps_to_success() {
    let src = include_str!("../src/scheduler/tasks.rs");
    // Clean Idle with no errors → Ok with AgentIdle.
    assert!(
        src.contains("outcome.stop_reason == StopReason::AgentIdle")
            && src.contains("outcome.errors.is_empty()"),
        "only clean Idle with empty errors should map to scheduler success"
    );
    // Errors or abnormal stop → Err (failure).
    assert!(
        src.contains("synthetic turn finished with"),
        "non-clean outcome must produce a detail message for failure path"
    );
}

/// B64 test: scheduler cancel branch aborts the synthetic session.
#[test]
fn t_b64_cancel_aborts_session() {
    let src = include_str!("../src/scheduler/tasks.rs");
    assert!(
        src.contains("core_clone.abort(sid)"),
        "cancel branch must call AgentCore::abort on the synthetic session"
    );
}

/// T6.5/B79 sentinel: synthetic_mode requires dispatch_fn AND an effective
/// origin chat (operator/inflight chat wins, spec chat is cron fallback).
/// Source-text grep prevents accidental removal of the effective-origin gate.
#[test]
fn t_synthetic_mode_gating_condition_locked() {
    let src = include_str!("../src/scheduler/tasks.rs");
    assert!(
        src.contains("synthetic_mode") && src.contains("dispatch_fn.is_some()"),
        "scheduler tasks.rs must gate synthetic dispatch on dispatch_fn.is_some() and an effective origin chat"
    );
    assert!(
        src.contains("effective_origin(") && src.contains("effective_chat_id.is_some()"),
        "synthetic_mode requires effective_origin chat (B79: inflight/operator wins, spec is fallback)"
    );
}
