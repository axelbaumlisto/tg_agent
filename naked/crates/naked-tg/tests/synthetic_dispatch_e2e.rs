//! End-to-end test: scheduler → MockDispatcher → captured SyntheticMessage.
//!
//! Replaces the include_str! sentinel for T6.3 (PLAN_RESEARCH_AGENT_FLOW_v1)
//! with a real behaviour test. Drives the actual `ResearchScheduler::start`
//! tick loop with a [`MockDispatcher`] plugged into `SchedulerConfig::dispatch_fn`
//! and asserts that a spec whose `chat_id` is set produces a captured
//! synthetic message on the very first tick.
//!
//! NO live Bot, NO live LLM — only the in-process scheduler infrastructure,
//! a `FsResearchStore` in a tempdir, and a `DummyProvider` that errors if
//! ever called (the synthetic path bypasses `run_research_*` entirely).

use std::pin::Pin;
use std::sync::Arc;
use std::sync::atomic::Ordering;
use std::time::Duration;

use async_trait::async_trait;
use chrono::Utc;
use naked_core::AgentCore;
use naked_core::config::{Config, ResearchConfig};
use naked_core::provider::{ChatRequest, Provider};
use naked_core::research::ResearchSpec;
use naked_core::types::{ModelInfo, StreamChunk};
use naked_tg::research_scheduler::{ResearchScheduler, SchedulerConfig};
use naked_tg::synthetic::MockDispatcher;
use tempfile::TempDir;
use tokio_stream::Stream;

// ── Test fixtures ──────────────────────────────────────────────────────

/// Provider that panics if invoked — synthetic path should never reach LLM.
struct UnreachableProvider;

#[async_trait]
impl Provider for UnreachableProvider {
    fn name(&self) -> &str {
        "unreachable"
    }
    fn models(&self) -> Vec<ModelInfo> {
        vec![]
    }
    async fn stream_chat(
        &self,
        _request: ChatRequest,
    ) -> naked_core::error::Result<Pin<Box<dyn Stream<Item = StreamChunk> + Send>>> {
        // If we ever land here, the test took the legacy path instead of
        // synthetic — failure mode worth surfacing loudly.
        Err(naked_core::error::AgentError::Provider(
            "UnreachableProvider invoked — synthetic dispatch path was bypassed".into(),
        ))
    }
}

/// Provider that reports its whole key/provider pool as blacklisted
/// (`blacklisted == total == 2`), so `should_skip_dispatch` returns true.
/// B75 (D-INV-DISPATCH-SKIP-METRIC): drives the REAL scheduler skip path so
/// the dispatch-skip → counter wiring is guarded end-to-end. `stream_chat`
/// errors because it must never be reached on the skip path.
struct ExhaustedProvider;

#[async_trait]
impl Provider for ExhaustedProvider {
    fn name(&self) -> &str {
        "exhausted"
    }
    fn models(&self) -> Vec<ModelInfo> {
        vec![]
    }
    fn blacklisted_key_count(&self) -> usize {
        2
    }
    fn total_key_count(&self) -> usize {
        2
    }
    async fn stream_chat(
        &self,
        _request: ChatRequest,
    ) -> naked_core::error::Result<Pin<Box<dyn Stream<Item = StreamChunk> + Send>>> {
        Err(naked_core::error::AgentError::Provider(
            "ExhaustedProvider::stream_chat invoked — dispatch should have been skipped".into(),
        ))
    }
}

fn make_core(tmp: &TempDir) -> Arc<AgentCore> {
    make_core_with_auto_first_run(tmp, true)
}

/// Build an AgentCore whose default provider reports a fully-exhausted key
/// pool. `auto_first_run = true` so the spec is immediately due on tick 1.
fn make_exhausted_core(tmp: &TempDir) -> Arc<AgentCore> {
    let cfg = Config {
        workspace: tmp.path().to_path_buf(),
        session_dir: tmp.path().join("sessions"),
        research: ResearchConfig {
            enabled: true,
            storage_dir: Some(tmp.path().join("research")),
            auto_first_run: true,
            ..Default::default()
        },
        ..Default::default()
    };
    let agent = Arc::new(AgentCore::new(cfg, Box::new(ExhaustedProvider)));
    agent.init_self_ref();
    agent
}

fn make_core_with_auto_first_run(tmp: &TempDir, auto_first_run: bool) -> Arc<AgentCore> {
    let cfg = Config {
        workspace: tmp.path().to_path_buf(),
        session_dir: tmp.path().join("sessions"),
        research: ResearchConfig {
            enabled: true,
            storage_dir: Some(tmp.path().join("research")),
            // Most tests rely on a freshly-created spec being IMMEDIATELY
            // due on the first scheduler tick. `auto_first_run = true`
            // sets `run_at = now` on create_research, which is the
            // mechanism we want there. Operator-dispatch tests disable it
            // so `dispatch_immediate` is the only trigger under test.
            auto_first_run,
            ..Default::default()
        },
        ..Default::default()
    };
    let agent = Arc::new(AgentCore::new(cfg, Box::new(UnreachableProvider)));
    agent.init_self_ref();
    agent
}

fn make_spec_with_chat(id: &str, chat_id: i64) -> ResearchSpec {
    // (production specs come from store deserialisation; this is an
    // exhaustive shape pin so a renamed field would surface here loudly).
    // REGISTRY-WAIVE: exhaustive ctor — tests want a known shape.
    ResearchSpec {
        id: id.into(),
        topic: format!("topic for {id}"),
        sources: vec![],
        interval_seconds: Some(1),
        run_at: None,
        cron: None,
        task_timeout_seconds: None,
        session_id: None,
        chat_id: Some(chat_id),
        thread_id: None,
        provider: None,
        model: None,
        max_iterations: None,
        max_wall_seconds: None,
        created_at: Utc::now(),
        paused: false,
        pause_reason: None,
    }
}

// ── Tests ──────────────────────────────────────────────────────────────

/// Happy path: spec with `chat_id` + scheduler with `dispatch_fn` (mock)
/// → first scheduler tick captures exactly one synthetic message.
///
/// Verifies the full chain installed by T6.3 + T2.6:
///   SchedulerConfig.dispatch_fn = Some(mock.as_fn())
///   ResearchScheduler::start spawns supervisor → run_loop
///   tick → plan_dispatches yields spec
///   spawn_task gates synthetic_mode = dispatch_fn.is_some() && chat_id.is_some()
///   tokio::select! { dispatch_fut, cancel } → closure invoked
///   MockDispatcher.captured.push(msg)
///   shutdown
///   snapshot reveals captured SyntheticMessage
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn t_scheduler_dispatches_synthetic_message_to_mock() {
    let tmp = TempDir::new().expect("tempdir");
    let agent = make_core(&tmp);

    // Persist the spec with chat_id — auto_first_run = true → run_at = now.
    let spec = make_spec_with_chat("alpha", 555_123);
    agent
        .research_store()
        .create_spec(&spec)
        .await
        .expect("create_spec");

    // Mock dispatcher — installed via SchedulerConfig.
    let mock = MockDispatcher::new();
    let cfg = SchedulerConfig {
        // Fast tick so the test runs in <1s.
        tick_interval: Duration::from_millis(50),
        max_concurrent_runs: 1,
        task_timeout: Duration::from_secs(5),
        dispatch_fn: Some(mock.as_fn()),
        ..Default::default()
    };
    let (sched, hook) = ResearchScheduler::start(Arc::downgrade(&agent), cfg);
    agent.set_scheduler_hook(hook);

    // Give the loop two ticks to fire and the spawn_task future to
    // run the closure body (the mock awaits and pushes synchronously,
    // so 250ms is generous).
    tokio::time::sleep(Duration::from_millis(250)).await;
    sched.shutdown();
    tokio::time::sleep(Duration::from_millis(50)).await;

    let captured = mock.snapshot().await;
    assert!(
        !captured.is_empty(),
        "mock dispatcher must have captured at least one message — \
         scheduler did not invoke the dispatch_fn closure"
    );
    let msg = &captured[0];
    assert_eq!(msg.chat_id, 555_123, "chat_id propagates from spec");
    assert_eq!(msg.text, "/research run alpha");
    assert_eq!(
        msg.source.spec_id(),
        Some("alpha"),
        "SyntheticSource::Scheduler must carry the spec_id"
    );
    assert_eq!(msg.source.label(), "scheduler");
}

/// Negative path: spec WITHOUT `chat_id` (CLI / unattended) does NOT
/// reach the dispatch_fn closure even when one is installed. The legacy
/// run_research_* path takes over — which our UnreachableProvider makes
/// surface as a soft failure (Err logged), but importantly the mock
/// remains empty.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn t_operator_dispatch_uses_inflight_chat_without_mutating_spec() {
    let tmp = TempDir::new().expect("tempdir");
    let agent = make_core_with_auto_first_run(&tmp, false);

    // Spec WITHOUT chat_id: old/CLI specs should remain portable and must not
    // be mutated just because this attempt was launched from a chat.
    let mut spec = make_spec_with_chat("beta-operator", 0);
    spec.chat_id = None;
    spec.thread_id = None;
    spec.interval_seconds = None;
    agent
        .research_store()
        .create_spec(&spec)
        .await
        .expect("create_spec");

    let mock = MockDispatcher::new();
    let cfg = SchedulerConfig {
        tick_interval: Duration::from_secs(60),
        max_concurrent_runs: 1,
        task_timeout: Duration::from_secs(5),
        dispatch_fn: Some(mock.as_fn()),
        ..Default::default()
    };
    let (sched, hook) = ResearchScheduler::start(Arc::downgrade(&agent), cfg);
    agent.set_scheduler_hook(hook.clone());

    sched
        .dispatch_immediate(
            "beta-operator",
            321_654,
            Some(77),
            "research-beta-operator".to_string(),
            "/research run beta-operator".to_string(),
        )
        .await
        .expect("dispatch_immediate");

    tokio::time::sleep(Duration::from_millis(100)).await;
    sched.shutdown();
    tokio::time::sleep(Duration::from_millis(50)).await;

    let captured = mock.snapshot().await;
    assert_eq!(
        captured.len(),
        1,
        "operator dispatch should reach mock once"
    );
    let msg = &captured[0];
    assert_eq!(
        msg.chat_id, 321_654,
        "chat_id comes from Inflight operator context"
    );
    assert_eq!(
        msg.thread_id,
        Some(77),
        "thread_id comes from Inflight operator context"
    );
    assert_eq!(msg.text, "/research run beta-operator");
    assert_eq!(msg.source.spec_id(), Some("beta-operator"));

    let stored = agent
        .research_store()
        .load_spec("beta-operator")
        .await
        .expect("load stored spec");
    assert_eq!(
        stored.chat_id, None,
        "ResearchSpec.chat_id must not be mutated"
    );
    assert_eq!(
        stored.thread_id, None,
        "ResearchSpec.thread_id must not be mutated"
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn t_scheduler_skips_mock_when_chat_id_missing() {
    let tmp = TempDir::new().expect("tempdir");
    let agent = make_core(&tmp);

    // Spec WITHOUT chat_id.
    let mut spec = make_spec_with_chat("beta", 0);
    spec.chat_id = None;
    agent
        .research_store()
        .create_spec(&spec)
        .await
        .expect("create_spec");

    let mock = MockDispatcher::new();
    let cfg = SchedulerConfig {
        tick_interval: Duration::from_millis(50),
        max_concurrent_runs: 1,
        task_timeout: Duration::from_secs(2),
        dispatch_fn: Some(mock.as_fn()),
        ..Default::default()
    };
    let (sched, hook) = ResearchScheduler::start(Arc::downgrade(&agent), cfg);
    agent.set_scheduler_hook(hook);

    tokio::time::sleep(Duration::from_millis(250)).await;
    sched.shutdown();
    tokio::time::sleep(Duration::from_millis(50)).await;

    let captured = mock.snapshot().await;
    assert_eq!(
        captured.len(),
        0,
        "mock must NOT receive messages for specs without chat_id (T6.4 fallback gate)"
    );
}

/// B75 (D-INV-DISPATCH-SKIP-METRIC): the REAL dispatch-skip path must bump
/// `SCHEDULER_DISPATCH_SKIPPED_COUNT` **and** skip the dispatch closure.
///
/// The pre-existing unit tests only exercise `record_scheduler_dispatch_skipped()`
/// and `should_skip_dispatch()` in ISOLATION — neither guards that the
/// `spawn_task` skip branch actually calls the counter. A mutation probe
/// (deleting the `record_scheduler_dispatch_skipped();` call-site) leaves all
/// unit tests green. This end-to-end test closes that gap:
///
///   * default provider reports `blacklisted == total == 2`
///     → `should_skip_dispatch(2, 2) == true`
///   * scheduler tick → `spawn_task` → skip branch fires
///     → counter increments AND the dispatch_fn closure is never invoked.
///
/// We assert the DELTA on the process-wide atomic (>= 1, never an absolute
/// value) because other concurrently-running tests may also bump it, and we
/// assert the MockDispatcher stayed empty — the behavioural fingerprint that
/// fails if the skip branch is bypassed.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn t_scheduler_skip_path_bumps_dispatch_skipped_counter() {
    let tmp = TempDir::new().expect("tempdir");
    let agent = make_exhausted_core(&tmp);

    // Spec WITH chat_id + auto_first_run → due immediately on tick 1.
    let spec = make_spec_with_chat("exhausted-delta", 999_777);
    agent
        .research_store()
        .create_spec(&spec)
        .await
        .expect("create_spec");

    // dispatch_fn installed so the ONLY reason the mock stays empty is the
    // skip branch short-circuiting before the dispatch closure.
    let mock = MockDispatcher::new();
    let cfg = SchedulerConfig {
        tick_interval: Duration::from_millis(50),
        max_concurrent_runs: 1,
        task_timeout: Duration::from_secs(5),
        dispatch_fn: Some(mock.as_fn()),
        ..Default::default()
    };

    // Capture the counter immediately before triggering the run.
    let before = naked_core::types::SCHEDULER_DISPATCH_SKIPPED_COUNT.load(Ordering::Relaxed);

    let (sched, hook) = ResearchScheduler::start(Arc::downgrade(&agent), cfg);
    agent.set_scheduler_hook(hook);

    tokio::time::sleep(Duration::from_millis(250)).await;
    sched.shutdown();
    tokio::time::sleep(Duration::from_millis(50)).await;

    let after = naked_core::types::SCHEDULER_DISPATCH_SKIPPED_COUNT.load(Ordering::Relaxed);
    assert!(
        after.wrapping_sub(before) >= 1,
        "skip path must bump SCHEDULER_DISPATCH_SKIPPED_COUNT at least once \
         (before={before}, after={after}) — the record_scheduler_dispatch_skipped() \
         call-site wiring on the should_skip_dispatch branch is missing"
    );

    let captured = mock.snapshot().await;
    assert!(
        captured.is_empty(),
        "dispatch was skipped → the dispatch_fn closure must never run \
         (mock captured {} message(s))",
        captured.len()
    );
}

/// Negative path 2: spec WITH chat_id but scheduler config has NO
/// dispatch_fn — confirms the symmetric half of the T6.4 gate. The
/// mock can't even be reached because the closure isn't plugged in.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn t_scheduler_skips_synthetic_when_dispatch_fn_missing() {
    let tmp = TempDir::new().expect("tempdir");
    let agent = make_core(&tmp);

    let spec = make_spec_with_chat("gamma", 777_888);
    agent
        .research_store()
        .create_spec(&spec)
        .await
        .expect("create_spec");

    // dispatch_fn=None — same default as before T2.6 was wired.
    let cfg = SchedulerConfig {
        tick_interval: Duration::from_millis(50),
        max_concurrent_runs: 1,
        task_timeout: Duration::from_secs(2),
        dispatch_fn: None,
        ..Default::default()
    };
    let (sched, hook) = ResearchScheduler::start(Arc::downgrade(&agent), cfg);
    agent.set_scheduler_hook(hook);
    // Just verify no panic + scheduler tears down cleanly.
    tokio::time::sleep(Duration::from_millis(150)).await;
    sched.shutdown();
}
