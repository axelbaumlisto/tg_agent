//! End-to-end tests for scheduler boot-pass hygiene.
//!
//! The scheduler's `run_loop` performs three boot-time cleanup
//! steps before the first tick:
//!
//!   1. Phase 3.5 — sweep stale `*.tmp` orphans from interrupted
//!      atomic writes at BOTH the research root and each spec
//!      subdirectory.
//!   2. Phase 1.3 — resurrect non-terminal `inflight.json`
//!      records via the pull model.
//!   3. Phase 3.1 — purge terminal `inflight.json` records older
//!      than `inflight_terminal_retention`.
//!
//! These unit-tested helpers are straightforward, but the
//! composed boot sequence — with a real `AgentCore`, real
//! `FsResearchStore`, and a live `ResearchScheduler` running in a
//! tokio task — is what the bot actually runs on `systemctl
//! start`. This file proves that wiring-up, start-up, and
//! shutdown work end-to-end without reading into the private
//! `purge_stale_tmp_files` helper.

use std::pin::Pin;
use std::sync::Arc;
use std::time::Duration;

use async_trait::async_trait;
use chrono::Utc;
use naked_core::AgentCore;
use naked_core::config::{Config, ResearchConfig};
use naked_core::provider::{ChatRequest, Provider};
use naked_core::research::{FsResearchStore, Inflight, ResearchSpec, ResearchStore};
use naked_core::types::{ModelInfo, StreamChunk};
use naked_tg::research_scheduler::{ResearchScheduler, SchedulerConfig};
use tempfile::TempDir;
use tokio_stream::Stream;

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

fn make_core(research_root: std::path::PathBuf) -> (TempDir, Arc<AgentCore>) {
    let tmp = TempDir::new().expect("tempdir");
    let cfg = Config {
        workspace: tmp.path().to_path_buf(),
        session_dir: tmp.path().join("sessions"),
        research: ResearchConfig {
            enabled: true,
            storage_dir: Some(research_root),
            ..Default::default()
        },
        ..Default::default()
    };
    let agent = Arc::new(AgentCore::new(cfg, Box::new(DummyProvider)));
    agent.init_self_ref();
    (tmp, agent)
}

fn make_spec(id: &str) -> ResearchSpec {
    // REGISTRY-WAIVE: exhaustive test struct ctor — pre-2026-05-13 baseline; new tests must use ..Default::default()
    ResearchSpec {
        id: id.into(),
        topic: format!("boot-test topic {id}"),
        sources: vec![],
        interval_seconds: None, // never scheduled, boot-pass only
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
        paused: false,
        pause_reason: None,
    }
}

/// Poll `predicate` every 25ms until it returns true or `deadline`
/// (in wall-clock ms from now) expires.
async fn wait_until<F: Fn() -> bool>(deadline_ms: u64, predicate: F) -> bool {
    let start = std::time::Instant::now();
    while start.elapsed().as_millis() < deadline_ms as u128 {
        if predicate() {
            return true;
        }
        tokio::time::sleep(Duration::from_millis(25)).await;
    }
    predicate()
}

#[tokio::test]
async fn scheduler_boot_purges_stale_tmp_files_at_root_and_spec_level() {
    // Seed a research root that looks like we crashed mid-atomic
    // write: `*.tmp` orphans at the root AND inside a spec dir.
    // Phase 3.5 says `purge_stale_tmp_files` must sweep both.
    let research_root = TempDir::new().expect("research root");
    let root_path = research_root.path().to_path_buf();

    // Pre-create a spec on disk so the per-spec sweep has
    // something to walk.
    let store: Arc<dyn ResearchStore> = Arc::new(FsResearchStore::new(root_path.clone()));
    let spec = make_spec("boot-purge");
    store.create_spec(&spec).await.unwrap();

    let root_orphan = root_path.join("summary.json.tmp");
    let spec_orphan = root_path.join(&spec.id).join("inflight.json.tmp");
    let sentinel = root_path.join("summary.json");
    tokio::fs::write(&root_orphan, b"half-written-root")
        .await
        .unwrap();
    tokio::fs::write(&spec_orphan, b"half-written-spec")
        .await
        .unwrap();
    tokio::fs::write(&sentinel, b"{}").await.unwrap();

    let (_workspace, agent) = make_core(root_path.clone());

    // Fast tick so we don't wait long, but we only care about the
    // one-shot boot pass anyway.
    let sched_cfg = SchedulerConfig {
        tick_interval: Duration::from_millis(50),
        ..SchedulerConfig::default()
    };
    let (sched, _hook) = ResearchScheduler::start(Arc::downgrade(&agent), sched_cfg);

    // Wait for the boot pass to complete: both orphans gone, the
    // sentinel still alive.
    let all_cleaned = wait_until(2_000, || {
        !root_orphan.exists() && !spec_orphan.exists() && sentinel.exists()
    })
    .await;
    sched.shutdown();
    assert!(
        all_cleaned,
        "boot pass must sweep *.tmp orphans at root and spec level within 2s; \
         root_orphan.exists()={}, spec_orphan.exists()={}, sentinel.exists()={}",
        root_orphan.exists(),
        spec_orphan.exists(),
        sentinel.exists()
    );
}

#[tokio::test]
async fn scheduler_boot_purges_old_terminal_inflight_records() {
    // Plant an old `Failed` inflight record (finished 30 days
    // ago) and a recent one (finished 1h ago) in two different
    // specs. With `inflight_terminal_retention = 7 days`, only
    // the old one should be purged during the boot pass.
    let research_root = TempDir::new().expect("research root");
    let root_path = research_root.path().to_path_buf();
    let store: Arc<dyn ResearchStore> = Arc::new(FsResearchStore::new(root_path.clone()));

    let old = make_spec("old-terminal");
    let recent = make_spec("recent-terminal");
    store.create_spec(&old).await.unwrap();
    store.create_spec(&recent).await.unwrap();

    let now = Utc::now();
    let mut old_inflight = Inflight::scheduled(&old.id, 1);
    old_inflight.mark_failed("old");
    old_inflight.finished_at = Some(now - chrono::Duration::days(30));

    let mut recent_inflight = Inflight::scheduled(&recent.id, 1);
    recent_inflight.mark_failed("recent");
    recent_inflight.finished_at = Some(now - chrono::Duration::hours(1));

    store.save_inflight(&old.id, &old_inflight).await.unwrap();
    store
        .save_inflight(&recent.id, &recent_inflight)
        .await
        .unwrap();

    let (_workspace, agent) = make_core(root_path.clone());
    let sched_cfg = SchedulerConfig {
        tick_interval: Duration::from_millis(50),
        inflight_terminal_retention: Duration::from_secs(7 * 24 * 60 * 60),
        inflight_purge_interval: Duration::from_secs(60 * 60),
        ..SchedulerConfig::default()
    };
    let (sched, _hook) = ResearchScheduler::start(Arc::downgrade(&agent), sched_cfg);

    let old_path = root_path.join(&old.id).join("inflight.json");
    let recent_path = root_path.join(&recent.id).join("inflight.json");
    let purged = wait_until(2_000, || !old_path.exists() && recent_path.exists()).await;
    sched.shutdown();
    assert!(
        purged,
        "old terminal inflight must be purged, recent kept; \
         old.exists()={}, recent.exists()={}",
        old_path.exists(),
        recent_path.exists()
    );
}

#[tokio::test]
async fn scheduler_shutdown_is_prompt_and_idempotent() {
    // Sanity check the whole graceful-shutdown path used by
    // `main.rs` on SIGINT: `shutdown()` is non-blocking, can be
    // called multiple times, and the scheduler releases its
    // internal tokio task promptly.
    let research_root = TempDir::new().expect("research root");
    let root_path = research_root.path().to_path_buf();
    let (_workspace, agent) = make_core(root_path);
    let sched_cfg = SchedulerConfig {
        tick_interval: Duration::from_millis(50),
        ..SchedulerConfig::default()
    };
    let (sched, _hook) = ResearchScheduler::start(Arc::downgrade(&agent), sched_cfg);

    // Let it reach steady-state.
    tokio::time::sleep(Duration::from_millis(100)).await;

    // Shutdown twice — MUST NOT panic or hang.
    sched.shutdown();
    sched.shutdown();

    // Poke after shutdown — MUST be a no-op.
    sched.poke();
}
