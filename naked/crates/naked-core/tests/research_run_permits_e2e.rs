//! Phase-8 e2e for [`AgentCore::research_run_permits`] — the global
//! per-process semaphore that serialises every research run regardless of
//! origin (TG, scheduler, LLM tool, CLI).
//!
//! These tests don't actually invoke the coordinator (no LLM, no Playwright);
//! they exercise the semaphore contract directly: the AgentCore exposes a
//! single permit pool that all callers share, default permit count comes from
//! `Config.research.max_concurrent_runs`, and the helper
//! `acquire_research_permit` blocks only when the pool is exhausted.

use std::pin::Pin;
use std::sync::Arc;
use std::time::Duration;

use async_trait::async_trait;
use naked_core::AgentCore;
use naked_core::acquire_research_permit;
use naked_core::config::Config;
use naked_core::provider::{ChatRequest, Provider};
use naked_core::types::{ModelInfo, StreamChunk};
use tempfile::tempdir;
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
            "dummy provider should not be invoked in permit tests".into(),
        ))
    }
}

fn make_core_with_permits(permits: usize) -> Arc<AgentCore> {
    let tmp = tempdir().unwrap();
    let cfg = Config {
        workspace: tmp.path().to_path_buf(),
        session_dir: tmp.path().join("sessions"),
        research: naked_core::config::ResearchConfig {
            enabled: true,
            storage_dir: Some(tmp.path().join("research")),
            max_concurrent_runs: permits,
            ..Default::default()
        },
        ..Default::default()
    };
    let agent = Arc::new(AgentCore::new(cfg, Box::new(DummyProvider)));
    agent.init_self_ref();
    std::mem::forget(tmp);
    agent
}

#[tokio::test]
async fn permit_pool_size_follows_config() {
    let agent = make_core_with_permits(3);
    assert_eq!(agent.research_run_permits().available_permits(), 3);
}

#[tokio::test]
async fn zero_permits_in_config_clamps_to_one() {
    // We never want a configuration mistake to produce a 0-permit pool: the
    // first run would block forever and there'd be no operator-visible error.
    let agent = make_core_with_permits(0);
    assert_eq!(
        agent.research_run_permits().available_permits(),
        1,
        "0 permits in config must be clamped to 1"
    );
}

#[tokio::test]
async fn second_acquire_blocks_until_first_released() {
    let agent = make_core_with_permits(1);
    let sem = agent.research_run_permits();

    let permit1 = acquire_research_permit(&sem, "spec-a")
        .await
        .expect("first permit");
    assert_eq!(sem.available_permits(), 0, "first acquire used the slot");

    // Second concurrent acquire MUST block until permit1 is dropped.
    let sem2 = sem.clone();
    let blocked = tokio::spawn(async move {
        let _p = acquire_research_permit(&sem2, "spec-b")
            .await
            .expect("second permit");
        Utc::now()
    });

    // Give the spawned task a chance to start waiting.
    tokio::time::sleep(Duration::from_millis(50)).await;
    assert!(!blocked.is_finished(), "second acquire must be blocked");

    let release_at = Utc::now();
    drop(permit1);

    let acquired_at = blocked.await.expect("task panicked");
    let waited = (acquired_at - release_at).num_milliseconds();
    assert!(
        (0..1000).contains(&waited),
        "second acquire must complete shortly after release; waited {waited}ms"
    );
}

#[tokio::test]
async fn three_permits_admit_three_simultaneous_holders() {
    let agent = make_core_with_permits(3);
    let sem = agent.research_run_permits();

    let p1 = acquire_research_permit(&sem, "a").await.unwrap();
    let p2 = acquire_research_permit(&sem, "b").await.unwrap();
    let p3 = acquire_research_permit(&sem, "c").await.unwrap();
    assert_eq!(sem.available_permits(), 0, "all three slots used");

    // Fourth must block.
    let sem2 = sem.clone();
    let blocked = tokio::spawn(async move {
        let _p = acquire_research_permit(&sem2, "d").await.unwrap();
    });
    tokio::time::sleep(Duration::from_millis(50)).await;
    assert!(!blocked.is_finished(), "fourth acquire must block");

    drop(p1);
    blocked.await.expect("released task should complete");
    drop(p2);
    drop(p3);
    assert_eq!(sem.available_permits(), 3);
}

#[tokio::test]
async fn run_research_returns_disabled_error_when_subsystem_off() {
    // Sanity check: even with permits available, run_research must fail fast
    // when research is disabled — no permit is leaked.
    let tmp = tempdir().unwrap();
    let cfg = Config {
        workspace: tmp.path().to_path_buf(),
        session_dir: tmp.path().join("sessions"),
        research: naked_core::config::ResearchConfig {
            enabled: false,
            ..Default::default()
        },
        ..Default::default()
    };
    let agent = Arc::new(AgentCore::new(cfg, Box::new(DummyProvider)));
    agent.init_self_ref();
    let before = agent.research_run_permits().available_permits();
    let err = agent
        .clone()
        .run_research("nope")
        .await
        .expect_err("must fail when disabled");
    assert!(format!("{err}").contains("disabled"));
    assert_eq!(
        agent.research_run_permits().available_permits(),
        before,
        "no permit should be acquired when research is disabled"
    );
    std::mem::forget(tmp);
}

use chrono::Utc;
