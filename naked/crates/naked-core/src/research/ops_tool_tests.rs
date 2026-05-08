//! Integration-style tests for the research-ops tool surface. These
//! pin the system-level safety guards we rely on to keep the agent
//! honest — most importantly: `research_set_target` must refuse a
//! "clear without saving anything" cleanup so we never again ship a
//! run where the model burned 30 tool calls and produced 0 findings.
use super::*;
use crate::research::store_fs::FsResearchStore;
use chrono::Utc;
use tempfile::tempdir;

fn make_spec(id: &str) -> super::super::spec::ResearchSpec {
    super::super::spec::ResearchSpec {
        id: id.into(),
        topic: "topic".into(),
        sources: vec![],
        interval_seconds: None,
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

fn cwd() -> std::path::PathBuf {
    std::env::current_dir().unwrap()
}

async fn setup() -> (
    tempfile::TempDir,
    std::sync::Arc<dyn ResearchStore>,
    ResearchContext,
) {
    let tmp = tempdir().unwrap();
    let store: std::sync::Arc<dyn ResearchStore> =
        std::sync::Arc::new(FsResearchStore::new(tmp.path().to_path_buf()));
    let ctx = ResearchContext::new();
    (tmp, store, ctx)
}

#[tokio::test]
async fn set_target_bind_resets_save_count_and_returns_active() {
    let (_tmp, store, ctx) = setup().await;
    store.create_spec(&make_spec("s1")).await.unwrap();
    // Pretend a previous run had saves; rebinding must reset.
    ctx.note_save();
    ctx.note_save();
    assert_eq!(ctx.save_count(), 2);

    let tool = ResearchSetTargetTool::new(store.clone(), ctx.clone());
    let out = tool
        .execute(serde_json::json!({"spec_id":"s1"}), &cwd())
        .await;
    assert!(!out.is_error, "got: {}", out.output);
    assert_eq!(
        ctx.save_count(),
        0,
        "fresh bind must reset the save counter so the clear-guard \
         starts from zero for this run"
    );
    assert_eq!(ctx.id().as_deref(), Some("s1"));
}

#[tokio::test]
async fn set_target_clear_with_zero_saves_is_rejected() {
    let (_tmp, store, ctx) = setup().await;
    store.create_spec(&make_spec("s1")).await.unwrap();
    ctx.set_id(Some("s1".into()));
    ctx.set_run_id(Some("run1".into()));
    ctx.reset_saves();

    let tool = ResearchSetTargetTool::new(store.clone(), ctx.clone());
    let out = tool
        .execute(serde_json::json!({"spec_id":""}), &cwd())
        .await;
    assert!(
        out.is_error,
        "clear with 0 saves must error, but got success: {}",
        out.output
    );
    assert!(
        out.output.contains("research_save"),
        "error must point the agent at the save tool: {}",
        out.output
    );
    // Context must remain bound so the agent can recover.
    assert_eq!(ctx.id().as_deref(), Some("s1"));
    assert_eq!(ctx.run_id().as_deref(), Some("run1"));
}

#[tokio::test]
async fn set_target_clear_with_force_succeeds_even_on_zero_saves() {
    let (_tmp, store, ctx) = setup().await;
    store.create_spec(&make_spec("s1")).await.unwrap();
    ctx.set_id(Some("s1".into()));
    ctx.set_run_id(Some("run1".into()));
    ctx.reset_saves();

    let tool = ResearchSetTargetTool::new(store.clone(), ctx.clone());
    let out = tool
        .execute(
            serde_json::json!({
                "spec_id":"",
                "force": true,
                "note": "no listings matched the strict spec on visited pages"
            }),
            &cwd(),
        )
        .await;
    assert!(!out.is_error, "force-clear must succeed: {}", out.output);
    assert_eq!(ctx.id(), None);
    assert_eq!(ctx.run_id(), None);
    assert_eq!(ctx.save_count(), 0);
}

#[tokio::test]
async fn set_target_clear_with_at_least_one_save_succeeds() {
    let (_tmp, store, ctx) = setup().await;
    store.create_spec(&make_spec("s1")).await.unwrap();
    ctx.set_id(Some("s1".into()));
    ctx.set_run_id(Some("run1".into()));
    ctx.reset_saves();
    // Simulate a successful research_save during the run.
    ctx.note_save();

    let tool = ResearchSetTargetTool::new(store.clone(), ctx.clone());
    let out = tool
        .execute(serde_json::json!({"spec_id":""}), &cwd())
        .await;
    assert!(
        !out.is_error,
        "clear after a real save must succeed, got: {}",
        out.output
    );
    assert_eq!(ctx.id(), None);
}
