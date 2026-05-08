use super::*;
use crate::research::spec::ResearchSpec;
use crate::research::store_fs::FsResearchStore;
use tempfile::tempdir;

fn setup() -> (tempfile::TempDir, Arc<dyn ResearchStore>, ResearchContext) {
    let tmp = tempdir().unwrap();
    let store: Arc<dyn ResearchStore> = Arc::new(FsResearchStore::new(tmp.path().to_path_buf()));
    let ctx = ResearchContext::new();
    (tmp, store, ctx)
}

fn make_spec(id: &str) -> ResearchSpec {
    ResearchSpec {
        id: id.into(),
        topic: "t".into(),
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

#[tokio::test]
async fn research_save_stores_and_dedups() {
    let (_tmp, store, ctx) = setup();
    store.create_spec(&make_spec("r1")).await.unwrap();
    ctx.set_id(Some("r1".into()));
    let tool = ResearchSaveTool::new(store.clone(), ctx.clone(), Default::default());

    let cwd = std::env::current_dir().unwrap();
    let first = tool
        .execute(json!({"url":"https://ex.com/a","title":"A"}), &cwd)
        .await;
    assert!(!first.is_error);
    assert!(first.output.contains("\"stored\":true"));

    let dup = tool
        .execute(
            json!({"url":"https://ex.com/a?utm_source=x","title":"A updated"}),
            &cwd,
        )
        .await;
    assert!(!dup.is_error);
    assert!(dup.output.contains("\"updated\":true"));
    assert_eq!(store.count_findings("r1").await.unwrap(), 1);
}

#[tokio::test]
async fn research_save_rejects_missing_context() {
    let (_tmp, store, ctx) = setup();
    let tool = ResearchSaveTool::new(store.clone(), ctx, Default::default());
    let cwd = std::env::current_dir().unwrap();
    let r = tool.execute(json!({"url":"https://ex.com/a"}), &cwd).await;
    assert!(r.is_error);
}

#[tokio::test]
async fn research_save_rejects_bad_url() {
    let (_tmp, store, ctx) = setup();
    store.create_spec(&make_spec("r1")).await.unwrap();
    ctx.set_id(Some("r1".into()));
    let tool = ResearchSaveTool::new(store.clone(), ctx, Default::default());
    let cwd = std::env::current_dir().unwrap();
    let r = tool.execute(json!({"url":"ftp://x/"}), &cwd).await;
    assert!(r.is_error);
    let r2 = tool.execute(json!({"url":""}), &cwd).await;
    assert!(r2.is_error);
}

#[tokio::test]
async fn research_list_shows_recent_entries() {
    let (_tmp, store, ctx) = setup();
    store.create_spec(&make_spec("r2")).await.unwrap();
    ctx.set_id(Some("r2".into()));
    let save = ResearchSaveTool::new(store.clone(), ctx.clone(), Default::default());
    let cwd = std::env::current_dir().unwrap();
    save.execute(json!({"url":"https://ex.com/1","title":"One"}), &cwd)
        .await;
    save.execute(json!({"url":"https://ex.com/2","title":"Two"}), &cwd)
        .await;
    let list = ResearchListTool::new(store.clone(), ctx)
        .execute(json!({"limit":10}), &cwd)
        .await;
    assert!(!list.is_error);
    assert!(list.output.contains("https://ex.com/1"));
    assert!(list.output.contains("https://ex.com/2"));
}

#[tokio::test]
async fn research_save_cursor_roundtrip() {
    let (_tmp, store, ctx) = setup();
    store.create_spec(&make_spec("r3")).await.unwrap();
    ctx.set_id(Some("r3".into()));
    let tool = ResearchSaveCursorTool::new(store.clone(), ctx);
    let cwd = std::env::current_dir().unwrap();
    let r = tool
        .execute(json!({"cursor":{"page":7,"anchor":"abc"}}), &cwd)
        .await;
    assert!(!r.is_error, "{}", r.output);
    let reloaded = store.load_cursor("r3").await.unwrap();
    assert_eq!(reloaded.data.get("page"), Some(&json!(7)));
}

#[tokio::test]
async fn research_status_produces_markdown_for_known_id() {
    let (_tmp, store, _ctx) = setup();
    store.create_spec(&make_spec("r4")).await.unwrap();
    let tool = ResearchStatusTool::new(store.clone());
    let cwd = std::env::current_dir().unwrap();
    let r = tool.execute(json!({"research_id":"r4"}), &cwd).await;
    assert!(!r.is_error);
    assert!(r.output.contains("Research `r4`"));
    assert!(r.output.contains("**Total findings:** 0"));
}

#[tokio::test]
async fn research_status_rejects_unknown_id() {
    let (_tmp, store, _ctx) = setup();
    let tool = ResearchStatusTool::new(store.clone());
    let cwd = std::env::current_dir().unwrap();
    let r = tool
        .execute(json!({"research_id":"does-not-exist"}), &cwd)
        .await;
    assert!(r.is_error);
}

#[test]
fn strip_attribution_drops_leading_source_lines() {
    let s = "Nguồn: alonhadat.com.vn, đăng 18/04/2026\n\
             2BR, 70m², District 7. Contact: 0912345678";
    let out = strip_source_attribution(s);
    assert!(!out.contains("alonhadat"), "got: {out}");
    assert!(out.contains("0912345678"));
    assert!(out.starts_with("2BR"));
}

#[test]
fn strip_attribution_drops_trailing_clause_after_em_dash() {
    let s = "70m² fully furnished — Nguồn: chotot.com";
    let out = strip_source_attribution(s);
    assert!(!out.to_lowercase().contains("nguồn"));
    assert!(out.starts_with("70m²"));
    assert!(out.ends_with("furnished"));
}

#[test]
fn strip_attribution_drops_relative_time_trailers() {
    let s = "Apartment listing\nđăng 3 ngày trước";
    let out = strip_source_attribution(s);
    assert!(!out.contains("ngày trước"), "got: {out}");
    assert_eq!(out.trim(), "Apartment listing");
}

#[test]
fn strip_attribution_preserves_unrelated_text() {
    let s = "First line\nSecond line with phone 0987654321";
    let out = strip_source_attribution(s);
    assert_eq!(out, s);
}

#[tokio::test]
async fn research_save_strips_source_attribution_from_excerpt() {
    let (_tmp, store, ctx) = setup();
    store.create_spec(&make_spec("rs1")).await.unwrap();
    ctx.set_id(Some("rs1".into()));
    let tool = ResearchSaveTool::new(store.clone(), ctx, Default::default());
    let cwd = std::env::current_dir().unwrap();
    let body = "70m², District 1, fully furnished. Contact Ms. Lan 0912345678 \
                (Zalo). Available May 1.";
    let raw = format!("Nguồn: alonhadat.com.vn, đăng 18/04/2026\n{body}");
    let r = tool
        .execute(
            json!({"url":"https://ex.com/strip","title":"T","price":"$500","excerpt":raw}),
            &cwd,
        )
        .await;
    assert!(!r.is_error, "{}", r.output);
    let findings = store.list_findings("rs1", Some(10)).await.unwrap();
    let stored = findings[0].excerpt.as_deref().unwrap_or("");
    assert!(!stored.contains("alonhadat"), "got: {stored}");
    assert!(stored.contains("0912345678"));
}

#[test]
fn redact_strips_api_key_and_bearer() {
    let s = "log: api_key=sk-abc123 done; Authorization: Bearer eyJhbGci OK";
    let out = scan_and_redact(s);
    assert!(out.contains("api_key=[redacted]"), "got: {out}");
    assert!(
        out.contains("Bearer [redacted]") || out.contains("bearer [redacted]"),
        "got: {out}"
    );
    assert!(!out.contains("sk-abc123"), "got: {out}");
    assert!(!out.contains("eyJhbGci"), "got: {out}");
}

#[test]
fn redact_ignores_unrelated_text() {
    let s = "normal log line with https://example.com/path and nothing sensitive";
    let out = scan_and_redact(s);
    assert_eq!(out, s);
}

#[test]
fn redact_handles_quoted_values() {
    let s = r#"config: secret="topsecret" and token = "xyz""#;
    let out = scan_and_redact(s);
    assert!(!out.contains("topsecret"), "got: {out}");
    assert!(!out.contains("xyz"), "got: {out}");
}
