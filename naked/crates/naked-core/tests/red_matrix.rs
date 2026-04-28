//! Red-test matrix for `naked-core` — observed-bug regressions that
//! must stay failing until each category is driven green per the plan at
//! `/home/spex/.cursor/plans/naked-tg_red-test_matrix_eb5c5662.plan.md`.
//!
//! Run:
//!
//! ```bash
//! cargo test -p naked-core --test red_matrix
//! cargo test -p naked-core --test red_matrix red_a2
//! ```
//!
//! Categories covered here (naked-core side):
//!
//! - A2  session should emit a warn-event before provider context limit.
//! - A3  live effective system prompt must follow persona workspace, not the snapshot in session_meta.
//! - B2  `ToolResult` content must round-trip through `session.jsonl`.
//! - C1  `research_launch` must return `verified:true` only when the agent actually started.
//! - C2  `research_save` must reject category URLs (no concrete id-segment).
//! - C3  `research_save` must reject stale listings (>90d old).
//! - D1  provider-level errors must be typed (`QuotaExhausted`), not free text.
//! - D2  `web_fetch` must detect Cloudflare challenge and return a typed error, not empty body.

// ─────────────────────────────────────────────────────────────────────────────
// A2 — Session context budget watermark
// ─────────────────────────────────────────────────────────────────────────────

/// A session climbing toward the provider context limit (e.g. GLM 128k)
/// must emit a structured warning before it silently rolls. DM session
/// `093a7d74` hit 104 553 input_tokens on its last turn before being
/// abandoned; no warning was surfaced to operator or bot.
#[test]
fn red_a2_session_warns_before_context_limit() {
    use naked_core::session::budget::{ContextBudgetEvent, SessionBudget};

    let mut budget = SessionBudget::new(128_000);
    assert_eq!(budget.observe_turn(10_000), None);
    let ev = budget
        .observe_turn(110_000)
        .expect("threshold crossed but no event");
    match ev {
        ContextBudgetEvent::ApproachingLimit { percent, .. } => {
            assert!(percent >= 80, "expected >=80%, got {percent}");
        }
    }
    assert_eq!(
        budget.observe_turn(111_000),
        None,
        "must not repeat warning once fired"
    );
}

// ─────────────────────────────────────────────────────────────────────────────
// A3 — session_meta.system_prompt is a stale snapshot
// ─────────────────────────────────────────────────────────────────────────────

/// Today `SessionMetadata` has no `system_prompt` field directly, but
/// `SessionMeta` row in `session.jsonl` captures the prompt at creation.
/// For chat personas with `workspace_override`, the effective prompt must
/// be resolved from `<workspace>/.naked/system_prompt.md` on every
/// message, otherwise a persona prompt change silently doesn't propagate
/// to existing sessions.
#[test]
fn red_a3_effective_prompt_follows_workspace_override() {
    use naked_core::prompt::effective_system_prompt;

    let dir = tempfile::tempdir().unwrap();
    let ws = dir.path().join("income-ws");
    std::fs::create_dir_all(ws.join(".naked")).unwrap();
    std::fs::write(ws.join(".naked/system_prompt.md"), "Income persona v2").unwrap();
    let p = effective_system_prompt(&ws, "stale Друся").unwrap();
    assert_eq!(
        p, "Income persona v2",
        "workspace prompt must override session_meta snapshot"
    );

    let ws2 = dir.path().join("no-override");
    std::fs::create_dir_all(&ws2).unwrap();
    let p2 = effective_system_prompt(&ws2, "meta snapshot").unwrap();
    assert_eq!(p2, "meta snapshot", "falls back to meta when no override");
}

// ─────────────────────────────────────────────────────────────────────────────
// B2 — tool_result content round-trip through session.jsonl
// ─────────────────────────────────────────────────────────────────────────────

/// Inspecting live `session.jsonl` files (e.g. Income `5c73bed7`) shows
/// every `tool_result` entry with an empty string in the dumped `content`
/// view. That means the history-to-jsonl persistence drops the output, or
/// the dumper slices incorrectly — either way auditing broken.
///
/// The invariant: whatever string we push into a `ToolResult` block must
/// come back from a saved+reloaded session.
#[tokio::test]
async fn red_b2_tool_result_content_round_trips() {
    use naked_core::session::jsonl_store::JsonlSessionStore;
    use naked_core::session::store::SessionStore;
    use naked_core::session::{Session, SessionMetadata};
    use naked_core::types::ContentBlock;

    let tmp = tempfile::tempdir().unwrap();
    let store = JsonlSessionStore::new(tmp.path().to_path_buf());

    let mut session = Session::new(
        tmp.path().to_path_buf(),
        "system".into(),
        SessionMetadata {
            name: None,
            provider: "p".into(),
            model: "m".into(),
            channel: "cli".into(),
            channel_id: None,
        },
    );
    session.history.push_assistant(
        vec![ContentBlock::ToolUse {
            id: "call-1".into(),
            name: "bash".into(),
            input: serde_json::json!({"command":"echo hi"}),
        }],
        None,
    );
    // payload we require to survive round-trip
    session.history.push_tool_result("call-1", "hi\n", false);

    store.save(&session).await.expect("save");
    let loaded = store
        .load(&session.id)
        .await
        .expect("load")
        .expect("exists");
    let blocks: Vec<ContentBlock> = loaded
        .history
        .messages()
        .iter()
        .flat_map(|m| m.blocks.clone())
        .collect();

    let restored_output = blocks
        .iter()
        .find_map(|b| match b {
            ContentBlock::ToolResult { output, .. } => Some(output.clone()),
            _ => None,
        })
        .unwrap_or_default();

    assert_eq!(
        restored_output, "hi\n",
        "ToolResult.output must survive jsonl_store save→load round-trip (was empty/truncated)"
    );
    // Defence in depth — the on-disk file should literally contain "hi".
    let sess_file = tmp.path().join(&session.id).join("session.jsonl");
    let raw = std::fs::read_to_string(&sess_file).unwrap_or_default();
    assert!(
        raw.contains("\"output\":\"hi\\n\""),
        "session.jsonl must serialise ToolResult.output verbatim; got:\n{raw}"
    );
}

// ─────────────────────────────────────────────────────────────────────────────
// C1 — research_launch must verify, not just acknowledge
// ─────────────────────────────────────────────────────────────────────────────

/// Today `research_launch` returns `{"status":"launched","verified":true}`
/// immediately after enqueueing. The scheduler may never pick up the job
/// (incident: DM session 2026-04-22 06:48 "Агент не стартовал"), and the
/// LLM wastes turns polling a run that never started.
#[tokio::test]
async fn red_c1_research_launch_waits_for_actual_agent_start() {
    use naked_core::research::launch::{LaunchOptions, LaunchOutcome, verify_launch};
    use std::time::Duration;

    // Scenario 1: agent never starts → FailedToStart.
    let outcome_timeout = verify_launch(
        "spec-timeout".into(),
        LaunchOptions {
            verify_within: Duration::from_millis(80),
        },
        || async { Ok::<_, String>(None) },
    )
    .await;
    match outcome_timeout {
        LaunchOutcome::FailedToStart { .. } => {}
        other => panic!("expected FailedToStart, got {other:?}"),
    }

    // Scenario 2: agent becomes running on the second poll → Verified.
    let mut calls = 0u32;
    let outcome_ok = verify_launch(
        "spec-ok".into(),
        LaunchOptions {
            verify_within: Duration::from_millis(500),
        },
        || {
            calls += 1;
            let n = calls;
            async move {
                if n < 2 {
                    Ok::<_, String>(None)
                } else {
                    Ok(Some(4242))
                }
            }
        },
    )
    .await;
    match outcome_ok {
        LaunchOutcome::Verified { pid, .. } => assert_eq!(pid, 4242),
        other => panic!("expected Verified, got {other:?}"),
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// C2 — research_save must reject category URLs (no concrete id)
// ─────────────────────────────────────────────────────────────────────────────

/// DM session 2423e530 on 2026-04-22 sat through a whole pass of
/// `research_save` calls with URLs like
/// `https://mogi.vn/da-nang/quan-hai-chau/thue-mat-bang-cua-hang-shop`
/// — a listing *category* page, not a concrete offer. We want the
/// validator to reject these pre-save instead of letting the Gatekeeper
/// sweep them up post-hoc.
#[tokio::test]
async fn red_c2_save_rejects_category_url_pre_validation() {
    use naked_core::research::validators::UrlSpecificity;

    let verdict = UrlSpecificity::classify(
        "https://mogi.vn/da-nang/quan-hai-chau/thue-mat-bang-cua-hang-shop",
    );
    assert!(
        matches!(verdict, UrlSpecificity::CategoryPage { .. }),
        "category URL should be rejected, got {verdict:?}"
    );

    let ok = UrlSpecificity::classify("https://mogi.vn/quan-ngu-hanh-son/thue-can-ho-id22092735");
    assert!(
        matches!(ok, UrlSpecificity::ConcreteListing),
        "concrete listing should be accepted, got {ok:?}"
    );
}

// ─────────────────────────────────────────────────────────────────────────────
// C3 — research_save must reject stale listings (>90d)
// ─────────────────────────────────────────────────────────────────────────────

/// Same session wasted a turn on findings that the Gatekeeper later
/// flagged as `>90d` old. The first pass saved them anyway. We want a
/// pre-save freshness check so the LLM can decide to drop + re-query.
#[test]
fn red_c3_save_rejects_stale_listing_pre_validation() {
    use chrono::{Duration, Utc};
    use naked_core::research::validators::Freshness;

    let now = Utc::now();
    let stale = now - Duration::days(120);
    let fresh = now - Duration::days(15);
    assert!(Freshness::from_posted_at(stale).is_stale(Duration::days(90)));
    assert!(!Freshness::from_posted_at(fresh).is_stale(Duration::days(90)));
}

// ─────────────────────────────────────────────────────────────────────────────
// D1 — Typed provider errors (Exa quota exhausted)
// ─────────────────────────────────────────────────────────────────────────────

/// Exa ran out of quota mid-run on 2026-04-22; the tool output was plain
/// text and the LLM had to guess the next step. We want a typed enum so
/// the coordinator can auto-fallback without prompt heuristics.
#[test]
fn red_d1_exa_quota_exhausted_is_typed_error() {
    use naked_core::provider::error::{FallbackHint, ProviderError};

    let err = ProviderError::from_http(429, r#"{"error":"quota_exceeded"}"#);
    match err {
        ProviderError::QuotaExhausted { hint, .. } => {
            assert_eq!(hint, FallbackHint::UseWebSearch);
        }
        other => panic!("expected QuotaExhausted, got {other:?}"),
    }

    let ok = ProviderError::from_http(500, "internal error");
    assert!(matches!(ok, ProviderError::Other { .. }));
}

// ─────────────────────────────────────────────────────────────────────────────
// D2 — Cloudflare challenge must surface as typed error
// ─────────────────────────────────────────────────────────────────────────────

/// `web_fetch` against batdongsan.com.vn and others routinely returns an
/// empty body (HTML with `cf-browser-verification` marker). The LLM has
/// no signal to switch to Playwright; it just sees "no content".
#[test]
fn red_d2_cloudflare_challenge_detected() {
    use naked_core::tool::web_fetch::{BlockKind, detect_block};

    let html = r#"<html><head><title>Just a moment…</title></head>
                  <body><div id="cf-browser-verification"></div></body></html>"#;
    let verdict = detect_block(403, html);
    assert_eq!(
        verdict,
        Some(BlockKind::Cloudflare),
        "CF interstitial should be detected"
    );

    let ok_html = "<html><body>normal content</body></html>";
    assert_eq!(detect_block(200, ok_html), None);
}
