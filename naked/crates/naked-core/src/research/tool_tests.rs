use super::redact::{redact_for_log, scan_and_redact}; // pub(crate); tool/mod.rs no longer re-exports
use super::*;
use crate::research::spec::ResearchSpec;
use crate::research::store_fs::FsResearchStore;
use std::{borrow::Cow, hint::black_box, time::Instant};
use tempfile::tempdir;

fn setup() -> (tempfile::TempDir, Arc<dyn ResearchStore>, ResearchContext) {
    let tmp = tempdir().unwrap();
    let store: Arc<dyn ResearchStore> = Arc::new(FsResearchStore::new(tmp.path().to_path_buf()));
    let ctx = ResearchContext::new();
    (tmp, store, ctx)
}

fn make_spec(id: &str) -> ResearchSpec {
    // REGISTRY-WAIVE: exhaustive test struct ctor — pre-2026-05-13 baseline; new tests must use ..Default::default()
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

// ── T5.1 tests (PLAN_RESEARCH_AGENT_FLOW_v1) ───────────────────────────────
// Verify explicit spec_id arg: no active context required when spec_id is
// passed. Also verify arg wins over context (DIP / Tell-Don't-Ask).

#[tokio::test]
async fn research_save_with_explicit_spec_id_no_context() {
    // T5.1: spec_id arg means no active context needed
    let (_tmp, store, ctx) = setup(); // ctx has no id set
    store
        .create_spec(&make_spec("explicit-spec"))
        .await
        .unwrap();
    let tool = ResearchSaveTool::new(store.clone(), ctx, Default::default());
    let cwd = std::env::current_dir().unwrap();
    let r = tool
        .execute(
            json!({
                "spec_id": "explicit-spec",
                "url": "https://ex.com/via-spec-id",
                "title": "Via explicit spec_id"
            }),
            &cwd,
        )
        .await;
    assert!(
        !r.is_error,
        "should succeed with explicit spec_id: {}",
        r.output
    );
    assert!(
        r.output.contains("\"stored\":true"),
        "finding should be stored: {}",
        r.output
    );
    assert_eq!(store.count_findings("explicit-spec").await.unwrap(), 1);
}

#[tokio::test]
async fn research_save_explicit_spec_id_overrides_context() {
    // T5.1: explicit spec_id takes priority over active context id
    let (_tmp, store, ctx) = setup();
    store.create_spec(&make_spec("ctx-spec")).await.unwrap();
    store.create_spec(&make_spec("arg-spec")).await.unwrap();
    ctx.set_id(Some("ctx-spec".into()));
    let tool = ResearchSaveTool::new(store.clone(), ctx, Default::default());
    let cwd = std::env::current_dir().unwrap();
    let r = tool
        .execute(
            json!({
                "spec_id": "arg-spec",
                "url": "https://ex.com/arg-wins",
                "title": "Arg wins"
            }),
            &cwd,
        )
        .await;
    assert!(!r.is_error, "should succeed: {}", r.output);
    // arg-spec gets the finding, ctx-spec stays empty
    assert_eq!(store.count_findings("arg-spec").await.unwrap(), 1);
    assert_eq!(store.count_findings("ctx-spec").await.unwrap(), 0);
}

#[tokio::test]
async fn research_save_rejects_missing_context() {
    // T5.1 backward compat: still fails when NEITHER spec_id arg NOR context
    let (_tmp, store, ctx) = setup();
    let tool = ResearchSaveTool::new(store.clone(), ctx, Default::default());
    let cwd = std::env::current_dir().unwrap();
    let r = tool.execute(json!({"url":"https://ex.com/a"}), &cwd).await;
    assert!(r.is_error);
    // Error message should explain both ways to fix it
    assert!(
        r.output.contains("spec_id") || r.output.contains("research context"),
        "error should mention spec_id or context: {}",
        r.output
    );
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
fn redact_strips_authorization_token_scheme() {
    for (input, expected_scheme) in [
        ("Authorization: Token SYNTH_TOKEN_VALUE", "Token"),
        ("authorization: token SYNTH_TOKEN_VALUE", "token"),
        ("Authorization: Bearer SYNTH_BEARER_VALUE", "Bearer"),
    ] {
        let out = scan_and_redact(input);
        assert!(
            out.contains(&format!("{expected_scheme} [redacted]")),
            "got: {out}"
        );
        assert!(!out.contains("SYNTH_"), "got: {out}");
    }
}

#[test]
fn redact_preserves_existing_key_value_bearer_and_password_patterns() {
    let s = "api-key: sk-live bearer tok-123 password = hunter2 secret=abc123 token=xyz987";
    let out = scan_and_redact(s);
    assert_eq!(
        out,
        "api-key=[redacted] bearer [redacted] password=[redacted] secret=[redacted] token=[redacted]"
    );
    for leaked in ["sk-live", "tok-123", "hunter2", "abc123", "xyz987"] {
        assert!(!out.contains(leaked), "{leaked} leaked in {out}");
    }
}

#[tokio::test]
async fn redact_for_log_strips_telegram_bot_token_from_reqwest_error_url() {
    let bot_id = "123456789";
    let secret = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghi_012345";
    let url = format!("http://127.0.0.1:9/bot{bot_id}:{secret}/getUpdates");
    let err = reqwest::Client::new()
        .get(url)
        .send()
        .await
        .expect_err("closed local port should fail without external network");
    let rendered = redact_for_log(&err);
    assert!(rendered.contains("bot[redacted]"), "got: {rendered}");
    assert_no_synthetic_token_leak(&rendered, bot_id, &[secret]);
}

#[test]
fn redact_telegram_bot_and_query_token_shapes() {
    const BOT_ID: &str = "1234567890";
    const FULL_SECRET: &str = "AAFZZfake000000000000000000000000000";
    const TRUNCATED_SECRET: &str = "AAFZ";
    const DASH_UNDERSCORE_SECRET: &str = "AAFZZ_fake-0000";
    const QUERY_SECRET: &str = "AAFZZquery0000";
    const ACCESS_SECRET: &str = "AAFZZaccess0000";

    let cases = [
        (
            "full-length path token",
            format!("url https://api.telegram.org/bot{BOT_ID}:{FULL_SECRET}/getUpdates"),
            vec![FULL_SECRET],
            vec!["bot[redacted]/getUpdates"],
        ),
        (
            "truncated path secret",
            format!("url https://api.telegram.org/bot{BOT_ID}:{TRUNCATED_SECRET}/getUpdates"),
            vec![TRUNCATED_SECRET],
            vec!["bot[redacted]/getUpdates"],
        ),
        (
            "uppercase BOT prefix",
            format!("url https://api.telegram.org/BOT{BOT_ID}:{FULL_SECRET}/getUpdates"),
            vec![FULL_SECRET],
            vec!["bot[redacted]/getUpdates"],
        ),
        (
            "mixed-case host",
            format!("url HTTPS://API.Telegram.Org/bot{BOT_ID}:{FULL_SECRET}/getUpdates"),
            vec![FULL_SECRET],
            vec!["bot[redacted]/getUpdates"],
        ),
        (
            "dash underscore secret",
            format!(
                "url https://api.telegram.org/bot{BOT_ID}:{DASH_UNDERSCORE_SECRET}/sendMessage"
            ),
            vec![DASH_UNDERSCORE_SECRET],
            vec!["bot[redacted]/sendMessage"],
        ),
        (
            "adjacent punctuation and parens",
            format!("(https://api.telegram.org/bot{BOT_ID}:{TRUNCATED_SECRET})."),
            vec![TRUNCATED_SECRET],
            vec!["(https://api.telegram.org/bot[redacted])."],
        ),
        (
            "token twice in one string",
            format!(
                "first bot{BOT_ID}:{FULL_SECRET}/getUpdates second bot{BOT_ID}:{DASH_UNDERSCORE_SECRET}/sendMessage"
            ),
            vec![FULL_SECRET, DASH_UNDERSCORE_SECRET],
            vec!["first bot[redacted]/getUpdates second bot[redacted]/sendMessage"],
        ),
        (
            "bare token query key preserves following params",
            format!("https://example.invalid/hook?token={QUERY_SECRET}&ok=1"),
            vec![QUERY_SECRET],
            vec!["?token=[redacted]&ok=1"],
        ),
        (
            "prefixed bot_token query key",
            format!("https://example.invalid/hook?bot_token={BOT_ID}:{QUERY_SECRET}&ok=1"),
            vec![QUERY_SECRET],
            vec!["?bot_token=[redacted]&ok=1"],
        ),
        (
            "prefixed access_token query key",
            format!("https://example.invalid/hook?access_token={BOT_ID}:{ACCESS_SECRET}&ok=1"),
            vec![ACCESS_SECRET],
            vec!["?access_token=[redacted]&ok=1"],
        ),
    ];

    for (name, input, secrets, expected_fragments) in cases {
        let out = scan_and_redact(&input);
        assert_no_synthetic_token_leak(&out, BOT_ID, &secrets);
        for expected in expected_fragments {
            assert!(
                out.contains(expected),
                "case {name}: expected {expected:?}, got {out}"
            );
        }
    }
}

#[test]
fn redact_compound_secret_keys_require_separator_before_keyword() {
    let must_redact = [
        ("bot_token=tok-bot-value", "bot_token=[redacted]"),
        ("access_token=tok-access-value", "access_token=[redacted]"),
        ("api-key=sk-api-key", "api-key=[redacted]"),
        ("x_api_key=sk-x-api", "x_api_key=[redacted]"),
        ("api_key=sk-api", "api_key=[redacted]"),
        ("token=tok-bare", "token=[redacted]"),
        ("secret=sec-bare", "secret=[redacted]"),
        ("password=pw-bare", "password=[redacted]"),
        ("--api-key=sk-cli", "--api-key=[redacted]"),
        ("?token=tok-query&ok=1", "?token=[redacted]&ok=1"),
        ("client.secret=sec-client", "client.secret=[redacted]"),
    ];
    for (input, expected) in must_redact {
        assert_eq!(scan_and_redact(input), expected, "must redact {input:?}");
    }

    let must_not_touch = [
        "monkey=banana",
        "?monkey=banana&ok=1",
        "hotkey=ctrl-k",
        "turkey=bird",
        "donkey=grey",
        "--monkey=banana",
        "keynote=speech",
        "tokenizer=bert",
        "benign prose sentence containing the word token but no assignment",
    ];
    for input in must_not_touch {
        assert_eq!(scan_and_redact(input), input, "must not touch {input:?}");
    }
}

#[test]
fn redact_ignores_unrelated_text() {
    let s = "normal log line with https://example.com/path and nothing sensitive";
    let out = scan_and_redact(s);
    assert_eq!(out, s);
    assert!(matches!(out, Cow::Borrowed(_)), "no-match path allocated");

    let benign = "token budgeting is about context length, not credentials";
    let benign_out = scan_and_redact(benign);
    assert_eq!(benign_out, benign);
    assert!(
        matches!(benign_out, Cow::Borrowed(_)),
        "benign candidate no-match path allocated"
    );
}

#[test]
fn redact_no_secret_mid_size_fast_path_perf_guard() {
    let payload = no_secret_cyrillic_payload(16 * 1024);
    let iterations = 128;
    let start = Instant::now();
    for _ in 0..iterations {
        let out = scan_and_redact(black_box(&payload));
        assert!(
            matches!(out, Cow::Borrowed(_)),
            "no-secret payload allocated"
        );
        black_box(out);
    }
    let per_call_ms = start.elapsed().as_secs_f64() * 1000.0 / f64::from(iterations);
    assert!(
        per_call_ms < 10.0,
        "redactor no-secret fast path regressed: {per_call_ms:.3} ms/call"
    );
}

#[test]
fn scan_and_redact_is_idempotent_for_existing_shapes() {
    for input in redaction_idempotency_inputs() {
        let once = scan_and_redact(&input).into_owned();
        let twice = scan_and_redact(&once).into_owned();
        assert_eq!(twice, once, "redaction not idempotent for {input:?}");
    }
}

fn redaction_idempotency_inputs() -> Vec<String> {
    const BOT_ID: &str = "1234567890";
    const FULL_SECRET: &str = "AAFZZfake000000000000000000000000000";
    const TRUNCATED_SECRET: &str = "AAFZ";
    const DASH_UNDERSCORE_SECRET: &str = "AAFZZ_fake-0000";
    const QUERY_SECRET: &str = "AAFZZquery0000";
    const ACCESS_SECRET: &str = "AAFZZaccess0000";

    vec![
        "log: api_key=sk-abc123 done; Authorization: Bearer eyJhbGci OK".to_string(),
        "Authorization: Bearer [redacted]".to_string(),
        "Authorization: Token SYNTH_TOKEN_VALUE".to_string(),
        "Authorization: Token [redacted]".to_string(),
        "api-key: sk-live bearer tok-123 password = hunter2 secret=abc123 token=xyz987".to_string(),
        format!("url https://api.telegram.org/bot{BOT_ID}:{FULL_SECRET}/getUpdates"),
        format!("url https://api.telegram.org/bot{BOT_ID}:{TRUNCATED_SECRET}/getUpdates"),
        format!("url https://api.telegram.org/BOT{BOT_ID}:{FULL_SECRET}/getUpdates"),
        format!("url HTTPS://API.Telegram.Org/bot{BOT_ID}:{FULL_SECRET}/getUpdates"),
        format!("url https://api.telegram.org/bot{BOT_ID}:{DASH_UNDERSCORE_SECRET}/sendMessage"),
        format!("(https://api.telegram.org/bot{BOT_ID}:{TRUNCATED_SECRET})."),
        format!(
            "first bot{BOT_ID}:{FULL_SECRET}/getUpdates second bot{BOT_ID}:{DASH_UNDERSCORE_SECRET}/sendMessage"
        ),
        format!("https://example.invalid/hook?token={QUERY_SECRET}&ok=1"),
        format!("https://example.invalid/hook?bot_token={BOT_ID}:{QUERY_SECRET}&ok=1"),
        format!("https://example.invalid/hook?access_token={BOT_ID}:{ACCESS_SECRET}&ok=1"),
        "bot_token=tok-bot-value".to_string(),
        "access_token=tok-access-value".to_string(),
        "api-key=sk-api-key".to_string(),
        "x_api_key=sk-x-api".to_string(),
        "api_key=sk-api".to_string(),
        "token=tok-bare".to_string(),
        "secret=sec-bare".to_string(),
        "password=pw-bare".to_string(),
        "--api-key=sk-cli".to_string(),
        "?token=tok-query&ok=1".to_string(),
        "client.secret=sec-client".to_string(),
        "monkey=banana".to_string(),
        "?monkey=banana&ok=1".to_string(),
        "hotkey=ctrl-k".to_string(),
        "turkey=bird".to_string(),
        "donkey=grey".to_string(),
        "--monkey=banana".to_string(),
        "keynote=speech".to_string(),
        "tokenizer=bert".to_string(),
        "benign prose sentence containing the word token but no assignment".to_string(),
        "normal log line with https://example.com/path and nothing sensitive".to_string(),
        "token budgeting is about context length, not credentials".to_string(),
        r#"config: secret="topsecret" and token = "xyz""#.to_string(),
        "deepseek v4 flash is online.\n\n<blockquote expandable>\u{1F4AD} <b>thinking</b>\nuser wanted exactly 5 words".to_string(),
        "\u{1F4AD} request had api_key=AKIAEXAMPLE_SECRET_VALUE inside".to_string(),
    ]
}

fn no_secret_cyrillic_payload(bytes: usize) -> String {
    let chunk = "Это обычный ответ без чувствительных значений. ";
    let mut s = String::with_capacity(bytes + chunk.len());
    while s.len() < bytes {
        s.push_str(chunk);
    }
    let mut end = bytes.min(s.len());
    while !s.is_char_boundary(end) {
        end -= 1;
    }
    s.truncate(end);
    s
}

fn assert_no_synthetic_token_leak(rendered: &str, bot_id: &str, secrets: &[&str]) {
    assert!(
        !contains_bot_token_shape(rendered),
        "bot<digits>: shape leaked in {rendered}"
    );
    assert!(
        !rendered.contains(&format!("{bot_id}:")),
        "bot id + colon leaked in {rendered}"
    );
    for secret in secrets {
        assert!(
            !rendered.contains(secret),
            "secret {secret} leaked in {rendered}"
        );
    }
}

fn contains_bot_token_shape(rendered: &str) -> bool {
    let lower = rendered.to_ascii_lowercase();
    let bytes = lower.as_bytes();
    let mut i = 0;
    while i + 3 < bytes.len() {
        if &bytes[i..i + 3] == b"bot" {
            let mut cursor = i + 3;
            let digits_start = cursor;
            while cursor < bytes.len() && bytes[cursor].is_ascii_digit() {
                cursor += 1;
            }
            if (8..=10).contains(&(cursor - digits_start)) && bytes.get(cursor) == Some(&b':') {
                return true;
            }
        }
        i += 1;
    }
    false
}

#[test]
fn redact_handles_quoted_values() {
    let s = r#"config: secret="topsecret" and token = "xyz""#;
    let out = scan_and_redact(s);
    assert!(!out.contains("topsecret"), "got: {out}");
    assert!(!out.contains("xyz"), "got: {out}");
}

#[test]
fn redact_does_not_panic_on_multibyte_utf8_thinking_content() {
    // B48 regression: redactor walked byte-by-byte and tried to slice
    // `&hay[end..end + part.len()]` where `end` could fall mid-emoji.
    // Repro string from the original panic (2026-05-13 deepseek-v4-flash via airpx):
    let s = "deepseek v4 flash is online.\n\n<blockquote expandable>\u{1F4AD} <b>thinking</b>\nuser wanted exactly 5 words";
    // Should NOT panic. May be no-op (no credential patterns matched).
    let out = scan_and_redact(s);
    // Output preserves the emoji and structure intact.
    assert!(out.contains("\u{1F4AD}"), "thinking emoji preserved");
    assert!(out.contains("v4 flash"), "text preserved");
}

#[test]
fn redact_with_real_credentials_amid_multibyte() {
    // Mix of safe multi-byte text + actual credential to redact.
    let s = "\u{1F4AD} request had api_key=AKIAEXAMPLE_SECRET_VALUE inside";
    let out = scan_and_redact(s);
    assert!(!out.contains("AKIAEXAMPLE_SECRET_VALUE"), "got: {out}");
    assert!(out.contains("\u{1F4AD}"), "emoji preserved: {out}");
}
