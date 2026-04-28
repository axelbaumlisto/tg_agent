//! Red-test matrix for `naked-tg` — observed-bug regressions that must
//! stay failing until each category is driven green per the plan at
//! `/home/example/.cursor/plans/naked-tg_red-test_matrix.plan.md`.
//!
//! Every test name starts with `red_<id>_` so we can filter:
//!
//! ```bash
//! cargo test --test red_matrix
//! cargo test --test red_matrix red_a1   # just A1
//! ```
//!
//! Categories covered here (naked-tg side):
//!
//! - A1 `ChannelSessionMap` in-memory only → rebuild on every restart.
//! - A4 assistant-turn can end with tool_use chain and no text reply.
//! - B1 `thinking` block leaks into the outgoing Telegram message.
//! - E2 zero-byte download must retry transparently.
//! - E3 missing `tg_media.audio` config must emit ONE warn per session, not a silent placeholder per voice.
//! - E4 skill results (telegram-reader) must be tagged with `source_chat_id`/`title` to kill persona cross-talk.
//! - F2 slash-gate must cover `CallbackQuery.data`, not just `Message`.
//! - F3 session-handover quote `> @bot [your previous message]:` must NOT be treated as an addressed reply.

// ─────────────────────────────────────────────────────────────────────────────
// A1 — ChannelSessionMap persistence across restart
// ─────────────────────────────────────────────────────────────────────────────

/// After a process restart, the (chat_id, thread_id) → session_id mapping
/// must still be honoured. Today the map lives in `tokio::sync::RwLock<HashMap>`
/// and dies with the process — the fallback (`SessionStore::list` + rebuild
/// from `metadata.channel_id`) races with new traffic and can create ghost
/// sessions. We want a durable snapshot as part of normal bot state.
#[tokio::test]
async fn red_a1_channel_map_survives_restart() {
    use naked_tg::channel_map::ChannelSessionMap;
    let dir = tempfile::tempdir().unwrap();

    let map1 = ChannelSessionMap::open(dir.path()).await.unwrap();
    map1.set(-5084292206, None, "sess-a".into()).await;
    map1.set(-5084292206, Some(7), "sess-a-topic".into()).await;
    map1.enable_yolo(-5084292206, Some(7)).await;
    map1.allow_add(-5084292206, None, "bash").await;
    map1.flush().await.unwrap();
    drop(map1);

    let map2 = ChannelSessionMap::open(dir.path()).await.unwrap();
    assert_eq!(
        map2.get(-5084292206, None).await.as_deref(),
        Some("sess-a"),
        "base session mapping must survive restart"
    );
    assert_eq!(
        map2.get(-5084292206, Some(7)).await.as_deref(),
        Some("sess-a-topic"),
        "per-topic session mapping must survive restart"
    );
    assert!(
        map2.is_yolo(-5084292206, Some(7)).await,
        "YOLO flag must survive restart (TTL still valid)"
    );
    let allow = map2.allow_get(-5084292206, None).await;
    assert_eq!(allow, vec!["bash"], "allow-list must survive restart");
}

// ─────────────────────────────────────────────────────────────────────────────
// A4 — Turn must emit a user-visible text block
// ─────────────────────────────────────────────────────────────────────────────

/// An assistant turn that finishes with tool_use blocks and no final
/// `text` leaves the user staring at the last bot message forever. We
/// saw this in Income session `5c73bed7` at 2026-04-21 13:34-13:36:
/// seven consecutive tool_use calls, no reply. A post-turn invariant
/// check must flag this.
#[tokio::test]
async fn red_a4_tool_only_turn_is_flagged_incomplete() {
    use naked_core::session::turn_invariants::TurnInvariants;
    use naked_core::types::ContentBlock;

    let tool_only = vec![
        ContentBlock::ToolUse {
            id: "1".into(),
            name: "bash".into(),
            input: serde_json::json!({}),
        },
        ContentBlock::ToolUse {
            id: "2".into(),
            name: "bash".into(),
            input: serde_json::json!({}),
        },
    ];
    let report = TurnInvariants::check_assistant_blocks(&tool_only);
    assert!(
        report.incomplete_turn,
        "a turn with only tool_use blocks must flag incomplete_turn"
    );
    assert!(report.is_violated());

    let with_text = vec![
        ContentBlock::ToolUse {
            id: "1".into(),
            name: "bash".into(),
            input: serde_json::json!({}),
        },
        ContentBlock::Text {
            text: "Готово.".into(),
        },
    ];
    let report2 = TurnInvariants::check_assistant_blocks(&with_text);
    assert!(!report2.incomplete_turn);
}

// ─────────────────────────────────────────────────────────────────────────────
// B1 — Thinking must not leak into the outbound Telegram text
// ─────────────────────────────────────────────────────────────────────────────

/// Incident 2026-04-21 12:21 UTC (Income): the bot's final message
/// contained the prefix "The user is asking about exchange (обмен)…"
/// — raw thinking concatenated with the real answer. The render/stream
/// layer must split `[Thinking, Text]` so only `Text` hits TG.
#[test]
fn red_b1_thinking_block_must_not_leak_into_rendered_text() {
    use naked_core::types::ContentBlock;
    use naked_tg::render::render_live_user_text;

    let rendered = render_live_user_text(&[
        ContentBlock::Thinking {
            text: "The user is asking about exchange (обмен); plan: …".into(),
        },
        ContentBlock::Text {
            text: "Привет! Курс обмена…".into(),
        },
    ]);
    assert!(
        !rendered.contains("The user is asking about exchange"),
        "thinking must not leak into rendered text; got:\n{rendered}"
    );
    assert!(rendered.contains("Курс обмена"));
}

// ─────────────────────────────────────────────────────────────────────────────
// E2 — zero-byte download must be classified as transient (unit test)
// ─────────────────────────────────────────────────────────────────────────────

/// `download_to_artifacts` already retries on transient errors; we need a
/// unit-level guard that empty bodies are explicitly in the transient set.
/// Today the classifier is private to `media.rs` and only covered at the
/// live-call level — a unit test pins the contract.
#[test]
fn red_e2_zero_byte_body_is_transient() {
    use naked_tg::media_helpers::DownloadErrorClass;

    let err = anyhow::anyhow!("file download body read failed: empty body");
    assert!(
        DownloadErrorClass::from_anyhow(&err).is_transient(),
        "empty-body download errors must be classified as transient (retryable)"
    );

    let zero = anyhow::anyhow!("read 0 bytes from Telegram CDN");
    assert!(
        DownloadErrorClass::from_anyhow(&zero).is_transient(),
        "zero-byte body must be transient too"
    );

    // 404 is terminal (bad file_id), never transient.
    let missing = anyhow::anyhow!("getFile returned status: 404 Not Found");
    assert!(!DownloadErrorClass::from_anyhow(&missing).is_transient());
}

// ─────────────────────────────────────────────────────────────────────────────
// E3 — missing tg_media.audio config must warn once
// ─────────────────────────────────────────────────────────────────────────────

/// Current behaviour for a voice message when `tg_media.audio = None`:
/// the media pipeline returns the placeholder text
/// `[transcription not configured — set tg_media.audio]` for EVERY voice
/// in EVERY session, with no WARN in the log. Operators miss the misconfig.
///
/// The contract we want: at startup (or first voice in a session), emit
/// exactly one WARN with instructions, then either transcribe or
/// gracefully drop the media without leaking the placeholder as a
/// user-facing line.
#[test]
fn red_e3_missing_audio_config_warns_once() {
    use naked_tg::media_helpers::{AudioConfigDiagnostics, VoiceHandling};

    let diag = AudioConfigDiagnostics::new();
    // First voice → operator warn, media drops silently.
    match diag.handle_voice(None) {
        VoiceHandling::FirstWarnThenDrop { warn } => {
            assert!(
                warn.contains("tg_media.audio"),
                "warn must point operator at the config key"
            );
        }
        other => panic!("expected FirstWarnThenDrop, got {other:?}"),
    }
    // Next four voices — quiet drop, no log churn.
    for _ in 0..4 {
        match diag.handle_voice(None) {
            VoiceHandling::QuietDrop => {}
            other => panic!("expected QuietDrop after first warn, got {other:?}"),
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// E4 — skill result must tag source chat so personas do not cross-pollute
// ─────────────────────────────────────────────────────────────────────────────

/// Incident: Income persona claimed to "see 3 dog photos in Income"
/// when the dogs actually lived in DM with @avprilipko. The cause: the
/// telegram-reader skill returns raw JSON without `source_chat_id`/title.
/// The prompt context then loses the chat boundary.
#[test]
fn red_e4_skill_result_envelope_exposes_source_chat() {
    use naked_tg::skill::SkillResultEnvelope;

    let env = SkillResultEnvelope::wrap_telegram_reader(
        -5084292206,
        "Income",
        "zverozabr_session",
        serde_json::json!({"messages":[{"id":1,"text":"hi"}]}),
    );
    let rendered = env.render_for_prompt();

    // First line must carry the chat id so the prompt rule
    // "источник в первой строке" is structurally enforced.
    let first = rendered.lines().next().unwrap_or("");
    assert_eq!(first, "chat_id=-5084292206", "first line must pin chat_id");

    assert!(rendered.contains("chat_title=Income"));
    assert!(rendered.contains("session=zverozabr_session"));
    assert!(rendered.contains("skill=telegram-reader"));
    assert!(rendered.contains("\"messages\""));
}

// ─────────────────────────────────────────────────────────────────────────────
// F2 — slash-gate must cover CallbackQuery, not only Message
// ─────────────────────────────────────────────────────────────────────────────

/// For a persona with `allow_slash_commands = false` (e.g. Income), inline
/// keyboards with callback_data starting with `/` must not smuggle slash
/// commands past the gate. Today `drop_slash_for_persona` only inspects
/// `Message`, so a crafted callback button bypasses the rule.
#[test]
fn red_f2_slash_in_callback_data_is_dropped_when_persona_disallows() {
    use naked_tg::persona::{DispatchChannel, is_slash_dispatchable};

    // Income persona: `allow_slash_commands = false` must block slash
    // in BOTH message and callback paths.
    assert!(!is_slash_dispatchable(false, "/reset", DispatchChannel::Message));
    assert!(!is_slash_dispatchable(false, "/reset", DispatchChannel::Callback));

    // Default persona: allowed.
    assert!(is_slash_dispatchable(true, "/reset", DispatchChannel::Message));
    assert!(is_slash_dispatchable(true, "/reset", DispatchChannel::Callback));

    // Non-slash text is not the gate's concern.
    assert!(is_slash_dispatchable(false, "hello", DispatchChannel::Callback));
}

// ─────────────────────────────────────────────────────────────────────────────
// F3 — handover quote must not be treated as addressed reply
// ─────────────────────────────────────────────────────────────────────────────

/// When a session rolls (A1/A2), the new session receives a synthetic
/// first user message shaped like:
///
/// ```text
/// > @example_bot [your previous message]:
/// > ...
/// ```
///
/// The bot's addressing gate (`is_addressed_to_bot`) happens to walk
/// entities inside this quote, see a mention of itself, and treat the
/// message as a fresh reply — which causes it to answer its own echoed
/// paragraph. We want an explicit classifier that marks these blocks as
/// handover context, not dialogue.
#[test]
fn red_f3_session_handover_quote_is_ignored_by_addressing_gate() {
    use naked_tg::bot_identity::classify_handover;

    let raw = "> @example_bot [your previous message]:\n> Вот таблица…\n> строка 2\n\nчто дальше?";
    let cls = classify_handover(raw);
    assert!(
        cls.is_handover_resume,
        "leading `> @bot [your previous message]:` must be recognised as handover"
    );
    assert_eq!(
        cls.effective_user_text.trim(),
        "что дальше?",
        "stripped body must drop the quoted block entirely"
    );

    // Negative: a plain quoted reply without the sentinel stays untouched.
    let plain = "> человек написал что-то\nсогласен";
    let neg = classify_handover(plain);
    assert!(!neg.is_handover_resume);
    assert_eq!(neg.effective_user_text, plain);
}
