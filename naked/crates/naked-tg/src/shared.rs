//! Shared types, constants, statics, and helper functions.
//!
//! Everything here is `pub(crate)` so that `main.rs` can re-export the whole
//! lot via `pub(crate) use shared::*;` and child modules can keep their
//! existing `use super::*;` without modification.

// ── Re-export external types that child modules rely on ────────────────────

pub(crate) use anyhow::Result;
pub(crate) use std::collections::{HashMap, HashSet};
pub(crate) use std::sync::{Arc, LazyLock};
pub(crate) use std::time::Duration;

pub(crate) use teloxide::prelude::*;
pub(crate) use teloxide::types::{
    CallbackQuery, InlineKeyboardButton, InlineKeyboardMarkup, MessageId, ParseMode, ThreadId,
};
pub(crate) use tokio::sync::{RwLock, oneshot};

pub(crate) use naked_core::AgentCore;
pub(crate) use naked_core::config::Config;
pub(crate) use naked_core::types::{
    AgentEvent, AgentHandle, Permission, PermissionResponse, TurnUsage,
};

// ── Re-export from sibling crate modules ──────────────────────────────────
pub(crate) use naked_tg::channel_map::{ChannelSessionMap, format_tg_channel_id};

// ── Re-export naked_tg library helpers used throughout ──────────────────────
pub(crate) use naked_tg::helpers::parse_interval;
pub(crate) use naked_tg::markup::{self, MAX_TG_MSG as TG_MSG_LIMIT};
pub(crate) use naked_tg::memory_scheduler;
// B4: ReportMeta + render_report_html were only used by finalize_research_ui
// (now removed).  research_html.rs remains in the lib for potential future
// direct-HTML export use cases.
pub(crate) use naked_tg::research_scheduler;
// B4: research_ui re-exports removed — legacy launch_research_run_with_ui gone.
// render_waterfall / HeartbeatProgress / keyboard_* are now only used by
// research_ui.rs's own unit tests.

// ── Constants ──────────────────────────────────────────────────────────────

pub(crate) const MAX_TG_MSG: usize = TG_MSG_LIMIT;
pub(crate) const TYPING_INTERVAL: Duration = Duration::from_secs(3);
pub(crate) const PERMISSION_TIMEOUT: Duration = Duration::from_secs(120);
pub(crate) const REASONING_TAIL: usize = 600;
// PLAN_TG_INTERLEAVED_v1: TOOL_WINDOW removed — the old
// `tool_lines: Vec<String>` field is replaced by `events: Vec<TurnEvent>`
// with budget-driven head-truncation, so a fixed last-N window is no
// longer needed. (Kept in git history if anyone needs the constant.)
pub(crate) const MAX_THINKING_BYTES: usize = 64_000;
/// Hard cap on the reasoning chain we ship in the *final* TG message
/// (inside `<blockquote expandable>`). Telegram caps a message at 4096
/// chars — we leave generous room for the actual response. Anything
/// over this is tail-truncated; the prefix is dropped because the
/// conclusion / commitments live at the bottom of a CoT.
pub(crate) const MAX_FINAL_THINKING_BYTES: usize = 2_000;
pub(crate) const MAX_RESPONSE_BYTES: usize = 128_000;
pub(crate) const REPLY_QUOTE_MAX_CHARS: usize = 1000;

// ── Type aliases ───────────────────────────────────────────────────────────

/// Global rate limiter for Telegram API calls (edit_message_text).
/// Key: call_id → (sender, chat_id, thread_id) so /yolo can drain only matching topic.
pub(crate) type PendingPermissions =
    Arc<RwLock<HashMap<String, (oneshot::Sender<bool>, i64, Option<i32>)>>>;

// ── ChatCtx ────────────────────────────────────────────────────────────────

#[derive(Clone, Copy)]
pub(crate) struct ChatCtx {
    pub(crate) chat_id: ChatId,
    pub(crate) thread_id: Option<ThreadId>,
    /// Message id of the user's triggering message, used to anchor the
    /// bot's placeholder (and any subsequent streamed message) as a
    /// **reply** in group chats. In a busy group, the "waterfall"
    /// placeholder otherwise gets scrolled off-screen by unrelated
    /// chatter and the operator can't see the live tool/thinking
    /// stream. Threading it as a reply keeps the anchor link visible
    /// next to their original message. `None` in private chats (the
    /// UI is already 1:1, reply-threading just adds noise there) and
    /// for callback-driven flows.
    pub(crate) reply_to: Option<MessageId>,
}

impl ChatCtx {
    pub(crate) fn from_msg(msg: &Message) -> Self {
        // Always reply-thread: in groups for context anchoring,
        // in DMs for visual question→answer pairing.
        Self {
            chat_id: msg.chat.id,
            thread_id: msg.thread_id,
            reply_to: Some(msg.id),
        }
    }

    pub(crate) fn from_callback(q: &CallbackQuery) -> Self {
        let (chat_id, thread_id) = match &q.message {
            Some(msg) => {
                let cid = msg.chat().id;
                let tid = msg.regular_message().and_then(|m| m.thread_id);
                (cid, tid)
            }
            None => (ChatId(0), None),
        };
        Self {
            chat_id,
            thread_id,
            reply_to: None,
        }
    }

    pub(crate) fn raw_thread_id(&self) -> Option<i32> {
        self.thread_id.map(|tid| tid.0.0)
    }
}

// ── SendExt trait ──────────────────────────────────────────────────────────

pub(crate) trait SendExt {
    fn maybe_thread(self, thread_id: Option<ThreadId>) -> Self;
    /// Attach a `reply_parameters` pointing at `reply_to` if it is
    /// `Some`. Used to anchor the bot's "⏳ waterfall" placeholder (and
    /// any finalization message) as a reply to the user's triggering
    /// message in group chats — see `ChatCtx::reply_to` for rationale.
    /// `allow_sending_without_reply` is always set so we don't hard-fail
    /// if the user deleted their message mid-flight.
    fn maybe_reply_to(self, reply_to: Option<MessageId>) -> Self;
}

impl SendExt for teloxide::requests::JsonRequest<teloxide::payloads::SendMessage> {
    fn maybe_thread(self, thread_id: Option<ThreadId>) -> Self {
        match thread_id {
            Some(tid) => self.message_thread_id(tid),
            None => self,
        }
    }

    fn maybe_reply_to(self, reply_to: Option<MessageId>) -> Self {
        match reply_to {
            Some(mid) => {
                use teloxide::types::ReplyParameters;
                let params = ReplyParameters::new(mid).allow_sending_without_reply();
                self.reply_parameters(params)
            }
            None => self,
        }
    }
}

impl SendExt for teloxide::requests::MultipartRequest<teloxide::payloads::SendDocument> {
    fn maybe_thread(self, thread_id: Option<ThreadId>) -> Self {
        match thread_id {
            Some(tid) => self.message_thread_id(tid),
            None => self,
        }
    }

    fn maybe_reply_to(self, reply_to: Option<MessageId>) -> Self {
        match reply_to {
            Some(mid) => {
                use teloxide::types::ReplyParameters;
                let params = ReplyParameters::new(mid).allow_sending_without_reply();
                self.reply_parameters(params)
            }
            None => self,
        }
    }
}

// ── Safe send (PLAN_TG_SAFE_SEND_v1) ──────────────────────────────────
//
// ONE strategy for all outgoing text:
//   ≤ 4096 bytes  → send_message (normal)
//   ≤ 8192 bytes  → split into 2 chunks, send sequentially
//   > 8192 bytes  → tail-4000 in chat + full text as HTML attachment

const SAFE_SEND_FILE_THRESHOLD: usize = MAX_TG_MSG * 2;
const SAFE_SEND_TAIL_BUDGET: usize = 3900; // leave room for footer

/// Take the **last** `limit` bytes of `s`, char-boundary safe.
/// Prepends `…` when truncated.
pub(crate) fn tail_truncate(s: &str, limit: usize) -> String {
    if s.len() <= limit {
        return s.to_string();
    }
    let start = s.len().saturating_sub(limit);
    let start = s.ceil_char_boundary(start);
    format!("\u{2026}{}", &s[start..])
}

/// Wrap plain text into a minimal dark-themed HTML document.
pub(crate) fn wrap_html_doc(text: &str, title: &str) -> Vec<u8> {
    format!(
        r#"<!DOCTYPE html>
<html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>{title}</title>
<style>
:root {{ --bg:#1e1e2e; --fg:#cdd6f4; }}
body {{ background:var(--bg); color:var(--fg); font-family:monospace; font-size:14px;
  line-height:1.6; padding:24px; max-width:900px; margin:0 auto; white-space:pre-wrap; word-break:break-word; }}
@media(prefers-color-scheme:light) {{ :root {{ --bg:#eff1f5; --fg:#4c4f69; }} }}
</style></head><body>{body}</body></html>"#,
        title = markup::escape_html(title),
        body = markup::escape_html(text),
    )
    .into_bytes()
}

/// Unified safe-send: handles any length of text.
///
/// - ≤ 4096: single `send_message`
/// - ≤ 8192: split into 2 chunks
/// - > 8192: tail-truncated message + full text as HTML attachment
pub(crate) async fn safe_send(
    bot: &Bot,
    ctx: &ChatCtx,
    text: String,
    mode: Option<teloxide::types::ParseMode>,
) -> ResponseResult<Message> {
    // Short path: fits in one message.
    if text.len() <= MAX_TG_MSG {
        let mut req = bot
            .send_message(ctx.chat_id, &text)
            .maybe_thread(ctx.thread_id);
        if let Some(m) = mode {
            req = req.parse_mode(m);
        }
        return req.await;
    }

    // Medium path: split into 2 chunks.
    if text.len() <= SAFE_SEND_FILE_THRESHOLD {
        let chunks = split_html(&text, MAX_TG_MSG - 100);
        let mut last_msg = None;
        for chunk in &chunks {
            let mut req = bot
                .send_message(ctx.chat_id, chunk.as_str())
                .maybe_thread(ctx.thread_id);
            if let Some(m) = mode {
                req = req.parse_mode(m);
            }
            match req.await {
                Ok(m) => last_msg = Some(m),
                Err(e) => {
                    // HTML parse error → retry without parse_mode.
                    tracing::warn!("safe_send chunk (mode={mode:?}) failed: {e}, retrying plain");
                    let fallback = bot
                        .send_message(ctx.chat_id, chunk.as_str())
                        .maybe_thread(ctx.thread_id)
                        .await;
                    if let Ok(m) = fallback {
                        last_msg = Some(m);
                    }
                }
            }
        }
        // Return last successful message, or fabricate an error.
        return last_msg.ok_or_else(|| {
            teloxide::RequestError::Api(teloxide::ApiError::Unknown("all chunks failed".into()))
        });
    }

    // Long path: tail in chat + full HTML attachment.
    let tail = tail_truncate(&text, SAFE_SEND_TAIL_BUDGET);
    let visible = format!("{tail}\n\n\u{1f4c4} <i>Full text attached</i>");
    let mut req = bot
        .send_message(ctx.chat_id, &visible)
        .maybe_thread(ctx.thread_id)
        .parse_mode(teloxide::types::ParseMode::Html);
    if let Some(teloxide::types::ParseMode::Html) = mode {
        // already set
    } else {
        // Force HTML for the footer italic; tail is plain text anyway.
        req = req.parse_mode(teloxide::types::ParseMode::Html);
    }
    let msg = req.await?;

    // Attach full text as HTML file.
    let doc = wrap_html_doc(&text, "Full message");
    let input = teloxide::types::InputFile::memory(doc).file_name("message.html");
    if let Err(e) = bot
        .send_document(ctx.chat_id, input)
        .caption("\u{1f4c4} Full text")
        .maybe_thread(ctx.thread_id)
        .await
    {
        tracing::warn!("safe_send: send_document failed: {e}");
    }
    Ok(msg)
}

/// Send a plain-text message. Delegates to [`safe_send`].
pub(crate) async fn reply_text(
    bot: &Bot,
    ctx: &ChatCtx,
    text: impl Into<String>,
) -> ResponseResult<Message> {
    safe_send(bot, ctx, text.into(), None).await
}

/// Send an HTML-formatted message. Delegates to [`safe_send`].
pub(crate) async fn reply_html(
    bot: &Bot,
    ctx: &ChatCtx,
    text: impl Into<String>,
) -> ResponseResult<Message> {
    safe_send(
        bot,
        ctx,
        text.into(),
        Some(teloxide::types::ParseMode::Html),
    )
    .await
}

/// Send an HTML message with an inline keyboard.
/// If text overflows, keyboard is dropped and [`safe_send`] handles it.
pub(crate) async fn reply_html_kb(
    bot: &Bot,
    ctx: &ChatCtx,
    text: impl Into<String>,
    kb: teloxide::types::InlineKeyboardMarkup,
) -> ResponseResult<Message> {
    let text = text.into();
    if text.len() <= MAX_TG_MSG {
        return bot
            .send_message(ctx.chat_id, text)
            .parse_mode(teloxide::types::ParseMode::Html)
            .reply_markup(kb)
            .maybe_thread(ctx.thread_id)
            .await;
    }
    // Overflow: keyboard doesn't survive chunking/attachment. Drop it.
    safe_send(bot, ctx, text, Some(teloxide::types::ParseMode::Html)).await
}

pub(crate) async fn send_typing_raw(
    client: &reqwest::Client,
    base: &str,
    chat_id: i64,
    thread_id: Option<i32>,
) {
    let url = format!("{base}/sendChatAction");

    let mut body = serde_json::json!({
        "chat_id": chat_id,
        "action": "typing"
    });
    if let Some(tid) = thread_id {
        body["message_thread_id"] = serde_json::json!(tid);
    }

    match client.post(&url).json(&body).send().await {
        Ok(resp) => {
            if !resp.status().is_success() {
                tracing::debug!(
                    chat_id,
                    ?thread_id,
                    status = %resp.status(),
                    "typing: HTTP error"
                );
            } else if let Ok(json) = resp.json::<serde_json::Value>().await
                && json["ok"].as_bool() != Some(true)
            {
                tracing::warn!(chat_id, ?thread_id, "typing: API error: {}", json);
            }
        }
        Err(e) => tracing::debug!(chat_id, ?thread_id, "typing: network error: {e}"),
    }
}

// ── Static globals ─────────────────────────────────────────────────────────

pub(crate) static SLASH_HINT_SHOWN: LazyLock<tokio::sync::RwLock<HashSet<i64>>> =
    LazyLock::new(|| tokio::sync::RwLock::new(HashSet::new()));
// T3.3 (PLAN_RESEARCH_AGENT_FLOW_v1): PENDING_CLARIFICATIONS map removed.
// Was a per-(chat,thread) HashMap of paused research runs awaiting a
// clarification reply. Replaced by single-mechanism flow: just send a new
// message or `/research run <id>` to relaunch.
pub(crate) type ModelSwitchMap = HashMap<String, naked_tg::model_switch::SharedModelSwitch>;
pub(crate) static MODEL_SWITCHES: LazyLock<tokio::sync::RwLock<ModelSwitchMap>> =
    LazyLock::new(|| tokio::sync::RwLock::new(HashMap::new()));
/// Steer senders: (chat_id, thread_id) → Sender<SteerMessage>.
/// Populated when a streaming turn starts, removed when it ends.
pub(crate) type SteerSenderMap =
    HashMap<String, tokio::sync::mpsc::Sender<naked_core::types::SteerMessage>>;
pub(crate) static STEER_SENDERS: LazyLock<tokio::sync::RwLock<SteerSenderMap>> =
    LazyLock::new(|| tokio::sync::RwLock::new(HashMap::new()));

/// PLAN_NEXT_SESSION §A.2 S6 — "↩️ Принято" temp confirmations
/// awaiting deletion. Keyed by `(chat_id, thread_id, user_msg_id)`,
/// where `user_msg_id` is the original Telegram message_id of the
/// user's steer text. Value is `(chat_id, ack_message_id)` so the
/// streaming-side handler can issue `bot.delete_message(chat, ack)`
/// without needing to look up the chat again.
///
/// Inserted by `message_handler::handle_text` right after the bot
/// posts "↩️ Принято — доставлю между шагами".
/// Removed (and the message deleted) by the streaming pipeline when
/// `AgentEvent::SteerReceived { msg_ids, .. }` arrives — the very
/// moment the steer is in `history` and the model is guaranteed to
/// see it on the next iteration.
pub(crate) type SteerAckMap =
    HashMap<(String, i32), (teloxide::types::ChatId, teloxide::types::MessageId)>;
pub(crate) static STEER_ACK_IDS: LazyLock<tokio::sync::RwLock<SteerAckMap>> =
    LazyLock::new(|| tokio::sync::RwLock::new(HashMap::new()));

/// PLAN_NEXT_SESSION §A.2 (auto-fire UX) — in-stream control cards
/// (⏹ Стоп / ⏩ Send now buttons). Keyed by `(chat_id,
/// thread_id)`. Inserted by `streaming_mod::pipeline::stream_response`
/// right after the placeholder is sent; removed and the message
/// deleted on Idle / Error / model-switch / abort.
///
/// One control card per active streaming turn — we never want to
/// stack two (would let user click an orphan button and confuse the
/// callback router).
pub(crate) type ControlCardMap =
    HashMap<String, (teloxide::types::ChatId, teloxide::types::MessageId)>;
pub(crate) static CONTROL_CARDS: LazyLock<tokio::sync::RwLock<ControlCardMap>> =
    LazyLock::new(|| tokio::sync::RwLock::new(HashMap::new()));

/// BUG_REGISTRY D-VALIDATE-IP-TOKENS (B37 stream guard).
///
/// Populated at boot by `wiring::populate_novnc_ip_allowlist` from the
/// `novnc.sh url` JSON output. Stream-side code reads it to validate
/// outgoing assistant messages that mention noVNC — if a message
/// contains an `IP:port` token NOT in this list, we bump
/// `IP_TOKEN_HALLUCINATION_COUNT` and log WARN.
///
/// `std::sync::RwLock` (not `tokio::sync::RwLock`) because the access
/// pattern is many-readers, one-writer (only boot), and we want
/// non-blocking reads from the stream hot path.
pub(crate) static NOVNC_IP_ALLOWLIST: LazyLock<std::sync::RwLock<Vec<String>>> =
    LazyLock::new(|| std::sync::RwLock::new(Vec::new()));

/// F4 of PLAN_NEXT_SESSION (2026-05-10): process-wide handle to the
/// shared `LivenessRegistry`. Set exactly once by `wiring.rs::build`
/// (alongside the same `Arc` it stashes in `WiredBot.liveness`); read
/// by the `/health` slash-command and — when we add it later — by
/// the `/metrics` text renderer.
///
/// `OnceLock` (rather than `LazyLock`) because there is exactly one
/// correct value and it must be installed by wiring — we do NOT
/// want a lazy default registry that silently shadows the real one.
pub(crate) static LIVENESS_REGISTRY: std::sync::OnceLock<
    std::sync::Arc<naked_core::liveness::LivenessRegistry>,
> = std::sync::OnceLock::new();

/// S7 / RC-14: loaded config hash captured immediately after
/// `Config::load_with_source_bytes` at boot. Read by `/health` and
/// `/metrics`; observability only, never used to refuse boot.
pub(crate) static CONFIG_LOADED_HASH: std::sync::OnceLock<naked_tg::config_hash::ConfigLoadedHash> =
    std::sync::OnceLock::new();

/// F4: process start time. Stamped on first read (effectively when
/// the bot boots and any code path touches `shared`). `/health`
/// renders `now - PROCESS_STARTED_AT` as the uptime line.
pub(crate) static PROCESS_STARTED_AT: LazyLock<std::time::Instant> =
    LazyLock::new(std::time::Instant::now);

/// Global rate limiter instance — accessible from commands.rs for /metrics.
pub(crate) static RATE_LIMITER: LazyLock<naked_tg::rate_limit::RateLimiter> =
    LazyLock::new(naked_tg::rate_limit::RateLimiter::new);

/// CYCLE 2 Step 2: in-memory, kind-agnostic live Run control index.
pub(crate) static RUN_REGISTRY: LazyLock<naked_tg::run_registry::RunRegistry> =
    LazyLock::new(naked_tg::run_registry::RunRegistry::new);

// ── Message helper functions ────────────────────────────────────────────────

/// Extract `@username` (or first_name fallback) from a User-bearing message.
pub(crate) fn sender_label(msg: &Message) -> String {
    msg.from
        .as_ref()
        .and_then(|u| u.username.as_deref().map(|s| format!("@{s}")))
        .or_else(|| msg.from.as_ref().map(|u| u.first_name.clone()))
        .unwrap_or_else(|| "unknown".to_string())
}

/// True when message comes from a group/supergroup/channel (not a 1-on-1 chat).
pub(crate) fn is_group_chat(msg: &Message) -> bool {
    use teloxide::types::ChatKind;
    matches!(msg.chat.kind, ChatKind::Public(_))
}

/// Format a Telegram `reply_to_message` as a blockquote for LLM context.
/// Returns `None` when the message is not a reply.
///
/// Format follows zeroclaws `src/channels/telegram.rs::extract_reply_context`:
///   `> @sender[ marker]:\n> line1\n> line2`
/// with a bot marker `[your previous message]` so the model sees its own
/// cited output clearly in 1-on-1 chats.
pub(crate) fn extract_reply_context(msg: &Message) -> Option<String> {
    let reply = msg.reply_to_message()?;

    let sender = reply
        .from
        .as_ref()
        .and_then(|u| u.username.as_deref().map(|s| format!("@{s}")))
        .or_else(|| reply.from.as_ref().map(|u| u.first_name.clone()))
        .unwrap_or_else(|| "unknown".to_string());

    let is_bot = reply.from.as_ref().map(|u| u.is_bot).unwrap_or(false);
    let marker = if is_bot {
        " [your previous message]"
    } else {
        ""
    };

    // Text messages win; otherwise we describe media and include the user
    // caption if any (e.g. `[Photo: 'ship it']`) so the LLM sees intent.
    let caption = reply.caption().map(|c| c.trim()).filter(|c| !c.is_empty());
    let describe = |kind: &str| match caption {
        Some(c) => format!("[{kind}: {c}]"),
        None => format!("[{kind}]"),
    };
    let body = if let Some(t) = reply.text() {
        t.to_string()
    } else if reply.voice().is_some() {
        describe("Voice message")
    } else if reply.audio().is_some() {
        describe("Audio")
    } else if reply.photo().is_some() {
        describe("Photo")
    } else if reply.document().is_some() {
        describe("Document")
    } else if reply.video().is_some() {
        describe("Video")
    } else if reply.animation().is_some() {
        describe("Animation")
    } else if reply.sticker().is_some() {
        "[Sticker]".to_string()
    } else if let Some(c) = caption {
        format!("[Message: {c}]")
    } else {
        "[Message]".to_string()
    };

    let body = if body.chars().count() > REPLY_QUOTE_MAX_CHARS {
        let truncated: String = body.chars().take(REPLY_QUOTE_MAX_CHARS).collect();
        format!("{truncated}…")
    } else {
        body
    };

    let quoted: String = body
        .lines()
        .map(|l| format!("> {l}"))
        .collect::<Vec<_>>()
        .join("\n");

    Some(format!("> {sender}{marker}:\n{quoted}"))
}

// ── Access control ──────────────────────────────────────────────────────────

pub(crate) fn is_allowed(chat_id: i64, config: &Config) -> bool {
    if config.telegram.allowed_chat_ids.is_empty() {
        return false;
    }
    config.telegram.allowed_chat_ids.contains(&chat_id)
}

// ── Provider helpers ────────────────────────────────────────────────────────

pub(crate) fn provider_models(config: &Config, provider_name: &str) -> Vec<String> {
    config
        .providers
        .get(provider_name)
        .map(|pc| pc.models_with_aliases())
        .unwrap_or_default()
}

// ── Telegram markup helpers: delegate to markup module (DRY) ───────────────

pub(crate) fn escape_html(s: &str) -> String {
    markup::escape_html(s)
}

pub(crate) fn md_to_tg_html(text: &str) -> String {
    markup::md_to_tg_html(text)
}

pub(crate) fn split_html(text: &str, max_bytes: usize) -> Vec<String> {
    markup::split_html(text, max_bytes)
}

// ── Health check server ─────────────────────────────────────────────────────

pub(crate) async fn run_health_server(port: u16) {
    use tokio::io::AsyncWriteExt;
    let listener = match tokio::net::TcpListener::bind(("0.0.0.0", port)).await {
        Ok(l) => {
            tracing::info!("Health endpoint listening on :{port}");
            l
        }
        Err(e) => {
            tracing::warn!("Failed to bind health port {port}: {e}");
            return;
        }
    };
    loop {
        if let Ok((mut stream, _)) = listener.accept().await {
            let body = r#"{"status":"ok"}"#;
            let response = format!(
                "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\r\n{}",
                body.len(),
                body
            );
            let _ = stream.write_all(response.as_bytes()).await;
        }
    }
}

#[cfg(test)]
mod safe_send_tests {
    use super::*;

    // ── tail_truncate ─────────────────────────────────────────────

    #[test]
    fn tail_within_limit_unchanged() {
        assert_eq!(tail_truncate("hello", 100), "hello");
    }

    #[test]
    fn tail_at_limit_unchanged() {
        let s = "a".repeat(4096);
        assert_eq!(tail_truncate(&s, 4096), s);
    }

    #[test]
    fn tail_over_limit_keeps_end() {
        let s = format!("{}TAIL", "x".repeat(5000));
        let t = tail_truncate(&s, 100);
        assert!(t.len() <= 103); // 100 + … (3 bytes)
        assert!(t.ends_with("TAIL"), "must keep tail: {t}");
        assert!(t.starts_with('…'));
    }

    #[test]
    fn tail_multibyte_safe() {
        let s = "ю".repeat(1000); // 2000 bytes
        let t = tail_truncate(&s, 500);
        assert!(t.len() <= 503);
        assert!(t.starts_with('…'));
    }

    // ── wrap_html_doc ───────────────────────────────────────────

    #[test]
    fn wrap_html_doc_contains_doctype_and_body() {
        let doc = wrap_html_doc("hello world", "Test");
        let html = String::from_utf8(doc).unwrap();
        assert!(html.contains("<!DOCTYPE html>"));
        assert!(html.contains("hello world"));
        assert!(html.contains("Test")); // title
    }

    #[test]
    fn wrap_html_doc_escapes_html_entities() {
        let doc = wrap_html_doc("<script>alert(1)</script>", "XSS");
        let html = String::from_utf8(doc).unwrap();
        assert!(!html.contains("<script>"));
        assert!(html.contains("&lt;script&gt;"));
    }

    // ── structural sentinels ─────────────────────────────────────

    #[test]
    fn reply_text_delegates_to_safe_send() {
        let src = include_str!("shared.rs");
        // Split at #[cfg(test)] to check prod code only.
        let prod = src.split("#[cfg(test)]").next().unwrap_or("");
        assert!(
            prod.contains("safe_send(bot, ctx, text.into(), None)"),
            "reply_text must delegate to safe_send"
        );
    }

    #[test]
    fn send_long_text_removed() {
        let src = include_str!("shared.rs");
        let prod = src.split("#[cfg(test)]").next().unwrap_or("");
        assert!(
            !prod.contains("fn send_long_text"),
            "send_long_text must not exist in shared.rs (use safe_send)"
        );
    }

    #[test]
    fn tg_truncate_removed() {
        let src = include_str!("shared.rs");
        let prod = src.split("#[cfg(test)]").next().unwrap_or("");
        assert!(
            !prod.contains("fn tg_truncate"),
            "tg_truncate must not exist (replaced by tail_truncate + safe_send)"
        );
    }
}
