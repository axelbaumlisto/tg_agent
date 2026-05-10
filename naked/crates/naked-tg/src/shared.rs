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
pub(crate) use crate::channel_map::{ChannelSessionMap, format_tg_channel_id};

// ── Re-export naked_tg library helpers used throughout ──────────────────────
pub(crate) use naked_tg::helpers::parse_interval;
pub(crate) use naked_tg::markup::{self, MAX_TG_MSG as TG_MSG_LIMIT};
pub(crate) use naked_tg::memory_scheduler;
pub(crate) use naked_tg::research_html::{ReportMeta, render_report_html};
pub(crate) use naked_tg::research_scheduler;
pub(crate) use naked_tg::research_ui::{
    HeartbeatProgress, PendingClarification, keyboard_after_complete,
    keyboard_paused_awaiting_clarification, keyboard_stop, render_waterfall,
};

// ── Constants ──────────────────────────────────────────────────────────────

pub(crate) const MAX_TG_MSG: usize = TG_MSG_LIMIT;
pub(crate) const TYPING_INTERVAL: Duration = Duration::from_secs(3);
pub(crate) const PERMISSION_TIMEOUT: Duration = Duration::from_secs(120);
pub(crate) const REASONING_TAIL: usize = 600;
pub(crate) const TOOL_WINDOW: usize = 5;
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

// ── Reply helpers (DRY: replaces 43× repeated send_message chains) ──────────

/// Send a plain-text message. Handles thread_id automatically.
pub(crate) async fn reply_text(
    bot: &Bot,
    ctx: &ChatCtx,
    text: impl Into<String>,
) -> ResponseResult<Message> {
    bot.send_message(ctx.chat_id, text)
        .maybe_thread(ctx.thread_id)
        .await
}

/// Send an HTML-formatted message. Handles thread_id + ParseMode::Html.
pub(crate) async fn reply_html(
    bot: &Bot,
    ctx: &ChatCtx,
    text: impl Into<String>,
) -> ResponseResult<Message> {
    bot.send_message(ctx.chat_id, text)
        .parse_mode(teloxide::types::ParseMode::Html)
        .maybe_thread(ctx.thread_id)
        .await
}

/// Send an HTML message with an inline keyboard.
pub(crate) async fn reply_html_kb(
    bot: &Bot,
    ctx: &ChatCtx,
    text: impl Into<String>,
    kb: teloxide::types::InlineKeyboardMarkup,
) -> ResponseResult<Message> {
    bot.send_message(ctx.chat_id, text)
        .parse_mode(teloxide::types::ParseMode::Html)
        .reply_markup(kb)
        .maybe_thread(ctx.thread_id)
        .await
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
pub(crate) type PendingClarificationMap = HashMap<(i64, Option<i32>), PendingClarification>;
pub(crate) static PENDING_CLARIFICATIONS: LazyLock<tokio::sync::RwLock<PendingClarificationMap>> =
    LazyLock::new(|| tokio::sync::RwLock::new(HashMap::new()));
pub(crate) type ModelSwitchMap =
    HashMap<(i64, Option<i32>), naked_tg::model_switch::SharedModelSwitch>;
pub(crate) static MODEL_SWITCHES: LazyLock<tokio::sync::RwLock<ModelSwitchMap>> =
    LazyLock::new(|| tokio::sync::RwLock::new(HashMap::new()));
pub(crate) type QueueCountMap = HashMap<(i64, Option<i32>), Arc<std::sync::atomic::AtomicUsize>>;
pub(crate) static QUEUE_COUNTS: LazyLock<tokio::sync::RwLock<QueueCountMap>> =
    LazyLock::new(|| tokio::sync::RwLock::new(HashMap::new()));

/// Steer senders: (chat_id, thread_id) → Sender<SteerMessage>.
/// Populated when a streaming turn starts, removed when it ends.
pub(crate) type SteerSenderMap =
    HashMap<(i64, Option<i32>), tokio::sync::mpsc::Sender<naked_core::types::SteerMessage>>;
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
    HashMap<(i64, Option<i32>, i32), (teloxide::types::ChatId, teloxide::types::MessageId)>;
pub(crate) static STEER_ACK_IDS: LazyLock<tokio::sync::RwLock<SteerAckMap>> =
    LazyLock::new(|| tokio::sync::RwLock::new(HashMap::new()));

/// Global rate limiter instance — accessible from commands.rs for /metrics.
pub(crate) static RATE_LIMITER: LazyLock<naked_tg::rate_limit::RateLimiter> =
    LazyLock::new(naked_tg::rate_limit::RateLimiter::new);

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
