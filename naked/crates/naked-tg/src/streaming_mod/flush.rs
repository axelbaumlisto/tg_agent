//! Flush and send functions: live edit, final send, permission dialog.

use super::*;

const FILE_THRESHOLD: usize = MAX_TG_MSG * 2;
const SUMMARY_CHARS: usize = 500;

/// Edit with retry: if rate-limited, waits the indicated duration and retries.
pub(crate) async fn edit_with_retry(
    bot: &Bot,
    chat_id: ChatId,
    msg_id: MessageId,
    text: &str,
    parse_html: bool,
) -> bool {
    let rl = &*RATE_LIMITER;
    if parse_html {
        if rl.edit_html(bot, chat_id, msg_id, text).await {
            return true;
        }
        // HTML failed — try plain text
        let plain = strip_html_tags(text);
        return rl.edit_plain(bot, chat_id, msg_id, &plain).await;
    }
    rl.edit_plain(bot, chat_id, msg_id, text).await
}

pub(crate) async fn send_final(
    bot: Bot,
    ctx: ChatCtx,
    msg_id: MessageId,
    html: &str,
    view: &CompositeView,
) {
    let chat_id = ctx.chat_id;

    // Short: fits in one message
    if html.len() <= MAX_TG_MSG {
        edit_with_retry(&bot, chat_id, msg_id, html, true).await;
        return;
    }

    // Medium: fits in 2 chunks
    if html.len() <= FILE_THRESHOLD {
        let chunks = split_html(html, MAX_TG_MSG - 100);
        if let Some(first) = chunks.first() {
            edit_with_retry(&bot, chat_id, msg_id, first, true).await;
        }
        for chunk in chunks.iter().skip(1) {
            let res = bot
                .send_message(chat_id, chunk.as_str())
                .parse_mode(ParseMode::Html)
                .maybe_thread(ctx.thread_id)
                .maybe_reply_to(ctx.reply_to)
                .await;
            if let Err(e) = res {
                tracing::warn!("send chunk (HTML) failed: {e}, retrying plain text");
                if let Err(e2) = bot
                    .send_message(chat_id, chunk.as_str())
                    .maybe_thread(ctx.thread_id)
                    .maybe_reply_to(ctx.reply_to)
                    .await
                {
                    tracing::error!("send chunk (plain) also failed: {e2}");
                }
            }
        }
        return;
    }

    let summary = view.render_summary(SUMMARY_CHARS);
    edit_with_retry(&bot, chat_id, msg_id, &summary, true).await;

    let html_doc = render_html_document(view);
    let input_file = teloxide::types::InputFile::memory(html_doc).file_name("response.html");
    if let Err(e) = bot
        .send_document(chat_id, input_file)
        .caption("📄 Full response")
        .maybe_thread(ctx.thread_id)
        .maybe_reply_to(ctx.reply_to)
        .await
    {
        tracing::warn!("send_document failed: {e}");
    }
}

pub(crate) async fn send_long_text(
    bot: &Bot,
    ctx: ChatCtx,
    text: &str,
) -> Result<(), teloxide::RequestError> {
    if text.len() <= MAX_TG_MSG {
        reply_text(bot, &ctx, text).await?;
        return Ok(());
    }
    for chunk in split_html(text, MAX_TG_MSG - 100) {
        reply_text(bot, &ctx, chunk).await?;
    }
    Ok(())
}

// ── Permission prompt ───────────────────────────────────────────────────────

pub(crate) async fn ask_permission(
    bot: &Bot,
    ctx: ChatCtx,
    call_id: &str,
    tool_name: &str,
    input: &serde_json::Value,
    permission: &Permission,
    pending: &PendingPermissions,
) -> bool {
    let level = match permission {
        Permission::WorkspaceWrite => "write",
        Permission::Dangerous => "dangerous",
        Permission::ReadOnly => "read",
    };
    let preview = format_input_preview(input, 200);
    let text = format!(
        "🔐 <b>{}</b> [{level}]({preview})\n\
         <i>💡 /yolo = авто-approve | read_file/search — авто</i>",
        escape_html(tool_name),
    );

    let (tx, rx) = oneshot::channel();
    let chat_id_raw = ctx.chat_id.0;
    let tid_raw = ctx.raw_thread_id();
    pending
        .write()
        .await
        .insert(call_id.to_string(), (tx, chat_id_raw, tid_raw));

    let keyboard = InlineKeyboardMarkup::new(vec![vec![
        InlineKeyboardButton::callback("✅ Allow", format!("p:{call_id}:allow")),
        InlineKeyboardButton::callback("❌ Deny", format!("p:{call_id}:deny")),
        InlineKeyboardButton::callback("⚡ YOLO", format!("p:{call_id}:yolo")),
    ]]);

    let sent = bot
        .send_message(ctx.chat_id, &text)
        .parse_mode(ParseMode::Html)
        .reply_markup(keyboard)
        .maybe_thread(ctx.thread_id)
        .await;

    if sent.is_err() {
        pending.write().await.remove(call_id);
        return false;
    }

    match tokio::time::timeout(PERMISSION_TIMEOUT, rx).await {
        Ok(Ok(allowed)) => allowed,
        _ => {
            pending.write().await.remove(call_id);
            // FIX-2: Notify user that permission timed out.
            if let Ok(msg) = &sent {
                let text = format!(
                    "⏱ <b>{}</b> — время ожидания истекло ({}s). Запрос отменён.",
                    escape_html(tool_name),
                    PERMISSION_TIMEOUT.as_secs()
                );
                RATE_LIMITER
                    .edit_html(bot, ctx.chat_id, msg.id, &text)
                    .await;
            }
            false
        }
    }
}

pub(crate) async fn flush_live(
    bot: &Bot,
    chat_id: ChatId,
    msg_id: MessageId,
    view: &CompositeView,
    last_sent: &mut String,
    html_broken: &mut bool,
) -> bool {
    let html = view.render_live();
    let trimmed = truncate_str(&html, MAX_TG_MSG - 50);
    if trimmed == *last_sent {
        return true;
    }
    *last_sent = trimmed.clone();

    // All edits go through the global rate limiter.
    let rl = &*RATE_LIMITER;

    if *html_broken {
        let plain = strip_html_tags(&trimmed);
        rl.edit_plain(bot, chat_id, msg_id, &plain).await;
        return true;
    }

    let ok = rl.edit_html(bot, chat_id, msg_id, &trimmed).await;
    if !ok {
        // Rate limiter returned false — either 429 (parked) or HTML error.
        // Try plain text as fallback.
        let plain = strip_html_tags(&trimmed);
        if !rl.edit_plain(bot, chat_id, msg_id, &plain).await {
            *last_sent = String::new(); // force retry next tick
            return false;
        }
        *html_broken = true;
    }
    true
}
