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

    // BUG_REGISTRY D-VALIDATE-IP-TOKENS (B37 stream guard):
    // scan outgoing HTML for hallucinated noVNC IPs before sending.
    // No-op if message doesn't mention vnc / VNC / noVNC keyword.
    validate_novnc_ip_tokens(html);

    // B45 wire-up (2026-05-13): credential redactor.
    // Scrubs `api_key=X`, `Bearer X`, `Authorization:`, `password=`, etc.
    // before forwarding to Telegram. Returns original string if no match
    // (fast-path) — only allocates when something is actually redacted.
    let redacted = naked_core::research::tool::redact::scan_and_redact(html);
    let html: &str = if redacted == html {
        html
    } else {
        crate::metrics::record_redaction_applied();
        tracing::warn!(
            target: "naked_tg::redact",
            chat = chat_id.0,
            original_len = html.len(),
            redacted_len = redacted.len(),
            "B45: credential pattern stripped from outgoing message"
        );
        &redacted
    };

    // PLAN_TG_INTERLEAVED_v1 Q4: head-truncate path.
    // If render_final dropped any chronological events to fit in one
    // message, ALSO attach the full-history HTML document so the user
    // can review what was cut. Inline still shows the last (newest)
    // part — head-truncate keeps the tail.
    let dropped = view
        .last_dropped_events
        .load(std::sync::atomic::Ordering::Relaxed);

    // Short: fits in one message
    if html.len() <= MAX_TG_MSG {
        edit_with_retry(&bot, chat_id, msg_id, html, true).await;
        if dropped > 0 {
            // Inline was head-truncated — attach full timeline.
            let html_doc = render_html_document(view);
            let input_file = teloxide::types::InputFile::memory(html_doc)
                .file_name("transcript.html");
            if let Err(e) = bot
                .send_document(chat_id, input_file)
                .caption(format!("📄 Full timeline (+{dropped} earlier events)"))
                .maybe_thread(ctx.thread_id)
                .maybe_reply_to(ctx.reply_to)
                .await
            {
                tracing::warn!("head-truncate full-timeline send_document failed: {e}");
            }
        }
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
    // BUG_REGISTRY B36: format_input_preview returns RAW string (may
    // contain `<<` from bash heredoc, `<>` from comparisons, `&` from URLs).
    // Without HTML-escape, Telegram rejects the message with HTTP 400
    // "can't parse entities", `bot.send_message().is_err()`, and
    // ask_permission returns false — looking exactly like the user
    // pressed Deny. Real bug: user never saw the permission card.
    let preview = escape_html(&format_input_preview(input, 200));
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

    if let Err(e) = &sent {
        // BUG_REGISTRY B23 (silent let-else) + B36 fallout:
        // log loudly so operator can see HTML-injection / TG API failures
        // instead of seeing fake "Permission denied by user" tool results.
        tracing::error!(
            tool = %tool_name,
            error = %e,
            preview = %preview,
            "permission card send FAILED — user will see no prompt and \
             tool will be auto-denied. Likely HTML injection in preview."
        );
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

/// BUG_REGISTRY D-VALIDATE-IP-TOKENS (B37 stream guard).
///
/// Scans an outgoing HTML/text message for `IP:port` tokens and
/// compares against `NOVNC_IP_ALLOWLIST` (populated at boot from
/// `novnc.sh url`). If the message mentions a noVNC-related keyword
/// AND contains an IP token that's NOT in the allow-list, bumps
/// `IP_TOKEN_HALLUCINATION_COUNT` and logs WARN. Soft-warn — does not
/// block the send.
///
/// Trigger keywords: "vnc", "VNC", "noVNC", "novnc" (case-insensitive
/// via lowercase comparison). Without one of these we don't run the
/// check (would false-positive on every screenshot URL etc.).
///
/// Allow-list is permissive: matches if the candidate IP appears
/// anywhere in any allow-list entry (so `100.80.12.120` matches
/// `100.80.12.120:6080`).
pub(crate) fn validate_novnc_ip_tokens(html: &str) {
    let lc = html.to_ascii_lowercase();
    if !(lc.contains("vnc") || lc.contains("novnc")) {
        return;
    }
    // Cheap manual IP scanner: walk bytes, group sequences of digits
    // separated by '.' — when we see 4 groups, we have an IPv4. Append
    // `:port` if followed by `:<digits>`. Avoids regex dep.
    let bytes = html.as_bytes();
    let mut i = 0;
    let mut suspicious: Vec<String> = Vec::new();
    while i < bytes.len() {
        if !bytes[i].is_ascii_digit() {
            i += 1;
            continue;
        }
        // Try to consume octet.dotted.notation starting here.
        let start = i;
        let mut octets = 0;
        let mut cursor = i;
        while octets < 4 && cursor < bytes.len() {
            let octet_start = cursor;
            while cursor < bytes.len() && bytes[cursor].is_ascii_digit() {
                cursor += 1;
            }
            if cursor == octet_start {
                break;
            }
            octets += 1;
            if octets < 4 {
                if cursor < bytes.len() && bytes[cursor] == b'.' {
                    cursor += 1;
                } else {
                    break;
                }
            }
        }
        if octets == 4 {
            // Got an IPv4. Try to consume :port.
            let ip_end = cursor;
            let token_end = if cursor < bytes.len() && bytes[cursor] == b':' {
                let port_start = cursor + 1;
                let mut p = port_start;
                while p < bytes.len() && bytes[p].is_ascii_digit() {
                    p += 1;
                }
                if p > port_start { p } else { ip_end }
            } else {
                ip_end
            };
            let token = &html[start..token_end];
            // Compare against allow-list.
            let ok = {
                let guard = crate::shared::NOVNC_IP_ALLOWLIST.read();
                match guard {
                    Ok(list) => {
                        if list.is_empty() {
                            // Allow-list not populated — fail-open
                            true
                        } else {
                            list.iter().any(|entry| {
                                // Match if token is prefix or substring of allow entry
                                entry.contains(token) || token.contains(entry.as_str())
                            })
                        }
                    }
                    Err(_) => true, // poisoned lock — fail-open
                }
            };
            if !ok {
                suspicious.push(token.to_string());
            }
            i = token_end;
            continue;
        }
        i = cursor.max(start + 1);
    }
    if !suspicious.is_empty() {
        naked_core::types::IP_TOKEN_HALLUCINATION_COUNT.fetch_add(
            suspicious.len() as u64,
            std::sync::atomic::Ordering::Relaxed,
        );
        tracing::warn!(
            suspicious = ?suspicious,
            "B37: outgoing message mentions noVNC AND contains IP:port tokens \
             NOT in the boot-cached allow-list from novnc.sh url. Likely \
             model-hallucinated credentials. Cross-check against `novnc.sh url`."
        );
    }
}
