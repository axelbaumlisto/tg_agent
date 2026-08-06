//! Flush and send functions: live edit, final send, permission dialog.

use super::*;

const FILE_THRESHOLD: usize = MAX_TG_MSG * 2;
const LONG_ANSWER_SPLIT_UTF16_THRESHOLD: u64 = 32_768;
const SUMMARY_CHARS: usize = 500;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct SummaryAttachmentResult {
    summary_delivered: bool,
    attachment_delivered: bool,
}

impl SummaryAttachmentResult {
    fn outcome_for_full_answer(self) -> crate::metrics::FinalAnswerDeliveryOutcome {
        if !self.attachment_delivered {
            crate::metrics::FinalAnswerDeliveryOutcome::Failed
        } else if self.summary_delivered {
            crate::metrics::FinalAnswerDeliveryOutcome::Ok
        } else {
            crate::metrics::FinalAnswerDeliveryOutcome::Partial
        }
    }

    fn outcome_for_fallback(self) -> crate::metrics::FinalAnswerDeliveryOutcome {
        if !self.attachment_delivered {
            crate::metrics::FinalAnswerDeliveryOutcome::Failed
        } else {
            crate::metrics::FinalAnswerDeliveryOutcome::Partial
        }
    }
}

/// Edit with retry: if rate-limited, waits the indicated duration and retries.
#[allow(dead_code)] // available for non-critical edits that can tolerate drops
pub(crate) async fn edit_with_retry(
    bot: &Bot,
    chat_id: ChatId,
    msg_id: MessageId,
    text: &str,
    parse_html: bool,
) -> bool {
    _edit_impl(bot, chat_id, msg_id, text, parse_html, false).await
}

/// Like edit_with_retry but never drops — waits through rate limits.
pub(crate) async fn edit_must_deliver(
    bot: &Bot,
    chat_id: ChatId,
    msg_id: MessageId,
    text: &str,
    parse_html: bool,
) -> bool {
    _edit_impl(bot, chat_id, msg_id, text, parse_html, true).await
}

async fn _edit_impl(
    bot: &Bot,
    chat_id: ChatId,
    msg_id: MessageId,
    text: &str,
    parse_html: bool,
    must_deliver: bool,
) -> bool {
    let rl = &*RATE_LIMITER;
    if parse_html {
        let ok = if must_deliver {
            rl.edit_must_deliver(bot, chat_id, msg_id, text, true).await
        } else {
            rl.edit_html(bot, chat_id, msg_id, text).await
        };
        if ok {
            return true;
        }
        // HTML failed — try plain text
        let plain = strip_html_tags(text);
        return if must_deliver {
            rl.edit_must_deliver(bot, chat_id, msg_id, &plain, false)
                .await
        } else {
            rl.edit_plain(bot, chat_id, msg_id, &plain).await
        };
    }
    if must_deliver {
        rl.edit_must_deliver(bot, chat_id, msg_id, text, false)
            .await
    } else {
        rl.edit_plain(bot, chat_id, msg_id, text).await
    }
}

pub(crate) async fn send_final(
    bot: Bot,
    ctx: ChatCtx,
    msg_id: MessageId,
    html: &str,
    view: &CompositeView,
    tg_long_answer_fix_enabled: bool,
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
    let final_utf16_units = crate::metrics::record_final_answer_utf16_from_html(html);
    let final_truncated = view
        .last_final_answer_truncated
        .load(std::sync::atomic::Ordering::Relaxed);
    if final_truncated {
        crate::metrics::record_final_answer_truncated();
    }
    tracing::info!(
        final_utf16_units,
        html_len = html.len(),
        final_truncated,
        dropped_events = dropped,
        "metrics: final answer observed"
    );

    send_final_with_observed_budget(
        bot,
        ctx,
        msg_id,
        html,
        view,
        FinalDeliveryBudget {
            dropped,
            utf16_units: final_utf16_units,
            long_answer_fix_enabled: tg_long_answer_fix_enabled,
        },
    )
    .await;
}

#[derive(Clone, Copy)]
pub(crate) struct FinalDeliveryBudget {
    pub(crate) dropped: usize,
    pub(crate) utf16_units: u64,
    pub(crate) long_answer_fix_enabled: bool,
}

#[cfg(all(test, not(debug_assertions)))]
pub(crate) async fn send_final_with_budget_for_test(
    bot: Bot,
    ctx: ChatCtx,
    msg_id: MessageId,
    html: &str,
    view: &CompositeView,
    budget: FinalDeliveryBudget,
) {
    send_final_with_observed_budget(bot, ctx, msg_id, html, view, budget).await;
}

async fn send_final_with_observed_budget(
    bot: Bot,
    ctx: ChatCtx,
    msg_id: MessageId,
    html: &str,
    view: &CompositeView,
    budget: FinalDeliveryBudget,
) {
    let chat_id = ctx.chat_id;

    // Short: fits in one message. With the long-answer fix enabled,
    // Telegram's real budget is visible UTF-16 units after entity decoding and
    // tag stripping; the legacy path remains byte-gated for flag-off goldens.
    if fits_single_final_payload(html, budget.utf16_units, budget.long_answer_fix_enabled) {
        if !tg_payload_within_utf16_budget(html, "final single edit") {
            let result =
                send_summary_attachment(&bot, ctx, msg_id, view, html.len(), budget.dropped).await;
            crate::metrics::record_final_answer_delivery(result.outcome_for_fallback());
            return;
        }
        if !edit_must_deliver(&bot, chat_id, msg_id, html, true).await {
            tracing::warn!(
                html_len = html.len(),
                dropped_events = budget.dropped,
                "final answer single-message edit failed; falling back to summary attachment"
            );
            let result =
                send_summary_attachment(&bot, ctx, msg_id, view, html.len(), budget.dropped).await;
            crate::metrics::record_final_answer_delivery(result.outcome_for_fallback());
            return;
        }
        if budget.dropped > 0 {
            // Inline was head-truncated — attach full timeline.
            let html_doc = render_html_document(view);
            let input_file =
                teloxide::types::InputFile::memory(html_doc).file_name("transcript.html");
            if let Err(e) = bot
                .send_document(chat_id, input_file)
                .caption(format!(
                    "📄 Full timeline (+{} tool events)",
                    budget.dropped
                ))
                .maybe_thread(ctx.thread_id)
                .maybe_reply_to(ctx.reply_to)
                .await
            {
                crate::metrics::record_final_answer_attachment_send_failed();
                let safe = redact_for_log(&e);
                tracing::warn!("head-truncate full-timeline send_document failed: {safe}");
                crate::metrics::record_final_answer_delivery(
                    crate::metrics::FinalAnswerDeliveryOutcome::Partial,
                );
                return;
            }
        }
        crate::metrics::record_final_answer_delivery(
            crate::metrics::FinalAnswerDeliveryOutcome::Ok,
        );
        return;
    }

    // Medium: split into ordered Telegram messages. Flag-off keeps the legacy
    // byte threshold; flag-on uses the observed UTF-16 budget so multi-part text
    // answers up to the measured corpus ceiling avoid lossy attachment fallback.
    if fits_split_final_payload(html, budget.utf16_units, budget.long_answer_fix_enabled) {
        let chunks = split_html(html, MAX_TG_MSG - 100);
        if let Some((chunk_idx, _)) = chunks
            .iter()
            .enumerate()
            .find(|(_, chunk)| !tg_payload_within_utf16_budget(chunk, "final split chunk"))
        {
            tracing::warn!(
                html_len = html.len(),
                chunk_count = chunks.len(),
                chunk_idx,
                dropped_events = budget.dropped,
                "final answer split chunk exceeded Telegram budget; falling back to summary attachment"
            );
            let result =
                send_summary_attachment(&bot, ctx, msg_id, view, html.len(), budget.dropped).await;
            crate::metrics::record_final_answer_delivery(result.outcome_for_fallback());
            return;
        }
        if let Some(first) = chunks.first()
            && !edit_must_deliver(&bot, chat_id, msg_id, first, true).await
        {
            tracing::warn!(
                html_len = html.len(),
                chunk_count = chunks.len(),
                first_chunk_len = first.len(),
                dropped_events = budget.dropped,
                "final answer first chunk edit failed; falling back to summary attachment"
            );
            let result =
                send_summary_attachment(&bot, ctx, msg_id, view, html.len(), budget.dropped).await;
            crate::metrics::record_final_answer_delivery(result.outcome_for_fallback());
            return;
        }
        let mut later_chunk_failed = false;
        for chunk in chunks.iter().skip(1) {
            let res = bot
                .send_message(chat_id, chunk.as_str())
                .parse_mode(ParseMode::Html)
                .maybe_thread(ctx.thread_id)
                .maybe_reply_to(ctx.reply_to)
                .await;
            if let Err(e) = res {
                let safe = redact_for_log(&e);
                tracing::warn!("send chunk (HTML) failed: {safe}, retrying plain text");
                if let Err(e2) = bot
                    .send_message(chat_id, chunk.as_str())
                    .maybe_thread(ctx.thread_id)
                    .maybe_reply_to(ctx.reply_to)
                    .await
                {
                    later_chunk_failed = true;
                    let safe2 = redact_for_log(&e2);
                    tracing::error!("send chunk (plain) also failed: {safe2}");
                }
            }
        }
        crate::metrics::record_final_answer_delivery(if later_chunk_failed {
            crate::metrics::FinalAnswerDeliveryOutcome::Partial
        } else {
            crate::metrics::FinalAnswerDeliveryOutcome::Ok
        });
        return;
    }

    let result = send_summary_attachment(&bot, ctx, msg_id, view, html.len(), budget.dropped).await;
    crate::metrics::record_final_answer_delivery(result.outcome_for_full_answer());
}

fn fits_single_final_payload(
    html: &str,
    final_utf16_units: u64,
    tg_long_answer_fix_enabled: bool,
) -> bool {
    if tg_long_answer_fix_enabled {
        final_utf16_units <= MAX_TG_MSG as u64
    } else {
        html.len() <= MAX_TG_MSG
    }
}

fn fits_split_final_payload(
    html: &str,
    final_utf16_units: u64,
    tg_long_answer_fix_enabled: bool,
) -> bool {
    if tg_long_answer_fix_enabled {
        final_utf16_units <= LONG_ANSWER_SPLIT_UTF16_THRESHOLD
    } else {
        html.len() <= FILE_THRESHOLD
    }
}

fn tg_payload_within_utf16_budget(html: &str, payload_kind: &str) -> bool {
    let utf16_units = markup::telegram_html_text_utf16_units(html);
    let within_budget = utf16_units <= MAX_TG_MSG as u64;
    debug_assert!(
        within_budget,
        "{payload_kind} exceeds Telegram payload budget: {utf16_units} UTF-16 units > {MAX_TG_MSG}"
    );
    if !within_budget {
        tracing::warn!(
            payload_kind,
            utf16_units,
            max_utf16_units = MAX_TG_MSG,
            html_len = html.len(),
            "final answer payload exceeds Telegram UTF-16 budget; falling back to summary attachment"
        );
    }
    within_budget
}

async fn send_summary_attachment(
    bot: &Bot,
    ctx: ChatCtx,
    msg_id: MessageId,
    view: &CompositeView,
    html_len: usize,
    dropped: usize,
) -> SummaryAttachmentResult {
    let summary = view.render_summary(SUMMARY_CHARS);
    let summary_delivered = if tg_payload_within_utf16_budget(&summary, "final summary edit") {
        edit_must_deliver(bot, ctx.chat_id, msg_id, &summary, true).await
    } else {
        false
    };
    if !summary_delivered {
        tracing::warn!(
            html_len,
            summary_len = summary.len(),
            dropped_events = dropped,
            "final answer summary edit failed; sending attachment anyway"
        );
    }

    let html_doc = render_html_document(view);
    let input_file = teloxide::types::InputFile::memory(html_doc).file_name("response.html");
    let attachment_delivered = match bot
        .send_document(ctx.chat_id, input_file)
        .caption("📄 Full response")
        .maybe_thread(ctx.thread_id)
        .maybe_reply_to(ctx.reply_to)
        .await
    {
        Ok(_) => true,
        Err(e) => {
            crate::metrics::record_final_answer_attachment_send_failed();
            let safe = redact_for_log(&e);
            tracing::warn!("send_document failed: {safe}");
            false
        }
    };
    SummaryAttachmentResult {
        summary_delivered,
        attachment_delivered,
    }
}

// send_long_text removed (PLAN_TG_SAFE_SEND_v1 T3).
// All callers use crate::shared::safe_send now.

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
        let safe = redact_for_log(e);
        tracing::error!(
            tool = %tool_name,
            error = %safe,
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

pub(crate) async fn flush_live_html(
    bot: &Bot,
    chat_id: ChatId,
    msg_id: MessageId,
    html: &str,
    last_sent: &mut String,
    html_broken: &mut bool,
) -> bool {
    let trimmed = truncate_str(html, MAX_TG_MSG - 50);
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
