//! /tools command handlers.

use super::super::*;

#[allow(unused_variables)]
pub(crate) async fn cmd_skills(
    bot: &Bot,
    agent: &std::sync::Arc<AgentCore>,
    channel_map: &std::sync::Arc<ChannelSessionMap>,
    config: &Config,
    ctx: &ChatCtx,
) -> Result<(), teloxide::RequestError> {
    let chat_id = ctx.chat_id.0;
    let tid = ctx.raw_thread_id();
    let skills = agent.list_skills();
    if skills.is_empty() {
        reply_text(bot, ctx, "No skills loaded.").await?;
    } else {
        let list: Vec<String> = skills
            .iter()
            .map(|(name, path)| {
                format!(
                    "• <b>{}</b>\n  <code>{}</code>",
                    escape_html(name),
                    escape_html(path)
                )
            })
            .collect();
        let header = format!("📚 <b>{} skill(s)</b>\n\n{}", skills.len(), list.join("\n"));
        reply_html(bot, ctx, header).await?;
    }
    Ok(())
}

#[allow(unused_variables)]
pub(crate) async fn cmd_mcp(
    bot: &Bot,
    agent: &std::sync::Arc<AgentCore>,
    channel_map: &std::sync::Arc<ChannelSessionMap>,
    config: &Config,
    ctx: &ChatCtx,
) -> Result<(), teloxide::RequestError> {
    let chat_id = ctx.chat_id.0;
    let tid = ctx.raw_thread_id();
    let servers = agent.list_mcp_servers().await;
    if servers.is_empty() {
        reply_text(bot, ctx, "No MCP servers connected.").await?;
    } else {
        let list: Vec<String> = servers
            .iter()
            .map(|(name, n_tools)| format!("• <b>{}</b> — {n_tools} tool(s)", escape_html(name)))
            .collect();
        let header = format!(
            "🔌 <b>{} MCP server(s)</b>\n\n{}",
            servers.len(),
            list.join("\n")
        );
        reply_html(bot, ctx, header).await?;
    }
    Ok(())
}

#[allow(unused_variables)]
pub(crate) async fn cmd_refresh(
    bot: &Bot,
    agent: &std::sync::Arc<AgentCore>,
    channel_map: &std::sync::Arc<ChannelSessionMap>,
    config: &Config,
    ctx: &ChatCtx,
) -> Result<(), teloxide::RequestError> {
    let chat_id = ctx.chat_id.0;
    let tid = ctx.raw_thread_id();
    agent.refresh_skills_and_mcp().await;
    let skills = agent.list_skills();
    let servers = agent.list_mcp_servers().await;
    reply_text(
        bot,
        ctx,
        format!(
            "🔄 Refreshed\n• {} skill(s)\n• {} MCP server(s)",
            skills.len(),
            servers.len()
        ),
    )
    .await?;
    Ok(())
}

/// Build the `/yolo off` reply. Durable-revocation success is reported ONLY
/// when every persisted session grant was cleared (`failed == 0`) AND the
/// channel-map snapshot flush succeeded (`!flush_failed`). Either failure means
/// a grant can survive restart — an uncleared session `yolo_enabled_at` revives
/// a temporary grant, and a failed flush leaves a permanent `yolo_chat` row on
/// disk — so we must never claim durable revocation the user cannot rely on.
fn yolo_off_reply(failed: usize, flush_failed: bool) -> String {
    if failed == 0 && !flush_failed {
        "YOLO выключен для чата (сброшено).".to_string()
    } else if failed > 0 {
        format!(
            "⚠️ YOLO частично отключён — не удалось сбросить {failed} сессий, повтори /yolo off позже."
        )
    } else {
        // Sessions cleared but the snapshot rewrite failed: a permanent
        // `yolo_chat` row may survive restart, so the revocation is not durable.
        "⚠️ YOLO частично отключён — не удалось сохранить сброс, повтори /yolo off позже."
            .to_string()
    }
}

pub(crate) async fn cmd_yolo(
    bot: &Bot,
    agent: &std::sync::Arc<AgentCore>,
    channel_map: &std::sync::Arc<ChannelSessionMap>,
    _config: &Config,
    ctx: &ChatCtx,
    text: &str,
    pending_perms: &PendingPermissions,
) -> Result<(), teloxide::RequestError> {
    let chat_id = ctx.chat_id.0;
    let tid = ctx.raw_thread_id();

    // `/yolo off` clears ALL yolo state for the chat (temporary grants +
    // per-chat escalation). Any other form enables/escalates.
    if text.split_whitespace().nth(1) == Some("off") {
        channel_map.clear_yolo_chat(chat_id).await;
        // B87 (revocation completeness): drain this chat's pending permission
        // cards — every card carries a live "⚡ YOLO" button, and revocation
        // resets the escalation count, so a card left alive could be tapped
        // AFTER `/yolo off` to silently re-enable/escalate YOLO. Deny them
        // (fail-closed) chat-wide, matching `clear_yolo_chat`'s scope; other
        // chats' cards are untouched.
        let drained = deny_pending_perms(pending_perms, chat_id).await;
        if drained > 0 {
            tracing::info!(
                chat_id,
                drained,
                "/yolo off: denied pending permission card(s)"
            );
        }
        // Clear the persisted SessionConfig yolo grant for EVERY topic of the
        // chat, not just the current one — otherwise a sibling topic's
        // `yolo_enabled_at` would revive a temporary grant on restart
        // (revocation bypass). Track the sessions we could NOT clear: a failed
        // persist means that topic re-enables YOLO on restart, so we must never
        // claim durable revocation when any clear failed.
        let mut failed = 0usize;
        for sid in channel_map.sessions_for_chat(chat_id).await {
            if let Err(e) = agent.set_session_yolo(&sid, None).await {
                let safe = redact_for_log(&e);
                tracing::error!(%sid, "/yolo off: failed to clear persisted yolo grant: {safe}");
                failed += 1;
            }
        }
        // Flush the channel-map snapshot so the in-memory revocation is durable
        // immediately (don't wait for the periodic 30s writer to drop the grants
        // we just cleared). A failed flush leaves the permanent `yolo_chat` row
        // on disk, so it counts toward non-durable revocation in the reply.
        let flush_failed = if let Err(e) = channel_map.flush().await {
            let safe = redact_for_log(&e);
            tracing::error!("failed to flush channel_map after /yolo off: {safe}");
            true
        } else {
            false
        };
        reply_text(bot, ctx, yolo_off_reply(failed, flush_failed)).await?;
        return Ok(());
    }

    // Command path: one `/yolo` is one distinct action (no dedup key).
    let esc = channel_map.enable_yolo(chat_id, tid, None).await;
    tracing::info!(chat_id, ?tid, count = esc.count, "yolo: enabling");
    // Persist yolo timestamp to session config
    let yolo_ts = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs() as i64;
    if let Some(sid) = channel_map.get(chat_id, tid).await
        && let Err(e) = agent.set_session_yolo(&sid, Some(yolo_ts)).await
    {
        let safe = redact_for_log(&e);
        tracing::warn!("failed to persist yolo: {safe}");
    }
    // Fix 3: a permanent escalation approves the WHOLE chat (all topics); a
    // temporary enable approves only the current topic.
    let permanent = matches!(esc.tier, YoloTier::Permanent);
    let n = approve_pending_perms(pending_perms, chat_id, tid, permanent).await;
    if n > 0 {
        tracing::info!("yolo: auto-approved {n} pending permission(s)");
    }
    // Fix 2: persist immediately on permanent so a restart before the periodic
    // flush can't lose the chat-wide grant.
    persist_permanent(channel_map, &esc.tier).await;
    reply_text(bot, ctx, esc.toast(n)).await?;
    Ok(())
}

#[allow(unused_variables)]
pub(crate) async fn cmd_allow(
    bot: &Bot,
    agent: &std::sync::Arc<AgentCore>,
    channel_map: &std::sync::Arc<ChannelSessionMap>,
    config: &Config,
    ctx: &ChatCtx,
    text: &str,
    pending_perms: &PendingPermissions,
) -> Result<(), teloxide::RequestError> {
    let chat_id = ctx.chat_id.0;
    let tid = ctx.raw_thread_id();
    let args: Vec<&str> = text.split_whitespace().skip(1).collect();
    match args.first().copied() {
        Some("add") if args.len() >= 2 => {
            let tool = args[1];
            let added = channel_map.allow_add(chat_id, tid, tool).await;
            // Persist allow-list
            if let Some(sid) = channel_map.get(chat_id, tid).await {
                let list = channel_map.allow_get(chat_id, tid).await;
                if let Err(e) = agent.set_session_allow_list(&sid, &list).await {
                    let safe = redact_for_log(&e);
                    tracing::warn!("failed to persist allow-list: {safe}");
                }
            }
            let msg = if added {
                format!("✅ <b>{}</b> added to allow-list.", escape_html(tool))
            } else {
                format!("ℹ️ <b>{}</b> already in allow-list.", escape_html(tool))
            };
            reply_html(bot, ctx, msg).await?;
        }
        Some("rm" | "remove" | "del") if args.len() >= 2 => {
            let tool = args[1];
            let removed = channel_map.allow_remove(chat_id, tid, tool).await;
            // Persist allow-list
            if let Some(sid) = channel_map.get(chat_id, tid).await {
                let list = channel_map.allow_get(chat_id, tid).await;
                if let Err(e) = agent.set_session_allow_list(&sid, &list).await {
                    let safe = redact_for_log(&e);
                    tracing::warn!("failed to persist allow-list: {safe}");
                }
            }
            let msg = if removed {
                format!("🗑 <b>{}</b> removed from allow-list.", escape_html(tool))
            } else {
                format!("ℹ️ <b>{}</b> not in allow-list.", escape_html(tool))
            };
            reply_html(bot, ctx, msg).await?;
        }
        _ => {
            let list = channel_map.allow_get(chat_id, tid).await;
            let yolo = channel_map.is_yolo(chat_id, tid).await;
            let mut text = String::new();
            if yolo {
                text.push_str("⚡ YOLO mode — everything auto-approved\n\n");
            }
            if list.is_empty() {
                text.push_str("Allow-list is empty.\n");
            } else {
                text.push_str(&format!(
                    "📋 <b>{} tool(s)</b> in allow-list:\n",
                    list.len()
                ));
                for t in &list {
                    text.push_str(&format!("• <code>{}</code>\n", escape_html(t)));
                }
            }
            text.push_str("\nUsage:\n<code>/allow add bash</code>\n<code>/allow rm bash</code>");
            reply_html(bot, ctx, text).await?;
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::yolo_off_reply;

    #[test]
    fn yolo_off_reply_reports_durable_success_when_no_failures() {
        assert_eq!(
            yolo_off_reply(0, false),
            "YOLO выключен для чата (сброшено)."
        );
    }

    #[test]
    fn yolo_off_reply_partial_failure_is_not_durable_success() {
        let msg = yolo_off_reply(3, false);
        assert!(
            msg.contains("частично отключён"),
            "partial failure must warn, got: {msg}"
        );
        assert!(msg.contains('3'), "must name the failed session count");
        assert_ne!(
            msg,
            yolo_off_reply(0, false),
            "a partial failure must NOT reuse the durable-success reply"
        );
    }

    #[test]
    fn yolo_off_reply_flush_failure_is_not_durable_success() {
        // Even with zero session-clear failures, a failed snapshot flush leaves
        // a permanent yolo_chat row on disk → must NOT claim durable success.
        let msg = yolo_off_reply(0, true);
        assert!(
            msg.contains("частично отключён"),
            "flush failure must warn, got: {msg}"
        );
        assert_ne!(
            msg,
            yolo_off_reply(0, false),
            "a flush failure must NOT reuse the durable-success reply"
        );
    }
}
