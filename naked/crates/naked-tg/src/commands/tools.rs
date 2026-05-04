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

#[allow(unused_variables)]
pub(crate) async fn cmd_yolo(
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
    tracing::info!(chat_id, ?tid, "yolo: enabling");
    let already = channel_map.is_yolo(chat_id, tid).await;
    if already {
        reply_text(bot, ctx, "⚡ YOLO already active.").await?;
    } else {
        channel_map.enable_yolo(chat_id, tid).await;
        // Persist yolo timestamp to session config
        let yolo_ts = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs() as i64;
        if let Some(sid) = channel_map.get(chat_id, tid).await
            && let Err(e) = agent.set_session_yolo(&sid, Some(yolo_ts)).await
        {
            tracing::warn!("failed to persist yolo: {e}");
        }
        let mut perms = pending_perms.write().await;
        let matching_keys: Vec<String> = perms
            .iter()
            .filter(|(_, (_, cid, t))| *cid == chat_id && *t == tid)
            .map(|(k, _)| k.clone())
            .collect();
        let n = matching_keys.len();
        for key in matching_keys {
            if let Some((tx, _, _)) = perms.remove(&key) {
                let _ = tx.send(true);
            }
        }
        if n > 0 {
            tracing::info!("yolo: auto-approved {n} pending permission(s)");
        }
        let remaining_h = channel_map.yolo_remaining_secs(chat_id, tid).await / 3600;
        reply_text(
            bot,
            ctx,
            format!(
                "⚡ YOLO ON — all tools auto-approved ({remaining_h}h).{}",
                if n > 0 {
                    format!("\n✅ {n} pending request(s) approved.")
                } else {
                    String::new()
                }
            ),
        )
        .await?;
    }
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
                    tracing::warn!("failed to persist allow-list: {e}");
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
                    tracing::warn!("failed to persist allow-list: {e}");
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
