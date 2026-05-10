//! Session commands: /new, /sessions, /abort, /stop, /status, /compact.

use super::super::fmt_utils::escape_html_min;
use super::super::*;

pub(crate) async fn cmd_new(
    bot: &Bot,
    agent: &Arc<AgentCore>,
    channel_map: &Arc<ChannelSessionMap>,
    config: &Config,
    ctx: &ChatCtx,
) -> Result<(), teloxide::RequestError> {
    let chat_id = ctx.chat_id.0;
    let tid = ctx.raw_thread_id();
    if let Some(prev) = channel_map.get(chat_id, tid).await {
        agent.close_session_summary(&prev).await;
    }
    let session_id = agent
        .create_session_with_channel(&config.workspace, "telegram")
        .await;
    channel_map.set(chat_id, tid, session_id.clone()).await;
    channel_map.disable_yolo(chat_id, tid).await;
    let cid = format_tg_channel_id(chat_id, tid);
    agent.set_session_channel_id(&session_id, &cid).await;
    reply_text(bot, ctx, format!("🆕 {session_id}")).await?;
    Ok(())
}

pub(crate) async fn cmd_sessions(
    bot: &Bot,
    agent: &Arc<AgentCore>,
    _channel_map: &Arc<ChannelSessionMap>,
    _config: &Config,
    ctx: &ChatCtx,
    text: &str,
) -> Result<(), teloxide::RequestError> {
    const PAGE_SIZE: usize = 20;
    let page: usize = text
        .split_whitespace()
        .nth(1)
        .and_then(|s| s.parse::<usize>().ok())
        .filter(|&n| n >= 1)
        .unwrap_or(1);
    let total = agent.list_sessions().await.len();
    let total_pages = total.div_ceil(PAGE_SIZE).max(1);
    let skip = (page - 1).saturating_mul(PAGE_SIZE);
    let slice = agent.list_sessions_paged(skip, PAGE_SIZE).await;

    if total == 0 {
        reply_text(bot, ctx, "No sessions.").await?;
    } else if slice.is_empty() {
        reply_text(
            bot,
            ctx,
            format!("Page {page} is empty. Total pages: {total_pages}."),
        )
        .await?;
    } else {
        let mut list: Vec<String> = slice
            .iter()
            .map(|s| {
                let prefix = if s.id.len() >= 8 { &s.id[..8] } else { &s.id };
                format!("• {} ({} msgs, {:?})", prefix, s.message_count, s.state)
            })
            .collect();
        list.insert(
            0,
            format!("📄 page {page}/{total_pages} ({total} sessions total)"),
        );
        let text = list.join("\n");
        send_long_text(bot, *ctx, &text).await?;
    }
    Ok(())
}

pub(crate) async fn cmd_abort(
    bot: &Bot,
    agent: &Arc<AgentCore>,
    channel_map: &Arc<ChannelSessionMap>,
    _config: &Config,
    ctx: &ChatCtx,
) -> Result<(), teloxide::RequestError> {
    let chat_id = ctx.chat_id.0;
    let tid = ctx.raw_thread_id();
    if let Some(sid) = channel_map.get(chat_id, tid).await {
        agent.abort(&sid).await;
        reply_text(bot, ctx, "Aborted.").await?;
    } else {
        reply_text(bot, ctx, "No active session.").await?;
    }
    Ok(())
}

pub(crate) async fn cmd_status(
    bot: &Bot,
    agent: &Arc<AgentCore>,
    channel_map: &Arc<ChannelSessionMap>,
    _config: &Config,
    ctx: &ChatCtx,
) -> Result<(), teloxide::RequestError> {
    let chat_id = ctx.chat_id.0;
    let tid = ctx.raw_thread_id();
    let sid = match channel_map.get(chat_id, tid).await {
        Some(s) => s,
        None => {
            reply_text(bot, ctx, "No active session. Send a message first.").await?;
            return Ok(());
        }
    };
    let (prov, model) = agent.session_provider_model(&sid).await;
    let reasoning = agent.session_reasoning(&sid).await;
    let usage = agent.session_total_usage(&sid).await;
    let (_, _, cost) = usage.estimate_cost(&format!("{prov}/{model}"));

    use naked_tg::markup::format_tokens;
    let mut lines = vec![
        format!(
            "<b>Model:</b> <code>{}/{}</code>",
            escape_html_min(&prov),
            escape_html_min(&model)
        ),
        format!(
            "<b>Reasoning:</b> <code>{}</code>",
            reasoning.as_deref().unwrap_or("off")
        ),
        format!(
            "<b>Usage:</b> ↑{} ↓{}",
            format_tokens(usage.input_tokens),
            format_tokens(usage.output_tokens)
        ),
    ];
    if usage.cache_read_tokens > 0 || usage.cache_write_tokens > 0 {
        lines.push(format!(
            "<b>Cache:</b> R{} W{}",
            format_tokens(usage.cache_read_tokens),
            format_tokens(usage.cache_write_tokens),
        ));
    }
    if cost > 0.001 {
        lines.push(format!("<b>Cost:</b> ${cost:.3}"));
    }
    let ctx_est = agent.session_context_usage(&sid).await;
    if let Some((est, cw)) = ctx_est {
        let pct = if cw > 0 {
            format!(" ({:.0}%)", est as f64 / cw as f64 * 100.0)
        } else {
            String::new()
        };
        lines.push(format!(
            "<b>Context:</b> ~{} / {}{pct}",
            format_tokens(est as u64),
            format_tokens(cw as u64),
        ));
        // Coherence state:
        let ratio = est as f32 / cw.max(1) as f32;
        let coh = naked_core::coherence::from_capacity(ratio as f64);
        lines.push(format!(
            "<b>Health:</b> {} {}",
            coh.emoji(),
            crate::fmt_utils::escape_html_min(coh.label()),
        ));
    }
    let (read_files, modified_files) = agent.session_file_stats(&sid).await;
    if !modified_files.is_empty() {
        lines.push(format!("<b>Modified:</b> {} file(s)", modified_files.len()));
    }
    if !read_files.is_empty() {
        lines.push(format!("<b>Read:</b> {} file(s)", read_files.len()));
    }

    let kb = teloxide::types::InlineKeyboardMarkup::new(vec![vec![
        teloxide::types::InlineKeyboardButton::callback("🤖 Model", "cmd:model"),
        teloxide::types::InlineKeyboardButton::callback("💭 Reasoning", "cmd:reasoning"),
    ]]);
    reply_html_kb(bot, ctx, lines.join("\n"), kb).await?;
    Ok(())
}

pub(crate) async fn cmd_compact(
    bot: &Bot,
    agent: &Arc<AgentCore>,
    channel_map: &Arc<ChannelSessionMap>,
    _config: &Config,
    ctx: &ChatCtx,
) -> Result<(), teloxide::RequestError> {
    let chat_id = ctx.chat_id.0;
    let tid = ctx.raw_thread_id();
    if let Some(sid) = channel_map.get(chat_id, tid).await {
        if agent.is_session_active(&sid).await {
            reply_text(bot, ctx, "⏳ Wait for current task to finish.").await?;
        } else {
            match agent.compact_session(&sid).await {
                Some((before, after)) => {
                    reply_text(
                        bot,
                        ctx,
                        format!("✅ Compacted: {before} messages → {after}"),
                    )
                    .await?;
                }
                None => {
                    reply_text(bot, ctx, "ℹ️ No compaction needed.").await?;
                }
            }
        }
    } else {
        reply_text(bot, ctx, "No active session.").await?;
    }
    Ok(())
}

pub(crate) async fn cmd_usage(
    bot: &Bot,
    agent: &Arc<AgentCore>,
    ctx: &ChatCtx,
) -> Result<(), teloxide::RequestError> {
    let summary = agent.token_tracker.summary();
    let text = format!(
        "<b>Token Usage</b>\n<pre>{}</pre>",
        crate::fmt_utils::escape_html_min(&summary)
    );
    reply_html(bot, ctx, &text).await?;
    Ok(())
}
