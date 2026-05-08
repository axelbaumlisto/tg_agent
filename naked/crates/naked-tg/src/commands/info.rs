//! /info command handlers.

use super::super::*;

#[allow(unused_variables)]
pub(crate) async fn cmd_help(
    bot: &Bot,
    agent: &std::sync::Arc<AgentCore>,
    channel_map: &std::sync::Arc<ChannelSessionMap>,
    config: &Config,
    ctx: &ChatCtx,
) -> Result<(), teloxide::RequestError> {
    let chat_id = ctx.chat_id.0;
    let tid = ctx.raw_thread_id();
    let help = "\
<b>naked agent</b> — send any message to start a conversation.

\
<b>Commands:</b>
/new — start a new session
/stop — cancel running task
/status — session info, usage, cost
/compact — compact session history
/model — switch model
/reasoning — set thinking level
/sessions — list active sessions
/metrics — bot performance stats
/reload — reload config
/health — provider health & key status
/help — show this message";
    reply_html(bot, ctx, help).await?;
    Ok(())
}

#[allow(unused_variables)]
pub(crate) async fn cmd_health(
    bot: &Bot,
    agent: &std::sync::Arc<AgentCore>,
    channel_map: &std::sync::Arc<ChannelSessionMap>,
    config: &Config,
    ctx: &ChatCtx,
    text: &str,
) -> Result<(), teloxide::RequestError> {
    let chat_id = ctx.chat_id.0;
    let tid = ctx.raw_thread_id();
    let mut lines = vec!["🏥 <b>Provider Health</b>".to_string()];
    for (name, pc) in &config.providers {
        let resolved = pc.resolved_all_keys();
        let total = resolved.len();
        if total == 0 {
            lines.push(format!("  <b>{name}</b>: ⚠️ no keys"));
            continue;
        }
        // Check which keys are alive via quick balance/auth probe
        // For now, show key count + provider status from the resilient wrapper
        let provider = agent.provider_for(name).await;
        let bl = provider.blacklisted_key_count();
        let info = if bl > 0 {
            let alive = total - bl;
            format!("⚠️ {alive}/{total} keys ({bl} blacklisted)")
        } else {
            format!("✅ {total} key(s)")
        };
        lines.push(format!("  <b>{name}</b>: {info}"));
    }
    reply_html(bot, ctx, lines.join("\n")).await?;
    Ok(())
}

#[allow(unused_variables)]
pub(crate) async fn cmd_metrics(
    bot: &Bot,
    agent: &std::sync::Arc<AgentCore>,
    channel_map: &std::sync::Arc<ChannelSessionMap>,
    config: &Config,
    ctx: &ChatCtx,
    text: &str,
) -> Result<(), teloxide::RequestError> {
    let chat_id = ctx.chat_id.0;
    let tid = ctx.raw_thread_id();
    let snap = crate::metrics::snapshot();
    let active = (*RATE_LIMITER).active_count().await;
    let interval = (*RATE_LIMITER).interval().await;
    let mut text = snap.render_text();
    text.push_str(&format!(
        "\n\nRate limiter (proactive):\n\
         • active chats: {}\n\
         • edit interval: {}ms",
        active,
        interval.as_millis(),
    ));
    reply_text(bot, ctx, text).await?;
    Ok(())
}
