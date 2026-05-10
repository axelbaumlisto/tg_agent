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
    // F4 of PLAN_NEXT_SESSION (2026-05-10): one compact health
    // dashboard. The four sections — providers, liveness sources,
    // operational counters, uptime — are pure reads; the command
    // never blocks on network or LLM calls so it stays responsive
    // even when the bot is otherwise wedged.
    let mut lines: Vec<String> = Vec::new();

    // ── 1) Providers (existing behaviour) ────────────────
    lines.push("🏥 <b>Provider Health</b>".to_string());
    for (name, pc) in &config.providers {
        let resolved = pc.resolved_all_keys();
        let total = resolved.len();
        if total == 0 {
            lines.push(format!("  <b>{name}</b>: ⚠️ no keys"));
            continue;
        }
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

    // ── 2) Liveness sources (F2 wiring) ─────────────────
    if let Some(reg) = crate::shared::LIVENESS_REGISTRY.get() {
        lines.push(String::new());
        lines.push("🪺 <b>Liveness</b>".to_string());
        let mut snap = reg.snapshot();
        snap.sort_by_key(|(name, _)| *name);
        for (name, age) in snap {
            let line = match age {
                Some(d) => {
                    let icon = if d.as_secs() <= 60 { "✅" } else { "⚠️" };
                    format!("  {icon} <code>{name}</code>: {}", fmt_age(d))
                }
                None => format!("  ❔ <code>{name}</code>: never beat"),
            };
            lines.push(line);
        }
    }

    // ── 3) Operational counters (F3 instrumentation) ───────────
    use std::sync::atomic::Ordering;
    let snap = crate::metrics::snapshot();
    let steer_delivered = naked_core::types::STEER_DELIVERED_COUNT.load(Ordering::Relaxed);
    let steer_soft = naked_core::types::STEER_SOFT_INTERRUPTED_COUNT.load(Ordering::Relaxed);
    let steer_drained = naked_core::types::STEER_DRAINED_ON_ABORT_COUNT.load(Ordering::Relaxed);
    let supervisor_restart =
        naked_tg::supervised::SUPERVISOR_PANIC_RESTART_COUNT.load(Ordering::Relaxed);
    let turn_ok = naked_core::types::TURN_COMPLETED_COUNT.load(Ordering::Relaxed);
    let turn_err = naked_core::types::TURN_ERROR_COUNT.load(Ordering::Relaxed);
    let empty_retries = naked_core::types::EMPTY_CONTENT_RETRY_COUNT.load(Ordering::Relaxed);
    lines.push(String::new());
    lines.push("📊 <b>Counters (process lifetime)</b>".to_string());
    lines.push(format!(
        "  turns: {turn_ok} ✓ / {turn_err} ✗   empty-retries: {empty_retries}"
    ));
    lines.push(format!(
        "  steer: {steer_delivered} delivered / {steer_soft} mid-stream / {steer_drained} rescued"
    ));
    lines.push(format!(
        "  buttons: ⏹ {} / ⏩ {}    supervisor restarts: {}",
        snap.stream_button_click_abort, snap.stream_button_click_sendnow, supervisor_restart,
    ));

    // ── 4) Uptime ──────────────────────────────────
    let up = crate::shared::PROCESS_STARTED_AT.elapsed();
    lines.push(String::new());
    lines.push(format!("⏱ <b>Uptime</b>: {}", fmt_age(up)));

    reply_html(bot, ctx, lines.join("\n")).await?;
    Ok(())
}

/// Format a [`Duration`] as a compact `Hh Mm Ss` / `Mm Ss` / `Ss`
/// string. Used by `/health`'s liveness + uptime rows.
fn fmt_age(d: std::time::Duration) -> String {
    let total = d.as_secs();
    let h = total / 3600;
    let m = (total % 3600) / 60;
    let s = total % 60;
    if h > 0 {
        format!("{h}h {m}m {s}s ago")
    } else if m > 0 {
        format!("{m}m {s}s ago")
    } else {
        format!("{s}s ago")
    }
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

#[cfg(test)]
mod tests {
    use super::fmt_age;
    use std::time::Duration;

    #[test]
    fn fmt_age_seconds_only() {
        assert_eq!(fmt_age(Duration::from_secs(0)), "0s ago");
        assert_eq!(fmt_age(Duration::from_secs(45)), "45s ago");
    }

    #[test]
    fn fmt_age_minutes() {
        assert_eq!(fmt_age(Duration::from_secs(60)), "1m 0s ago");
        assert_eq!(fmt_age(Duration::from_secs(125)), "2m 5s ago");
    }

    #[test]
    fn fmt_age_hours() {
        assert_eq!(fmt_age(Duration::from_secs(3_600)), "1h 0m 0s ago");
        assert_eq!(fmt_age(Duration::from_secs(3_725)), "1h 2m 5s ago");
        assert_eq!(fmt_age(Duration::from_secs(86_400)), "24h 0m 0s ago");
    }
}
