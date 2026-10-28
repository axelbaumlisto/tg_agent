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
/yolo — auto-approve tools (1st: 30d for this topic; 2nd in chat: permanent)
/yolo off — turn off & reset YOLO for this chat
/sessions — list active sessions
/metrics — bot performance stats
/reload — reload config
/health — provider health & key status
/help — show this message";
    reply_html(bot, ctx, help).await?;
    Ok(())
}

/// `/start` — first-contact screen for a brand-new chat.
///
/// Separate from `/help` on purpose: `/help` is a reference sheet for
/// someone who already knows the bot, while `/start` has to answer
/// "what is this and what do I do now" in the first screenful. A new
/// user who is shown fourteen commands learns nothing about the one
/// thing that matters: you can just type.
#[allow(unused_variables)]
pub(crate) async fn cmd_start(
    bot: &Bot,
    agent: &std::sync::Arc<AgentCore>,
    channel_map: &std::sync::Arc<ChannelSessionMap>,
    config: &Config,
    ctx: &ChatCtx,
) -> Result<(), teloxide::RequestError> {
    let welcome = "\
<b>naked agent</b>

\
Just send a message — no command needed. Ask a question, paste code,
attach an image or a voice note; the reply streams back as it is written.

\
<b>Three things worth knowing</b>

\
• <code>/new</code> — start over when the topic changes. Context is kept
  per chat, so a fresh session gives cleaner answers.
• <code>/stop</code> — interrupt a long answer at any moment.
• <code>/model</code> — switch the model; <code>/status</code> shows which
  one is active, plus usage and cost.

\
<code>/help</code> lists every command.";
    reply_html(bot, ctx, welcome).await?;
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

    // ── 2) Loaded config hash (S7 / RC-14 B42 observability) ───────
    lines.push(naked_tg::config_hash::render_health_config_hash(
        crate::shared::CONFIG_LOADED_HASH.get(),
    ));

    // ── 3) Liveness sources (F2 wiring) ─────────────────
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

    // ── 4) Operational counters (F3 instrumentation) ───────────
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

    // ── 5) Coherence ladder (T7 of PLAN_QUALITY_v1) ─────────
    // Pure derivation from existing process-wide counters — no
    // separate state, no I/O. Healthy is the default; we degrade
    // when empty-retries climb and recover after a successful
    // compaction or steer rescue.
    let coh = derive_coherence_state();
    lines.push(String::new());
    lines.push(format!(
        "{} <b>Coherence</b>: {} — <i>{}</i>",
        coh.icon(),
        coh.label(),
        coh.description(),
    ));

    // ── 6) Uptime ──────────────────────────────────
    let up = crate::shared::PROCESS_STARTED_AT.elapsed();
    lines.push(String::new());
    lines.push(format!("⏱ <b>Uptime</b>: {}", fmt_age(up)));

    reply_html(bot, ctx, lines.join("\n")).await?;
    Ok(())
}

/// Derive coherence state from the current snapshot of process-wide
/// counters. Cheap (a few atomic loads); no caching needed.
fn derive_coherence_state() -> naked_core::coherence::CoherenceState {
    use naked_core::coherence::{CoherenceSignal, CoherenceState, next_coherence_state};
    use std::sync::atomic::Ordering;
    let mut state = CoherenceState::Healthy;
    let empty = naked_core::types::EMPTY_CONTENT_RETRY_COUNT.load(Ordering::Relaxed);
    let drained = naked_core::types::STEER_DRAINED_ON_ABORT_COUNT.load(Ordering::Relaxed);
    // Heuristic: any nonzero recent retries imply context pressure.
    // We don't have rate-windows here; the counters are monotonic,
    // so we just treat "observed empties at all" as the warm signal.
    // Operators get the precise counts in the panel above; the
    // ladder is a one-glance overview.
    if empty > 0 {
        state = next_coherence_state(state, CoherenceSignal::EmptyContentRetriesRising);
    }
    if drained > 0 {
        state = next_coherence_state(state, CoherenceSignal::SteerRescued);
    }
    state
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

    /// Source-text assertion (same convention as research.rs help tests):
    /// the /help command literal is the operator-facing contract, and
    /// `/yolo off` is the ONLY way to revoke a permanent chat-wide YOLO
    /// grant, so it must stay discoverable in help.
    #[test]
    fn help_documents_yolo_commands() {
        let src = include_str!("info.rs");
        let prod = src.split("#[cfg(test)]").next().unwrap_or("");
        assert!(
            prod.contains("/yolo —"),
            "/help must document the /yolo command"
        );
        assert!(
            prod.contains("/yolo off"),
            "/help must document /yolo off (only way to revoke permanent YOLO)"
        );
    }

    /// `/start` and `/help` answer different questions. If they ever
    /// collapse back into one handler the new-user path silently
    /// regresses to a command dump, which is invisible in tests that
    /// only check that *something* is sent.
    #[test]
    fn start_is_not_the_help_sheet() {
        let src = include_str!("info.rs");
        let prod = src.split("#[cfg(test)]").next().unwrap_or("");
        assert!(
            prod.contains("pub(crate) async fn cmd_start("),
            "/start must have its own handler, separate from /help"
        );
    }

    /// The first screen must teach the core interaction, not the command
    /// list. Everything here is a promise to a first-time user.
    #[test]
    fn start_welcome_teaches_plain_messaging() {
        let src = include_str!("info.rs");
        let start = src
            .split("pub(crate) async fn cmd_start(")
            .nth(1)
            .and_then(|tail| tail.split("async fn cmd_health").next())
            .unwrap_or("");
        assert!(
            start.contains("Just send a message"),
            "/start must tell a new user that plain messages work"
        );
        for command in ["/new", "/stop", "/model"] {
            assert!(
                start.contains(command),
                "/start must surface {command}, the commands a new user needs first"
            );
        }
        assert!(
            start.contains("/help"),
            "/start must point at /help for the full list"
        );
    }

    /// A welcome screen that scrolls is not a welcome screen. The help
    /// sheet is allowed to be long; this one is not.
    #[test]
    fn start_welcome_stays_short() {
        let src = include_str!("info.rs");
        let start = src
            .split("let welcome = ")
            .nth(1)
            .and_then(|tail| tail.split("reply_html").next())
            .unwrap_or("");
        let lines = start.lines().count();
        assert!(
            lines <= 30,
            "/start welcome grew to {lines} source lines; keep it to one screen"
        );
    }

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
