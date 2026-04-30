//! Slash-command handlers extracted from main.rs.
//!
//! All functions are `pub(crate)` so main.rs can call them.
//! Uses `use super::*` to access types and statics from main.

use super::fmt_utils::{escape_html_min, format_age, format_interval, safe_slug};
use super::*;

// ── Commands ────────────────────────────────────────────────────────────────

/// Returns `Ok(true)` if the command was handled, `Ok(false)` if unrecognized
/// (caller should pass the message to the agent).
#[allow(clippy::too_many_arguments)]
pub(crate) async fn handle_command(
    bot: &Bot,
    _msg: &Message,
    text: &str,
    agent: &Arc<AgentCore>,
    channel_map: &Arc<ChannelSessionMap>,
    config: &Config,
    ctx: ChatCtx,
    pending_perms: &PendingPermissions,
    attribution_flag: &Arc<std::sync::atomic::AtomicBool>,
) -> Result<bool, teloxide::RequestError> {
    let chat_id = ctx.chat_id.0;
    let tid = ctx.raw_thread_id();

    let cmd_word = text.split_whitespace().next().unwrap_or("");
    let cmd = cmd_word.split('@').next().unwrap_or(cmd_word);
    tracing::debug!(chat_id, cmd, cmd_word, "handle_command");
    match cmd {
        "/start" | "/help" => {
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
            reply_html(bot, &ctx, help).await?;
        }
        "/attribution" => {
            use std::sync::atomic::Ordering;
            let arg = text[cmd_word.len()..].trim().to_ascii_lowercase();
            let reply = match arg.as_str() {
                "on" | "1" | "true" | "enable" => {
                    attribution_flag.store(true, Ordering::Relaxed);
                    "✅ sender attribution: ON (groups will see `@username:` prefix)".to_string()
                }
                "off" | "0" | "false" | "disable" => {
                    attribution_flag.store(false, Ordering::Relaxed);
                    "⛔ sender attribution: OFF".to_string()
                }
                "" | "status" => {
                    let on = attribution_flag.load(Ordering::Relaxed);
                    let state = if on { "ON" } else { "OFF" };
                    format!(
                        "sender attribution: {state}\nUsage: /attribution on|off|status\n(resets to config.tg_sender_attribution on restart)"
                    )
                }
                other => {
                    format!("Unknown arg `{other}`. Usage: /attribution on|off|status")
                }
            };
            reply_text(bot, &ctx, reply).await?;
        }
        "/new" => {
            // Best-effort: snapshot the closing session into a daily
            // memory draft before we cut the channel binding to it.
            // Runs in the background; never blocks `/new`.
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
            reply_text(bot, &ctx, format!("🆕 {session_id}")).await?;
        }
        "/sessions" => {
            // `/sessions [page]` — 1-indexed, 20 per page, newest first.
            // Backward-compatible: `/sessions` with no arg behaves exactly
            // like before (page 1) thanks to the default.
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
                reply_text(bot, &ctx, "No sessions.").await?;
            } else if slice.is_empty() {
                reply_text(
                    bot,
                    &ctx,
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
                send_long_text(bot, ctx, &text).await?;
            }
        }
        "/abort" | "/stop" => {
            if let Some(sid) = channel_map.get(chat_id, tid).await {
                agent.abort(&sid).await;
                reply_text(bot, &ctx, "Aborted.").await?;
            } else {
                reply_text(bot, &ctx, "No active session.").await?;
            }
        }
        "/status" => {
            let sid = match channel_map.get(chat_id, tid).await {
                Some(s) => s,
                None => {
                    reply_text(bot, &ctx, "No active session. Send a message first.").await?;
                    return Ok(true);
                }
            };
            let (prov, model) = agent.session_provider_model(&sid).await;
            let reasoning = agent.session_reasoning(&sid).await;
            let usage = agent.session_total_usage(&sid).await;
            let (_, _, cost) = usage.estimate_cost(&format!("{prov}/{model}"));

            use naked_tg::tg_markup::format_tokens;
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
                    format_tokens(usage.cache_write_tokens)
                ));
            }
            if let Some((est, cw)) = agent.session_context_usage(&sid).await
                && cw > 0
            {
                let pct = (est as f64 / cw as f64) * 100.0;
                lines.push(format!(
                    "<b>Context:</b> {:.1}%/{}",
                    pct,
                    format_tokens(cw as u64)
                ));
            }
            if cost > 0.0 {
                lines.push(format!("<b>Cost:</b> <code>${cost:.4}</code>"));
            }

            let kb = teloxide::types::InlineKeyboardMarkup::new(vec![vec![
                teloxide::types::InlineKeyboardButton::callback("🤖 Model", "cmd:model"),
                teloxide::types::InlineKeyboardButton::callback("💭 Reasoning", "cmd:reasoning"),
            ]]);
            reply_html_kb(bot, &ctx, lines.join("\n"), kb).await?;
        }
        "/health" => {
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
                let info = if let Some(rp) = provider.as_resilient() {
                    let bl = rp.blacklisted_count().await;
                    let alive = total - bl;
                    if bl > 0 {
                        format!("⚠️ {alive}/{total} keys ({bl} blacklisted)")
                    } else {
                        format!("✅ {total} key(s)")
                    }
                } else {
                    format!("✅ {total} key(s)")
                };
                lines.push(format!("  <b>{name}</b>: {info}"));
            }
            reply_html(bot, &ctx, lines.join("\n")).await?;
        }
        "/metrics" => {
            let snap = crate::metrics::snapshot();
            let rl_stats = (*RATE_LIMITER).stats().await;
            let mut text = snap.render_text();
            text.push_str(&format!(
                "\n\nRate limiter:\n\
                 • active chats: {}\n\
                 • global calls/min: {}",
                rl_stats.active_chats, rl_stats.global_calls_last_min,
            ));
            for (key, gap_ms, calls) in &rl_stats.chat_gaps {
                text.push_str(&format!(
                    "\n  chat {}: gap={}ms, calls/min={}",
                    key.0, gap_ms, calls
                ));
            }
            reply_text(bot, &ctx, text).await?;
        }
        "/compact" => {
            if let Some(sid) = channel_map.get(chat_id, tid).await {
                if agent.is_session_active(&sid).await {
                    reply_text(bot, &ctx, "⏳ Wait for current task to finish.").await?;
                } else {
                    match agent.compact_session(&sid).await {
                        Some((before, after)) => {
                            reply_text(
                                bot,
                                &ctx,
                                format!("✅ Compacted: {before} messages → {after}"),
                            )
                            .await?;
                        }
                        None => {
                            reply_text(bot, &ctx, "ℹ️ No compaction needed.").await?;
                        }
                    }
                }
            } else {
                reply_text(bot, &ctx, "No active session.").await?;
            }
        }
        "/reload" => {
            agent.refresh_skills_and_mcp().await;
            let skills = agent.list_skills();
            let reply = format!(
                "✅ Reloaded.\n• skills: {}\n• Use /metrics for more.",
                skills.len()
            );
            reply_text(bot, &ctx, reply).await?;
        }
        "/provider" | "/providers" => {
            let arg = text[cmd_word.len()..].trim();
            tracing::debug!(chat_id, arg, "cmd /provider");
            if arg.is_empty() {
                let providers = agent.list_providers();
                tracing::debug!(n_providers = providers.len(), "listing providers");
                let (prov, model) = if let Some(sid) = channel_map.get(chat_id, tid).await {
                    agent.session_provider_model(&sid).await
                } else {
                    (
                        config.default_provider.clone(),
                        config.default_model.clone(),
                    )
                };
                tracing::debug!(%prov, %model, "current provider/model");
                let rows: Vec<Vec<InlineKeyboardButton>> = providers
                    .iter()
                    .map(|p| {
                        let mark = if p.name == prov { " ✅" } else { "" };
                        vec![InlineKeyboardButton::callback(
                            format!("{}{mark}", p.name),
                            format!("sp:{}", p.name),
                        )]
                    })
                    .collect();
                let kb = InlineKeyboardMarkup::new(rows);
                match bot
                    .send_message(
                        ctx.chat_id,
                        format!(
                            "Current: <b>{}</b> / <b>{}</b>\n\nSelect provider:",
                            escape_html(&prov),
                            escape_html(&model)
                        ),
                    )
                    .parse_mode(ParseMode::Html)
                    .reply_markup(kb)
                    .maybe_thread(ctx.thread_id)
                    .await
                {
                    Ok(m) => tracing::debug!(msg_id = m.id.0, "sent provider keyboard"),
                    Err(e) => {
                        tracing::error!("failed to send provider keyboard: {e}");
                        return Err(e);
                    }
                }
            } else {
                let sid = get_or_create_session(ctx, agent, channel_map, config).await;
                match agent.set_session_provider(&sid, Some(arg), None).await {
                    Ok(()) => {
                        let (prov, model) = agent.session_provider_model(&sid).await;
                        reply_text(bot, &ctx, format!("Switched to: {prov}/{model}")).await?;
                    }
                    Err(e) => {
                        reply_text(bot, &ctx, format!("Error: {e}")).await?;
                    }
                }
            }
        }
        "/model" | "/models" => {
            let arg = text[cmd_word.len()..].trim();
            if arg.is_empty() {
                let (prov, current_model) = if let Some(sid) = channel_map.get(chat_id, tid).await {
                    agent.session_provider_model(&sid).await
                } else {
                    (
                        config.default_provider.clone(),
                        config.default_model.clone(),
                    )
                };
                // Build model list: scoped (if configured) or all
                let all_models: Vec<(String, String)> = config
                    .providers
                    .iter()
                    .flat_map(|(p, pc)| pc.models.iter().map(move |m| (p.clone(), m.clone())))
                    .collect();
                let scope = &config.model_scope;
                let display_models = if scope.is_empty() {
                    // No scope: show current provider's models (legacy behavior)
                    provider_models(config, &prov)
                        .into_iter()
                        .map(|m| (prov.clone(), m))
                        .collect::<Vec<_>>()
                } else {
                    naked_tg::tg_markup::filter_models_by_scope(&all_models, scope)
                        .into_iter()
                        .cloned()
                        .collect()
                };

                if display_models.is_empty() {
                    reply_text(bot, &ctx, "No models match scope.").await?;
                } else {
                    const PAGE_SIZE: usize = 8;
                    let page = 0usize;
                    let total_pages = display_models.len().div_ceil(PAGE_SIZE);
                    let page_models = &display_models[page * PAGE_SIZE
                        ..(page * PAGE_SIZE + PAGE_SIZE).min(display_models.len())];

                    let mut rows: Vec<Vec<InlineKeyboardButton>> = page_models
                        .iter()
                        .map(|(p, m)| {
                            let label = if scope.is_empty() {
                                m.clone()
                            } else {
                                format!("{p}/{m}")
                            };
                            let mark = if *m == current_model && *p == prov {
                                " ✅"
                            } else {
                                ""
                            };
                            vec![InlineKeyboardButton::callback(
                                format!("{label}{mark}"),
                                format!("sm:{m}"),
                            )]
                        })
                        .collect();

                    // Pagination buttons
                    if total_pages > 1 {
                        let mut nav = Vec::new();
                        if page > 0 {
                            nav.push(InlineKeyboardButton::callback(
                                "◀ Prev",
                                format!("mp:{}", page - 1),
                            ));
                        }
                        nav.push(InlineKeyboardButton::callback(
                            format!("{}/{total_pages}", page + 1),
                            "mp:noop".to_string(),
                        ));
                        if page + 1 < total_pages {
                            nav.push(InlineKeyboardButton::callback(
                                "Next ▶",
                                format!("mp:{}", page + 1),
                            ));
                        }
                        rows.push(nav);
                    }

                    let kb = InlineKeyboardMarkup::new(rows);
                    let header = if scope.is_empty() {
                        format!(
                            "Provider: <b>{}</b>\nCurrent: <b>{}</b>",
                            escape_html_min(&prov),
                            escape_html_min(&current_model)
                        )
                    } else {
                        format!(
                            "Current: <b>{}/{}</b>\nShowing scoped models:",
                            escape_html_min(&prov),
                            escape_html_min(&current_model)
                        )
                    };
                    reply_html_kb(bot, &ctx, header, kb).await?;
                }
            } else {
                let sid = get_or_create_session(ctx, agent, channel_map, config).await;
                match agent.set_session_provider(&sid, None, Some(arg)).await {
                    Ok(()) => {
                        let (prov, model) = agent.session_provider_model(&sid).await;
                        reply_text(bot, &ctx, format!("Model set: {prov}/{model}")).await?;
                    }
                    Err(e) => {
                        reply_text(bot, &ctx, format!("Error: {e}")).await?;
                    }
                }
            }
        }
        "/reasoning" => {
            let sid = get_or_create_session(ctx, agent, channel_map, config).await;
            let current = agent.session_reasoning(&sid).await;
            let current_level = current.as_deref().unwrap_or("off");
            let levels = ["off", "low", "medium", "high"];
            let rows: Vec<Vec<InlineKeyboardButton>> = levels
                .iter()
                .map(|lvl| {
                    let mark = if *lvl == current_level { " ✅" } else { "" };
                    vec![InlineKeyboardButton::callback(
                        format!("{lvl}{mark}"),
                        format!("sr:{lvl}"),
                    )]
                })
                .collect();
            let kb = InlineKeyboardMarkup::new(rows);
            reply_html_kb(
                bot,
                &ctx,
                format!(
                    "💭 Reasoning: <b>{}</b>\n\nSelect level:",
                    escape_html(current_level)
                ),
                kb,
            )
            .await?;
        }
        "/skills" => {
            let skills = agent.list_skills();
            if skills.is_empty() {
                reply_text(bot, &ctx, "No skills loaded.").await?;
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
                reply_html(bot, &ctx, header).await?;
            }
        }
        "/mcp" => {
            let servers = agent.list_mcp_servers().await;
            if servers.is_empty() {
                reply_text(bot, &ctx, "No MCP servers connected.").await?;
            } else {
                let list: Vec<String> = servers
                    .iter()
                    .map(|(name, n_tools)| {
                        format!("• <b>{}</b> — {n_tools} tool(s)", escape_html(name))
                    })
                    .collect();
                let header = format!(
                    "🔌 <b>{} MCP server(s)</b>\n\n{}",
                    servers.len(),
                    list.join("\n")
                );
                reply_html(bot, &ctx, header).await?;
            }
        }
        "/refresh" => {
            agent.refresh_skills_and_mcp().await;
            let skills = agent.list_skills();
            let servers = agent.list_mcp_servers().await;
            reply_text(
                bot,
                &ctx,
                format!(
                    "🔄 Refreshed\n• {} skill(s)\n• {} MCP server(s)",
                    skills.len(),
                    servers.len()
                ),
            )
            .await?;
        }
        "/approve" | "/yolo" => {
            tracing::info!(chat_id, ?tid, "yolo: enabling");
            let already = channel_map.is_yolo(chat_id, tid).await;
            if already {
                reply_text(bot, &ctx, "⚡ YOLO already active.").await?;
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
                    &ctx,
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
        }
        "/allow" => {
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
                    reply_html(bot, &ctx, msg).await?;
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
                    reply_html(bot, &ctx, msg).await?;
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
                    text.push_str(
                        "\nUsage:\n<code>/allow add bash</code>\n<code>/allow rm bash</code>",
                    );
                    reply_html(bot, &ctx, text).await?;
                }
            }
        }
        "/research" => {
            handle_research_cmd(bot, agent, config, &ctx, text, cmd_word).await?;
        }
        "/memory" => {
            handle_memory_cmd(bot, agent, channel_map, &ctx, text, cmd_word).await?;
        }
        _ => {
            return Ok(false);
        }
    }

    Ok(true)
}

/// Start a research run for `spec_id` and install the live-progress
/// waterfall UI in the given chat. Returns the short confirmation
/// string that the caller sends as a separate message only if the
/// placeholder could not be posted — on the happy path this returns
/// `""` because the placeholder *is* the acknowledgement and the
/// heartbeat task keeps editing it.
///
/// Architecture:
///   1. Send a placeholder message with the `[⏸ Stop & clarify]` button,
///      captured `message_id` feeds the heartbeat + completion edits.
///   2. Spawn the research task on `agent.run_research*`. Result lands
///      on a oneshot that the heartbeat `select!`-awaits.
///   3. Spawn the heartbeat: every 20 s, snapshot `run_events`, edit the
///      placeholder. When the research future resolves, the heartbeat
///      flips into the completion branch — renders the report as HTML,
///      sends it as a document, then rewrites the placeholder with a
///      `[🔁 Run again]` button.
pub(crate) async fn launch_research_run_with_ui(
    bot: Bot,
    agent: Arc<AgentCore>,
    config: Config,
    chat_id: ChatId,
    thread_id: Option<ThreadId>,
    spec_id: String,
) -> String {
    let verify = config.research.verify_by_default;
    let max_rounds = config.research.gatekeeper.max_rounds;
    let iteration_cap = config.research.max_iterations.max(1);

    let spec_topic = match agent.research_store().load_spec(&spec_id).await {
        Ok(s) => s.topic,
        Err(e) => {
            return format!("error: spec `{spec_id}` not found ({e})");
        }
    };
    let findings_baseline = agent
        .research_store()
        .count_findings(&spec_id)
        .await
        .unwrap_or(0);

    let started_at = chrono::Utc::now();
    let progress0 = HeartbeatProgress {
        topic: spec_topic.clone(),
        started_at,
        findings_total: findings_baseline,
        findings_baseline,
        iteration_estimate: None,
        iteration_cap,
    };
    let initial_body = render_waterfall(&spec_id, &progress0, &[], started_at);

    let placeholder = match bot
        .send_message(chat_id, &initial_body)
        .maybe_thread(thread_id)
        .reply_markup(keyboard_stop(&spec_id))
        .await
    {
        Ok(m) => m,
        Err(e) => {
            return format!("error: failed to post placeholder: {e}");
        }
    };
    let msg_id = placeholder.id;

    // Oneshot carries the research future's result to the heartbeat
    // task. Using a channel (rather than `JoinHandle`) means the
    // heartbeat can `select!` on both the tick and the completion.
    let (done_tx, done_rx) = oneshot::channel::<ResearchOutcome>();

    let agent_for_run = agent.clone();
    let spec_for_run = spec_id.clone();
    tokio::spawn(async move {
        let outcome = if verify {
            match agent_for_run
                .run_research_verified(&spec_for_run, max_rounds)
                .await
            {
                Ok(vr) => ResearchOutcome::Verified(Box::new(vr)),
                Err(e) => ResearchOutcome::Error(format!("{e:#}")),
            }
        } else {
            match agent_for_run.run_research(&spec_for_run).await {
                Ok(r) => ResearchOutcome::Plain(Box::new(r)),
                Err(e) => ResearchOutcome::Error(format!("{e:#}")),
            }
        };
        let _ = done_tx.send(outcome);
    });

    // Heartbeat loop: edit the placeholder every 20 s with the latest
    // waterfall, or take the completion branch as soon as the run
    // future resolves.
    let agent_for_hb = agent.clone();
    let bot_for_hb = bot.clone();
    let spec_for_hb = spec_id.clone();
    let topic_for_hb = spec_topic.clone();
    tokio::spawn(async move {
        let mut tick = tokio::time::interval(Duration::from_secs(20));
        // Skip the immediate tick — the placeholder already reflects the
        // initial state.
        tick.tick().await;
        let mut done_rx = done_rx;
        let outcome: ResearchOutcome = loop {
            tokio::select! {
                res = &mut done_rx => {
                    break res.unwrap_or(ResearchOutcome::Error(
                        "internal: research task dropped".to_string(),
                    ));
                }
                _ = tick.tick() => {
                    let events = agent_for_hb
                        .research_run_events_snapshot(&spec_for_hb, 16)
                        .await;
                    let total = agent_for_hb
                        .research_store()
                        .count_findings(&spec_for_hb)
                        .await
                        .unwrap_or(findings_baseline);
                    let progress = HeartbeatProgress {
                        topic: topic_for_hb.clone(),
                        started_at,
                        findings_total: total,
                        findings_baseline,
                        iteration_estimate: None,
                        iteration_cap,
                    };
                    let body = render_waterfall(
                        &spec_for_hb,
                        &progress,
                        &events,
                        chrono::Utc::now(),
                    );
                    let edit = bot_for_hb
                        .edit_message_text(chat_id, msg_id, body)
                        .reply_markup(keyboard_stop(&spec_for_hb))
                        .await;
                    if let Err(e) = edit {
                        // Don't bail — a transient 400 "message is not
                        // modified" or rate-limit is routine. We keep
                        // ticking; the next edit will succeed or the
                        // completion path will replace the message anyway.
                        tracing::debug!(spec_id = %spec_for_hb, ?e, "heartbeat edit failed");
                    }
                }
            }
        };

        finalize_research_ui(
            &bot_for_hb,
            chat_id,
            thread_id,
            msg_id,
            &agent_for_hb,
            &spec_for_hb,
            &topic_for_hb,
            findings_baseline,
            started_at,
            outcome,
        )
        .await;
    });

    String::new()
}

/// Result of a background `run_research*` call, shuttled from the
/// launcher task to the heartbeat's completion branch. Boxed so the
/// enum stays small and copy-cheap for the oneshot channel.
pub(crate) enum ResearchOutcome {
    Plain(Box<naked_core::research::RunReport>),
    Verified(Box<naked_core::research::VerifiedRunReport>),
    Error(String),
}

#[allow(clippy::too_many_arguments)]
pub(crate) async fn finalize_research_ui(
    bot: &Bot,
    chat_id: ChatId,
    thread_id: Option<ThreadId>,
    msg_id: MessageId,
    agent: &Arc<AgentCore>,
    spec_id: &str,
    topic: &str,
    findings_baseline: u32,
    started_at: chrono::DateTime<chrono::Utc>,
    outcome: ResearchOutcome,
) {
    let store = agent.research_store();
    let total_after = store.count_findings(spec_id).await.unwrap_or(0);
    let new_this_run = total_after.saturating_sub(findings_baseline);
    let elapsed = (chrono::Utc::now() - started_at).num_seconds().max(0);

    let (summary, run_id_opt, is_error) = match &outcome {
        ResearchOutcome::Plain(r) => {
            let mut s = format!(
                "✅ <b>{}</b> — run complete\nspec <code>{}</code> · +{} finding(s) · total {} · {}s · stop={}",
                escape_html_min(topic),
                escape_html_min(spec_id),
                r.new_findings,
                r.total_findings_after,
                elapsed,
                r.stop_reason.as_str(),
            );
            s.push('\n');
            (s, Some(r.run_id.clone()), false)
        }
        ResearchOutcome::Verified(vr) => {
            let r = &vr.last_run;
            let s = format!(
                "✅ <b>{}</b> — run complete (gatekeeper {} round(s))\nspec <code>{}</code> · +{} new · total {} · removed={} · replacements={} · final={} · {}s\n",
                escape_html_min(topic),
                vr.verification_rounds,
                escape_html_min(spec_id),
                r.new_findings,
                r.total_findings_after,
                vr.dead_removed,
                vr.replacements_found,
                vr.final_findings,
                elapsed,
            );
            (s, Some(r.run_id.clone()), false)
        }
        ResearchOutcome::Error(err) => (
            format!(
                "❌ research run <code>{}</code> failed after {}s\n<pre>{}</pre>",
                escape_html_min(spec_id),
                elapsed,
                escape_html_min(err),
            ),
            None,
            true,
        ),
    };

    let _ = bot
        .edit_message_text(chat_id, msg_id, &summary)
        .parse_mode(ParseMode::Html)
        .reply_markup(keyboard_after_complete(spec_id))
        .await;

    if is_error {
        return;
    }

    // Render and ship the HTML report. Failure to materialise the
    // report is non-fatal — the run summary is already on the chat and
    // the caller can always pull `report.md` from disk.
    let report_md = match store.read_report(spec_id).await {
        Ok(Some(md)) => md,
        Ok(None) => {
            tracing::info!(spec_id, "no report.md to ship (empty run)");
            return;
        }
        Err(e) => {
            tracing::warn!(spec_id, ?e, "reading report.md failed");
            return;
        }
    };

    let meta = ReportMeta {
        spec_id,
        topic,
        run_id: run_id_opt.as_deref(),
        findings_total: total_after,
        new_findings: new_this_run,
        generated_at: chrono::Utc::now(),
    };
    let html = render_report_html(&meta, &report_md);
    let filename = format!("research-{}.html", safe_slug(spec_id));
    let caption = format!(
        "📄 Report for {} · {} finding(s) (+{} this run)",
        topic, total_after, new_this_run,
    );
    let input = teloxide::types::InputFile::memory(html).file_name(filename);
    if let Err(e) = bot
        .send_document(chat_id, input)
        .caption(caption)
        .maybe_thread(thread_id)
        .await
    {
        tracing::warn!(spec_id, ?e, "send_document(report.html) failed");
    }
}

/// `/research ...` — operator surface in Telegram for the research subsystem.
///
/// Mirrors the CLI (`naked research ...`) with two concessions:
///   1. `run` is spawned into a background task and replies "launched" so we
///      don't hold up the bot loop for the full agent turn (can be 20+ min).
///      The run result is not pushed back to this chat unless the spec has
///      its own `deliver_to` config (future v6); operators tail the log.
///   2. `schedule <id> <cron>` is stubbed with an explicit "not implemented
///      yet" reply — systemd timer templating is deferred to v6.
pub(crate) async fn handle_research_cmd(
    bot: &Bot,
    agent: &Arc<AgentCore>,
    config: &Config,
    ctx: &ChatCtx,
    text: &str,
    cmd_word: &str,
) -> Result<(), teloxide::RequestError> {
    if !config.research.enabled {
        reply_text(bot, ctx, "Research subsystem is disabled in config.").await?;
        return Ok(());
    }
    let rest = text[cmd_word.len()..].trim();
    let mut parts = rest.splitn(2, char::is_whitespace);
    let sub = parts.next().unwrap_or("").to_string();
    let tail = parts.next().unwrap_or("").trim().to_string();

    let reply = match sub.as_str() {
        "" | "help" => "\
/research new <topic>          create a new research spec
/research ls                    list specs with schedule + last-run metrics
/research show <id>             show spec summary + recent findings
/research state <id>            deep scheduler state — inflight ledger, failure streak, recent runs
/research fresh <id>            show only findings from the latest run
/research run <id>              launch a one-off run (background, gatekeeper-verified by default)
/research metrics <id>          detailed metrics for the latest run (gatekeeper rounds, etc.)
/research ask <id> <question>   ask the LLM a question grounded in the known findings
/research pause <id>            pause scheduled runs
/research resume <id>           resume scheduled runs
/research reset <id>            clear failure streak + pause_reason and resume (rearm after fixing the cause)
/research stop <id>             alias for pause
/research rm <id>               delete all data for a spec
/research schedule <id> on <interval>   schedule periodic runs (e.g. 30m, 1h, 1d, or seconds)
/research schedule <id> off              clear schedule
/research schedule <id> status           show current schedule
/research delta <id>            show findings from the latest run only
/research <свободный текст>     (soft-fallback) создаст spec и сразу запустит прогон

Tip: you can also just talk to me — \"расскажи как идёт исследование X\", \
\"исправь расписание X на каждый час\", \"добавь источник Y в X\" — \
the LLM has tools for all of this. А ещё свободный текст без слэша \
(\"исследуй помещения в Дананге, до $3000, на апрель 2026\") поднимает \
research-skill через LLM."
            .to_string(),
        "new" => {
            if tail.is_empty() {
                "Usage: /research new <topic>".to_string()
            } else {
                match agent
                    .create_research(
                        &tail,
                        Vec::new(),
                        None,
                        Some(ctx.chat_id.0),
                        ctx.raw_thread_id(),
                    )
                    .await
                {
                    Ok(spec) => format!("🔬 created `{}` — topic: {}", spec.id, spec.topic),
                    Err(e) => format!("error: {e}"),
                }
            }
        }
        "ls" => match agent.list_research().await {
            Ok(list) if list.is_empty() => "No research specs defined.".to_string(),
            Ok(list) => {
                let store = agent.research_store();
                let mut out = String::from("🔬 research specs:\n");
                for s in list {
                    let total = store.count_findings(&s.id).await.unwrap_or(0);
                    let runs = store.list_runs(&s.id, Some(20)).await.unwrap_or_default();
                    let last = runs
                        .iter()
                        .find(|r| r.verification_rounds.is_some())
                        .or_else(|| runs.first());
                    let inflight = store.load_inflight(&s.id).await.ok().flatten();
                    let status = if s.paused { "⏸" } else { "▶" };
                    let schedule = match s.interval_seconds {
                        Some(secs) => format_interval(secs),
                        None => "manual".to_string(),
                    };
                    out.push_str(&format!(
                        "{status} `{}` — {}\n   schedule: {} · findings: {}",
                        s.id, s.topic, schedule, total
                    ));
                    if s.paused
                        && let Some(reason) = s.pause_reason.as_deref()
                        && !reason.is_empty()
                    {
                        let short: String = reason.chars().take(120).collect();
                        out.push_str(&format!("\n   ⏸ {short}"));
                    }
                    if let Some(r) = last {
                        let age = (chrono::Utc::now() - r.finished_at).num_seconds().max(0) as u64;
                        out.push_str(&format!(
                            " · last: {} ago (+{} new",
                            format_age(age),
                            r.new_findings,
                        ));
                        if let Some(rounds) = r.verification_rounds {
                            out.push_str(&format!(
                                ", {rounds} rd, removed={}, replaced={}",
                                r.dead_removed.unwrap_or(0),
                                r.replacements_found.unwrap_or(0),
                            ));
                        }
                        out.push(')');
                    }
                    if let Some(infl) = inflight {
                        let icon = match infl.state {
                            naked_core::research::RunState::Scheduled => "🟡",
                            naked_core::research::RunState::Running => "🔵",
                            naked_core::research::RunState::Completed => "✅",
                            naked_core::research::RunState::Failed => "❌",
                        };
                        out.push_str(&format!(
                            "\n   state: {icon} {} (attempt {})",
                            infl.state.ru_label(),
                            infl.attempt,
                        ));
                        if let Some(err) = infl.error.as_deref()
                            && !err.is_empty()
                        {
                            let short: String = err.chars().take(80).collect();
                            out.push_str(&format!(" · err: {short}"));
                        }
                    }
                    out.push('\n');
                }
                out
            }
            Err(e) => format!("error: {e}"),
        },
        "metrics" => format_research_metrics(agent, &tail).await,
        "state" => {
            if tail.is_empty() {
                "Usage: /research state <id>".to_string()
            } else {
                match agent.load_research(&tail).await {
                    Ok(spec) => {
                        let store = agent.research_store();
                        let inflight = store.load_inflight(&spec.id).await.ok().flatten();
                        let recent_runs =
                            store.list_runs(&spec.id, Some(5)).await.unwrap_or_default();
                        let total_findings =
                            store.count_findings(&spec.id).await.unwrap_or(0) as u64;
                        let (failure_streak, alert_fired) = agent
                            .scheduler_failure_snapshot(&spec.id)
                            .await
                            .unwrap_or((0, false));
                        let view = naked_core::research::StateView {
                            spec: &spec,
                            inflight: inflight.as_ref(),
                            recent_runs: &recent_runs,
                            recent_runs_limit: 5,
                            failure_streak,
                            alert_fired,
                            total_findings,
                            now: chrono::Utc::now(),
                        };
                        naked_core::research::render_state(&view)
                    }
                    Err(e) => format!("error: {e}"),
                }
            }
        }
        "show" | "fresh" | "delta" => {
            format_research_show(agent, &tail, sub == "fresh" || sub == "delta").await
        }
        "run" => {
            if tail.is_empty() {
                "Usage: /research run <id>".to_string()
            } else {
                launch_research_run_with_ui(
                    bot.clone(),
                    agent.clone(),
                    config.clone(),
                    ctx.chat_id,
                    ctx.thread_id,
                    tail.clone(),
                )
                .await
            }
        }
        "ask" => {
            let mut ap = tail.splitn(2, char::is_whitespace);
            let id = ap.next().unwrap_or("").to_string();
            let question = ap.next().unwrap_or("").trim().to_string();
            if id.is_empty() || question.is_empty() {
                "Usage: /research ask <id> <question>".to_string()
            } else {
                match agent.ask_research(&id, &question).await {
                    Ok(answer) => format!("🔬 `{id}` — _{question}_\n\n{answer}"),
                    Err(e) => format!("error: {e}"),
                }
            }
        }
        "pause" | "stop" => {
            if tail.is_empty() {
                "Usage: /research pause <id>".to_string()
            } else {
                match agent.set_research_paused(&tail, true).await {
                    Ok(()) => format!("⏸ paused `{tail}`"),
                    Err(e) => format!("error: {e}"),
                }
            }
        }
        "resume" => {
            if tail.is_empty() {
                "Usage: /research resume <id>".to_string()
            } else {
                match agent.set_research_paused(&tail, false).await {
                    Ok(()) => format!("▶ resumed `{tail}`"),
                    Err(e) => format!("error: {e}"),
                }
            }
        }
        "reset" => {
            if tail.is_empty() {
                "Usage: /research reset <id>".to_string()
            } else {
                match agent.reset_research_failures(&tail).await {
                    Ok(()) => format!(
                        "🔄 reset `{tail}` — failure streak cleared, pause_reason cleared, resumed"
                    ),
                    Err(e) => format!("error: {e}"),
                }
            }
        }
        "rm" => {
            if tail.is_empty() {
                "Usage: /research rm <id>".to_string()
            } else {
                match agent.delete_research(&tail).await {
                    Ok(()) => format!("🗑 deleted `{tail}`"),
                    Err(e) => format!("error: {e}"),
                }
            }
        }
        "schedule" => {
            // In-process scheduler — no systemd. Stores `interval_seconds` on
            // the spec; the scheduler thread picks the change up via
            // SchedulerHook::notify and reschedules immediately.
            let mut sp = tail.splitn(3, char::is_whitespace);
            let id = sp.next().unwrap_or("").trim();
            let action = sp.next().unwrap_or("").trim();
            let arg = sp.next().unwrap_or("").trim();

            if id.is_empty() {
                "Usage: /research schedule <id> on <interval>|off|status\n\
                 <interval> can be seconds (`3600`) or a shorthand like `30m`, `1h`, `1d`."
                    .to_string()
            } else if !agent.config().research.schedule_enabled {
                "schedule_enabled=false in config — scheduling disabled".to_string()
            } else {
                match action {
                    "off" | "disable" => {
                        let patch = naked_core::ResearchPatch {
                            interval_seconds: Some(None),
                            ..Default::default()
                        };
                        match agent.update_research(id, patch).await {
                            Ok(_) => format!("⏹ schedule cleared for `{id}`"),
                            Err(e) => format!("error: {e}"),
                        }
                    }
                    "" | "on" | "enable" => {
                        let secs = if arg.is_empty() { Some(3600) } else { parse_interval(arg) };
                        schedule_research_on(agent, id, secs, arg).await
                    }
                    "status" => match agent.load_research(id).await {
                        Ok(spec) => match spec.interval_seconds {
                            Some(s) => format!(
                                "`{id}` schedule: every {}{}",
                                format_interval(s),
                                if spec.paused { " (paused)" } else { "" }
                            ),
                            None => format!("`{id}` schedule: off"),
                        },
                        Err(e) => format!("error: {e}"),
                    },
                    other => format!(
                        "unknown schedule action: {other}\n\
                         Usage: /research schedule <id> on <interval>|off|status"
                    ),
                }
            }
        }
        other => {
            // Soft-fallback: treat the whole tail as a free-text research
            // topic, create a spec and auto-run it in the background. This
            // rescues users who instinctively type `/research <тема>` — the
            // slash handler used to reply "unknown subcommand" and the LLM
            // never saw the request (see plan `fix_research_routing_and_anti-block`).
            //
            // Same UX as `/research run <id>`: we hand off to
            // `launch_research_run_with_ui`, which posts the live
            // waterfall + Stop & clarify button and ships the HTML
            // report on completion. Returning the empty string keeps
            // the trailing `send_message` quiet — the placeholder is
            // the acknowledgement.
            let topic = if tail.is_empty() {
                other.to_string()
            } else {
                format!("{other} {tail}")
            };
            match agent
                .create_research(
                    &topic,
                    Vec::new(),
                    None,
                    Some(ctx.chat_id.0),
                    ctx.raw_thread_id(),
                )
                .await
            {
                Ok(spec) => {
                    let spec_id = spec.id.clone();
                    let header = format!(
                        "🚀 создал `{}` — запускаю фоновый прогон…\nтема: {}",
                        spec_id, spec.topic
                    );
                    let _ = bot
                        .send_message(ctx.chat_id, header)
                        .maybe_thread(ctx.thread_id)
                        .await;
                    launch_research_run_with_ui(
                        bot.clone(),
                        agent.clone(),
                        config.clone(),
                        ctx.chat_id,
                        ctx.thread_id,
                        spec_id,
                    )
                    .await
                }
                Err(e) => format!(
                    "не смог создать research spec из `{other} {tail}`: {e}\n\
                     Попробуй `/research help`, либо отправь запрос свободным текстом без слэша."
                ),
            }
        }
    };

    // Plain text: research IDs contain hyphens and URLs can include
    // markdown-reserved chars (`_`, `*`, `[`). Escaping everything every time
    // is not worth the readability hit.
    //
    // The `run` arm manages its own live-progress message + keyboard, so
    // it returns an empty string here and we skip the trailing send.
    if reply.is_empty() {
        return Ok(());
    }
    reply_text(bot, ctx, reply).await?;
    Ok(())
}

/// `/memory ...` — operator surface over the daily-digest memory subsystem.
///
/// Resolves the workspace from the active session for this chat/topic so
/// project-scoped queries hit the same `MEMORY.md` the agent sees.
/// Falls back to `config.workspace` if no session is bound yet.
///
/// Subcommands:
///   - `ls` / `list`            — durable rules (MEMORY.md) for the project scope.
///   - `dreams`                 — last 7 entries of the digest audit log.
///   - `drafts`                 — today's draft buffer (pre-promotion).
///   - `stats`                  — counters (rules, drafts, promoted/rejected 7d).
///   - `help` / empty           — usage hint.
pub(crate) async fn handle_memory_cmd(
    bot: &Bot,
    agent: &Arc<AgentCore>,
    channel_map: &Arc<ChannelSessionMap>,
    ctx: &ChatCtx,
    text: &str,
    cmd_word: &str,
) -> Result<(), teloxide::RequestError> {
    use std::path::PathBuf;

    use naked_core::memory::dreams as memory_dreams;
    use naked_core::memory::service::MemoryService;
    use naked_core::memory::store::MarkdownMemoryStore;
    use naked_core::memory::types::MemoryScope;

    let rest = text[cmd_word.len()..].trim();
    let mut parts = rest.splitn(2, char::is_whitespace);
    let sub = parts.next().unwrap_or("").to_string();
    let _tail = parts.next().unwrap_or("").trim().to_string();

    // Resolve workspace via the session bound to (chat_id, thread). If
    // nothing is bound (fresh chat / pre-/new), `MemoryService::list`
    // tolerates a non-existent path and returns an empty list, so we
    // pick the agent's CWD as a sane fallback.
    let chat_id = ctx.chat_id.0;
    let tid = ctx.raw_thread_id();
    let workspace = if let Some(sid) = channel_map.get(chat_id, tid).await {
        agent
            .session_workspace(&sid)
            .await
            .unwrap_or_else(|| std::env::current_dir().unwrap_or_else(|_| PathBuf::from(".")))
    } else {
        std::env::current_dir().unwrap_or_else(|_| PathBuf::from("."))
    };

    let scope = MemoryScope::Project;

    let reply = match sub.as_str() {
        "" | "help" => "\
/memory ls            durable rules from project MEMORY.md
/memory dreams        last 7 daily-digest audit entries
/memory drafts        today's draft buffer (pre-promotion)
/memory stats         rules / drafts / promoted&rejected (7d)

The digest runs at 04:00 UTC by default. Drafts are auto-collected from \
session closes and pre-compaction flushes; rules promote when they \
re-appear across days."
            .to_string(),

        "ls" | "list" => {
            let entries = MemoryService::list(&workspace, Some(scope.clone()));
            if entries.is_empty() {
                "🧠 No durable rules in project MEMORY.md yet.".to_string()
            } else {
                let lines: Vec<String> = entries
                    .iter()
                    .map(|e| {
                        format!(
                            "• [{}] {}",
                            e.memory_type,
                            e.content.lines().next().unwrap_or("")
                        )
                    })
                    .collect();
                format!(
                    "🧠 <b>{} rule(s)</b> (project)\n\n{}",
                    entries.len(),
                    escape_html(&lines.join("\n"))
                )
            }
        }

        "dreams" => {
            let entries = memory_dreams::read_dreams(&workspace, &scope);
            if entries.is_empty() {
                "💭 No dream entries yet — the digest hasn't run for this project.".to_string()
            } else {
                let recent: Vec<_> = entries.iter().rev().take(7).collect();
                let mut lines = Vec::new();
                lines.push(format!("💭 <b>Last {} dream entries</b>", recent.len()));
                for d in recent {
                    lines.push(format!("\n<b>{}</b>", d.date.format("%Y-%m-%d")));
                    if !d.summary.trim().is_empty() {
                        lines.push(escape_html(&d.summary));
                    }
                    if !d.promoted.is_empty() {
                        lines.push(format!("✅ promoted ({}):", d.promoted.len()));
                        for p in &d.promoted {
                            lines.push(format!("  + {}", escape_html(p)));
                        }
                    }
                    if !d.rejected.is_empty() {
                        lines.push(format!("❌ rejected ({}):", d.rejected.len()));
                        for r in &d.rejected {
                            let reason = r.reason.as_deref().unwrap_or("");
                            lines.push(format!(
                                "  − {} {}",
                                escape_html(&r.content),
                                escape_html(reason)
                            ));
                        }
                    }
                }
                lines.join("\n")
            }
        }

        "drafts" => {
            let today = chrono::Utc::now().date_naive();
            let entries = MarkdownMemoryStore::read_daily(&workspace, &scope, today);
            if entries.is_empty() {
                format!("📝 No draft entries for {today} yet.")
            } else {
                let lines: Vec<String> = entries
                    .iter()
                    .map(|e| format!("• [{}/{}] {}", e.memory_type, e.source, e.content))
                    .collect();
                format!(
                    "📝 <b>{} draft(s) for {today}</b>\n\n{}",
                    entries.len(),
                    escape_html(&lines.join("\n"))
                )
            }
        }

        "stats" => {
            let durable = MemoryService::list(&workspace, Some(scope.clone())).len();
            let today = chrono::Utc::now().date_naive();
            let drafts_today = MarkdownMemoryStore::read_daily(&workspace, &scope, today).len();
            let dream_entries = memory_dreams::read_dreams(&workspace, &scope);
            let week_cutoff = today - chrono::Duration::days(7);
            let recent: Vec<_> = dream_entries
                .iter()
                .filter(|d| d.date >= week_cutoff)
                .collect();
            let promoted_7d: usize = recent.iter().map(|d| d.promoted.len()).sum();
            let rejected_7d: usize = recent.iter().map(|d| d.rejected.len()).sum();
            let last_run = dream_entries
                .iter()
                .map(|d| d.date)
                .max()
                .map(|d| d.to_string())
                .unwrap_or_else(|| "never".to_string());
            format!(
                "📊 <b>Memory stats</b> (project)\n\
                 • durable rules: <b>{durable}</b>\n\
                 • drafts today: <b>{drafts_today}</b>\n\
                 • dream entries (7d): <b>{}</b>\n\
                 • promoted (7d): <b>{promoted_7d}</b>\n\
                 • rejected (7d): <b>{rejected_7d}</b>\n\
                 • last digest run: <b>{last_run}</b>",
                recent.len(),
            )
        }

        other => format!("Unknown subcommand: {other}. Try /memory help"),
    };

    reply_html(bot, ctx, reply).await?;
    Ok(())
}

pub(crate) async fn get_or_create_session(
    ctx: ChatCtx,
    agent: &Arc<AgentCore>,
    channel_map: &Arc<ChannelSessionMap>,
    config: &Config,
) -> String {
    let chat_id = ctx.chat_id.0;
    let tid = ctx.raw_thread_id();
    if let Some(sid) = channel_map.get(chat_id, tid).await {
        return sid;
    }

    // Per-chat persona: when `naked.json` declares `chat_personas[<chat_id>]`,
    // the new session is rooted in the persona's dedicated workspace instead
    // of the global `config.workspace`. This is the single switch that gives
    // each persona its own system prompt, project memory namespace,
    // CLAUDE.md/AGENTS.md walk, and default `bash` cwd — without forking the
    // bot process. See `naked-core::config::ChatPersona` for the contract.
    let workspace = resolve_session_workspace(chat_id, config).await;

    let session_id = agent
        .create_session_with_channel(&workspace, "telegram")
        .await;
    channel_map.set(chat_id, tid, session_id.clone()).await;

    let channel_id = format_tg_channel_id(chat_id, tid);
    agent.set_session_channel_id(&session_id, &channel_id).await;
    session_id
}

/// Drop a Telegram slash command in a persona chat that opted out of the
/// command surface.
///
/// Returns `true` when the message must be discarded (i.e. the chat has a
/// persona and `allow_slash_commands == false`). The caller should
/// short-circuit `handle_message` immediately. On the FIRST drop per chat
/// we also reply with a one-line hint so the operator knows their `/cmd`
/// was intentional dead air, not a bot bug.
pub(crate) async fn drop_slash_for_persona(
    bot: &Bot,
    chat_id: i64,
    msg: &Message,
    config: &Config,
) -> bool {
    let Some(persona) = config.chat_personas.get(&chat_id) else {
        return false;
    };
    if persona.allow_slash_commands {
        return false;
    }

    let already_hinted = SLASH_HINT_SHOWN.read().await.contains(&chat_id);
    if !already_hinted {
        let inserted = SLASH_HINT_SHOWN.write().await.insert(chat_id);
        if inserted {
            let hint = "В этом чате слеш-команды отключены. Скажи естественным \
                        языком, что нужно (например: «выручка вчера», «маржа \
                        за март», «переключи модель на gpt-5») — я разберусь.";
            if let Err(e) = bot
                .send_message(msg.chat.id, hint)
                .maybe_thread(msg.thread_id)
                .await
            {
                tracing::warn!(
                    chat_id,
                    persona = %persona.name,
                    "failed to send slash-disabled hint: {e}"
                );
            }
        }
    }
    tracing::info!(
        chat_id,
        persona = %persona.name,
        "dropping slash command — persona has allow_slash_commands=false"
    );
    true
}

/// Pick the workspace path for a brand-new session bound to `chat_id`.
///
/// Falls back to `config.workspace` (the legacy single-workspace bot
/// behaviour) when no persona is configured for this chat. When a persona
/// IS configured, ensures its workspace directory and the
/// `<workspace>/.naked` subdirectory exist so the first turn doesn't fail
/// reading a non-existent prompt file. Failures during `mkdir` are logged
/// at warn-level but do not abort session creation — the agent will still
/// boot using the built-in default prompt if the persona's prompt file is
/// unreadable, which is strictly better than refusing to answer.
pub(crate) async fn resolve_session_workspace(chat_id: i64, config: &Config) -> std::path::PathBuf {
    if let Some(persona) = config.chat_personas.get(&chat_id) {
        let ws = persona.workspace_expanded();
        if let Err(e) = tokio::fs::create_dir_all(ws.join(".naked")).await {
            tracing::warn!(
                chat_id,
                persona = %persona.name,
                workspace = %ws.display(),
                "failed to ensure persona workspace dir exists: {e}"
            );
        }
        tracing::info!(
            chat_id,
            persona = %persona.name,
            workspace = %ws.display(),
            "creating new session under persona workspace"
        );
        return ws;
    }
    config.workspace.clone()
}

// ── Formatting helpers ──────────────────────────────────────────────────────

async fn schedule_research_on(
    agent: &Arc<AgentCore>,
    id: &str,
    secs: Option<u64>,
    arg: &str,
) -> String {
    let Some(s) = secs else {
        return format!("bad interval `{arg}` — try `3600`, `30m`, `1h`, `1d`");
    };
    let patch = naked_core::ResearchPatch {
        interval_seconds: Some(Some(s)),
        ..Default::default()
    };
    match agent.update_research(id, patch).await {
        Ok(spec) => format!(
            "⏰ scheduled `{id}` — every {} (verify={})",
            format_interval(spec.interval_seconds.unwrap_or(s)),
            agent.config().research.verify_by_default
        ),
        Err(e) => format!("error: {e}"),
    }
}

async fn format_research_show(agent: &Arc<AgentCore>, tail: &str, fresh_only: bool) -> String {
    let usage = if fresh_only {
        "Usage: /research fresh <id>"
    } else {
        "Usage: /research show <id>"
    };
    if tail.is_empty() {
        return usage.to_string();
    }
    let spec = match agent.load_research(tail).await {
        Ok(s) => s,
        Err(e) => return format!("error: {e}"),
    };
    let store = agent.research_store();
    let total = store.count_findings(&spec.id).await.unwrap_or(0);
    let runs = store.list_runs(&spec.id, Some(1)).await.unwrap_or_default();
    let mut findings = if fresh_only {
        store
            .list_findings(&spec.id, None)
            .await
            .unwrap_or_default()
    } else {
        store
            .list_findings(&spec.id, Some(5))
            .await
            .unwrap_or_default()
    };
    if fresh_only && let Some(last_run) = runs.last() {
        let run_id = &last_run.run_id;
        findings.retain(|f| f.run_id == *run_id);
    }
    let sources = if spec.sources.is_empty() {
        "(auto)".to_string()
    } else {
        spec.sources.join(", ")
    };
    let mut out = if fresh_only {
        format!(
            "🔬 `{}`\ntopic: {}\nfindings: {} total, {} fresh (latest run)\n",
            spec.id,
            spec.topic,
            total,
            findings.len()
        )
    } else {
        format!(
            "🔬 `{}`\ntopic: {}\nsources: {sources}\npaused: {}\nfindings: {total}\n",
            spec.id, spec.topic, spec.paused,
        )
    };
    if !findings.is_empty() {
        out.push_str(if fresh_only {
            "\nfresh:\n"
        } else {
            "\nrecent:\n"
        });
        for f in findings.iter().rev() {
            let title = f.title.as_deref().unwrap_or("(untitled)");
            let date_str = f.listing_date.as_deref().unwrap_or("");
            if date_str.is_empty() {
                out.push_str(&format!("• {title} — {}\n", f.url));
            } else {
                out.push_str(&format!("• {title} [{date_str}] — {}\n", f.url));
            }
        }
    }
    out
}

async fn format_research_metrics(agent: &Arc<AgentCore>, tail: &str) -> String {
    if tail.is_empty() {
        return "Usage: /research metrics <id>".to_string();
    }
    let spec = match agent.load_research(tail).await {
        Ok(s) => s,
        Err(e) => return format!("error: {e}"),
    };
    let store = agent.research_store();
    let total = store.count_findings(&spec.id).await.unwrap_or(0);
    let runs = store.list_runs(&spec.id, Some(5)).await.unwrap_or_default();
    let sources = if spec.sources.is_empty() {
        "(auto)".to_string()
    } else {
        spec.sources.join(", ")
    };
    let schedule = match spec.interval_seconds {
        Some(s) => format_interval(s),
        None => "manual".to_string(),
    };
    let mut out = format!(
        "🔬 metrics for `{}`\ntopic: {}\npaused: {}\nsources: {sources}\nschedule: {schedule}\ntotal findings: {total}\n",
        spec.id, spec.topic, spec.paused,
    );
    if runs.is_empty() {
        out.push_str(&format!(
            "\n(no runs yet — `/research run {}` to start)",
            spec.id
        ));
    } else {
        out.push_str("\nrecent runs:\n");
        format_run_list(&runs, &mut out);
    }
    out
}

fn format_run_list(runs: &[naked_core::research::RunRecord], out: &mut String) {
    for r in runs {
        let age = (chrono::Utc::now() - r.finished_at).num_seconds().max(0) as u64;
        out.push_str(&format!(
            "• `{}` — {} ago · stop={} · +{} new (total {})",
            r.run_id,
            format_age(age),
            r.stop_reason,
            r.new_findings,
            r.total_findings_after,
        ));
        if let Some(elapsed) = r.elapsed_secs {
            out.push_str(&format!(" · {}s", elapsed));
        }
        if let Some(rounds) = r.verification_rounds {
            out.push_str(&format!(
                "\n   gatekeeper: {} rd · removed={} · replaced={} · remaining={}",
                rounds,
                r.dead_removed.unwrap_or(0),
                r.replacements_found.unwrap_or(0),
                r.remaining_issues.unwrap_or(0),
            ));
        }
        out.push('\n');
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn safe_slug_basic() {
        assert_eq!(safe_slug("hello world"), "hello-world");
        assert_eq!(safe_slug("my-topic_v2"), "my-topic_v2");
        assert_eq!(safe_slug("  spaces  everywhere  "), "-spaces-everywhere-");
    }

    #[test]
    fn safe_slug_unicode() {
        assert_eq!(safe_slug("квартиры Самуи"), "-");
        assert_eq!(safe_slug("test квартиры"), "test-");
    }

    #[test]
    fn safe_slug_empty() {
        assert_eq!(safe_slug(""), "run");
        assert_eq!(safe_slug("   "), "-"); // all spaces collapse to single dash
    }

    #[test]
    fn escape_html_min_entities() {
        assert_eq!(escape_html_min("a < b > c & d"), "a &lt; b &gt; c &amp; d");
        assert_eq!(escape_html_min("no special"), "no special");
        assert_eq!(escape_html_min(""), "");
    }

    #[test]
    fn escape_html_min_preserves_quotes() {
        // Unlike full escape_html, this minimal version does NOT escape quotes
        assert_eq!(escape_html_min("he said \"hi\""), "he said \"hi\"");
    }
}
