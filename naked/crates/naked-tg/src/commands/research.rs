//! Research command handlers.
//!
//! B4 (PLAN_RESEARCH_FLOW_CLOSURE_v1): `launch_research_run_with_ui`,
//! `finalize_research_ui`, `ResearchOutcome`, and the heartbeat-waterfall
//! UI have been removed.  All research runs now go through the unified
//! synthetic-dispatch path (`synthetic_dispatch::dispatch_for_chat`) which
//! lets the agent's normal turn loop handle progress output and abort.

use super::super::fmt_utils::{format_age, format_interval};
use super::super::*;

use std::collections::HashMap;
use std::sync::LazyLock;
use tokio::sync::RwLock;

/// Per-chat mapping of index (1-based) → spec ID, updated on each `/research list`.
/// Allows `/research show 3` instead of copying the full ID.
static LIST_INDEX: LazyLock<RwLock<HashMap<i64, Vec<String>>>> =
    LazyLock::new(|| RwLock::new(HashMap::new()));

/// If `tail` is a decimal number, resolve it to a spec ID from the last
/// `/research list` for this chat. Otherwise return the original string.
async fn resolve_spec_ref(chat_id: i64, tail: &str) -> String {
    if let Ok(n) = tail.parse::<usize>()
        && n >= 1
    {
        // Try cached LIST_INDEX first (populated by /research ls).
        let map = LIST_INDEX.read().await;
        if let Some(ids) = map.get(&chat_id)
            && let Some(id) = ids.get(n - 1)
        {
            return id.clone();
        }
        drop(map);
        // Fallback: scan research dir directly so `/research run 2`
        // works even without a prior `/research ls` in this session.
        let research_dir = naked_core::research::store::research_root();
        if research_dir.is_dir() {
            let mut ids: Vec<String> = std::fs::read_dir(&research_dir)
                .into_iter()
                .flatten()
                .filter_map(|e| {
                    let p = e.ok()?.path();
                    if p.join("spec.json").exists() {
                        Some(p.file_name()?.to_string_lossy().into_owned())
                    } else {
                        None
                    }
                })
                .collect();
            ids.sort();
            if let Some(id) = ids.get(n - 1) {
                return id.clone();
            }
        }
    }
    tail.to_string()
}

fn recurring_marker(spec: &naked_core::research::ResearchSpec) -> &'static str {
    if spec.interval_seconds.is_some() || spec.cron.is_some() {
        " 🔁"
    } else {
        ""
    }
}

// Dead-code marker — the legacy orchestration layer (heartbeat waterfall,
// stop/restart keyboards, ResearchOutcome enum, finalize_research_ui) is gone.
// All research runs go through synthetic_dispatch::dispatch_for_chat.

/// `/research ...` — operator surface in Telegram for the research subsystem.
///
/// Mirrors the CLI (`naked research ...`) with two concessions:
///   1. `run` injects a synthetic prompt into the current chat session via
///      [`naked_tg::synthetic::dispatch_for_chat`] so the agent's
///      normal turn loop calls `research_run(spec_id=X)`.  Abort works via
///      the standard `/abort` command or the ⏹ inline button.
///   2. `schedule <id> <cron>` is stubbed with an explicit "not implemented
///      yet" reply — systemd timer templating is deferred to v6.
pub(crate) async fn handle_research_cmd(
    deps: &crate::message_handler::BotDeps,
    ctx: &ChatCtx,
    text: &str,
    cmd_word: &str,
) -> Result<(), teloxide::RequestError> {
    let bot = &deps.bot;
    let agent = &deps.agent;
    let channel_map = &deps.channel_map;
    let config = &deps.config;
    if !config.research.enabled {
        reply_text(bot, ctx, "Research subsystem is disabled in config.").await?;
        return Ok(());
    }
    let rest = text[cmd_word.len()..].trim();
    let mut parts = rest.splitn(2, char::is_whitespace);
    let sub = parts.next().unwrap_or("").to_string();
    let raw_tail = parts.next().unwrap_or("").trim().to_string();
    // Resolve numeric refs: "/research show 3" → spec ID from last list.
    let tail = resolve_spec_ref(ctx.chat_id.0, &raw_tail).await;

    let reply = match sub.as_str() {
        "" | "help" => "\
/research list                  нумерованный список
/research runs                  live runs in this chat/thread
/research show <run_id>         live mirror текущего in-flight run
/research steer <run_id> <text> steer a specific live run
/research move <run_id> here    move live run bubble to this chat/thread
/research fresh <N|id>          stored spec + свежие находки
/research delta <N|id>          alias fresh
/research run <N|id>            запустить прогон
/research ask <N|id> <вопрос>   спросить по находкам
/research pause <N|id>          пауза
/research resume <N|id>         снять паузу
/research reset <N|id>          сбросить ошибки + resume
/research rm <N|id>             удалить
/research schedule <N|id> on <30m|1h|1d>
/research schedule <N|id> off
/research new <тема>            новый spec
/research <свободный текст>     создаст spec + запустит

<N> — номер из /research list"
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
                        naked_core::CreateSchedule::OneShotNow,
                    )
                    .await
                {
                    Ok(spec) => format!("🔬 created `{}` — topic: {}", spec.id, spec.topic),
                    Err(e) => format!("error: {e}"),
                }
            }
        }
        // T1 (PLAN_RESEARCH_AGENT_FLOW_v1): `list` alias for `ls`.
        // Operator instinct: every other tool calls it `list`. Without this
        // alias `/research list` would hit the soft-fallback arm and create
        // a spec with topic="list" (incident 2026-05-16 14:14 UTC,
        // BUG_REGISTRY B55 COMMAND-FALLTHROUGH-CREATES-RESOURCE).
        "ls" | "list" => match agent.list_research().await {
            Ok(list) if list.is_empty() => "No research specs defined.".to_string(),
            Ok(list) => {
                let store = agent.research_store();
                let mut ids: Vec<String> = Vec::with_capacity(list.len());
                let mut out = String::from("🔬 research specs:\n");
                for (i, s) in list.iter().enumerate() {
                    let n = i + 1;
                    let total = store.count_findings(&s.id).await.unwrap_or(0);
                    let status = if s.paused { "⏸" } else { "▶" };
                    let short_topic: String = s.topic.chars().take(50).collect();
                    let ellip = if s.topic.chars().count() > 50 {
                        "…"
                    } else {
                        ""
                    };
                    let last_run = store
                        .list_runs(&s.id, Some(1))
                        .await
                        .ok()
                        .and_then(|r| r.first().map(|r| r.finished_at))
                        .map(|dt| {
                            let age = (chrono::Utc::now() - dt).num_seconds().max(0) as u64;
                            format_age(age)
                        })
                        .unwrap_or_else(|| "—".into());
                    let recurring = recurring_marker(s);
                    out.push_str(&format!(
                        "{status} {n}. {short_topic}{ellip} ({total}) · {last_run}{recurring}\n"
                    ));
                    ids.push(s.id.clone());
                }
                out.push_str(
                    "\n`/research fresh|run|pause|rm <N>` · off: `/research schedule <N> off` (live: `/research show <run_id>`)",
                );
                // Store index for numeric refs.
                LIST_INDEX.write().await.insert(ctx.chat_id.0, ids);
                out
            }
            Err(e) => format!("error: {e}"),
        },
        "metrics" => format_research_metrics(agent, &tail).await,
        "runs" => {
            reply_html(
                bot,
                ctx,
                format_live_runs_for_thread(
                    &crate::shared::RUN_REGISTRY,
                    ctx.chat_id.0,
                    ctx.raw_thread_id(),
                ),
            )
            .await?;
            String::new()
        }
        "steer" => {
            let msg = explicit_steer_live_run(&crate::shared::RUN_REGISTRY, ctx, &raw_tail).await;
            reply_html(bot, ctx, msg).await?;
            String::new()
        }
        "move" => {
            move_live_run_here(deps, ctx, &raw_tail).await?;
            String::new()
        }
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
        "show" => {
            send_live_run_snapshot(bot, ctx, &crate::shared::RUN_REGISTRY, &raw_tail).await?;
            String::new()
        }
        "fresh" | "delta" => format_research_show(agent, &tail, true).await,
        "run" => {
            // PLAN_BG_UNIFY_v2 T3: dispatch through scheduler for
            // heartbeat, timeout, resurrect, and concurrency cap.
            if tail.is_empty() {
                "Usage: /research run <id>".to_string()
            } else {
                dispatch_research(deps, ctx, agent, channel_map, config, &tail).await
            }
        }
        "ask" => {
            let mut ap = raw_tail.splitn(2, char::is_whitespace);
            let raw_id = ap.next().unwrap_or("").to_string();
            let id = resolve_spec_ref(ctx.chat_id.0, &raw_id).await;
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
        // T1 alias: `delete` for symmetry с `list`/`ls`.
        "rm" | "delete" => {
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
            let mut sp = raw_tail.splitn(3, char::is_whitespace);
            let raw_id = sp.next().unwrap_or("").trim();
            let id_resolved = resolve_spec_ref(ctx.chat_id.0, raw_id).await;
            let id = id_resolved.as_str();
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
                        match agent
                            .set_research_schedule(id, naked_core::ScheduleUpdate::Off)
                            .await
                        {
                            Ok(()) => format!("⏹ schedule cleared for `{id}`"),
                            Err(e) => format!("error: {e}"),
                        }
                    }
                    "" | "on" | "enable" => {
                        let secs = if arg.is_empty() {
                            Some(3600)
                        } else {
                            parse_interval(arg)
                        };
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
            // B1 (PLAN_RESEARCH_FLOW_CLOSURE_v1): same synthetic-dispatch
            // path as the explicit `/research run <id>` arm.  The agent's
            // turn loop calls `research_run` which does the actual work.
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
                    naked_core::CreateSchedule::OneShotNow,
                )
                .await
            {
                Ok(spec) => {
                    let spec_id = spec.id.clone();
                    let header = format!(
                        "🚀 создал `{}` — запускаю фоновый прогон…\nтема: {}",
                        spec_id, spec.topic
                    );
                    let _ = crate::shared::safe_send(bot, ctx, header, None).await;
                    dispatch_research(deps, ctx, agent, channel_map, config, &spec_id).await
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

/// DRY helper — session + yolo + dispatch_immediate.
async fn dispatch_research(
    deps: &crate::message_handler::BotDeps,
    ctx: &crate::shared::ChatCtx,
    agent: &std::sync::Arc<naked_core::AgentCore>,
    channel_map: &std::sync::Arc<naked_tg::channel_map::ChannelSessionMap>,
    config: &naked_core::config::Config,
    spec_id: &str,
) -> String {
    let Some(sched) = &deps.research_scheduler else {
        return "Research scheduler not running".into();
    };
    let session_id = super::get_or_create_session(*ctx, agent, channel_map, config).await;
    channel_map
        .enable_yolo(ctx.chat_id.0, ctx.raw_thread_id())
        .await;
    let prompt = format!(
        "Run research spec_id={spec_id}. Call research_run(spec_id=\"{spec_id}\") directly."
    );
    match sched
        .dispatch_immediate(
            spec_id,
            ctx.chat_id.0,
            ctx.raw_thread_id(),
            session_id,
            prompt,
        )
        .await
    {
        Ok(attempt_id) => format!("\u{1f52c} research `{spec_id}` dispatched ({attempt_id})"),
        Err(e) => format!("dispatch error: {e}"),
    }
}

async fn schedule_research_on(
    agent: &Arc<AgentCore>,
    id: &str,
    secs: Option<u64>,
    arg: &str,
) -> String {
    let Some(s) = secs else {
        return format!("bad interval `{arg}` — try `3600`, `30m`, `1h`, `1d`");
    };
    match agent
        .set_research_schedule(id, naked_core::ScheduleUpdate::Interval(s))
        .await
    {
        Ok(()) => format!(
            "⏰ scheduled `{id}` — every {} (verify={})",
            format_interval(s),
            agent.config().research.verify_by_default
        ),
        Err(e) => format!("error: {e}"),
    }
}

pub(crate) async fn send_live_run_snapshot(
    bot: &Bot,
    ctx: &ChatCtx,
    registry: &naked_tg::run_registry::RunRegistry,
    run_id: &str,
) -> Result<(), teloxide::RequestError> {
    let run_id = run_id.trim();
    let Some(run) = registry.get_run(run_id) else {
        reply_html(bot, ctx, format_live_run_snapshot(registry, run_id)).await?;
        return Ok(());
    };

    if !matches!(
        run.status,
        naked_tg::run_registry::RunStatus::Streaming
            | naked_tg::run_registry::RunStatus::AwaitingTool
            | naked_tg::run_registry::RunStatus::Idle
    ) {
        reply_html(bot, ctx, format_live_run_snapshot(registry, run_id)).await?;
        return Ok(());
    }

    if registry.mirror_sinks(run_id).len() >= naked_tg::run_registry::MIRROR_SINK_CAP {
        reply_text(
            bot,
            ctx,
            "⚠️ mirror limit reached for this run (home + 2 mirrors).",
        )
        .await?;
        return Ok(());
    }

    let initial_html = if run.rendered.live_html.trim().is_empty() {
        format!(
            "⏳ mirroring run <code>{}</code>…",
            naked_tg::markup::escape_html(run_id)
        )
    } else {
        run.rendered.live_html.clone()
    };
    let sent = bot
        .send_message(ctx.chat_id, initial_html)
        .parse_mode(ParseMode::Html)
        .maybe_thread(ctx.thread_id)
        .maybe_reply_to(ctx.reply_to)
        .reply_markup(crate::streaming::streaming_control_kb_for_run(run_id))
        .await?;
    match registry.add_mirror_sink(
        run_id,
        naked_tg::run_registry::RunSink::new(ctx.chat_id.0, ctx.raw_thread_id(), sent.id.0),
    ) {
        Ok(_) => {}
        Err(naked_tg::run_registry::MirrorSinkError::SinkCapacityExceeded { .. }) => {
            let _ = bot
                .edit_message_text(
                    ctx.chat_id,
                    sent.id,
                    "⚠️ mirror limit reached for this run (home + 2 mirrors).",
                )
                .await;
            let _ = bot.edit_message_reply_markup(ctx.chat_id, sent.id).await;
        }
        Err(_) => {
            let _ = bot
                .edit_message_text(ctx.chat_id, sent.id, "run not found or expired")
                .await;
            let _ = bot.edit_message_reply_markup(ctx.chat_id, sent.id).await;
        }
    }
    Ok(())
}

pub(crate) fn format_live_runs_for_thread(
    registry: &naked_tg::run_registry::RunRegistry,
    chat_id: i64,
    thread_id: Option<i32>,
) -> String {
    let runs = registry.list_for_thread(naked_tg::run_registry::ChatThreadKey::new(
        chat_id, thread_id,
    ));
    if runs.is_empty() {
        return "No live runs in this chat/thread.".to_string();
    }
    let mut out = String::from("🏃 <b>live runs here</b>\n");
    for (idx, run) in runs
        .iter()
        .take(naked_tg::run_registry::MULTI_RUN_THREAD_CAP)
        .enumerate()
    {
        let n = idx + 1;
        let run_id = naked_tg::markup::escape_html(&run.run_id);
        let label = run
            .source_ref
            .as_deref()
            .map(naked_tg::markup::escape_html)
            .unwrap_or_else(|| match &run.kind {
                naked_tg::run_registry::RunKind::ChatTurn => "chat-turn".to_string(),
                naked_tg::run_registry::RunKind::Research { spec_id } => {
                    naked_tg::markup::escape_html(spec_id)
                }
                naked_tg::run_registry::RunKind::SubAgent { label } => {
                    naked_tg::markup::escape_html(label)
                }
            });
        let status = naked_tg::markup::escape_html(&format!("{:?}", run.status));
        out.push_str(&format!(
            "{n}. <code>{run_id}</code> · {label} · <code>{status}</code> · origin <code>{}:{:?}</code>\n",
            run.origin.chat_id, run.origin.thread_id
        ));
    }
    out.push_str("\nUse <code>/research steer &lt;run_id&gt; &lt;text&gt;</code> or <code>/research show &lt;run_id&gt;</code>.");
    out
}

pub(crate) async fn move_live_run_here(
    deps: &crate::message_handler::BotDeps,
    ctx: &ChatCtx,
    raw_tail: &str,
) -> Result<(), teloxide::RequestError> {
    let mut parts = raw_tail.split_whitespace();
    let run_id = parts.next().unwrap_or("");
    let here = parts.next().unwrap_or("");
    if run_id.is_empty() || here != "here" || parts.next().is_some() {
        reply_text(&deps.bot, ctx, "Usage: /research move <run_id> here").await?;
        return Ok(());
    }
    let registry = &crate::shared::RUN_REGISTRY;
    let new_origin = naked_tg::run_registry::RunOrigin::new(ctx.chat_id.0, ctx.raw_thread_id());
    let options = if deps.config.run_registry_multi_stream_enabled {
        naked_tg::run_registry::RegisterRunOptions::cap_three()
    } else {
        naked_tg::run_registry::RegisterRunOptions::step2_single_run()
    };

    let sent = deps
        .bot
        .send_message(
            ctx.chat_id,
            format!(
                "↪ moved here: <code>{}</code>",
                naked_tg::markup::escape_html(run_id)
            ),
        )
        .parse_mode(ParseMode::Html)
        .maybe_thread(ctx.thread_id)
        .await?;

    let plan = match registry.move_run(run_id, new_origin.clone(), options) {
        Ok(plan) => plan,
        Err(naked_tg::run_registry::MoveRunError::RunNotFound { .. }) => {
            let _ = deps
                .bot
                .edit_message_text(ctx.chat_id, sent.id, "run not found or expired")
                .await;
            let _ = deps
                .bot
                .edit_message_reply_markup(ctx.chat_id, sent.id)
                .await;
            return Ok(());
        }
        Err(naked_tg::run_registry::MoveRunError::ThreadCapacityExceeded {
            active, cap, ..
        }) => {
            let _ = deps
                .bot
                .edit_message_text(
                    ctx.chat_id,
                    sent.id,
                    format!(
                        "⚠️ destination already has {active}/{cap} active runs — abort or wait before moving here."
                    ),
                )
                .await;
            let _ = deps
                .bot
                .edit_message_reply_markup(ctx.chat_id, sent.id)
                .await;
            return Ok(());
        }
    };

    let bound = registry.bind_message(
        &plan.run_id,
        naked_tg::run_registry::MessageKey::new(ctx.chat_id.0, sent.id.0),
    );
    match bound {
        Ok(_) => {
            crate::shared::CONTROL_CARDS
                .write()
                .await
                .insert(plan.run_id.clone(), (ctx.chat_id, sent.id));
            if let Err(e) = deps
                .bot
                .edit_message_reply_markup(ctx.chat_id, sent.id)
                .reply_markup(crate::streaming::streaming_control_kb_for_run(&plan.run_id))
                .await
            {
                tracing::warn!(
                    run_id = %plan.run_id,
                    chat = ctx.chat_id.0,
                    message_id = sent.id.0,
                    error = %e,
                    "move re-home: failed to attach controls to destination bubble after bind"
                );
            }
        }
        Err(e) => {
            tracing::warn!(
                run_id = %plan.run_id,
                chat = ctx.chat_id.0,
                message_id = sent.id.0,
                error = ?e,
                "move re-home: bind_message failed after successful registry move; destination bubble left without controls"
            );
        }
    }
    deps.channel_map
        .set(ctx.chat_id.0, ctx.raw_thread_id(), plan.session_id.clone())
        .await;

    if let Some(old) = plan.old_bubble.as_ref() {
        let _ = deps
            .bot
            .edit_message_text(
                teloxide::types::ChatId(old.chat_id),
                teloxide::types::MessageId(old.message_id),
                format!(
                    "↪ moved to chat {}{}",
                    plan.new_origin.chat_id,
                    plan.new_origin
                        .thread_id
                        .map(|tid| format!(" / thread {tid}"))
                        .unwrap_or_default()
                ),
            )
            .await;
        let _ = deps
            .bot
            .edit_message_reply_markup(
                teloxide::types::ChatId(old.chat_id),
                teloxide::types::MessageId(old.message_id),
            )
            .await;
    }
    Ok(())
}

pub(crate) async fn explicit_steer_live_run(
    registry: &naked_tg::run_registry::RunRegistry,
    ctx: &ChatCtx,
    raw_tail: &str,
) -> String {
    let mut parts = raw_tail.splitn(2, char::is_whitespace);
    let run_id = parts.next().unwrap_or("").trim();
    let text = parts.next().unwrap_or("").trim();
    if run_id.is_empty() || text.is_empty() {
        return "Usage: /research steer <run_id> <text>".to_string();
    }
    let Some(control) = registry.control_for_run(run_id) else {
        return format!(
            "run not found or expired: <code>{}</code>",
            naked_tg::markup::escape_html(run_id)
        );
    };
    let msg_id = ctx.reply_to.map(|id| id.0).unwrap_or(0);
    match control.steer.try_send(naked_core::types::SteerMessage {
        msg_id,
        text: text.to_string(),
        is_edit: false,
    }) {
        Ok(()) => format!(
            "↩️ accepted steer for <code>{}</code>",
            naked_tg::markup::escape_html(&control.run_id)
        ),
        Err(_) => format!(
            "run <code>{}</code> is busy; try again shortly",
            naked_tg::markup::escape_html(&control.run_id)
        ),
    }
}

pub(crate) fn format_live_run_snapshot(
    registry: &naked_tg::run_registry::RunRegistry,
    run_id: &str,
) -> String {
    let run_id = run_id.trim();
    if run_id.is_empty() {
        return "Usage: /research show <run_id>\n\
                This command shows in-flight live runs only; use /research fresh <id> for stored findings."
            .to_string();
    }

    let Some(run) = registry.get_run(run_id) else {
        let run_id_html = naked_tg::markup::escape_html(run_id);
        return format!(
            "run not found or expired: <code>{run_id_html}</code>\n\
             This shows in-flight runs only; use /research fresh &lt;id&gt; for stored findings."
        );
    };

    let run_id_html = naked_tg::markup::escape_html(&run.run_id);
    let source = run
        .source_ref
        .as_deref()
        .map(naked_tg::markup::escape_html)
        .unwrap_or_else(|| "—".to_string());
    let status = naked_tg::markup::escape_html(&format!("{:?}", run.status));
    let status_line = run
        .rendered
        .status_line
        .as_deref()
        .filter(|line| !line.trim().is_empty())
        .map(|line| {
            format!(
                "\nline: <code>{}</code>",
                naked_tg::markup::escape_html(line)
            )
        })
        .unwrap_or_default();
    let body = run
        .rendered
        .final_html
        .as_deref()
        .filter(|html| !html.trim().is_empty())
        .or_else(|| {
            let html = run.rendered.live_html.trim();
            (!html.is_empty()).then_some(html)
        })
        .unwrap_or("<i>(no cached render yet)</i>");

    format!(
        "📸 <b>Live run snapshot</b>\n\
         run: <code>{run_id_html}</code>\n\
         source: <code>{source}</code>\n\
         status: <code>{status}</code>{status_line}\n\n{body}\n\n\
         <i>Static snapshot only — use /research fresh &lt;id&gt; for stored findings.</i>"
    )
}

async fn format_research_show(agent: &Arc<AgentCore>, tail: &str, fresh_only: bool) -> String {
    let usage = "Usage: /research fresh <id>";
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
    //! T1 (PLAN_RESEARCH_AGENT_FLOW_v1): regression tests for subcommand
    //! aliases. These read the file source itself because handle_research_cmd
    //! needs a real Bot + AgentCore to invoke — too heavy for a parser test.
    //!
    //! The dispatching match-arm string literals are part of the public UX
    //! contract: adding/removing them changes how operators invoke commands.
    //! Source-text assertion is the simplest defence against silent removal.

    use teloxide::Bot;

    fn source() -> &'static str {
        include_str!("research.rs")
    }

    #[test]
    fn list_alias_is_handled_with_ls() {
        // /research list and /research ls must both list specs.
        // Without this match arm `list` falls into the soft-fallback
        // arm which creates a spec named `list-c1e4` (incident 2026-05-16).
        let src = source();
        assert!(
            src.contains("\"ls\" | \"list\" =>"),
            "list alias for ls missing — see B55 in BUG_REGISTRY.md. \
             Required: match arm `\"ls\" | \"list\" =>`"
        );
    }

    #[test]
    fn delete_alias_is_handled_with_rm() {
        let src = source();
        assert!(
            src.contains("\"rm\" | \"delete\" =>"),
            "delete alias for rm missing — symmetry with ls|list alias"
        );
    }

    #[test]
    fn help_documents_list_command() {
        let src = source();
        assert!(
            src.contains("/research list"),
            "help text must mention /research list"
        );
    }

    #[test]
    fn help_documents_rm_command() {
        let src = source();
        assert!(
            src.contains("/research rm"),
            "help text must mention /research rm"
        );
    }

    #[test]
    fn soft_fallback_arm_still_exists() {
        let src = source();
        assert!(
            src.contains("Soft-fallback"),
            "soft-fallback arm comment missing — free-text research path \
             must remain for ergonomic `/research <тема>` invocations"
        );
    }

    // ---- PLAN_UNIFIED_TURN_v1 sentinel tests ----

    #[test]
    fn run_subcommand_uses_scheduler_dispatch() {
        // PLAN_BG_UNIFY_v2: /research run must dispatch through scheduler.
        let src = source();
        let prod = src.split("#[cfg(test)]").next().unwrap_or("");
        assert!(
            prod.contains("dispatch_research("),
            "'run' arm must use dispatch_research helper (DRY)"
        );
        assert!(
            prod.contains("dispatch_immediate("),
            "dispatch_research must call scheduler.dispatch_immediate"
        );
        assert!(
            !prod.contains("BgGuard"),
            "must NOT use BgGuard (replaced by scheduler Inflight)"
        );
    }

    fn registry_with_cached_run() -> naked_tg::run_registry::RunRegistry {
        use naked_core::types::SteerMessage;
        use naked_tg::run_registry::{
            RegisterRunInput, RegisterRunOptions, RenderedRunState, RunKind, RunOrigin, RunRegistry,
        };
        use tokio::sync::mpsc;
        use tokio_util::sync::CancellationToken;

        let registry = RunRegistry::new();
        let (steer, _rx) = mpsc::channel::<SteerMessage>(4);
        registry
            .register_run(
                RegisterRunInput {
                    requested_run_id: Some("run-live-1".to_string()),
                    session_id: "session-live-1".to_string(),
                    origin: RunOrigin::new(6_196_099_449, Some(42)),
                    kind: RunKind::Research {
                        spec_id: "spec-live-1".to_string(),
                    },
                    source_ref: Some("spec-live-1".to_string()),
                    steer,
                    abort: CancellationToken::new(),
                },
                RegisterRunOptions::step2_single_run(),
            )
            .expect("register cached run");
        registry.update_rendered_state(
            "run-live-1",
            RenderedRunState {
                live_html: "<b>cached live tail</b>\nlast tool: research_run".to_string(),
                final_html: None,
                status_line: Some("Streaming #7".to_string()),
            },
        );
        registry
    }

    #[test]
    fn research_runs_lists_max_three_live_runs() {
        use naked_core::types::SteerMessage;
        use naked_tg::run_registry::{
            RegisterRunInput, RegisterRunOptions, RunKind, RunOrigin, RunRegistry,
        };
        use tokio::sync::mpsc;
        use tokio_util::sync::CancellationToken;

        let registry = RunRegistry::new();
        for idx in 0..3 {
            let (steer, _rx) = mpsc::channel::<SteerMessage>(4);
            registry
                .register_run(
                    RegisterRunInput {
                        requested_run_id: Some(format!("run-{idx}")),
                        session_id: format!("sid-{idx}"),
                        origin: RunOrigin::new(777, Some(9)),
                        kind: RunKind::Research {
                            spec_id: format!("spec-{idx}"),
                        },
                        source_ref: Some(format!("spec-{idx}")),
                        steer,
                        abort: CancellationToken::new(),
                    },
                    RegisterRunOptions::cap_three(),
                )
                .unwrap();
        }
        let out = super::format_live_runs_for_thread(&registry, 777, Some(9));
        assert!(out.contains("live runs here"), "got: {out}");
        for idx in 0..3 {
            assert!(out.contains(&format!("run-{idx}")), "got: {out}");
            assert!(out.contains(&format!("spec-{idx}")), "got: {out}");
        }
        assert!(
            out.contains("origin <code>777:Some(9)</code>"),
            "got: {out}"
        );
    }

    #[tokio::test]
    async fn research_steer_explicit_run_id_targets_one() {
        use naked_core::types::SteerMessage;
        use naked_tg::run_registry::{
            RegisterRunInput, RegisterRunOptions, RunKind, RunOrigin, RunRegistry,
        };
        use teloxide::types::{ChatId, MessageId};
        use tokio::sync::mpsc;
        use tokio_util::sync::CancellationToken;

        let registry = RunRegistry::new();
        let (tx_a, mut rx_a) = mpsc::channel::<SteerMessage>(4);
        let (tx_b, mut rx_b) = mpsc::channel::<SteerMessage>(4);
        for (run_id, sid, steer) in [("run-a", "sid-a", tx_a), ("run-b", "sid-b", tx_b)] {
            registry
                .register_run(
                    RegisterRunInput {
                        requested_run_id: Some(run_id.to_string()),
                        session_id: sid.to_string(),
                        origin: RunOrigin::new(777, None),
                        kind: RunKind::ChatTurn,
                        source_ref: None,
                        steer,
                        abort: CancellationToken::new(),
                    },
                    RegisterRunOptions::cap_three(),
                )
                .unwrap();
        }
        let ctx = crate::shared::ChatCtx {
            chat_id: ChatId(777),
            thread_id: None,
            reply_to: Some(MessageId(1234)),
        };
        let out = super::explicit_steer_live_run(&registry, &ctx, "run-b please focus").await;
        assert!(out.contains("run-b"), "got: {out}");
        assert!(rx_a.try_recv().is_err(), "run A must be untouched");
        let msg = rx_b.try_recv().expect("run B receives steer");
        assert_eq!(msg.msg_id, 1234);
        assert_eq!(msg.text, "please focus");
        assert!(!msg.is_edit);
    }

    #[test]
    fn show_snapshot_reads_cached_rendered_state_without_polling_events() {
        let registry = registry_with_cached_run();
        let out = super::format_live_run_snapshot(&registry, "run-live-1");
        assert!(out.contains("📸 <b>Live run snapshot</b>"), "got: {out}");
        assert!(out.contains("<code>run-live-1</code>"), "got: {out}");
        assert!(out.contains("<b>cached live tail</b>"), "got: {out}");
        assert!(out.contains("Streaming #7"), "got: {out}");
        assert!(out.contains("/research fresh &lt;id&gt;"), "got: {out}");

        // Type-level isolation is the real single-consumer proof: the formatter
        // accepts only (&RunRegistry, run_id) and returns a String. There is no
        // AgentHandle/events receiver/stream_response value to poll or clone.
        let prod = source().split("#[cfg(test)]").next().unwrap_or("");
        let formatter = prod
            .split("pub(crate) fn format_live_run_snapshot")
            .nth(1)
            .and_then(|tail| tail.split("async fn format_research_show").next())
            .expect("formatter source section");
        assert!(formatter.contains("registry: &naked_tg::run_registry::RunRegistry"));
        assert!(formatter.contains("run_id: &str"));
        assert!(
            !formatter.contains("AgentHandle"),
            "formatter must not mention AgentHandle"
        );
        assert!(
            !formatter.contains("events"),
            "formatter must not mention events"
        );
        assert!(
            !formatter.contains("stream_response"),
            "formatter must not call stream_response"
        );
    }

    #[test]
    fn show_snapshot_unknown_run_friendly() {
        let registry = naked_tg::run_registry::RunRegistry::new();
        let out = super::format_live_run_snapshot(&registry, "stale-run");
        assert!(out.contains("run not found or expired"), "got: {out}");
        assert!(out.contains("in-flight runs only"), "got: {out}");
        assert!(out.contains("/research fresh &lt;id&gt;"), "got: {out}");
        assert!(
            !out.contains("findings:"),
            "must not fall back to stored report: {out}"
        );
        assert!(
            !out.contains("recent:"),
            "must not fall back to stored report: {out}"
        );
    }

    fn mock_bot(mock_url: &str) -> Bot {
        let url = reqwest::Url::parse(mock_url).unwrap();
        Bot::new("0:TEST_TOKEN").set_api_url(url)
    }

    fn test_research_spec(
        interval_seconds: Option<u64>,
        cron: Option<&str>,
    ) -> naked_core::research::ResearchSpec {
        naked_core::research::ResearchSpec {
            id: "test-spec".into(),
            topic: "test topic".into(),
            interval_seconds,
            cron: cron.map(str::to_owned),
            ..Default::default()
        }
    }

    #[test]
    fn research_list_marks_recurring_not_oneshot() {
        let interval = test_research_spec(Some(21_600), None);
        assert_eq!(super::recurring_marker(&interval), " 🔁");

        let cron = test_research_spec(None, Some("0 * * * *"));
        assert_eq!(super::recurring_marker(&cron), " 🔁");

        let one_shot = test_research_spec(None, None);
        assert_eq!(super::recurring_marker(&one_shot), "");

        assert!(
            source().contains("/research schedule <N> off"),
            "list footer must advertise the recurring off-switch"
        );
    }

    #[tokio::test]
    async fn show_live_run_mockbot_adds_one_mirror_bubble() {
        use teloxide::types::ChatId;
        use wiremock::matchers::method;
        use wiremock::{Mock, MockServer, ResponseTemplate};

        let registry = registry_with_cached_run();
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "ok": true,
                "result": {
                    "message_id": 4242,
                    "date": 0,
                    "chat": {"id": 777, "type": "private", "first_name": "x"},
                    "text": "ack"
                }
            })))
            .expect(1)
            .mount(&server)
            .await;

        let bot = mock_bot(&server.uri());
        let ctx = crate::shared::ChatCtx {
            chat_id: ChatId(777),
            thread_id: None,
            reply_to: None,
        };
        let _ = super::send_live_run_snapshot(&bot, &ctx, &registry, "run-live-1").await;

        let received = server.received_requests().await.unwrap();
        assert_eq!(
            received.len(),
            1,
            "live /research show must send exactly one mirror bubble"
        );
        let body = std::str::from_utf8(&received[0].body).unwrap_or("");
        assert!(body.contains("cached live tail"), "body={body}");
        assert!(
            body.contains("HTML"),
            "mirror must be sent as HTML: body={body}"
        );
        assert!(
            body.contains("s:abort:run-live-1"),
            "mirror controls must target run_id: body={body}"
        );
        assert_eq!(registry.mirror_sinks("run-live-1").len(), 1);
    }
}
