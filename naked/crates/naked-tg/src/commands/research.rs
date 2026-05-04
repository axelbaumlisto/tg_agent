//! Research command handlers + research UI helpers.
//!
//! Extracted from commands.rs.

use super::super::fmt_utils::{escape_html_min, format_age, format_interval, safe_slug};
use super::super::*;

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
