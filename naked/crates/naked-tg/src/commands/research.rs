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
/research show <N|id>           spec + находки
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
                    out.push_str(&format!(
                        "{status} {n}. {short_topic}{ellip} ({total}) · {last_run}\n"
                    ));
                    ids.push(s.id.clone());
                }
                out.push_str("\n`/research show|run|pause|rm <N>`");
                // Store index for numeric refs.
                LIST_INDEX.write().await.insert(ctx.chat_id.0, ids);
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
                        let patch = naked_core::ResearchPatch {
                            interval_seconds: naked_core::PatchField::Clear,
                            ..Default::default()
                        };
                        match agent.update_research(id, patch).await {
                            Ok(_) => format!("⏹ schedule cleared for `{id}`"),
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
    let patch = naked_core::ResearchPatch {
        interval_seconds: naked_core::PatchField::Set(s),
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
    //! T1 (PLAN_RESEARCH_AGENT_FLOW_v1): regression tests for subcommand
    //! aliases. These read the file source itself because handle_research_cmd
    //! needs a real Bot + AgentCore to invoke — too heavy for a parser test.
    //!
    //! The dispatching match-arm string literals are part of the public UX
    //! contract: adding/removing them changes how operators invoke commands.
    //! Source-text assertion is the simplest defence against silent removal.

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
}
