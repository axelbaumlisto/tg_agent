use super::*;
use crate::fmt_utils::escape_html_min;

pub(super) async fn handle_research_callback(
    deps: &crate::message_handler::BotDeps,
    q: &CallbackQuery,
    action: &str,
    spec_id: &str,
) -> Result<(), teloxide::RequestError> {
    let bot = deps.bot.clone();
    let agent = deps.agent.clone();
    let channel_map = deps.channel_map.clone();
    match action {
        "stop" => handle_research_stop(&bot, &agent, &channel_map, q, spec_id).await,
        "restart" => handle_research_restart(deps, &bot, &agent, &channel_map, q, spec_id).await,
        "sch" => handle_research_schedule_opt_in(&bot, &agent, q, spec_id).await,
        "uns" => handle_research_unschedule(&bot, &agent, q, spec_id).await,
        "rm" => handle_research_delete_request(&bot, &agent, q, spec_id).await,
        "rmc" => handle_research_delete_confirm(&bot, &agent, q, spec_id).await,
        _ => {
            bot.answer_callback_query(q.id.clone())
                .text(format!("unknown research action: {action}"))
                .await?;
            Ok(())
        }
    }
}

#[derive(Clone, Copy)]
struct ResearchCallbackContext {
    chat_id: ChatId,
    thread_id: Option<ThreadId>,
    message_id: MessageId,
}

fn owner_gate_ok(
    spec_chat: Option<i64>,
    spec_thread: Option<i32>,
    cb_chat: i64,
    cb_thread: Option<i32>,
) -> bool {
    // Legacy/unbound specs have no owner chat. The global allowed_chat_ids gate
    // already ran in callbacks/mod.rs, so allow these old specs through.
    if spec_chat.is_some_and(|owner| owner != cb_chat) {
        return false;
    }
    if spec_thread.is_some_and(|owner_thread| Some(owner_thread) != cb_thread) {
        return false;
    }
    true
}

fn raw_thread_id(thread_id: Option<ThreadId>) -> Option<i32> {
    thread_id.map(|tid| tid.0.0)
}

fn research_callback_context(q: &CallbackQuery) -> Option<ResearchCallbackContext> {
    let msg = q.message.as_ref()?;
    Some(ResearchCallbackContext {
        chat_id: msg.chat().id,
        thread_id: msg.regular_message().and_then(|m| m.thread_id),
        message_id: msg.id(),
    })
}

async fn load_authorized_research_callback(
    bot: &Bot,
    agent: &Arc<AgentCore>,
    q: &CallbackQuery,
    spec_id: &str,
) -> Result<Option<ResearchCallbackContext>, teloxide::RequestError> {
    let Some(ctx) = research_callback_context(q) else {
        bot.answer_callback_query(q.id.clone())
            .text("missing chat context")
            .await?;
        return Ok(None);
    };

    let spec = match agent.load_research(spec_id).await {
        Ok(spec) => spec,
        Err(_) => {
            bot.answer_callback_query(q.id.clone())
                .text("уже удалено или истекло")
                .await?;
            let _ = bot
                .edit_message_reply_markup(ctx.chat_id, ctx.message_id)
                .await;
            return Ok(None);
        }
    };

    if !owner_gate_ok(
        spec.chat_id,
        spec.thread_id,
        ctx.chat_id.0,
        raw_thread_id(ctx.thread_id),
    ) {
        bot.answer_callback_query(q.id.clone())
            .text("это не ваш ресёрч")
            .await?;
        return Ok(None);
    }

    Ok(Some(ctx))
}

async fn handle_research_schedule_opt_in(
    bot: &Bot,
    agent: &Arc<AgentCore>,
    q: &CallbackQuery,
    spec_id: &str,
) -> Result<(), teloxide::RequestError> {
    let Some(ctx) = load_authorized_research_callback(bot, agent, q, spec_id).await? else {
        return Ok(());
    };
    let configured = agent.config().research.default_interval_seconds;
    let interval_seconds = if configured == 0 { 21_600 } else { configured };

    match agent
        .set_research_schedule(
            spec_id,
            naked_core::ScheduleUpdate::Interval(interval_seconds),
        )
        .await
    {
        Ok(()) => {
            bot.answer_callback_query(q.id.clone())
                .text("⏰ будет повторяться")
                .await?;
            let _ = bot
                .edit_message_reply_markup(ctx.chat_id, ctx.message_id)
                .reply_markup(naked_tg::research_controls::keyboard_scheduled(spec_id))
                .await;
        }
        Err(e) => {
            bot.answer_callback_query(q.id.clone())
                .text(format!("error: {e}"))
                .await?;
        }
    }
    Ok(())
}

async fn handle_research_unschedule(
    bot: &Bot,
    agent: &Arc<AgentCore>,
    q: &CallbackQuery,
    spec_id: &str,
) -> Result<(), teloxide::RequestError> {
    let Some(ctx) = load_authorized_research_callback(bot, agent, q, spec_id).await? else {
        return Ok(());
    };

    match agent
        .set_research_schedule(spec_id, naked_core::ScheduleUpdate::Off)
        .await
    {
        Ok(()) => {
            bot.answer_callback_query(q.id.clone())
                .text("⏹ расписание отменено, оставлено в списке")
                .await?;
            let _ = bot
                .edit_message_reply_markup(ctx.chat_id, ctx.message_id)
                .await;
        }
        Err(e) => {
            bot.answer_callback_query(q.id.clone())
                .text(format!("error: {e}"))
                .await?;
        }
    }
    Ok(())
}

async fn handle_research_delete_request(
    bot: &Bot,
    agent: &Arc<AgentCore>,
    q: &CallbackQuery,
    spec_id: &str,
) -> Result<(), teloxide::RequestError> {
    let Some(ctx) = load_authorized_research_callback(bot, agent, q, spec_id).await? else {
        return Ok(());
    };

    bot.answer_callback_query(q.id.clone())
        .text("подтвердите удаление")
        .await?;
    let _ = bot
        .edit_message_reply_markup(ctx.chat_id, ctx.message_id)
        .reply_markup(naked_tg::research_controls::keyboard_delete_confirm(
            spec_id,
        ))
        .await;
    Ok(())
}

async fn handle_research_delete_confirm(
    bot: &Bot,
    agent: &Arc<AgentCore>,
    q: &CallbackQuery,
    spec_id: &str,
) -> Result<(), teloxide::RequestError> {
    let Some(ctx) = load_authorized_research_callback(bot, agent, q, spec_id).await? else {
        return Ok(());
    };

    if crate::shared::RUN_REGISTRY.has_active_run_for_source_ref(spec_id) {
        bot.answer_callback_query(q.id.clone())
            .text("прогон активен, сначала остановите")
            .await?;
        return Ok(());
    }

    match agent.delete_research(spec_id).await {
        Ok(()) => {
            bot.answer_callback_query(q.id.clone())
                .text("🗑 удалено")
                .await?;
            let _ = bot
                .edit_message_reply_markup(ctx.chat_id, ctx.message_id)
                .await;
        }
        Err(e) => {
            bot.answer_callback_query(q.id.clone())
                .text(format!("error: {e}"))
                .await?;
        }
    }
    Ok(())
}

async fn handle_research_stop(
    bot: &Bot,
    agent: &Arc<AgentCore>,
    channel_map: &Arc<ChannelSessionMap>,
    q: &CallbackQuery,
    spec_id: &str,
) -> Result<(), teloxide::RequestError> {
    // B4 (PLAN_RESEARCH_FLOW_CLOSURE_v1): unified abort.
    // All research runs now create proper sessions in ChannelSessionMap,
    // so agent.abort(session_id) is the only cancel mechanism. The
    // legacy cancel_research_run fallback has been removed (INV-CANCEL-1).
    let cancel_started = std::time::Instant::now();
    let aborted = if let Some(msg) = &q.message
        && let Some(regular) = msg.regular_message()
    {
        let outcome = crate::session_control::abort_mapped_session(
            agent,
            channel_map,
            regular.chat.id,
            regular.thread_id,
        )
        .await;
        match &outcome {
            crate::session_control::AbortMappedSession::Aborted { session_id } => tracing::debug!(
                spec_id = %spec_id,
                %session_id,
                "stop callback: aborting via agent.abort(session_id)"
            ),
            crate::session_control::AbortMappedSession::NoMappedSession => tracing::warn!(
                spec_id = %spec_id,
                "stop callback: no session found for spec — run may have already finished"
            ),
        }
        outcome.aborted()
    } else {
        tracing::warn!(
            spec_id = %spec_id,
            "stop callback: no message context to resolve session"
        );
        false
    };
    let elapsed_ms = cancel_started.elapsed().as_millis() as u64;
    crate::metrics::record_research_cancel_propagation(elapsed_ms);

    if let Some(msg) = &q.message
        && let Some(regular) = msg.regular_message()
    {
        let note = research_stop_note(spec_id, aborted);
        let _ = bot
            .edit_message_text(regular.chat.id, regular.id, note)
            .parse_mode(ParseMode::Html)
            .await;
    }
    bot.answer_callback_query(q.id.clone())
        .text(if aborted { "Aborted" } else { "Already done" })
        .await?;
    Ok(())
}

async fn handle_research_restart(
    deps: &crate::message_handler::BotDeps,
    bot: &Bot,
    agent: &Arc<AgentCore>,
    channel_map: &Arc<ChannelSessionMap>,
    q: &CallbackQuery,
    spec_id: &str,
) -> Result<(), teloxide::RequestError> {
    let (chat_id, thread_id) = match &q.message {
        Some(msg) => {
            let cid = msg.chat().id;
            let tid = msg.regular_message().and_then(|m| m.thread_id);
            (cid, tid)
        }
        None => {
            bot.answer_callback_query(q.id.clone())
                .text("missing chat context")
                .await?;
            return Ok(());
        }
    };
    // T3.3 (PLAN_RESEARCH_AGENT_FLOW_v1): PENDING_CLARIFICATIONS
    // removed — clarification = next normal user message in thread (no
    // special UI). Re-run is just `/research run X`.
    bot.answer_callback_query(q.id.clone())
        .text("Restarting…")
        .await?;
    // PLAN_UNIFIED_TURN_v1 T4: restart goes through normal send_prompt →
    // stream_response so operator sees progress.
    let cb_ctx = ChatCtx {
        chat_id,
        thread_id,
        reply_to: None,
    };
    let tid_raw = thread_id.map(|teloxide::types::ThreadId(mid)| mid.0);
    let session_id = if let Some(sid) = channel_map.get(chat_id.0, tid_raw).await {
        sid
    } else {
        let ws = std::path::PathBuf::from(
            std::env::var("NAKED_WORKSPACE").unwrap_or_else(|_| ".".into()),
        );
        let sid = agent.create_session_with_channel(&ws, "telegram").await;
        channel_map.set(chat_id.0, tid_raw, sid.clone()).await;
        sid
    };
    let prompt = format!("/research run {spec_id}");
    match agent.send_prompt(&session_id, &prompt).await {
        Ok(handle) => {
            let pm = agent.session_provider_model(&session_id).await;
            let deps_clone = deps.clone();
            let spec_id_owned = spec_id.to_string();
            tokio::spawn(async move {
                let run_ctx = crate::streaming::StreamRunContext {
                    requested_run_id: None,
                    session_id: session_id.clone(),
                    kind: naked_tg::run_registry::RunKind::Research {
                        spec_id: spec_id_owned.clone(),
                    },
                    source_ref: Some(spec_id_owned.clone()),
                    final_reply_markup: None,
                };
                let _ = crate::streaming::stream_response(
                    &deps_clone,
                    cb_ctx,
                    handle,
                    format!("{}/{}", pm.0, pm.1),
                    run_ctx,
                )
                .await;
            });
        }
        Err(e) => {
            let _ =
                crate::shared::safe_send(bot, &cb_ctx, format!("restart error: {e}"), None).await;
        }
    }
    Ok(())
}

fn research_stop_note(spec_id: &str, aborted: bool) -> String {
    if aborted {
        format!(
            "⏸ Aborted <code>{spec}</code>.\n\
             Reply with a clarification or relaunch with \
             <code>/research run {spec}</code>.",
            spec = escape_html_min(spec_id)
        )
    } else {
        format!(
            "ℹ️ Run <code>{}</code> already finished.\n\
             Reply with a note or relaunch with \
             <code>/research run {}</code>.",
            escape_html_min(spec_id),
            escape_html_min(spec_id)
        )
    }
}

#[cfg(test)]
mod tests {
    use super::{owner_gate_ok, research_stop_note};
    use crate::callbacks::CallbackAction;

    fn source() -> &'static str {
        include_str!("research.rs")
    }

    fn prod_source() -> &'static str {
        source().split("#[cfg(test)]").next().unwrap_or("")
    }

    fn function_body<'a>(src: &'a str, name: &str, next_name: &str) -> &'a str {
        src.split(name)
            .nth(1)
            .unwrap_or_else(|| panic!("missing function {name}"))
            .split(next_name)
            .next()
            .unwrap_or_else(|| panic!("missing next function {next_name}"))
    }

    #[test]
    fn research_callbacks_route_through_typed_action() {
        assert_eq!(
            CallbackAction::parse("r:stop:da-nang"),
            CallbackAction::Research {
                action: "stop",
                spec_id: "da-nang",
            }
        );
        assert_eq!(
            CallbackAction::parse("r:restart:da-nang"),
            CallbackAction::Research {
                action: "restart",
                spec_id: "da-nang",
            }
        );
    }

    #[test]
    fn stop_callback_has_no_legacy_cancel_fallback() {
        let prod = prod_source();
        assert!(
            !prod.contains("cancel_research_run("),
            "stop callback must NOT call cancel_research_run — agent.abort is the only path"
        );
    }

    #[test]
    fn d8_owner_gate_allows_owner_denies_non_owner_allows_legacy_unbound() {
        assert!(owner_gate_ok(Some(111), None, 111, None));
        assert!(!owner_gate_ok(Some(111), None, 999, None));
        assert!(owner_gate_ok(None, None, 999, None));
        assert!(owner_gate_ok(Some(111), Some(7), 111, Some(7)));
        assert!(!owner_gate_ok(Some(111), Some(7), 111, Some(8)));
    }

    #[test]
    fn d9_research_callbacks_load_and_handle_missing_before_mutating() {
        let prod = prod_source();
        let load = prod
            .find("agent.load_research(spec_id).await")
            .expect("callback helper must load spec");
        let missing = prod[load..]
            .find("Err(_) =>")
            .expect("missing spec must be handled gracefully")
            + load;
        let first_set = prod
            .find(".set_research_schedule(")
            .expect("schedule callback must mutate through set_research_schedule");
        let first_delete = prod
            .find("agent.delete_research(spec_id)")
            .expect("delete confirm callback must call delete_research");

        assert!(
            load < first_set,
            "spec must be loaded before schedule mutation"
        );
        assert!(
            load < first_delete,
            "spec must be loaded before delete mutation"
        );
        assert!(
            missing < first_set && missing < first_delete,
            "missing-spec branch must appear before any mutation"
        );
    }

    #[test]
    fn d8_delete_request_is_two_tap_only_confirm_deletes() {
        let prod = prod_source();
        let request = function_body(
            prod,
            "fn handle_research_delete_request",
            "fn handle_research_delete_confirm",
        );
        assert!(request.contains("keyboard_delete_confirm("));
        assert!(
            !request.contains("delete_research"),
            "r:rm step 1 must never delete"
        );

        let confirm = function_body(
            prod,
            "fn handle_research_delete_confirm",
            "fn handle_research_stop",
        );
        assert!(
            confirm.contains("agent.delete_research(spec_id).await"),
            "only r:rmc confirmation deletes"
        );
    }

    #[test]
    fn d10_delete_confirm_refuses_inflight_before_delete() {
        let prod = prod_source();
        let confirm = function_body(
            prod,
            "fn handle_research_delete_confirm",
            "fn handle_research_stop",
        );
        let active_check = confirm
            .find("has_active_run_for_source_ref(spec_id)")
            .expect("confirmed delete must check active source_ref run");
        let delete = confirm
            .find("agent.delete_research(spec_id).await")
            .expect("confirmed delete must call delete_research");
        assert!(
            active_check < delete,
            "active run check must precede delete"
        );
        assert!(confirm.contains("прогон активен, сначала остановите"));
    }

    #[test]
    fn stop_callback_uses_aborted_not_paused_wording() {
        let note = research_stop_note("da<nang>", true);
        assert!(note.contains("Aborted <code>"));
        assert!(note.contains("da&lt;nang&gt;"));
        assert!(!note.contains("Paused"));
    }

    #[test]
    fn stop_callback_finished_note_is_escaped() {
        let note = research_stop_note("x&y", false);
        assert!(note.contains("Run <code>x&amp;y</code> already finished"));
        assert!(note.contains("/research run x&amp;y"));
    }
}
