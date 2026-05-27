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
        _ => {
            bot.answer_callback_query(q.id.clone())
                .text(format!("unknown research action: {action}"))
                .await?;
            Ok(())
        }
    }
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
            tokio::spawn(async move {
                crate::streaming::stream_response(
                    &deps_clone,
                    cb_ctx,
                    handle,
                    format!("{}/{}", pm.0, pm.1),
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
    use super::research_stop_note;
    use crate::callbacks::CallbackAction;

    fn source() -> &'static str {
        include_str!("research.rs")
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
        let src = source();
        let prod = src.split("#[cfg(test)]").next().unwrap_or("");
        assert!(
            !prod.contains("cancel_research_run("),
            "stop callback must NOT call cancel_research_run — agent.abort is the only path"
        );
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
