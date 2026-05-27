use super::*;

// ── A3: Streaming control card buttons ────────────────────────────────────
//
// Two semantically distinct actions:
//
// * stream:abort  — hard stop. Cancels the active turn's CancellationToken.
//   The run-loop's drain-on-error path preserves pending steers / queued
//   input for the next turn (via the SteerPipeline rescue). Use when the
//   user wants to throw away current work entirely.
//
// * stream:sendnow — soft nudge via the steer pipeline. Injects a synthetic
//   SteerMessage into the running turn telling the model to stop tool-calling
//   and reply with what it has. The S2/S3 mid-stream interrupt then re-issues
//   the iteration with the nudge in history; the model produces its
//   best-available answer immediately. NO abort — the session keeps streaming.
//   Falls back to abort if the steer channel is unavailable.

pub(super) async fn handle_stream_action(
    bot: &Bot,
    agent: &Arc<AgentCore>,
    channel_map: &Arc<ChannelSessionMap>,
    q: &CallbackQuery,
    action: &str,
) -> Result<(), teloxide::RequestError> {
    // F3: bump per-action click counter for /metrics. Done before the
    // action so the counter advances even if there's no active session.
    crate::metrics::record_stream_button_click(action);
    let cb_ctx = ChatCtx::from_callback(q);
    let cid = cb_ctx.chat_id.0;
    let tid = cb_ctx.raw_thread_id();

    let toast: &str = match action {
        "abort" => {
            let aborted = crate::session_control::abort_mapped_session(
                agent,
                channel_map,
                cb_ctx.chat_id,
                cb_ctx.thread_id,
            )
            .await
            .aborted();
            // M4/B02: clear kbd from placeholder, don't delete the
            // message (it now holds streamed assistant content).
            crate::session_control::clear_control_card(bot, cid, tid).await;
            if aborted {
                "⏹ Остановлено"
            } else {
                "⏹ Нет активной сессии"
            }
        }
        "sendnow" => handle_send_now(bot, agent, channel_map, cb_ctx).await,
        _ => "…",
    };
    bot.answer_callback_query(q.id.clone()).text(toast).await?;
    Ok(())
}

async fn handle_send_now(
    bot: &Bot,
    agent: &Arc<AgentCore>,
    channel_map: &Arc<ChannelSessionMap>,
    cb_ctx: ChatCtx,
) -> &'static str {
    let cid = cb_ctx.chat_id.0;
    let tid = cb_ctx.raw_thread_id();
    // Try to inject the synthetic steer first. If the steer channel
    // exists, leave the session running (the model will pick up the
    // nudge via S2/S3 and wrap up). The control card stays — the next
    // assistant message-end tears it down via streaming cleanup.
    let key = (cid, tid);
    let nudged = {
        let map = crate::shared::STEER_SENDERS.read().await;
        if let Some(steer_tx) = map.get(&key) {
            steer_tx
                .try_send(naked_core::types::SteerMessage {
                    // Synthetic — no Telegram message is associated,
                    // so use a sentinel negative id to avoid colliding
                    // with real msg_id values (Telegram ids are positive).
                    msg_id: -1,
                    text: super::SEND_NOW_NUDGE_TEXT.into(),
                    is_edit: false,
                })
                .is_ok()
        } else {
            false
        }
    };
    if nudged {
        "⏩ Нудж отправлен — модель завершит с тем, что есть"
    } else {
        // Fallback: no live steer channel → abort so the user gets an
        // observable effect rather than a silent no-op.
        let aborted = crate::session_control::abort_mapped_session(
            agent,
            channel_map,
            cb_ctx.chat_id,
            cb_ctx.thread_id,
        )
        .await
        .aborted();
        // M4/B02: clear kbd, don't delete content.
        crate::session_control::clear_control_card(bot, cid, tid).await;
        if aborted {
            "⏩ Нудж не прошёл — остановил. Отправь сообщение, весь контекст сохранён"
        } else {
            "⏩ Нет активной сессии"
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::callbacks::CallbackAction;

    #[test]
    fn stream_callbacks_route_through_typed_action() {
        assert_eq!(
            CallbackAction::parse("stream:abort"),
            CallbackAction::Stream { action: "abort" }
        );
        assert_eq!(
            CallbackAction::parse("stream:sendnow"),
            CallbackAction::Stream { action: "sendnow" }
        );
    }
}
