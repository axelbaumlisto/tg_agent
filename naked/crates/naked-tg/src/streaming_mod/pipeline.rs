//! Main streaming pipeline: event loop + `stream_response`.

use super::*;

#[allow(clippy::too_many_arguments)]
pub(crate) async fn stream_response(
    bot: Bot,
    ctx: ChatCtx,
    handle: AgentHandle,
    channel_map: &ChannelSessionMap,
    pending_perms: &PendingPermissions,
    model_tag: String,
    http_client: &reqwest::Client,
    base_url: &str,
    tg_attach_queue: &naked_tg::tg_attach::AttachmentQueue,
    rate_limiter: &naked_tg::rate_limit::RateLimiter,
) {
    let AgentHandle {
        mut events,
        permissions,
        steer,
    } = handle;
    let chat_id_raw = ctx.chat_id.0;
    let tid = ctx.raw_thread_id();
    let chat_key_for_steer = (chat_id_raw, tid);

    // Store steer sender so message_handler can reach us.
    STEER_SENDERS
        .write()
        .await
        .insert(chat_key_for_steer, steer);

    tracing::debug!(
        chat_id = chat_id_raw,
        ?tid,
        thread_id_raw = ?ctx.thread_id,
        "stream_response: sending typing"
    );
    send_typing_raw(http_client, base_url, chat_id_raw, tid).await;

    let placeholder = match bot
        .send_message(ctx.chat_id, "⏳")
        .maybe_thread(ctx.thread_id)
        .maybe_reply_to(ctx.reply_to)
        .await
    {
        Ok(m) => m.id,
        Err(e) => {
            tracing::error!("Failed to send placeholder: {e}");
            return;
        }
    };

    let typing_client = http_client.clone();
    let typing_base = base_url.to_string();
    let typing_cancel = tokio_util::sync::CancellationToken::new();
    let typing_token = typing_cancel.clone();
    tokio::spawn(async move {
        loop {
            tokio::select! {
                _ = typing_token.cancelled() => break,
                _ = tokio::time::sleep(TYPING_INTERVAL) => {
                    send_typing_raw(&typing_client, &typing_base, chat_id_raw, tid).await;
                }
            }
        }
    });

    // Register per-chat model-switch state so `/model` callbacks can
    // request an in-flight switch during this stream.
    let chat_key = (chat_id_raw, tid);
    let model_switch = naked_tg::model_switch::new_shared();
    MODEL_SWITCHES
        .write()
        .await
        .insert(chat_key, model_switch.clone());

    // Register queue counter (reset to 0 for this turn).
    let queue_counter = Arc::new(std::sync::atomic::AtomicUsize::new(0));
    QUEUE_COUNTS
        .write()
        .await
        .insert(chat_key, queue_counter.clone());

    let mut view = CompositeView::new(model_tag, queue_counter);
    let mut dirty = false;
    let mut last_sent = String::new();
    let mut html_broken = false;
    let mut aborted_for_switch = false;
    let mut last_event_at = tokio::time::Instant::now();
    let mut stall_level: u8 = 0; // 0=none, 1=warned 60s, 2=critical 120s

    // Fixed-interval ticker for streaming flushes.
    // The actual rate limiting happens inside RATE_LIMITER.edit() — the
    // ticker just decides when to ATTEMPT a flush.
    let mut flush_interval = tokio::time::interval(std::time::Duration::from_millis(2_400));
    flush_interval.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);
    flush_interval.tick().await; // consume first immediate tick

    let mut got_idle = false;
    loop {
        let event = tokio::select! {
            ev = events.recv() => match ev {
                Some(e) => Some(e),
                None => break, // channel dropped — agent task died
            },
            _ = flush_interval.tick() => None,
        };

        // FIX-3: Detect stalled agent (no events for 90s).
        if event.is_some() {
            last_event_at = tokio::time::Instant::now();
            stall_level = 0;
        } else {
            let elapsed = last_event_at.elapsed().as_secs();
            if stall_level == 0 && elapsed >= 60 {
                stall_level = 1;
                view.tool_lines
                    .push("⚠️ Нет ответа 60с — возможно, зависло".to_string());
                dirty = true;
            } else if stall_level == 1 && elapsed >= 120 {
                stall_level = 2;
                view.tool_lines
                    .push("🔴 Зависло 2 мин — /abort чтобы прервать".to_string());
                dirty = true;
            }
        }

        let has_event = event.is_some();
        if let Some(event) = event {
            match event {
                AgentEvent::ThinkingDelta(t) => {
                    dirty = handlers::handle_thinking_delta(&mut view, &t) != ViewAction::Clean;
                }
                AgentEvent::TextDelta(t) => {
                    dirty = handlers::handle_text_delta(&mut view, &t) != ViewAction::Clean;
                }
                AgentEvent::ToolStart { name, input, .. } => {
                    let _ = handlers::handle_tool_start(&mut view, &name, &input);
                    dirty = true;
                }
                AgentEvent::ToolEnd {
                    name,
                    state,
                    output,
                    ..
                } => {
                    let is_error = matches!(state, naked_core::types::ToolState::Error);
                    let (_, detail_msg) =
                        handlers::handle_tool_end(&mut view, &name, is_error, &output);
                    if let Some(msg) = detail_msg {
                        let _ = bot
                            .send_message(ctx.chat_id, &msg)
                            .maybe_thread(ctx.thread_id)
                            .parse_mode(teloxide::types::ParseMode::Html)
                            .await;
                    }
                    dirty = true;
                }
                AgentEvent::PermissionRequest {
                    call_id,
                    tool_name,
                    input,
                    permission,
                } => {
                    // Flush before showing permission dialog.
                    let _ = flush_live(
                        &bot,
                        ctx.chat_id,
                        placeholder,
                        &view,
                        &mut last_sent,
                        &mut html_broken,
                    )
                    .await;
                    dirty = false;

                    let auto = channel_map
                        .should_auto_approve(chat_id_raw, tid, &tool_name)
                        .await;
                    tracing::info!(
                        chat_id = chat_id_raw, ?tid,
                        tool = %tool_name, auto_approve = auto,
                        "permission check"
                    );
                    if auto {
                        let _ = permissions
                            .send(PermissionResponse {
                                call_id,
                                allowed: true,
                            })
                            .await;
                    } else {
                        let allowed = ask_permission(
                            &bot,
                            ctx,
                            &call_id,
                            &tool_name,
                            &input,
                            &permission,
                            pending_perms,
                        )
                        .await;
                        let _ = permissions
                            .send(PermissionResponse { call_id, allowed })
                            .await;
                    }
                }
                AgentEvent::ContextCompacted {
                    before_msgs,
                    after_msgs,
                    summary_hint,
                    files_count,
                } => {
                    let note = handlers::handle_compaction(
                        before_msgs,
                        after_msgs,
                        files_count,
                        summary_hint.as_deref(),
                    );
                    let _ = bot
                        .send_message(ctx.chat_id, &note)
                        .maybe_thread(ctx.thread_id)
                        .maybe_reply_to(ctx.reply_to)
                        .parse_mode(teloxide::types::ParseMode::Html)
                        .await;
                }
                AgentEvent::CycleRestarted { .. } => {}
                AgentEvent::Heartbeat => {
                    handlers::handle_heartbeat(&mut view);
                    dirty = true;
                }
                AgentEvent::SubAgentProgress {
                    agent_id,
                    event: sa_ev,
                } => {
                    let _ = handlers::handle_sub_agent(&mut view, agent_id, sa_ev);
                    dirty = true;
                }
                AgentEvent::UsageUpdate(u) => {
                    handlers::handle_usage(&mut view, u);
                    dirty = true;
                }
                AgentEvent::ToolOutput { chunk, .. } => {
                    view.tool_output = Some(chunk);
                    dirty = true;
                }
                AgentEvent::SteerReceived { text, msg_ids } => {
                    // S6 of PLAN_NEXT_SESSION: visual proof of
                    // delivery. Render the steer text as a <pre>
                    // (code block) tool-line so the user sees their
                    // exact words echoed back — and DELETE the
                    // matching "↩️ Принято" temp confirmations
                    // (the ones whose user-msg-ids appear in
                    // `msg_ids`).
                    view.tool_lines.push(format!(
                        "✅ <b>Доставлено</b>\n<pre>{}</pre>",
                        crate::fmt_utils::escape_html_min(&text)
                    ));
                    dirty = true;

                    // Drain ack ids matching the delivered steer
                    // msg_ids and best-effort delete them. We don't
                    // propagate delete errors — a stale ack is a
                    // cosmetic problem, never worth crashing the
                    // streaming pipeline over.
                    let to_delete: Vec<(teloxide::types::ChatId, teloxide::types::MessageId)> = {
                        let mut acks = crate::shared::STEER_ACK_IDS.write().await;
                        msg_ids
                            .iter()
                            .filter_map(|mid| acks.remove(&(chat_id_raw, tid, *mid)))
                            .collect()
                    };
                    for (chat, ack_id) in to_delete {
                        if let Err(e) = bot.delete_message(chat, ack_id).await {
                            tracing::debug!(
                                chat = chat.0,
                                msg = ack_id.0,
                                "steer ack delete failed (likely already gone): {e}"
                            );
                        }
                    }
                }
                AgentEvent::Error(e) => {
                    // "cancelled" = normal turn displacement, not a crash.
                    if e.contains("cancelled") || e.contains("Cancelled") {
                        got_idle = true;
                    }
                    let _ = handlers::handle_error(&mut view, &e);
                    dirty = true;
                }
                AgentEvent::Idle => {
                    got_idle = true;
                    break;
                }
            }
        } // end if let Some(event)

        // Check for in-flight model switch at every yield point.
        if naked_tg::model_switch::check_and_take(&model_switch)
            .await
            .is_some()
        {
            aborted_for_switch = true;
            break;
        }

        // Proactive flush: only on tick interval, never faster.
        // The interval ticker fires in select! above → event=None.
        // On event: just set dirty. On tick: flush if dirty.
        let is_tick = !has_event; // tick = no event received, interval fired
        if is_tick {
            // Tick fired — time to flush.
            view.tick += 1;
            if dirty {
                flush_live(
                    &bot,
                    ctx.chat_id,
                    placeholder,
                    &view,
                    &mut last_sent,
                    &mut html_broken,
                )
                .await;
                dirty = false;

                // If the rate limiter has this chat blocked (429),
                // stretch the ticker to avoid hammering.
                let backoff = rate_limiter.streaming_interval(ctx.chat_id.0).await;
                if backoff > flush_interval.period() {
                    flush_interval = tokio::time::interval(backoff);
                    flush_interval.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);
                    flush_interval.tick().await;
                }
            }
        }
    }

    // If the agent task died without sending Idle (panic/crash),
    // notify the user so they know something broke.
    if !got_idle {
        tracing::error!(
            chat = ctx.chat_id.0,
            "agent task died without Idle — likely panicked"
        );
        let _ = bot
            .send_message(
                ctx.chat_id,
                "🔴 Внутренняя ошибка — задача аварийно завершилась. Попробуйте ещё раз.",
            )
            .maybe_thread(ctx.thread_id)
            .await;
    }

    typing_cancel.cancel();

    // Deregister model-switch state, queue counter, and steer sender.
    MODEL_SWITCHES.write().await.remove(&chat_key);
    QUEUE_COUNTS.write().await.remove(&chat_key);
    STEER_SENDERS.write().await.remove(&chat_key_for_steer);
    // S6 cleanup: any ack ids still parked for this chat/thread are
    // unreachable now (turn ended without an Idle-time SteerReceived
    // for them). Best-effort delete — prevents the temp
    // "Принято" message from sticking around forever.
    {
        let stale: Vec<(teloxide::types::ChatId, teloxide::types::MessageId)> = {
            let mut acks = crate::shared::STEER_ACK_IDS.write().await;
            let keys: Vec<_> = acks
                .keys()
                .filter(|(c, t, _)| *c == chat_id_raw && *t == tid)
                .cloned()
                .collect();
            keys.into_iter().filter_map(|k| acks.remove(&k)).collect()
        };
        for (chat, ack_id) in stale {
            if let Err(e) = bot.delete_message(chat, ack_id).await {
                tracing::debug!(
                    chat = chat.0,
                    msg = ack_id.0,
                    "end-of-turn steer ack cleanup failed: {e}"
                );
            }
        }
    }

    if aborted_for_switch {
        // Don't send final — the turn was interrupted.
        // Edit placeholder to indicate switch in progress.
        RATE_LIMITER
            .edit_plain(&bot, ctx.chat_id, placeholder, "⚡ Switching model…")
            .await;
        return;
    }

    let final_html = view.render_final();
    send_final(bot.clone(), ctx, placeholder, &final_html, &view).await;

    // ── A3: Send error card with retry button if provider failed ────
    if view.had_provider_error {
        let keyboard = InlineKeyboardMarkup::new(vec![vec![
            InlineKeyboardButton::callback("🔄 Retry", "err:retry".to_string()),
            InlineKeyboardButton::callback("🔀 Switch model", "err:switch".to_string()),
        ]]);
        let _ = bot
            .send_message(
                ctx.chat_id,
                "⚠️ Ответ содержит ошибку провайдера. Повторить?",
            )
            .maybe_thread(ctx.thread_id)
            .reply_markup(keyboard)
            .await;
    }

    // Deliver any files queued by telegram_attach tool.
    let attachments: Vec<naked_tg::tg_attach::StagedAttachment> =
        tg_attach_queue.lock().await.drain(..).collect();
    for att in attachments {
        let method = if naked_tg::tg_attach::is_image_path(&att.path) {
            "sendPhoto"
        } else {
            "sendDocument"
        };
        let field = if method == "sendPhoto" {
            "photo"
        } else {
            "document"
        };
        let mut form = reqwest::multipart::Form::new().text("chat_id", ctx.chat_id.0.to_string());
        if let Some(tid) = ctx.thread_id {
            form = form.text("message_thread_id", tid.0.0.to_string());
        }
        let form = form.file(field, &att.path).await;
        match form {
            Ok(form) => {
                let url = format!("{base_url}/{method}");
                match http_client.post(&url).multipart(form).send().await {
                    Ok(resp) if resp.status().is_success() => {
                        tracing::info!(file = %att.file_name, "telegram_attach delivered");
                    }
                    Ok(resp) => {
                        tracing::warn!(
                            file = %att.file_name,
                            status = %resp.status(),
                            "telegram_attach delivery failed"
                        );
                    }
                    Err(e) => {
                        tracing::warn!(file = %att.file_name, error = %e, "telegram_attach send error");
                    }
                }
            }
            Err(e) => {
                tracing::warn!(file = %att.file_name, error = %e, "telegram_attach form build failed");
            }
        }
    }
}
