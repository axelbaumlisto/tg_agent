//! Main streaming pipeline: event loop + `stream_response`.

use super::*;

/// PLAN_MEDIA_UX_v1 M4: single source of truth for the streaming
/// control keyboard. Used by both `stream_response` (initial
/// placeholder send) and may be re-attached by future callback
/// paths if needed. Kept here so the literal lives in ONE place
/// (DRY).
/// PLAN_MEDIA_UX_v1 M4 / BUG_REGISTRY D-INV-STREAM-BUBBLE-COUNT.
///
/// Sends the SINGLE stream-start message: a placeholder text ("⏳")
/// with the inline-keyboard control card attached. Returns the
/// message id on success, None on error (already logged).
///
/// Extracted from `stream_response` so wiremock tests can assert it
/// calls `bot.send_message` exactly ONCE — the regression guard for
/// B02 (3-bubble stream-start that motivated M4).
pub(crate) async fn send_stream_placeholder(
    bot: &Bot,
    ctx: &ChatCtx,
    run_id: &str,
) -> Option<teloxide::types::MessageId> {
    // Retry with exponential backoff: 2s, 5s, 10s.
    // Network blips to Telegram DC are transient (10-30s); without
    // retry the entire agent turn result is silently dropped.
    const DELAYS: [u64; 3] = [2, 5, 10];
    let mut last_err;
    // First attempt (immediate)
    match bot
        .send_message(ctx.chat_id, "⏳ thinking…")
        .maybe_thread(ctx.thread_id)
        .maybe_reply_to(ctx.reply_to)
        .reply_markup(streaming_control_kb_for_run(run_id))
        .await
    {
        Ok(m) => return Some(m.id),
        Err(e) => {
            last_err = format!("{e}");
            tracing::warn!("placeholder send failed, will retry: {e}");
        }
    }
    // Retries
    for delay in DELAYS {
        tokio::time::sleep(std::time::Duration::from_secs(delay)).await;
        match bot
            .send_message(ctx.chat_id, "⏳ thinking…")
            .maybe_thread(ctx.thread_id)
            .maybe_reply_to(ctx.reply_to)
            .reply_markup(streaming_control_kb_for_run(run_id))
            .await
        {
            Ok(m) => {
                tracing::info!("placeholder sent after {delay}s retry");
                return Some(m.id);
            }
            Err(e) => {
                last_err = format!("{e}");
                tracing::warn!("placeholder retry after {delay}s failed: {e}");
            }
        }
    }
    tracing::error!("Failed to send placeholder after all retries: {last_err}");
    None
}

pub(crate) fn streaming_control_kb_for_run(run_id: &str) -> teloxide::types::InlineKeyboardMarkup {
    teloxide::types::InlineKeyboardMarkup::new(vec![vec![
        teloxide::types::InlineKeyboardButton::callback("⏹ Stop", format!("s:abort:{run_id}")),
        teloxide::types::InlineKeyboardButton::callback("⏩ Send", format!("s:sendnow:{run_id}")),
    ]])
}

pub(crate) async fn register_turn_routing(_ctx: ChatCtx, _handle: &AgentHandle) {
    // Step 4: run routing is registered by run_id inside stream_response.
}

#[derive(Debug, Clone)]
pub(crate) struct StreamRunContext {
    /// Optional pre-existing external id. Research maps Inflight.attempt_id here;
    /// chat turns leave it empty and let RunRegistry generate a kind-agnostic id.
    pub(crate) requested_run_id: Option<naked_tg::run_registry::RunId>,
    pub(crate) session_id: naked_tg::run_registry::SessionId,
    pub(crate) kind: naked_tg::run_registry::RunKind,
    pub(crate) source_ref: Option<naked_tg::run_registry::SourceRef>,
}

impl StreamRunContext {
    pub(crate) fn chat_turn(session_id: String) -> Self {
        Self {
            requested_run_id: None,
            session_id,
            kind: naked_tg::run_registry::RunKind::ChatTurn,
            source_ref: None,
        }
    }
}

#[derive(Debug, Clone, Copy, Default)]
pub(crate) struct StreamResponseOutcome {
    /// True if the turn reached `AgentEvent::Idle` (clean completion/flush).
    pub(crate) got_idle: bool,
    /// True if the turn emitted B73's wall-clock timeout error before Idle.
    pub(crate) wall_timeout: bool,
}

/// T3 (PLAN_v13_SOLID_AUDIT): takes `&BotDeps` for shared infra
/// instead of 7 individual args.  Turn-specific params remain separate.
/// Stream an agent turn's events to Telegram as live message edits.
///
/// Returns an outcome that distinguishes clean Idle from B73's timeout+Idle
/// cleanup path; callers that do not need the distinction can ignore it.
pub(crate) async fn stream_response(
    deps: &crate::message_handler::BotDeps,
    ctx: ChatCtx,
    handle: AgentHandle,
    model_tag: String,
    run_ctx: StreamRunContext,
) -> StreamResponseOutcome {
    let bot = deps.bot.clone();
    let channel_map = &deps.channel_map;
    let pending_perms = &deps.pending_perms;
    let http_client = &deps.http_client;
    let base_url: &str = &deps.base_url;
    let tg_attach_queue = &deps.tg_attach_queue;
    let rate_limiter = &deps.rate_limiter;
    let AgentHandle {
        mut events,
        permissions,
        steer,
        abort,
    } = handle;
    let chat_id_raw = ctx.chat_id.0;
    let tid = ctx.raw_thread_id();
    let _chat_key_for_steer = (chat_id_raw, tid);
    let abort_for_reject = abort.clone();
    let session_id_for_attach = run_ctx.session_id.clone();
    let run_summary = match crate::shared::RUN_REGISTRY.register_run(
        naked_tg::run_registry::RegisterRunInput {
            requested_run_id: run_ctx.requested_run_id,
            session_id: run_ctx.session_id,
            origin: naked_tg::run_registry::RunOrigin::new(chat_id_raw, tid),
            kind: run_ctx.kind,
            source_ref: run_ctx.source_ref,
            steer: steer.clone(),
            abort,
        },
        {
            if deps.config.run_registry_multi_stream_enabled {
                naked_tg::run_registry::RegisterRunOptions::cap_three()
            } else {
                naked_tg::run_registry::RegisterRunOptions::step2_single_run()
            }
        },
    ) {
        Ok(summary) => summary,
        Err(err) => {
            if matches!(
                err,
                naked_tg::run_registry::RegisterRunError::ThreadCapacityExceeded { .. }
            ) {
                crate::metrics::record_run_registry_cap_reject();
                let _ = bot
                    .send_message(
                        ctx.chat_id,
                        "⚠️ 3 runs already active in this thread — abort or wait for one to finish before starting another.",
                    )
                    .maybe_thread(ctx.thread_id)
                    .await;
            }
            tracing::warn!(chat_id = chat_id_raw, ?tid, error = ?err, "run registry rejected stream start");
            abort_for_reject.cancel();
            drain_rejected_stream(events, permissions).await;
            return StreamResponseOutcome::default();
        }
    };
    crate::metrics::record_run_registry_register();
    let run_id = run_summary.run_id.clone();
    let attach_workspace = deps.agent.session_workspace(&session_id_for_attach).await;
    if let Some(workspace) = attach_workspace.clone() {
        naked_tg::tg_attach::bind_workspace_run(workspace, run_id.clone()).await;
    }

    STEER_SENDERS.write().await.insert(run_id.clone(), steer);

    tracing::debug!(
        chat_id = chat_id_raw,
        ?tid,
        thread_id_raw = ?ctx.thread_id,
        "stream_response: sending typing"
    );
    send_typing_raw(http_client, base_url, chat_id_raw, tid).await;

    // PLAN_MEDIA_UX_v1 M4 / BUG_REGISTRY B02: single-bubble
    // stream-start. Previously sent TWO messages (⏳ placeholder
    // + ⏯️ control card with buttons), creating empty visual
    // clutter while the first chunk landed. Now ONE message
    // carries both the placeholder content (edited as chunks
    // arrive via edit_message_text) AND the [⏹ Стоп] [⏩ Send
    // now] inline keyboard. Telegram's editMessageText preserves
    // reply_markup when the `reply_markup` field is omitted, so
    // the buttons ride untouched through every streaming flush.
    // End-of-turn clears them with editMessageReplyMarkup (no
    // `.reply_markup()` arg) instead of deleting the message.
    let placeholder = match send_stream_placeholder(&bot, &ctx, &run_id).await {
        Some(id) => id,
        None => {
            if let Some(workspace) = attach_workspace.as_deref() {
                naked_tg::tg_attach::unbind_workspace_run(workspace, &run_id).await;
            }
            if crate::shared::RUN_REGISTRY.remove_run(&run_id).is_some() {
                crate::metrics::record_run_registry_remove();
            }
            abort_for_reject.cancel();
            drain_rejected_stream(events, permissions).await;
            return StreamResponseOutcome::default();
        }
    };
    let _ = crate::shared::RUN_REGISTRY.bind_message(
        &run_id,
        naked_tg::run_registry::MessageKey::new(chat_id_raw, placeholder.0),
    );
    crate::shared::CONTROL_CARDS
        .write()
        .await
        .insert(run_id.clone(), (ctx.chat_id, placeholder));

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
    let model_switch = naked_tg::model_switch::new_shared();
    MODEL_SWITCHES
        .write()
        .await
        .insert(run_id.clone(), model_switch.clone());

    let mut view = CompositeView::new(model_tag);
    update_cached_run_state(&run_id, &view, None);
    let mut dirty = false;
    let mut last_sent = String::new();
    let mut html_broken = false;
    let mut aborted_for_switch = false;
    let mut last_event_at = tokio::time::Instant::now();
    let mut stall_level: u8 = 0; // 0=none, 1=warned 60s, 2=critical 120s

    // Fixed-interval ticker for streaming flushes.
    // The actual rate limiting happens inside RATE_LIMITER.edit() — the
    // ticker just decides when to ATTEMPT a flush.
    let mut flush_interval = tokio::time::interval(std::time::Duration::from_millis(
        naked_tg::rate_limit::MIN_GAP_MS,
    ));
    flush_interval.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);
    flush_interval.tick().await; // consume first immediate tick

    let mut got_idle = false;
    let mut wall_timeout = false;
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
                view.events.push(TurnEvent::Note(
                    "⚠️ Нет ответа 60с — возможно, зависло".to_string(),
                ));
                dirty = true;
            } else if stall_level == 1 && elapsed >= 120 {
                stall_level = 2;
                view.events.push(TurnEvent::Note(
                    "🔴 Зависло 2 мин — /abort чтобы прервать".to_string(),
                ));
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
                        let _ = crate::shared::safe_send(
                            &bot,
                            &ctx,
                            msg,
                            Some(teloxide::types::ParseMode::Html),
                        )
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
                    let _ = flush_live_to_all_sinks(
                        &bot,
                        &run_id,
                        primary_sink_or_fallback(&run_id, ctx, placeholder),
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
                    let _ = crate::shared::safe_send(
                        &bot,
                        &ctx,
                        note,
                        Some(teloxide::types::ParseMode::Html),
                    )
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
                    view.events.push(TurnEvent::Note(format!(
                        "✅ <b>Доставлено</b>\n<pre>{}</pre>",
                        crate::fmt_utils::escape_html_min(&text)
                    )));
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
                            .filter_map(|mid| acks.remove(&(run_id.clone(), *mid)))
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

                    // User-requested 2026-05-13 (img_20260513_e710.png):
                    // also delete the user's ORIGINAL steer message,
                    // since the streaming view now эхоит the same text
                    // as `✅ Доставлено → <pre>...</pre>`. Leaving the
                    // user msg below the streaming bubble is redundant
                    // visual noise.
                    //
                    // Best-effort: bot can only delete user messages
                    // in groups where it has `can_delete_messages`
                    // admin perm, AND within 48h. In private chats
                    // (where the bot lacks that capability) the delete
                    // simply returns Forbidden — we log at debug
                    // (never error) so this can't crash streaming.
                    for mid in &msg_ids {
                        let tg_mid = teloxide::types::MessageId(*mid);
                        if let Err(e) = bot.delete_message(ctx.chat_id, tg_mid).await {
                            tracing::debug!(
                                chat = ctx.chat_id.0,
                                msg = *mid,
                                "steer user-msg delete skipped \
                                 (not admin / >48h / private chat / already gone): {e}"
                            );
                        }
                    }
                }
                AgentEvent::Error(e) => {
                    // "cancelled" = normal turn displacement, not a crash.
                    if e.contains("cancelled") || e.contains("Cancelled") {
                        got_idle = true;
                    }
                    if e == naked_core::error::WALL_TIMEOUT_MESSAGE {
                        wall_timeout = true;
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
                flush_live_to_all_sinks(
                    &bot,
                    &run_id,
                    primary_sink_or_fallback(&run_id, ctx, placeholder),
                    &view,
                    &mut last_sent,
                    &mut html_broken,
                )
                .await;
                update_cached_run_state(&run_id, &view, None);
                dirty = false;

                // If the rate limiter has this chat blocked (429),
                // stretch the ticker to avoid hammering.
                let primary = primary_sink_or_fallback(&run_id, ctx, placeholder);
                let backoff = rate_limiter.streaming_interval(primary.ctx.chat_id.0).await;
                if backoff > flush_interval.period() {
                    flush_interval = tokio::time::interval(backoff);
                    flush_interval.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);
                    flush_interval.tick().await;
                }
            }
        }
    }

    notify_if_agent_died_without_idle(&bot, ctx, got_idle).await;
    typing_cancel.cancel();
    cleanup_stream_registries(&bot, &run_id).await;

    if aborted_for_switch {
        // Don't send final — the turn was interrupted. Edit placeholder to
        // indicate switch in progress.
        RATE_LIMITER
            .edit_plain(&bot, ctx.chat_id, placeholder, "⚡ Switching model…")
            .await;
        if let Some(workspace) = attach_workspace.as_deref() {
            naked_tg::tg_attach::unbind_workspace_run(workspace, &run_id).await;
        }
        if crate::shared::RUN_REGISTRY.remove_run(&run_id).is_some() {
            crate::metrics::record_run_registry_remove();
        }
        return StreamResponseOutcome {
            got_idle,
            wall_timeout,
        };
    }

    let final_html = view.render_final();
    update_cached_run_state(&run_id, &view, Some(final_html.clone()));
    let primary = primary_sink_or_fallback(&run_id, ctx, placeholder);
    send_final_to_all_sinks(bot.clone(), &run_id, primary, &final_html, &view).await;
    send_provider_error_card_if_needed(&bot, primary.ctx, &view).await;
    deliver_queued_attachments(http_client, base_url, primary.ctx, tg_attach_queue, &run_id).await;
    if let Some(workspace) = attach_workspace.as_deref() {
        naked_tg::tg_attach::unbind_workspace_run(workspace, &run_id).await;
    }
    if crate::shared::RUN_REGISTRY.remove_run(&run_id).is_some() {
        crate::metrics::record_run_registry_remove();
    }
    StreamResponseOutcome {
        got_idle,
        wall_timeout,
    }
}

#[derive(Clone, Copy)]
struct PrimaryRunSink {
    ctx: ChatCtx,
    message_id: teloxide::types::MessageId,
}

fn primary_sink_or_fallback(
    run_id: &str,
    fallback_ctx: ChatCtx,
    fallback_msg: teloxide::types::MessageId,
) -> PrimaryRunSink {
    crate::shared::RUN_REGISTRY
        .primary_sink(run_id)
        .map(|sink| PrimaryRunSink {
            ctx: ChatCtx {
                chat_id: teloxide::types::ChatId(sink.chat_id),
                thread_id: sink
                    .thread_id
                    .map(|id| teloxide::types::ThreadId(teloxide::types::MessageId(id))),
                reply_to: None,
            },
            message_id: teloxide::types::MessageId(sink.message_id),
        })
        .unwrap_or(PrimaryRunSink {
            ctx: fallback_ctx,
            message_id: fallback_msg,
        })
}

async fn flush_live_to_all_sinks(
    bot: &Bot,
    run_id: &str,
    primary: PrimaryRunSink,
    view: &CompositeView,
    last_sent: &mut String,
    html_broken: &mut bool,
) -> bool {
    let html = view.render_live();
    let primary_ok = flush_live_html(
        bot,
        primary.ctx.chat_id,
        primary.message_id,
        &html,
        last_sent,
        html_broken,
    )
    .await;
    for sink in crate::shared::RUN_REGISTRY.mirror_sinks(run_id) {
        let mut mirror_last_sent = String::new();
        let mut mirror_html_broken = false;
        let _ = flush_live_html(
            bot,
            teloxide::types::ChatId(sink.chat_id),
            teloxide::types::MessageId(sink.message_id),
            &html,
            &mut mirror_last_sent,
            &mut mirror_html_broken,
        )
        .await;
    }
    primary_ok
}

async fn send_final_to_all_sinks(
    bot: Bot,
    run_id: &str,
    primary: PrimaryRunSink,
    final_html: &str,
    view: &CompositeView,
) {
    send_final(
        bot.clone(),
        primary.ctx,
        primary.message_id,
        final_html,
        view,
    )
    .await;
    for sink in crate::shared::RUN_REGISTRY.mirror_sinks(run_id) {
        let mirror_ctx = ChatCtx {
            chat_id: teloxide::types::ChatId(sink.chat_id),
            thread_id: sink
                .thread_id
                .map(|id| teloxide::types::ThreadId(teloxide::types::MessageId(id))),
            reply_to: None,
        };
        send_final(
            bot.clone(),
            mirror_ctx,
            teloxide::types::MessageId(sink.message_id),
            final_html,
            view,
        )
        .await;
    }
}

async fn drain_rejected_stream(
    mut events: tokio::sync::mpsc::Receiver<AgentEvent>,
    permissions: tokio::sync::mpsc::Sender<PermissionResponse>,
) {
    while let Some(event) = events.recv().await {
        match event {
            AgentEvent::PermissionRequest { call_id, .. } => {
                let _ = permissions
                    .send(PermissionResponse {
                        call_id,
                        allowed: false,
                    })
                    .await;
            }
            AgentEvent::Idle => break,
            _ => {}
        }
    }
}

async fn notify_if_agent_died_without_idle(bot: &Bot, ctx: ChatCtx, got_idle: bool) {
    // If the agent task died without sending Idle (panic/crash), notify the user
    // so they know something broke.
    if got_idle {
        return;
    }
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

async fn cleanup_stream_registries(bot: &Bot, run_id: &str) {
    MODEL_SWITCHES.write().await.remove(run_id);
    STEER_SENDERS.write().await.remove(run_id);

    // PLAN_MEDIA_UX_v1 M4 / B02: clear the [⏹ Стоп] [⏩ Send now] inline
    // keyboard from the placeholder (which now ALSO holds the final text).
    let mut control_keys: Vec<_> = crate::shared::RUN_REGISTRY
        .bound_message_keys(run_id)
        .into_iter()
        .map(|key| {
            (
                teloxide::types::ChatId(key.chat_id),
                teloxide::types::MessageId(key.message_id),
            )
        })
        .collect();
    if let Some((chat, mid)) = crate::shared::CONTROL_CARDS.write().await.remove(run_id) {
        control_keys.push((chat, mid));
    }
    control_keys.sort_by_key(|(chat, mid)| (chat.0, mid.0));
    control_keys.dedup();
    for (chat, mid) in control_keys {
        if let Err(e) = bot.edit_message_reply_markup(chat, mid).await {
            tracing::debug!(
                chat = chat.0,
                msg = mid.0,
                "control card clear failed (likely already cleared): {e}"
            );
        }
    }
    cleanup_stale_steer_acks(bot, run_id).await;
}

fn update_cached_run_state(run_id: &str, view: &CompositeView, final_html: Option<String>) {
    let status_line = Some(format!("{} #{}", view.phase, view.tick));
    let _ = crate::shared::RUN_REGISTRY.update_rendered_state(
        run_id,
        naked_tg::run_registry::RenderedRunState {
            live_html: view.render_live(),
            final_html,
            status_line,
        },
    );
}

async fn cleanup_stale_steer_acks(bot: &Bot, run_id: &str) {
    // S6 cleanup: any ack ids still parked for this run are unreachable now.
    let stale: Vec<(teloxide::types::ChatId, teloxide::types::MessageId)> = {
        let mut acks = crate::shared::STEER_ACK_IDS.write().await;
        let keys: Vec<_> = acks
            .keys()
            .filter(|(rid, _)| rid == run_id)
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

async fn send_provider_error_card_if_needed(bot: &Bot, ctx: ChatCtx, view: &CompositeView) {
    if !view.had_provider_error {
        return;
    }
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

async fn deliver_queued_attachments(
    http_client: &reqwest::Client,
    base_url: &str,
    ctx: ChatCtx,
    tg_attach_queue: &naked_tg::tg_attach::AttachmentQueue,
    run_id: &str,
) {
    let attachments = naked_tg::tg_attach::drain_for_run(tg_attach_queue, run_id).await;
    for att in attachments {
        deliver_one_attachment(http_client, base_url, ctx, att).await;
    }
}

async fn deliver_one_attachment(
    http_client: &reqwest::Client,
    base_url: &str,
    ctx: ChatCtx,
    att: naked_tg::tg_attach::StagedAttachment,
) {
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
    match form.file(field, &att.path).await {
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
