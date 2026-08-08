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
            last_err = redact_for_log(&e);
            tracing::warn!("placeholder send failed, will retry: {last_err}");
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
                last_err = redact_for_log(&e);
                tracing::warn!("placeholder retry after {delay}s failed: {last_err}");
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
    /// Optional final keyboard to leave on the primary stream bubble after the
    /// live [Stop]/[Send] controls are removed. Streaming remains generic: the
    /// caller decides what markup, if any, belongs to this run kind.
    pub(crate) final_reply_markup: Option<teloxide::types::InlineKeyboardMarkup>,
}

impl StreamRunContext {
    pub(crate) fn chat_turn(session_id: String) -> Self {
        Self {
            requested_run_id: None,
            session_id,
            kind: naked_tg::run_registry::RunKind::ChatTurn,
            source_ref: None,
            final_reply_markup: None,
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

/// B113: single definition of "the agent made progress".
///
/// Consumed by BOTH the run-registry silence warner and the UI stall detector.
/// A keep-alive `Heartbeat` is explicitly NOT progress: it is emitted while a
/// tool or provider call is blocked, so counting it as progress is what let a
/// wedged turn look healthy to the user. Any future keep-alive variant is
/// classified once, here.
fn is_progress_event(event: &AgentEvent) -> bool {
    !matches!(event, AgentEvent::Heartbeat)
}

/// B106: annotate the view when the provider the user picked did not actually
/// serve the turn, so `render_final` can warn at the top of the answer.
///
/// Extracted from `stream_response` (already ~500 lines): "decide whether a
/// fallback happened" is its own concern, and keeping it here makes it
/// testable without driving a whole stream.
async fn attach_fallback_notice(view: &mut CompositeView, deps: &crate::message_handler::BotDeps) {
    let Some(requested) = view.model_tag.split('/').next().filter(|r| !r.is_empty()) else {
        return;
    };
    let Some(info) = deps.agent.last_fallback_for(requested).await else {
        return;
    };
    if info.served_by == info.requested {
        return;
    }
    tracing::info!(
        requested = %info.requested,
        served_by = %info.served_by,
        "B106: surfacing provider fallback to user"
    );
    view.fallback_notice = Some(format!(
        "Отвечал {} — {} не смог: {}",
        info.served_by, info.requested, info.reason
    ));
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
    let final_reply_markup = run_ctx.final_reply_markup.clone();
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
            }
            // B119a: EVERY rejection must reach the user. Only the capacity
            // case used to, so a duplicate session/source/run-id made the bot
            // answer nothing at all: no placeholder, no text, and no
            // registered run to /abort. The user saw a message that vanished.
            let _ = bot
                .send_message(ctx.chat_id, reject_message(&err))
                .maybe_thread(ctx.thread_id)
                .await;
            let safe = redact_for_log(format!("{err:?}"));
            tracing::warn!(chat_id = chat_id_raw, ?tid, error = %safe, "run registry rejected stream start");
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
        // B113: "did the agent make progress?" must be ONE predicate. The
        // run-registry gate below excluded `Heartbeat`; this reset did not, so
        // an agent wedged inside a provider call kept its keep-alive pump
        // running, `stall_level` never left 0, and the user never saw the 60s /
        // 2min warnings — while `warn_silent_runs` DID fire in the journal.
        // Ops saw a silent run, the user saw a healthy spinner.
        if event.as_ref().is_some_and(is_progress_event) {
            last_event_at = tokio::time::Instant::now();
            stall_level = 0;
        } else if event.is_none() {
            let elapsed = last_event_at.elapsed().as_secs();
            // B118b: the bubble Note keeps the live view honest; `stall_notice`
            // is what survives into the final answer. Both are set from this
            // one place so they can never disagree about whether a turn stalled.
            if stall_level == 0 && elapsed >= 60 {
                stall_level = 1;
                let note = "⚠️ Нет ответа 60с — возможно, зависло".to_string();
                view.stall_notice = Some(note.clone());
                view.events.push(TurnEvent::Note(note));
                dirty = true;
            } else if stall_level == 1 && elapsed >= 120 {
                stall_level = 2;
                let note = "🔴 Зависло 2 мин — /abort чтобы прервать".to_string();
                view.stall_notice = Some(note.clone());
                view.events.push(TurnEvent::Note(note));
                dirty = true;
            }
        }

        let has_event = event.is_some();
        if let Some(event) = event {
            if is_progress_event(&event) {
                let _ = crate::shared::RUN_REGISTRY.mark_run_progress(&run_id);
            }
            match event {
                // B113: `|=`, not `=`. Every other arm sets `dirty = true`;
                // these two OVERWROTE it, so a delta whose handler reports
                // `Clean` (a coalescing no-op) cleared a pending flush queued
                // by an earlier `ToolEnd` \u2014 the tool result then sat in the view
                // unrendered, and if the next event was `Idle` the live bubble
                // never showed it at all.
                AgentEvent::ThinkingDelta(t) => {
                    dirty |= handlers::handle_thinking_delta(&mut view, &t) != ViewAction::Clean;
                }
                AgentEvent::TextDelta(t) => {
                    dirty |= handlers::handle_text_delta(&mut view, &t) != ViewAction::Clean;
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
                            let safe = redact_for_log(&e);
                            tracing::debug!(
                                chat = chat.0,
                                msg = ack_id.0,
                                "steer ack delete failed (likely already gone): {safe}"
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
                            let safe = redact_for_log(&e);
                            tracing::debug!(
                                chat = ctx.chat_id.0,
                                msg = *mid,
                                "steer user-msg delete skipped \
                                 (not admin / >48h / private chat / already gone): {safe}"
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

        crate::shared::RUN_REGISTRY
            .warn_silent_runs(naked_tg::run_registry::ACTIVE_RUN_SILENCE_WARN_THRESHOLD);

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

    // B113: a deliberate in-flight `/model` switch breaks out of the loop
    // WITHOUT setting `got_idle`, so an unconditional call here told the user
    // "\u{1f534} \u{412}\u{43d}\u{443}\u{442}\u{440}\u{435}\u{43d}\u{43d}\u{44f}\u{44f} \u{43e}\u{448}\u{438}\u{431}\u{43a}\u{430}" and logged `error!("agent task died
    // without Idle \u{2014} likely panicked")` on a path the code itself calls
    // "the turn was interrupted" \u2014 poisoning log-based crash alerting too.
    if !aborted_for_switch {
        notify_if_agent_died_without_idle(&bot, ctx, got_idle).await;
    }
    typing_cancel.cancel();

    if aborted_for_switch {
        cleanup_stream_registries(&bot, &run_id, None).await;
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

    attach_fallback_notice(&mut view, deps).await;

    let tg_long_answer_fix_enabled = deps.config.telegram.tg_long_answer_fix_enabled;
    let final_html = view.render_final_with_long_answer_fix(tg_long_answer_fix_enabled);
    update_cached_run_state(&run_id, &view, Some(final_html.clone()));
    let primary = primary_sink_or_fallback(&run_id, ctx, placeholder);
    send_final_to_all_sinks(
        bot.clone(),
        &run_id,
        primary,
        &final_html,
        &view,
        tg_long_answer_fix_enabled,
    )
    .await;
    cleanup_stream_registries(&bot, &run_id, final_reply_markup).await;
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
    tg_long_answer_fix_enabled: bool,
) {
    send_final(
        bot.clone(),
        primary.ctx,
        primary.message_id,
        final_html,
        view,
        tg_long_answer_fix_enabled,
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
            tg_long_answer_fix_enabled,
        )
        .await;
    }
}

/// B119a: user-facing text for a rejected stream start.
///
/// One place decides what each rejection says, so a new `RegisterRunError`
/// variant cannot be added without the compiler pointing here — which is how
/// the three silent variants slipped in behind the one that spoke.
pub(crate) fn reject_message(err: &naked_tg::run_registry::RegisterRunError) -> &'static str {
    use naked_tg::run_registry::RegisterRunError as E;
    match err {
        E::ThreadCapacityExceeded { .. } => {
            "⚠️ 3 runs already active in this thread — abort or wait for one to finish before starting another."
        }
        E::DuplicateSessionId { .. } => {
            "⚠️ Для этой сессии уже идёт ход. Дождись его окончания или /abort."
        }
        E::DuplicateSourceRef { .. } => {
            "⚠️ Эта задача уже выполняется. Дождись результата или /abort."
        }
        E::DuplicateRunId { .. } => "⚠️ Внутренняя коллизия id хода — повтори запрос.",
    }
}

/// B119b: bound the drain.
///
/// This used to await `events.recv()` with no timeout. A rejected start whose
/// agent task ignores its cancelled token parked this future — and the task
/// holding it — forever. The drain is best-effort cleanup, so giving up is
/// strictly better than leaking.
pub(crate) const REJECTED_DRAIN_TIMEOUT: std::time::Duration = std::time::Duration::from_secs(10);

pub(crate) async fn drain_rejected_stream(
    events: tokio::sync::mpsc::Receiver<AgentEvent>,
    permissions: tokio::sync::mpsc::Sender<PermissionResponse>,
) {
    if tokio::time::timeout(
        REJECTED_DRAIN_TIMEOUT,
        drain_rejected_stream_inner(events, permissions),
    )
    .await
    .is_err()
    {
        tracing::warn!(
            timeout_secs = REJECTED_DRAIN_TIMEOUT.as_secs(),
            "rejected-stream drain timed out — abandoning it rather than parking the task"
        );
    }
}

async fn drain_rejected_stream_inner(
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

async fn cleanup_stream_registries(
    bot: &Bot,
    run_id: &str,
    final_reply_markup: Option<teloxide::types::InlineKeyboardMarkup>,
) {
    MODEL_SWITCHES.write().await.remove(run_id);
    STEER_SENDERS.write().await.remove(run_id);

    // PLAN_MEDIA_UX_v1 M4 / B02: clear the [⏹ Стоп] [⏩ Send now] inline
    // keyboard from stream bubbles. B86 may replace that live-control keyboard
    // with a caller-supplied final keyboard on exactly one primary bubble;
    // streaming does not inspect the domain semantics of that markup.
    let bound_keys = crate::shared::RUN_REGISTRY.bound_message_keys(run_id);
    let final_target = final_reply_markup.as_ref().and_then(|_| {
        bound_keys
            .iter()
            .min_by_key(|key| (key.message_id, key.chat_id))
            .map(|key| (key.chat_id, key.message_id))
    });
    let mut control_keys: Vec<_> = bound_keys
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
        let result = if final_target == Some((chat.0, mid.0)) {
            if let Some(markup) = final_reply_markup.clone() {
                bot.edit_message_reply_markup(chat, mid)
                    .reply_markup(markup)
                    .await
            } else {
                bot.edit_message_reply_markup(chat, mid).await
            }
        } else {
            bot.edit_message_reply_markup(chat, mid).await
        };
        if let Err(e) = result {
            let safe = redact_for_log(&e);
            tracing::debug!(
                chat = chat.0,
                msg = mid.0,
                "control card cleanup failed (likely already cleared): {safe}"
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
            let safe = redact_for_log(&e);
            tracing::debug!(
                chat = chat.0,
                msg = ack_id.0,
                "end-of-turn steer ack cleanup failed: {safe}"
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
                    let safe = redact_for_log(&e);
                    tracing::warn!(file = %att.file_name, error = %safe, "telegram_attach send error");
                }
            }
        }
        Err(e) => {
            let safe = redact_for_log(&e);
            tracing::warn!(file = %att.file_name, error = %safe, "telegram_attach form build failed");
        }
    }
}

#[cfg(test)]
mod b113_tests {
    use super::*;

    /// B113: the run-registry silence warner and the UI stall detector must
    /// share one notion of progress. They did not: the registry excluded
    /// `Heartbeat`, the stall timer counted it, so a wedged turn kept a healthy
    /// spinner while ops saw a silent run.
    #[test]
    fn b113_heartbeat_is_not_progress() {
        assert!(
            !is_progress_event(&AgentEvent::Heartbeat),
            "a keep-alive emitted while a call is blocked must not reset the \
             stall timer — that is what hid wedged turns from the user"
        );
    }

    #[test]
    fn b113_real_events_are_progress() {
        assert!(is_progress_event(&AgentEvent::TextDelta("hi".into())));
        assert!(is_progress_event(&AgentEvent::ThinkingDelta("t".into())));
        assert!(is_progress_event(&AgentEvent::Idle));
        assert!(is_progress_event(&AgentEvent::ToolStart {
            call_id: "c".into(),
            name: "bash".into(),
            input: serde_json::json!({}),
        }));
    }
}
