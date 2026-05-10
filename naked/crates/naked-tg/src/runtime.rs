//! Main event loop, signal handling, watchdog, and graceful shutdown.
//!
//! [`run_event_loop`] receives all wired state from [`crate::wiring::build`]
//! and runs until a SIGINT (or cancellation) is received.

use std::sync::Arc;
use std::time::Duration;

use teloxide::prelude::*;
use teloxide::types::CallbackQuery;

use crate::album;
use crate::callbacks::handle_callback;
use crate::message_handler::BotDeps;
use crate::shared::{HashMap, PendingPermissions, RwLock, STEER_SENDERS, run_health_server};
use crate::wiring::WiredBot;

/// Run the Telegram polling event loop until shutdown.
///
/// Handles messages, edited messages, callback queries, and message reactions.
/// Blocks until a SIGINT is received, then waits up to 30s for in-flight
/// tasks to drain before returning.
pub(crate) async fn run_event_loop(wb: WiredBot) {
    let WiredBot {
        agent,
        channel_map,
        config,
        bot,
        bot_token,
        bot_identity,
        http_client,
        base_url,
        tg_attach_queue,
        mcp_failures,
        rate_limiter,
        _scheduler_lock,
        _memory_scheduler,
    } = wb;

    // Permissions are per-process state, not part of the DI graph.
    let pending_perms: PendingPermissions = Arc::new(RwLock::new(HashMap::new()));

    // ── A7: Send MCP startup diagnostics to owner chat ────────────────
    if !mcp_failures.is_empty() && !config.telegram.allowed_chat_ids.is_empty() {
        let owner_chat = ChatId(config.telegram.allowed_chat_ids[0]);
        let mut lines = vec!["🔌 <b>MCP startup report</b>".to_string()];
        // Show connected servers
        let connected = agent.list_mcp_servers().await;
        for (name, tools) in &connected {
            lines.push(format!("  ✅ <b>{name}</b>: {tools} tools"));
        }
        // Show failures
        for f in &mcp_failures {
            let err_short: String = f.error.chars().take(120).collect();
            lines.push(format!(
                "  ❌ <b>{}</b>: {}",
                crate::markup::escape_html(&f.name),
                crate::markup::escape_html(&err_short),
            ));
        }
        let text = lines.join("\n");
        let _ = bot
            .send_message(owner_chat, &text)
            .parse_mode(teloxide::types::ParseMode::Html)
            .await;
    }

    if config.telegram.allowed_chat_ids.is_empty() {
        tracing::warn!(
            "allowed_chat_ids is empty — ALL messages will be rejected! \
             Add your chat IDs to naked.json."
        );
    }

    // ── FIX-1: Notify ONLY chats with genuinely interrupted sessions ─────
    //
    // drain_interrupted_sessions filters to sessions updated within 5 min.
    // Stale entries (accumulated across SIGKILL restarts) are silently skipped.
    {
        let crashed_sessions = agent
            .drain_interrupted_sessions(chrono::Duration::minutes(5))
            .await;
        if !crashed_sessions.is_empty() {
            let entries = channel_map.all_entries().await;
            let mut notified = 0u32;
            for (chat_id, thread_id_raw, session_id) in &entries {
                if !crashed_sessions.contains(session_id) {
                    continue;
                }
                let cid = ChatId(*chat_id);
                let tid = if *thread_id_raw != 0 {
                    Some(teloxide::types::ThreadId(teloxide::types::MessageId(
                        *thread_id_raw as i32,
                    )))
                } else {
                    None
                };
                let text = "⚠️ Бот перезапустился. Последний запрос потерян — повтори.";
                let mut req = bot.send_message(cid, text);
                if let Some(t) = tid {
                    req = req.message_thread_id(t);
                }
                let _ = req.await;
                notified += 1;
            }
            tracing::info!(
                "crash recovery: {} session(s) interrupted, notified {notified} chat(s)",
                crashed_sessions.len()
            );
        }
    }

    // Health check endpoint (lightweight TCP)
    let health_port: u16 = std::env::var("HEALTH_PORT")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(0);
    if health_port > 0 {
        tokio::spawn(run_health_server(health_port));
    }

    let shutdown = tokio_util::sync::CancellationToken::new();
    let shutdown_signal = shutdown.clone();
    tokio::spawn(async move {
        let _ = tokio::signal::ctrl_c().await;
        tracing::info!("Received SIGINT, shutting down gracefully…");
        shutdown_signal.cancel();
    });

    // ── systemd watchdog (no-op outside `Type=notify`) ─────────────────
    // Wired after the bot is fully constructed but before the polling
    // loop starts, so READY=1 is sent only when the bot is actually
    // ready to handle work. Liveness arbiter (T2 of
    // PLAN_LIVENESS_v1) gates sd_notify on real polling progress —
    // not just process aliveness. If polling silently dies (incident
    // 2026-05-10 12:47), the arbiter stops pinging and systemd
    // restarts us within WatchdogSec.
    use naked_tg::watchdog::WatchdogNotifier as _;
    let watchdog_notifier: Arc<dyn naked_tg::watchdog::WatchdogNotifier> =
        match naked_tg::watchdog::SystemdWatchdog::detect_from_env() {
            Some(wd) => {
                let interval_secs = wd.interval().map(|d| d.as_secs()).unwrap_or(0);
                tracing::info!(interval_secs, "systemd watchdog enabled");
                Arc::new(wd)
            }
            None => Arc::new(naked_tg::watchdog::NoopWatchdog),
        };
    watchdog_notifier.notify_ready().await;
    let liveness = Arc::new(naked_core::liveness::LivenessRegistry::new());
    liveness.register("tg_polling.tick");
    let watchdog_shutdown = Arc::new(tokio::sync::Notify::new());
    let watchdog_handle = naked_tg::watchdog::spawn_watchdog_with_liveness(
        watchdog_notifier.clone(),
        liveness.clone(),
        vec![naked_tg::watchdog::LivenessRequirement {
            source: "tg_polling.tick",
            // Long-poll is ~30s + reasonable network slack. If we don't
            // see a tick in 90s, the loop is wedged and systemd should
            // recycle us.
            max_silence: Duration::from_secs(90),
        }],
        // Grace covers boot wiring + first long-poll round trip.
        Duration::from_secs(60),
        watchdog_shutdown.clone(),
    );
    {
        let watchdog_notifier = watchdog_notifier.clone();
        let watchdog_shutdown = watchdog_shutdown.clone();
        let shutdown_token = shutdown.clone();
        tokio::spawn(async move {
            shutdown_token.cancelled().await;
            watchdog_shutdown.notify_waiters();
            watchdog_notifier.notify_stopping().await;
        });
    }
    let _watchdog_handle = watchdog_handle;

    let task_tracker = Arc::new(tokio::sync::Semaphore::new(50));
    let album_buffer = album::AlbumBuffer::default();
    // Runtime toggle for `tg_sender_attribution`. Seeded from config; the
    // `/attribution on|off` command flips this atomic without restarting.
    let attribution_flag: Arc<std::sync::atomic::AtomicBool> = Arc::new(
        std::sync::atomic::AtomicBool::new(config.telegram.tg_sender_attribution),
    );
    let mut offset: i64 = 0;

    // Use the pre-computed base URL and token as borrowed slices.
    let base: &str = &base_url;
    let client: &reqwest::Client = &http_client;

    // T4 of PLAN_LIVENESS_v1: getUpdates round trip is wrapped in one
    // tokio::time::timeout. Without this, a half-broken TCP stream
    // could hang resp.json().await forever (incident 2026-05-10).
    // Long-poll is 30s on the server side; we add 30s of network
    // slack and another 15s for body decode — total 75s.
    const POLL_TIMEOUT: Duration = Duration::from_secs(75);

    while !shutdown.is_cancelled() {
        // Heartbeat the liveness registry every iteration. The
        // arbiter forwards sd_notify only while this stays fresh
        // (max_silence = 90s, set above). One missed iteration is
        // tolerated; two (~150s of silence) trips the watchdog.
        liveness.beat("tg_polling.tick");

        let body = serde_json::json!({
            "offset": offset,
            "timeout": 30,
            "allowed_updates": ["message", "edited_message", "callback_query", "message_reaction"]
        });
        tracing::debug!(offset, "polling getUpdates");

        let payload: serde_json::Value = tokio::select! {
            _ = shutdown.cancelled() => break,
            outcome = tokio::time::timeout(
                POLL_TIMEOUT,
                fetch_updates(client, base, &body),
            ) => match outcome {
                Ok(Ok(p)) => p,
                Ok(Err(e)) => {
                    tracing::error!("getUpdates error: {e}");
                    tokio::time::sleep(Duration::from_secs(3)).await;
                    continue;
                }
                Err(_) => {
                    tracing::warn!(
                        timeout_secs = POLL_TIMEOUT.as_secs(),
                        "getUpdates timed out — cycling connection"
                    );
                    tokio::time::sleep(Duration::from_secs(2)).await;
                    continue;
                }
            }
        };
        if payload["ok"].as_bool() != Some(true) {
            let desc = payload["description"].as_str().unwrap_or("unknown");
            tracing::warn!("getUpdates API error: {desc}");
            tokio::time::sleep(Duration::from_secs(5)).await;
            continue;
        }
        let updates = match payload["result"].as_array() {
            Some(arr) => arr.clone(),
            None => continue,
        };
        if !updates.is_empty() {
            tracing::info!("Received {} update(s), offset now={offset}", updates.len());
            for u in &updates {
                let keys: Vec<&str> = u
                    .as_object()
                    .map(|o| o.keys().map(|k| k.as_str()).collect())
                    .unwrap_or_default();
                tracing::debug!(?keys, "update keys");
            }
        }
        for upd in &updates {
            if let Some(uid) = upd["update_id"].as_i64()
                && uid >= offset
            {
                offset = uid + 1;
            }
            if let Some(msg_val) = upd.get("message") {
                let thread_id = msg_val.get("message_thread_id").and_then(|v| v.as_i64());
                let is_topic = msg_val
                    .get("is_topic_message")
                    .and_then(|v| v.as_bool())
                    .unwrap_or(false);
                let msg: Message = match serde_json::from_value(msg_val.clone()) {
                    Ok(m) => m,
                    Err(e) => {
                        tracing::warn!("Failed to parse message: {e}");
                        continue;
                    }
                };
                tracing::info!(
                    chat_id = msg.chat.id.0,
                    ?thread_id,
                    is_topic,
                    "Dispatching message"
                );
                // One cheap clone-bag covers both the album-flush callback
                // and the sync dispatch path below. Replaces the 11-line
                // manual plumbing this function used to carry.
                let deps = BotDeps {
                    bot: bot.clone(),
                    agent: agent.clone(),
                    channel_map: channel_map.clone(),
                    config: config.clone(),
                    pending_perms: pending_perms.clone(),
                    http_client: http_client.clone(),
                    base_url: base_url.clone(),
                    rate_limiter: rate_limiter.clone(),
                    attribution_flag: attribution_flag.clone(),
                    bot_token: bot_token.clone(),
                    bot_identity: bot_identity.clone(),
                    tg_attach_queue: tg_attach_queue.clone(),
                };
                let permit = task_tracker.clone();
                let album = album_buffer.clone();
                let task_tracker_for_flush = task_tracker.clone();
                let guard_chat = msg.chat.id;
                let guard_thread = msg.thread_id;
                naked_tg::guarded::spawn_guarded(
                    bot.clone(),
                    guard_chat,
                    guard_thread,
                    "message",
                    async move {
                        // Album-coalescing wrapper: messages tagged with a
                        // `media_group_id` are buffered and flushed once the
                        // debounce window closes; everything else dispatches
                        // immediately.
                        let deps_for_flush = deps.clone();
                        let outcome = album
                            .submit(msg, move |mut msgs| async move {
                                let _permit = task_tracker_for_flush.acquire().await;
                                // Sort by message_id so the user's perceived order
                                // matches the order of images in the agent prompt.
                                msgs.sort_by_key(|m| m.id.0);
                                let primary = msgs.remove(0);
                                if let Err(e) = deps_for_flush.handle(primary, msgs).await {
                                    tracing::error!("handle_message (album) error: {e}");
                                }
                            })
                            .await;
                        if let album::Decision::Solo(msg) = outcome {
                            let _permit = permit.acquire().await;
                            if let Err(e) = deps.handle(*msg, Vec::new()).await {
                                tracing::error!("handle_message error: {e}");
                            }
                        }
                    },
                );
            }
            // edited_message → treat as a new message (simplest useful behavior).
            // If the original was already processed, the agent sees the edit as
            // follow-up context. If still pending, it appears as a correction.
            if let Some(msg_val) = upd.get("edited_message") {
                let msg: Message = match serde_json::from_value(msg_val.clone()) {
                    Ok(m) => m,
                    Err(e) => {
                        tracing::warn!("Failed to parse edited_message: {e}");
                        continue;
                    }
                };
                let chat_id_raw = msg.chat.id.0;
                let tid_raw = msg.thread_id.map(|t| t.0.0);
                let msg_id = msg.id.0;
                let edit_text = msg.text().or(msg.caption()).unwrap_or_default().to_string();

                // If a turn is actively streaming for this chat,
                // inject as steer edit rather than a new message.
                let steer_key = (chat_id_raw, tid_raw);
                let steered = {
                    let map = STEER_SENDERS.read().await;
                    if let Some(tx) = map.get(&steer_key) {
                        tx.try_send(naked_core::types::SteerMessage {
                            msg_id,
                            text: edit_text.clone(),
                            is_edit: true,
                        })
                        .is_ok()
                    } else {
                        false
                    }
                };

                if steered {
                    tracing::info!(
                        chat_id = chat_id_raw,
                        msg_id,
                        "edited_message routed as steer edit"
                    );
                } else {
                    tracing::info!(
                        chat_id = chat_id_raw,
                        msg_id,
                        "Dispatching edited_message as new message"
                    );
                    let deps = BotDeps {
                        bot: bot.clone(),
                        agent: agent.clone(),
                        channel_map: channel_map.clone(),
                        config: config.clone(),
                        pending_perms: pending_perms.clone(),
                        http_client: http_client.clone(),
                        base_url: base_url.clone(),
                        rate_limiter: rate_limiter.clone(),
                        attribution_flag: attribution_flag.clone(),
                        bot_token: bot_token.clone(),
                        bot_identity: bot_identity.clone(),
                        tg_attach_queue: tg_attach_queue.clone(),
                    };
                    let permit = task_tracker.clone();
                    let guard_chat = msg.chat.id;
                    let guard_thread = msg.thread_id;
                    naked_tg::guarded::spawn_guarded(
                        bot.clone(),
                        guard_chat,
                        guard_thread,
                        "edited_message",
                        async move {
                            let _permit = permit.acquire().await;
                            if let Err(e) = deps.handle(msg, Vec::new()).await {
                                tracing::error!("handle edited_message error: {e}");
                            }
                        },
                    );
                }
            }
            if let Some(cb_val) = upd.get("callback_query") {
                tracing::info!("Dispatching callback_query");
                let q: CallbackQuery = match serde_json::from_value(cb_val.clone()) {
                    Ok(q) => q,
                    Err(e) => {
                        tracing::warn!("Failed to parse callback: {e}");
                        continue;
                    }
                };
                let bot = bot.clone();
                let pending_perms = pending_perms.clone();
                let agent = agent.clone();
                let channel_map = channel_map.clone();
                let config = config.clone();
                let permit = task_tracker.clone();
                // Callback chat_id: from the message the button was on.
                let cb_chat = q.message.as_ref().map(|m| m.chat().id).unwrap_or(ChatId(0));
                let cb_thread = q.message.as_ref().and_then(|m| match m {
                    teloxide::types::MaybeInaccessibleMessage::Regular(msg) => msg.thread_id,
                    _ => None,
                });
                naked_tg::guarded::spawn_guarded(
                    bot.clone(),
                    cb_chat,
                    cb_thread,
                    "callback",
                    async move {
                        let _permit = permit.acquire().await;
                        if let Err(e) =
                            handle_callback(bot, q, pending_perms, agent, channel_map, config).await
                        {
                            tracing::error!("handle_callback error: {e}");
                        }
                    },
                );
            }
            // message_reaction → 👎 removes/cancels queued message hint.
            // Since messages go directly into conversation history, we can't
            // truly remove them. Instead, append a correction note.
            if let Some(reaction_val) = upd.get("message_reaction") {
                let chat_id = reaction_val
                    .get("chat")
                    .and_then(|c| c.get("id"))
                    .and_then(|v| v.as_i64())
                    .unwrap_or(0);
                let msg_id = reaction_val
                    .get("message_id")
                    .and_then(|v| v.as_i64())
                    .unwrap_or(0);
                let new_reactions: Vec<String> = reaction_val
                    .get("new_reaction")
                    .and_then(|v| v.as_array())
                    .map(|arr| {
                        arr.iter()
                            .filter_map(|r| r.get("emoji").and_then(|e| e.as_str()))
                            .map(String::from)
                            .collect()
                    })
                    .unwrap_or_default();
                tracing::debug!(chat_id, msg_id, ?new_reactions, "message_reaction");
                // 👎 → queue cancellation note for the agent
                if new_reactions.iter().any(|e| e == "👎") {
                    let tid = reaction_val
                        .get("chat")
                        .and_then(|c| c.get("message_thread_id"))
                        .and_then(|v| v.as_i64())
                        .map(|v| v as i32);
                    if let Some(sid) = channel_map.get(chat_id, tid).await {
                        agent
                            .queue_message(
                                &sid,
                                &format!(
                                    "[The user reacted with 👎 to message #{msg_id}. \
                                     Disregard that message if you haven't started working on it yet.]"
                                ),
                            )
                            .await;
                        tracing::info!(chat_id, msg_id, "queued 👎 cancellation note");
                    }
                }
            }
        }
    }

    tracing::info!("Waiting for in-flight tasks to complete…");
    // Wait for all permits to be returned (all tasks finished)
    let _ = tokio::time::timeout(Duration::from_secs(30), task_tracker.acquire_many(50)).await;
    tracing::info!("Shutdown complete.");
}

/// Fetch + parse a `getUpdates` response in one await.
///
/// Combines the previously-separate `.send().await` and `.json().await`
/// into a single future so callers can wrap the entire round trip in
/// a single `tokio::time::timeout`. The pre-T4 split allowed
/// `.json()` to hang indefinitely on a half-broken TCP stream
/// (incident 2026-05-10 12:47).
async fn fetch_updates(
    client: &reqwest::Client,
    base: &str,
    body: &serde_json::Value,
) -> Result<serde_json::Value, reqwest::Error> {
    let resp = client
        .post(format!("{base}/getUpdates"))
        .json(body)
        .send()
        .await?;
    resp.json::<serde_json::Value>().await
}
