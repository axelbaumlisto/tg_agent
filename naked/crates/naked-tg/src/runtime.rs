//! Main event loop, signal handling, watchdog, and graceful shutdown.
//!
//! [`run_event_loop`] receives all wired state from [`crate::wiring::build`]
//! and runs until a SIGINT (or cancellation) is received.

use std::sync::Arc;
use std::time::Duration;

use teloxide::prelude::*;
use teloxide::types::{CallbackQuery, Message};

use naked_core::AgentCore;
use naked_core::config::Config;
use naked_tg::channel_map::ChannelSessionMap;

use crate::album;
use crate::callbacks::handle_callback;
use crate::message_handler::BotDeps;
use crate::shared::{
    ChatCtx, HashMap, PendingPermissions, RwLock, STEER_SENDERS, is_allowed, run_health_server,
};
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
        research_scheduler,
        _scheduler_lock,
        _memory_scheduler,
        liveness,
    } = wb;

    // Permissions are per-process state, not part of the DI graph.
    let pending_perms: PendingPermissions = Arc::new(RwLock::new(HashMap::new()));

    send_mcp_startup_report(&bot, &agent, &config, &mcp_failures).await;

    if config.telegram.allowed_chat_ids.is_empty() {
        tracing::warn!(
            "allowed_chat_ids is empty — ALL messages will be rejected! \
             Add your chat IDs to naked.json."
        );
    }

    notify_interrupted_sessions(&bot, &agent, &channel_map).await;

    spawn_health_server_from_env();

    let shutdown = tokio_util::sync::CancellationToken::new();
    install_ctrl_c_shutdown_handler(shutdown.clone());
    let _watchdog_handle = start_watchdog(liveness.clone(), shutdown.clone()).await;

    let task_tracker = Arc::new(tokio::sync::Semaphore::new(50));
    let album_buffer = album::InboundCoalescer::default();
    let per_chat_locks = Arc::new(crate::per_chat_locks::PerChatLocks::new());
    // Runtime toggle for `tg_sender_attribution`. Seeded from config; the
    // `/attribution on|off` command flips this atomic without restarting.
    let attribution_flag: Arc<std::sync::atomic::AtomicBool> = Arc::new(
        std::sync::atomic::AtomicBool::new(config.telegram.tg_sender_attribution),
    );

    // Single BotDeps construction — cloned into every handler arm (DRY).
    let shared_deps = BotDeps {
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
        research_scheduler: research_scheduler.clone(),
        per_chat_locks: per_chat_locks.clone(),
    };
    let update_dispatcher = UpdateDispatcher {
        bot: bot.clone(),
        agent: agent.clone(),
        channel_map: channel_map.clone(),
        pending_perms: pending_perms.clone(),
        task_tracker: task_tracker.clone(),
        album_buffer: album_buffer.clone(),
        shared_deps,
    };

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
            update_dispatcher.dispatch_update(upd).await;
        }
    }

    tracing::info!("Waiting for in-flight tasks to complete…");
    // Wait for all permits to be returned (all tasks finished)
    let _ = tokio::time::timeout(Duration::from_secs(30), task_tracker.acquire_many(50)).await;
    tracing::info!("Shutdown complete.");
}

struct UpdateDispatcher {
    bot: Bot,
    agent: Arc<AgentCore>,
    channel_map: Arc<ChannelSessionMap>,
    pending_perms: PendingPermissions,
    task_tracker: Arc<tokio::sync::Semaphore>,
    album_buffer: album::InboundCoalescer,
    shared_deps: BotDeps,
}

impl UpdateDispatcher {
    async fn dispatch_update(&self, upd: &serde_json::Value) {
        if let Some(msg_val) = upd.get("message") {
            self.dispatch_message(msg_val).await;
        }
        if let Some(msg_val) = upd.get("edited_message") {
            self.dispatch_edited_message(msg_val).await;
        }
        if let Some(cb_val) = upd.get("callback_query") {
            self.dispatch_callback_query(cb_val).await;
        }
        if let Some(reaction_val) = upd.get("message_reaction") {
            self.dispatch_message_reaction(reaction_val).await;
        }
    }

    async fn dispatch_message(&self, msg_val: &serde_json::Value) {
        let thread_id = msg_val.get("message_thread_id").and_then(|v| v.as_i64());
        let is_topic = msg_val
            .get("is_topic_message")
            .and_then(|v| v.as_bool())
            .unwrap_or(false);
        let msg: Message = match serde_json::from_value(msg_val.clone()) {
            Ok(m) => m,
            Err(e) => {
                tracing::warn!("Failed to parse message: {e}");
                return;
            }
        };
        tracing::info!(
            chat_id = msg.chat.id.0,
            ?thread_id,
            is_topic,
            "Dispatching message"
        );

        // One cheap clone-bag covers both the album-flush callback and the sync
        // dispatch path below. Replaces the 11-line manual plumbing this
        // function used to carry.
        let deps = self.shared_deps.clone();
        let permit = self.task_tracker.clone();
        let album = self.album_buffer.clone();
        let task_tracker_for_flush = self.task_tracker.clone();
        let guard_chat = msg.chat.id;
        let guard_thread = msg.thread_id;
        naked_tg::guarded::spawn_guarded(
            self.bot.clone(),
            guard_chat,
            guard_thread,
            "message",
            async move {
                // Inbound coalescing wrapper: runtime owns eligibility
                // (allow/address/active-turn gates), while the coalescer owns
                // the shared debounce/timer machinery for albums and text.
                let deps_for_flush = deps.clone();
                let text_key = text_burst_key_if_eligible(&deps, &msg).await;
                let outcome = if let Some(key) = text_key {
                    let text_debounce =
                        Duration::from_millis(deps.config.telegram.coalesce_text_ms);
                    album
                        .submit_text(msg, key, text_debounce, move |mut msgs| async move {
                            let _permit = task_tracker_for_flush.acquire().await;
                            msgs.sort_by_key(|m| m.id.0);
                            let merged = msgs.len();
                            let mut primary = msgs.remove(0);
                            if should_count_coalesced(merged) {
                                crate::metrics::record_text_coalesced();
                            }
                            let joined = join_text_messages(&primary, &msgs);
                            if let Err(e) = overwrite_message_text(&mut primary, joined) {
                                tracing::error!("text burst synthesis failed: {e}");
                                return;
                            }
                            if let Err(e) = deps_for_flush.handle(primary, Vec::new()).await {
                                tracing::error!("handle_message (text burst) error: {e}");
                            }
                        })
                        .await
                } else {
                    album
                        .submit_album(msg, move |mut msgs| async move {
                            let _permit = task_tracker_for_flush.acquire().await;
                            // Sort by message_id so the user's perceived order
                            // matches the order of images in the agent prompt.
                            msgs.sort_by_key(|m| m.id.0);
                            let primary = msgs.remove(0);
                            if let Err(e) = deps_for_flush.handle(primary, msgs).await {
                                tracing::error!("handle_message (album) error: {e}");
                            }
                        })
                        .await
                };
                if let album::Decision::Solo(msg) = outcome {
                    let _permit = permit.acquire().await;
                    if let Err(e) = deps.handle(*msg, Vec::new()).await {
                        tracing::error!("handle_message error: {e}");
                    }
                }
            },
        );
    }

    async fn dispatch_edited_message(&self, msg_val: &serde_json::Value) {
        // edited_message → treat as a new message (simplest useful behavior).
        // If the original was already processed, the agent sees the edit as
        // follow-up context. If still pending, it appears as a correction.
        let msg: Message = match serde_json::from_value(msg_val.clone()) {
            Ok(m) => m,
            Err(e) => {
                tracing::warn!("Failed to parse edited_message: {e}");
                return;
            }
        };
        let chat_id_raw = msg.chat.id.0;
        let tid_raw = msg.thread_id.map(|t| t.0.0);
        let msg_id = msg.id.0;
        let edit_text = msg.text().or(msg.caption()).unwrap_or_default().to_string();

        // If a turn is actively streaming for this chat, inject as steer edit
        // rather than a new message.
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
            return;
        }

        tracing::info!(
            chat_id = chat_id_raw,
            msg_id,
            "Dispatching edited_message as new message"
        );
        let deps = self.shared_deps.clone();
        let permit = self.task_tracker.clone();
        let guard_chat = msg.chat.id;
        let guard_thread = msg.thread_id;
        naked_tg::guarded::spawn_guarded(
            self.bot.clone(),
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

    async fn dispatch_callback_query(&self, cb_val: &serde_json::Value) {
        tracing::info!("Dispatching callback_query");
        let q: CallbackQuery = match serde_json::from_value(cb_val.clone()) {
            Ok(q) => q,
            Err(e) => {
                tracing::warn!("Failed to parse callback: {e}");
                return;
            }
        };
        let cb_deps = self.shared_deps.clone();
        let cb_pending = self.pending_perms.clone();
        let permit = self.task_tracker.clone();
        let cb_chat = q.message.as_ref().map(|m| m.chat().id).unwrap_or(ChatId(0));
        let cb_thread = q.message.as_ref().and_then(|m| match m {
            teloxide::types::MaybeInaccessibleMessage::Regular(msg) => msg.thread_id,
            _ => None,
        });
        naked_tg::guarded::spawn_guarded(
            self.bot.clone(),
            cb_chat,
            cb_thread,
            "callback",
            async move {
                let _permit = permit.acquire().await;
                if let Err(e) = handle_callback(cb_deps, q, cb_pending).await {
                    tracing::error!("handle_callback error: {e}");
                }
            },
        );
    }

    async fn dispatch_message_reaction(&self, reaction_val: &serde_json::Value) {
        // message_reaction → 👎 removes/cancels queued message hint. Since
        // messages go directly into conversation history, we can't truly remove
        // them. Instead, append a correction note.
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

        if !new_reactions.iter().any(|e| e == "👎") {
            return;
        }

        let tid = reaction_val
            .get("chat")
            .and_then(|c| c.get("message_thread_id"))
            .and_then(|v| v.as_i64())
            .map(|v| v as i32);
        if let Some(sid) = self.channel_map.get(chat_id, tid).await {
            self.agent
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

async fn text_burst_key_if_eligible(deps: &BotDeps, msg: &Message) -> Option<album::TextBurstKey> {
    // Feature gate first: default 0 preserves current Solo behaviour.
    if deps.config.telegram.coalesce_text_ms == 0 {
        return None;
    }

    let text = msg.text()?;
    if text.trim().is_empty() || text.starts_with('/') {
        return None;
    }
    if msg.media_group_id().is_some() || msg.caption().is_some() {
        return None;
    }
    if !crate::extract_media_items(msg).is_empty() {
        return None;
    }

    let chat_id = msg.chat.id.0;
    if !is_allowed(chat_id, &deps.config) {
        return None;
    }
    if !naked_tg::bot_identity::is_addressed_to_bot(msg, &deps.bot_identity) {
        return None;
    }

    let thread_id = msg.thread_id.map(|t| t.0.0);
    let chat_key = (chat_id, thread_id);
    let existing_session_id = deps.channel_map.get(chat_id, thread_id).await;
    if let Some(session_id) = existing_session_id {
        if deps.agent.is_session_active(&session_id).await || steer_sender_exists(chat_key).await {
            return None;
        }
    } else if steer_sender_exists(chat_key).await {
        return None;
    }

    // Sender is mandatory in groups to prevent cross-user merges. Private
    // chats normally have `from`, too; if Telegram omits it, keep current Solo
    // behaviour rather than inventing an unsafe synthetic sender.
    let sender_id = msg.from.as_ref()?.id.0;
    Some(album::TextBurstKey {
        chat_id,
        thread_id,
        sender_id,
        reply_to_message_id: msg.reply_to_message().map(|m| m.id.0),
    })
}

async fn steer_sender_exists(key: (i64, Option<i32>)) -> bool {
    let map = STEER_SENDERS.read().await;
    map.contains_key(&key)
}

fn should_count_coalesced(merged_len: usize) -> bool {
    merged_len >= 2
}

fn join_text_messages(primary: &Message, extras: &[Message]) -> String {
    std::iter::once(primary)
        .chain(extras.iter())
        .filter_map(|m| m.text())
        .map(str::to_string)
        .collect::<Vec<_>>()
        .join("\n")
}

fn overwrite_message_text(msg: &mut Message, text: String) -> Result<(), serde_json::Error> {
    let mut value = serde_json::to_value(&*msg)?;
    if let Some(obj) = value.as_object_mut() {
        obj.insert("text".to_string(), serde_json::Value::String(text));
        // Keep the synthesized message text-only from the handler's point of
        // view; eligibility already rejected captions/media, but removing an
        // accidental caption here makes the invariant explicit.
        obj.remove("caption");
    }
    *msg = serde_json::from_value(value)?;
    Ok(())
}

async fn send_mcp_startup_report(
    bot: &Bot,
    agent: &Arc<AgentCore>,
    config: &Config,
    mcp_failures: &[naked_core::mcp::client::McpConnectFailure],
) {
    // ── A7: Send MCP startup diagnostics to owner chat ────────────────
    if mcp_failures.is_empty() || config.telegram.allowed_chat_ids.is_empty() {
        return;
    }

    let owner_chat = ChatId(config.telegram.allowed_chat_ids[0]);
    let mut lines = vec!["🔌 <b>MCP startup report</b>".to_string()];
    let connected = agent.list_mcp_servers().await;
    for (name, tools) in &connected {
        lines.push(format!("  ✅ <b>{name}</b>: {tools} tools"));
    }
    for f in mcp_failures {
        let err_short: String = f.error.chars().take(120).collect();
        lines.push(format!(
            "  ❌ <b>{}</b>: {}",
            crate::markup::escape_html(&f.name),
            crate::markup::escape_html(&err_short),
        ));
    }
    let mcp_ctx = ChatCtx {
        chat_id: owner_chat,
        thread_id: None,
        reply_to: None,
    };
    let _ = crate::shared::safe_send(
        bot,
        &mcp_ctx,
        lines.join("\n"),
        Some(teloxide::types::ParseMode::Html),
    )
    .await;
}

async fn notify_interrupted_sessions(
    bot: &Bot,
    agent: &Arc<AgentCore>,
    channel_map: &Arc<ChannelSessionMap>,
) {
    // ── FIX-1: Notify ONLY chats with genuinely interrupted sessions ─────
    //
    // drain_interrupted_sessions filters to sessions updated within 5 min.
    // Stale entries (accumulated across SIGKILL restarts) are silently skipped.
    let crashed_sessions = agent
        .drain_interrupted_sessions(chrono::Duration::minutes(5))
        .await;
    if crashed_sessions.is_empty() {
        return;
    }

    let entries = channel_map.all_entries().await;
    // R4 of PLAN_RESILIENCE_v1: track TG-bound vs. research sessions
    // separately. Research sessions don't have a user-facing chat (no entry in
    // channel_map.jsonl) and get resurrected by the scheduler — they don't need
    // a notification.
    let mut notified = 0u32;
    let mut research_orphans = 0u32;
    for session_id in &crashed_sessions {
        let mapping = entries.iter().find(|(_, _, sid)| sid == session_id);
        let Some((chat_id, thread_id_raw, _sid)) = mapping else {
            research_orphans += 1;
            continue;
        };
        let cid = ChatId(*chat_id);
        let tid = if *thread_id_raw != 0 {
            Some(teloxide::types::ThreadId(teloxide::types::MessageId(
                *thread_id_raw as i32,
            )))
        } else {
            None
        };
        let short = &session_id[..session_id.len().min(8)];
        let text =
            format!("⚠️ Бот перезапустился. Последний запрос (сессия {short}…) потерян — повтори.");
        let mut req = bot.send_message(cid, text);
        if let Some(t) = tid {
            req = req.message_thread_id(t);
        }
        match req.await {
            Ok(_) => {
                notified += 1;
                naked_core::types::CRASH_RECOVERY_NOTIFIED_COUNT
                    .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            }
            Err(e) => {
                tracing::info!(
                    chat = chat_id,
                    session = %session_id,
                    "crash recovery notify failed (chat blocked / not found?): {e}"
                );
            }
        }
    }
    tracing::info!(
        total = crashed_sessions.len(),
        notified = notified,
        research_orphans = research_orphans,
        "crash recovery: {} session(s) interrupted, notified {notified} chat(s), {research_orphans} research (will resurrect via scheduler)",
        crashed_sessions.len()
    );
}

fn spawn_health_server_from_env() {
    let health_port: u16 = std::env::var("HEALTH_PORT")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(0);
    if health_port > 0 {
        tokio::spawn(run_health_server(health_port));
    }
}

fn install_ctrl_c_shutdown_handler(shutdown: tokio_util::sync::CancellationToken) {
    tokio::spawn(async move {
        let _ = tokio::signal::ctrl_c().await;
        tracing::info!("Received SIGINT, shutting down gracefully…");
        shutdown.cancel();
    });
}

async fn start_watchdog(
    liveness: Arc<naked_core::liveness::LivenessRegistry>,
    shutdown: tokio_util::sync::CancellationToken,
) -> Option<tokio::task::JoinHandle<()>> {
    // ── systemd watchdog (no-op outside `Type=notify`) ─────────────────
    // Wired after the bot is fully constructed but before the polling loop
    // starts, so READY=1 is sent only when the bot is actually ready to handle
    // work. Liveness arbiter (T2 of PLAN_LIVENESS_v1) gates sd_notify on real
    // polling progress — not just process aliveness.
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

    let watchdog_shutdown = Arc::new(tokio::sync::Notify::new());
    let watchdog_handle = naked_tg::watchdog::spawn_watchdog_with_liveness(
        watchdog_notifier.clone(),
        liveness,
        vec![
            naked_tg::watchdog::LivenessRequirement {
                source: "tg_polling.tick",
                // Long-poll is ~30s + reasonable network slack. If we don't
                // see a tick in 90s, the loop is wedged and systemd should
                // recycle us.
                max_silence: Duration::from_secs(90),
            },
            naked_tg::watchdog::LivenessRequirement {
                source: "scheduler.tick",
                // F2 of PLAN_NEXT_SESSION: scheduler.tick is a 30s cadence by
                // default; allow up to 3x that before declaring it wedged.
                max_silence: Duration::from_secs(120),
            },
        ],
        // Grace covers boot wiring + first long-poll round trip.
        Duration::from_secs(60),
        watchdog_shutdown.clone(),
    );

    tokio::spawn(async move {
        shutdown.cancelled().await;
        watchdog_shutdown.notify_waiters();
        watchdog_notifier.notify_stopping().await;
    });
    watchdog_handle
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

#[cfg(test)]
mod tests {
    use super::should_count_coalesced;

    #[test]
    fn text_coalesced_counter_bumps_once_per_merged_text_flush() {
        assert!(
            !should_count_coalesced(0),
            "empty flushes are never counted"
        );
        assert!(
            !should_count_coalesced(1),
            "single-message flushes are Solo-equivalent"
        );
        assert!(
            should_count_coalesced(2),
            "two messages form one merged burst"
        );
        assert!(
            should_count_coalesced(3),
            "larger text bursts are still one coalesced flush"
        );
    }
}
