mod album;
mod channel_map;
mod media;
mod metrics;

use naked_tg::helpers::parse_interval;
use naked_tg::memory_scheduler;
use naked_tg::research_html::{ReportMeta, render_report_html};
use naked_tg::research_scheduler;
use naked_tg::research_ui::{
    HeartbeatProgress, PendingClarification, keyboard_after_complete,
    keyboard_paused_awaiting_clarification, keyboard_stop, render_waterfall,
};
use naked_tg::tg_markup::{self, MAX_TG_MSG as TG_MSG_LIMIT};

use std::collections::{HashMap, HashSet};
use std::sync::{Arc, LazyLock};
use std::time::Duration;

use anyhow::Result;
use std::collections::VecDeque;
use teloxide::prelude::*;
use teloxide::types::{
    CallbackQuery, InlineKeyboardButton, InlineKeyboardMarkup, MessageId, ParseMode, ThreadId,
};
use tokio::sync::{Mutex, RwLock, oneshot};

use naked_core::AgentCore;
use naked_core::config::Config;
use naked_core::types::{AgentEvent, AgentHandle, Permission, PermissionResponse, TurnUsage};

use channel_map::{ChannelSessionMap, format_tg_channel_id};

const MAX_TG_MSG: usize = TG_MSG_LIMIT;
const EDIT_INTERVAL: Duration = Duration::from_secs(10);
const MIN_EDIT_GAP: Duration = Duration::from_secs(5);
const TYPING_INTERVAL: Duration = Duration::from_secs(3);
const RATE_LIMIT_PER_MIN: usize = 60;
const PERMISSION_TIMEOUT: Duration = Duration::from_secs(120);
const REASONING_TAIL: usize = 600;
const TOOL_WINDOW: usize = 5;
const MAX_THINKING_BYTES: usize = 64_000;
/// Hard cap on the reasoning chain we ship in the *final* TG message
/// (inside `<blockquote expandable>`). Telegram caps a message at 4096
/// chars — we leave generous room for the actual response. Anything
/// over this is tail-truncated; the prefix is dropped because the
/// conclusion / commitments live at the bottom of a CoT.
const MAX_FINAL_THINKING_BYTES: usize = 2_000;
const MAX_RESPONSE_BYTES: usize = 128_000;
const REPLY_QUOTE_MAX_CHARS: usize = 1000;

/// Global rate limiter for Telegram API calls (edit_message_text).
/// Tracks timestamps of recent calls; blocks if over RATE_LIMIT_PER_MIN.
#[derive(Clone)]
pub(crate) struct TgRateLimiter(Arc<Mutex<VecDeque<tokio::time::Instant>>>);

impl TgRateLimiter {
    fn new() -> Self {
        Self(Arc::new(Mutex::new(VecDeque::new())))
    }

    async fn acquire(&self) {
        let window = Duration::from_secs(60);
        let mut delayed_once = false;
        loop {
            let mut q = self.0.lock().await;
            let now = tokio::time::Instant::now();
            while q.front().is_some_and(|&t| now.duration_since(t) >= window) {
                q.pop_front();
            }
            if q.len() < RATE_LIMIT_PER_MIN {
                q.push_back(now);
                return;
            }
            let wait_until = q.front().unwrap().checked_add(window).unwrap();
            drop(q);
            // Only count the first delay per call — otherwise a single
            // blocked caller woken up repeatedly by timer slop would inflate
            // the counter. The semantic is "this send had to wait", not
            // "how many microsleeps it took".
            if !delayed_once {
                crate::metrics::record_rate_limit_delay();
                delayed_once = true;
            }
            tokio::time::sleep_until(wait_until).await;
        }
    }
}

/// Key: call_id → (sender, chat_id, thread_id) so /yolo can drain only matching topic.
pub(crate) type PendingPermissions =
    Arc<RwLock<HashMap<String, (oneshot::Sender<bool>, i64, Option<i32>)>>>;

#[derive(Clone, Copy)]
pub(crate) struct ChatCtx {
    chat_id: ChatId,
    thread_id: Option<ThreadId>,
    /// Message id of the user's triggering message, used to anchor the
    /// bot's placeholder (and any subsequent streamed message) as a
    /// **reply** in group chats. In a busy group, the "waterfall"
    /// placeholder otherwise gets scrolled off-screen by unrelated
    /// chatter and the operator can't see the live tool/thinking
    /// stream. Threading it as a reply keeps the anchor link visible
    /// next to their original message. `None` in private chats (the
    /// UI is already 1:1, reply-threading just adds noise there) and
    /// for callback-driven flows.
    reply_to: Option<MessageId>,
}

impl ChatCtx {
    fn from_msg(msg: &Message) -> Self {
        use teloxide::types::ChatKind;
        let reply_to = matches!(msg.chat.kind, ChatKind::Public(_)).then_some(msg.id);
        Self {
            chat_id: msg.chat.id,
            thread_id: msg.thread_id,
            reply_to,
        }
    }

    fn from_callback(q: &CallbackQuery) -> Self {
        let (chat_id, thread_id) = match &q.message {
            Some(msg) => {
                let cid = msg.chat().id;
                let tid = msg.regular_message().and_then(|m| m.thread_id);
                (cid, tid)
            }
            None => (ChatId(0), None),
        };
        Self {
            chat_id,
            thread_id,
            reply_to: None,
        }
    }

    fn raw_thread_id(&self) -> Option<i32> {
        self.thread_id.map(|tid| tid.0.0)
    }
}

trait SendExt {
    fn maybe_thread(self, thread_id: Option<ThreadId>) -> Self;
    /// Attach a `reply_parameters` pointing at `reply_to` if it is
    /// `Some`. Used to anchor the bot's "⏳ waterfall" placeholder (and
    /// any finalization message) as a reply to the user's triggering
    /// message in group chats — see `ChatCtx::reply_to` for rationale.
    /// `allow_sending_without_reply` is always set so we don't hard-fail
    /// if the user deleted their message mid-flight.
    fn maybe_reply_to(self, reply_to: Option<MessageId>) -> Self;
}

impl SendExt for teloxide::requests::JsonRequest<teloxide::payloads::SendMessage> {
    fn maybe_thread(self, thread_id: Option<ThreadId>) -> Self {
        match thread_id {
            Some(tid) => self.message_thread_id(tid),
            None => self,
        }
    }

    fn maybe_reply_to(self, reply_to: Option<MessageId>) -> Self {
        match reply_to {
            Some(mid) => {
                use teloxide::types::ReplyParameters;
                let params = ReplyParameters::new(mid).allow_sending_without_reply();
                self.reply_parameters(params)
            }
            None => self,
        }
    }
}

impl SendExt for teloxide::requests::MultipartRequest<teloxide::payloads::SendDocument> {
    fn maybe_thread(self, thread_id: Option<ThreadId>) -> Self {
        match thread_id {
            Some(tid) => self.message_thread_id(tid),
            None => self,
        }
    }

    fn maybe_reply_to(self, reply_to: Option<MessageId>) -> Self {
        match reply_to {
            Some(mid) => {
                use teloxide::types::ReplyParameters;
                let params = ReplyParameters::new(mid).allow_sending_without_reply();
                self.reply_parameters(params)
            }
            None => self,
        }
    }
}

async fn send_typing_raw(
    client: &reqwest::Client,
    base: &str,
    chat_id: i64,
    thread_id: Option<i32>,
) {
    let url = format!("{base}/sendChatAction");

    let mut body = serde_json::json!({
        "chat_id": chat_id,
        "action": "typing"
    });
    if let Some(tid) = thread_id {
        body["message_thread_id"] = serde_json::json!(tid);
    }

    match client.post(&url).json(&body).send().await {
        Ok(resp) => {
            if !resp.status().is_success() {
                tracing::debug!(
                    chat_id,
                    ?thread_id,
                    status = %resp.status(),
                    "typing: HTTP error"
                );
            } else if let Ok(json) = resp.json::<serde_json::Value>().await
                && json["ok"].as_bool() != Some(true)
            {
                tracing::warn!(chat_id, ?thread_id, "typing: API error: {}", json);
            }
        }
        Err(e) => tracing::debug!(chat_id, ?thread_id, "typing: network error: {e}"),
    }
}

#[tokio::main]
async fn main() {
    let home = std::env::var("HOME").unwrap_or_else(|_| ".".into());
    let naked_dir = std::path::PathBuf::from(&home).join(".naked");
    let log_dir = naked_dir.join("logs");
    std::fs::create_dir_all(&log_dir).ok();

    // ── pid-lock: kill ALL stale instances before starting ─────────────
    let my_pid = std::process::id();
    let pid_path = naked_dir.join("naked-tg.pid");

    // 1) Kill process from pid file
    if let Ok(old) = std::fs::read_to_string(&pid_path)
        && let Ok(pid) = old.trim().parse::<u32>()
        && pid != my_pid
        && std::path::Path::new(&format!("/proc/{pid}")).exists()
    {
        eprintln!("Killing stale naked-tg pid={pid}");
        let _ = std::process::Command::new("kill")
            .args(["-9", &pid.to_string()])
            .status();
    }
    // 2) Kill any other naked-tg binaries (exact process name match)
    if let Ok(output) = std::process::Command::new("pgrep")
        .args(["-x", "naked-tg"])
        .output()
    {
        for line in String::from_utf8_lossy(&output.stdout).lines() {
            if let Ok(pid) = line.trim().parse::<u32>()
                && pid != my_pid
            {
                eprintln!("Killing extra naked-tg pid={pid}");
                let _ = std::process::Command::new("kill")
                    .args(["-9", &pid.to_string()])
                    .status();
            }
        }
    }

    std::fs::write(&pid_path, my_pid.to_string()).ok();

    // 14-day daily rotation: one file per day, auto-delete anything older
    // than two weeks. Previous setting kept only 3 days which made it
    // painful to investigate incidents that got reported late.
    let file_appender = tracing_appender::rolling::RollingFileAppender::builder()
        .rotation(tracing_appender::rolling::Rotation::DAILY)
        .filename_prefix("naked-tg")
        .filename_suffix("log")
        .max_log_files(14)
        .build(&log_dir)
        .expect("failed to init log appender");
    let (non_blocking_file, _guard) = tracing_appender::non_blocking(file_appender);

    let env_filter = tracing_subscriber::EnvFilter::from_default_env()
        .add_directive("naked=info".parse().unwrap());

    use tracing_subscriber::fmt::writer::MakeWriterExt;
    let combined = std::io::stderr.and(non_blocking_file);

    tracing_subscriber::fmt()
        .with_env_filter(env_filter)
        .with_writer(combined)
        .with_ansi(false)
        .init();

    // Spawn Prometheus /metrics endpoint if NAKED_METRICS_ADDR is set.
    // No-op otherwise; production only opts in explicitly.
    metrics::serve_prometheus_if_enabled();

    let config = Config::load().expect("Failed to load config");
    let provider =
        naked_core::build_provider_from_config(&config).expect("Failed to build provider");
    let agent = Arc::new(AgentCore::new(config.clone(), provider));
    agent.init_self_ref();

    // Register telegram_attach tool — lets the agent send files to chat.
    // The attachment queue is per-turn (created in stream_response), but
    // the tool factory captures a global queue that stream_response swaps.
    let tg_attach_queue: naked_tg::tg_attach::AttachmentQueue = naked_tg::tg_attach::new_queue();
    {
        let q = tg_attach_queue.clone();
        agent
            .register_extra_tool(move || {
                Box::new(naked_tg::tg_attach::TelegramAttachTool::new(q.clone()))
            })
            .await;
    }

    // Cross-process advisory lock guarding `<NAKED_HOME>/research/`.
    // Acquired BEFORE we wire the scheduler so a second `naked-tg`
    // instance pointed at the same NAKED_HOME aborts immediately
    // instead of corrupting `inflight.json` and `runs.jsonl` via
    // append races. Held by binding to `_scheduler_lock` so it lives
    // for the lifetime of the bot process; drop on exit releases it.
    // We keep an `Option` so test or future tooling can run without a
    // research subsystem at all.
    let _scheduler_lock: Option<naked_tg::scheduler_lock::SchedulerLock> = if config
        .research
        .enabled
    {
        let research_root = config
            .research
            .storage_dir
            .clone()
            .unwrap_or_else(naked_core::research::research_root);
        match naked_tg::scheduler_lock::SchedulerLock::try_acquire(&research_root) {
            Ok(lock) => Some(lock),
            Err(naked_tg::scheduler_lock::LockError::Held { path, existing_pid }) => {
                tracing::error!(
                    lock = %path.display(),
                    holder_pid = ?existing_pid,
                    "research scheduler lock is held by another naked-tg process; \
                     refusing to start the scheduler to avoid corrupting state. \
                     Stop the other instance or point NAKED_HOME at a different \
                     directory."
                );
                None
            }
            Err(naked_tg::scheduler_lock::LockError::Io(e)) => {
                tracing::error!(
                    "failed to acquire scheduler lock under {}: {e}; refusing to start scheduler",
                    research_root.display()
                );
                None
            }
        }
    } else {
        None
    };

    if config.research.enabled && _scheduler_lock.is_some() {
        let scheduler_cfg = research_scheduler::SchedulerConfig {
            verify_by_default: config.research.verify_by_default,
            max_verification_rounds: config.research.gatekeeper.max_rounds,
            max_concurrent_runs: config.research.max_concurrent_runs.max(1),
            task_timeout: std::time::Duration::from_secs(config.research.task_timeout_seconds),
            max_retries_before_alert: config.research.max_retries_before_alert,
            ..Default::default()
        };
        let (_scheduler, hook) =
            research_scheduler::ResearchScheduler::start(Arc::downgrade(&agent), scheduler_cfg);
        agent.set_scheduler_hook(hook);
        tracing::info!("research scheduler online");
    }

    // Daily-memory digest scheduler. Runs `memory::daily::run_daily`
    // for the project + every on-disk user scope at the configured
    // cron time. Idempotent — safe to spawn unconditionally; the loop
    // honors `memory.daily_enabled`.
    let _memory_scheduler = memory_scheduler::spawn(
        agent.clone(),
        config.workspace.clone(),
        config.memory.clone(),
    );
    tracing::info!(
        cron = %config.memory.daily_cron,
        enabled = config.memory.daily_enabled,
        "memory scheduler online"
    );

    agent.init_mcp().await;

    let restored = agent.restore_sessions().await.unwrap_or_default();
    if !restored.is_empty() {
        tracing::info!("Restored {} session(s)", restored.len());
    }

    // Open the durable channel-map snapshot. On any failure we fall
    // back to an in-memory map so the bot still starts; restoration via
    // session-meta below remains the authoritative recovery path.
    let channel_map = match ChannelSessionMap::open(&naked_dir).await {
        Ok(m) => Arc::new(m),
        Err(e) => {
            tracing::warn!("channel_map snapshot open failed ({e:#}); using in-memory only");
            Arc::new(ChannelSessionMap::new())
        }
    };

    // Rebuild channel→session mapping from persisted metadata. This is
    // the authoritative path — the JSONL snapshot above is a cache that
    // gets refreshed below once the in-memory state is fully populated.
    let mappings = agent.channel_session_mappings().await;
    let restored_links = channel_map.restore_from(&mappings).await;
    if restored_links > 0 {
        tracing::info!("Restored {restored_links} channel→session link(s)");
    }

    // Restore yolo + allow_list from persisted session configs
    for (channel_id, session_id) in &mappings {
        let sc = agent.load_session_config_pub(session_id);
        let parts: Vec<&str> = channel_id.splitn(3, ':').collect();
        if parts.len() == 3
            && parts[0] == "tg"
            && let (Ok(cid), Ok(raw_tid)) = (parts[1].parse::<i64>(), parts[2].parse::<i64>())
        {
            let tid = if raw_tid == 0 {
                None
            } else {
                Some(raw_tid as i32)
            };
            if let Some(enabled_at) = sc.yolo_enabled_at {
                channel_map.enable_yolo_at(cid, tid, enabled_at).await;
                if channel_map.is_yolo(cid, tid).await {
                    let remaining_h = channel_map.yolo_remaining_secs(cid, tid).await / 3600;
                    tracing::info!(
                        cid,
                        ?tid,
                        remaining_h,
                        "restored yolo for session {} ({remaining_h}h left)",
                        &session_id[..8]
                    );
                } else {
                    tracing::info!(cid, ?tid, "yolo expired for session {}", &session_id[..8]);
                }
            }
            if let Some(tools) = &sc.allow_list {
                for tool in tools {
                    channel_map.allow_add(cid, tid, tool).await;
                }
                if !tools.is_empty() {
                    tracing::info!(
                        cid,
                        ?tid,
                        n = tools.len(),
                        "restored allow-list for session {}",
                        &session_id[..8]
                    );
                }
            }
        }
    }

    // Refresh the durable snapshot to reflect everything we just
    // restored. After this point the bot can crash at any moment and
    // the next start-up will see the same channel→session links even
    // if the session-meta path fails for some reason.
    if let Err(e) = channel_map.flush().await {
        tracing::warn!("channel_map: initial flush failed: {e:#}");
    }

    // Periodic snapshot writer: cheap atomic temp+rename every 30s.
    // 30s is fine because the session-meta path already covers gaps;
    // this is defense-in-depth, not the source of truth.
    {
        let cm = channel_map.clone();
        tokio::spawn(async move {
            let mut tick = tokio::time::interval(std::time::Duration::from_secs(30));
            tick.tick().await; // skip the immediate first tick
            loop {
                tick.tick().await;
                if let Err(e) = cm.flush().await {
                    tracing::warn!("channel_map: periodic flush failed: {e:#}");
                }
            }
        });
    }

    let pending_perms: PendingPermissions = Arc::new(RwLock::new(HashMap::new()));

    let bot_token = config
        .telegram_bot_token
        .clone()
        .or_else(|| std::env::var("TELEGRAM_BOT_TOKEN").ok())
        .expect("telegram_bot_token must be set in config or TELEGRAM_BOT_TOKEN env var");

    let bot = Bot::new(&bot_token);

    register_commands(&bot).await;

    tracing::info!(
        "Starting Telegram bot ({}/{})",
        config.default_provider,
        config.default_model
    );

    // Manual polling loop — avoids teloxide's Dispatcher/Polling which
    // conflicts with stale getUpdates connections from other processes.
    let client = reqwest::Client::builder()
        .timeout(Duration::from_secs(60))
        .build()
        .unwrap();
    let base = format!("https://api.telegram.org/bot{bot_token}");
    let http_client: Arc<reqwest::Client> = Arc::new(client.clone());
    let base_url: Arc<String> = Arc::new(base.clone());
    let bot_token_arc: Arc<String> = Arc::new(bot_token.clone());

    // Resolve our own bot identity (id + @username) so the group-chat
    // gate can tell "this message is for us" apart from "humans
    // chatting with each other while the bot lurks". Privacy mode is
    // OFF for this bot (`can_read_all_group_messages: true`), which
    // means Telegram delivers every group message — without this
    // identity-aware filter the bot would respond to all of them.
    let bot_identity: Arc<naked_tg::bot_identity::BotIdentity> = match client
        .get(format!("{base}/getMe"))
        .send()
        .await
        .and_then(|r| r.error_for_status())
    {
        Ok(resp) => match resp.json::<serde_json::Value>().await {
            Ok(v) => {
                let id = v
                    .get("result")
                    .and_then(|r| r.get("id"))
                    .and_then(|i| i.as_u64())
                    .unwrap_or(0);
                let username = v
                    .get("result")
                    .and_then(|r| r.get("username"))
                    .and_then(|u| u.as_str())
                    .unwrap_or("")
                    .to_string();
                if id == 0 || username.is_empty() {
                    tracing::warn!(
                        "getMe returned malformed payload — group-chat addressing filter \
                         will refuse every group message. Check the bot token."
                    );
                }
                tracing::info!(bot_id = id, %username, "bot identity resolved via getMe");
                Arc::new(naked_tg::bot_identity::BotIdentity { id, username })
            }
            Err(e) => {
                tracing::error!(
                    "getMe parse error: {e} — using zero identity (group filter will reject everything)"
                );
                Arc::new(naked_tg::bot_identity::BotIdentity {
                    id: 0,
                    username: String::new(),
                })
            }
        },
        Err(e) => {
            tracing::error!(
                "getMe request failed: {e} — using zero identity (group filter will reject everything)"
            );
            Arc::new(naked_tg::bot_identity::BotIdentity {
                id: 0,
                username: String::new(),
            })
        }
    };

    // Startup janitor: nuke old media artifacts outside the retention window.
    media::sweep_old_artifacts(&config.workspace, config.tg_media.artifact_retention_days);

    // Drop any pending updates + delete webhook on startup
    let _ = client
        .post(format!("{base}/deleteWebhook"))
        .json(&serde_json::json!({"drop_pending_updates": true}))
        .send()
        .await;
    tracing::info!("Webhook cleared, starting polling loop");

    if config.allowed_chat_ids.is_empty() {
        tracing::warn!(
            "allowed_chat_ids is empty — ALL messages will be rejected! Add your chat IDs to naked.json."
        );
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
    // Wired after the bot is fully constructed (provider, scheduler,
    // Telegram client) but before we enter the polling loop, so the
    // `READY=1` notification is sent only when the bot is actually
    // ready to handle work. Drops the handle on shutdown via the
    // `Notify` we wake from the `ctrl_c` task — see below.
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
    let watchdog_handle = naked_tg::watchdog::spawn_watchdog_ticks(
        watchdog_notifier.clone(),
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
    let rate_limiter = TgRateLimiter::new();
    let album_buffer = album::AlbumBuffer::default();
    // Runtime toggle for `tg_sender_attribution`. Seeded from config; the
    // `/attribution on|off` command flips this atomic without restarting.
    let attribution_flag: Arc<std::sync::atomic::AtomicBool> = Arc::new(
        std::sync::atomic::AtomicBool::new(config.tg_sender_attribution),
    );
    let mut offset: i64 = 0;

    while !shutdown.is_cancelled() {
        let body = serde_json::json!({
            "offset": offset,
            "timeout": 30,
            "allowed_updates": ["message", "edited_message", "callback_query", "message_reaction"]
        });
        tracing::debug!(offset, "polling getUpdates");

        let resp = tokio::select! {
            _ = shutdown.cancelled() => break,
            r = client.post(format!("{base}/getUpdates")).json(&body).send() => {
                match r {
                    Ok(r) => r,
                    Err(e) => {
                        tracing::error!("getUpdates network error: {e}");
                        tokio::time::sleep(Duration::from_secs(3)).await;
                        continue;
                    }
                }
            }
        };

        let payload: serde_json::Value = match resp.json().await {
            Ok(v) => v,
            Err(e) => {
                tracing::error!("getUpdates parse error: {e}");
                tokio::time::sleep(Duration::from_secs(3)).await;
                continue;
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
                    bot_token: bot_token_arc.clone(),
                    bot_identity: bot_identity.clone(),
                    tg_attach_queue: tg_attach_queue.clone(),
                };
                let permit = task_tracker.clone();
                let album = album_buffer.clone();
                let task_tracker_for_flush = task_tracker.clone();
                tokio::spawn(async move {
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
                });
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
                tracing::info!(
                    chat_id = msg.chat.id.0,
                    msg_id = msg.id.0,
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
                    bot_token: bot_token_arc.clone(),
                    bot_identity: bot_identity.clone(),
                    tg_attach_queue: tg_attach_queue.clone(),
                };
                let permit = task_tracker.clone();
                tokio::spawn(async move {
                    let _permit = permit.acquire().await;
                    if let Err(e) = deps.handle(msg, Vec::new()).await {
                        tracing::error!("handle edited_message error: {e}");
                    }
                });
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
                tokio::spawn(async move {
                    let _permit = permit.acquire().await;
                    if let Err(e) =
                        handle_callback(bot, q, pending_perms, agent, channel_map, config).await
                    {
                        tracing::error!("handle_callback error: {e}");
                    }
                });
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

async fn register_commands(bot: &Bot) {
    use teloxide::types::BotCommand;
    let commands = vec![
        BotCommand::new("new", "Start a new session"),
        BotCommand::new("sessions", "List active sessions"),
        BotCommand::new("abort", "Cancel running task"),
        BotCommand::new("provider", "Switch provider"),
        BotCommand::new("model", "Switch model"),
        BotCommand::new("skills", "List loaded skills"),
        BotCommand::new("mcp", "List MCP servers"),
        BotCommand::new("refresh", "Reload skills & MCP"),
        BotCommand::new("reasoning", "Set thinking/reasoning level"),
        BotCommand::new("yolo", "Auto-approve ALL tools in this topic"),
        BotCommand::new("allow", "Manage tool allow-list for this topic"),
        BotCommand::new("memory", "Memory: rules / dreams / drafts / stats"),
        BotCommand::new("research", "Run, list, pause or resume research"),
    ];
    if let Err(e) = bot.set_my_commands(commands).await {
        tracing::warn!("Failed to set bot commands: {e}");
    }
}

// ── Message handler ─────────────────────────────────────────────────────────

/// Extract `@username` (or first_name fallback) from a User-bearing message.
fn sender_label(msg: &Message) -> String {
    msg.from
        .as_ref()
        .and_then(|u| u.username.as_deref().map(|s| format!("@{s}")))
        .or_else(|| msg.from.as_ref().map(|u| u.first_name.clone()))
        .unwrap_or_else(|| "unknown".to_string())
}

/// True when message comes from a group/supergroup/channel (not a 1-on-1 chat).
fn is_group_chat(msg: &Message) -> bool {
    use teloxide::types::ChatKind;
    matches!(msg.chat.kind, ChatKind::Public(_))
}

/// Format a Telegram `reply_to_message` as a blockquote for LLM context.
/// Returns `None` when the message is not a reply.
///
/// Format follows zeroclaws `src/channels/telegram.rs::extract_reply_context`:
///   `> @sender[ marker]:\n> line1\n> line2`
/// with a bot marker `[your previous message]` so the model sees its own
/// cited output clearly in 1-on-1 chats.
fn extract_reply_context(msg: &Message) -> Option<String> {
    let reply = msg.reply_to_message()?;

    let sender = reply
        .from
        .as_ref()
        .and_then(|u| u.username.as_deref().map(|s| format!("@{s}")))
        .or_else(|| reply.from.as_ref().map(|u| u.first_name.clone()))
        .unwrap_or_else(|| "unknown".to_string());

    let is_bot = reply.from.as_ref().map(|u| u.is_bot).unwrap_or(false);
    let marker = if is_bot {
        " [your previous message]"
    } else {
        ""
    };

    // Text messages win; otherwise we describe media and include the user
    // caption if any (e.g. `[Photo: 'ship it']`) so the LLM sees intent.
    let caption = reply.caption().map(|c| c.trim()).filter(|c| !c.is_empty());
    let describe = |kind: &str| match caption {
        Some(c) => format!("[{kind}: {c}]"),
        None => format!("[{kind}]"),
    };
    let body = if let Some(t) = reply.text() {
        t.to_string()
    } else if reply.voice().is_some() {
        describe("Voice message")
    } else if reply.audio().is_some() {
        describe("Audio")
    } else if reply.photo().is_some() {
        describe("Photo")
    } else if reply.document().is_some() {
        describe("Document")
    } else if reply.video().is_some() {
        describe("Video")
    } else if reply.animation().is_some() {
        describe("Animation")
    } else if reply.sticker().is_some() {
        "[Sticker]".to_string()
    } else if let Some(c) = caption {
        format!("[Message: {c}]")
    } else {
        "[Message]".to_string()
    };

    let body = if body.chars().count() > REPLY_QUOTE_MAX_CHARS {
        let truncated: String = body.chars().take(REPLY_QUOTE_MAX_CHARS).collect();
        format!("{truncated}…")
    } else {
        body
    };

    let quoted: String = body
        .lines()
        .map(|l| format!("> {l}"))
        .collect::<Vec<_>>()
        .join("\n");

    Some(format!("> {sender}{marker}:\n{quoted}"))
}

// ── Media extraction & dispatch ─────────────────────────────────────────

/// A single media attachment that we're willing to download and process.
///
/// A Telegram message never contains more than one media "kind" at a time
/// (photo+video are mutually exclusive), but we still return a `Vec` so the
/// caller can decide the policy in one place.
#[derive(Debug, Clone)]
struct MediaItem {
    kind: media::MediaKind,
    file_id: String,
    file_name: String,
    mime_hint: Option<String>,
    duration_secs: Option<u64>,
    emoji: Option<String>,
    /// Raw byte size hint from Telegram metadata (best PhotoSize / file.size).
    /// Used to short-circuit downloads or skip native vision routing **before**
    /// we spend bandwidth pulling the file from `api.telegram.org`.
    size_hint: Option<u32>,
    /// Sticker format. `None` for non-stickers. Animated/video stickers
    /// (`.tgs` / `.webm`) cannot be consumed by vision models, so we skip the
    /// native multimodal path for them.
    sticker_format: Option<StickerFormat>,
}

/// Subset of `teloxide_types::sticker::StickerFormat` we need locally.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum StickerFormat {
    /// Static `.webp` raster — vision-model compatible.
    Static,
    /// Animated `.tgs` (Lottie) — NOT vision-compatible.
    Animated,
    /// Video `.webm` — NOT vision-compatible.
    Video,
}

fn extract_media_items(msg: &Message) -> Vec<MediaItem> {
    let mid = msg.id.0;

    if let Some(v) = msg.voice() {
        return vec![MediaItem {
            kind: media::MediaKind::Voice,
            file_id: v.file.id.clone().to_string(),
            file_name: format!("voice_{mid}.ogg"),
            mime_hint: v.mime_type.as_ref().map(|m| m.to_string()),
            duration_secs: Some(u64::from(v.duration.seconds())),
            emoji: None,
            size_hint: Some(v.file.size),
            sticker_format: None,
        }];
    }
    if let Some(a) = msg.audio() {
        let ext = a
            .mime_type
            .as_ref()
            .and_then(|m| mime_ext(m.essence_str()))
            .unwrap_or("bin");
        let name = a
            .file_name
            .clone()
            .unwrap_or_else(|| format!("audio_{mid}.{ext}"));
        return vec![MediaItem {
            kind: media::MediaKind::Audio,
            file_id: a.file.id.clone().to_string(),
            file_name: name,
            mime_hint: a.mime_type.as_ref().map(|m| m.to_string()),
            duration_secs: Some(u64::from(a.duration.seconds())),
            emoji: None,
            size_hint: Some(a.file.size),
            sticker_format: None,
        }];
    }
    if let Some(photos) = msg.photo()
        && let Some(best) = photos.iter().max_by_key(|p| p.width * p.height)
    {
        return vec![MediaItem {
            kind: media::MediaKind::Photo,
            file_id: best.file.id.clone().to_string(),
            file_name: format!("photo_{mid}.jpg"),
            mime_hint: Some("image/jpeg".to_string()),
            duration_secs: None,
            emoji: None,
            size_hint: Some(best.file.size),
            sticker_format: None,
        }];
    }
    if let Some(v) = msg.video() {
        let name = v
            .file_name
            .clone()
            .unwrap_or_else(|| format!("video_{mid}.mp4"));
        return vec![MediaItem {
            kind: media::MediaKind::Video,
            file_id: v.file.id.clone().to_string(),
            file_name: name,
            mime_hint: v.mime_type.as_ref().map(|m| m.to_string()),
            duration_secs: Some(u64::from(v.duration.seconds())),
            emoji: None,
            size_hint: Some(v.file.size),
            sticker_format: None,
        }];
    }
    if let Some(a) = msg.animation() {
        let name = a
            .file_name
            .clone()
            .unwrap_or_else(|| format!("animation_{mid}.mp4"));
        return vec![MediaItem {
            kind: media::MediaKind::Animation,
            file_id: a.file.id.clone().to_string(),
            file_name: name,
            mime_hint: a.mime_type.as_ref().map(|m| m.to_string()),
            duration_secs: Some(u64::from(a.duration.seconds())),
            emoji: None,
            size_hint: Some(a.file.size),
            sticker_format: None,
        }];
    }
    if let Some(d) = msg.document() {
        let name = d
            .file_name
            .clone()
            .unwrap_or_else(|| format!("document_{mid}.bin"));
        return vec![MediaItem {
            kind: media::MediaKind::Document,
            file_id: d.file.id.clone().to_string(),
            file_name: name,
            mime_hint: d.mime_type.as_ref().map(|m| m.to_string()),
            duration_secs: None,
            emoji: None,
            size_hint: Some(d.file.size),
            sticker_format: None,
        }];
    }
    if let Some(s) = msg.sticker() {
        let format = if s.is_animated() {
            StickerFormat::Animated
        } else if s.is_video() {
            StickerFormat::Video
        } else {
            StickerFormat::Static
        };
        // .tgs is Lottie JSON, .webm is video — vision models can't ingest
        // either. Use the proper extension so the on-disk artifact is sane.
        let ext = match format {
            StickerFormat::Static => "webp",
            StickerFormat::Animated => "tgs",
            StickerFormat::Video => "webm",
        };
        let mime = match format {
            StickerFormat::Static => "image/webp",
            StickerFormat::Animated => "application/x-tgsticker",
            StickerFormat::Video => "video/webm",
        };
        return vec![MediaItem {
            kind: media::MediaKind::Sticker,
            file_id: s.file.id.clone().to_string(),
            file_name: format!("sticker_{mid}.{ext}"),
            mime_hint: Some(mime.to_string()),
            duration_secs: None,
            emoji: s.emoji.clone(),
            size_hint: Some(s.file.size),
            sticker_format: Some(format),
        }];
    }
    Vec::new()
}

fn mime_ext(mime: &str) -> Option<&'static str> {
    match mime {
        "audio/ogg" | "audio/opus" => Some("ogg"),
        "audio/mpeg" | "audio/mp3" => Some("mp3"),
        "audio/x-wav" | "audio/wav" => Some("wav"),
        "audio/flac" | "audio/x-flac" => Some("flac"),
        "audio/mp4" | "audio/m4a" | "audio/x-m4a" => Some("m4a"),
        _ => None,
    }
}

/// Format a short `MM:SS` duration, capped at `99:59`.
fn fmt_duration(secs: u64) -> String {
    let total = secs.min(60 * 99 + 59);
    format!("{:02}:{:02}", total / 60, total % 60)
}

/// Render a polling interval (in seconds) as something a human-readable
/// label suitable for `/research ls` ("every 30m", "every 2h", "every 1d").
fn format_interval(secs: u64) -> String {
    if secs == 0 {
        return "manual".to_string();
    }
    if secs.is_multiple_of(86_400) {
        format!("every {}d", secs / 86_400)
    } else if secs.is_multiple_of(3_600) {
        format!("every {}h", secs / 3_600)
    } else if secs.is_multiple_of(60) {
        format!("every {}m", secs / 60)
    } else {
        format!("every {secs}s")
    }
}

/// Render a "time since" duration in seconds with one unit of precision.
fn format_age(secs: u64) -> String {
    if secs < 60 {
        format!("{secs}s")
    } else if secs < 3_600 {
        format!("{}m", secs / 60)
    } else if secs < 86_400 {
        format!("{}h", secs / 3_600)
    } else {
        format!("{}d", secs / 86_400)
    }
}

/// Outcome of processing a batch of Telegram media items: the text block to
/// prepend to the user prompt, plus any **raw** image bytes that should be
/// passed natively to the main model as image content blocks.
///
/// `native_images` is non-empty only when the caller passed
/// `route_images_natively = true` (i.e. the active model is vision-capable
/// and `tg_media.native_image_context` is on). In all other cases images are
/// described via `tg_media.vision` and the description is folded into `text`.
#[derive(Debug, Default)]
pub struct MediaProcessed {
    pub text: String,
    pub native_images: Vec<NativeImage>,
}

#[derive(Debug, Clone)]
pub struct NativeImage {
    pub mime: String,
    pub bytes: Vec<u8>,
}

/// Process one or more media items → produce the user-facing "media block"
/// that gets prepended to the agent prompt. Errors degrade to `[⚠ … error: …]`
/// and a path-only fallback whenever we've managed to save the file.
#[allow(clippy::too_many_arguments)]
async fn process_media_items(
    items: &[MediaItem],
    bot_token: &str,
    config: &Config,
    http: Arc<reqwest::Client>,
    base_url: Arc<String>,
    user_caption: Option<&str>,
    msg_id: i32,
    route_images_natively: bool,
    native_cap_bytes: u64,
    active_model: &str,
) -> MediaProcessed {
    let mut text_blocks: Vec<String> = Vec::with_capacity(items.len());
    let mut native_images: Vec<NativeImage> = Vec::new();
    for item in items {
        match process_one_media(
            item,
            bot_token,
            config,
            http.clone(),
            base_url.clone(),
            user_caption,
            msg_id,
            route_images_natively,
            native_cap_bytes,
            active_model,
        )
        .await
        {
            Ok(out) => {
                text_blocks.push(out.text);
                native_images.extend(out.native_images);
            }
            Err(e) => {
                tracing::warn!(kind = item.kind.as_str(), error = %e, "media processing failed");
                text_blocks.push(format!("[\u{26A0} {} error: {}]", item.kind.as_str(), e));
            }
        }
    }
    MediaProcessed {
        text: text_blocks.join("\n\n"),
        native_images,
    }
}

/// Pure routing predicate: should this `MediaItem` be sent through the **native**
/// multimodal path (raw image bytes attached to the chat request) or fall back to
/// the legacy describer path (vision-provider summary inlined as text)?
///
/// The native path is taken only when **all** are true:
///   * the caller already decided the model+config support native routing,
///   * the item is a `Photo` or a static `Sticker` (animated `.tgs` and video
///     `.webm` cannot be ingested by Anthropic / OpenAI vision endpoints),
///   * the size hint from Telegram metadata fits under `native_image_max_bytes`
///     (pre-download guard — saves bandwidth and avoids API "image too large"
///     errors at request time).
///
/// Non-photo / non-sticker media (audio, video, files, …) bypass this predicate
/// — the caller's `route_images_natively` flag is forwarded as-is for them so the
/// rest of `process_one_media` can keep its branching logic uniform.
/// Sniff the first few bytes for a recognised image-format magic header.
/// Used as a defence-in-depth check before attaching `bytes` as raw image
/// content to a vision API: if the magic is wrong (truncated download,
/// mis-typed mime, repurposed extension), the upstream API will reject the
/// request with an opaque 400 — instead we downgrade to the describer path,
/// which can at least produce a useful "I cannot decode this image" reply.
///
/// Recognised: JPEG (`FF D8 FF`), PNG (`89 50 4E 47 0D 0A 1A 0A`), GIF
/// (`GIF87a` / `GIF89a`), WebP (`RIFF....WEBP`). Returns `false` for any
/// payload shorter than 12 bytes or whose header doesn't match.
pub(crate) fn looks_like_supported_image(bytes: &[u8]) -> bool {
    if bytes.len() < 12 {
        return false;
    }
    if bytes.starts_with(&[0xFF, 0xD8, 0xFF]) {
        return true;
    }
    if bytes.starts_with(&[0x89, b'P', b'N', b'G', 0x0D, 0x0A, 0x1A, 0x0A]) {
        return true;
    }
    if &bytes[..6] == b"GIF87a" || &bytes[..6] == b"GIF89a" {
        return true;
    }
    if &bytes[..4] == b"RIFF" && &bytes[8..12] == b"WEBP" {
        return true;
    }
    false
}

fn decide_native_route(item: &MediaItem, route_images_natively: bool, native_cap: u32) -> bool {
    if !matches!(
        item.kind,
        media::MediaKind::Photo | media::MediaKind::Sticker
    ) {
        return route_images_natively;
    }
    let static_image = matches!(item.sticker_format, None | Some(StickerFormat::Static));
    let too_big = item.size_hint.is_some_and(|s| s > native_cap);
    route_images_natively && static_image && !too_big
}

#[allow(clippy::too_many_arguments)]
async fn process_one_media(
    item: &MediaItem,
    bot_token: &str,
    config: &Config,
    http: Arc<reqwest::Client>,
    base_url: Arc<String>,
    user_caption: Option<&str>,
    _msg_id: i32,
    route_images_natively: bool,
    native_cap_bytes: u64,
    active_model: &str,
) -> anyhow::Result<MediaProcessed> {
    use media::MediaKind;
    let pre_model_for_err = active_model.to_string();

    // Pre-download routing: if Telegram already told us this image is bigger
    // than the effective native cap (per-provider floor of the global
    // `native_image_max_bytes`), force the legacy (text-describer) path so we
    // don't waste bandwidth pulling a file we'd discard for native routing
    // anyway. The download still happens (we always need the bytes for either
    // the describer or the file artifact), but `route_natively` is downgraded
    // here, which surfaces the right routing decision in logs.
    let native_cap_u32 = u32::try_from(native_cap_bytes).unwrap_or(u32::MAX);
    let route_natively = decide_native_route(item, route_images_natively, native_cap_u32);
    if matches!(
        item.kind,
        media::MediaKind::Photo | media::MediaKind::Sticker
    ) {
        let static_image = matches!(item.sticker_format, None | Some(StickerFormat::Static));
        if route_images_natively && !static_image {
            tracing::info!(
                kind = item.kind.as_str(),
                fmt = ?item.sticker_format,
                "downgrading native route: animated/video stickers not vision-capable"
            );
        }
        if route_images_natively && item.size_hint.is_some_and(|s| s > native_cap_u32) {
            tracing::info!(
                kind = item.kind.as_str(),
                size_hint = item.size_hint,
                native_cap = native_cap_u32,
                "downgrading native route: image larger than native_image_max_bytes"
            );
        }
    }

    let cap = item.kind.cap(&config.tg_media);
    let dl = media::download_to_artifacts(
        http.clone(),
        bot_token,
        &base_url,
        &item.file_id,
        &config.workspace,
        &item.file_name,
        cap,
    )
    .await?;
    let mime = item.mime_hint.as_deref().unwrap_or(&dl.mime);
    let size = dl.bytes.len();
    let rel_path = dl.path.display().to_string();

    match item.kind {
        MediaKind::Voice | MediaKind::Audio => {
            let dur = item
                .duration_secs
                .map(fmt_duration)
                .unwrap_or_else(|| "?".to_string());
            let kind_label = if matches!(item.kind, MediaKind::Voice) {
                "\u{1F3A4} voice"
            } else {
                "\u{1F3B5} audio"
            };
            let header = format!("[{kind_label} {dur}, saved: {rel_path}]");
            let text = match &config.tg_media.audio {
                Some(audio_cfg) => {
                    match media::transcribe_audio(
                        http.clone(),
                        audio_cfg,
                        &dl.bytes,
                        &item.file_name,
                        mime,
                    )
                    .await
                    {
                        Ok(text) => format!("{header}\nTranscript:\n{text}"),
                        Err(e) => {
                            tracing::warn!(error = %e, "transcription failed, falling back to path");
                            format!("{header}\n[\u{26A0} transcription failed: {e}]")
                        }
                    }
                }
                None => {
                    format!("{header}\n[transcription not configured \u{2014} set tg_media.audio]")
                }
            };
            Ok(MediaProcessed {
                text,
                native_images: Vec::new(),
            })
        }
        MediaKind::Photo | MediaKind::Sticker => {
            let kind_label = if matches!(item.kind, MediaKind::Photo) {
                "\u{1F4F8} photo"
            } else {
                "\u{1F5BC} sticker"
            };
            let emoji = item
                .emoji
                .as_deref()
                .map(|e| format!(" {e}"))
                .unwrap_or_default();
            let header = format!("[{kind_label}{emoji}, saved: {rel_path}]");

            // Native path: hand raw bytes to the main model. We still emit a
            // tiny text header so the conversation log is human-readable and
            // the artifact path is preserved for `read_file` retrieval.
            //
            // Animated/video stickers (`.tgs`, `.webm`) are NOT consumable by
            // vision models — Anthropic and OpenAI both reject them — so we
            // force the legacy (text-describer) path for those, which will at
            // least produce a fallback "[sticker not supported]" line instead
            // of an opaque API 4xx.
            let native_cap = native_cap_bytes as usize;
            // Magic-byte sniff: if Telegram tagged the file `image/*` but the
            // payload obviously isn't (truncated / malformed / wrong mime),
            // skip the native path. The describer often produces a useful
            // "I can't read this image" reply, while a vision provider would
            // return an opaque 400.
            let valid_image_magic = looks_like_supported_image(&dl.bytes);
            let native_ok = route_natively
                && size > 0
                && size <= native_cap
                && mime.starts_with("image/")
                && valid_image_magic;
            if route_natively && !valid_image_magic && mime.starts_with("image/") {
                tracing::warn!(
                    bytes = size,
                    mime = %mime,
                    "downgrading native route: payload does not match a known image magic header"
                );
            }
            if native_ok {
                let user_caption_part = user_caption
                    .map(|c| format!("\nUser caption: {c}"))
                    .unwrap_or_default();
                return Ok(MediaProcessed {
                    text: format!("{header}{user_caption_part}"),
                    native_images: vec![NativeImage {
                        mime: mime.to_string(),
                        bytes: dl.bytes.clone(),
                    }],
                });
            }
            // Animated/video stickers: vision providers can't ingest .tgs/.webm
            // either — short-circuit with a clear note instead of pretending
            // to call the describer (which would error out anyway).
            let is_animated = matches!(
                item.sticker_format,
                Some(StickerFormat::Animated) | Some(StickerFormat::Video)
            );
            let text = if is_animated {
                let fmt = match item.sticker_format {
                    Some(StickerFormat::Animated) => "animated (.tgs)",
                    Some(StickerFormat::Video) => "video (.webm)",
                    _ => "non-static",
                };
                format!(
                    "{header}\n[{fmt} sticker \u{2014} vision models cannot describe this format; artifact saved on disk]"
                )
            } else {
                // Legacy text-only path: describe via vision provider, inline the text.
                match &config.tg_media.vision {
                    Some(vision_cfg) => {
                        match media::describe_image(
                            http.clone(),
                            vision_cfg,
                            &dl.bytes,
                            mime,
                            user_caption,
                        )
                        .await
                        {
                            Ok(text) => format!("{header}\nDescription:\n{text}"),
                            Err(e) => {
                                let msg = e.to_string();
                                // Classify the failure so the user can act on it instead
                                // of staring at an opaque 4xx. Only the most common
                                // upstream errors are explicitly handled — the catch-all
                                // arm preserves the raw error tail (truncated) for
                                // debugging.
                                let lc = msg.to_ascii_lowercase();
                                let hint = if lc.contains("429") || lc.contains("rate") {
                                    " (rate-limited \u{2014} the describer provider \
                                     is throttling; consider a different model in \
                                     `tg_media.vision`)"
                                } else if lc.contains("401")
                                    || lc.contains("403")
                                    || lc.contains("unauthorized")
                                {
                                    " (auth rejected \u{2014} check the describer's \
                                     `api_key`)"
                                } else if lc.contains("413")
                                    || lc.contains("too large")
                                    || lc.contains("payload")
                                {
                                    " (image too large for the describer; lower \
                                     `tg_media.limits.photo_max_bytes` or pick a \
                                     model with a higher cap)"
                                } else if lc.contains("timeout") || lc.contains("timed out") {
                                    " (describer timed out; the upstream is slow \
                                     or unreachable)"
                                } else {
                                    ""
                                };
                                tracing::warn!(error = %e, classified = %hint, "vision describer failed");
                                let tail = msg.chars().take(180).collect::<String>();
                                format!(
                                    "{header}\n[\u{26A0} vision describer failed{hint}: {tail}]"
                                )
                            }
                        }
                    }
                    None => {
                        format!(
                            "{header}\n[\u{26A0} vision not configured \u{2014} the active \
                             model `{model}` doesn't accept images natively and no \
                             `tg_media.vision` describer is set; only the file path was \
                             saved. Either switch to a vision-capable model (e.g. \
                             `gpt-4o`, `claude-sonnet-4`, `llama-4-scout`) or set \
                             `tg_media.vision`.]",
                            model = pre_model_for_err.as_str(),
                        )
                    }
                }
            };
            Ok(MediaProcessed {
                text,
                native_images: Vec::new(),
            })
        }
        MediaKind::Document => {
            let header = format!(
                "[\u{1F4C4} document: {} ({} bytes, {mime}), saved: {rel_path}]",
                item.file_name, size
            );
            let text = if media::is_inlineable_doc(
                &item.file_name,
                mime,
                size as u64,
                config.tg_media.docs_inline_max_bytes,
            ) {
                match std::str::from_utf8(&dl.bytes) {
                    Ok(text) => {
                        let safe = media::truncate_for_inline(text, 64 * 1024);
                        format!("{header}\n```\n{safe}\n```")
                    }
                    Err(_) => format!("{header}\n[binary content \u{2014} path only]"),
                }
            } else {
                header
            };
            Ok(MediaProcessed {
                text,
                native_images: Vec::new(),
            })
        }
        MediaKind::Video | MediaKind::Animation => {
            let dur = item
                .duration_secs
                .map(fmt_duration)
                .unwrap_or_else(|| "?".to_string());
            let kind_label = if matches!(item.kind, MediaKind::Video) {
                "\u{1F3AC} video"
            } else {
                "\u{1F3A1} animation"
            };
            Ok(MediaProcessed {
                text: format!("[{kind_label} {dur}, {size} bytes, saved: {rel_path}]"),
                native_images: Vec::new(),
            })
        }
    }
}

async fn send_text(
    bot: &Bot,
    chat_id: ChatId,
    thread_id: Option<ThreadId>,
    text: &str,
) -> Result<(), teloxide::RequestError> {
    bot.send_message(chat_id, text)
        .maybe_thread(thread_id)
        .await?;
    Ok(())
}

/// Immutable-ish bag of every dependency `handle_message` needs.
///
/// Introduced so we don't have to pass 11+ positional args at every call
/// site (which tripped `clippy::too_many_arguments` and made the call
/// sites hard to read). Every field is cheap to clone (`Arc`, `Bot`,
/// small structs) so `BotDeps::clone()` is safe to splatter around.
#[derive(Clone)]
pub(crate) struct BotDeps {
    pub bot: Bot,
    pub agent: Arc<AgentCore>,
    pub channel_map: Arc<ChannelSessionMap>,
    pub config: Config,
    pub pending_perms: PendingPermissions,
    pub http_client: Arc<reqwest::Client>,
    pub base_url: Arc<String>,
    pub rate_limiter: TgRateLimiter,
    pub attribution_flag: Arc<std::sync::atomic::AtomicBool>,
    pub bot_token: Arc<String>,
    pub bot_identity: Arc<naked_tg::bot_identity::BotIdentity>,
    pub tg_attach_queue: naked_tg::tg_attach::AttachmentQueue,
}

impl BotDeps {
    /// Entry point matching the old free-function `handle_message`; exists
    /// so callers can write `deps.handle(msg, extras).await` instead of
    /// unpacking eleven arguments at every dispatch site.
    pub async fn handle(
        self,
        msg: Message,
        extra_album_msgs: Vec<Message>,
    ) -> Result<(), teloxide::RequestError> {
        // Every TG message handle gets its own span with (chat_id, msg_id,
        // thread_id) so operators can follow one conversation through the
        // logs without grep-ing by free-form text. `album_size` counts
        // the current + any extras coalesced from a media-group burst.
        use tracing::Instrument;
        let chat_id = msg.chat.id.0;
        let msg_id = msg.id.0;
        let thread_id = msg.thread_id.map(|t| t.0.0).unwrap_or(0);
        let album_size = 1 + extra_album_msgs.len();
        let span = tracing::info_span!(
            "tg_handle",
            chat = chat_id,
            msg = msg_id,
            thread = thread_id,
            album = album_size,
        );

        async move {
            handle_message(
                self.bot,
                msg,
                self.agent,
                self.channel_map,
                self.config,
                self.pending_perms,
                self.http_client,
                self.base_url,
                self.rate_limiter,
                self.attribution_flag,
                self.bot_token,
                self.bot_identity,
                self.tg_attach_queue,
                extra_album_msgs,
            )
            .await
        }
        .instrument(span)
        .await
    }
}

#[allow(clippy::too_many_arguments)]
async fn handle_message(
    bot: Bot,
    msg: Message,
    agent: Arc<AgentCore>,
    channel_map: Arc<ChannelSessionMap>,
    config: Config,
    pending_perms: PendingPermissions,
    http_client: Arc<reqwest::Client>,
    base_url: Arc<String>,
    rate_limiter: TgRateLimiter,
    attribution_flag: Arc<std::sync::atomic::AtomicBool>,
    bot_token: Arc<String>,
    bot_identity: Arc<naked_tg::bot_identity::BotIdentity>,
    tg_attach_queue: naked_tg::tg_attach::AttachmentQueue,
    extra_album_msgs: Vec<Message>,
) -> Result<(), teloxide::RequestError> {
    let ctx = ChatCtx::from_msg(&msg);
    let chat_id_raw = ctx.chat_id.0;

    // Pass-1: classify incoming content. Either `msg.text()`, or a caption on
    // top of media, or pure media — or literally nothing (service messages).
    let text_direct = msg.text().map(str::to_string).filter(|s| !s.is_empty());
    // Album captions: Telegram only attaches the caption to the **first**
    // photo in the group; for any subsequent message its `.caption()` is
    // empty. Search the whole batch for the first non-empty caption so the
    // user's intent isn't dropped just because the first part happened to
    // be processed without a caption.
    let caption = std::iter::once(&msg)
        .chain(extra_album_msgs.iter())
        .find_map(|m| {
            m.caption()
                .map(str::to_string)
                .filter(|s| !s.trim().is_empty())
        });
    let mut media_items = extract_media_items(&msg);
    for em in &extra_album_msgs {
        media_items.extend(extract_media_items(em));
    }

    // Replying to a message with media == "look at THIS". Pull the
    // attachments from the reply target into the current turn so the
    // agent sees actual bytes (and routes through vision / whisper /
    // artifact saving), not just a `[Photo]` placeholder in the quoted
    // reply context.
    //
    // Telegram's `reply_to_message` always points at a single message
    // id — so for an album we only get the specific photo the user
    // tapped Reply on (siblings in the same `media_group_id` aren't
    // reachable via Bot API). Empirically operators usually tap the
    // one they care about most, and we log what we pulled so it's
    // obvious in the trace.
    if let Some(reply) = msg.reply_to_message() {
        let reply_media = extract_media_items(reply);
        if !reply_media.is_empty() {
            tracing::info!(
                chat_id = chat_id_raw,
                count = reply_media.len(),
                reply_msg_id = reply.id.0,
                "pulled media from reply target into current turn"
            );
            media_items.extend(reply_media);
        }
    }

    if text_direct.is_none() && caption.is_none() && media_items.is_empty() {
        return Ok(());
    }

    // Permission check before spending any time on media processing or
    // touching the agent core.
    if !is_allowed(chat_id_raw, &config) {
        tracing::warn!("Rejected message from chat_id={chat_id_raw}");
        return Ok(());
    }

    // Research clarification intercept. A prior `r:stop:<spec>` callback
    // stashed a `PendingClarification` keyed on (chat, thread); the next
    // non-empty text message becomes a topic update + "Restart" prompt.
    // We intercept before the addressing gate because DMs are the normal
    // research delivery surface and we don't want to force a bot mention
    // in private chats just to reply to an inline button.
    if let Some(text) = text_direct
        .as_deref()
        .map(str::trim)
        .filter(|s| !s.is_empty())
        && !text.starts_with('/')
    {
        let key = (chat_id_raw, ctx.raw_thread_id());
        let maybe_pending = PENDING_CLARIFICATIONS.write().await.remove(&key);
        if let Some(pending) = maybe_pending {
            let spec_id = pending.spec_id;
            let store = agent.research_store();
            let outcome = match store.load_spec(&spec_id).await {
                Ok(mut spec) => {
                    let stamp = chrono::Utc::now().format("%Y-%m-%d %H:%M UTC").to_string();
                    if !spec.topic.trim_end().ends_with('\n') && !spec.topic.is_empty() {
                        spec.topic.push('\n');
                    }
                    spec.topic.push_str(&format!("\nUPDATE {stamp}: {text}"));
                    store.save_spec(&spec).await
                }
                Err(e) => Err(e),
            };
            match outcome {
                Ok(()) => {
                    let body = format!(
                        "✅ Clarification saved to <code>{}</code>.\nTap below to relaunch with the updated topic.",
                        escape_html_min(&spec_id)
                    );
                    let _ = bot
                        .edit_message_text(ctx.chat_id, pending.message_id, body)
                        .parse_mode(ParseMode::Html)
                        .reply_markup(keyboard_paused_awaiting_clarification(&spec_id))
                        .await;
                }
                Err(e) => {
                    let body = format!(
                        "⚠️ Could not save clarification for <code>{}</code>: {}",
                        escape_html_min(&spec_id),
                        escape_html_min(&format!("{e:#}"))
                    );
                    let _ = bot
                        .send_message(ctx.chat_id, body)
                        .parse_mode(ParseMode::Html)
                        .maybe_thread(ctx.thread_id)
                        .await;
                }
            }
            return Ok(());
        }
    }

    // Group-chat addressing gate. Privacy mode is OFF for this bot
    // (`can_read_all_group_messages: true` from getMe), so Telegram
    // delivers every message in every group the bot has joined. We
    // **read** all of them (logged via the `tg_handle` span above so
    // the LLM-side memory loop can pick them up later if desired) but
    // only **respond** when the message is explicitly addressed to us
    // — see `naked_tg::bot_identity::is_addressed_to_bot` for the
    // exact rules. Private chats always pass this gate.
    if !naked_tg::bot_identity::is_addressed_to_bot(&msg, &bot_identity) {
        // INFO-level on purpose: in groups with privacy-mode OFF this
        // is the only way to confirm "yes, we saw the message, and we
        // intentionally chose not to respond". The volume is bounded
        // by group activity (the bot is silent in DMs — those bypass
        // the gate). If this becomes too chatty in a high-traffic
        // group, demote to debug! and add a counter metric instead.
        tracing::info!(
            chat_id = chat_id_raw,
            addressed = false,
            "group msg not addressed to bot — read but not answered"
        );
        return Ok(());
    }

    // Peek the existing session (without creating one) so we know the active
    // model and can decide whether photos go through the **native multimodal**
    // path (raw bytes → main vision-capable model) or the legacy text-only
    // path (vision provider → text description → main model). We deliberately
    // do NOT call `get_or_create_session` here — that would spawn an empty
    // session for unrecognized commands like `/help`, `/clear`, `/new` sent as
    // the first message in a fresh chat.
    let chat_id_for_peek = ctx.chat_id.0;
    let tid_for_peek = ctx.raw_thread_id();
    let existing_session_id = channel_map.get(chat_id_for_peek, tid_for_peek).await;
    let (pre_prov, pre_model) = match existing_session_id.as_deref() {
        Some(sid) => agent.session_provider_model(sid).await,
        // No session yet — use the global default; this is read-only and never
        // creates session directories on disk.
        None => agent.default_provider_model(),
    };
    let pre_provider_cfg = config.providers.get(&pre_prov);
    let route_images_natively = config.tg_media.native_image_context
        && config
            .tg_media
            .is_vision_capable_with_provider(&pre_model, pre_provider_cfg);
    // Per-provider effective image cap = floor(global cap, provider hard limit).
    let (provider_type, provider_base_url): (String, Option<String>) = match pre_provider_cfg {
        Some(pc) => (pc.provider_type.clone(), pc.base_url.clone()),
        None => (String::new(), None),
    };
    let native_cap_bytes = config
        .tg_media
        .provider_image_cap(&provider_type, provider_base_url.as_deref());
    if !media_items.is_empty() {
        let has_oversize = media_items
            .iter()
            .any(|i| i.size_hint.is_some_and(|s| u64::from(s) > native_cap_bytes));
        tracing::info!(
            model = %pre_model,
            provider = %pre_prov,
            native = route_images_natively,
            native_cap_bytes,
            count = media_items.len(),
            "media routing decision"
        );
        crate::metrics::record_media_routing(
            route_images_natively,
            has_oversize,
            !route_images_natively && config.tg_media.vision.is_some(),
        );
    }

    // Pass-2: if there's media, acknowledge and process it (download + transform).
    let media_processed = if !media_items.is_empty() {
        let _ = send_text(
            &bot,
            ctx.chat_id,
            ctx.thread_id,
            "\u{1F4E5} processing media\u{2026}",
        )
        .await;
        Some(
            process_media_items(
                &media_items,
                &bot_token,
                &config,
                http_client.clone(),
                base_url.clone(),
                caption.as_deref(),
                msg.id.0,
                route_images_natively,
                native_cap_bytes,
                &pre_model,
            )
            .await,
        )
    } else {
        None
    };

    // Pass-3: merge media_block + text into a single user prompt.
    let (media_text, native_images) = match media_processed {
        Some(m) => (Some(m.text), m.native_images),
        None => (None, Vec::new()),
    };
    let base_text = {
        let mut parts: Vec<String> = Vec::new();
        if let Some(block) = media_text {
            parts.push(block);
        }
        if let Some(t) = text_direct.as_ref() {
            parts.push(t.clone());
        } else if let Some(c) = caption.as_ref() {
            parts.push(c.clone());
        }
        parts.join("\n\n")
    };
    if base_text.trim().is_empty() && native_images.is_empty() {
        return Ok(());
    }

    let in_group = is_group_chat(&msg);
    let sender = sender_label(&msg);
    let has_reply = msg.reply_to_message().is_some();
    tracing::info!(
        chat_id = chat_id_raw,
        thread_id = ?ctx.thread_id,
        raw_thread = ?ctx.raw_thread_id(),
        sender = %sender,
        is_group = in_group,
        has_reply,
        media_count = media_items.len(),
        "handle_message: ctx"
    );

    // Compose the final text for the agent: optional reply quote + optional
    // `@sender:` attribution prefix (groups only). Commands skip composition
    // so `/clear`, `/new`, etc. still work when sent as a reply.
    let text = if base_text.starts_with('/') {
        base_text.clone()
    } else {
        let quote = extract_reply_context(&msg);
        let attr_enabled = attribution_flag.load(std::sync::atomic::Ordering::Relaxed);
        let need_attr = attr_enabled && in_group;
        let attributed = if need_attr {
            format!("{sender}: {base_text}")
        } else {
            base_text.clone()
        };
        match quote {
            Some(q) => format!("{q}\n\n{attributed}"),
            None => attributed,
        }
    };

    if text.starts_with('/') {
        // Persona chats with `allow_slash_commands=false` are pure
        // natural-language conversations — drop any `/cmd` here before
        // dispatch and (on the first hit per chat) drop a one-line hint
        // so the operator knows the silence is intentional. See
        // `Config.chat_personas` in `naked-core::config` for the contract.
        if drop_slash_for_persona(&bot, chat_id_raw, &msg, &config).await {
            return Ok(());
        }
        // `/start@zGsR_bot args` → `/start args` so command parsing
        // doesn't have to know about the @-suffix Telegram appends in
        // groups. `/start@OtherBot` is already filtered out by the
        // addressing gate above (returned as not-addressed), so any
        // `@bot` suffix that survives to this point either targets
        // us or doesn't exist at all.
        let canonical = naked_tg::bot_identity::strip_bot_command_suffix(&text, &bot_identity)
            .unwrap_or_else(|| text.clone());
        let handled = handle_command(
            &bot,
            &msg,
            &canonical,
            &agent,
            &channel_map,
            &config,
            ctx,
            &pending_perms,
            &attribution_flag,
        )
        .await?;
        if handled {
            return Ok(());
        }
        // Unrecognized /command — fall through to agent as regular message
    }

    // Now we know the message is going to the agent — only now do we create
    // a session if one didn't exist yet.
    let session_id = match existing_session_id {
        Some(sid) => sid,
        None => get_or_create_session(ctx, &agent, &channel_map, &config).await,
    };

    // Build optional native-multimodal blocks. When present, these go through
    // `send_prompt_multimodal` / `queue_message_multimodal`; otherwise we fall
    // back to the text-only entry points.
    //
    // Block ordering: images FIRST, text LAST. Anthropic's vision docs
    // explicitly recommend placing image blocks before text for best response
    // quality; OpenAI-compatible vision models accept either order. See
    // https://docs.anthropic.com/en/docs/build-with-claude/vision
    let multimodal_blocks: Option<Vec<naked_core::types::ContentBlock>> =
        if !native_images.is_empty() {
            let mut blocks: Vec<naked_core::types::ContentBlock> = Vec::new();
            for img in &native_images {
                use base64::Engine as _;
                let b64 = base64::engine::general_purpose::STANDARD.encode(&img.bytes);
                blocks.push(naked_core::types::ContentBlock::Image {
                    mime: img.mime.clone(),
                    data_base64: b64,
                    detail: config.tg_media.image_detail,
                });
            }
            if !text.is_empty() {
                blocks.push(naked_core::types::ContentBlock::Text { text: text.clone() });
            }
            Some(blocks)
        } else {
            None
        };

    if agent.is_session_active(&session_id).await {
        match multimodal_blocks {
            Some(blocks) => agent.queue_message_multimodal(&session_id, blocks).await,
            None => agent.queue_message(&session_id, &text).await,
        }
        // Increment queued message counter for status preview.
        {
            let key = (ctx.chat_id.0, ctx.raw_thread_id());
            let map = QUEUE_COUNTS.read().await;
            if let Some(counter) = map.get(&key) {
                counter.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            }
        }
        let qcount = {
            let key = (ctx.chat_id.0, ctx.raw_thread_id());
            let map = QUEUE_COUNTS.read().await;
            map.get(&key)
                .map(|c| c.load(std::sync::atomic::Ordering::Relaxed))
                .unwrap_or(0)
        };
        let note = if qcount > 0 {
            format!("⏳ +{qcount} in queue")
        } else {
            "⏳ Queued".into()
        };
        bot.send_message(ctx.chat_id, note)
            .maybe_thread(ctx.thread_id)
            .maybe_reply_to(ctx.reply_to)
            .await?;
        return Ok(());
    }

    // Reuse the (provider, model) we resolved earlier for the multimodal
    // routing decision — for both existing and freshly-created sessions the
    // result is identical, so calling `session_provider_model` again would be
    // a redundant lock acquisition.
    let (prov, model) = (pre_prov, pre_model);
    let model_tag = format!("{prov}/{model}");

    // Record the current author for this turn so the memory tool can resolve
    // `scope=user` without extra parameters. Use the raw numeric id (stable
    // across username changes).
    let sender_id = msg.from.as_ref().map(|u| u.id.0.to_string());
    agent.set_session_sender(&session_id, sender_id).await;

    let send_result = match multimodal_blocks {
        Some(blocks) => {
            agent
                .send_prompt_multimodal(&session_id, blocks, text.clone())
                .await
        }
        None => agent.send_prompt(&session_id, &text).await,
    };
    let handle = match send_result {
        Ok(h) => h,
        Err(e) => {
            bot.send_message(ctx.chat_id, format!("Error: {e}"))
                .maybe_thread(ctx.thread_id)
                .maybe_reply_to(ctx.reply_to)
                .await?;
            return Ok(());
        }
    };

    stream_response(
        bot,
        ctx,
        handle,
        &channel_map,
        &pending_perms,
        model_tag,
        &http_client,
        &base_url,
        &tg_attach_queue,
        &rate_limiter,
    )
    .await;

    Ok(())
}

// ── Callback handler (permissions) ──────────────────────────────────────────

async fn handle_callback(
    bot: Bot,
    q: CallbackQuery,
    pending_perms: PendingPermissions,
    agent: Arc<AgentCore>,
    channel_map: Arc<ChannelSessionMap>,
    config: Config,
) -> Result<(), teloxide::RequestError> {
    let data = match &q.data {
        Some(d) => d.clone(),
        None => return Ok(()),
    };

    let parts: Vec<&str> = data.splitn(3, ':').collect();
    let prefix = parts.first().copied().unwrap_or("");
    tracing::info!(callback_data = %data, prefix, "handle_callback");

    match prefix {
        "p" if parts.len() == 3 => {
            let call_id = parts[1].to_string();
            let action = parts[2];
            if action == "yolo" {
                let cb_ctx = ChatCtx::from_callback(&q);
                let cid = cb_ctx.chat_id.0;
                let tid = cb_ctx.raw_thread_id();
                tracing::info!(cid, ?tid, "yolo callback: enabling");
                channel_map.enable_yolo(cid, tid).await;
                let yolo_ts = std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap_or_default()
                    .as_secs() as i64;
                if let Some(sid) = channel_map.get(cid, tid).await
                    && let Err(e) = agent.set_session_yolo(&sid, Some(yolo_ts)).await
                {
                    tracing::warn!("failed to persist yolo: {e}");
                }
                let mut perms = pending_perms.write().await;
                let matching: Vec<String> = perms
                    .iter()
                    .filter(|(_, (_, c, t))| *c == cid && *t == tid)
                    .map(|(k, _)| k.clone())
                    .collect();
                let n = matching.len();
                for key in matching {
                    if let Some((tx, _, _)) = perms.remove(&key) {
                        let _ = tx.send(true);
                    }
                }
                drop(perms);
                if let Some(msg) = &q.message
                    && let Some(regular) = msg.regular_message()
                {
                    let _ = bot.delete_message(regular.chat.id, regular.id).await;
                }
                let remaining_h = channel_map.yolo_remaining_secs(cid, tid).await / 3600;
                bot.answer_callback_query(q.id.clone())
                    .text(format!("⚡ YOLO ON ({remaining_h}h) — {n} approved"))
                    .await?;
            } else {
                let allowed = action == "allow";
                let sender = pending_perms.write().await.remove(&call_id);
                if let Some((tx, _, _)) = sender {
                    let _ = tx.send(allowed);
                }
                let label = if allowed { "✅" } else { "❌ Denied" };
                if let Some(msg) = &q.message
                    && let Some(regular) = msg.regular_message()
                {
                    let _ = bot.delete_message(regular.chat.id, regular.id).await;
                }
                bot.answer_callback_query(q.id.clone()).text(label).await?;
            }
        }
        "sp" if parts.len() >= 2 => {
            let provider_name = parts[1..].join(":");
            let cb_ctx = ChatCtx::from_callback(&q);
            let sid = get_or_create_session(cb_ctx, &agent, &channel_map, &config).await;
            match agent
                .set_session_provider(&sid, Some(&provider_name), None)
                .await
            {
                Ok(()) => {
                    let (prov, model) = agent.session_provider_model(&sid).await;
                    let models = provider_models(&config, &prov);
                    if models.is_empty() {
                        if let Some(msg) = &q.message
                            && let Some(regular) = msg.regular_message()
                        {
                            let _ = bot
                                .edit_message_text(
                                    regular.chat.id,
                                    regular.id,
                                    format!("✅ {prov}/{model}"),
                                )
                                .await;
                        }
                    } else {
                        let rows: Vec<Vec<InlineKeyboardButton>> = models
                            .iter()
                            .map(|m| {
                                let mark = if *m == model { " ✅" } else { "" };
                                vec![InlineKeyboardButton::callback(
                                    format!("{m}{mark}"),
                                    format!("sm:{m}"),
                                )]
                            })
                            .collect();
                        let kb = InlineKeyboardMarkup::new(rows);
                        if let Some(msg) = &q.message
                            && let Some(regular) = msg.regular_message()
                        {
                            let _ = bot
                                .edit_message_text(
                                    regular.chat.id,
                                    regular.id,
                                    format!("✅ <b>{prov}</b>\nSelect model:"),
                                )
                                .parse_mode(ParseMode::Html)
                                .reply_markup(kb)
                                .await;
                        }
                    }
                    bot.answer_callback_query(q.id.clone())
                        .text(format!("Provider: {prov}"))
                        .await?;
                }
                Err(e) => {
                    bot.answer_callback_query(q.id.clone())
                        .text(format!("Error: {e}"))
                        .await?;
                }
            }
        }
        "sm" if parts.len() >= 2 => {
            let model_name = parts[1..].join(":");
            let cb_ctx = ChatCtx::from_callback(&q);
            let sid = get_or_create_session(cb_ctx, &agent, &channel_map, &config).await;
            match agent
                .set_session_provider(&sid, None, Some(&model_name))
                .await
            {
                Ok(()) => {
                    let (prov, model) = agent.session_provider_model(&sid).await;
                    if let Some(msg) = &q.message
                        && let Some(regular) = msg.regular_message()
                    {
                        let _ = bot
                            .edit_message_text(
                                regular.chat.id,
                                regular.id,
                                format!("✅ <b>{prov}</b> / <b>{model}</b>"),
                            )
                            .parse_mode(ParseMode::Html)
                            .await;
                    }
                    // If agent is busy on this chat, trigger in-flight
                    // model switch: abort current turn and re-dispatch.
                    let chat_key = (cb_ctx.chat_id.0, cb_ctx.raw_thread_id());
                    if agent.is_session_active(&sid).await {
                        let switch = naked_tg::model_switch::PendingSwitch {
                            provider: Some(prov.clone()),
                            model: model.clone(),
                            continuation: None,
                        };
                        let map = MODEL_SWITCHES.read().await;
                        if let Some(ms) = map.get(&chat_key) {
                            let continuation = naked_tg::model_switch::build_continuation(&switch);
                            ms.lock().await.request(switch);
                            agent.abort(&sid).await;
                            // Queue continuation so the agent picks up
                            // where it left off with the new model.
                            agent.queue_message(&sid, &continuation).await;
                            bot.answer_callback_query(q.id.clone())
                                .text(format!("⚡ Switching to {model}…"))
                                .await?;
                        } else {
                            drop(map);
                            bot.answer_callback_query(q.id.clone())
                                .text(format!("Model: {model} (next turn)"))
                                .await?;
                        }
                    } else {
                        bot.answer_callback_query(q.id.clone())
                            .text(format!("Model: {model}"))
                            .await?;
                    }
                }
                Err(e) => {
                    bot.answer_callback_query(q.id.clone())
                        .text(format!("Error: {e}"))
                        .await?;
                }
            }
        }
        "sr" if parts.len() >= 2 => {
            let level = parts[1];
            let cb_ctx = ChatCtx::from_callback(&q);
            let sid = get_or_create_session(cb_ctx, &agent, &channel_map, &config).await;
            match agent.set_session_reasoning(&sid, level).await {
                Ok(()) => {
                    let label = if level == "off" { "off" } else { level };
                    if let Some(msg) = &q.message
                        && let Some(regular) = msg.regular_message()
                    {
                        let _ = bot
                            .edit_message_text(
                                regular.chat.id,
                                regular.id,
                                format!("💭 Reasoning: <b>{label}</b>"),
                            )
                            .parse_mode(ParseMode::Html)
                            .await;
                    }
                    bot.answer_callback_query(q.id.clone())
                        .text(format!("Reasoning: {label}"))
                        .await?;
                }
                Err(e) => {
                    bot.answer_callback_query(q.id.clone())
                        .text(format!("Error: {e}"))
                        .await?;
                }
            }
        }
        "r" if parts.len() >= 3 => {
            let action = parts[1];
            let spec_id = parts[2].to_string();
            match action {
                "stop" => {
                    let cancelled = agent.cancel_research_run(&spec_id).await;
                    if let Some(msg) = &q.message
                        && let Some(regular) = msg.regular_message()
                    {
                        let cid_raw = regular.chat.id.0;
                        let tid_raw = regular.thread_id.map(|t| t.0.0);
                        let note = if cancelled {
                            format!(
                                "⏸ Paused <code>{}</code>\n\nReply with a clarification — the next message in this chat will be appended to the spec's topic and a Restart button will appear.",
                                escape_html_min(&spec_id)
                            )
                        } else {
                            format!(
                                "ℹ️ Run <code>{}</code> already finished.\nYou can still type a note and press Restart on the final message.",
                                escape_html_min(&spec_id)
                            )
                        };
                        let _ = bot
                            .edit_message_text(regular.chat.id, regular.id, note)
                            .parse_mode(ParseMode::Html)
                            .reply_markup(keyboard_paused_awaiting_clarification(&spec_id))
                            .await;
                        PENDING_CLARIFICATIONS.write().await.insert(
                            (cid_raw, tid_raw),
                            PendingClarification {
                                spec_id: spec_id.clone(),
                                message_id: regular.id,
                                paused_at: chrono::Utc::now(),
                            },
                        );
                    }
                    bot.answer_callback_query(q.id.clone())
                        .text(if cancelled { "Paused" } else { "Already done" })
                        .await?;
                }
                "restart" => {
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
                    let cid_raw = chat_id.0;
                    let tid_raw = thread_id.map(|t| t.0.0);
                    PENDING_CLARIFICATIONS
                        .write()
                        .await
                        .remove(&(cid_raw, tid_raw));
                    bot.answer_callback_query(q.id.clone())
                        .text("Restarting…")
                        .await?;
                    let reply = launch_research_run_with_ui(
                        bot.clone(),
                        agent.clone(),
                        config.clone(),
                        chat_id,
                        thread_id,
                        spec_id,
                    )
                    .await;
                    if !reply.is_empty() {
                        let _ = bot
                            .send_message(chat_id, reply)
                            .maybe_thread(thread_id)
                            .await;
                    }
                }
                _ => {
                    bot.answer_callback_query(q.id.clone())
                        .text(format!("unknown research action: {action}"))
                        .await?;
                }
            }
        }
        _ => {
            bot.answer_callback_query(q.id.clone()).await?;
        }
    }
    Ok(())
}

pub(crate) fn provider_models(config: &Config, provider_name: &str) -> Vec<String> {
    config
        .providers
        .get(provider_name)
        .map(|pc| pc.models_with_aliases())
        .unwrap_or_default()
}

// ── Streaming (extracted to streaming.rs) ────────────────────────────────────
mod streaming;
use streaming::{send_long_text, stream_response};

// ── Access control ──────────────────────────────────────────────────────────

fn is_allowed(chat_id: i64, config: &Config) -> bool {
    if config.allowed_chat_ids.is_empty() {
        return false;
    }
    config.allowed_chat_ids.contains(&chat_id)
}

// ── Commands (extracted to commands.rs) ──────────────────────────────────────

static SLASH_HINT_SHOWN: LazyLock<tokio::sync::RwLock<HashSet<i64>>> =
    LazyLock::new(|| tokio::sync::RwLock::new(HashSet::new()));
type PendingClarificationMap = HashMap<(i64, Option<i32>), PendingClarification>;
static PENDING_CLARIFICATIONS: LazyLock<tokio::sync::RwLock<PendingClarificationMap>> =
    LazyLock::new(|| tokio::sync::RwLock::new(HashMap::new()));
type ModelSwitchMap = HashMap<(i64, Option<i32>), naked_tg::model_switch::SharedModelSwitch>;
static MODEL_SWITCHES: LazyLock<tokio::sync::RwLock<ModelSwitchMap>> =
    LazyLock::new(|| tokio::sync::RwLock::new(HashMap::new()));
type QueueCountMap = HashMap<(i64, Option<i32>), Arc<std::sync::atomic::AtomicUsize>>;
static QUEUE_COUNTS: LazyLock<tokio::sync::RwLock<QueueCountMap>> =
    LazyLock::new(|| tokio::sync::RwLock::new(HashMap::new()));

mod commands;
use commands::{
    drop_slash_for_persona, escape_html_min, get_or_create_session, handle_command,
    launch_research_run_with_ui,
};

// ── Formatting helpers ──────────────────────────────────────────────────────

// ── Telegram markup: delegate to tg_markup module (DRY) ─────────────────

pub(crate) fn escape_html(s: &str) -> String {
    tg_markup::escape_html(s)
}

fn md_to_tg_html(text: &str) -> String {
    tg_markup::md_to_tg_html(text)
}

fn split_html(text: &str, max_bytes: usize) -> Vec<String> {
    tg_markup::split_html(text, max_bytes)
}

// ── Health check server ─────────────────────────────────────────────────────

async fn run_health_server(port: u16) {
    use tokio::io::AsyncWriteExt;
    let listener = match tokio::net::TcpListener::bind(("0.0.0.0", port)).await {
        Ok(l) => {
            tracing::info!("Health endpoint listening on :{port}");
            l
        }
        Err(e) => {
            tracing::warn!("Failed to bind health port {port}: {e}");
            return;
        }
    };
    loop {
        if let Ok((mut stream, _)) = listener.accept().await {
            let body = r#"{"status":"ok"}"#;
            let response = format!(
                "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\r\n{}",
                body.len(),
                body
            );
            let _ = stream.write_all(response.as_bytes()).await;
        }
    }
}
