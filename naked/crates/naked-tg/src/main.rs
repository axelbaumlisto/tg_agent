mod album;
mod channel_map;
mod media;
mod metrics;

use naked_tg::helpers::parse_interval;
use naked_tg::memory_scheduler;
use naked_tg::research_html::{ReportMeta, render_report_html};
use naked_tg::research_scheduler;
use naked_tg::research_ui::{
    HeartbeatProgress, PendingClarification, keyboard_after_complete, keyboard_paused_awaiting_clarification,
    keyboard_stop, render_waterfall,
};

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

const MAX_TG_MSG: usize = 4096;
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
struct TgRateLimiter(Arc<Mutex<VecDeque<tokio::time::Instant>>>);

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
type PendingPermissions = Arc<RwLock<HashMap<String, (oneshot::Sender<bool>, i64, Option<i32>)>>>;

#[derive(Clone, Copy)]
struct ChatCtx {
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

    // Cross-process advisory lock guarding `<NAKED_HOME>/research/`.
    // Acquired BEFORE we wire the scheduler so a second `naked-tg`
    // instance pointed at the same NAKED_HOME aborts immediately
    // instead of corrupting `inflight.json` and `runs.jsonl` via
    // append races. Held by binding to `_scheduler_lock` so it lives
    // for the lifetime of the bot process; drop on exit releases it.
    // We keep an `Option` so test or future tooling can run without a
    // research subsystem at all.
    let _scheduler_lock: Option<naked_tg::scheduler_lock::SchedulerLock> = if config.research.enabled
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
            task_timeout: std::time::Duration::from_secs(
                config.research.task_timeout_seconds,
            ),
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
            tracing::warn!(
                "channel_map snapshot open failed ({e:#}); using in-memory only"
            );
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
                tracing::error!("getMe parse error: {e} — using zero identity (group filter will reject everything)");
                Arc::new(naked_tg::bot_identity::BotIdentity {
                    id: 0,
                    username: String::new(),
                })
            }
        },
        Err(e) => {
            tracing::error!("getMe request failed: {e} — using zero identity (group filter will reject everything)");
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
                let interval_secs =
                    wd.interval().map(|d| d.as_secs()).unwrap_or(0);
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
            "allowed_updates": ["message", "callback_query"]
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
    if let Some(text) = text_direct.as_deref().map(str::trim).filter(|s| !s.is_empty())
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
                    spec.topic
                        .push_str(&format!("\nUPDATE {stamp}: {text}"));
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
        // `/start@example_bot args` → `/start args` so command parsing
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
        bot.send_message(
            ctx.chat_id,
            "⏳ Сообщение добавлено в очередь — дождись завершения текущей задачи.",
        )
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
                    bot.answer_callback_query(q.id.clone())
                        .text(format!("Model: {model}"))
                        .await?;
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
                    bot.answer_callback_query(q.id.clone()).text("Restarting…").await?;
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

fn provider_models(config: &Config, provider_name: &str) -> Vec<String> {
    config
        .providers
        .get(provider_name)
        .map(|pc| pc.models_with_aliases())
        .unwrap_or_default()
}

// ── Streaming response ──────────────────────────────────────────────────────

struct SubAgentState {
    prompt: String,
    status: &'static str,
    last_tool: Option<String>,
    tool_count: u32,
}

struct CompositeView {
    thinking: String,
    in_thinking: bool,
    tool_lines: Vec<String>,
    sub_agents: std::collections::HashMap<String, SubAgentState>,
    sub_agent_order: Vec<String>,
    response_text: String,
    usage: Option<TurnUsage>,
    model_tag: String,
    tick: usize,
    phase: &'static str,
    started_at: std::time::Instant,
}

const SPINNER: &[&str] = &["⏳", "⌛", "⏳", "⌛"];

impl CompositeView {
    fn new(model_tag: String) -> Self {
        Self {
            thinking: String::new(),
            in_thinking: false,
            tool_lines: Vec::new(),
            sub_agents: std::collections::HashMap::new(),
            sub_agent_order: Vec::new(),
            response_text: String::new(),
            usage: None,
            model_tag,
            tick: 0,
            phase: "thinking",
            started_at: std::time::Instant::now(),
        }
    }

    fn elapsed_label(&self) -> String {
        let secs = self.started_at.elapsed().as_secs();
        if secs < 60 {
            format!("🕐 {secs}s")
        } else {
            let mins = secs / 60;
            format!("🕐 {mins}min")
        }
    }

    fn spinner(&self) -> &'static str {
        SPINNER[self.tick % SPINNER.len()]
    }

    /// Live composite: spinner status + reasoning tail + last N tools.
    fn render_live(&self) -> String {
        let spin = self.spinner();
        let tokens = if let Some(u) = &self.usage {
            format!(" · {} in / {} out", u.input_tokens, u.output_tokens)
        } else {
            String::new()
        };
        let elapsed = self.elapsed_label();
        let status = format!("{spin} <i>{} · {elapsed}{tokens}</i>", self.phase);

        let mut parts = vec![status];

        if self.in_thinking && !self.thinking.is_empty() {
            let tail = if self.thinking.len() > REASONING_TAIL {
                let mut boundary = self.thinking.len() - REASONING_TAIL;
                while boundary < self.thinking.len() && !self.thinking.is_char_boundary(boundary) {
                    boundary += 1;
                }
                format!("…{}", &self.thinking[boundary..])
            } else {
                self.thinking.clone()
            };
            parts.push(format!("💭 <i>{}</i>", escape_html(&tail)));
        }

        let start = self.tool_lines.len().saturating_sub(TOOL_WINDOW);
        for line in &self.tool_lines[start..] {
            parts.push(line.clone());
        }

        if !self.sub_agents.is_empty() {
            let last_ids: Vec<_> = self
                .sub_agent_order
                .iter()
                .rev()
                .take(3)
                .rev()
                .cloned()
                .collect();
            for id in &last_ids {
                if let Some(sa) = self.sub_agents.get(id) {
                    let icon = match sa.status {
                        "running" => "🤖",
                        "done" => "✅",
                        "error" => "❌",
                        _ => "⏳",
                    };
                    let tool_info = sa
                        .last_tool
                        .as_deref()
                        .map(|t| {
                            if sa.tool_count > 1 {
                                format!(" · {t} ×{}", sa.tool_count)
                            } else {
                                format!(" · {t}")
                            }
                        })
                        .unwrap_or_default();
                    let short = truncate_str(&sa.prompt, 40);
                    parts.push(format!(
                        "{icon} <b>{id}</b> {}{tool_info}",
                        escape_html(&short)
                    ));
                }
            }
        }

        parts.join("\n")
    }

    fn usage_footer(&self) -> String {
        if let Some(u) = &self.usage {
            let (_, _, cost) = u.estimate_cost(&self.model_tag);
            let cost_str = if cost >= 0.01 {
                format!(" · ${cost:.2}")
            } else if cost > 0.0 {
                format!(" · ${cost:.4}")
            } else {
                String::new()
            };
            format!(
                "{} in / {} out{}",
                u.input_tokens, u.output_tokens, cost_str
            )
        } else {
            String::new()
        }
    }

    /// Final: replace everything with clean response text + reasoning block + footer.
    ///
    /// The reasoning chain (when present) is wrapped in
    /// `<blockquote expandable>` so it ships collapsed by default — users
    /// who want to see the model's thinking just tap to expand. The
    /// thinking block is dynamically squeezed so the whole final message
    /// fits in a single Telegram message (`MAX_TG_MSG`) — no more
    /// "echo-chunk" second posts that leak CoT drafts after the real
    /// answer.
    fn render_final(&self) -> String {
        let text = self.response_text.trim();
        let thinking = self.thinking.trim();
        if text.is_empty() && thinking.is_empty() {
            return "<i>— модель закрыла ход без ответа. Попробуй переформулировать или <code>/new</code>.</i>".into();
        }
        let footer = self.usage_footer();
        let footer_rendered = if footer.is_empty() {
            String::new()
        } else {
            format!("\n\n<i>✓ {footer}</i>")
        };
        let body = if text.is_empty() {
            "<i>(no text — reasoning only)</i>".to_string()
        } else {
            md_to_tg_html(text)
        };

        // Budget: we want `body + thinking_block + footer_rendered`
        // to fit into one TG message. The thinking block is expendable
        // (it's collapsed by default anyway); the primary answer is not.
        // `SINGLE_MESSAGE_TARGET` is a safety margin below MAX_TG_MSG to
        // absorb HTML tag overhead from `<blockquote>` etc.
        const SINGLE_MESSAGE_TARGET: usize = MAX_TG_MSG - 200;
        let fixed_len = body.len() + footer_rendered.len() + 2 /* \n\n separator */;
        let thinking_budget = SINGLE_MESSAGE_TARGET.saturating_sub(fixed_len);

        let thinking_block = self.render_thinking_block_budgeted(thinking, thinking_budget);

        let mut out = String::with_capacity(body.len() + 256);
        out.push_str(&body);
        if let Some(block) = thinking_block {
            out.push_str("\n\n");
            out.push_str(&block);
        }
        out.push_str(&footer_rendered);
        out
    }

    /// Thinking block that respects a dynamic byte budget.
    ///
    /// Returns `None` if the raw chain is empty or if the budget is too
    /// small to fit even a minimal `<blockquote>` header (we'd rather
    /// drop the thinking entirely than half-render a broken tag). The
    /// payload is tail-preserved (the conclusion is at the end of the
    /// CoT) and capped by the lower of `MAX_FINAL_THINKING_BYTES` and
    /// the caller-provided budget.
    fn render_thinking_block_budgeted(
        &self,
        trimmed: &str,
        budget: usize,
    ) -> Option<String> {
        if trimmed.is_empty() {
            return None;
        }
        // Overhead of the wrapper tags + our small prefix.
        const WRAPPER_OVERHEAD: usize = 64;
        const MIN_PAYLOAD: usize = 80;
        if budget < WRAPPER_OVERHEAD + MIN_PAYLOAD {
            return None;
        }
        let payload_cap = budget
            .saturating_sub(WRAPPER_OVERHEAD)
            .min(MAX_FINAL_THINKING_BYTES);
        let payload = if trimmed.len() > payload_cap {
            let mut start = trimmed.len() - payload_cap;
            while start < trimmed.len() && !trimmed.is_char_boundary(start) {
                start += 1;
            }
            format!("…{}", &trimmed[start..])
        } else {
            trimmed.to_string()
        };
        Some(format!(
            "<blockquote expandable>💭 <b>thinking</b>\n{}</blockquote>",
            escape_html(&payload)
        ))
    }

    /// Collapsible reasoning block with the default byte budget.
    ///
    /// Retained for tests only — real rendering happens through
    /// `render_thinking_block_budgeted` so that `render_final` can
    /// shrink the CoT to fit the outgoing TG message.
    #[cfg(test)]
    fn render_thinking_block(&self) -> Option<String> {
        let trimmed = self.thinking.trim();
        self.render_thinking_block_budgeted(trimmed, MAX_FINAL_THINKING_BYTES + 64)
    }

    /// Truncated preview for the placeholder message when sending a file.
    fn render_summary(&self, max_chars: usize) -> String {
        let text = self.response_text.trim();
        let converted = md_to_tg_html(text);
        let preview: String = if converted.chars().count() > max_chars {
            let truncated: String = converted.chars().take(max_chars).collect();
            format!("{truncated}…")
        } else {
            converted
        };
        format!(
            "{preview}\n\n📄 <i>Full response attached as file</i>\n<i>{}</i>",
            self.usage_footer()
        )
    }
}

fn render_html_document(view: &CompositeView) -> Vec<u8> {
    let text = view.response_text.trim();
    let content_html = md_to_tg_html(text).replace('\n', "<br>\n");

    // The full reasoning chain — no truncation here, this is the
    // attached HTML file precisely so users can see everything.
    // We render it in its own <details> section so it's collapsed by
    // default just like the inline message blockquote.
    let thinking = view.thinking.trim();
    let thinking_block = if thinking.is_empty() {
        String::new()
    } else {
        format!(
            "<details class=\"thinking\"><summary>💭 reasoning ({} chars)</summary><pre>{}</pre></details>",
            thinking.len(),
            escape_html(thinking)
        )
    };

    let footer = view.usage_footer().replace(" · ", " &middot; ");

    let doc = format!(
        r#"<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Response</title>
<style>
:root {{
  --bg: #1e1e2e;
  --fg: #cdd6f4;
  --muted: #6c7086;
  --surface: #313244;
  --code-bg: #181825;
  --accent: #89b4fa;
  --border: #45475a;
}}
* {{ margin: 0; padding: 0; box-sizing: border-box; }}
body {{
  background: var(--bg);
  color: var(--fg);
  font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
  font-size: 15px;
  line-height: 1.7;
  padding: 24px;
  max-width: 900px;
  margin: 0 auto;
}}
pre {{
  background: var(--code-bg);
  border: 1px solid var(--border);
  border-radius: 8px;
  padding: 16px;
  overflow-x: auto;
  margin: 12px 0;
  font-family: 'JetBrains Mono', 'Fira Code', 'Cascadia Code', monospace;
  font-size: 13px;
  line-height: 1.5;
}}
code {{
  background: var(--code-bg);
  border-radius: 4px;
  padding: 2px 6px;
  font-family: 'JetBrains Mono', 'Fira Code', 'Cascadia Code', monospace;
  font-size: 13px;
}}
pre code {{
  background: none;
  padding: 0;
}}
a {{
  color: var(--accent);
  text-decoration: none;
}}
a:hover {{
  text-decoration: underline;
}}
blockquote {{
  border-left: 3px solid var(--accent);
  padding-left: 16px;
  margin: 12px 0;
  color: var(--muted);
}}
hr {{
  border: none;
  border-top: 1px solid var(--border);
  margin: 20px 0;
}}
.footer {{
  margin-top: 32px;
  padding-top: 16px;
  border-top: 1px solid var(--border);
  color: var(--muted);
  font-size: 13px;
}}
.thinking {{
  margin-top: 24px;
  padding: 12px 16px;
  background: var(--surface);
  border: 1px solid var(--border);
  border-radius: 8px;
  color: var(--muted);
  font-size: 13px;
}}
.thinking summary {{
  cursor: pointer;
  font-weight: 600;
  color: var(--accent);
}}
.thinking pre {{
  margin-top: 12px;
  background: var(--code-bg);
  white-space: pre-wrap;
}}
</style>
</head>
<body>
<div class="content">
{content_html}
</div>
{thinking_block}
<div class="footer">{footer}</div>
</body>
</html>"#
    );

    doc.into_bytes()
}

async fn stream_response(
    bot: Bot,
    ctx: ChatCtx,
    handle: AgentHandle,
    channel_map: &ChannelSessionMap,
    pending_perms: &PendingPermissions,
    model_tag: String,
    http_client: &reqwest::Client,
    base_url: &str,
    rate_limiter: &TgRateLimiter,
) {
    let AgentHandle {
        mut events,
        permissions,
    } = handle;
    let chat_id_raw = ctx.chat_id.0;
    let tid = ctx.raw_thread_id();

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

    let mut view = CompositeView::new(model_tag);
    let mut last_edit = tokio::time::Instant::now();
    let mut dirty = false;
    let mut last_sent = String::new();

    loop {
        let event = tokio::select! {
            ev = events.recv() => match ev {
                Some(e) => Some(e),
                None => break,
            },
            _ = tokio::time::sleep(EDIT_INTERVAL) => None,
        };

        let mut force_flush = false;

        if let Some(event) = event {
            match event {
                AgentEvent::ThinkingDelta(t) => {
                    view.in_thinking = true;
                    view.phase = "thinking";
                    if view.thinking.len() < MAX_THINKING_BYTES {
                        view.thinking.push_str(&t);
                    }
                    dirty = true;
                }
                AgentEvent::TextDelta(t) => {
                    view.in_thinking = false;
                    view.phase = "generating";
                    if view.response_text.len() < MAX_RESPONSE_BYTES {
                        view.response_text.push_str(&t);
                    }
                    dirty = true;
                }
                AgentEvent::ToolStart { name, input, .. } => {
                    view.in_thinking = false;
                    view.phase = "tool use";
                    let preview = format_input_preview(&input, 200);
                    if view.tool_lines.len() >= TOOL_WINDOW * 4 {
                        view.tool_lines.drain(..view.tool_lines.len() - TOOL_WINDOW);
                    }
                    view.tool_lines
                        .push(format!("🔧 <b>{}</b>({preview})…", escape_html(&name)));
                    dirty = true;
                    force_flush = true;
                }
                AgentEvent::ToolEnd {
                    name,
                    state,
                    output,
                    ..
                } => {
                    let icon = match state {
                        naked_core::types::ToolState::Completed => "✅",
                        naked_core::types::ToolState::Error => "❌",
                    };
                    let title = truncate_str(&output, 80);
                    view.tool_lines.push(format!(
                        "{icon} <b>{}</b> — {}",
                        escape_html(&name),
                        escape_html(&title)
                    ));
                    dirty = true;
                    force_flush = true;
                }
                AgentEvent::PermissionRequest {
                    call_id,
                    tool_name,
                    input,
                    permission,
                } => {
                    flush_live(
                        &bot,
                        ctx.chat_id,
                        placeholder,
                        &view,
                        &mut last_sent,
                        rate_limiter,
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
                } => {
                    let note = format!(
                        "📦 контекст был сжат: {} сообщений → {}",
                        before_msgs, after_msgs
                    );
                    let _ = bot
                        .send_message(ctx.chat_id, &note)
                        .maybe_thread(ctx.thread_id)
                        .maybe_reply_to(ctx.reply_to)
                        .await;
                }
                AgentEvent::Heartbeat => {
                    view.tick += 1;
                    dirty = true;
                }
                AgentEvent::SubAgentProgress {
                    agent_id,
                    event: sa_ev,
                } => {
                    use naked_core::types::SubAgentEvent;
                    match sa_ev {
                        SubAgentEvent::Started { prompt_preview } => {
                            view.phase = "sub_agent";
                            if !view.sub_agent_order.contains(&agent_id) {
                                view.sub_agent_order.push(agent_id.clone());
                            }
                            view.sub_agents.insert(
                                agent_id,
                                SubAgentState {
                                    prompt: prompt_preview,
                                    status: "running",
                                    last_tool: None,
                                    tool_count: 0,
                                },
                            );
                        }
                        SubAgentEvent::ToolUse { name, .. } => {
                            if let Some(sa) = view.sub_agents.get_mut(&agent_id) {
                                sa.last_tool = Some(name);
                                sa.tool_count = sa.tool_count.saturating_add(1);
                            }
                        }
                        SubAgentEvent::ToolDone { .. } => {}
                        SubAgentEvent::TextDelta(_) => {}
                        SubAgentEvent::Finished { .. } => {
                            if let Some(sa) = view.sub_agents.get_mut(&agent_id) {
                                sa.status = "done";
                                sa.last_tool = None;
                            }
                        }
                        SubAgentEvent::Error(_) => {
                            if let Some(sa) = view.sub_agents.get_mut(&agent_id) {
                                sa.status = "error";
                            }
                        }
                    }
                    dirty = true;
                    force_flush = true;
                }
                AgentEvent::UsageUpdate(u) => {
                    view.usage = Some(u);
                    dirty = true;
                }
                AgentEvent::Error(e) => {
                    // Friendly mapping for the one specific class of
                    // provider failures we see often enough to warrant
                    // a human-readable hint: the "0-token refusal" that
                    // glm-5-turbo and a few OpenAI-compatible gateways
                    // fall into when they decline without explaining.
                    // The agent loop surfaces these as:
                    //   "provider returned no content ..."
                    // Surfacing that raw string is confusing; we replace
                    // it with an actionable suggestion.
                    let pretty = if e.contains("no content") {
                        "— модель закрыла ход без ответа (0 токенов). \
                         Попробуй переформулировать запрос или открой новую сессию: /new."
                            .to_string()
                    } else {
                        format!("❌ {e}")
                    };
                    if !view.response_text.is_empty() {
                        view.response_text.push('\n');
                    }
                    view.response_text.push_str(&pretty);
                    dirty = true;
                    force_flush = true;
                }
                AgentEvent::Idle => break,
            }
        } // end if let Some(event)

        // Tick spinner + flush periodically
        if force_flush || last_edit.elapsed() >= EDIT_INTERVAL {
            view.tick += 1;
            dirty = true;
        }

        let elapsed = last_edit.elapsed();
        let can_flush = if force_flush {
            elapsed >= MIN_EDIT_GAP
        } else {
            elapsed >= EDIT_INTERVAL
        };
        if dirty && can_flush {
            flush_live(
                &bot,
                ctx.chat_id,
                placeholder,
                &view,
                &mut last_sent,
                rate_limiter,
            )
            .await;
            last_edit = tokio::time::Instant::now();
            dirty = false;
        }
    }

    typing_cancel.cancel();

    let final_html = view.render_final();
    send_final(bot, ctx, placeholder, &final_html, &view).await;
}

const FILE_THRESHOLD: usize = MAX_TG_MSG * 2;
const SUMMARY_CHARS: usize = 500;

/// Edit with retry: if rate-limited, waits the indicated duration and retries.
async fn edit_with_retry(
    bot: &Bot,
    chat_id: ChatId,
    msg_id: MessageId,
    text: &str,
    parse_html: bool,
) -> bool {
    let modes: &[bool] = if parse_html { &[true, false] } else { &[false] };

    for &use_html in modes {
        for attempt in 0..3 {
            let result = if use_html {
                bot.edit_message_text(chat_id, msg_id, text)
                    .parse_mode(ParseMode::Html)
                    .await
            } else {
                bot.edit_message_text(chat_id, msg_id, text).await
            };
            match result {
                Ok(_) => return true,
                Err(e) => {
                    let err_str = e.to_string();
                    if let Some(wait) = parse_retry_after(&err_str) {
                        let wait = wait.min(60);
                        tracing::warn!(
                            attempt,
                            wait,
                            use_html,
                            "final edit rate-limited, waiting {wait}s"
                        );
                        tokio::time::sleep(Duration::from_secs(wait + 1)).await;
                        continue;
                    }
                    if use_html && attempt == 0 {
                        tracing::warn!("final edit (HTML) failed: {e}, falling back to plain text");
                        break;
                    }
                    tracing::error!("final edit failed: {e}");
                    return false;
                }
            }
        }
    }
    false
}

async fn send_final(bot: Bot, ctx: ChatCtx, msg_id: MessageId, html: &str, view: &CompositeView) {
    let chat_id = ctx.chat_id;

    // Short: fits in one message
    if html.len() <= MAX_TG_MSG {
        edit_with_retry(&bot, chat_id, msg_id, html, true).await;
        return;
    }

    // Medium: fits in 2 chunks
    if html.len() <= FILE_THRESHOLD {
        let chunks = split_html(html, MAX_TG_MSG - 100);
        if let Some(first) = chunks.first() {
            edit_with_retry(&bot, chat_id, msg_id, first, true).await;
        }
        for chunk in chunks.iter().skip(1) {
            let res = bot
                .send_message(chat_id, *chunk)
                .parse_mode(ParseMode::Html)
                .maybe_thread(ctx.thread_id)
                .maybe_reply_to(ctx.reply_to)
                .await;
            if let Err(e) = res {
                tracing::warn!("send chunk (HTML) failed: {e}, retrying plain text");
                if let Err(e2) = bot
                    .send_message(chat_id, *chunk)
                    .maybe_thread(ctx.thread_id)
                    .maybe_reply_to(ctx.reply_to)
                    .await
                {
                    tracing::error!("send chunk (plain) also failed: {e2}");
                }
            }
        }
        return;
    }

    let summary = view.render_summary(SUMMARY_CHARS);
    edit_with_retry(&bot, chat_id, msg_id, &summary, true).await;

    let html_doc = render_html_document(view);
    let input_file = teloxide::types::InputFile::memory(html_doc).file_name("response.html");
    if let Err(e) = bot
        .send_document(chat_id, input_file)
        .caption("📄 Full response")
        .maybe_thread(ctx.thread_id)
        .maybe_reply_to(ctx.reply_to)
        .await
    {
        tracing::warn!("send_document failed: {e}");
    }
}

async fn send_long_text(bot: &Bot, ctx: ChatCtx, text: &str) -> Result<(), teloxide::RequestError> {
    if text.len() <= MAX_TG_MSG {
        bot.send_message(ctx.chat_id, text)
            .maybe_thread(ctx.thread_id)
            .await?;
        return Ok(());
    }
    for chunk in split_html(text, MAX_TG_MSG - 100) {
        bot.send_message(ctx.chat_id, chunk)
            .maybe_thread(ctx.thread_id)
            .await?;
    }
    Ok(())
}

// ── Permission prompt ───────────────────────────────────────────────────────

async fn ask_permission(
    bot: &Bot,
    ctx: ChatCtx,
    call_id: &str,
    tool_name: &str,
    input: &serde_json::Value,
    permission: &Permission,
    pending: &PendingPermissions,
) -> bool {
    let level = match permission {
        Permission::WorkspaceWrite => "write",
        Permission::Dangerous => "dangerous",
        Permission::ReadOnly => "read",
    };
    let preview = format_input_preview(input, 200);
    let text = format!(
        "🔐 <b>Permission required</b> [{level}]\n\n<b>{}</b>({preview})",
        escape_html(tool_name),
    );

    let (tx, rx) = oneshot::channel();
    let chat_id_raw = ctx.chat_id.0;
    let tid_raw = ctx.raw_thread_id();
    pending
        .write()
        .await
        .insert(call_id.to_string(), (tx, chat_id_raw, tid_raw));

    let keyboard = InlineKeyboardMarkup::new(vec![vec![
        InlineKeyboardButton::callback("✅ Allow", format!("p:{call_id}:allow")),
        InlineKeyboardButton::callback("❌ Deny", format!("p:{call_id}:deny")),
        InlineKeyboardButton::callback("⚡ YOLO", format!("p:{call_id}:yolo")),
    ]]);

    let sent = bot
        .send_message(ctx.chat_id, &text)
        .parse_mode(ParseMode::Html)
        .reply_markup(keyboard)
        .maybe_thread(ctx.thread_id)
        .await;

    if sent.is_err() {
        pending.write().await.remove(call_id);
        return false;
    }

    match tokio::time::timeout(PERMISSION_TIMEOUT, rx).await {
        Ok(Ok(allowed)) => allowed,
        _ => {
            pending.write().await.remove(call_id);
            false
        }
    }
}

/// Parse "Retry after Xs" from Telegram error string.
fn parse_retry_after(err: &str) -> Option<u64> {
    let s = err.to_lowercase();
    if let Some(pos) = s.find("retry after") {
        let after = &s[pos + 12..];
        let digits: String = after.chars().take_while(|c| c.is_ascii_digit()).collect();
        digits.parse().ok()
    } else {
        None
    }
}

/// Deduplicated edit: only sends if content changed. Acquires global rate limiter
/// slot before sending. On rate limit from Telegram, backs off.
async fn flush_live(
    bot: &Bot,
    chat_id: ChatId,
    msg_id: MessageId,
    view: &CompositeView,
    last_sent: &mut String,
    rate_limiter: &TgRateLimiter,
) {
    let html = view.render_live();
    let trimmed = truncate_str(&html, MAX_TG_MSG - 50);
    if trimmed == *last_sent {
        return;
    }
    rate_limiter.acquire().await;
    *last_sent = trimmed.clone();
    let result = bot
        .edit_message_text(chat_id, msg_id, &trimmed)
        .parse_mode(ParseMode::Html)
        .await;
    if let Err(e) = result {
        let err_str = e.to_string();
        if err_str.contains("429") || err_str.contains("Too Many Requests") {
            let wait = parse_retry_after(&err_str).unwrap_or(5);
            tracing::debug!("rate-limited on live edit, backing off {wait}s");
            tokio::time::sleep(Duration::from_secs(wait)).await;
            *last_sent = String::new();
        } else if !err_str.contains("not modified") {
            tracing::warn!("edit_message_text error: {e}");
        }
    }
}

// ── Access control ──────────────────────────────────────────────────────────

fn is_allowed(chat_id: i64, config: &Config) -> bool {
    if config.allowed_chat_ids.is_empty() {
        return false;
    }
    config.allowed_chat_ids.contains(&chat_id)
}

// ── Commands ────────────────────────────────────────────────────────────────

/// Returns `Ok(true)` if the command was handled, `Ok(false)` if unrecognized
/// (caller should pass the message to the agent).
async fn handle_command(
    bot: &Bot,
    _msg: &Message,
    text: &str,
    agent: &Arc<AgentCore>,
    channel_map: &Arc<ChannelSessionMap>,
    config: &Config,
    ctx: ChatCtx,
    pending_perms: &PendingPermissions,
    attribution_flag: &Arc<std::sync::atomic::AtomicBool>,
) -> Result<bool, teloxide::RequestError> {
    let chat_id = ctx.chat_id.0;
    let tid = ctx.raw_thread_id();

    let cmd_word = text.split_whitespace().next().unwrap_or("");
    let cmd = cmd_word.split('@').next().unwrap_or(cmd_word);
    tracing::debug!(chat_id, cmd, cmd_word, "handle_command");
    match cmd {
        "/start" => {
            bot.send_message(
                ctx.chat_id,
                "naked agent ready. Send me a message to start.",
            )
            .maybe_thread(ctx.thread_id)
            .await?;
        }
        "/attribution" => {
            use std::sync::atomic::Ordering;
            let arg = text[cmd_word.len()..].trim().to_ascii_lowercase();
            let reply = match arg.as_str() {
                "on" | "1" | "true" | "enable" => {
                    attribution_flag.store(true, Ordering::Relaxed);
                    "✅ sender attribution: ON (groups will see `@username:` prefix)".to_string()
                }
                "off" | "0" | "false" | "disable" => {
                    attribution_flag.store(false, Ordering::Relaxed);
                    "⛔ sender attribution: OFF".to_string()
                }
                "" | "status" => {
                    let on = attribution_flag.load(Ordering::Relaxed);
                    let state = if on { "ON" } else { "OFF" };
                    format!(
                        "sender attribution: {state}\nUsage: /attribution on|off|status\n(resets to config.tg_sender_attribution on restart)"
                    )
                }
                other => {
                    format!("Unknown arg `{other}`. Usage: /attribution on|off|status")
                }
            };
            bot.send_message(ctx.chat_id, reply)
                .maybe_thread(ctx.thread_id)
                .await?;
        }
        "/new" => {
            // Best-effort: snapshot the closing session into a daily
            // memory draft before we cut the channel binding to it.
            // Runs in the background; never blocks `/new`.
            if let Some(prev) = channel_map.get(chat_id, tid).await {
                agent.close_session_summary(&prev).await;
            }
            let session_id = agent
                .create_session_with_channel(&config.workspace, "telegram")
                .await;
            channel_map.set(chat_id, tid, session_id.clone()).await;
            channel_map.disable_yolo(chat_id, tid).await;
            let cid = format_tg_channel_id(chat_id, tid);
            agent.set_session_channel_id(&session_id, &cid).await;
            bot.send_message(ctx.chat_id, format!("🆕 {session_id}"))
                .maybe_thread(ctx.thread_id)
                .await?;
        }
        "/sessions" => {
            // `/sessions [page]` — 1-indexed, 20 per page, newest first.
            // Backward-compatible: `/sessions` with no arg behaves exactly
            // like before (page 1) thanks to the default.
            const PAGE_SIZE: usize = 20;
            let page: usize = text
                .split_whitespace()
                .nth(1)
                .and_then(|s| s.parse::<usize>().ok())
                .filter(|&n| n >= 1)
                .unwrap_or(1);
            let total = agent.list_sessions().await.len();
            let total_pages = total.div_ceil(PAGE_SIZE).max(1);
            let skip = (page - 1).saturating_mul(PAGE_SIZE);
            let slice = agent.list_sessions_paged(skip, PAGE_SIZE).await;

            if total == 0 {
                bot.send_message(ctx.chat_id, "No sessions.")
                    .maybe_thread(ctx.thread_id)
                    .await?;
            } else if slice.is_empty() {
                bot.send_message(
                    ctx.chat_id,
                    format!("Page {page} is empty. Total pages: {total_pages}."),
                )
                .maybe_thread(ctx.thread_id)
                .await?;
            } else {
                let mut list: Vec<String> = slice
                    .iter()
                    .map(|s| {
                        let prefix = if s.id.len() >= 8 { &s.id[..8] } else { &s.id };
                        format!("• {} ({} msgs, {:?})", prefix, s.message_count, s.state)
                    })
                    .collect();
                list.insert(
                    0,
                    format!("📄 page {page}/{total_pages} ({total} sessions total)"),
                );
                let text = list.join("\n");
                send_long_text(bot, ctx, &text).await?;
            }
        }
        "/abort" => {
            if let Some(sid) = channel_map.get(chat_id, tid).await {
                agent.abort(&sid).await;
                bot.send_message(ctx.chat_id, "Aborted.")
                    .maybe_thread(ctx.thread_id)
                    .await?;
            } else {
                bot.send_message(ctx.chat_id, "No active session.")
                    .maybe_thread(ctx.thread_id)
                    .await?;
            }
        }
        "/metrics" => {
            let snap = crate::metrics::snapshot();
            bot.send_message(ctx.chat_id, snap.render_text())
                .maybe_thread(ctx.thread_id)
                .await?;
        }
        "/provider" | "/providers" => {
            let arg = text[cmd_word.len()..].trim();
            tracing::debug!(chat_id, arg, "cmd /provider");
            if arg.is_empty() {
                let providers = agent.list_providers();
                tracing::debug!(n_providers = providers.len(), "listing providers");
                let (prov, model) = if let Some(sid) = channel_map.get(chat_id, tid).await {
                    agent.session_provider_model(&sid).await
                } else {
                    (
                        config.default_provider.clone(),
                        config.default_model.clone(),
                    )
                };
                tracing::debug!(%prov, %model, "current provider/model");
                let rows: Vec<Vec<InlineKeyboardButton>> = providers
                    .iter()
                    .map(|p| {
                        let mark = if p.name == prov { " ✅" } else { "" };
                        vec![InlineKeyboardButton::callback(
                            format!("{}{mark}", p.name),
                            format!("sp:{}", p.name),
                        )]
                    })
                    .collect();
                let kb = InlineKeyboardMarkup::new(rows);
                match bot
                    .send_message(
                        ctx.chat_id,
                        format!(
                            "Current: <b>{}</b> / <b>{}</b>\n\nSelect provider:",
                            escape_html(&prov),
                            escape_html(&model)
                        ),
                    )
                    .parse_mode(ParseMode::Html)
                    .reply_markup(kb)
                    .maybe_thread(ctx.thread_id)
                    .await
                {
                    Ok(m) => tracing::debug!(msg_id = m.id.0, "sent provider keyboard"),
                    Err(e) => {
                        tracing::error!("failed to send provider keyboard: {e}");
                        return Err(e);
                    }
                }
            } else {
                let sid = get_or_create_session(ctx, agent, channel_map, config).await;
                match agent.set_session_provider(&sid, Some(arg), None).await {
                    Ok(()) => {
                        let (prov, model) = agent.session_provider_model(&sid).await;
                        bot.send_message(ctx.chat_id, format!("Switched to: {prov}/{model}"))
                            .maybe_thread(ctx.thread_id)
                            .await?;
                    }
                    Err(e) => {
                        bot.send_message(ctx.chat_id, format!("Error: {e}"))
                            .maybe_thread(ctx.thread_id)
                            .await?;
                    }
                }
            }
        }
        "/model" | "/models" => {
            let arg = text[cmd_word.len()..].trim();
            if arg.is_empty() {
                let (prov, current_model) = if let Some(sid) = channel_map.get(chat_id, tid).await {
                    agent.session_provider_model(&sid).await
                } else {
                    (
                        config.default_provider.clone(),
                        config.default_model.clone(),
                    )
                };
                let models = provider_models(config, &prov);
                if models.is_empty() {
                    bot.send_message(ctx.chat_id, "No models for current provider.")
                        .maybe_thread(ctx.thread_id)
                        .await?;
                } else {
                    let rows: Vec<Vec<InlineKeyboardButton>> = models
                        .iter()
                        .map(|m| {
                            let mark = if *m == current_model { " ✅" } else { "" };
                            vec![InlineKeyboardButton::callback(
                                format!("{m}{mark}"),
                                format!("sm:{m}"),
                            )]
                        })
                        .collect();
                    let kb = InlineKeyboardMarkup::new(rows);
                    bot.send_message(
                        ctx.chat_id,
                        format!(
                            "Provider: <b>{}</b>\nCurrent: <b>{}</b>\n\nSelect model:",
                            escape_html(&prov),
                            escape_html(&current_model)
                        ),
                    )
                    .parse_mode(ParseMode::Html)
                    .reply_markup(kb)
                    .maybe_thread(ctx.thread_id)
                    .await?;
                }
            } else {
                let sid = get_or_create_session(ctx, agent, channel_map, config).await;
                match agent.set_session_provider(&sid, None, Some(arg)).await {
                    Ok(()) => {
                        let (prov, model) = agent.session_provider_model(&sid).await;
                        bot.send_message(ctx.chat_id, format!("Model set: {prov}/{model}"))
                            .maybe_thread(ctx.thread_id)
                            .await?;
                    }
                    Err(e) => {
                        bot.send_message(ctx.chat_id, format!("Error: {e}"))
                            .maybe_thread(ctx.thread_id)
                            .await?;
                    }
                }
            }
        }
        "/reasoning" => {
            let sid = get_or_create_session(ctx, agent, channel_map, config).await;
            let current = agent.session_reasoning(&sid).await;
            let current_level = current.as_deref().unwrap_or("off");
            let levels = ["off", "low", "medium", "high"];
            let rows: Vec<Vec<InlineKeyboardButton>> = levels
                .iter()
                .map(|lvl| {
                    let mark = if *lvl == current_level { " ✅" } else { "" };
                    vec![InlineKeyboardButton::callback(
                        format!("{lvl}{mark}"),
                        format!("sr:{lvl}"),
                    )]
                })
                .collect();
            let kb = InlineKeyboardMarkup::new(rows);
            bot.send_message(
                ctx.chat_id,
                format!(
                    "💭 Reasoning: <b>{}</b>\n\nSelect level:",
                    escape_html(current_level)
                ),
            )
            .parse_mode(ParseMode::Html)
            .reply_markup(kb)
            .maybe_thread(ctx.thread_id)
            .await?;
        }
        "/skills" => {
            let skills = agent.list_skills();
            if skills.is_empty() {
                bot.send_message(ctx.chat_id, "No skills loaded.")
                    .maybe_thread(ctx.thread_id)
                    .await?;
            } else {
                let list: Vec<String> = skills
                    .iter()
                    .map(|(name, path)| {
                        format!(
                            "• <b>{}</b>\n  <code>{}</code>",
                            escape_html(name),
                            escape_html(path)
                        )
                    })
                    .collect();
                let header = format!("📚 <b>{} skill(s)</b>\n\n{}", skills.len(), list.join("\n"));
                bot.send_message(ctx.chat_id, header)
                    .parse_mode(ParseMode::Html)
                    .maybe_thread(ctx.thread_id)
                    .await?;
            }
        }
        "/mcp" => {
            let servers = agent.list_mcp_servers().await;
            if servers.is_empty() {
                bot.send_message(ctx.chat_id, "No MCP servers connected.")
                    .maybe_thread(ctx.thread_id)
                    .await?;
            } else {
                let list: Vec<String> = servers
                    .iter()
                    .map(|(name, n_tools)| {
                        format!("• <b>{}</b> — {n_tools} tool(s)", escape_html(name))
                    })
                    .collect();
                let header = format!(
                    "🔌 <b>{} MCP server(s)</b>\n\n{}",
                    servers.len(),
                    list.join("\n")
                );
                bot.send_message(ctx.chat_id, header)
                    .parse_mode(ParseMode::Html)
                    .maybe_thread(ctx.thread_id)
                    .await?;
            }
        }
        "/refresh" => {
            agent.refresh_skills_and_mcp().await;
            let skills = agent.list_skills();
            let servers = agent.list_mcp_servers().await;
            bot.send_message(
                ctx.chat_id,
                format!(
                    "🔄 Refreshed\n• {} skill(s)\n• {} MCP server(s)",
                    skills.len(),
                    servers.len()
                ),
            )
            .maybe_thread(ctx.thread_id)
            .await?;
        }
        "/approve" | "/yolo" => {
            tracing::info!(chat_id, ?tid, "yolo: enabling");
            let already = channel_map.is_yolo(chat_id, tid).await;
            if already {
                bot.send_message(ctx.chat_id, "⚡ YOLO already active.")
                    .maybe_thread(ctx.thread_id)
                    .await?;
            } else {
                channel_map.enable_yolo(chat_id, tid).await;
                // Persist yolo timestamp to session config
                let yolo_ts = std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap_or_default()
                    .as_secs() as i64;
                if let Some(sid) = channel_map.get(chat_id, tid).await
                    && let Err(e) = agent.set_session_yolo(&sid, Some(yolo_ts)).await
                {
                    tracing::warn!("failed to persist yolo: {e}");
                }
                let mut perms = pending_perms.write().await;
                let matching_keys: Vec<String> = perms
                    .iter()
                    .filter(|(_, (_, cid, t))| *cid == chat_id && *t == tid)
                    .map(|(k, _)| k.clone())
                    .collect();
                let n = matching_keys.len();
                for key in matching_keys {
                    if let Some((tx, _, _)) = perms.remove(&key) {
                        let _ = tx.send(true);
                    }
                }
                if n > 0 {
                    tracing::info!("yolo: auto-approved {n} pending permission(s)");
                }
                let remaining_h = channel_map.yolo_remaining_secs(chat_id, tid).await / 3600;
                bot.send_message(
                    ctx.chat_id,
                    format!(
                        "⚡ YOLO ON — all tools auto-approved ({remaining_h}h).{}",
                        if n > 0 {
                            format!("\n✅ {n} pending request(s) approved.")
                        } else {
                            String::new()
                        }
                    ),
                )
                .maybe_thread(ctx.thread_id)
                .await?;
            }
        }
        "/allow" => {
            let args: Vec<&str> = text.split_whitespace().skip(1).collect();
            match args.first().copied() {
                Some("add") if args.len() >= 2 => {
                    let tool = args[1];
                    let added = channel_map.allow_add(chat_id, tid, tool).await;
                    // Persist allow-list
                    if let Some(sid) = channel_map.get(chat_id, tid).await {
                        let list = channel_map.allow_get(chat_id, tid).await;
                        if let Err(e) = agent.set_session_allow_list(&sid, &list).await {
                            tracing::warn!("failed to persist allow-list: {e}");
                        }
                    }
                    let msg = if added {
                        format!("✅ <b>{}</b> added to allow-list.", escape_html(tool))
                    } else {
                        format!("ℹ️ <b>{}</b> already in allow-list.", escape_html(tool))
                    };
                    bot.send_message(ctx.chat_id, msg)
                        .parse_mode(ParseMode::Html)
                        .maybe_thread(ctx.thread_id)
                        .await?;
                }
                Some("rm" | "remove" | "del") if args.len() >= 2 => {
                    let tool = args[1];
                    let removed = channel_map.allow_remove(chat_id, tid, tool).await;
                    // Persist allow-list
                    if let Some(sid) = channel_map.get(chat_id, tid).await {
                        let list = channel_map.allow_get(chat_id, tid).await;
                        if let Err(e) = agent.set_session_allow_list(&sid, &list).await {
                            tracing::warn!("failed to persist allow-list: {e}");
                        }
                    }
                    let msg = if removed {
                        format!("🗑 <b>{}</b> removed from allow-list.", escape_html(tool))
                    } else {
                        format!("ℹ️ <b>{}</b> not in allow-list.", escape_html(tool))
                    };
                    bot.send_message(ctx.chat_id, msg)
                        .parse_mode(ParseMode::Html)
                        .maybe_thread(ctx.thread_id)
                        .await?;
                }
                _ => {
                    let list = channel_map.allow_get(chat_id, tid).await;
                    let yolo = channel_map.is_yolo(chat_id, tid).await;
                    let mut text = String::new();
                    if yolo {
                        text.push_str("⚡ YOLO mode — everything auto-approved\n\n");
                    }
                    if list.is_empty() {
                        text.push_str("Allow-list is empty.\n");
                    } else {
                        text.push_str(&format!(
                            "📋 <b>{} tool(s)</b> in allow-list:\n",
                            list.len()
                        ));
                        for t in &list {
                            text.push_str(&format!("• <code>{}</code>\n", escape_html(t)));
                        }
                    }
                    text.push_str(
                        "\nUsage:\n<code>/allow add bash</code>\n<code>/allow rm bash</code>",
                    );
                    bot.send_message(ctx.chat_id, text)
                        .parse_mode(ParseMode::Html)
                        .maybe_thread(ctx.thread_id)
                        .await?;
                }
            }
        }
        "/research" => {
            handle_research_cmd(bot, agent, config, &ctx, text, cmd_word).await?;
        }
        "/memory" => {
            handle_memory_cmd(bot, agent, channel_map, &ctx, text, cmd_word).await?;
        }
        _ => {
            return Ok(false);
        }
    }

    Ok(true)
}

/// Start a research run for `spec_id` and install the live-progress
/// waterfall UI in the given chat. Returns the short confirmation
/// string that the caller sends as a separate message only if the
/// placeholder could not be posted — on the happy path this returns
/// `""` because the placeholder *is* the acknowledgement and the
/// heartbeat task keeps editing it.
///
/// Architecture:
///   1. Send a placeholder message with the `[⏸ Stop & clarify]` button,
///      captured `message_id` feeds the heartbeat + completion edits.
///   2. Spawn the research task on `agent.run_research*`. Result lands
///      on a oneshot that the heartbeat `select!`-awaits.
///   3. Spawn the heartbeat: every 20 s, snapshot `run_events`, edit the
///      placeholder. When the research future resolves, the heartbeat
///      flips into the completion branch — renders the report as HTML,
///      sends it as a document, then rewrites the placeholder with a
///      `[🔁 Run again]` button.
async fn launch_research_run_with_ui(
    bot: Bot,
    agent: Arc<AgentCore>,
    config: Config,
    chat_id: ChatId,
    thread_id: Option<ThreadId>,
    spec_id: String,
) -> String {
    let verify = config.research.verify_by_default;
    let max_rounds = config.research.gatekeeper.max_rounds;
    let iteration_cap = config.research.max_iterations.max(1);

    let spec_topic = match agent.research_store().load_spec(&spec_id).await {
        Ok(s) => s.topic,
        Err(e) => {
            return format!("error: spec `{spec_id}` not found ({e})");
        }
    };
    let findings_baseline = agent
        .research_store()
        .count_findings(&spec_id)
        .await
        .unwrap_or(0);

    let started_at = chrono::Utc::now();
    let progress0 = HeartbeatProgress {
        topic: spec_topic.clone(),
        started_at,
        findings_total: findings_baseline,
        findings_baseline,
        iteration_estimate: None,
        iteration_cap,
    };
    let initial_body = render_waterfall(&spec_id, &progress0, &[], started_at);

    let placeholder = match bot
        .send_message(chat_id, &initial_body)
        .maybe_thread(thread_id)
        .reply_markup(keyboard_stop(&spec_id))
        .await
    {
        Ok(m) => m,
        Err(e) => {
            return format!("error: failed to post placeholder: {e}");
        }
    };
    let msg_id = placeholder.id;

    // Oneshot carries the research future's result to the heartbeat
    // task. Using a channel (rather than `JoinHandle`) means the
    // heartbeat can `select!` on both the tick and the completion.
    let (done_tx, done_rx) = oneshot::channel::<ResearchOutcome>();

    let agent_for_run = agent.clone();
    let spec_for_run = spec_id.clone();
    tokio::spawn(async move {
        let outcome = if verify {
            match agent_for_run
                .run_research_verified(&spec_for_run, max_rounds)
                .await
            {
                Ok(vr) => ResearchOutcome::Verified(Box::new(vr)),
                Err(e) => ResearchOutcome::Error(format!("{e:#}")),
            }
        } else {
            match agent_for_run.run_research(&spec_for_run).await {
                Ok(r) => ResearchOutcome::Plain(Box::new(r)),
                Err(e) => ResearchOutcome::Error(format!("{e:#}")),
            }
        };
        let _ = done_tx.send(outcome);
    });

    // Heartbeat loop: edit the placeholder every 20 s with the latest
    // waterfall, or take the completion branch as soon as the run
    // future resolves.
    let agent_for_hb = agent.clone();
    let bot_for_hb = bot.clone();
    let spec_for_hb = spec_id.clone();
    let topic_for_hb = spec_topic.clone();
    tokio::spawn(async move {
        let mut tick = tokio::time::interval(Duration::from_secs(20));
        // Skip the immediate tick — the placeholder already reflects the
        // initial state.
        tick.tick().await;
        let mut done_rx = done_rx;
        let outcome: ResearchOutcome = loop {
            tokio::select! {
                res = &mut done_rx => {
                    break res.unwrap_or(ResearchOutcome::Error(
                        "internal: research task dropped".to_string(),
                    ));
                }
                _ = tick.tick() => {
                    let events = agent_for_hb
                        .research_run_events_snapshot(&spec_for_hb, 16)
                        .await;
                    let total = agent_for_hb
                        .research_store()
                        .count_findings(&spec_for_hb)
                        .await
                        .unwrap_or(findings_baseline);
                    let progress = HeartbeatProgress {
                        topic: topic_for_hb.clone(),
                        started_at,
                        findings_total: total,
                        findings_baseline,
                        iteration_estimate: None,
                        iteration_cap,
                    };
                    let body = render_waterfall(
                        &spec_for_hb,
                        &progress,
                        &events,
                        chrono::Utc::now(),
                    );
                    let edit = bot_for_hb
                        .edit_message_text(chat_id, msg_id, body)
                        .reply_markup(keyboard_stop(&spec_for_hb))
                        .await;
                    if let Err(e) = edit {
                        // Don't bail — a transient 400 "message is not
                        // modified" or rate-limit is routine. We keep
                        // ticking; the next edit will succeed or the
                        // completion path will replace the message anyway.
                        tracing::debug!(spec_id = %spec_for_hb, ?e, "heartbeat edit failed");
                    }
                }
            }
        };

        finalize_research_ui(
            &bot_for_hb,
            chat_id,
            thread_id,
            msg_id,
            &agent_for_hb,
            &spec_for_hb,
            &topic_for_hb,
            findings_baseline,
            started_at,
            outcome,
        )
        .await;
    });

    String::new()
}

/// Result of a background `run_research*` call, shuttled from the
/// launcher task to the heartbeat's completion branch. Boxed so the
/// enum stays small and copy-cheap for the oneshot channel.
enum ResearchOutcome {
    Plain(Box<naked_core::research::RunReport>),
    Verified(Box<naked_core::research::VerifiedRunReport>),
    Error(String),
}

#[allow(clippy::too_many_arguments)]
async fn finalize_research_ui(
    bot: &Bot,
    chat_id: ChatId,
    thread_id: Option<ThreadId>,
    msg_id: MessageId,
    agent: &Arc<AgentCore>,
    spec_id: &str,
    topic: &str,
    findings_baseline: u32,
    started_at: chrono::DateTime<chrono::Utc>,
    outcome: ResearchOutcome,
) {
    let store = agent.research_store();
    let total_after = store.count_findings(spec_id).await.unwrap_or(0);
    let new_this_run = total_after.saturating_sub(findings_baseline);
    let elapsed = (chrono::Utc::now() - started_at).num_seconds().max(0);

    let (summary, run_id_opt, is_error) = match &outcome {
        ResearchOutcome::Plain(r) => {
            let mut s = format!(
                "✅ <b>{}</b> — run complete\nspec <code>{}</code> · +{} finding(s) · total {} · {}s · stop={}",
                escape_html_min(topic),
                escape_html_min(spec_id),
                r.new_findings,
                r.total_findings_after,
                elapsed,
                r.stop_reason.as_str(),
            );
            s.push('\n');
            (s, Some(r.run_id.clone()), false)
        }
        ResearchOutcome::Verified(vr) => {
            let r = &vr.last_run;
            let s = format!(
                "✅ <b>{}</b> — run complete (gatekeeper {} round(s))\nspec <code>{}</code> · +{} new · total {} · removed={} · replacements={} · final={} · {}s\n",
                escape_html_min(topic),
                vr.verification_rounds,
                escape_html_min(spec_id),
                r.new_findings,
                r.total_findings_after,
                vr.dead_removed,
                vr.replacements_found,
                vr.final_findings,
                elapsed,
            );
            (s, Some(r.run_id.clone()), false)
        }
        ResearchOutcome::Error(err) => (
            format!(
                "❌ research run <code>{}</code> failed after {}s\n<pre>{}</pre>",
                escape_html_min(spec_id),
                elapsed,
                escape_html_min(err),
            ),
            None,
            true,
        ),
    };

    let _ = bot
        .edit_message_text(chat_id, msg_id, &summary)
        .parse_mode(ParseMode::Html)
        .reply_markup(keyboard_after_complete(spec_id))
        .await;

    if is_error {
        return;
    }

    // Render and ship the HTML report. Failure to materialise the
    // report is non-fatal — the run summary is already on the chat and
    // the caller can always pull `report.md` from disk.
    let report_md = match store.read_report(spec_id).await {
        Ok(Some(md)) => md,
        Ok(None) => {
            tracing::info!(spec_id, "no report.md to ship (empty run)");
            return;
        }
        Err(e) => {
            tracing::warn!(spec_id, ?e, "reading report.md failed");
            return;
        }
    };

    let meta = ReportMeta {
        spec_id,
        topic,
        run_id: run_id_opt.as_deref(),
        findings_total: total_after,
        new_findings: new_this_run,
        generated_at: chrono::Utc::now(),
    };
    let html = render_report_html(&meta, &report_md);
    let filename = format!("research-{}.html", safe_slug(spec_id));
    let caption = format!(
        "📄 Report for {} · {} finding(s) (+{} this run)",
        topic, total_after, new_this_run,
    );
    let input = teloxide::types::InputFile::memory(html).file_name(filename);
    if let Err(e) = bot
        .send_document(chat_id, input)
        .caption(caption)
        .maybe_thread(thread_id)
        .await
    {
        tracing::warn!(spec_id, ?e, "send_document(report.html) failed");
    }
}

/// Filename-safe slug for use in `research-<slug>.html`. Keeps ASCII
/// alnum + dash/underscore; everything else collapses to `-`.
fn safe_slug(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    let mut last_dash = false;
    for c in s.chars() {
        if c.is_ascii_alphanumeric() || c == '-' || c == '_' {
            out.push(c);
            last_dash = false;
        } else if !last_dash {
            out.push('-');
            last_dash = true;
        }
    }
    if out.is_empty() {
        return "run".to_string();
    }
    out
}

/// Minimal HTML escaper for summary messages. Not a security boundary —
/// user-provided fields here are topic / ids / error strings and the
/// Telegram parser only cares about `<`, `>`, `&`.
fn escape_html_min(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    for c in s.chars() {
        match c {
            '<' => out.push_str("&lt;"),
            '>' => out.push_str("&gt;"),
            '&' => out.push_str("&amp;"),
            _ => out.push(c),
        }
    }
    out
}

/// `/research ...` — operator surface in Telegram for the research subsystem.
///
/// Mirrors the CLI (`naked research ...`) with two concessions:
///   1. `run` is spawned into a background task and replies "launched" so we
///      don't hold up the bot loop for the full agent turn (can be 20+ min).
///      The run result is not pushed back to this chat unless the spec has
///      its own `deliver_to` config (future v6); operators tail the log.
///   2. `schedule <id> <cron>` is stubbed with an explicit "not implemented
///      yet" reply — systemd timer templating is deferred to v6.
async fn handle_research_cmd(
    bot: &Bot,
    agent: &Arc<AgentCore>,
    config: &Config,
    ctx: &ChatCtx,
    text: &str,
    cmd_word: &str,
) -> Result<(), teloxide::RequestError> {
    if !config.research.enabled {
        bot.send_message(ctx.chat_id, "Research subsystem is disabled in config.")
            .maybe_thread(ctx.thread_id)
            .await?;
        return Ok(());
    }
    let rest = text[cmd_word.len()..].trim();
    let mut parts = rest.splitn(2, char::is_whitespace);
    let sub = parts.next().unwrap_or("").to_string();
    let tail = parts.next().unwrap_or("").trim().to_string();

    let reply = match sub.as_str() {
        "" | "help" => "\
/research new <topic>          create a new research spec
/research ls                    list specs with schedule + last-run metrics
/research show <id>             show spec summary + recent findings
/research state <id>            deep scheduler state — inflight ledger, failure streak, recent runs
/research fresh <id>            show only findings from the latest run
/research run <id>              launch a one-off run (background, gatekeeper-verified by default)
/research metrics <id>          detailed metrics for the latest run (gatekeeper rounds, etc.)
/research ask <id> <question>   ask the LLM a question grounded in the known findings
/research pause <id>            pause scheduled runs
/research resume <id>           resume scheduled runs
/research reset <id>            clear failure streak + pause_reason and resume (rearm after fixing the cause)
/research stop <id>             alias for pause
/research rm <id>               delete all data for a spec
/research schedule <id> on <interval>   schedule periodic runs (e.g. 30m, 1h, 1d, or seconds)
/research schedule <id> off              clear schedule
/research schedule <id> status           show current schedule
/research delta <id>            show findings from the latest run only
/research <свободный текст>     (soft-fallback) создаст spec и сразу запустит прогон

Tip: you can also just talk to me — \"расскажи как идёт исследование X\", \
\"исправь расписание X на каждый час\", \"добавь источник Y в X\" — \
the LLM has tools for all of this. А ещё свободный текст без слэша \
(\"исследуй помещения в Дананге, до $3000, на апрель 2026\") поднимает \
research-skill через LLM."
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
        "ls" => match agent.list_research().await {
            Ok(list) if list.is_empty() => "No research specs defined.".to_string(),
            Ok(list) => {
                let store = agent.research_store();
                let mut out = String::from("🔬 research specs:\n");
                for s in list {
                    let total = store.count_findings(&s.id).await.unwrap_or(0);
                    let runs = store.list_runs(&s.id, Some(20)).await.unwrap_or_default();
                    let last = runs
                        .iter()
                        .find(|r| r.verification_rounds.is_some())
                        .or_else(|| runs.first());
                    let inflight = store.load_inflight(&s.id).await.ok().flatten();
                    let status = if s.paused { "⏸" } else { "▶" };
                    let schedule = match s.interval_seconds {
                        Some(secs) => format_interval(secs),
                        None => "manual".to_string(),
                    };
                    out.push_str(&format!(
                        "{status} `{}` — {}\n   schedule: {} · findings: {}",
                        s.id, s.topic, schedule, total
                    ));
                    if s.paused
                        && let Some(reason) = s.pause_reason.as_deref()
                        && !reason.is_empty()
                    {
                        let short: String = reason.chars().take(120).collect();
                        out.push_str(&format!("\n   ⏸ {short}"));
                    }
                    if let Some(r) = last {
                        let age = (chrono::Utc::now() - r.finished_at).num_seconds().max(0) as u64;
                        out.push_str(&format!(
                            " · last: {} ago (+{} new",
                            format_age(age),
                            r.new_findings,
                        ));
                        if let Some(rounds) = r.verification_rounds {
                            out.push_str(&format!(
                                ", {rounds} rd, removed={}, replaced={}",
                                r.dead_removed.unwrap_or(0),
                                r.replacements_found.unwrap_or(0),
                            ));
                        }
                        out.push(')');
                    }
                    if let Some(infl) = inflight {
                        let icon = match infl.state {
                            naked_core::research::RunState::Scheduled => "🟡",
                            naked_core::research::RunState::Running => "🔵",
                            naked_core::research::RunState::Completed => "✅",
                            naked_core::research::RunState::Failed => "❌",
                        };
                        out.push_str(&format!(
                            "\n   state: {icon} {} (attempt {})",
                            infl.state.ru_label(),
                            infl.attempt,
                        ));
                        if let Some(err) = infl.error.as_deref()
                            && !err.is_empty()
                        {
                            let short: String = err.chars().take(80).collect();
                            out.push_str(&format!(" · err: {short}"));
                        }
                    }
                    out.push('\n');
                }
                out
            }
            Err(e) => format!("error: {e}"),
        },
        "metrics" => {
            if tail.is_empty() {
                "Usage: /research metrics <id>".to_string()
            } else {
                match agent.load_research(&tail).await {
                    Ok(spec) => {
                        let store = agent.research_store();
                        let total = store.count_findings(&spec.id).await.unwrap_or(0);
                        let runs = store.list_runs(&spec.id, Some(5)).await.unwrap_or_default();
                        let mut out = format!(
                            "🔬 metrics for `{}`\ntopic: {}\npaused: {}\nsources: {}\nschedule: {}\ntotal findings: {}\n",
                            spec.id,
                            spec.topic,
                            spec.paused,
                            if spec.sources.is_empty() {
                                "(auto)".to_string()
                            } else {
                                spec.sources.join(", ")
                            },
                            match spec.interval_seconds {
                                Some(s) => format_interval(s),
                                None => "manual".to_string(),
                            },
                            total,
                        );
                        if runs.is_empty() {
                            out.push_str("\n(no runs yet — `/research run ");
                            out.push_str(&spec.id);
                            out.push_str("` to start)");
                        } else {
                            out.push_str("\nrecent runs:\n");
                            for r in &runs {
                                let age = (chrono::Utc::now() - r.finished_at).num_seconds().max(0)
                                    as u64;
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
                        out
                    }
                    Err(e) => format!("error: {e}"),
                }
            }
        }
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
            let fresh_only = sub == "fresh" || sub == "delta";
            let usage = if fresh_only {
                "Usage: /research fresh <id>"
            } else {
                "Usage: /research show <id>"
            };
            if tail.is_empty() {
                usage.to_string()
            } else {
                match agent.load_research(&tail).await {
                    Ok(spec) => {
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
                        let header = if fresh_only {
                            format!(
                                "🔬 `{}`\ntopic: {}\nfindings: {} total, {} fresh (latest run)\n",
                                spec.id,
                                spec.topic,
                                total,
                                findings.len()
                            )
                        } else {
                            format!(
                                "🔬 `{}`\ntopic: {}\nsources: {}\npaused: {}\nfindings: {}\n",
                                spec.id,
                                spec.topic,
                                if spec.sources.is_empty() {
                                    "(auto)".to_string()
                                } else {
                                    spec.sources.join(", ")
                                },
                                spec.paused,
                                total
                            )
                        };
                        let mut out = header;
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
                    Err(e) => format!("error: {e}"),
                }
            }
        }
        "run" => {
            if tail.is_empty() {
                "Usage: /research run <id>".to_string()
            } else {
                launch_research_run_with_ui(
                    bot.clone(),
                    agent.clone(),
                    config.clone(),
                    ctx.chat_id,
                    ctx.thread_id,
                    tail.clone(),
                )
                .await
            }
        }
        "ask" => {
            let mut ap = tail.splitn(2, char::is_whitespace);
            let id = ap.next().unwrap_or("").to_string();
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
        "rm" => {
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
            let mut sp = tail.splitn(3, char::is_whitespace);
            let id = sp.next().unwrap_or("").trim();
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
                            interval_seconds: Some(None),
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
                        match secs {
                            None => format!("bad interval `{arg}` — try `3600`, `30m`, `1h`, `1d`"),
                            Some(s) => {
                                let patch = naked_core::ResearchPatch {
                                    interval_seconds: Some(Some(s)),
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
                        }
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
            // Same UX as `/research run <id>`: we hand off to
            // `launch_research_run_with_ui`, which posts the live
            // waterfall + Stop & clarify button and ships the HTML
            // report on completion. Returning the empty string keeps
            // the trailing `send_message` quiet — the placeholder is
            // the acknowledgement.
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
                    let _ = bot
                        .send_message(ctx.chat_id, header)
                        .maybe_thread(ctx.thread_id)
                        .await;
                    launch_research_run_with_ui(
                        bot.clone(),
                        agent.clone(),
                        config.clone(),
                        ctx.chat_id,
                        ctx.thread_id,
                        spec_id,
                    )
                    .await
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
    bot.send_message(ctx.chat_id, reply)
        .maybe_thread(ctx.thread_id)
        .await?;
    Ok(())
}

/// `/memory ...` — operator surface over the daily-digest memory subsystem.
///
/// Resolves the workspace from the active session for this chat/topic so
/// project-scoped queries hit the same `MEMORY.md` the agent sees.
/// Falls back to `config.workspace` if no session is bound yet.
///
/// Subcommands:
///   - `ls` / `list`            — durable rules (MEMORY.md) for the project scope.
///   - `dreams`                 — last 7 entries of the digest audit log.
///   - `drafts`                 — today's draft buffer (pre-promotion).
///   - `stats`                  — counters (rules, drafts, promoted/rejected 7d).
///   - `help` / empty           — usage hint.
async fn handle_memory_cmd(
    bot: &Bot,
    agent: &Arc<AgentCore>,
    channel_map: &Arc<ChannelSessionMap>,
    ctx: &ChatCtx,
    text: &str,
    cmd_word: &str,
) -> Result<(), teloxide::RequestError> {
    use std::path::PathBuf;

    use naked_core::memory::dreams as memory_dreams;
    use naked_core::memory::service::MemoryService;
    use naked_core::memory::store::MarkdownMemoryStore;
    use naked_core::memory::types::MemoryScope;

    let rest = text[cmd_word.len()..].trim();
    let mut parts = rest.splitn(2, char::is_whitespace);
    let sub = parts.next().unwrap_or("").to_string();
    let _tail = parts.next().unwrap_or("").trim().to_string();

    // Resolve workspace via the session bound to (chat_id, thread). If
    // nothing is bound (fresh chat / pre-/new), `MemoryService::list`
    // tolerates a non-existent path and returns an empty list, so we
    // pick the agent's CWD as a sane fallback.
    let chat_id = ctx.chat_id.0;
    let tid = ctx.raw_thread_id();
    let workspace = if let Some(sid) = channel_map.get(chat_id, tid).await {
        agent
            .session_workspace(&sid)
            .await
            .unwrap_or_else(|| std::env::current_dir().unwrap_or_else(|_| PathBuf::from(".")))
    } else {
        std::env::current_dir().unwrap_or_else(|_| PathBuf::from("."))
    };

    let scope = MemoryScope::Project;

    let reply = match sub.as_str() {
        "" | "help" => "\
/memory ls            durable rules from project MEMORY.md
/memory dreams        last 7 daily-digest audit entries
/memory drafts        today's draft buffer (pre-promotion)
/memory stats         rules / drafts / promoted&rejected (7d)

The digest runs at 04:00 UTC by default. Drafts are auto-collected from \
session closes and pre-compaction flushes; rules promote when they \
re-appear across days."
            .to_string(),

        "ls" | "list" => {
            let entries = MemoryService::list(&workspace, Some(scope.clone()));
            if entries.is_empty() {
                "🧠 No durable rules in project MEMORY.md yet.".to_string()
            } else {
                let lines: Vec<String> = entries
                    .iter()
                    .map(|e| {
                        format!(
                            "• [{}] {}",
                            e.memory_type,
                            e.content.lines().next().unwrap_or("")
                        )
                    })
                    .collect();
                format!(
                    "🧠 <b>{} rule(s)</b> (project)\n\n{}",
                    entries.len(),
                    escape_html(&lines.join("\n"))
                )
            }
        }

        "dreams" => {
            let entries = memory_dreams::read_dreams(&workspace, &scope);
            if entries.is_empty() {
                "💭 No dream entries yet — the digest hasn't run for this project.".to_string()
            } else {
                let recent: Vec<_> = entries.iter().rev().take(7).collect();
                let mut lines = Vec::new();
                lines.push(format!("💭 <b>Last {} dream entries</b>", recent.len()));
                for d in recent {
                    lines.push(format!("\n<b>{}</b>", d.date.format("%Y-%m-%d")));
                    if !d.summary.trim().is_empty() {
                        lines.push(escape_html(&d.summary));
                    }
                    if !d.promoted.is_empty() {
                        lines.push(format!("✅ promoted ({}):", d.promoted.len()));
                        for p in &d.promoted {
                            lines.push(format!("  + {}", escape_html(p)));
                        }
                    }
                    if !d.rejected.is_empty() {
                        lines.push(format!("❌ rejected ({}):", d.rejected.len()));
                        for r in &d.rejected {
                            let reason = r.reason.as_deref().unwrap_or("");
                            lines.push(format!(
                                "  − {} {}",
                                escape_html(&r.content),
                                escape_html(reason)
                            ));
                        }
                    }
                }
                lines.join("\n")
            }
        }

        "drafts" => {
            let today = chrono::Utc::now().date_naive();
            let entries = MarkdownMemoryStore::read_daily(&workspace, &scope, today);
            if entries.is_empty() {
                format!("📝 No draft entries for {today} yet.")
            } else {
                let lines: Vec<String> = entries
                    .iter()
                    .map(|e| format!("• [{}/{}] {}", e.memory_type, e.source, e.content))
                    .collect();
                format!(
                    "📝 <b>{} draft(s) for {today}</b>\n\n{}",
                    entries.len(),
                    escape_html(&lines.join("\n"))
                )
            }
        }

        "stats" => {
            let durable = MemoryService::list(&workspace, Some(scope.clone())).len();
            let today = chrono::Utc::now().date_naive();
            let drafts_today = MarkdownMemoryStore::read_daily(&workspace, &scope, today).len();
            let dream_entries = memory_dreams::read_dreams(&workspace, &scope);
            let week_cutoff = today - chrono::Duration::days(7);
            let recent: Vec<_> = dream_entries
                .iter()
                .filter(|d| d.date >= week_cutoff)
                .collect();
            let promoted_7d: usize = recent.iter().map(|d| d.promoted.len()).sum();
            let rejected_7d: usize = recent.iter().map(|d| d.rejected.len()).sum();
            let last_run = dream_entries
                .iter()
                .map(|d| d.date)
                .max()
                .map(|d| d.to_string())
                .unwrap_or_else(|| "never".to_string());
            format!(
                "📊 <b>Memory stats</b> (project)\n\
                 • durable rules: <b>{durable}</b>\n\
                 • drafts today: <b>{drafts_today}</b>\n\
                 • dream entries (7d): <b>{}</b>\n\
                 • promoted (7d): <b>{promoted_7d}</b>\n\
                 • rejected (7d): <b>{rejected_7d}</b>\n\
                 • last digest run: <b>{last_run}</b>",
                recent.len(),
            )
        }

        other => format!("Unknown subcommand: {other}. Try /memory help"),
    };

    bot.send_message(ctx.chat_id, reply)
        .parse_mode(ParseMode::Html)
        .maybe_thread(ctx.thread_id)
        .await?;
    Ok(())
}

async fn get_or_create_session(
    ctx: ChatCtx,
    agent: &Arc<AgentCore>,
    channel_map: &Arc<ChannelSessionMap>,
    config: &Config,
) -> String {
    let chat_id = ctx.chat_id.0;
    let tid = ctx.raw_thread_id();
    if let Some(sid) = channel_map.get(chat_id, tid).await {
        return sid;
    }

    // Per-chat persona: when `naked.json` declares `chat_personas[<chat_id>]`,
    // the new session is rooted in the persona's dedicated workspace instead
    // of the global `config.workspace`. This is the single switch that gives
    // each persona its own system prompt, project memory namespace,
    // CLAUDE.md/AGENTS.md walk, and default `bash` cwd — without forking the
    // bot process. See `naked-core::config::ChatPersona` for the contract.
    let workspace = resolve_session_workspace(chat_id, config).await;

    let session_id = agent
        .create_session_with_channel(&workspace, "telegram")
        .await;
    channel_map.set(chat_id, tid, session_id.clone()).await;

    let channel_id = format_tg_channel_id(chat_id, tid);
    agent.set_session_channel_id(&session_id, &channel_id).await;
    session_id
}

/// Per-chat set of chats where we already sent the "no slash commands"
/// hint at least once. We only nag the operator on their FIRST `/cmd`
/// in a persona chat — repeat slashes are silently dropped to keep the
/// chat clean. The set lives for the bot's lifetime and is rebuilt from
/// scratch on restart, which is fine: a fresh hint after a deploy is
/// useful, not annoying.
static SLASH_HINT_SHOWN: LazyLock<tokio::sync::RwLock<HashSet<i64>>> =
    LazyLock::new(|| tokio::sync::RwLock::new(HashSet::new()));

/// Per (chat_id, thread_id) state recording a research run that the user
/// paused via the "Stop & clarify" inline button. The next regular text
/// message coming from that conversation is treated as the clarification
/// — we append it to the spec's topic and offer a restart button.
///
/// Holding this as a module-level map (rather than threading yet another
/// `Arc<RwLock<..>>` through every handler signature) matches the style
/// already used for `SLASH_HINT_SHOWN` and keeps the /research plumbing
/// self-contained.
type PendingClarificationMap = HashMap<(i64, Option<i32>), PendingClarification>;
static PENDING_CLARIFICATIONS: LazyLock<tokio::sync::RwLock<PendingClarificationMap>> =
    LazyLock::new(|| tokio::sync::RwLock::new(HashMap::new()));

/// Drop a Telegram slash command in a persona chat that opted out of the
/// command surface.
///
/// Returns `true` when the message must be discarded (i.e. the chat has a
/// persona and `allow_slash_commands == false`). The caller should
/// short-circuit `handle_message` immediately. On the FIRST drop per chat
/// we also reply with a one-line hint so the operator knows their `/cmd`
/// was intentional dead air, not a bot bug.
async fn drop_slash_for_persona(bot: &Bot, chat_id: i64, msg: &Message, config: &Config) -> bool {
    let Some(persona) = config.chat_personas.get(&chat_id) else {
        return false;
    };
    if persona.allow_slash_commands {
        return false;
    }

    let already_hinted = SLASH_HINT_SHOWN.read().await.contains(&chat_id);
    if !already_hinted {
        let inserted = SLASH_HINT_SHOWN.write().await.insert(chat_id);
        if inserted {
            let hint = "В этом чате слеш-команды отключены. Скажи естественным \
                        языком, что нужно (например: «выручка вчера», «маржа \
                        за март», «переключи модель на gpt-5») — я разберусь.";
            if let Err(e) = bot
                .send_message(msg.chat.id, hint)
                .maybe_thread(msg.thread_id)
                .await
            {
                tracing::warn!(
                    chat_id,
                    persona = %persona.name,
                    "failed to send slash-disabled hint: {e}"
                );
            }
        }
    }
    tracing::info!(
        chat_id,
        persona = %persona.name,
        "dropping slash command — persona has allow_slash_commands=false"
    );
    true
}

/// Pick the workspace path for a brand-new session bound to `chat_id`.
///
/// Falls back to `config.workspace` (the legacy single-workspace bot
/// behaviour) when no persona is configured for this chat. When a persona
/// IS configured, ensures its workspace directory and the
/// `<workspace>/.naked` subdirectory exist so the first turn doesn't fail
/// reading a non-existent prompt file. Failures during `mkdir` are logged
/// at warn-level but do not abort session creation — the agent will still
/// boot using the built-in default prompt if the persona's prompt file is
/// unreadable, which is strictly better than refusing to answer.
async fn resolve_session_workspace(chat_id: i64, config: &Config) -> std::path::PathBuf {
    if let Some(persona) = config.chat_personas.get(&chat_id) {
        let ws = persona.workspace_expanded();
        if let Err(e) = tokio::fs::create_dir_all(ws.join(".naked")).await {
            tracing::warn!(
                chat_id,
                persona = %persona.name,
                workspace = %ws.display(),
                "failed to ensure persona workspace dir exists: {e}"
            );
        }
        tracing::info!(
            chat_id,
            persona = %persona.name,
            workspace = %ws.display(),
            "creating new session under persona workspace"
        );
        return ws;
    }
    config.workspace.clone()
}

// ── Formatting helpers ──────────────────────────────────────────────────────

fn format_input_preview(input: &serde_json::Value, max_len: usize) -> String {
    if let Some(map) = input.as_object() {
        if map.len() == 1 {
            let (key, val) = map.iter().next().unwrap();
            let fallback = val.to_string();
            let v = val.as_str().unwrap_or(&fallback);
            return format!("{key}: {}", truncate_str(v, max_len));
        }
        let parts: Vec<String> = map
            .iter()
            .map(|(k, v)| {
                let fallback = v.to_string();
                let s = v.as_str().unwrap_or(&fallback);
                format!("{k}: {}", truncate_str(s, 60))
            })
            .collect();
        truncate_str(&parts.join(", "), max_len)
    } else {
        truncate_str(&input.to_string(), max_len)
    }
}

fn truncate_str(s: &str, max_chars: usize) -> String {
    let mut last_boundary = 0;
    for (i, (byte_pos, _)) in s.char_indices().enumerate() {
        if i >= max_chars {
            return format!("{}…", &s[..last_boundary]);
        }
        last_boundary = byte_pos;
    }
    s.to_string()
}

fn escape_html(s: &str) -> String {
    s.replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
}

/// Convert markdown to Telegram-compatible HTML.
fn md_to_tg_html(text: &str) -> String {
    let mut out = String::with_capacity(text.len() + 256);
    let mut in_code_block = false;

    for line in text.lines() {
        let trimmed = line.trim();

        // Fenced code block toggle
        if trimmed.starts_with("```") {
            if in_code_block {
                out.push_str("</pre>\n");
                in_code_block = false;
            } else {
                out.push_str("<pre>");
                in_code_block = true;
            }
            continue;
        }

        if in_code_block {
            out.push_str(&escape_html(line));
            out.push('\n');
            continue;
        }

        // Horizontal rules
        if trimmed == "---" || trimmed == "***" || trimmed == "___" {
            continue;
        }

        // Table separator rows
        if trimmed.starts_with('|') && trimmed.contains("---") {
            continue;
        }

        // Table rows → strip pipes, format inline
        if trimmed.starts_with('|') && trimmed.ends_with('|') {
            let cells: Vec<&str> = trimmed
                .trim_matches('|')
                .split('|')
                .map(|c| c.trim())
                .filter(|c| !c.is_empty())
                .collect();
            if !cells.is_empty() {
                let rendered: Vec<String> =
                    cells.iter().map(|c| md_inline(&escape_html(c))).collect();
                out.push_str(&rendered.join(" · "));
                out.push('\n');
            }
            continue;
        }

        // Headers → bold
        if let Some(rest) = trimmed.strip_prefix("### ") {
            out.push_str(&format!("<b>{}</b>\n", md_inline(&escape_html(rest))));
            continue;
        }
        if let Some(rest) = trimmed.strip_prefix("## ") {
            out.push_str(&format!("\n<b>{}</b>\n", md_inline(&escape_html(rest))));
            continue;
        }
        if let Some(rest) = trimmed.strip_prefix("# ") {
            out.push_str(&format!("\n<b>{}</b>\n", md_inline(&escape_html(rest))));
            continue;
        }

        // Regular line → escape + inline formatting
        let escaped = escape_html(line);
        out.push_str(&md_inline(&escaped));
        out.push('\n');
    }

    if in_code_block {
        out.push_str("</pre>\n");
    }

    // Clean up excessive newlines
    while out.contains("\n\n\n") {
        out = out.replace("\n\n\n", "\n\n");
    }
    out.trim().to_string()
}

/// Apply inline markdown formatting to already-escaped text.
fn md_inline(s: &str) -> String {
    let mut result = s.to_string();

    // Links: [text](url) — url won't have < > since those are escaped
    while let Some(start) = result.find('[') {
        let after_bracket = start + 1;
        if let Some(close) = result[after_bracket..].find("](") {
            let text_end = after_bracket + close;
            let url_start = text_end + 2;
            if let Some(url_end) = result[url_start..].find(')') {
                let link_text = &result[after_bracket..text_end];
                let url = &result[url_start..url_start + url_end];
                let replacement = format!("<a href=\"{}\">{}</a>", url, link_text);
                result = format!(
                    "{}{}{}",
                    &result[..start],
                    replacement,
                    &result[url_start + url_end + 1..]
                );
                continue;
            }
        }
        break;
    }

    // Bold: **text** (before italic to avoid conflict)
    result = apply_pair(&result, "**", "<b>", "</b>");
    // Bold: __text__
    result = apply_pair(&result, "__", "<b>", "</b>");
    // Italic: *text* (single)
    result = apply_pair(&result, "*", "<i>", "</i>");
    // Inline code: `text`
    result = apply_pair(&result, "`", "<code>", "</code>");
    // Strikethrough: ~~text~~
    result = apply_pair(&result, "~~", "<s>", "</s>");

    result
}

fn apply_pair(s: &str, marker: &str, open: &str, close: &str) -> String {
    let mut result = String::with_capacity(s.len());
    let mut rest = s;
    loop {
        if let Some(start) = rest.find(marker) {
            let after = start + marker.len();
            if let Some(end) = rest[after..].find(marker) {
                let inner = &rest[after..after + end];
                if !inner.is_empty() && !inner.starts_with(' ') && !inner.ends_with(' ') {
                    result.push_str(&rest[..start]);
                    result.push_str(open);
                    result.push_str(inner);
                    result.push_str(close);
                    rest = &rest[after + end + marker.len()..];
                    continue;
                }
            }
            result.push_str(&rest[..start + marker.len()]);
            rest = &rest[start + marker.len()..];
        } else {
            result.push_str(rest);
            break;
        }
    }
    result
}

fn split_html(text: &str, max_bytes: usize) -> Vec<&str> {
    if text.len() <= max_bytes {
        return vec![text];
    }

    let mut chunks = Vec::new();
    let mut start = 0;
    while start < text.len() {
        let tentative_end = (start + max_bytes).min(text.len());
        let end = if tentative_end == text.len() {
            tentative_end
        } else {
            let mut e = tentative_end;
            while e > start && !text.is_char_boundary(e) {
                e -= 1;
            }
            e
        };
        if end <= start {
            break;
        }
        let split_at = if end == text.len() {
            end
        } else {
            text[start..end]
                .rfind('\n')
                .map(|pos| start + pos + 1)
                .unwrap_or(end)
        };
        chunks.push(&text[start..split_at]);
        start = split_at;
    }
    chunks
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

#[cfg(test)]
mod tests {
    use super::*;

    // ── render_thinking_block ───────────────────────────────────────────

    fn view_with_response_and_thinking(response: &str, thinking: &str) -> CompositeView {
        let mut v = CompositeView::new("test-model".into());
        v.response_text = response.to_string();
        v.thinking = thinking.to_string();
        v
    }

    #[test]
    fn render_thinking_block_omitted_when_empty() {
        let v = view_with_response_and_thinking("hi", "");
        assert!(v.render_thinking_block().is_none());
        // Final must not contain blockquote when there's no reasoning.
        assert!(!v.render_final().contains("<blockquote"));
    }

    #[test]
    fn render_thinking_block_emits_collapsible_blockquote() {
        let v = view_with_response_and_thinking("answer", "step 1\nstep 2");
        let block = v.render_thinking_block().expect("has thinking");
        assert!(block.starts_with("<blockquote expandable>"));
        assert!(block.ends_with("</blockquote>"));
        assert!(block.contains("💭 <b>thinking</b>"));
        assert!(block.contains("step 1"));
        assert!(block.contains("step 2"));
    }

    #[test]
    fn render_thinking_block_escapes_html() {
        let v = view_with_response_and_thinking("ok", "<script>alert(1)</script>");
        let block = v.render_thinking_block().unwrap();
        assert!(
            !block.contains("<script>"),
            "raw HTML inside reasoning must be escaped — telegram parser \
             would otherwise reject the message or, worse, the model could \
             smuggle markup that breaks our blockquote envelope. got: {block}"
        );
        assert!(block.contains("&lt;script&gt;"));
    }

    #[test]
    fn render_thinking_block_tail_truncates_long_chain() {
        // Use a chain comfortably longer than MAX_FINAL_THINKING_BYTES so
        // we exercise the cap. We expect the *prefix* to be dropped: the
        // commitment / conclusion in a CoT lives at the bottom.
        let prefix = "PREFIX_THAT_SHOULD_BE_DROPPED ".repeat(200);
        let suffix = "FINAL_DECISION";
        let mut chain = String::new();
        chain.push_str(&prefix);
        chain.push_str(suffix);
        assert!(chain.len() > MAX_FINAL_THINKING_BYTES);

        let v = view_with_response_and_thinking("done", &chain);
        let block = v.render_thinking_block().unwrap();
        assert!(
            block.contains(suffix),
            "tail must survive truncation — that's where the conclusion is"
        );
        assert!(
            block.contains('…'),
            "truncated chain must announce itself with an ellipsis"
        );
    }

    #[test]
    fn render_final_includes_thinking_block_when_present() {
        let v = view_with_response_and_thinking("hello world", "let me think");
        let out = v.render_final();
        assert!(out.contains("hello world"));
        assert!(out.contains("<blockquote expandable>"));
        assert!(out.contains("let me think"));
    }

    #[test]
    fn render_final_handles_thinking_only_no_text() {
        let v = view_with_response_and_thinking("", "i was thinking but said nothing");
        let out = v.render_final();
        assert_ne!(out, "(empty response)");
        assert!(out.contains("(no text — reasoning only)"));
        assert!(out.contains("i was thinking but said nothing"));
    }

    #[test]
    fn render_final_empty_returns_helpful_fallback() {
        // When the provider closes the turn with zero text AND zero
        // reasoning the user used to see a cryptic "(empty response)".
        // Now they get an actionable hint that points at `/new`.
        let v = view_with_response_and_thinking("", "");
        let out = v.render_final();
        assert!(!out.contains("(empty response)"));
        assert!(
            out.contains("/new"),
            "empty-turn fallback must mention /new recovery, got: {out}"
        );
    }

    #[test]
    fn render_final_fits_single_telegram_message_even_with_huge_thinking() {
        // Regression for the "echo-thinking leak": if text + thinking
        // exceed MAX_TG_MSG, send_final would split into two TG messages
        // and the second one was almost pure reasoning. Now the thinking
        // block must shrink so the whole final render fits in one
        // message.
        let response = "Here is the final answer. ".repeat(50); // ~1.3 kB
        let thinking = "intermediate reasoning chunk. ".repeat(400); // ~12 kB
        let v = view_with_response_and_thinking(&response, &thinking);
        let out = v.render_final();
        assert!(
            out.len() <= MAX_TG_MSG,
            "render_final must fit in one TG message ({} <= {}), got len={}",
            out.len(),
            MAX_TG_MSG,
            out.len()
        );
        // The primary answer must be preserved verbatim — it's what
        // the user actually wants. Only thinking may be squeezed.
        assert!(out.contains("Here is the final answer."));
    }

    #[test]
    fn render_final_drops_thinking_entirely_when_text_already_full() {
        // Extreme case: response alone nearly fills the message. The
        // thinking block must be dropped outright instead of spilling
        // into a second message.
        let response = "A".repeat(MAX_TG_MSG - 250);
        let thinking = "thought. ".repeat(500);
        let v = view_with_response_and_thinking(&response, &thinking);
        let out = v.render_final();
        assert!(out.len() <= MAX_TG_MSG);
        assert!(
            !out.contains("<blockquote expandable>"),
            "thinking must be dropped (not half-rendered) when budget is tight"
        );
    }

    // ── is_allowed ──────────────────────────────────────────────────────

    #[test]
    fn is_allowed_empty_list_denies_all() {
        let config = Config {
            allowed_chat_ids: vec![],
            ..Config::default()
        };
        assert!(!is_allowed(123, &config));
        assert!(!is_allowed(0, &config));
    }

    #[test]
    fn is_allowed_with_ids_checks_membership() {
        let config = Config {
            allowed_chat_ids: vec![100, 200],
            ..Config::default()
        };
        assert!(is_allowed(100, &config));
        assert!(is_allowed(200, &config));
        assert!(!is_allowed(300, &config));
    }

    // ── escape_html ─────────────────────────────────────────────────────

    #[test]
    fn escape_html_special_chars() {
        assert_eq!(escape_html("<b>hi</b>"), "&lt;b&gt;hi&lt;/b&gt;");
        assert_eq!(escape_html("a & b"), "a &amp; b");
        assert_eq!(escape_html("plain"), "plain");
    }

    // ── truncate_str ────────────────────────────────────────────────────

    #[test]
    fn truncate_str_short_unchanged() {
        assert_eq!(truncate_str("hello", 10), "hello");
    }

    #[test]
    fn truncate_str_exact_length() {
        assert_eq!(truncate_str("hello", 5), "hello");
    }

    #[test]
    fn truncate_str_adds_ellipsis() {
        let result = truncate_str("hello world", 5);
        assert!(result.ends_with('…'));
        assert!(result.len() < "hello world".len() + 3);
    }

    #[test]
    fn truncate_str_multibyte_safe() {
        let s = "日本語テスト";
        let result = truncate_str(s, 3);
        assert!(result.ends_with('…'));
        assert!(result.starts_with("日本"));
    }

    // ── split_html ──────────────────────────────────────────────────────

    #[test]
    fn split_html_short_returns_single() {
        let chunks = split_html("hello", 100);
        assert_eq!(chunks, vec!["hello"]);
    }

    #[test]
    fn split_html_splits_on_newline() {
        let text = "line1\nline2\nline3\nline4\nline5";
        let chunks = split_html(text, 12);
        assert!(chunks.len() > 1);
        let joined: String = chunks.concat();
        assert_eq!(joined, text);
    }

    #[test]
    fn split_html_respects_char_boundaries() {
        let text = "Привет мир, это тест юникода";
        let chunks = split_html(text, 10);
        assert!(chunks.len() > 1);
        let joined: String = chunks.concat();
        assert_eq!(joined, text);
    }

    // ── format_input_preview ────────────────────────────────────────────

    #[test]
    fn format_input_preview_single_key() {
        let input = serde_json::json!({"command": "ls -la"});
        let result = format_input_preview(&input, 100);
        assert!(result.contains("command"));
        assert!(result.contains("ls -la"));
    }

    #[test]
    fn format_input_preview_multi_key() {
        let input = serde_json::json!({"file": "test.rs", "content": "fn main()"});
        let result = format_input_preview(&input, 200);
        assert!(result.contains("file"));
        assert!(result.contains("content"));
    }

    #[test]
    fn format_input_preview_truncates() {
        let long_val = "x".repeat(500);
        let input = serde_json::json!({"data": long_val});
        let result = format_input_preview(&input, 50);
        assert!(result.len() < 200);
    }

    // ── md_to_tg_html ────────────────────────────────────────────────────

    #[test]
    fn md_bold_italic() {
        assert!(md_to_tg_html("**hello**").contains("<b>hello</b>"));
        assert!(md_to_tg_html("*world*").contains("<i>world</i>"));
    }

    #[test]
    fn md_inline_code() {
        assert!(md_to_tg_html("`code`").contains("<code>code</code>"));
    }

    #[test]
    fn md_code_block() {
        let input = "before\n```rust\nfn main() {}\n```\nafter";
        let result = md_to_tg_html(input);
        assert!(result.contains("<pre>"));
        assert!(result.contains("fn main()"));
        assert!(result.contains("</pre>"));
    }

    #[test]
    fn md_headers() {
        assert!(md_to_tg_html("# Big").contains("<b>Big</b>"));
        assert!(md_to_tg_html("## Medium").contains("<b>Medium</b>"));
        assert!(md_to_tg_html("### Small").contains("<b>Small</b>"));
    }

    #[test]
    fn md_link() {
        let result = md_to_tg_html("[click](https://example.com)");
        assert!(result.contains("<a href=\"https://example.com\">click</a>"));
    }

    #[test]
    fn md_table_to_text() {
        let input = "| Name | Score |\n|---|---|\n| Alice | 100 |";
        let result = md_to_tg_html(input);
        assert!(!result.contains('|'));
        assert!(result.contains("Alice"));
        assert!(result.contains("Score"));
    }

    #[test]
    fn md_hr_stripped() {
        let result = md_to_tg_html("above\n---\nbelow");
        assert!(!result.contains("---"));
        assert!(result.contains("above"));
        assert!(result.contains("below"));
    }

    #[test]
    fn md_escapes_html_entities() {
        let result = md_to_tg_html("a < b & c > d");
        assert!(result.contains("&lt;"));
        assert!(result.contains("&amp;"));
        assert!(result.contains("&gt;"));
    }

    // ── sender_label / is_group_chat / extract_reply_context ───────────
    //
    // We build `Message` fixtures by parsing raw Telegram API JSON — this
    // is the same path the dispatcher takes, and it avoids depending on
    // teloxide private constructors.

    fn make_message(v: serde_json::Value) -> Message {
        serde_json::from_value(v).expect("valid Message JSON")
    }

    fn base_private_chat() -> serde_json::Value {
        serde_json::json!({
            "message_id": 1,
            "date": 1_700_000_000,
            "chat": { "id": 42, "type": "private", "first_name": "Alice" },
        })
    }

    fn base_group_chat() -> serde_json::Value {
        serde_json::json!({
            "message_id": 1,
            "date": 1_700_000_000,
            "chat": { "id": -1001, "type": "supergroup", "title": "team" },
        })
    }

    fn user(username: Option<&str>, first: &str, is_bot: bool) -> serde_json::Value {
        let mut u = serde_json::json!({
            "id": 7,
            "is_bot": is_bot,
            "first_name": first,
        });
        if let Some(n) = username {
            u["username"] = serde_json::Value::String(n.to_string());
        }
        u
    }

    #[test]
    fn sender_label_prefers_username_with_at() {
        let mut m = base_private_chat();
        m["from"] = user(Some("alice"), "Alice", false);
        m["text"] = serde_json::Value::String("hi".into());
        assert_eq!(sender_label(&make_message(m)), "@alice");
    }

    #[test]
    fn sender_label_falls_back_to_first_name() {
        let mut m = base_private_chat();
        m["from"] = user(None, "Bob", false);
        m["text"] = serde_json::Value::String("hi".into());
        assert_eq!(sender_label(&make_message(m)), "Bob");
    }

    #[test]
    fn sender_label_unknown_when_no_sender() {
        let mut m = base_private_chat();
        m["text"] = serde_json::Value::String("hi".into());
        assert_eq!(sender_label(&make_message(m)), "unknown");
    }

    #[test]
    fn is_group_chat_true_for_supergroup() {
        let mut m = base_group_chat();
        m["text"] = serde_json::Value::String("hi".into());
        assert!(is_group_chat(&make_message(m)));
    }

    #[test]
    fn is_group_chat_false_for_private() {
        let mut m = base_private_chat();
        m["text"] = serde_json::Value::String("hi".into());
        assert!(!is_group_chat(&make_message(m)));
    }

    #[test]
    fn extract_reply_context_none_when_not_a_reply() {
        let mut m = base_private_chat();
        m["from"] = user(Some("alice"), "Alice", false);
        m["text"] = serde_json::Value::String("hi".into());
        assert!(extract_reply_context(&make_message(m)).is_none());
    }

    #[test]
    fn extract_reply_context_formats_text_reply_with_username() {
        let reply = serde_json::json!({
            "message_id": 10,
            "date": 1_699_999_900,
            "chat": { "id": 42, "type": "private", "first_name": "Alice" },
            "from": user(Some("bob"), "Bob", false),
            "text": "line1\nline2",
        });
        let mut m = base_private_chat();
        m["from"] = user(Some("alice"), "Alice", false);
        m["text"] = serde_json::Value::String("ok".into());
        m["reply_to_message"] = reply;

        let q = extract_reply_context(&make_message(m)).expect("some");
        assert!(q.starts_with("> @bob:\n"), "got: {q}");
        assert!(q.contains("> line1"));
        assert!(q.contains("> line2"));
    }

    #[test]
    fn extract_reply_context_marks_bot_previous_message() {
        let reply = serde_json::json!({
            "message_id": 10,
            "date": 1_699_999_900,
            "chat": { "id": 42, "type": "private", "first_name": "Alice" },
            "from": user(Some("naked_bot"), "naked", true),
            "text": "done.",
        });
        let mut m = base_private_chat();
        m["text"] = serde_json::Value::String("ok".into());
        m["reply_to_message"] = reply;

        let q = extract_reply_context(&make_message(m)).unwrap();
        assert!(q.contains("[your previous message]"), "got: {q}");
    }

    #[test]
    fn extract_reply_context_photo_with_caption() {
        let reply = serde_json::json!({
            "message_id": 10,
            "date": 1_699_999_900,
            "chat": { "id": 42, "type": "private", "first_name": "Alice" },
            "from": user(Some("bob"), "Bob", false),
            "photo": [
                {"file_id":"abc","file_unique_id":"u","width":10,"height":10}
            ],
            "caption": "ship it",
        });
        let mut m = base_private_chat();
        m["text"] = serde_json::Value::String("yep".into());
        m["reply_to_message"] = reply;

        let q = extract_reply_context(&make_message(m)).unwrap();
        assert!(q.contains("[Photo: ship it]"), "got: {q}");
    }

    #[test]
    fn extract_reply_context_photo_without_caption() {
        let reply = serde_json::json!({
            "message_id": 10,
            "date": 1_699_999_900,
            "chat": { "id": 42, "type": "private", "first_name": "Alice" },
            "from": user(Some("bob"), "Bob", false),
            "photo": [
                {"file_id":"abc","file_unique_id":"u","width":10,"height":10}
            ],
        });
        let mut m = base_private_chat();
        m["text"] = serde_json::Value::String("yep".into());
        m["reply_to_message"] = reply;

        let q = extract_reply_context(&make_message(m)).unwrap();
        assert!(q.contains("[Photo]"));
        assert!(!q.contains("[Photo:"));
    }

    // ── extract_media_items / fmt_duration ─────────────────────────────

    #[test]
    fn fmt_duration_formats_mm_ss() {
        assert_eq!(fmt_duration(0), "00:00");
        assert_eq!(fmt_duration(9), "00:09");
        assert_eq!(fmt_duration(65), "01:05");
        assert_eq!(fmt_duration(3599), "59:59");
    }

    #[test]
    fn fmt_duration_caps_long_values() {
        // Anything past 99:59 is clamped.
        assert_eq!(fmt_duration(60 * 99 + 59), "99:59");
        assert_eq!(fmt_duration(60 * 200), "99:59");
    }

    #[test]
    fn extract_media_items_voice() {
        let mut m = base_private_chat();
        m["from"] = user(Some("alice"), "Alice", false);
        m["voice"] = serde_json::json!({
            "file_id": "voice-abc",
            "file_unique_id": "u",
            "duration": 12,
            "mime_type": "audio/ogg"
        });
        let items = extract_media_items(&make_message(m));
        assert_eq!(items.len(), 1);
        assert_eq!(items[0].kind, media::MediaKind::Voice);
        assert_eq!(items[0].file_id, "voice-abc");
        assert!(items[0].file_name.ends_with(".ogg"));
        assert_eq!(items[0].duration_secs, Some(12));
    }

    #[test]
    fn extract_media_items_photo_picks_highest_resolution() {
        let mut m = base_private_chat();
        m["from"] = user(Some("alice"), "Alice", false);
        m["photo"] = serde_json::json!([
            {"file_id":"small","file_unique_id":"s","width":90,"height":90,"file_size":1000},
            {"file_id":"big","file_unique_id":"b","width":1280,"height":720,"file_size":200000}
        ]);
        let items = extract_media_items(&make_message(m));
        assert_eq!(items.len(), 1);
        assert_eq!(items[0].kind, media::MediaKind::Photo);
        assert_eq!(items[0].file_id, "big");
        assert_eq!(items[0].mime_hint.as_deref(), Some("image/jpeg"));
    }

    #[test]
    fn extract_media_items_document_with_name() {
        let mut m = base_private_chat();
        m["from"] = user(Some("alice"), "Alice", false);
        m["document"] = serde_json::json!({
            "file_id": "doc-1",
            "file_unique_id": "u",
            "file_name": "notes.md",
            "mime_type": "text/markdown"
        });
        let items = extract_media_items(&make_message(m));
        assert_eq!(items.len(), 1);
        assert_eq!(items[0].kind, media::MediaKind::Document);
        assert_eq!(items[0].file_name, "notes.md");
        assert_eq!(items[0].mime_hint.as_deref(), Some("text/markdown"));
    }

    #[test]
    fn extract_media_items_none_for_plain_text() {
        let mut m = base_private_chat();
        m["from"] = user(Some("alice"), "Alice", false);
        m["text"] = serde_json::Value::String("just text".into());
        let items = extract_media_items(&make_message(m));
        assert!(items.is_empty());
    }

    #[test]
    fn extract_media_items_pulls_photo_from_reply_target() {
        // The user replies to an old photo with a textual question; we
        // need the photo bytes for the current turn, so handle_message
        // forwards `extract_media_items(reply_to_message)` into the
        // pipeline. Verify the helper itself does the right thing on a
        // reply-target Message: it reads media off whichever Message
        // shape it's handed, so passing the reply target Just Works.
        // (handle_message-side wiring is exercised by the live e2e
        // test; here we pin down the building block.)
        let reply_target = serde_json::json!({
            "message_id": 99,
            "date": 0,
            "chat": { "id": -5084292206_i64, "type": "group", "title": "income" },
            "from": { "id": 123456789, "is_bot": false, "first_name": "u" },
            "photo": [
                {"file_id":"reply-small","file_unique_id":"a","width":90,"height":90,"file_size":1000},
                {"file_id":"reply-big","file_unique_id":"b","width":1280,"height":720,"file_size":200000}
            ]
        });
        let items = extract_media_items(&make_message(reply_target));
        assert_eq!(items.len(), 1, "should extract the single photo");
        assert_eq!(items[0].kind, media::MediaKind::Photo);
        assert_eq!(
            items[0].file_id, "reply-big",
            "should pick the highest-resolution PhotoSize for vision routing"
        );
    }

    // ── Native multimodal routing ───────────────────────────────────────
    //
    // These tests exercise the *decision* path (`is_vision_capable_model` +
    // `native_image_context` + `native_image_max_bytes`). The actual byte-to-
    // base64 conversion happens in `handle_message` and is covered by the
    // live e2e test `live_native_image_roundtrip_via_groq`.

    #[test]
    fn vision_routing_off_when_native_image_context_disabled() {
        use naked_core::config::TgMediaConfig;
        let cfg = TgMediaConfig {
            native_image_context: false,
            ..TgMediaConfig::default()
        };
        // Even a Claude 3 model goes through the legacy text-only path.
        let route =
            cfg.native_image_context && cfg.is_vision_capable_model("claude-sonnet-4-20250514");
        assert!(!route);
    }

    #[test]
    fn vision_routing_on_for_capable_model() {
        use naked_core::config::TgMediaConfig;
        let cfg = TgMediaConfig::default();
        for model in [
            "claude-sonnet-4-20250514",
            "claude-haiku-4-5-20251001",
            "gpt-4o",
            "gpt-4o-mini",
            "meta-llama/llama-4-scout-17b-16e-instruct",
            "grok-2-vision-latest",
        ] {
            let route = cfg.native_image_context && cfg.is_vision_capable_model(model);
            assert!(route, "{model} should route natively");
        }
    }

    #[test]
    fn looks_like_supported_image_accepts_known_formats() {
        // Real magic headers (header bytes only — body is irrelevant).
        let jpeg: Vec<u8> = [&[0xFF, 0xD8, 0xFFu8] as &[u8], &[0u8; 16]].concat();
        let png: Vec<u8> = [
            &[0x89, b'P', b'N', b'G', 0x0D, 0x0A, 0x1A, 0x0A] as &[u8],
            &[0u8; 16],
        ]
        .concat();
        let gif87 = b"GIF87a\0\0\0\0\0\0".to_vec();
        let gif89 = b"GIF89a\0\0\0\0\0\0".to_vec();
        let mut webp = b"RIFF".to_vec();
        webp.extend_from_slice(&[0u8; 4]);
        webp.extend_from_slice(b"WEBP");
        webp.extend_from_slice(&[0u8; 4]);
        for (name, payload) in [
            ("jpeg", jpeg),
            ("png", png),
            ("gif87", gif87),
            ("gif89", gif89),
            ("webp", webp),
        ] {
            assert!(
                looks_like_supported_image(&payload),
                "{name} magic header must be recognised"
            );
        }
    }

    #[test]
    fn looks_like_supported_image_rejects_garbage_and_short_payloads() {
        // Empty / too short / random bytes / repurposed text.
        assert!(!looks_like_supported_image(&[]));
        assert!(!looks_like_supported_image(&[0xFF, 0xD8])); // truncated jpeg
        assert!(!looks_like_supported_image(b"hello world"));
        assert!(!looks_like_supported_image(b"<?xml version=1.0?>"));
        // RIFF without WEBP marker (e.g. WAV) must not be claimed as image.
        let mut wav = b"RIFF".to_vec();
        wav.extend_from_slice(&[0u8; 4]);
        wav.extend_from_slice(b"WAVE");
        wav.extend_from_slice(&[0u8; 4]);
        assert!(!looks_like_supported_image(&wav));
    }

    #[test]
    fn vision_routing_off_for_text_only_model() {
        use naked_core::config::TgMediaConfig;
        let cfg = TgMediaConfig::default();
        for model in [
            "llama-3.3-70b-versatile",
            "glm-5-turbo",
            "MiniMax-Text-01",
            "deepseek-chat",
        ] {
            let route = cfg.native_image_context && cfg.is_vision_capable_model(model);
            assert!(!route, "{model} should NOT route natively");
        }
    }

    fn item_photo(size_hint: Option<u32>) -> MediaItem {
        MediaItem {
            kind: media::MediaKind::Photo,
            file_id: "f".into(),
            file_name: "x.jpg".into(),
            mime_hint: Some("image/jpeg".into()),
            duration_secs: None,
            emoji: None,
            size_hint,
            sticker_format: None,
        }
    }

    #[test]
    fn decide_native_route_off_when_caller_disabled() {
        let item = item_photo(Some(10_000));
        assert!(!decide_native_route(&item, false, u32::MAX));
    }

    #[test]
    fn decide_native_route_on_for_photo_under_cap() {
        let item = item_photo(Some(10_000));
        assert!(decide_native_route(&item, true, 1_000_000));
    }

    #[test]
    fn decide_native_route_on_for_photo_with_no_size_hint() {
        // Telegram sometimes omits `file.size` for cached PhotoSize entries —
        // we should let the download proceed natively rather than degrading
        // pre-emptively. The post-download cap in `process_one_media` will
        // still catch oversized images.
        let item = item_photo(None);
        assert!(decide_native_route(&item, true, 5 * 1024 * 1024));
    }

    #[test]
    fn decide_native_route_off_when_size_exceeds_cap() {
        // Pre-download fallback path: size_hint > native_image_max_bytes must
        // force the legacy describer route so we don't waste bandwidth nor
        // get rejected by the provider for "image too large".
        let item = item_photo(Some(20 * 1024 * 1024));
        assert!(!decide_native_route(&item, true, 5 * 1024 * 1024));
    }

    #[test]
    fn decide_native_route_off_for_animated_sticker() {
        let mut item = item_photo(Some(10_000));
        item.kind = media::MediaKind::Sticker;
        item.sticker_format = Some(StickerFormat::Animated);
        assert!(!decide_native_route(&item, true, u32::MAX));
    }

    #[test]
    fn decide_native_route_off_for_video_sticker() {
        let mut item = item_photo(Some(10_000));
        item.kind = media::MediaKind::Sticker;
        item.sticker_format = Some(StickerFormat::Video);
        assert!(!decide_native_route(&item, true, u32::MAX));
    }

    #[test]
    fn decide_native_route_on_for_static_sticker() {
        let mut item = item_photo(Some(10_000));
        item.kind = media::MediaKind::Sticker;
        item.sticker_format = Some(StickerFormat::Static);
        item.mime_hint = Some("image/webp".into());
        assert!(decide_native_route(&item, true, u32::MAX));
    }

    #[test]
    fn decide_native_route_passthrough_for_non_image_media() {
        // Audio/video/file media are not gated by the photo-specific predicate;
        // the caller's flag wins for them. (`process_one_media` then routes
        // them through audio transcription / file artifact paths.)
        let item = MediaItem {
            kind: media::MediaKind::Voice,
            file_id: "v".into(),
            file_name: "v.ogg".into(),
            mime_hint: Some("audio/ogg".into()),
            duration_secs: Some(5),
            emoji: None,
            size_hint: Some(50_000_000), // intentionally huge
            sticker_format: None,
        };
        assert!(decide_native_route(&item, true, 1));
        assert!(!decide_native_route(&item, false, u32::MAX));
    }

    #[test]
    fn media_processed_default_is_empty() {
        let mp = MediaProcessed::default();
        assert!(mp.text.is_empty());
        assert!(mp.native_images.is_empty());
    }

    #[test]
    fn native_image_struct_carries_mime_and_bytes() {
        let img = NativeImage {
            mime: "image/png".into(),
            bytes: vec![0x89, 0x50, 0x4E, 0x47],
        };
        assert_eq!(img.mime, "image/png");
        assert_eq!(img.bytes.len(), 4);
        let cloned = img.clone();
        assert_eq!(cloned.bytes, img.bytes);
    }

    #[test]
    fn extract_media_items_sticker_carries_emoji() {
        let mut m = base_private_chat();
        m["from"] = user(Some("alice"), "Alice", false);
        m["sticker"] = serde_json::json!({
            "file_id": "stk-1",
            "file_unique_id": "u",
            "width": 512,
            "height": 512,
            "type": "regular",
            "is_animated": false,
            "is_video": false,
            "emoji": "\u{1F525}"
        });
        let items = extract_media_items(&make_message(m));
        assert_eq!(items.len(), 1);
        assert_eq!(items[0].kind, media::MediaKind::Sticker);
        assert_eq!(items[0].emoji.as_deref(), Some("\u{1F525}"));
    }

    #[test]
    fn extract_static_sticker_routes_natively() {
        let mut m = base_private_chat();
        m["from"] = user(Some("alice"), "Alice", false);
        m["sticker"] = serde_json::json!({
            "file_id": "stk-static",
            "file_unique_id": "u",
            "width": 512, "height": 512,
            "type": "regular",
            "is_animated": false, "is_video": false,
            "file_size": 32_000,
        });
        let items = extract_media_items(&make_message(m));
        assert_eq!(items[0].sticker_format, Some(StickerFormat::Static));
        assert_eq!(items[0].mime_hint.as_deref(), Some("image/webp"));
        assert!(items[0].file_name.ends_with(".webp"));
        assert_eq!(items[0].size_hint, Some(32_000));
    }

    #[test]
    fn extract_animated_sticker_marked_non_native() {
        let mut m = base_private_chat();
        m["from"] = user(Some("alice"), "Alice", false);
        m["sticker"] = serde_json::json!({
            "file_id": "stk-anim",
            "file_unique_id": "u",
            "width": 512, "height": 512,
            "type": "regular",
            "is_animated": true, "is_video": false,
            "file_size": 12_000,
        });
        let items = extract_media_items(&make_message(m));
        assert_eq!(items[0].sticker_format, Some(StickerFormat::Animated));
        assert!(items[0].file_name.ends_with(".tgs"));
        assert_eq!(
            items[0].mime_hint.as_deref(),
            Some("application/x-tgsticker"),
            "animated stickers must NOT be advertised as image/* — vision providers will reject them"
        );
    }

    #[test]
    fn extract_video_sticker_marked_non_native() {
        let mut m = base_private_chat();
        m["from"] = user(Some("alice"), "Alice", false);
        m["sticker"] = serde_json::json!({
            "file_id": "stk-vid",
            "file_unique_id": "u",
            "width": 512, "height": 512,
            "type": "regular",
            "is_animated": false, "is_video": true,
            "file_size": 80_000,
        });
        let items = extract_media_items(&make_message(m));
        assert_eq!(items[0].sticker_format, Some(StickerFormat::Video));
        assert!(items[0].file_name.ends_with(".webm"));
        assert_eq!(items[0].mime_hint.as_deref(), Some("video/webm"));
    }

    #[test]
    fn extract_photo_carries_size_hint() {
        let mut m = base_private_chat();
        m["from"] = user(Some("alice"), "Alice", false);
        m["photo"] = serde_json::json!([
            { "file_id": "p1", "file_unique_id": "u1", "width": 90,  "height": 60,  "file_size": 4_000 },
            { "file_id": "p2", "file_unique_id": "u2", "width": 320, "height": 240, "file_size": 32_000 },
            { "file_id": "p3", "file_unique_id": "u3", "width": 800, "height": 600, "file_size": 200_000 },
        ]);
        let items = extract_media_items(&make_message(m));
        assert_eq!(items.len(), 1);
        assert_eq!(items[0].kind, media::MediaKind::Photo);
        // We pick the largest resolution → its size_hint should be 200_000.
        assert_eq!(items[0].size_hint, Some(200_000));
    }

    // ── TG HTTP mock (wiremock) ─────────────────────────────────────────
    //
    // These tests stand up a local HTTP server that pretends to be the
    // Telegram Bot API and verify our outgoing `sendMessage` plumbing
    // talks to it correctly. They exist to catch regressions in the
    // low-level `Bot`/`reqwest` layer — higher-level dispatch logic is
    // covered by the `album::tests` module.

    use wiremock::matchers::method;
    use wiremock::{Mock, MockServer, ResponseTemplate};

    /// Build a `Bot` pointing at the provided mock URL instead of
    /// `api.telegram.org`. Token is a throwaway.
    fn mock_bot(mock_url: &str) -> Bot {
        let url = reqwest::Url::parse(mock_url).unwrap();
        Bot::new("0:TEST_TOKEN").set_api_url(url)
    }

    #[tokio::test]
    async fn send_text_hits_mock_server_with_sendmessage() {
        let server = MockServer::start().await;
        // Match every POST. teloxide's URL shape is cosmetic for a mock —
        // what we care about is "the request reached the HTTP server".
        Mock::given(method("POST"))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "ok": true,
                "result": {
                    "message_id": 999,
                    "date": 0,
                    "chat": {"id": 1, "type": "private", "first_name": "x"},
                    "text": "ack"
                }
            })))
            .expect(1)
            .mount(&server)
            .await;

        let bot = mock_bot(&server.uri());
        let result = send_text(&bot, ChatId(1), None, "hello").await;
        // The test is about reaching the mock, not round-tripping the
        // full Message. Some teloxide versions are strict about the
        // serialized response shape; tolerate either Ok or a
        // deserialisation error as long as the request was sent.
        let _ = result;

        // `expect(1)` on Drop: wiremock panics if the mock wasn't hit
        // exactly once. Belt-and-braces: explicitly count received reqs.
        let received = server.received_requests().await.unwrap();
        assert_eq!(
            received.len(),
            1,
            "expected exactly one TG API request, got {}",
            received.len()
        );
    }

    #[tokio::test]
    async fn send_text_serialises_chat_id_and_text() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "ok": true,
                "result": {
                    "message_id": 12,
                    "date": 0,
                    "chat": {"id": 777, "type": "private", "first_name": "x"},
                    "text": "ok"
                }
            })))
            .mount(&server)
            .await;

        let bot = mock_bot(&server.uri());
        let _ = send_text(&bot, ChatId(777), None, "Привет мир 🌍").await;

        let received = server.received_requests().await.unwrap();
        assert_eq!(received.len(), 1);
        let body = std::str::from_utf8(&received[0].body).unwrap();
        assert!(
            body.contains("777"),
            "chat_id must appear in POST body: {body}"
        );
        // Non-ASCII payload must pass through unmangled (url-encoded or
        // JSON-escaped both count — we just need to see the logical text).
        assert!(
            body.contains("%D0%9F%D1%80%D0%B8") || body.contains("Привет"),
            "cyrillic/emoji must survive serialisation: {body}"
        );
    }
}
