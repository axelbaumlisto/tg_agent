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

use teloxide::prelude::*;
use teloxide::types::{
    CallbackQuery, InlineKeyboardButton, InlineKeyboardMarkup, MessageId, ParseMode, ThreadId,
};
use tokio::sync::{RwLock, oneshot};

use naked_core::AgentCore;
use naked_core::config::Config;
use naked_core::types::{AgentEvent, AgentHandle, Permission, PermissionResponse, TurnUsage};

use channel_map::{ChannelSessionMap, format_tg_channel_id};

const MAX_TG_MSG: usize = TG_MSG_LIMIT;
const TYPING_INTERVAL: Duration = Duration::from_secs(3);
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
        // Always reply-thread: in groups for context anchoring,
        // in DMs for visual question→answer pairing.
        Self {
            chat_id: msg.chat.id,
            thread_id: msg.thread_id,
            reply_to: Some(msg.id),
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

// ── Reply helpers (DRY: replaces 43× repeated send_message chains) ──────────

/// Send a plain-text message. Handles thread_id automatically.
async fn reply_text(bot: &Bot, ctx: &ChatCtx, text: impl Into<String>) -> ResponseResult<Message> {
    bot.send_message(ctx.chat_id, text)
        .maybe_thread(ctx.thread_id)
        .await
}

/// Send an HTML-formatted message. Handles thread_id + ParseMode::Html.
async fn reply_html(bot: &Bot, ctx: &ChatCtx, text: impl Into<String>) -> ResponseResult<Message> {
    bot.send_message(ctx.chat_id, text)
        .parse_mode(teloxide::types::ParseMode::Html)
        .maybe_thread(ctx.thread_id)
        .await
}

/// Send an HTML message with an inline keyboard.
async fn reply_html_kb(
    bot: &Bot,
    ctx: &ChatCtx,
    text: impl Into<String>,
    kb: teloxide::types::InlineKeyboardMarkup,
) -> ResponseResult<Message> {
    bot.send_message(ctx.chat_id, text)
        .parse_mode(teloxide::types::ParseMode::Html)
        .reply_markup(kb)
        .maybe_thread(ctx.thread_id)
        .await
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
        .add_directive("naked=info".parse().expect("static directive"));

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

    let mcp_failures = agent.init_mcp().await;

    // B6: Register built-in context hook — inject short git status.
    // Helps the model know if there are uncommitted changes.
    agent
        .hooks()
        .on_context(std::sync::Arc::new(
            |msgs: &mut Vec<naked_core::types::ConversationMessage>| {
                // Only inject if the first message is a system prompt
                // and we're in a git repo (workspace is set in system prompt).
                if msgs.is_empty() {
                    return;
                }
                // Quick check with timeout — skip if git is slow or not a repo
                let output = std::process::Command::new("git")
                    .args(["diff", "--stat", "HEAD"])
                    .stdout(std::process::Stdio::piped())
                    .stderr(std::process::Stdio::null())
                    .output();
                if let Ok(out) = output {
                    let stat = String::from_utf8_lossy(&out.stdout);
                    let stat = stat.trim();
                    if !stat.is_empty() && stat.len() < 500 {
                        msgs.push(naked_core::types::ConversationMessage::user(format!(
                            "[git diff --stat]\n{stat}"
                        )));
                    }
                }
            },
        ))
        .await;

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
        .expect("reqwest client");
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

    // ── A7: Send MCP startup diagnostics to owner chat ────────────────
    if !mcp_failures.is_empty() && !config.allowed_chat_ids.is_empty() {
        let owner_chat = ChatId(config.allowed_chat_ids[0]);
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
                crate::tg_markup::escape_html(&f.name),
                crate::tg_markup::escape_html(&err_short),
            ));
        }
        let text = lines.join("\n");
        let _ = bot
            .send_message(owner_chat, &text)
            .parse_mode(teloxide::types::ParseMode::Html)
            .await;
    }

    if config.allowed_chat_ids.is_empty() {
        tracing::warn!(
            "allowed_chat_ids is empty — ALL messages will be rejected! Add your chat IDs to naked.json."
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
    let rate_limiter = RATE_LIMITER.clone();
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
        BotCommand::new("status", "Session status, usage, cost"),
        BotCommand::new("compact", "Compact session history"),
        BotCommand::new("new", "Start a new session"),
        BotCommand::new("sessions", "List active sessions"),
        BotCommand::new("stop", "Cancel running task"),
        BotCommand::new("abort", "Cancel running task (alias)"),
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
        BotCommand::new("health", "Provider health & key status"),
        BotCommand::new("commit", "Git commit modified files"),
        BotCommand::new("remote", "Switch to SSH remote host"),
        BotCommand::new("help", "Show all commands"),
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

// ── Media (extracted to media_dispatch.rs) ───────────────────────────────────
mod media_dispatch;
use media_dispatch::{extract_media_items, process_media_items, send_text};

// ── Message handler (extracted to message_handler.rs) ────────────────────────
mod message_handler;
pub(crate) use message_handler::BotDeps;

// ── Callback handler (extracted to callbacks.rs) ────────────────────────────

pub(crate) fn provider_models(config: &Config, provider_name: &str) -> Vec<String> {
    config
        .providers
        .get(provider_name)
        .map(|pc| pc.models_with_aliases())
        .unwrap_or_default()
}

mod callbacks;
use callbacks::handle_callback;

// ── Streaming (extracted to streaming.rs) ────────────────────────────────────
#[path = "streaming_mod/mod.rs"]
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

/// Steer senders: (chat_id, thread_id) → Sender<SteerMessage>.
/// Populated when a streaming turn starts, removed when it ends.
type SteerSenderMap =
    HashMap<(i64, Option<i32>), tokio::sync::mpsc::Sender<naked_core::types::SteerMessage>>;
pub(crate) static STEER_SENDERS: LazyLock<tokio::sync::RwLock<SteerSenderMap>> =
    LazyLock::new(|| tokio::sync::RwLock::new(HashMap::new()));

/// Global rate limiter instance — accessible from commands.rs for /metrics.
pub(crate) static RATE_LIMITER: LazyLock<naked_tg::rate_limit::RateLimiter> =
    LazyLock::new(naked_tg::rate_limit::RateLimiter::new);

mod commands;
mod fmt_utils;
use commands::{
    drop_slash_for_persona, get_or_create_session, handle_command,
    research::launch_research_run_with_ui,
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
