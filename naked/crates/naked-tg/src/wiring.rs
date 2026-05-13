//! Dependency injection: build all components needed by the bot.
//!
//! [`build`] constructs every stateful piece (provider, agent, schedulers,
//! MCP, channel-map, bot client, bot identity) and returns a [`WiredBot`]
//! that the event loop in [`crate::runtime`] can use directly.

use std::sync::Arc;
use std::time::Duration;

use teloxide::prelude::*;

use naked_core::AgentCore;
use naked_core::config::Config;

use crate::channel_map::ChannelSessionMap;
use crate::shared::{RATE_LIMITER, memory_scheduler, research_scheduler};

/// All state produced by [`build`] and consumed by the event loop.
pub(crate) struct WiredBot {
    pub(crate) agent: Arc<AgentCore>,
    pub(crate) channel_map: Arc<ChannelSessionMap>,
    pub(crate) config: Config,
    pub(crate) bot: Bot,
    pub(crate) bot_token: Arc<String>,
    pub(crate) bot_identity: Arc<naked_tg::bot_identity::BotIdentity>,
    pub(crate) http_client: Arc<reqwest::Client>,
    pub(crate) base_url: Arc<String>,
    pub(crate) tg_attach_queue: naked_tg::tg_attach::AttachmentQueue,
    pub(crate) mcp_failures: Vec<naked_core::mcp::client::McpConnectFailure>,
    pub(crate) rate_limiter: naked_tg::rate_limit::RateLimiter,
    // Kept alive for the lifetime of the process:
    pub(crate) _scheduler_lock: Option<naked_tg::scheduler_lock::SchedulerLock>,
    pub(crate) _memory_scheduler: naked_tg::memory_scheduler::MemoryScheduler,
    /// Shared liveness registry. The polling loop and the research
    /// scheduler both `beat` into this; the watchdog arbiter
    /// (`spawn_watchdog_with_liveness` in runtime) reads it.
    /// Created in `build()` so every long-running task can be wired
    /// to the same instance — single source of truth (DRY).
    pub(crate) liveness: Arc<naked_core::liveness::LivenessRegistry>,
}

/// Build the complete DI graph and return a ready-to-run [`WiredBot`].
///
/// Must be called **after** tracing is initialised (in [`crate::bootstrap`]).
pub(crate) async fn build() -> WiredBot {
    let config = Config::load().expect("Failed to load config");
    let provider =
        naked_core::build_provider_from_config(&config).expect("Failed to build provider");
    let agent = Arc::new(AgentCore::new(config.clone(), provider));
    agent.init_self_ref();

    // R2 of PLAN_RESILIENCE_v1: fire-and-forget boot-time key audit
    // for EVERY configured provider. Each provider's chain (key
    // rotation + nested fallbacks) gets probed in parallel.
    // Permanent failures (401/402) bump
    // `naked_core_provider_permanent_blacklist_total` and remove
    // the key from rotation before the first real turn would hit
    // it. The audit runs in the background so boot time is
    // unaffected; first few turns may still try a dead key.
    {
        let agent_for_audit = agent.clone();
        let provider_names: Vec<String> = config.providers.keys().cloned().collect();
        tokio::spawn(async move {
            // Sequential rather than join_all — avoids pulling in
            // futures_util at this layer. Audits are fast (5s
            // timeout per probe) so serial is fine.
            for name in provider_names {
                let p = agent_for_audit.provider_for(&name).await;
                p.audit_keys_on_boot().await;
            }
            tracing::info!("R2 boot-time provider audit complete for all configured providers");
        });
    }

    // PLAN_QUALITY_v1 wiring (T2/T5/T6): install pluggable managers.
    // Each is opt-in via config / disk presence; missing = silent
    // off-path (zero overhead).
    {
        // T2 LSP manager. ENABLED by default (post-edit compiler
        // feedback is the biggest quality multiplier in
        // PLAN_QUALITY_v1). Lazy: LSP servers spawn on first edit
        // per language. Operators who want to disable can set
        // NAKED_LSP_DISABLED=1.
        let lsp_cfg = naked_core::lsp::LspConfig::default();
        tracing::info!(
            lsp_enabled = lsp_cfg.enabled,
            lsp_warn_included = lsp_cfg.include_warnings,
            lsp_max_diagnostics = lsp_cfg.max_diagnostics_per_file,
            "PLAN_QUALITY_v1 LSP manager configured"
        );
        let lsp = std::sync::Arc::new(naked_core::lsp::LspManager::new(lsp_cfg));
        agent.set_lsp(lsp);

        // T6 lifecycle hooks: load ~/.naked/hooks.json if present.
        // Empty file / missing path = no hooks installed (silent).
        let hooks = std::sync::Arc::new(naked_core::lifecycle_hooks::LifecycleHookRunner::new());
        hooks.load_default().await;
        agent.set_lifecycle_hooks(hooks);

        // T5 permission ruleset: load ~/.naked/permissions.json if
        // present. Empty file / missing path = empty ruleset = every
        // tool falls through to the existing UI prompt (Ask).
        let ruleset = naked_core::permissions::Store::load();
        let permissions = std::sync::Arc::new(tokio::sync::RwLock::new(ruleset));
        agent.set_permissions(permissions);
        tracing::info!("PLAN_QUALITY_v1 wiring installed: lsp + hooks + permissions");
    }

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

    // F2 of PLAN_NEXT_SESSION: shared liveness registry — the
    // polling loop and the scheduler both beat into it, the watchdog
    // arbiter (in runtime.rs) reads it.
    let liveness = Arc::new(naked_core::liveness::LivenessRegistry::new());
    liveness.register("tg_polling.tick");
    liveness.register("scheduler.tick");
    // F4: install the singleton so `/health` (and any future
    // ops-side reader) can reach the registry without us having to
    // thread it through every command handler. set() returns Err if
    // already installed; we ignore — a duplicate install during
    // tests is harmless.
    let _ = crate::shared::LIVENESS_REGISTRY.set(liveness.clone());
    // Force-stamp process start time so `/health` uptime is honest
    // (LazyLock is initialised on first deref).
    let _ = *crate::shared::PROCESS_STARTED_AT;

    if config.research.enabled && _scheduler_lock.is_some() {
        let scheduler_cfg = research_scheduler::SchedulerConfig {
            verify_by_default: config.research.verify_by_default,
            max_verification_rounds: config.research.gatekeeper.max_rounds,
            max_concurrent_runs: config.research.max_concurrent_runs.max(1),
            task_timeout: std::time::Duration::from_secs(config.research.task_timeout_seconds),
            max_retries_before_alert: config.research.max_retries_before_alert,
            liveness: Some(liveness.clone()),
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

    // Reconstruct the naked home dir (same formula as bootstrap.rs).
    let home = std::env::var("HOME").unwrap_or_else(|_| ".".into());
    let naked_dir = std::path::PathBuf::from(&home).join(".naked");

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
    // restored.
    if let Err(e) = channel_map.flush().await {
        tracing::warn!("channel_map: initial flush failed: {e:#}");
    }

    // Periodic snapshot writer: cheap atomic temp+rename every 30s.
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

    let bot_token = config
        .telegram
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

    // Manual polling loop client — avoids teloxide's Dispatcher/Polling
    // which conflicts with stale getUpdates connections from other processes.
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
    crate::media::sweep_old_artifacts(&config.workspace, config.tg_media.artifact_retention_days);

    // Drop any pending updates + delete webhook on startup
    let _ = client
        .post(format!("{base}/deleteWebhook"))
        .json(&serde_json::json!({"drop_pending_updates": true}))
        .send()
        .await;
    tracing::info!("Webhook cleared, starting polling loop");

    let rate_limiter = RATE_LIMITER.clone();

    WiredBot {
        agent,
        channel_map,
        config,
        bot,
        bot_token: bot_token_arc,
        bot_identity,
        http_client,
        base_url,
        tg_attach_queue,
        mcp_failures,
        rate_limiter,
        _scheduler_lock,
        _memory_scheduler,
        liveness,
    }
}

/// Register bot commands with Telegram so the slash-menu is populated.
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
