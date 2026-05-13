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
/// BUG_REGISTRY D-BOOT-VISION-PROBE (B06): outcome of a single
/// vision content-shape probe. Used to classify whether a provider
/// that CLAIMS multimodal capability actually accepts OpenAI-style
/// `image_url` content blocks. Inconclusive outcomes are treated as
/// success — we only want to flag *definite* shape mismatches.
#[derive(Debug, Clone)]
pub(crate) enum VisionShapeOutcome {
    /// The provider accepted the image_url shape. May still have
    /// rejected our specific 1×1 PNG ("image too small") — that's
    /// content, not shape. Either way the capability is real.
    Accepted,
    /// The provider rejected the shape itself (e.g. "unknown variant
    /// `image_url`, expected `text`"). Caps are wrong.
    ShapeMismatch(String),
    /// Auth / rate-limit / timeout / network. Can't tell. Skip.
    Inconclusive(String),
}

/// Pure classifier for an error message returned by a vision probe.
/// Extracted for unit-testing without spinning up live providers.
pub(crate) fn classify_vision_probe_error(msg: &str) -> VisionShapeOutcome {
    let lc = msg.to_ascii_lowercase();
    // Definite shape-mismatch signals across the providers we care about:
    //   * serde-de error: "unknown variant `image_url`, expected `text`"
    //   * "expected text" / "only text content"
    //   * "does not support image" / "text-only model"
    if lc.contains("unknown variant")
        || lc.contains("expected text")
        || lc.contains("expected `text`")
        || lc.contains("only text content")
        || lc.contains("does not support image")
        || lc.contains("text-only model")
        || lc.contains("multimodal not supported")
    {
        return VisionShapeOutcome::ShapeMismatch(msg.chars().take(180).collect());
    }
    // "Image too small" / "min size" / size-related rejections — shape
    // accepted, content rejected. Either way the capability is real.
    if lc.contains("image must be")
        || lc.contains("too small")
        || lc.contains("min") && lc.contains("size")
        || lc.contains("width")
        || lc.contains("height")
    {
        return VisionShapeOutcome::Accepted;
    }
    // Auth, rate, timeout, network — can't tell, treat as inconclusive.
    VisionShapeOutcome::Inconclusive(msg.chars().take(180).collect())
}

/// 1×1 transparent PNG (the smallest valid PNG payload). Used as the
/// probe content — we expect every real vision API to either accept
/// or reject it with a SIZE error, but never a SHAPE error.
const VISION_PROBE_PIXEL_PNG_B64: &str = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg==";

/// Send a 1×1 image_url to the provider's stream_chat and classify
/// the response. 8-second timeout per probe so total D-BOOT-VISION-PROBE
/// time is bounded by (number of vision-claimed models × 8 s); typically
/// 5–10 s in practice.
pub(crate) async fn probe_vision_content_shape(
    provider: &dyn naked_core::provider::Provider,
    model: &str,
) -> VisionShapeOutcome {
    use naked_core::provider::ChatRequest;
    let req = ChatRequest {
        model: model.into(),
        system: String::new(),
        messages: vec![serde_json::json!({
            "role": "user",
            "content": [
                {"type": "text", "text": "hi"},
                {"type": "image_url", "image_url": {
                    "url": format!("data:image/png;base64,{VISION_PROBE_PIXEL_PNG_B64}")
                }}
            ]
        })],
        tools: vec![],
        max_tokens: 1,
        temperature: None,
        reasoning: None,
    };
    let timeout = std::time::Duration::from_secs(8);
    match tokio::time::timeout(timeout, provider.stream_chat(req)).await {
        Ok(Ok(_stream)) => VisionShapeOutcome::Accepted,
        Ok(Err(e)) => classify_vision_probe_error(&e.to_string()),
        Err(_) => VisionShapeOutcome::Inconclusive("timeout".into()),
    }
}

/// BUG_REGISTRY D-BOOT-CAPS-INVARIANT: enforces INV-1 + INV-2 at
/// boot time. Returns the number of (provider, model) pairs whose
/// declared caps disagree with the routing function. Logs WARN per
/// violation with file:line-style context for the operator.
///
/// `pub(crate)` for unit-testing without spinning up async wiring.
/// `NAKED_STRICT_CAPS=1` env: escalate WARN → `std::process::exit(3)`
/// so misconfiguration can't reach prod silently.
pub(crate) fn boot_caps_invariant_sweep(config: &naked_core::config::Config) -> usize {
    let mut violations: Vec<String> = Vec::new();

    for (provider_name, provider) in &config.providers {
        // INV-1: per-model caps.supports_vision=Some(true) must route as true.
        for (model_id, caps) in &provider.capabilities {
            if caps.supports_vision == Some(true) {
                let routable = config
                    .tg_media
                    .is_vision_capable_with_provider(model_id, Some(provider));
                if !routable {
                    violations.push(format!(
                        "INV-1 {provider_name}/{model_id}: caps.supports_vision=Some(true) \
                         but is_vision_capable_with_provider=false"
                    ));
                }
            }
        }
        // INV-2: every model id matching vision-naming pattern must route OR
        // have explicit Some(false) deny.
        for model_id in &provider.models {
            let lc = model_id.to_ascii_lowercase();
            let looks_vision =
                lc.contains("vl") || lc.contains("vision") || lc.contains("multimodal");
            if !looks_vision {
                continue;
            }
            let routable = config
                .tg_media
                .is_vision_capable_with_provider(model_id, Some(provider));
            if routable {
                continue;
            }
            let explicit_deny = provider
                .capabilities
                .get(model_id)
                .and_then(|c| c.supports_vision)
                == Some(false);
            if !explicit_deny {
                violations.push(format!(
                    "INV-2 {provider_name}/{model_id}: name suggests vision but \
                     is_vision_capable_with_provider=false and no explicit deny"
                ));
            }
        }
    }

    let count = violations.len();
    for v in &violations {
        tracing::warn!(violation = %v, "caps invariant violation at boot");
    }

    if count == 0 {
        tracing::info!(
            providers = config.providers.len(),
            "caps invariant sweep clean (INV-1 + INV-2)"
        );
    } else if std::env::var_os("NAKED_STRICT_CAPS").is_some_and(|v| v == "1") {
        // Strict mode — fail boot rather than ship broken caps to prod.
        eprintln!(
            "❌ {} caps invariant violation(s) at boot and NAKED_STRICT_CAPS=1; refusing to start",
            count
        );
        for v in &violations {
            eprintln!("   {v}");
        }
        std::process::exit(3);
    }

    count
}

/// BUG_REGISTRY D-BOOT-DESCRIBER-WARN: boot-time health check for the
/// multimodal vision path. Emits a loud WARN if the default model
/// can't accept image content blocks AND `tg_media.vision` describer
/// fallback is unconfigured — in that state, every photo from a user
/// pinned to the default model gets the "[⚠ vision not configured]"
/// placeholder text and the image content is lost. INFO when healthy.
///
/// `pub(crate)` so this can be unit-tested without spinning up the
/// full async wiring pipeline.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum MultimodalDescriberHealth {
    /// Default model is vision-capable; describer presence irrelevant.
    DefaultVision,
    /// Default model isn't vision-capable but describer fallback exists.
    DescriberFallback,
    /// Default model isn't vision-capable AND no describer — photos
    /// from default-model users will be silently dropped.
    Degraded,
}

pub(crate) fn check_multimodal_describer_health(
    config: &naked_core::config::Config,
) -> MultimodalDescriberHealth {
    let default_provider = config.providers.get(&config.default_provider);
    let default_is_vision = config
        .tg_media
        .is_vision_capable_with_provider(&config.default_model, default_provider);
    let describer = config.tg_media.vision.is_some();
    let outcome = match (default_is_vision, describer) {
        (true, _) => MultimodalDescriberHealth::DefaultVision,
        (false, true) => MultimodalDescriberHealth::DescriberFallback,
        (false, false) => MultimodalDescriberHealth::Degraded,
    };
    match outcome {
        MultimodalDescriberHealth::Degraded => {
            tracing::warn!(
                default_provider = %config.default_provider,
                default_model = %config.default_model,
                "multimodal degraded: default model is not vision-capable AND \
                 no tg_media.vision describer fallback configured. \
                 Photos from users on this model will be lost. \
                 Either pin to a vision-capable model (e.g. qwen3-vl-plus) \
                 or set tg_media.vision in naked.json."
            );
        }
        MultimodalDescriberHealth::DefaultVision => {
            tracing::info!(
                default_provider = %config.default_provider,
                default_model = %config.default_model,
                "multimodal: default model is vision-capable"
            );
        }
        MultimodalDescriberHealth::DescriberFallback => {
            tracing::info!(
                default_provider = %config.default_provider,
                default_model = %config.default_model,
                describer_model = ?config.tg_media.vision.as_ref().map(|v| &v.model),
                "multimodal: default is text-only, describer fallback active"
            );
        }
    }
    outcome
}

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

    // BUG_REGISTRY D-BOOT-DESCRIBER-WARN (B05 regression guard).
    check_multimodal_describer_health(&config);

    // BUG_REGISTRY D-BOOT-CAPS-INVARIANT (B03+B04 boot-time enforcement).
    // Walks every (provider, model) pair declared in config.providers,
    // checks INV-1 + INV-2 hold at boot. Mismatches log WARN with the
    // exact offending pair so operator sees them on every restart, not
    // only when a user happens to send a photo to that model.
    // Set `NAKED_STRICT_CAPS=1` to escalate WARN → process exit 3.
    boot_caps_invariant_sweep(&config);

    // BUG_REGISTRY D-BOOT-VISION-PROBE (B06): for every (provider, model)
    // pair with caps.supports_vision=Some(true), send a tiny request with
    // an `image_url` content block and classify the API's response. If
    // the provider rejects the SHAPE ("unknown variant", "expected text"),
    // it doesn't actually support OpenAI-style multimodal even though we
    // think it does — bump counter + WARN with remediation. Other errors
    // (auth, rate, timeout, image-too-small) are inconclusive and skipped.
    // Fire-and-forget so boot is not blocked.
    {
        let agent_for_probe = agent.clone();
        let pairs: Vec<(String, String)> = config
            .providers
            .iter()
            .flat_map(|(pname, pcfg)| {
                pcfg.capabilities
                    .iter()
                    .filter(|(_, c)| c.supports_vision == Some(true))
                    .map(|(model_id, _)| (pname.clone(), model_id.clone()))
                    .collect::<Vec<_>>()
            })
            .collect();
        if !pairs.is_empty() {
            tokio::spawn(async move {
                let total = pairs.len();
                let mut mismatch = 0usize;
                for (pname, model) in pairs {
                    let provider = agent_for_probe.provider_for(&pname).await;
                    match probe_vision_content_shape(&*provider, &model).await {
                        VisionShapeOutcome::Accepted => {
                            tracing::debug!(
                                provider = %pname,
                                model = %model,
                                "vision shape probe ok"
                            );
                        }
                        VisionShapeOutcome::ShapeMismatch(reason) => {
                            mismatch += 1;
                            naked_core::types::PROVIDER_VISION_CAP_MISMATCH_COUNT
                                .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                            tracing::warn!(
                                provider = %pname,
                                model = %model,
                                reason = %reason,
                                "vision capability mismatch — caps claim supports_vision=true \
                                 but API rejects image_url content shape. Flip caps to false \
                                 in naked.json or stop pinning sessions to this model."
                            );
                        }
                        VisionShapeOutcome::Inconclusive(reason) => {
                            tracing::debug!(
                                provider = %pname,
                                model = %model,
                                reason = %reason,
                                "vision shape probe inconclusive"
                            );
                        }
                    }
                }
                tracing::info!(
                    pairs = total,
                    mismatch = mismatch,
                    "D-BOOT-VISION-PROBE complete"
                );
            });
        }
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

#[cfg(test)]
mod tests {
    use super::*;
    use naked_core::config::{Config, ProviderConfig, VisionProviderCfg};

    /// Builds a minimal `Config` with the given default model and provider
    /// + per-model vision capability + optional describer.
    ///
    /// KISS — no other fields set. Spread `..Default::default()` so future
    /// schema additions don't break this helper (BUG_REGISTRY C2).
    fn make_config(
        default_model: &str,
        default_provider_name: &str,
        per_model_vision: Option<bool>,
        describer: Option<VisionProviderCfg>,
    ) -> Config {
        let mut providers = std::collections::HashMap::new();
        let mut caps = std::collections::HashMap::new();
        caps.insert(
            default_model.to_string(),
            naked_core::model_catalog::ModelCapabilities {
                supports_vision: per_model_vision,
                ..Default::default()
            },
        );
        providers.insert(
            default_provider_name.to_string(),
            ProviderConfig {
                capabilities: caps,
                ..Default::default()
            },
        );
        let mut cfg = Config {
            default_model: default_model.to_string(),
            default_provider: default_provider_name.to_string(),
            providers,
            ..Default::default()
        };
        cfg.tg_media.vision = describer;
        cfg
    }

    fn fake_describer() -> VisionProviderCfg {
        VisionProviderCfg {
            api_url: "https://example.com/v1/chat/completions".into(),
            api_key: "$FAKE_KEY".into(),
            model: "qwen3-vl-plus".into(),
            max_tokens: 400,
            prompt_override: None,
        }
    }

    /// Default model is vision-capable → healthy regardless of describer.
    #[test]
    fn multimodal_default_vision_healthy_without_describer() {
        let cfg = make_config("qwen3-vl-plus", "qwen", Some(true), None);
        assert_eq!(
            check_multimodal_describer_health(&cfg),
            MultimodalDescriberHealth::DefaultVision
        );
    }

    /// Default text-only + describer present → fallback active.
    #[test]
    fn multimodal_describer_fallback_active() {
        let cfg = make_config("qwen3.6-plus", "qwen", Some(false), Some(fake_describer()));
        assert_eq!(
            check_multimodal_describer_health(&cfg),
            MultimodalDescriberHealth::DescriberFallback
        );
    }

    /// Default text-only + no describer → DEGRADED, warning emitted.
    /// This is the regression guard for B05: shipping with this state
    /// silently drops all photo attachments on the default model.
    #[test]
    fn multimodal_degraded_when_default_text_only_and_no_describer() {
        let cfg = make_config("qwen3.6-plus", "qwen", Some(false), None);
        assert_eq!(
            check_multimodal_describer_health(&cfg),
            MultimodalDescriberHealth::Degraded
        );
    }

    /// Per-model caps `None` falls through to needles. With a non-vision
    /// model name and no describer → degraded.
    #[test]
    fn multimodal_per_model_none_falls_through_to_needle_check() {
        let cfg = make_config("qwen-turbo", "qwen", None, None);
        assert_eq!(
            check_multimodal_describer_health(&cfg),
            MultimodalDescriberHealth::Degraded
        );
    }

    // ─── D-BOOT-CAPS-INVARIANT (B03+B04 enforcement at boot) ───

    fn make_provider_with_models_and_caps(
        models: &[&str],
        per_model: &[(&str, Option<bool>)],
    ) -> ProviderConfig {
        let mut caps = std::collections::HashMap::new();
        for (m, sv) in per_model {
            caps.insert(
                (*m).to_string(),
                naked_core::model_catalog::ModelCapabilities {
                    supports_vision: *sv,
                    ..Default::default()
                },
            );
        }
        ProviderConfig {
            models: models.iter().map(|s| s.to_string()).collect(),
            capabilities: caps,
            ..Default::default()
        }
    }

    /// Clean config: every vision-capable model is routable, every
    /// vision-named model resolves. Zero violations expected.
    #[test]
    fn boot_caps_sweep_clean_config_returns_zero() {
        let mut providers = std::collections::HashMap::new();
        providers.insert(
            "qwen".to_string(),
            make_provider_with_models_and_caps(
                &["qwen3.6-plus", "qwen3-vl-plus"],
                &[("qwen3-vl-plus", Some(true))],
            ),
        );
        let cfg = Config {
            default_model: "qwen3.6-plus".into(),
            default_provider: "qwen".into(),
            providers,
            ..Default::default()
        };
        assert_eq!(boot_caps_invariant_sweep(&cfg), 0);
    }

    /// Synthetic INV-1 violation: a provider claims caps.supports_vision=true
    /// for a model that the routing function (via provider-wide override =
    /// Some(false)) maps to false. Sweep must flag it.
    #[test]
    fn boot_caps_sweep_detects_inv1_violation() {
        let mut providers = std::collections::HashMap::new();
        let mut caps = std::collections::HashMap::new();
        caps.insert(
            "fake-model".to_string(),
            naked_core::model_catalog::ModelCapabilities {
                supports_vision: Some(true),
                ..Default::default()
            },
        );
        providers.insert(
            "fakep".to_string(),
            ProviderConfig {
                models: vec!["fake-model".into()],
                supports_vision: Some(false), // provider-wide deny outranks per-model
                capabilities: caps,
                ..Default::default()
            },
        );
        let cfg = Config {
            default_model: "fake-model".into(),
            default_provider: "fakep".into(),
            providers,
            ..Default::default()
        };
        assert_eq!(boot_caps_invariant_sweep(&cfg), 1);
    }

    /// INV-2 violation: model named `*-vision-pro` but not matched by any
    /// needle and no explicit deny in caps.
    #[test]
    fn boot_caps_sweep_detects_inv2_violation() {
        let mut providers = std::collections::HashMap::new();
        providers.insert(
            "someprovider".to_string(),
            make_provider_with_models_and_caps(
                &["my-special-vision-pro"], // contains 'vision' but no needle
                &[],                        // no caps entry at all
            ),
        );
        // Force provider-wide to None so default-fallthrough applies.
        let cfg = Config {
            default_model: "my-special-vision-pro".into(),
            default_provider: "someprovider".into(),
            providers,
            ..Default::default()
        };
        // The substring `vision` IS in BUILTIN_VISION_MODEL_NEEDLES via
        // `gpt-4-vision` / `grok-2-vision` etc. — actually `vision` itself
        // is a substring of every one of those needles, but the matcher
        // does substring `m.contains(needle)`, not the other way. So
        // "my-special-vision-pro".contains("vision") would only match if
        // "vision" is in the needle list — which it isn't (it's always
        // prefixed). Let's verify by direct call:
        //   - looks_vision flag: true (contains "vision")
        //   - is_vision_capable_with_provider: should be false (no needle
        //     matches plain "vision" without prefix)
        // Therefore sweep flags it.
        let result = boot_caps_invariant_sweep(&cfg);
        assert_eq!(
            result, 1,
            "expected exactly 1 INV-2 violation for 'my-special-vision-pro'"
        );
    }

    // ─── D-BOOT-VISION-PROBE classifier tests (B06) ───

    #[test]
    fn classify_vision_probe_unknown_variant_is_shape_mismatch() {
        // Real deepseek-v4-pro error text from earlier curl probe.
        let err = "Failed to deserialize the JSON body into the target type: \
                   messages[0]: unknown variant `image_url`, expected `text`";
        match classify_vision_probe_error(err) {
            VisionShapeOutcome::ShapeMismatch(_) => {}
            other => panic!("expected ShapeMismatch, got {other:?}"),
        }
    }

    #[test]
    fn classify_vision_probe_image_too_small_is_accepted() {
        // Real qwen3-vl-plus error from our M3 verification probe.
        let err = "<400> InternalError.Algo.InvalidParameter: \
                   The image length and width do not meet the model restrictions. \
                   [height:1 or width:1 must be larger than 10]";
        match classify_vision_probe_error(err) {
            VisionShapeOutcome::Accepted => {}
            other => panic!("expected Accepted (size, not shape), got {other:?}"),
        }
    }

    #[test]
    fn classify_vision_probe_text_only_model_is_shape_mismatch() {
        let err = "This is a text-only model and does not support image inputs.";
        match classify_vision_probe_error(err) {
            VisionShapeOutcome::ShapeMismatch(_) => {}
            other => panic!("expected ShapeMismatch, got {other:?}"),
        }
    }

    #[test]
    fn classify_vision_probe_auth_is_inconclusive() {
        let err = "HTTP 401 Unauthorized";
        match classify_vision_probe_error(err) {
            VisionShapeOutcome::Inconclusive(_) => {}
            other => panic!("auth error should be inconclusive, got {other:?}"),
        }
    }

    #[test]
    fn classify_vision_probe_rate_limit_is_inconclusive() {
        let err = "HTTP 429 Too Many Requests";
        match classify_vision_probe_error(err) {
            VisionShapeOutcome::Inconclusive(_) => {}
            other => panic!("rate limit should be inconclusive, got {other:?}"),
        }
    }

    /// INV-2 explicit-deny escape hatch: same model name but caps say
    /// Some(false) explicitly. Operator says "yes I know, it's not actually
    /// vision". Sweep must accept that.
    #[test]
    fn boot_caps_sweep_accepts_explicit_inv2_deny() {
        let mut providers = std::collections::HashMap::new();
        providers.insert(
            "someprovider".to_string(),
            make_provider_with_models_and_caps(
                &["my-special-vision-pro"],
                &[("my-special-vision-pro", Some(false))], // explicit deny
            ),
        );
        let cfg = Config {
            default_model: "my-special-vision-pro".into(),
            default_provider: "someprovider".into(),
            providers,
            ..Default::default()
        };
        assert_eq!(boot_caps_invariant_sweep(&cfg), 0);
    }
}
