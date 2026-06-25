//! Dependency injection: build all components needed by the bot.
//!
//! [`build`] constructs every stateful piece (provider, agent, schedulers,
//! MCP, channel-map, bot client, bot identity) and returns a [`WiredBot`]
//! that the event loop in [`crate::runtime`] can use directly.

use std::sync::Arc;
use teloxide::prelude::*;

use naked_core::AgentCore;
use naked_core::config::Config;

use crate::shared::{RATE_LIMITER, memory_scheduler};
use naked_tg::channel_map::ChannelSessionMap;

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
    pub(crate) research_scheduler:
        Option<Arc<crate::shared::research_scheduler::ResearchScheduler>>,
    pub(crate) _scheduler_lock: Option<naked_tg::scheduler_lock::SchedulerLock>,
    pub(crate) _memory_scheduler: naked_tg::memory_scheduler::MemoryScheduler,
    /// Shared liveness registry. The polling loop and the research
    /// scheduler both `beat` into this; the watchdog arbiter
    /// (`spawn_watchdog_with_liveness` in runtime) reads it.
    /// Created in `build()` so every long-running task can be wired
    /// to the same instance — single source of truth (DRY).
    pub(crate) liveness: Arc<naked_core::liveness::LivenessRegistry>,
}

pub(crate) mod channel_state;
pub(crate) mod context_hooks;
pub(crate) mod health;
pub(crate) mod invariants;
pub(crate) mod quality;
pub(crate) mod research_scheduler;
pub(crate) mod scheduler_lock;
pub(crate) mod telegram;

use health::{
    MultimodalDescriberHealth, VisionShapeOutcome, audit_all_providers,
    audit_deduped_provider_targets, boot_audit_dedup_enabled, boot_caps_invariant_sweep,
    check_multimodal_describer_health, probe_deduped_target_with_created_provider,
    probe_vision_content_shape,
};
use invariants::{
    check_config_symlink_invariant, check_system_prompt_paths, populate_novnc_ip_allowlist,
    spawn_config_mtime_watcher,
};

pub(crate) async fn build() -> WiredBot {
    let (config, config_source) = Config::load_with_source_bytes().expect("Failed to load config");
    let loaded_config_hash = naked_tg::config_hash::loaded_hash_from_source(config_source, &config);
    tracing::info!(
        hash = %loaded_config_hash.hash_hex,
        source = %loaded_config_hash.source,
        "loaded config hash"
    );
    let _ = crate::shared::CONFIG_LOADED_HASH.set(loaded_config_hash);
    let provider =
        naked_core::build_provider_from_config(&config).expect("Failed to build provider");
    let agent = Arc::new(AgentCore::new(config.clone(), provider));
    agent.init_self_ref();

    // R2 of PLAN_RESILIENCE_v1 + B63: fire-and-forget boot-time key audit
    // for EVERY configured provider name, but in default mode probe each
    // physical resolved credential once. The legacy per-name chain audit is
    // kept behind NAKED_BOOT_AUDIT_DEDUP=0 for one-restart rollback.
    if boot_audit_dedup_enabled() {
        let provider_entries: Vec<(String, naked_core::config::ProviderConfig)> = config
            .providers
            .iter()
            .map(|(name, cfg)| (name.clone(), cfg.clone()))
            .collect();
        tokio::spawn(async move {
            let summary = audit_deduped_provider_targets(provider_entries, |target| async move {
                probe_deduped_target_with_created_provider(target).await
            })
            .await;
            tracing::info!(
                providers_seen = summary.providers_seen,
                targets_probed = summary.targets_probed,
                targets_skipped_dup = summary.targets_skipped_dup,
                known_dead = summary.known_dead,
                inconclusive = summary.inconclusive,
                "R2 boot-time provider audit complete for all configured providers"
            );
        });
    } else {
        let agent_for_audit = agent.clone();
        let provider_names: Vec<String> = config.providers.keys().cloned().collect();
        tokio::spawn(async move {
            audit_all_providers(provider_names, |name| {
                let agent = agent_for_audit.clone();
                async move { agent.provider_for(&name).await }
            })
            .await;
            tracing::info!(
                mode = "legacy",
                "R2 boot-time provider audit complete for all configured providers"
            );
        });
    }

    quality::install_quality_managers(&agent).await;

    // BUG_REGISTRY D-BOOT-CONFIG-SYMLINK (B41 regression guard).
    // Asserts that `naked/naked.json` is a symlink pointing at
    // `../state/naked.json` per AGENTS.md layout. When the symlink
    // gets replaced by a stale copy (B41), bot can load wrong config.
    check_config_symlink_invariant();

    // BUG_REGISTRY D-CONFIG-MTIME-WATCH (B42 detector): record boot
    // snapshot of state/naked.json (mtime+sha256), spawn periodic
    // poller that bumps CONFIG_EXTERNAL_WRITE_COUNT + WARN if changed.
    spawn_config_mtime_watcher();

    // BUG_REGISTRY D-CHECK-SYSPROMPT-PATHS (B38/B37 regression guard).
    // Scans the loaded system_prompt for file paths and asserts each
    // exists. Catches stale references like ~/.zeroclaw/workspace/
    // before the model sees them and confabulates.
    check_system_prompt_paths();

    // BUG_REGISTRY D-VALIDATE-IP-TOKENS (B37 stream-level guard):
    // pre-populate the IP allow-list from `novnc.sh url` so the
    // stream pipeline can validate outgoing noVNC mentions against
    // a known-good set. Fire-and-forget — if novnc.sh is unreachable,
    // allow-list stays empty and IP validation is a no-op (fail-open).
    populate_novnc_ip_allowlist();

    // BUG_REGISTRY D-BOOT-DESCRIBER-WARN (B05 regression guard).
    let describer_health = check_multimodal_describer_health(&config);
    crate::metrics::set_config_describer_missing(matches!(
        describer_health,
        MultimodalDescriberHealth::Degraded
    ));

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

    let _scheduler_lock = scheduler_lock::acquire_scheduler_lock(&config);

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

    // T2.6 (PLAN_RESEARCH_AGENT_FLOW_v1): scheduler init moved BELOW
    // bot + channel_map creation so we can capture them in the
    // SyntheticDispatchFn closure. See [`research_scheduler_with_dispatch`]
    // helper invoked after bot is ready. Hook installation moved with
    // it — nothing between this point and bot creation requires the
    // research scheduler hook (memory_scheduler + hooks/init_mcp are
    // independent of research).

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

    context_hooks::install_builtin_context_hooks(&agent).await;

    let restored = agent.restore_sessions().await.unwrap_or_default();
    if !restored.is_empty() {
        tracing::info!("Restored {} session(s)", restored.len());
    }

    let channel_map = channel_state::restore_channel_state(&agent).await;

    let telegram = telegram::boot_telegram_client(&config).await;
    let bot = telegram.bot;
    let bot_token_arc = telegram.bot_token;
    let bot_identity = telegram.bot_identity;
    let http_client = telegram.http_client;
    let base_url = telegram.base_url;

    // Startup janitor: nuke old media artifacts outside the retention window.
    crate::media::sweep_old_artifacts(&config.workspace, config.tg_media.artifact_retention_days);

    telegram::clear_webhook(&http_client, &base_url).await;

    let stream_deps = research_scheduler::StreamDeps {
        bot: bot.clone(),
        config: config.clone(),
        http_client: http_client.clone(),
        base_url: base_url.clone(),
        tg_attach_queue: tg_attach_queue.clone(),
        bot_token: bot_token_arc.clone(),
        bot_identity: bot_identity.clone(),
    };
    let research_scheduler_handle = research_scheduler::start_research_scheduler(
        &agent,
        &config,
        &channel_map,
        &liveness,
        _scheduler_lock.is_some(),
        Some(stream_deps),
    );

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
        research_scheduler: research_scheduler_handle,
        _scheduler_lock,
        _memory_scheduler,
        liveness,
    }
}

#[cfg(test)]
#[path = "tests.rs"]
mod tests;
