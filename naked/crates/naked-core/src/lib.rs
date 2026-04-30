pub mod agent_registry;
pub mod agent_role;
pub mod agent_run;
pub mod agent_store;
pub mod agent_validator;
pub mod config;
pub mod error;
#[path = "history_mod/mod.rs"]
pub mod history;
pub mod keys;
pub mod loop_;
pub mod mcp;
pub mod memory;
pub mod model_catalog;
pub mod prompt;
pub mod provider;
mod provider_ops;
pub mod research;
mod research_ops;
pub mod scrape;
pub mod search;
pub mod session;
mod session_ops;
pub mod skill;
pub mod tool;
pub mod types;

#[cfg(test)]
pub mod test_support;

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Weak};

use tokio::sync::{RwLock, mpsc};
use tokio_util::sync::CancellationToken;

use agent_registry::AgentRegistry;
use config::{Config, EffectiveSessionConfig, ResolvedProvider, SessionConfig};
use error::{AgentError, Result};
use keys::pool::KeyPool;
use loop_::{AgentLoop, LoopConfig};
use mcp::client::{McpRegistry, McpServer};
use mcp::wrapper::McpToolWrapper;
use provider::Provider;
use provider::anthropic::AnthropicProvider;
use provider::copilot::CopilotProvider;
use provider::openai_compat::OpenAiCompatProvider;
use provider::resilient::ResilientProvider;
use research::{
    CoordinatorConfig, FsResearchStore, ResearchContext, ResearchCoordinator, ResearchCreateTool,
    ResearchFindingsTool, ResearchHelpTool, ResearchLaunchTool, ResearchListSpecsTool,
    ResearchListTool, ResearchMetricsTool, ResearchPauseTool, ResearchResumeTool,
    ResearchSaveCursorTool, ResearchSaveTool, ResearchSetScheduleTool, ResearchSetTargetTool,
    ResearchSpec, ResearchStatusTool, ResearchStore, ResearchUpdateSpecTool, RunRecord, RunReport,
    VerifiedRunReport, new_research_id, parse_provider_model_pair,
};
use session::jsonl_store::JsonlSessionStore;
use session::store::SessionStore;
use session::{Session, SessionMetadata, SessionState, SessionSummary};
use skill::resolver::SkillResolver;
use skill::tool::SkillTool;
use tool::agent_control::{AgentStatusTool, AgentStopTool};
use tool::bash::BashTool;
use tool::file_ops::{EditFileTool, ReadFileTool, WriteFileTool};
use tool::memory::MemoryTool;
use tool::registry::ToolRegistry;
use tool::search::{GlobSearchTool, GrepSearchTool};
use tool::sub_agent::SubAgentTool;
use tool::web_fetch::WebFetchTool;
use tool::web_fetch_tls::WebFetchTlsTool;
use tool::web_fetch_wayback::WebFetchWaybackTool;
use tool::web_search::WebSearchTool;
use types::{AgentEvent, AgentHandle, ContentBlock, PermissionResponse};

/// What to push to history at the start of a turn. Internal — public callers
/// pick the variant via `send_prompt` (text-only) or `send_prompt_multimodal`
/// (text + images).
enum UserPush {
    Text(String),
    Multimodal {
        blocks: Vec<ContentBlock>,
        classifier_text: String,
    },
}

/// Create a single Provider instance for one key.
fn create_single_provider(name: &str, cfg: config::ProviderConfig) -> Box<dyn Provider> {
    match cfg.provider_type.as_str() {
        "anthropic" => Box::new(AnthropicProvider::new(name.to_string(), cfg)),
        "copilot" => Box::new(CopilotProvider::new(name.to_string(), cfg)),
        _ => Box::new(OpenAiCompatProvider::new(name.to_string(), cfg)),
    }
}

/// Create a Provider from a resolved config.
/// If multiple keys are available, wraps them in ResilientProvider for
/// automatic key rotation on failure.
pub fn create_provider(name: &str, resolved: ResolvedProvider) -> Box<dyn Provider> {
    if resolved.all_keys.len() <= 1 {
        let cfg = config::ProviderConfig {
            provider_type: resolved.provider_type,
            api_key: resolved.api_key,
            api_keys: Vec::new(),
            base_url: resolved.base_url,
            models: resolved.models,
            max_tokens: resolved.max_tokens,
            temperature: resolved.temperature,
            context_window: None,
            headers: resolved.headers,
            supports_vision: None,
            model_aliases: resolved.model_aliases,
            capabilities: HashMap::new(),
        };
        return create_single_provider(name, cfg);
    }

    let providers: Vec<Box<dyn Provider>> = resolved
        .all_keys
        .iter()
        .enumerate()
        .map(|(i, key)| {
            let tag = format!("{name}[key-{i}]");
            let cfg = config::ProviderConfig {
                provider_type: resolved.provider_type.clone(),
                api_key: key.clone(),
                api_keys: Vec::new(),
                base_url: resolved.base_url.clone(),
                models: resolved.models.clone(),
                max_tokens: resolved.max_tokens,
                temperature: resolved.temperature,
                context_window: None,
                headers: resolved.headers.clone(),
                supports_vision: None,
                model_aliases: resolved.model_aliases.clone(),
                capabilities: HashMap::new(),
            };
            create_single_provider(&tag, cfg)
        })
        .collect();

    tracing::info!(
        "Provider '{name}': {} keys configured for rotation",
        providers.len()
    );
    Box::new(ResilientProvider::new(providers))
}

/// Acquire one permit from the process-wide research-run semaphore.
///
/// The label is logged at debug level when the call has to wait, so operators
/// can see in `RUST_LOG=naked_core=debug` which spec is queueing behind which
/// other run. Returns an [`AgentError::Provider`] only if the semaphore was
/// closed (which we never do in production code).
///
/// Used by [`AgentCore::run_research`], [`AgentCore::run_research_verified`],
/// the in-process `naked-tg` scheduler, and any future entry point. Bypassing
/// this helper means bypassing the global concurrency cap — don't.
pub async fn acquire_research_permit(
    sem: &Arc<tokio::sync::Semaphore>,
    label: &str,
) -> Result<tokio::sync::OwnedSemaphorePermit> {
    if sem.available_permits() == 0 {
        tracing::debug!(spec = %label, "research run waiting for permit");
    }
    sem.clone()
        .acquire_owned()
        .await
        .map_err(|e| AgentError::Provider(format!("research semaphore closed: {e}")))
}

/// Build a provider (possibly resilient with fallbacks) from config.
pub fn build_provider_from_config(config: &Config) -> Result<Box<dyn Provider>> {
    let (primary_name, primary_resolved) = config.resolve_default_provider()?;
    let mut providers: Vec<Box<dyn Provider>> =
        vec![create_provider(&primary_name, primary_resolved)];

    for (fb_provider, _fb_model) in config.fallback_providers() {
        if fb_provider == primary_name {
            continue;
        }
        if let Some(pc) = config.provider_config(&fb_provider)
            && let Ok(resolved) = pc.resolved()
        {
            providers.push(create_provider(&fb_provider, resolved));
        }
    }

    if providers.len() == 1 {
        Ok(providers.remove(0))
    } else {
        Ok(Box::new(ResilientProvider::new(providers)))
    }
}

/// Summary info about a configured provider.
#[derive(Debug, Clone)]
pub struct ProviderInfo {
    pub name: String,
    pub models: Vec<String>,
    pub active: bool,
}

/// Top-level facade: manages sessions, tools, and the agent loop.
type ExtraToolFactories = Vec<Arc<dyn Fn() -> Box<dyn tool::Tool> + Send + Sync>>;

pub struct AgentCore {
    config: Config,
    provider: Arc<dyn Provider>,
    store: Arc<dyn SessionStore>,
    mcp_registry: Arc<RwLock<McpRegistry>>,
    /// Per-provider-name cache so we don't rebuild the same provider on every turn.
    provider_cache: RwLock<HashMap<String, Arc<dyn Provider>>>,
    /// Per-session MCP servers (connected lazily from session config.json).
    session_mcp: RwLock<HashMap<String, Vec<Arc<McpServer>>>>,
    sessions: Arc<RwLock<HashMap<String, Session>>>,
    cancels: RwLock<HashMap<String, CancellationToken>>,
    agent_registry: AgentRegistry,
    /// Optional per-session "current author" (Telegram user id). Set by the
    /// channel before a turn runs and consumed by the memory tool to implement
    /// `scope=user` without guessing.
    session_senders: RwLock<HashMap<String, String>>,
    /// Shared research storage. Always present — `perfection-plan-v5` treats
    /// disabled research as "no `/research *` commands accepted" rather than
    /// a separate data path, so the store is cheap to keep live.
    research_store: Arc<dyn ResearchStore>,
    /// Ambient "which research is this turn driving?" handle, consumed by the
    /// research tools. The research coordinator sets it before each run and
    /// clears it after; outside of a research run it stays `None`.
    research_context: ResearchContext,
    /// Weak self-reference so orchestration tools (e.g. `research_launch`) can
    /// upgrade to `Arc<Self>` and call methods like `run_research`. Set once via
    /// `init_self_ref()` after Arc construction.
    self_ref: std::sync::RwLock<Option<Weak<AgentCore>>>,
    /// In-process scheduler hook. Defaults to a no-op so the CLI / tests don't
    /// need an explicit setup; the TG bot installs a real implementation via
    /// [`AgentCore::set_scheduler_hook`] right after `init_self_ref()`.
    scheduler_hook: std::sync::RwLock<Arc<dyn research::SchedulerHook>>,
    /// Process-wide concurrency cap for research runs. Every entry point
    /// (`run_research`, `run_research_verified`, the LLM `research_launch`
    /// tool, the in-process scheduler) must acquire a permit before kicking
    /// off a coordinator. Defaults to 1 permit so we never reach the
    /// Playwright "Browser is already in use" race.
    research_run_semaphore: Arc<tokio::sync::Semaphore>,
    /// Catalog of agent roles loaded from `Config::agent_dirs` at
    /// startup. Cheap to clone (`Arc` inside) so callers can hand it
    /// to coordinators / CLI subcommands without lock contention.
    /// Empty when no `agent_dirs` exist on disk — that's a valid
    /// config for hosts that build roles programmatically.
    agent_store: Arc<agent_store::AgentStore>,
    /// Runtime health tracker (Phase 3). Shared across every agent loop
    /// spawned by this core so the 24h rolling window is
    /// process-global, and its durable jsonl survives restarts.
    /// Constructed in [`AgentCore::new`] from `config.model_health`.
    model_health: Arc<crate::model_catalog::ModelHealth>,
    /// Per-run event registry (waterfall) for the research subsystem.
    /// Shared with every coordinator spawned from this core so the
    /// TG heartbeat task can poll by `run_id` without reaching into
    /// coordinator internals.
    research_run_events: research::RunEventRegistry,
    /// Live cancellation tokens keyed by research `run_id`. Populated
    /// when [`Self::run_research*`] spawns a coordinator and cleared
    /// on completion. The TG `r:stop:<run_id>` callback looks up the
    /// token here and signals it — callers who kicked the run off
    /// via `tokio::spawn` still own the `JoinHandle`, but the
    /// cooperative cancel path works without them.
    research_cancels: Arc<RwLock<HashMap<String, CancellationToken>>>,
    /// Per-provider search/scrape API key pools. Built once at startup
    /// from `~/.naked/secrets/search_pool.alive.json` (primary) plus
    /// CSV env-vars (fallback) and cached for the lifetime of the
    /// process. `WebSearchTool::new` borrows them on every turn, so
    /// rebuilding the tool registry stays cheap.
    exa_key_pool: Arc<KeyPool>,
    tavily_key_pool: Arc<KeyPool>,
    serpapi_key_pool: Arc<KeyPool>,
    /// Cloud-scrape cascade injected into [`WebFetchTool`] as Tier 3.5.
    /// Built once at startup from the same key pools as search; `None`
    /// when no ScrapingBee or Firecrawl keys are available so the
    /// existing 4-tier cascade remains untouched on hosts without them.
    cloud_scraper: Option<Arc<crate::scrape::multi::MultiCloudScraper>>,
    /// Adaptive per-host fetch-tier selector. Lives on `AgentCore` so a
    /// single shared instance is reused across every `WebFetchTool`
    /// built per session — per-host learning persists for the entire
    /// lifetime of the agent process (not just one tool invocation).
    host_policy: Arc<crate::scrape::host_policy::HostPolicy>,
    /// Extra tools injected by the embedding binary (e.g. naked-tg).
    /// Appended to every session's tool registry after the built-in
    /// tools. Factory closures produce fresh instances per session.
    extra_tool_factories: RwLock<ExtraToolFactories>,
}

impl AgentCore {
    pub fn new(config: Config, provider: Box<dyn Provider>) -> Self {
        let store = Arc::new(JsonlSessionStore::new(config.session_dir_abs()));
        let research_root = config
            .research
            .storage_dir
            .clone()
            .unwrap_or_else(research::store::research_root);
        let research_store: Arc<dyn ResearchStore> = Arc::new(FsResearchStore::new(research_root));
        let max_concurrent = config.research.max_concurrent_runs.max(1);
        let research_run_semaphore = Arc::new(tokio::sync::Semaphore::new(max_concurrent));

        // Load agent roles from disk. A failure here is a hard config
        // error — we'd rather refuse to start than silently run with
        // an empty catalog and confuse every later CLI subcommand
        // with "unknown role". On a fresh checkout with no
        // `agent_dirs` present on disk, `load_dirs` returns an empty
        // store (missing roots are not errors), which is fine.
        let agent_store = match agent_store::AgentStore::load_dirs(&config.agent_dirs) {
            Ok(s) => Arc::new(s),
            Err(e) => {
                tracing::error!(error = %e, "agent_store: load failed, continuing with empty catalog");
                Arc::new(agent_store::AgentStore::empty())
            }
        };

        let model_health = Arc::new(crate::model_catalog::ModelHealth::load(
            config.model_health.clone(),
        ));
        if let Some(observer) = crate::model_catalog::ObservationRecorder::default_location() {
            model_health.attach_observer(Arc::new(observer));
        }

        let research_run_events = research::RunEventRegistry::new();
        let research_context = ResearchContext::new();
        research_context.set_run_events(Some(research_run_events.clone()));

        let (exa_key_pool, tavily_key_pool, serpapi_key_pool) =
            build_search_key_pools(&config.exa_api_keys);
        let cloud_scraper = build_cloud_scraper();
        let host_policy = Arc::new(crate::scrape::host_policy::HostPolicy::new());

        Self {
            config,
            provider: Arc::from(provider),
            store,
            mcp_registry: Arc::new(RwLock::new(McpRegistry::new())),
            provider_cache: RwLock::new(HashMap::new()),
            session_mcp: RwLock::new(HashMap::new()),
            sessions: Arc::new(RwLock::new(HashMap::new())),
            cancels: RwLock::new(HashMap::new()),
            agent_registry: AgentRegistry::new(),
            session_senders: RwLock::new(HashMap::new()),
            research_store,
            research_context,
            self_ref: std::sync::RwLock::new(None),
            scheduler_hook: std::sync::RwLock::new(research::noop_hook()),
            research_run_semaphore,
            agent_store,
            model_health,
            research_run_events,
            research_cancels: Arc::new(RwLock::new(HashMap::new())),
            exa_key_pool,
            tavily_key_pool,
            serpapi_key_pool,
            cloud_scraper,
            host_policy,
            extra_tool_factories: RwLock::new(Vec::new()),
        }
    }

    /// Shared runtime health tracker. Exposed so telemetry surfaces
    /// (Phase 3 Prometheus exporter, `/model health` CLI) can query
    /// rolling counters without round-tripping through the coordinator.
    pub fn model_health(&self) -> Arc<crate::model_catalog::ModelHealth> {
        self.model_health.clone()
    }

    /// Shared catalog of disk-loaded agent roles. Use
    /// `naked_core::agent_store::resolve_role(name, &core.agent_store(),
    /// &config.agent_roles)` to pick a role with overrides applied.
    pub fn agent_store(&self) -> Arc<agent_store::AgentStore> {
        self.agent_store.clone()
    }

    /// Expose the process-wide research-run semaphore. Schedulers and other
    /// internal callers may need to inspect it (e.g. `available_permits()` for
    /// dispatch planning) without going through `run_research*`.
    pub fn research_run_permits(&self) -> Arc<tokio::sync::Semaphore> {
        self.research_run_semaphore.clone()
    }

    /// Install an in-process scheduler hook. The TG bot calls this once at
    /// startup so spec mutations (`research_create`, `research_update_spec`,
    /// pause/resume) trigger immediate rescheduling.
    pub fn set_scheduler_hook(&self, hook: Arc<dyn research::SchedulerHook>) {
        *self.scheduler_hook.write().unwrap() = hook;
    }

    /// Register an extra tool factory. Each factory is called once per
    /// session turn to produce a fresh tool instance. Use this to inject
    /// tools from the embedding binary (e.g. `telegram_attach` from
    /// `naked-tg`) without coupling naked-core to Telegram.
    pub async fn register_extra_tool<F>(&self, factory: F)
    where
        F: Fn() -> Box<dyn tool::Tool> + Send + Sync + 'static,
    {
        self.extra_tool_factories
            .write()
            .await
            .push(Arc::new(factory));
    }

    fn scheduler_hook(&self) -> Arc<dyn research::SchedulerHook> {
        self.scheduler_hook.read().unwrap().clone()
    }

    /// Snapshot the scheduler's in-memory failure tracker for `spec_id`.
    /// Returns `None` when no scheduler is wired (CLI / tests) or when
    /// the spec has never failed under the current process. Surfaces
    /// the data needed by `/research state <id>` without leaking any
    /// scheduler internals to the bot crate.
    pub async fn scheduler_failure_snapshot(&self, spec_id: &str) -> Option<(u32, bool)> {
        self.scheduler_hook().failure_snapshot(spec_id).await
    }

    /// Manually rearm a research spec after operator intervention.
    ///
    /// Wipes the in-memory failure streak / alert flag, clears
    /// `pause_reason`, and resumes the spec on disk in one atomic
    /// move. Idempotent — calling it on a healthy spec is a no-op.
    /// Powers the `/research reset <id>` command so an operator can
    /// undo an auto-pause without grepping for the right knobs to
    /// twist.
    pub async fn reset_research_failures(&self, id: &str) -> Result<()> {
        self.scheduler_hook().reset_failures(id).await;
        let mut spec = self.research_store.load_spec(id).await?;
        let needs_save = spec.paused || spec.pause_reason.is_some();
        spec.paused = false;
        spec.pause_reason = None;
        if needs_save {
            self.research_store.save_spec(&spec).await?;
        }
        self.scheduler_hook()
            .notify(research::SchedulerEvent::SpecUpdated {
                spec_id: id.to_string(),
            })
            .await;
        Ok(())
    }

    /// Must be called once after wrapping in `Arc` so orchestration tools can
    /// obtain a reference back to the core (e.g. `research_launch`).
    pub fn init_self_ref(self: &Arc<Self>) {
        *self.self_ref.write().unwrap() = Some(Arc::downgrade(self));
    }

    pub fn config(&self) -> &Config {
        &self.config
    }

    /// Expose the research store so CLI / TG handlers can read & write without
    /// going through the full research coordinator. The store is always live;
    /// callers still need to check `config.research.enabled` before offering
    /// `/research *` surfaces to the user.
    pub fn research_store(&self) -> Arc<dyn ResearchStore> {
        self.research_store.clone()
    }

    /// Expose the underlying provider so out-of-loop consumers (e.g. the
    /// [`crate::agent_validator::GatekeeperValidator`]) can issue one-shot
    /// completions without spinning up a session. Returns the same `Arc`
    /// the agent uses for its own loop, so token budgets and rate limits
    /// stay shared.
    pub fn provider(&self) -> Arc<dyn Provider> {
        self.provider.clone()
    }
}

/// RAII guard that owns the lifecycle of a research run's
/// `(cancel_token, events)` registration. Inserts the entry on
/// `install`, removes it on drop — so every exit path of
/// `run_research_*` (Ok, Err, panic) cleans up without boilerplate.
struct ResearchCancelGuard {
    cancels: Arc<RwLock<HashMap<String, CancellationToken>>>,
    events: research::RunEventRegistry,
    spec_id: String,
}

impl ResearchCancelGuard {
    async fn install(
        cancels: Arc<RwLock<HashMap<String, CancellationToken>>>,
        events: research::RunEventRegistry,
        spec_id: &str,
        token: CancellationToken,
    ) -> Self {
        cancels.write().await.insert(spec_id.to_string(), token);
        Self {
            cancels,
            events,
            spec_id: spec_id.to_string(),
        }
    }
}

impl Drop for ResearchCancelGuard {
    fn drop(&mut self) {
        // Take ownership of the necessary fields so the spawned future
        // doesn't outlive the guard. All state is cheaply clone-able.
        let cancels = self.cancels.clone();
        let events = self.events.clone();
        let spec_id = std::mem::take(&mut self.spec_id);
        tokio::spawn(async move {
            cancels.write().await.remove(&spec_id);
            events.drop_run(&spec_id).await;
        });
    }
}

/// Adapter so the research coordinator can drive `AgentCore` without AgentCore
/// having a direct dep on the coordinator's `AgentRunner` trait bounds (keeps
/// the coordinator unit-testable with stubs).
///
/// `run_session_map` keeps a `run_id → session_id` table so
/// [`Self::cleanup_research_session`] can abort the underlying
/// session's [`AgentLoop`] task. Without this, cancelling a research
/// run would only stop the coordinator's `drain_events` loop while the
/// background `tokio::spawn` keeps running tools (the cancel-safety
/// bug observed in production: worker continued executing for minutes
/// after a `Stop` button press).
struct AgentCoreResearchRunner {
    core: Arc<AgentCore>,
    run_session_map: Arc<RwLock<HashMap<String, String>>>,
}

impl AgentCoreResearchRunner {
    fn new(core: Arc<AgentCore>) -> Self {
        Self {
            core,
            run_session_map: Arc::new(RwLock::new(HashMap::new())),
        }
    }
}

#[async_trait::async_trait]
impl research::AgentRunner for AgentCoreResearchRunner {
    async fn start_research_turn(
        &self,
        spec: &ResearchSpec,
        prompt: &str,
        config: &CoordinatorConfig,
        run_id: &str,
    ) -> Result<(types::AgentHandle, String, String)> {
        // 1. Provision an ephemeral session on the "research" channel. This
        //    lives on disk under `session_dir/<id>` — intentional, because it
        //    makes `/sessions` show a breadcrumb of every research run for
        //    later inspection.
        let workspace = config.workspace.clone();
        let session_id = self
            .core
            .create_session_with_channel(&workspace, "research")
            .await;

        // 2. Apply per-session provider/model override so this turn runs on
        //    the research model (default: kimi-for-coding via kimi-code).
        //    The precedence is spec > config.research > global default.
        //
        //    `config.default_model` may carry a `provider/model` pair (the
        //    fallback chain uses this to switch to qwen when kimi fails — see
        //    `try_start_with_fallback`). When it does, the embedded provider
        //    overrides everything else for this turn.
        let (parsed_provider, parsed_model) = config
            .default_model
            .as_deref()
            .and_then(parse_provider_model_pair)
            .map(|(p, m)| (Some(p), Some(m)))
            .unwrap_or_else(|| {
                (
                    config.default_provider.clone(),
                    config.default_model.clone(),
                )
            });
        let provider = spec.provider.clone().or(parsed_provider);
        let model = spec.model.clone().or(parsed_model);
        if (provider.is_some() || model.is_some())
            && let Err(e) = self
                .core
                .set_session_provider(&session_id, provider.as_deref(), model.as_deref())
                .await
        {
            tracing::warn!("research override failed: {e}");
        }

        // 2b. Apply reasoning level (e.g. "medium" for kimi-for-coding) so
        //     it surfaces as `reasoning_effort` in the OAI-compat request.
        if let Some(level) = config.reasoning.as_deref()
            && let Err(e) = self.core.set_session_reasoning(&session_id, level).await
        {
            tracing::warn!("research reasoning override failed: {e}");
        }

        // 3. Auto-approve tool permissions for the research session.
        //    Research runs are headless — nobody is watching to click "allow".
        let now_ts = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_secs() as i64)
            .unwrap_or(0);
        if let Err(e) = self.core.set_session_yolo(&session_id, Some(now_ts)).await {
            tracing::warn!("failed to enable yolo for research session: {e}");
        }

        // 4. Install the ambient research context so the `research_save`
        //    tools know where to write and which run_id to tag findings with.
        //    Reset the per-run save counter so the new clear-target guard
        //    (`research_set_target`) starts at 0 for this run rather than
        //    inheriting a count from whatever happened previously.
        self.core.research_context.set_id(Some(spec.id.clone()));
        self.core
            .research_context
            .set_run_id(Some(run_id.to_string()));
        self.core.research_context.reset_saves();

        let handle = self.core.send_prompt(&session_id, prompt).await?;

        // Remember which session backs this run so cleanup (incl. the cancel
        // path) can abort the spawned AgentLoop task. Without this, a
        // cancelled research run keeps burning model tokens / proxy
        // bandwidth in the background.
        self.run_session_map
            .write()
            .await
            .insert(run_id.to_string(), session_id.clone());

        // Resolve what we actually settled on after overrides were applied, so
        // the run record is truthful.
        let effective_provider = provider
            .clone()
            .unwrap_or_else(|| self.core.config.default_provider.clone());
        let effective_model = model
            .clone()
            .unwrap_or_else(|| self.core.config.default_model.clone());

        Ok((handle, effective_provider, effective_model))
    }

    async fn cleanup_research_session(&self, run_id: &str) {
        // Abort the AgentLoop task that backs this run. Idempotent — if the
        // task already finished naturally, the cancel call is a no-op and
        // the session state is already `Idle`. The mapping is removed
        // unconditionally so we don't accumulate stale entries.
        let session_id = self.run_session_map.write().await.remove(run_id);
        if let Some(sid) = session_id {
            tracing::info!(
                run_id, session = %sid,
                "research cleanup: aborting underlying agent session",
            );
            self.core.abort(&sid).await;
        }

        self.core.research_context.set_id(None);
        self.core.research_context.set_run_id(None);
        self.core.research_context.reset_saves();
    }
}

fn provider_to_box(provider: &Arc<dyn Provider>) -> Box<dyn Provider> {
    Box::new(ArcProvider(provider.clone()))
}

/// Partial mutation applied to a [`ResearchSpec`] by [`AgentCore::update_research`].
///
/// Every field is optional. The double-`Option` on `interval_seconds`
/// distinguishes "do not touch" (`None`) from "clear the schedule"
/// (`Some(None)`); same for `provider`, `model`, `max_iterations`, and
/// `max_wall_seconds` (where an empty string / explicit `null` clears).
#[derive(Debug, Default, Clone)]
pub struct ResearchPatch {
    pub topic: Option<String>,
    /// Replace the entire sources list (after dedup, empty entries dropped).
    pub sources_replace: Option<Vec<String>>,
    /// Append to the sources list (skipping duplicates).
    pub sources_add: Option<Vec<String>>,
    pub interval_seconds: Option<Option<u64>>,
    /// One-shot at-time trigger. `Some(Some(t))` = set to `t`,
    /// `Some(None)` = clear, `None` = leave unchanged.
    pub run_at: Option<Option<chrono::DateTime<chrono::Utc>>>,
    /// Recurring cron expression. Same triple-state semantics.
    pub cron: Option<Option<String>>,
    /// Per-spec scheduler-task timeout override (seconds).
    pub task_timeout_seconds: Option<Option<u64>>,
    pub provider: Option<String>,
    pub model: Option<String>,
    pub max_iterations: Option<Option<u32>>,
    pub max_wall_seconds: Option<Option<u64>>,
}

/// Apply a [`ResearchPatch`] to an in-memory [`ResearchSpec`] following the
/// same semantics used by [`AgentCore::update_research`]. Exposed so tests can
/// assert patch behavior without spinning up an [`AgentCore`].
pub fn apply_research_patch(spec: &mut ResearchSpec, patch: ResearchPatch) {
    if let Some(topic) = patch.topic {
        let trimmed = topic.trim();
        if !trimmed.is_empty() {
            spec.topic = trimmed.to_string();
        }
    }
    if let Some(sources) = patch.sources_replace {
        let mut seen = std::collections::HashSet::new();
        spec.sources = sources
            .into_iter()
            .filter(|s| !s.trim().is_empty())
            .filter(|s| seen.insert(s.clone()))
            .collect();
    }
    if let Some(extra) = patch.sources_add {
        for s in extra {
            let s = s.trim().to_string();
            if !s.is_empty() && !spec.sources.contains(&s) {
                spec.sources.push(s);
            }
        }
    }
    if let Some(interval) = patch.interval_seconds {
        spec.interval_seconds = interval;
    }
    if let Some(at) = patch.run_at {
        spec.run_at = at;
    }
    if let Some(cron) = patch.cron {
        spec.cron = cron;
    }
    if let Some(timeout) = patch.task_timeout_seconds {
        spec.task_timeout_seconds = timeout;
    }
    if let Some(provider) = patch.provider {
        spec.provider = if provider.trim().is_empty() {
            None
        } else {
            Some(provider)
        };
    }
    if let Some(model) = patch.model {
        spec.model = if model.trim().is_empty() {
            None
        } else {
            Some(model)
        };
    }
    if let Some(iters) = patch.max_iterations {
        spec.max_iterations = iters;
    }
    if let Some(secs) = patch.max_wall_seconds {
        spec.max_wall_seconds = secs;
    }
}

/// Implementation behind [`AgentCore::write_research_memory_link`].
/// Extracted into a free function so it can be exercised by tests without
/// having to spin up a full `AgentCore`.
///
/// Run-completion lines are appended to the **research run-log**
/// (`research::store::research_runlog_path()` by default), *not* to any
/// `MEMORY.md`. The durable memory files are reserved for promoted rules
/// from the daily-digest pipeline; flooding them with one entry per
/// research run pollutes the system prompt and trips the per-file cap.
///
/// If `memory_path_override` is `Some`, the line is appended to that
/// exact file (used by tests). The parameter name is kept for API
/// stability — callers in `naked-core` already pass `None`.
pub async fn write_research_memory_link_for(
    store: &dyn ResearchStore,
    _workspace: &Path,
    spec_id: &str,
    run_id: &str,
    verified: Option<&VerifiedRunReport>,
    memory_path_override: Option<&Path>,
) -> Result<()> {
    let spec = store.load_spec(spec_id).await?;
    let runs = store.list_runs(spec_id, Some(50)).await.unwrap_or_default();
    let record: Option<RunRecord> = runs.into_iter().find(|r| r.run_id == run_id);
    let total_after = record.as_ref().map(|r| r.total_findings_after).unwrap_or(0);
    let new_findings = record.as_ref().map(|r| r.new_findings).unwrap_or(0);
    let elapsed_secs = record.as_ref().and_then(|r| r.elapsed_secs).unwrap_or(0);

    let report_path = store
        .report_path(spec_id)
        .map(|p| p.display().to_string())
        .unwrap_or_else(|| "<unavailable>".to_string());

    let topic = spec.topic.replace('"', "'");
    let mut body = format!(
        "research:{spec_id} | topic=\"{topic}\" | run={run_id} | new={new_findings} total={total_after}",
    );
    if let Some(vr) = verified {
        body.push_str(&format!(
            " | verified={r} rounds, removed={d}, replaced={p}, remaining={rem}",
            r = vr.verification_rounds,
            d = vr.dead_removed,
            p = vr.replacements_found,
            rem = vr.remaining_issues.len(),
        ));
    }
    body.push_str(&format!(
        " | elapsed={elapsed_secs}s | report={report_path}"
    ));

    // UTF-8 safe truncation: `MAX_ENTRY_CHARS` is named in chars, not bytes,
    // and a plain byte slice (`&body[..max - 3]`) panics inside multi-byte
    // codepoints — e.g. the Cyrillic 'й' (2 bytes) in a Russian research
    // topic split exactly at byte 497. Walk char boundaries instead and
    // never split inside a codepoint.
    let max = memory::store::MAX_ENTRY_CHARS;
    let body = if body.chars().count() > max {
        let mut truncated: String = body.chars().take(max.saturating_sub(3)).collect();
        truncated.push_str("...");
        truncated
    } else {
        body
    };

    let path: PathBuf = match memory_path_override {
        Some(p) => p.to_path_buf(),
        None => research::store::research_runlog_path(),
    };
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)
            .map_err(|e| AgentError::Provider(format!("research runlog dir: {e}")))?;
    }
    let stamp = chrono::Utc::now().format("%Y-%m-%dT%H:%M:%SZ");
    let line = format!("- {stamp} {body}\n");
    use std::io::Write;
    let mut f = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(&path)
        .map_err(|e| AgentError::Provider(format!("research runlog open: {e}")))?;
    f.write_all(line.as_bytes())
        .map_err(|e| AgentError::Provider(format!("research runlog write: {e}")))?;
    Ok(())
}

/// Build the per-provider [`KeyPool`]s used by [`WebSearchTool`].
///
/// Sources, in order:
/// 1. [`keys::fs::FilesystemKeyProvider`] reading the standalone JSON pool
///    at `~/.naked/secrets/search_pool.alive.json` — the **primary** source
///    populated by `naked/scripts/import_search_keys.py`.
/// 2. [`keys::env::EnvKeyProvider`] (CSV env-vars like `EXA_API_KEYS`) —
///    legacy fallback for hosts that don't ship the pool file.
/// 3. For Exa specifically, the legacy `Config::exa_api_keys` slice is
///    folded in too so existing `.env` setups keep working until everyone
///    migrates.
///
/// TTL: 6 hours, matching the recommended refresh cadence in
/// `naked/scripts/README-search-pool.md`.
fn build_search_key_pools(
    legacy_exa_keys: &[String],
) -> (Arc<KeyPool>, Arc<KeyPool>, Arc<KeyPool>) {
    use keys::KeyProvider;
    use keys::env::EnvKeyProvider;
    use keys::fs::FilesystemKeyProvider;
    use std::time::Duration;

    let fs_path = FilesystemKeyProvider::default_path();
    let fs_provider: Arc<dyn KeyProvider> = Arc::new(FilesystemKeyProvider::new(fs_path.clone()));
    let env_provider: Arc<dyn KeyProvider> = Arc::new(EnvKeyProvider::defaults());

    // Bridge legacy `Config::exa_api_keys` (set from `.env` at startup) into
    // the Exa pool so a host without the JSON file still gets its keys.
    struct StaticProvider {
        keys: Vec<String>,
    }
    impl KeyProvider for StaticProvider {
        fn fetch(&self, _: &str) -> std::result::Result<Vec<String>, String> {
            Ok(self.keys.clone())
        }
    }
    let legacy_exa: Arc<dyn KeyProvider> = Arc::new(StaticProvider {
        keys: legacy_exa_keys.to_vec(),
    });

    let ttl = Duration::from_secs(6 * 3600);

    let exa = Arc::new(KeyPool::new(
        vec![fs_provider.clone(), env_provider.clone(), legacy_exa],
        "exa",
        ttl,
    ));
    let tavily = Arc::new(KeyPool::new(
        vec![fs_provider.clone(), env_provider.clone()],
        "tavily",
        ttl,
    ));
    let serpapi = Arc::new(KeyPool::new(
        vec![fs_provider.clone(), env_provider.clone()],
        "serpapi",
        ttl,
    ));

    tracing::info!(
        pool_path = %fs_path.display(),
        exa = exa.size(),
        tavily = tavily.size(),
        serpapi = serpapi.size(),
        "search key pools initialized"
    );
    (exa, tavily, serpapi)
}

/// Build the optional cloud-scrape cascade for [`WebFetchTool`] Tier 3.5.
///
/// Reads ScrapingBee + Firecrawl keys from the same `~/.naked/secrets/
/// search_pool.alive.json` source as the search engines, with the
/// `SCRAPINGBEE_API_KEYS` / `FIRECRAWL_API_KEYS` CSV env-vars as
/// fallback. Returns `None` when no keys are available so the cascade
/// silently skips the tier — the existing 4-tier path still works on
/// hosts that haven't provisioned cloud-scrape credentials.
fn build_cloud_scraper() -> Option<Arc<crate::scrape::multi::MultiCloudScraper>> {
    use crate::scrape::CloudScraper;
    use crate::scrape::firecrawl::FirecrawlEngine;
    use crate::scrape::multi::MultiCloudScraper;
    use crate::scrape::scrapingbee::ScrapingBeeEngine;
    use keys::KeyProvider;
    use keys::env::EnvKeyProvider;
    use keys::fs::FilesystemKeyProvider;
    use std::time::Duration;

    let fs_provider: Arc<dyn KeyProvider> = Arc::new(FilesystemKeyProvider::new(
        FilesystemKeyProvider::default_path(),
    ));
    let env_provider: Arc<dyn KeyProvider> = Arc::new(EnvKeyProvider::defaults());
    let ttl = Duration::from_secs(6 * 3600);

    let scrapingbee_pool = Arc::new(KeyPool::new(
        vec![fs_provider.clone(), env_provider.clone()],
        "scrapingbee",
        ttl,
    ));
    let firecrawl_pool = Arc::new(KeyPool::new(
        vec![fs_provider, env_provider],
        "firecrawl",
        ttl,
    ));

    let mut engines: Vec<Arc<dyn CloudScraper>> = Vec::new();
    if scrapingbee_pool.size() > 0 {
        engines.push(Arc::new(ScrapingBeeEngine::new(scrapingbee_pool.clone())));
    }
    if firecrawl_pool.size() > 0 {
        engines.push(Arc::new(FirecrawlEngine::new(firecrawl_pool.clone())));
    }

    if engines.is_empty() {
        tracing::info!("cloud-scrape: no keys available, Tier 3.5 disabled");
        return None;
    }

    let summary = engines
        .iter()
        .map(|e| e.name().to_string())
        .collect::<Vec<_>>()
        .join(",");
    tracing::info!(
        engines = %summary,
        scrapingbee_keys = scrapingbee_pool.size(),
        firecrawl_keys = firecrawl_pool.size(),
        "cloud-scrape: cascade initialized",
    );
    Some(Arc::new(MultiCloudScraper::new(engines)))
}

struct ArcProvider(Arc<dyn Provider>);

#[async_trait::async_trait]
impl Provider for ArcProvider {
    fn name(&self) -> &str {
        self.0.name()
    }
    fn models(&self) -> Vec<types::ModelInfo> {
        self.0.models()
    }
    async fn stream_chat(
        &self,
        request: provider::ChatRequest,
    ) -> Result<std::pin::Pin<Box<dyn tokio_stream::Stream<Item = types::StreamChunk> + Send>>>
    {
        self.0.stream_chat(request).await
    }
}
