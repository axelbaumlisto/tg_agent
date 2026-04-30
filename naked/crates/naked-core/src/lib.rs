pub mod agent_registry;
pub mod agent_role;
pub mod agent_run;
pub mod agent_store;
pub mod agent_validator;
pub mod config;
pub mod error;
pub mod history;
pub mod keys;
pub mod loop_;
pub mod mcp;
pub mod memory;
pub mod model_catalog;
pub mod prompt;
pub mod provider;
pub mod research;
pub mod scrape;
pub mod search;
pub mod session;
pub mod skill;
pub mod tool;
pub mod types;

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

    /// Record the current author for an upcoming turn. Used by the memory tool
    /// to resolve `scope=user` without an explicit `user_id`. `None` clears it.
    pub async fn set_session_sender(&self, session_id: &str, sender_id: Option<String>) {
        let mut map = self.session_senders.write().await;
        match sender_id {
            Some(id) if !id.is_empty() => {
                map.insert(session_id.to_string(), id);
            }
            _ => {
                map.remove(session_id);
            }
        }
    }

    /// Look up the currently-recorded author for this session, if any.
    pub async fn session_sender(&self, session_id: &str) -> Option<String> {
        self.session_senders.read().await.get(session_id).cloned()
    }

    /// Connect to configured MCP servers.
    pub async fn init_mcp(&self) {
        let servers = self.config.mcp_server_list();
        if !servers.is_empty() {
            {
                let old = self.mcp_registry.read().await;
                old.close_all().await;
            }
            let registry = McpRegistry::connect_all(&servers).await;
            tracing::info!(
                "MCP: {} tools from {} servers",
                registry.all_tools().len(),
                registry.servers().len()
            );
            *self.mcp_registry.write().await = registry;
        }
    }

    /// Borrow the underlying session store. Exposed for operator commands
    /// (`vacuum-sessions`, future `gc` task) that need to walk all sessions
    /// without going through the in-memory cache.
    pub fn store(&self) -> Arc<dyn SessionStore> {
        self.store.clone()
    }

    /// Load per-session config.json if it exists.
    pub fn load_session_config_pub(&self, session_id: &str) -> SessionConfig {
        let path = self.store.session_root(session_id).join("config.json");
        if path.exists() {
            match SessionConfig::from_file(&path) {
                Ok(sc) => {
                    tracing::info!("loaded per-session config for {}", &session_id[..8]);
                    sc
                }
                Err(e) => {
                    tracing::warn!("bad session config.json for {}: {e}", &session_id[..8]);
                    SessionConfig::default()
                }
            }
        } else {
            SessionConfig::default()
        }
    }

    /// Call the current model to summarize conversation for compaction.
    async fn llm_summarize(
        provider: &dyn Provider,
        model: &str,
        conversation_text: &str,
    ) -> Result<String> {
        use tokio_stream::StreamExt;

        const MAX_COMPACTION_INPUT: usize = 16_000;
        // UTF-8 safe truncation — conversation_text routinely contains
        // multi-byte text (Russian, Vietnamese, emoji), and a raw byte
        // slice panics inside a codepoint. Allocating a new String here is
        // cheap relative to the model call that follows.
        let truncated_owned;
        let input: &str = if conversation_text.chars().count() > MAX_COMPACTION_INPUT {
            truncated_owned = conversation_text
                .chars()
                .take(MAX_COMPACTION_INPUT)
                .collect::<String>();
            truncated_owned.as_str()
        } else {
            conversation_text
        };

        tracing::info!(
            input_chars = input.len(),
            "LLM compaction: sending to model"
        );

        let system = "Ты — помощник для сжатия контекста. Сделай краткое резюме разговора ниже. \
            Сохрани: ключевые решения, текущую задачу, важные файлы/пути, незавершённую работу. \
            Формат: компактный текст, без markdown заголовков, максимум 1500 символов.";

        let request = provider::ChatRequest {
            model: model.to_string(),
            system: system.to_string(),
            messages: vec![serde_json::json!({
                "role": "user",
                "content": input,
            })],
            tools: vec![],
            max_tokens: 1024,
            temperature: Some(0.0),
            reasoning: None,
        };

        let mut stream = provider
            .stream_chat(request)
            .await
            .map_err(|e| AgentError::Provider(format!("compaction LLM call failed: {e}")))?;

        let mut text = String::new();
        while let Some(chunk) = stream.next().await {
            match chunk {
                types::StreamChunk::Text(t) => text.push_str(&t),
                types::StreamChunk::Done => break,
                types::StreamChunk::Error(e) => {
                    return Err(AgentError::Provider(format!(
                        "compaction stream error: {e}"
                    )));
                }
                _ => {}
            }
        }

        if text.trim().is_empty() {
            return Err(AgentError::Provider("compaction LLM returned empty".into()));
        }

        Ok(text)
    }

    /// Get or build a provider by name from the global provider catalog.
    /// Returns the global default if `name` matches `self.config.default_provider`.
    /// Resolve the named provider from the config catalog, building &
    /// caching it on first request. Empty / unknown names fall back to
    /// the default provider this `AgentCore` was constructed with.
    /// Public so out-of-loop callers (CLI gatekeeper, validators) can
    /// pin a non-default provider per call without rebuilding the
    /// whole agent.
    pub async fn provider_for(&self, provider_name: &str) -> Arc<dyn Provider> {
        if provider_name.is_empty() || provider_name == self.config.default_provider {
            return self.provider.clone();
        }

        if let Some(cached) = self.provider_cache.read().await.get(provider_name) {
            return cached.clone();
        }

        let built: Arc<dyn Provider> = if let Some(pc) = self.config.providers.get(provider_name)
            && let Ok(resolved) = pc.resolved()
        {
            Arc::from(create_provider(provider_name, resolved))
        } else {
            tracing::warn!(
                "session requests provider '{provider_name}' not in catalog, using default"
            );
            return self.provider.clone();
        };

        self.provider_cache
            .write()
            .await
            .insert(provider_name.to_string(), built.clone());
        built
    }

    /// Connect any extra MCP servers needed by a session (additive over global).
    async fn session_mcp_servers(
        &self,
        session_id: &str,
        effective: &EffectiveSessionConfig,
    ) -> Vec<Arc<McpServer>> {
        let extra_names: Vec<String> = effective
            .mcp_servers
            .keys()
            .filter(|name| !self.config.mcp_servers.contains_key(*name))
            .cloned()
            .collect();

        if extra_names.is_empty() {
            return Vec::new();
        }

        if let Some(cached) = self.session_mcp.read().await.get(session_id) {
            return cached.clone();
        }

        let mut servers = Vec::new();
        for name in &extra_names {
            if let Some(mut cfg) = effective.mcp_servers.get(name).cloned() {
                if cfg.name.is_empty() {
                    cfg.name = name.clone();
                }
                match McpServer::connect(&cfg).await {
                    Ok(s) => {
                        tracing::info!("session MCP '{}': {} tools", name, s.tools().len());
                        servers.push(Arc::new(s));
                    }
                    Err(e) => tracing::warn!("session MCP '{name}' connect failed: {e}"),
                }
            }
        }

        self.session_mcp
            .write()
            .await
            .insert(session_id.to_string(), servers.clone());
        servers
    }

    pub async fn create_session(&self, workspace: &Path) -> String {
        self.create_session_with_channel(workspace, "cli").await
    }

    pub async fn create_session_with_channel(&self, workspace: &Path, channel: &str) -> String {
        let system_prompt =
            prompt::resolve_system_prompt(workspace, self.config.system_prompt_path.as_deref());
        let capabilities = self.capabilities_section().await;
        let mut full_prompt = format!(
            "{}\n\n{}\n\n{}",
            system_prompt,
            prompt::environment_section(workspace),
            capabilities
        );
        if self.config.research.enabled {
            full_prompt.push_str("\n\n---\n");
            full_prompt.push_str(&research::briefing::short(&self.config.research));
        }

        if channel == "telegram" {
            full_prompt.push_str("\n\n---\n");
            full_prompt.push_str(
                "Telegram bridge is active.\n\
                 - Messages from the user are forwarded from Telegram.\n\
                 - To send a file back to the user, use the telegram_attach tool with the absolute file path.\n\
                 - Mentioning a file path in plain text will NOT deliver it — you must call telegram_attach.\n\
                 - Keep responses concise — Telegram messages are read on mobile screens.",
            );
        }

        let metadata = SessionMetadata {
            name: None,
            provider: self.config.default_provider.clone(),
            model: self.config.default_model.clone(),
            channel: channel.into(),
            channel_id: None,
        };

        let session = Session::new(workspace.to_path_buf(), full_prompt, metadata);
        let id = session.id.clone();

        if let Err(e) = self.store.save(&session).await {
            tracing::error!("failed to persist new session: {e}");
        }

        // Apply per-session config.json overrides to metadata
        let sc = self.load_session_config_pub(&id);
        let effective = self.config.merge_session(&sc);

        let mut sessions = self.sessions.write().await;
        let mut session = session;
        session.metadata.provider = effective.provider;
        session.metadata.model = effective.model;
        sessions.insert(id.clone(), session);
        id
    }

    pub async fn send_prompt(&self, session_id: &str, text: &str) -> Result<AgentHandle> {
        self.dispatch_turn(session_id, UserPush::Text(text.to_string()))
            .await
    }

    /// Like `send_prompt` but takes pre-built content blocks (text + inline
    /// images). The `classifier_text` is what the background memory classifier
    /// will see — pass the human-readable summary of the message.
    pub async fn send_prompt_multimodal(
        &self,
        session_id: &str,
        blocks: Vec<ContentBlock>,
        classifier_text: String,
    ) -> Result<AgentHandle> {
        self.dispatch_turn(
            session_id,
            UserPush::Multimodal {
                blocks,
                classifier_text,
            },
        )
        .await
    }

    async fn dispatch_turn(&self, session_id: &str, push: UserPush) -> Result<AgentHandle> {
        let (tx, rx) = mpsc::channel(64);
        let (perm_tx, perm_rx) = mpsc::channel::<PermissionResponse>(4);

        // Load per-session config overlay (re-read each turn so edits take effect)
        let sc = self.load_session_config_pub(session_id);
        let effective = self.config.merge_session(&sc);

        let mut sessions = self.sessions.write().await;
        let session = sessions
            .get_mut(session_id)
            .ok_or_else(|| AgentError::Session("session not found".into()))?;
        session.state = SessionState::Active;

        // Update metadata to match effective config (model/provider may change between turns)
        session.metadata.provider = effective.provider.clone();
        session.metadata.model = effective.model.clone();

        let model = session.metadata.model.clone();
        let provider_name = session.metadata.provider.clone();

        // Resolve context window: per-session > per-provider > model lookup > global > 128K
        let provider_ctx = self
            .config
            .providers
            .get(&provider_name)
            .and_then(|pc| pc.context_window);
        let cw = effective
            .context_window
            .or(provider_ctx)
            .unwrap_or_else(|| history::model_context_window(&model));
        session.history.set_context_window_tokens(cw);
        tracing::info!(
            context_window = cw,
            estimated_tokens = session.history.estimated_tokens(),
            message_count = session.history.message_count(),
            needs_compaction = session.history.needs_compaction(),
            last_input_tokens = ?session.history.last_input_tokens(),
            provider = %provider_name,
            "pre-compact check"
        );

        let classifier_text = match &push {
            UserPush::Text(t) => t.clone(),
            UserPush::Multimodal {
                classifier_text, ..
            } => classifier_text.clone(),
        };
        match push {
            UserPush::Text(t) => session.history.push_user(&t),
            UserPush::Multimodal { blocks, .. } => session.history.push_user_multimodal(blocks),
        }
        session.updated_at = chrono::Utc::now();

        // Background memory classification (non-blocking, fire-and-forget).
        //
        // By default the hit lands in *today's draft file*, NOT durable
        // `MEMORY.md`. The daily digest later promotes it iff the same
        // rule re-appears across `promote_min_repeat_days` days. This
        // turns the classifier into a low-precision, high-recall feeder
        // for the scoring gate instead of a one-shot writer.
        //
        // Toggle with `memory.auto_classify_to_drafts = false` to fall
        // back to direct writes (legacy behaviour).
        {
            let ws = session.workspace.clone();
            let mdl = model.clone();
            let msg = classifier_text.clone();
            // Use the session's provider — not the global default — so the
            // model name is valid for the API endpoint. (Bug: using
            // self.provider sent "kimi-for-coding" to qwen → 404.)
            let provider_ref = self.provider_for(&provider_name).await;
            let sender_id = self.session_sender(session_id).await;
            let to_drafts = self.config.memory.auto_classify_to_drafts;
            tokio::spawn(async move {
                let provider_arc: std::sync::Arc<dyn Provider> = provider_ref;
                let Some(result) =
                    memory::classifier::classify(&*provider_arc, &mdl, &msg, sender_id.as_deref())
                        .await
                else {
                    return;
                };

                if to_drafts {
                    let entry = memory::types::MemoryEntry::new(
                        result.memory_type,
                        result.content.clone(),
                        "auto_classify",
                        result.scope.clone(),
                    );
                    match memory::store::MarkdownMemoryStore::append_daily(&ws, &entry, true) {
                        Ok(true) => tracing::info!(
                            scope = %result.scope,
                            ty = %result.memory_type,
                            "memory auto-captured to drafts: {}",
                            result.content
                        ),
                        Ok(false) => {
                            tracing::debug!("memory auto-capture (drafts): duplicate skipped")
                        }
                        Err(e) => tracing::warn!("memory auto-capture (drafts) write failed: {e}"),
                    }
                } else {
                    match memory::service::MemoryService::store(
                        &ws,
                        result.scope,
                        result.memory_type,
                        &result.content,
                        "auto",
                    ) {
                        Ok(true) => tracing::info!(
                            "memory auto-captured: [{}] {}",
                            result.memory_type,
                            result.content
                        ),
                        Ok(false) => tracing::debug!("memory auto-capture: duplicate skipped"),
                        Err(e) => tracing::warn!("memory auto-capture write failed: {e}"),
                    }
                }
            });
        }

        let needs_compact = session.history.needs_compaction();
        let compact_text = if needs_compact {
            session.history.messages_for_compaction(4)
        } else {
            None
        };
        let before_msgs = session.history.message_count();
        let workspace_for_compaction = session.workspace.clone();

        // Drop sessions lock before LLM call to avoid blocking other requests
        drop(sessions);

        // LLM-based compaction with deterministic fallback
        let llm_summary = if let Some(text_for_llm) = compact_text {
            tracing::info!(before_msgs, "attempting LLM-based compaction");
            let provider_arc = self.provider_for(&provider_name).await;

            // Pre-compaction flush: ask the model (silent turn) to extract
            // any rules-of-thumb / corrections from the history we are
            // about to discard, and append them to the project's daily
            // draft file. Best-effort; never blocks compaction.
            if self.config.memory.daily_enabled && self.config.memory.pre_compaction_flush {
                memory::digest::pre_compaction_flush(
                    &*provider_arc,
                    &model,
                    &workspace_for_compaction,
                    &memory::types::MemoryScope::Project,
                    &text_for_llm,
                )
                .await;
            }

            match Self::llm_summarize(&*provider_arc, &model, &text_for_llm).await {
                Ok(summary) => {
                    tracing::info!("LLM compaction succeeded");
                    Some(summary)
                }
                Err(e) => {
                    tracing::warn!("LLM compaction failed, falling back to deterministic: {e}");
                    None
                }
            }
        } else {
            None
        };

        // Re-acquire sessions lock to apply compaction
        let mut sessions = self.sessions.write().await;
        let session = sessions
            .get_mut(session_id)
            .ok_or_else(|| AgentError::Session("session not found".into()))?;

        let compacted = if needs_compact {
            if let Some(summary) = llm_summary {
                session.history.compact_with_llm_summary(&summary, 4);
                session.history.set_last_input_tokens(None);
            } else {
                session.history.auto_compact();
            }
            let after = session.history.message_count();
            Some((before_msgs, after))
        } else {
            None
        };

        if let Some((before, after)) = compacted {
            tracing::info!("context compacted: {before} msgs -> {after} msgs");
            let _ = tx
                .send(AgentEvent::ContextCompacted {
                    before_msgs: before,
                    after_msgs: after,
                })
                .await;
            // After compaction, older turns (and any image blocks they
            // owned) are gone from history. Run a best-effort GC over the
            // session's artifacts dir to reclaim disk for images that no
            // JSONL line still references. Never block the user reply on
            // GC failures — log and move on.
            match self.store.gc_orphan_image_artifacts(session_id).await {
                Ok(n) if n > 0 => {
                    tracing::info!(
                        removed = n,
                        session = session_id,
                        "post-compaction artifact GC reclaimed {n} orphan images"
                    );
                }
                Ok(_) => {}
                Err(e) => {
                    tracing::warn!(
                        session = session_id,
                        "post-compaction artifact GC failed: {e}"
                    );
                }
            }
        }

        let mut history = session.history.clone();
        let original_system_prompt = history.system_prompt().to_string();

        // Inject per-session prompt.md as context (if present)
        let session_root = self.store.session_root(session_id);
        let prompt_path = effective
            .system_prompt_path
            .as_ref()
            .map(|p| session_root.join(p))
            .unwrap_or_else(|| session_root.join("prompt.md"));
        if let Ok(extra) = tokio::fs::read_to_string(&prompt_path).await {
            let trimmed = extra.trim();
            if !trimmed.is_empty() {
                history.inject_system_context(&format!("\n\n[Session instructions]\n{trimmed}"));
            }
        }

        // Inject persistent memory rules into system prompt (plus the active
        // author's per-user rules when a Telegram sender is set for this turn).
        let sender_for_rules = self.session_sender(session_id).await;
        let memory_rules = memory::service::MemoryService::load_rules_for(
            &session.workspace,
            sender_for_rules.as_deref(),
        );
        if !memory_rules.is_empty() {
            history.inject_system_context(&format!("\n\n{memory_rules}"));
        }

        // "Recent shift": surface the last few days of un-promoted draft
        // memory entries so the model sees fresh context without waiting
        // for a daily-digest promotion. Cheap (just reads ≤2 small md
        // files per scope) and bumps the recall counter as a side
        // effect, which feeds promotion scoring.
        if self.config.memory.daily_enabled {
            let mut shift_blocks: Vec<String> = Vec::new();
            if let Some(b) = memory::daily::recent_shift_block(
                &session.workspace,
                &memory::types::MemoryScope::Project,
                &self.config.memory,
            ) {
                shift_blocks.push(b);
            }
            if let Some(sender) = sender_for_rules.as_deref()
                && let Some(b) = memory::daily::recent_shift_block(
                    &session.workspace,
                    &memory::types::MemoryScope::User(sender.to_string()),
                    &self.config.memory,
                )
            {
                shift_blocks.push(b);
            }
            if !shift_blocks.is_empty() {
                history.inject_system_context(&format!("\n\n{}", shift_blocks.join("\n\n")));
            }
        }

        let artifacts = self.store.artifacts_dir(session_id);
        if let Err(e) = tokio::fs::create_dir_all(&artifacts).await {
            tracing::warn!("could not create artifacts dir: {e}");
        }

        let cwd = if session.workspace.as_os_str().is_empty() || !session.workspace.exists() {
            artifacts
        } else {
            session.workspace.clone()
        };

        // Per-provider max_tokens / temperature (provider-level override > session > global)
        let eff_max_tokens = self
            .config
            .providers
            .get(&provider_name)
            .and_then(|pc| pc.max_tokens)
            .unwrap_or(effective.max_tokens);
        let eff_temperature = self
            .config
            .providers
            .get(&provider_name)
            .and_then(|pc| pc.temperature)
            .or(effective.temperature);

        let loop_config = LoopConfig {
            max_iterations: effective.max_iterations,
            cwd,
            model: model.clone(),
            max_tokens: eff_max_tokens,
            temperature: eff_temperature,
            reasoning: effective.reasoning.clone(),
            provider: provider_name.clone(),
            health: Some(self.model_health.clone()),
        };

        let session_workspace = session.workspace.clone();

        let cancel = CancellationToken::new();
        self.cancels
            .write()
            .await
            .insert(session_id.to_string(), cancel.clone());
        drop(sessions);

        // Validate model belongs to provider before making any API calls.
        if let Some(pc) = self.config.providers.get(&provider_name) {
            let valid = pc.models.iter().any(|x| x == &model)
                || pc.model_aliases.contains_key(&model)
                || pc.model_aliases.values().any(|v| v == &model);
            if !valid {
                let available: Vec<_> = pc
                    .models
                    .iter()
                    .chain(pc.model_aliases.keys())
                    .take(6)
                    .cloned()
                    .collect();
                let err_msg = format!(
                    "Model '{}' not found on provider '{}'. Try: {}",
                    model,
                    provider_name,
                    available.join(", ")
                );
                let _ = tx.send(AgentEvent::Error(err_msg)).await;
                let _ = tx.send(AgentEvent::Idle).await;
                if let Some(s) = self.sessions.write().await.get_mut(session_id) {
                    s.state = SessionState::Idle;
                }
                return Ok(AgentHandle {
                    events: rx,
                    permissions: perm_tx,
                });
            }
        }

        // Per-session provider (falls back to global if unchanged)
        let session_provider = self.provider_for(&provider_name).await;

        // Build tool registry with per-session MCP + skills
        let tools = self
            .build_tool_registry_for(
                session_id,
                &effective,
                &session_provider,
                &model,
                &session_workspace,
            )
            .await;
        let agent_loop = AgentLoop::new(provider_to_box(&session_provider), tools, loop_config);

        let session_id_owned = session_id.to_string();
        let sessions_ref = self.sessions.clone();
        let store_ref = self.store.clone();

        // Every turn gets a span with (session_id, provider, model). All
        // events emitted from `agent_loop.run` — tool calls, usage, errors
        // — inherit these attributes, so operators can grep one session's
        // worth of logs by a single `session` field without hunting
        // through chat/thread IDs.
        let turn_span = tracing::info_span!(
            "agent_turn",
            session = %session_id_owned,
            provider = %provider_name,
            model = %model,
        );
        use tracing::Instrument;

        tokio::spawn(
            async move {
                let result = agent_loop
                    .run(&mut history, tx.clone(), cancel, Some(perm_rx))
                    .await;
                match &result {
                    Ok(usage) => {
                        crate::types::TURN_COMPLETED_COUNT
                            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                        tracing::info!(
                            "turn complete [{}]: {} tokens",
                            session_id_owned,
                            usage.total_tokens()
                        );
                        if usage.input_tokens > 0 {
                            history.set_last_input_tokens(usage.input_tokens);
                        }
                    }
                    Err(e) => {
                        crate::types::TURN_ERROR_COUNT
                            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                        tracing::error!("turn error [{}]: {e}", session_id_owned);
                        let _ = tx.send(AgentEvent::Error(e.to_string())).await;
                    }
                }

                // Merge mutated history back into the session and persist
                history.restore_system_prompt(original_system_prompt);
                let mut sessions = sessions_ref.write().await;
                if let Some(session) = sessions.get_mut(&session_id_owned) {
                    session.history = history;
                    session.state = SessionState::Idle;
                    session.updated_at = chrono::Utc::now();
                    if let Err(e) = store_ref.save(session).await {
                        tracing::error!("failed to persist session [{}]: {e}", session_id_owned);
                    }
                }
            }
            .instrument(turn_span),
        );

        Ok(AgentHandle {
            events: rx,
            permissions: perm_tx,
        })
    }

    pub async fn is_session_active(&self, session_id: &str) -> bool {
        self.sessions
            .read()
            .await
            .get(session_id)
            .is_some_and(|s| s.state == SessionState::Active)
    }

    /// Append a user message to the session history without starting a new turn.
    pub async fn queue_message(&self, session_id: &str, text: &str) {
        if let Some(session) = self.sessions.write().await.get_mut(session_id) {
            session.history.push_user(text);
        }
    }

    /// Append a multimodal user message (text + images) to the history without
    /// starting a new turn. Used when the bot is busy and a new media-bearing
    /// message arrives mid-turn.
    pub async fn queue_message_multimodal(&self, session_id: &str, blocks: Vec<ContentBlock>) {
        if let Some(session) = self.sessions.write().await.get_mut(session_id) {
            session.history.push_user_multimodal(blocks);
        }
    }

    /// Trigger history compaction for a session. Returns (before, after) message counts.
    /// No-op if compaction not needed.
    pub async fn compact_session(&self, session_id: &str) -> Option<(usize, usize)> {
        if let Some(session) = self.sessions.write().await.get_mut(session_id) {
            session.history.auto_compact()
        } else {
            None
        }
    }

    pub async fn abort(&self, session_id: &str) {
        if let Some(cancel) = self.cancels.read().await.get(session_id) {
            cancel.cancel();
        }
        if let Some(session) = self.sessions.write().await.get_mut(session_id) {
            session.state = SessionState::Idle;
        }
    }

    /// Fire-and-forget: ask the LLM to extract any rules-of-thumb from
    /// the (closing) session's transcript and append them to the
    /// project's daily memory draft. Called from `/new`-style handlers
    /// that swap one session for another. Returns immediately; the
    /// summary runs in a background tokio task and never blocks the
    /// caller. No-op when `memory.session_close_summary = false`.
    pub async fn close_session_summary(&self, session_id: &str) {
        if !self.config.memory.daily_enabled || !self.config.memory.session_close_summary {
            return;
        }
        let (workspace, transcript, provider_name, model) = {
            let sessions = self.sessions.read().await;
            let Some(session) = sessions.get(session_id) else {
                return;
            };
            // `messages_for_compaction(0)` returns the full transcript
            // (no recent-tail kept). If there is nothing to summarize
            // (e.g. fresh session), skip.
            let Some(text) = session.history.messages_for_compaction(0) else {
                return;
            };
            (
                session.workspace.clone(),
                text,
                session.metadata.provider.clone(),
                session.metadata.model.clone(),
            )
        };
        let provider = self.provider_for(&provider_name).await;
        tokio::spawn(async move {
            memory::digest::session_close(
                &*provider,
                &model,
                &workspace,
                &memory::types::MemoryScope::Project,
                &transcript,
            )
            .await;
        });
    }

    pub async fn list_sessions(&self) -> Vec<SessionSummary> {
        self.list_sessions_paged(0, usize::MAX).await
    }

    /// Look up the workspace path tied to a live session id.
    ///
    /// Returns `None` if the session was never created or has been
    /// evicted. Callers (e.g. the TG `/memory` operator command) use
    /// this to scope project-memory queries to the same directory the
    /// agent was talking from.
    pub async fn session_workspace(&self, session_id: &str) -> Option<std::path::PathBuf> {
        self.sessions
            .read()
            .await
            .get(session_id)
            .map(|s| s.workspace.clone())
    }

    /// Paged variant of `list_sessions`. Sorts by `updated_at` descending
    /// (most recently touched session first), then applies `skip` + `limit`.
    ///
    /// Callers can pass `limit = usize::MAX` to disable truncation. A
    /// `skip` beyond the total count returns an empty vec — never panics.
    /// Intended for CLI `/sessions --skip N --limit M` and future UI
    /// paging where listing 500 stale sessions would be useless.
    pub async fn list_sessions_paged(&self, skip: usize, limit: usize) -> Vec<SessionSummary> {
        let sessions = self.sessions.read().await;
        let mut summaries: Vec<SessionSummary> = sessions.values().map(|s| s.summary()).collect();
        // Newest first — operators almost always want the recent tail.
        summaries.sort_by(|a, b| b.updated_at.cmp(&a.updated_at));
        summaries.into_iter().skip(skip).take(limit).collect()
    }

    pub fn list_models(&self) -> Vec<types::ModelInfo> {
        self.provider.models()
    }

    /// List models for a specific provider as `(provider_name, model_id)` pairs.
    pub async fn provider_models(&self, provider_name: &str) -> Vec<(String, String)> {
        if let Some(pc) = self.config.providers.get(provider_name) {
            pc.models
                .iter()
                .map(|m| (provider_name.to_string(), m.clone()))
                .collect()
        } else {
            Vec::new()
        }
    }

    /// List all configured providers with their available models.
    pub fn list_providers(&self) -> Vec<ProviderInfo> {
        let mut result = Vec::new();
        for (name, pc) in &self.config.providers {
            let active = name == &self.config.default_provider;
            result.push(ProviderInfo {
                name: name.clone(),
                models: pc.models.clone(),
                active,
            });
        }
        result.sort_by(|a, b| a.name.cmp(&b.name));
        result
    }

    /// Switch the provider and model for a specific session.
    /// Writes config.json into the session directory.
    pub async fn set_session_provider(
        &self,
        session_id: &str,
        provider: Option<&str>,
        model: Option<&str>,
    ) -> Result<()> {
        if let Some(p) = provider
            && !self.config.providers.contains_key(p)
        {
            return Err(AgentError::Config(format!("unknown provider: {p}")));
        }

        // Validate that the model belongs to the target provider.
        // Resolve the effective provider (explicit or current session's).
        if let Some(m) = model {
            let target_provider = provider
                .map(|s| s.to_string())
                .or_else(|| {
                    let sessions = self.sessions.try_read().ok()?;
                    sessions
                        .get(session_id)
                        .map(|s| s.metadata.provider.clone())
                })
                .unwrap_or_else(|| self.config.default_provider.clone());
            if let Some(pc) = self.config.providers.get(&target_provider) {
                let valid = pc.models.iter().any(|x| x == m)
                    || pc.model_aliases.contains_key(m)
                    || pc.model_aliases.values().any(|v| v == m);
                if !valid {
                    let available: Vec<_> = pc
                        .models
                        .iter()
                        .chain(pc.model_aliases.keys())
                        .take(8)
                        .cloned()
                        .collect();
                    return Err(AgentError::Config(format!(
                        "model '{m}' not found on provider '{target_provider}'. Available: {}",
                        available.join(", ")
                    )));
                }
            }
        }

        let session_root = self.store.session_root(session_id);
        let config_path = session_root.join("config.json");

        let mut sc = if config_path.exists() {
            SessionConfig::from_file(&config_path).unwrap_or_default()
        } else {
            SessionConfig::default()
        };

        if let Some(p) = provider {
            sc.default_provider = Some(p.to_string());
        }
        if let Some(m) = model {
            sc.default_model = Some(m.to_string());
        }

        tokio::fs::create_dir_all(&session_root)
            .await
            .map_err(|e| AgentError::Session(format!("cannot create session dir: {e}")))?;
        let json = serde_json::to_string_pretty(&sc)
            .map_err(|e| AgentError::Config(format!("serialize: {e}")))?;
        tokio::fs::write(&config_path, json)
            .await
            .map_err(|e| AgentError::Session(format!("cannot write config.json: {e}")))?;

        // Update in-memory metadata immediately
        let effective = self.config.merge_session(&sc);
        if let Some(session) = self.sessions.write().await.get_mut(session_id) {
            session.metadata.provider = effective.provider;
            session.metadata.model = effective.model;
        }

        // Invalidate cached provider so next turn rebuilds it (cache is keyed by provider name)
        if let Some(p) = provider {
            self.provider_cache.write().await.remove(p);
        }

        Ok(())
    }

    /// Set reasoning level for a session. Writes to config.json.
    pub async fn set_session_reasoning(&self, session_id: &str, reasoning: &str) -> Result<()> {
        let val = match reasoning {
            "off" | "low" | "medium" | "high" => reasoning.to_string(),
            _ => {
                return Err(AgentError::Config(format!(
                    "invalid reasoning level: {reasoning}"
                )));
            }
        };

        let session_root = self.store.session_root(session_id);
        let config_path = session_root.join("config.json");

        let mut sc = if config_path.exists() {
            SessionConfig::from_file(&config_path).unwrap_or_default()
        } else {
            SessionConfig::default()
        };

        sc.reasoning = if val == "off" { None } else { Some(val) };

        tokio::fs::create_dir_all(&session_root)
            .await
            .map_err(|e| AgentError::Session(format!("cannot create session dir: {e}")))?;
        let json = serde_json::to_string_pretty(&sc)
            .map_err(|e| AgentError::Config(format!("serialize: {e}")))?;
        tokio::fs::write(&config_path, json)
            .await
            .map_err(|e| AgentError::Session(format!("cannot write config.json: {e}")))?;

        Ok(())
    }

    /// Set yolo timestamp for a session. Writes to config.json.
    /// Pass `Some(ts)` to enable with a specific unix timestamp, `None` to disable.
    pub async fn set_session_yolo(&self, session_id: &str, enabled_at: Option<i64>) -> Result<()> {
        let session_root = self.store.session_root(session_id);
        let config_path = session_root.join("config.json");

        let mut sc = if config_path.exists() {
            SessionConfig::from_file(&config_path).unwrap_or_default()
        } else {
            SessionConfig::default()
        };

        sc.yolo_enabled_at = enabled_at;

        tokio::fs::create_dir_all(&session_root)
            .await
            .map_err(|e| AgentError::Session(format!("cannot create session dir: {e}")))?;
        let json = serde_json::to_string_pretty(&sc)
            .map_err(|e| AgentError::Config(format!("serialize: {e}")))?;
        tokio::fs::write(&config_path, json)
            .await
            .map_err(|e| AgentError::Session(format!("cannot write config.json: {e}")))?;

        Ok(())
    }

    /// Set allow-list for a session. Writes to config.json.
    pub async fn set_session_allow_list(&self, session_id: &str, tools: &[String]) -> Result<()> {
        let session_root = self.store.session_root(session_id);
        let config_path = session_root.join("config.json");

        let mut sc = if config_path.exists() {
            SessionConfig::from_file(&config_path).unwrap_or_default()
        } else {
            SessionConfig::default()
        };

        sc.allow_list = if tools.is_empty() {
            None
        } else {
            Some(tools.to_vec())
        };

        tokio::fs::create_dir_all(&session_root)
            .await
            .map_err(|e| AgentError::Session(format!("cannot create session dir: {e}")))?;
        let json = serde_json::to_string_pretty(&sc)
            .map_err(|e| AgentError::Config(format!("serialize: {e}")))?;
        tokio::fs::write(&config_path, json)
            .await
            .map_err(|e| AgentError::Session(format!("cannot write config.json: {e}")))?;

        Ok(())
    }

    /// Get the current reasoning level for a session.
    pub async fn session_reasoning(&self, session_id: &str) -> Option<String> {
        let sc = self.load_session_config_pub(session_id);
        sc.reasoning
    }

    /// Persist a channel-specific key so the channel→session mapping survives restarts.
    /// For Telegram: `"tg:{chat_id}:{thread_id}"`.
    pub async fn set_session_channel_id(&self, session_id: &str, channel_id: &str) {
        let mut sessions = self.sessions.write().await;
        if let Some(session) = sessions.get_mut(session_id) {
            session.metadata.channel_id = Some(channel_id.to_string());
            if let Err(e) = self.store.save(session).await {
                tracing::error!("failed to persist channel_id for {session_id}: {e}");
            }
        }
    }

    /// Return `(channel_id, session_id)` pairs for sessions that have a channel_id.
    /// When multiple sessions share the same channel_id, only the most recently
    /// updated one is returned.
    pub async fn channel_session_mappings(&self) -> Vec<(String, String)> {
        let sessions = self.sessions.read().await;
        let mut best: std::collections::HashMap<String, (&str, chrono::DateTime<chrono::Utc>)> =
            std::collections::HashMap::new();
        for s in sessions.values() {
            if let Some(cid) = &s.metadata.channel_id {
                let entry = best.entry(cid.clone()).or_insert((&s.id, s.updated_at));
                if s.updated_at > entry.1 {
                    *entry = (&s.id, s.updated_at);
                }
            }
        }
        best.into_iter()
            .map(|(cid, (sid, _))| (cid, sid.to_string()))
            .collect()
    }

    /// Get the currently active provider and model for a session.
    /// Sum of token usage across all assistant turns in a session.
    pub async fn session_total_usage(&self, session_id: &str) -> types::TurnUsage {
        let sessions = self.sessions.read().await;
        let mut total = types::TurnUsage::default();
        if let Some(session) = sessions.get(session_id) {
            for msg in session.history.messages() {
                if let Some(u) = &msg.usage {
                    total.input_tokens += u.input_tokens;
                    total.output_tokens += u.output_tokens;
                    total.cache_read_tokens += u.cache_read_tokens;
                    total.cache_write_tokens += u.cache_write_tokens;
                }
            }
        }
        total
    }

    /// Returns (estimated_tokens, context_window_tokens) for a session.
    pub async fn session_context_usage(&self, session_id: &str) -> Option<(usize, u32)> {
        let sessions = self.sessions.read().await;
        sessions.get(session_id).map(|s| {
            (
                s.history.estimated_tokens(),
                s.history.context_window_tokens(),
            )
        })
    }

    pub async fn session_provider_model(&self, session_id: &str) -> (String, String) {
        let sc = self.load_session_config_pub(session_id);
        let effective = self.config.merge_session(&sc);
        (effective.provider, effective.model)
    }

    /// Resolve the **default** (provider, model) tuple — what a brand-new
    /// session would inherit before any per-session overrides are applied.
    ///
    /// Use this when you need to make a routing decision (vision-capability,
    /// reasoning policy, …) **before** a session ID exists. Replaces the
    /// previous `session_provider_model("__nonexistent__")` hack which leaned
    /// on the fact that `load_session_config_pub` of a missing dir returns
    /// `SessionConfig::default()`. That worked, but baking a magic
    /// session-id sentinel into the contract was a code smell — this method
    /// makes the intent explicit and never touches the filesystem.
    pub fn default_provider_model(&self) -> (String, String) {
        let effective = self.config.merge_session(&SessionConfig::default());
        (effective.provider, effective.model)
    }

    pub async fn restore_sessions(&self) -> Result<Vec<String>> {
        let summaries = self.store.list().await?;
        let mut restored = Vec::new();
        for summary in summaries {
            match self.store.load(&summary.id).await {
                Ok(Some(mut session)) => {
                    // Apply per-session config.json overrides (provider/model)
                    let sc = self.load_session_config_pub(&session.id);
                    let effective = self.config.merge_session(&sc);
                    session.metadata.provider = effective.provider;
                    session.metadata.model = effective.model;

                    restored.push(session.id.clone());
                    self.sessions
                        .write()
                        .await
                        .insert(session.id.clone(), session);
                }
                Ok(None) => {
                    tracing::warn!(
                        "session {} listed but not loadable",
                        &summary.id[..8.min(summary.id.len())]
                    );
                }
                Err(e) => {
                    tracing::warn!(
                        "failed to load session {}: {e}",
                        &summary.id[..8.min(summary.id.len())]
                    );
                }
            }
        }
        Ok(restored)
    }

    pub async fn fork_session(
        &self,
        session_id: &str,
        branch_name: Option<String>,
    ) -> Result<String> {
        let sessions = self.sessions.read().await;
        let parent = sessions
            .get(session_id)
            .ok_or_else(|| AgentError::Session("session not found".into()))?;
        let forked = parent.fork(branch_name);
        let new_id = forked.id.clone();
        self.store.save(&forked).await?;
        drop(sessions);
        self.sessions.write().await.insert(new_id.clone(), forked);
        Ok(new_id)
    }

    pub fn list_skills(&self) -> Vec<(String, String)> {
        let resolver = SkillResolver::new(self.config.skill_roots.clone());
        resolver
            .list()
            .into_iter()
            .map(|(name, hit)| (name, hit.path.display().to_string()))
            .collect()
    }

    pub async fn list_mcp_servers(&self) -> Vec<(String, usize)> {
        let reg = self.mcp_registry.read().await;
        reg.servers()
            .iter()
            .map(|s| (s.name.clone(), s.tools().len()))
            .collect()
    }

    async fn capabilities_section(&self) -> String {
        let mut parts = Vec::new();

        // Skills — same description source as the SkillTool catalog
        // so the LLM sees identical hints in the system preface and
        // in the tool's JSON schema. JSON skills get their description
        // from the spec; MD skills from front-matter.
        let skills = self.list_skills();
        if !skills.is_empty() {
            let mut s = String::from("Available skills (use the Skill tool to activate):\n");
            let resolver = SkillResolver::new(self.config.skill_roots.clone());
            for (name, _path) in &skills {
                let desc = resolver
                    .resolve(name)
                    .and_then(|hit| skill::resolver::read_skill_description(&hit));
                match desc {
                    Some(d) => s.push_str(&format!("- {name}: {d}\n")),
                    None => s.push_str(&format!("- {name}\n")),
                }
            }
            parts.push(s);
        }

        // MCP servers & tools
        let mcp_reg = self.mcp_registry.read().await;
        let servers = mcp_reg.servers();
        if !servers.is_empty() {
            let mut s = String::from("Connected MCP servers and their tools:\n");
            for server in servers {
                let tools = server.tools();
                s.push_str(&format!("- {} ({} tools):", server.name, tools.len()));
                for tool in tools {
                    let desc = tool.description.as_deref().unwrap_or("");
                    s.push_str(&format!("\n  • {}: {desc}", tool.name));
                }
                s.push('\n');
            }
            parts.push(s);
        }

        let out = if parts.is_empty() {
            String::new()
        } else {
            format!("---\nCapabilities:\n{}", parts.join("\n"))
        };
        tracing::debug!(
            target: "naked::capabilities",
            chars = out.len(),
            "[capabilities-prefix] {}",
            out.replace('\n', " ⏎ ").chars().take(2_000).collect::<String>()
        );
        out
    }

    pub async fn refresh_skills_and_mcp(&self) {
        self.init_mcp().await;
        let skills = self.list_skills();
        tracing::info!(
            "Refresh: {} skills, {} MCP servers",
            skills.len(),
            self.mcp_registry.read().await.servers().len()
        );
    }

    async fn build_tool_registry_for(
        &self,
        session_id: &str,
        effective: &EffectiveSessionConfig,
        provider: &Arc<dyn Provider>,
        model: &str,
        workspace: &Path,
    ) -> ToolRegistry {
        let sub_agent = SubAgentTool::new(
            provider.clone(),
            model.to_string(),
            self.config.tool_timeout_secs,
            self.config.exa_api_keys.clone(),
        )
        .with_registry(self.agent_registry.clone());

        let mut tools: Vec<Box<dyn tool::Tool>> = vec![
            Box::new(BashTool::new(self.config.tool_timeout_secs)),
            Box::new(ReadFileTool),
            Box::new(WriteFileTool),
            Box::new(EditFileTool),
            Box::new(GlobSearchTool),
            Box::new(GrepSearchTool),
            Box::new(sub_agent),
            Box::new(AgentStatusTool::new(self.agent_registry.clone())),
            Box::new(AgentStopTool::new(self.agent_registry.clone())),
            Box::new(WebSearchTool::new(
                self.exa_key_pool.clone(),
                self.tavily_key_pool.clone(),
                self.serpapi_key_pool.clone(),
            )),
            Box::new(WebFetchTool::with_components(
                self.cloud_scraper.clone(),
                self.host_policy.clone(),
            )),
            Box::new(WebFetchTlsTool::new()),
            Box::new(WebFetchWaybackTool::new()),
            Box::new({
                let ctx = tool::memory::MemoryContext::new();
                ctx.set_user_id(self.session_sender(session_id).await);
                MemoryTool::with_context(workspace.to_path_buf(), ctx)
            }),
        ];

        // Research tools. Always registered so the `/research ask` flow can
        // call `research_status` from any session — but `research_save`,
        // `research_list`, and `research_save_cursor` early-return with an
        // error unless `research_context` is set (coordinator does this for
        // the turn and clears it after).
        if self.config.research.enabled {
            tools.push(Box::new(ResearchSaveTool::new(
                self.research_store.clone(),
                self.research_context.clone(),
                self.config.research.gatekeeper.clone(),
            )));
            tools.push(Box::new(ResearchListTool::new(
                self.research_store.clone(),
                self.research_context.clone(),
            )));
            tools.push(Box::new(ResearchSaveCursorTool::new(
                self.research_store.clone(),
                self.research_context.clone(),
            )));
            tools.push(Box::new(ResearchStatusTool::new(
                self.research_store.clone(),
            )));

            // High-level orchestration tools (usable from any chat turn)
            tools.push(Box::new(ResearchCreateTool::new(
                self.research_store.clone(),
                self.config.research.clone(),
            )));
            tools.push(Box::new(ResearchListSpecsTool::new(
                self.research_store.clone(),
                self.config.research.clone(),
            )));
            tools.push(Box::new(ResearchMetricsTool::new(
                self.research_store.clone(),
            )));
            tools.push(Box::new(ResearchHelpTool::new(
                self.config.research.clone(),
            )));
            tools.push(Box::new(ResearchFindingsTool::new(
                self.research_store.clone(),
            )));
            tools.push(Box::new(ResearchSetTargetTool::new(
                self.research_store.clone(),
                self.research_context.clone(),
            )));
            if let Some(weak) = self.self_ref.read().unwrap().clone() {
                tools.push(Box::new(ResearchLaunchTool::new(weak.clone())));
                tools.push(Box::new(ResearchUpdateSpecTool::new(weak.clone())));
                tools.push(Box::new(ResearchSetScheduleTool::new(weak.clone())));
                tools.push(Box::new(ResearchPauseTool::new(weak.clone())));
                tools.push(Box::new(ResearchResumeTool::new(weak)));
            }
        }

        let skill_roots = &effective.skill_roots;
        tracing::debug!("skill_roots: {:?}", skill_roots);
        let resolver = SkillResolver::new(skill_roots.clone());
        let available = resolver.list();
        tracing::info!(
            "Skills: {} found in {} roots",
            available.len(),
            skill_roots.len()
        );
        for (name, hit) in &available {
            tracing::debug!("  skill: {name} -> {}", hit.path.display());
        }
        let orphans = resolver.find_orphans();
        if !orphans.is_empty() {
            tracing::warn!(
                "Skills: {} directory(ies) in skill_roots have NO SKILL.{{json,md,toml}} manifest — \
                 invisible to the `Skill` tool. Add a manifest or remove the directory:",
                orphans.len()
            );
            for (root, path) in &orphans {
                tracing::warn!(
                    "  orphan skill dir: {} (root: {})",
                    path.display(),
                    root.display()
                );
            }
        }
        tools.push(Box::new(SkillTool::new(resolver, &available)));

        // Global MCP servers
        let mcp_reg = self.mcp_registry.read().await;
        for server in mcp_reg.servers() {
            tools.extend(McpToolWrapper::wrap_all(Arc::clone(server)));
        }

        // Per-session MCP servers (additive)
        let extra = self.session_mcp_servers(session_id, effective).await;
        for server in &extra {
            tools.extend(McpToolWrapper::wrap_all(Arc::clone(server)));
        }

        // Append extra tools injected by the embedding binary.
        for factory in self.extra_tool_factories.read().await.iter() {
            tools.push(factory());
        }

        ToolRegistry::new(tools)
    }

    // ── Research public API ────────────────────────────────────────────
    //
    // These methods are the thin layer the Telegram bot and CLI call into.
    // They hide the coordinator / store plumbing so callers don't need to
    // build it themselves; the trade-off is that AgentCore carries a research
    // store by construction, which is cheap (no network, one directory).

    /// Create a new research and persist its spec.
    ///
    /// `topic` is free text; the returned id is derived from it (slug + short
    /// random suffix) so it fits in a URL / systemd template.
    pub async fn create_research(
        &self,
        topic: &str,
        sources: Vec<String>,
        session_id: Option<String>,
        chat_id: Option<i64>,
        thread_id: Option<i32>,
    ) -> Result<ResearchSpec> {
        if !self.config.research.enabled {
            return Err(AgentError::Config("research subsystem is disabled".into()));
        }
        let mut seeds = sources;
        if seeds.is_empty() {
            seeds = self.config.research.default_sources.clone();
        }
        let spec = ResearchSpec {
            id: new_research_id(topic),
            topic: topic.trim().to_string(),
            sources: seeds,
            interval_seconds: None,
            run_at: None,
            cron: None,
            task_timeout_seconds: None,
            session_id,
            chat_id,
            thread_id,
            provider: self.config.research.provider.clone(),
            model: self.config.research.model.clone(),
            max_iterations: Some(self.config.research.max_iterations),
            max_wall_seconds: Some(self.config.research.max_wall_seconds),
            created_at: chrono::Utc::now(),
            paused: false,
            pause_reason: None,
        };
        self.research_store.create_spec(&spec).await?;
        self.scheduler_hook()
            .notify(research::SchedulerEvent::SpecCreated {
                spec_id: spec.id.clone(),
            })
            .await;
        Ok(spec)
    }

    pub async fn list_research(&self) -> Result<Vec<ResearchSpec>> {
        self.research_store.list_specs().await
    }

    pub async fn load_research(&self, id: &str) -> Result<ResearchSpec> {
        self.research_store.load_spec(id).await
    }

    pub async fn delete_research(&self, id: &str) -> Result<()> {
        self.research_store.delete_spec(id).await?;
        self.scheduler_hook()
            .notify(research::SchedulerEvent::SpecRemoved {
                spec_id: id.to_string(),
            })
            .await;
        Ok(())
    }

    pub async fn set_research_paused(&self, id: &str, paused: bool) -> Result<()> {
        self.set_research_paused_with_reason(id, paused, None).await
    }

    /// Pause/resume a research spec and stamp a human-readable reason.
    /// `reason` is honoured ONLY when `paused == true`; on resume it is
    /// always cleared back to `None` so a subsequent `pause` doesn't
    /// inherit the previous reason silently.
    ///
    /// Used by the scheduler when auto-pausing after a failure streak
    /// (`reason = Some("auto: 5 consecutive failures — last error: …")`)
    /// so `/research ls` can show "auto" vs. user-initiated pauses
    /// without operators having to dig through `journalctl`.
    pub async fn set_research_paused_with_reason(
        &self,
        id: &str,
        paused: bool,
        reason: Option<String>,
    ) -> Result<()> {
        let mut spec = self.research_store.load_spec(id).await?;
        spec.paused = paused;
        spec.pause_reason = if paused { reason } else { None };
        self.research_store.save_spec(&spec).await?;
        self.scheduler_hook()
            .notify(research::SchedulerEvent::SpecUpdated {
                spec_id: id.to_string(),
            })
            .await;
        Ok(())
    }

    /// Answer a free-form question against a research's accumulated findings.
    ///
    /// Builds a one-shot prompt containing the topic + the latest N findings
    /// (title, price, date, url, excerpt) and calls the configured research
    /// provider/model with no tools. The LLM is asked to answer ONLY from the
    /// supplied corpus and to cite URLs by `[1]`-style numeric indices.
    ///
    /// Used by `/research ask` in the Telegram bot. Returns the raw text the
    /// model produced; callers should surface it as-is.
    pub async fn ask_research(&self, id: &str, question: &str) -> Result<String> {
        use tokio_stream::StreamExt;

        if !self.config.research.enabled {
            return Err(AgentError::Config("research subsystem is disabled".into()));
        }
        let q = question.trim();
        if q.is_empty() {
            return Err(AgentError::Config("question must not be empty".into()));
        }

        let spec = self.research_store.load_spec(id).await?;
        // Cap context: 30 most recent findings keeps us safely under typical
        // 16k token windows even for very long excerpts.
        const MAX_FINDINGS: usize = 30;
        const EXCERPT_BUDGET: usize = 600;
        let findings = self
            .research_store
            .list_findings(id, Some(MAX_FINDINGS))
            .await?;

        if findings.is_empty() {
            return Ok(format!(
                "No findings yet for `{id}` — run `/research run {id}` first."
            ));
        }

        let mut corpus = String::new();
        for (i, f) in findings.iter().rev().enumerate() {
            let title = f.title.as_deref().unwrap_or("(untitled)");
            let price = f.price.as_deref().unwrap_or("?");
            let date = f.listing_date.as_deref().unwrap_or("?");
            let excerpt = f
                .excerpt
                .as_deref()
                .map(|e| {
                    if e.chars().count() > EXCERPT_BUDGET {
                        format!("{}…", e.chars().take(EXCERPT_BUDGET).collect::<String>())
                    } else {
                        e.to_string()
                    }
                })
                .unwrap_or_default();
            corpus.push_str(&format!(
                "[{idx}] {title} — {price} ({date})\n  url: {url}\n  excerpt: {excerpt}\n\n",
                idx = i + 1,
                url = f.url,
            ));
        }

        let provider_name = spec
            .provider
            .clone()
            .or_else(|| self.config.research.provider.clone())
            .unwrap_or_else(|| self.config.default_provider.clone());
        let model = spec
            .model
            .clone()
            .or_else(|| self.config.research.model.clone())
            .unwrap_or_else(|| self.config.default_model.clone());
        let provider = self.provider_for(&provider_name).await;

        let system = "You are a research assistant. Answer the user's question \
                      using ONLY the numbered findings provided. If the corpus \
                      does not contain the answer, say so plainly — do NOT \
                      invent details. When you cite specific findings, refer to \
                      them by their bracketed number, e.g. `[3]`. Keep the \
                      reply concise (under 1500 chars) and in the same language \
                      as the question.";
        let user = format!(
            "Research topic: {topic}\n\n# Findings\n\n{corpus}# Question\n\n{q}",
            topic = spec.topic,
        );

        let request = provider::ChatRequest {
            model,
            system: system.to_string(),
            messages: vec![serde_json::json!({"role": "user", "content": user})],
            tools: vec![],
            max_tokens: 1500,
            temperature: Some(0.2),
            reasoning: None,
        };

        let mut stream = provider
            .stream_chat(request)
            .await
            .map_err(|e| AgentError::Provider(format!("research_ask LLM call failed: {e}")))?;

        let mut text = String::new();
        while let Some(chunk) = stream.next().await {
            match chunk {
                types::StreamChunk::Text(t) => text.push_str(&t),
                types::StreamChunk::Done => break,
                types::StreamChunk::Error(e) => {
                    return Err(AgentError::Provider(format!(
                        "research_ask stream error: {e}"
                    )));
                }
                _ => {}
            }
        }
        if text.trim().is_empty() {
            return Err(AgentError::Provider(
                "research_ask LLM returned empty response".into(),
            ));
        }
        Ok(text)
    }

    /// Apply a partial mutation to a research spec. Only the fields set on
    /// `patch` (non-`None`) are touched; everything else is preserved.
    ///
    /// `interval_seconds` uses a double-`Option` so callers can distinguish
    /// "leave the schedule alone" (`None`) from "clear the schedule"
    /// (`Some(None)`).
    ///
    /// Returns the post-update spec for echoing back to LLM/UI callers.
    pub async fn update_research(&self, id: &str, patch: ResearchPatch) -> Result<ResearchSpec> {
        let mut spec = self.research_store.load_spec(id).await?;
        apply_research_patch(&mut spec, patch);
        self.research_store.save_spec(&spec).await?;
        tracing::info!(
            spec_id = %spec.id,
            topic = %spec.topic,
            interval_seconds = ?spec.interval_seconds,
            sources = spec.sources.len(),
            "research spec updated"
        );
        self.scheduler_hook()
            .notify(research::SchedulerEvent::SpecUpdated {
                spec_id: spec.id.clone(),
            })
            .await;
        Ok(spec)
    }

    /// Execute a single research pass synchronously. Returns the run summary.
    /// Telegram/CLI callers typically `tokio::spawn` this — a live run can
    /// take up to `max_wall_seconds` (20 min default).
    pub async fn run_research(self: Arc<Self>, id: &str) -> Result<RunReport> {
        self.run_research_with_cancel(id, tokio_util::sync::CancellationToken::new())
            .await
    }

    /// Cancellation-aware variant of [`Self::run_research`]. The scheduler
    /// uses this so its two-step `cancel → abort` shutdown can stop a
    /// runaway research at the next coordinator `await` point — without
    /// it, `JoinHandle::abort` lands only at the worker's next yield
    /// point, which can be many seconds away inside an HTTP/LLM stream.
    pub async fn run_research_with_cancel(
        self: Arc<Self>,
        id: &str,
        cancel: tokio_util::sync::CancellationToken,
    ) -> Result<RunReport> {
        if !self.config.research.enabled {
            return Err(AgentError::Config("research subsystem is disabled".into()));
        }
        let _permit = acquire_research_permit(&self.research_run_semaphore, id).await?;
        // Register the cancel token so the TG "Stop & clarify" callback
        // can signal it by spec_id. Cleared on exit — every path below
        // goes through the guard's `drop`.
        let _guard = ResearchCancelGuard::install(
            self.research_cancels.clone(),
            self.research_run_events.clone(),
            id,
            cancel.clone(),
        )
        .await;
        let coord = self.build_coordinator();
        let report = coord.run_once_with_cancel(id, cancel).await?;
        if let Err(e) = self
            .write_research_memory_link(id, &report.run_id, None)
            .await
        {
            tracing::warn!(spec = %id, "failed to record research memory link: {e}");
        }
        Ok(report)
    }

    /// Execute research with gatekeeper verification loop.
    /// After collecting findings, verifies URLs are live and data is complete.
    /// Dead findings are removed and the agent is re-run with feedback to find
    /// replacements. Up to `max_rounds` verification passes.
    pub async fn run_research_verified(
        self: Arc<Self>,
        id: &str,
        max_rounds: u32,
    ) -> Result<research::VerifiedRunReport> {
        self.run_research_verified_with_cancel(
            id,
            max_rounds,
            tokio_util::sync::CancellationToken::new(),
        )
        .await
    }

    /// Cancellation-aware variant of [`Self::run_research_verified`]. See
    /// [`Self::run_research_with_cancel`] for the rationale.
    pub async fn run_research_verified_with_cancel(
        self: Arc<Self>,
        id: &str,
        max_rounds: u32,
        cancel: tokio_util::sync::CancellationToken,
    ) -> Result<research::VerifiedRunReport> {
        if !self.config.research.enabled {
            return Err(AgentError::Config("research subsystem is disabled".into()));
        }
        let _permit = acquire_research_permit(&self.research_run_semaphore, id).await?;
        let _guard = ResearchCancelGuard::install(
            self.research_cancels.clone(),
            self.research_run_events.clone(),
            id,
            cancel.clone(),
        )
        .await;
        let coord = self.build_coordinator();
        let report = coord
            .run_verified_with_cancel(id, max_rounds, cancel)
            .await?;
        if let Err(e) = self
            .write_research_memory_link(id, &report.last_run.run_id, Some(&report))
            .await
        {
            tracing::warn!(spec = %id, "failed to record research memory link: {e}");
        }
        Ok(report)
    }

    /// Append a single global memory entry summarising a finished research run.
    /// Used as a breadcrumb so the LLM can answer "как там наше исследование"
    /// without trawling `runs.jsonl`. The entry's `source` field is set to
    /// `research/<spec_id>` so memory listings group naturally.
    ///
    /// `verified` is `Some(..)` when the run came from `run_verified`, with
    /// gatekeeper round/dead/replacement counters folded into the line.
    /// `report_path` falls back to `report=<unavailable>` for non-fs stores.
    /// Failures are returned to the caller so they can be logged at the
    /// invocation site without the helper logging twice.
    pub async fn write_research_memory_link(
        &self,
        spec_id: &str,
        run_id: &str,
        verified: Option<&VerifiedRunReport>,
    ) -> Result<()> {
        write_research_memory_link_for(
            self.research_store.as_ref(),
            &self.config.workspace,
            spec_id,
            run_id,
            verified,
            None,
        )
        .await
    }

    fn build_coordinator(self: &Arc<Self>) -> ResearchCoordinator {
        let coord_cfg = CoordinatorConfig {
            default_provider: self.config.research.provider.clone(),
            default_model: self.config.research.model.clone(),
            fallback_models: self.config.research.fallback_models.clone(),
            default_max_iterations: self.config.research.max_iterations,
            default_max_wall_seconds: self.config.research.max_wall_seconds,
            workspace: self.config.workspace.clone(),
            gatekeeper: self.config.research.gatekeeper.clone(),
            reasoning: self.config.research.reasoning.clone(),
            provider_capabilities: self.config.providers.clone(),
            enforce_model_capabilities: self.config.enforce_model_capabilities,
            model_health: Some(self.model_health.clone()),
            run_events: Some(self.research_run_events.clone()),
        };
        let runner: Arc<dyn research::AgentRunner> =
            Arc::new(AgentCoreResearchRunner::new(self.clone()));
        ResearchCoordinator::new(self.research_store.clone(), runner, coord_cfg)
    }

    /// Shared waterfall registry. The TG heartbeat task polls this
    /// every ~20 s to render the live-progress message; no other
    /// consumer is expected today but the method is exposed so
    /// future telemetry (metrics, CLI `/research tail`) can tap the
    /// same source of truth.
    pub fn research_run_events(&self) -> research::RunEventRegistry {
        self.research_run_events.clone()
    }

    /// Snapshot the latest `limit` events for `run_id`. Empty when
    /// the run has completed and the registry has been cleaned up,
    /// or when the run_id is unknown.
    pub async fn research_run_events_snapshot(
        &self,
        run_id: &str,
        limit: usize,
    ) -> Vec<research::RunEvent> {
        self.research_run_events.snapshot(run_id, limit).await
    }

    /// Stop a live research run cooperatively. Looks up the
    /// cancellation token installed by `run_research*` and signals
    /// it; the coordinator's `select!` arm picks the signal up at
    /// the next `await` boundary and returns
    /// [`research::StopReason::Cancelled`]. Returns `false` when the
    /// `run_id` is unknown (already finished, or TG callback fired
    /// after cleanup), which the caller typically surfaces as a
    /// benign "run already done".
    pub async fn cancel_research_run(&self, run_id: &str) -> bool {
        if let Some(token) = self.research_cancels.read().await.get(run_id).cloned() {
            token.cancel();
            true
        } else {
            false
        }
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
