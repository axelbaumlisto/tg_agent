// ── Core engine ─────────────────────────────────────────────────────────────
pub mod config;
pub mod error;
pub mod error_taxonomy;
#[path = "history_mod/mod.rs"]
pub mod history;
pub mod hooks;
pub mod loop_;
pub mod loop_guard;
pub mod prompt;
pub mod services;
pub mod session;
mod session_config_ops;
mod session_ops;
pub mod turn;

// ── Agent identity & orchestration ──────────────────────────────────────────
pub mod agent_registry;
pub mod agent_role;
pub mod agent_run;
pub mod agent_store;
pub mod agent_validator;

// ── Tools & skills ──────────────────────────────────────────────────────────
pub mod skill;
pub mod tool;

// ── Providers & models ──────────────────────────────────────────────────────
pub mod model_catalog;
pub mod model_selector;
pub mod provider;
mod provider_ops;

// ── Memory & persistence ────────────────────────────────────────────────────
pub mod memory;
pub mod snapshot;
pub mod working_set;

// ── Research ────────────────────────────────────────────────────────────────
pub mod research;
mod research_ops;

// ── External integrations ───────────────────────────────────────────────────
pub mod keys;
pub mod mcp;
pub mod scrape;
pub mod search;

// ── Infrastructure & utilities ──────────────────────────────────────────────
pub mod active_turns;
pub mod audit;
pub mod auto_reasoning;
pub mod capacity;
pub mod coherence;
pub mod command_arity;
pub mod mentions;
pub mod network_policy;
pub mod retry;
pub mod schema_migration;
pub mod stream_filter;
pub mod token_tracker;
pub mod types;

#[cfg(test)]
mod core_tests {
    use super::*;

    #[test]
    fn lock_or_recover_normal() {
        let m = std::sync::Mutex::new(42);
        let g = lock_or_recover(&m);
        assert_eq!(*g, 42);
    }

    #[test]
    fn write_or_recover_normal() {
        let rw = std::sync::RwLock::new("hello");
        let g = write_or_recover(&rw);
        assert_eq!(*g, "hello");
    }

    #[test]
    fn read_or_recover_normal() {
        let rw = std::sync::RwLock::new(99);
        let g = read_or_recover(&rw);
        assert_eq!(*g, 99);
    }

    #[test]
    fn provider_factory_anthropic_registered() {
        assert!(
            PROVIDER_FACTORIES
                .iter()
                .any(|(name, _)| *name == "anthropic")
        );
    }

    #[test]
    fn provider_factory_copilot_registered() {
        assert!(
            PROVIDER_FACTORIES
                .iter()
                .any(|(name, _)| *name == "copilot")
        );
    }

    #[test]
    fn create_single_provider_anthropic() {
        let cfg = config::ProviderConfig {
            provider_type: "anthropic".into(),
            api_key: "test-key".into(),
            ..Default::default()
        };
        let p = create_single_provider("test", cfg);
        assert!(p.name().contains("test"));
    }

    #[test]
    fn create_single_provider_unknown_falls_back_to_openai() {
        let cfg = config::ProviderConfig {
            provider_type: "unknown_provider".into(),
            api_key: "test-key".into(),
            ..Default::default()
        };
        let p = create_single_provider("test", cfg);
        // OpenAiCompatProvider is the fallback
        assert!(p.name().contains("test"));
    }

    #[test]
    fn shared_tool_state_new() {
        let s = SharedToolState::new();
        // TodoList and PlanState are private but we can verify construction doesn't panic
        let _ = s;
    }
}

#[cfg(test)]
pub mod test_support;

use std::collections::HashMap;
use std::path::Path;
use std::sync::{Arc, Weak};

/// Lock a `std::sync::Mutex`, recovering from poison (panicked holder).
///
/// Replaces the verbose `.lock().unwrap_or_else(|e| e.into_inner())`
/// pattern used throughout the codebase.
pub fn lock_or_recover<T>(mutex: &std::sync::Mutex<T>) -> std::sync::MutexGuard<'_, T> {
    mutex.lock().unwrap_or_else(|e| e.into_inner())
}

/// Write-lock a `std::sync::RwLock`, recovering from poison.
pub fn write_or_recover<T>(lock: &std::sync::RwLock<T>) -> std::sync::RwLockWriteGuard<'_, T> {
    lock.write().unwrap_or_else(|e| e.into_inner())
}

/// Read-lock a `std::sync::RwLock`, recovering from poison.
pub fn read_or_recover<T>(lock: &std::sync::RwLock<T>) -> std::sync::RwLockReadGuard<'_, T> {
    lock.read().unwrap_or_else(|e| e.into_inner())
}

use tokio::sync::RwLock;

use agent_registry::AgentRegistry;
use config::{Config, EffectiveSessionConfig, ResolvedProvider, SessionConfig};
use error::{AgentError, Result};
use mcp::client::{McpRegistry, McpServer};
use provider::Provider;
use provider::anthropic::AnthropicProvider;
use provider::copilot::CopilotProvider;
use provider::openai_compat::OpenAiCompatProvider;
use provider::resilient::ResilientProvider;
use research::{
    CoordinatorConfig, FsResearchStore, ResearchContext, ResearchSpec, ResearchStore,
    new_research_id, parse_provider_model_pair,
};
use session::jsonl_store::JsonlSessionStore;

use types::{AgentHandle, ContentBlock};

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

/// Registry of provider factories — add new provider types here.
///
/// Open/Closed: adding a new provider type = one line in this array,
/// no changes to `create_single_provider`.
type ProviderFactory = fn(String, config::ProviderConfig) -> Box<dyn Provider>;
static PROVIDER_FACTORIES: &[(&str, ProviderFactory)] = &[
    ("anthropic", |name, cfg| {
        Box::new(AnthropicProvider::new(name, cfg))
    }),
    ("copilot", |name, cfg| {
        Box::new(CopilotProvider::new(name, cfg))
    }),
];

/// Create a single Provider instance for one key.
/// Looks up `cfg.provider_type` in [`PROVIDER_FACTORIES`]; falls back
/// to OpenAI-compatible if no match (covers openai, deepseek, kimi, etc.).
fn create_single_provider(name: &str, cfg: config::ProviderConfig) -> Box<dyn Provider> {
    for &(type_name, factory) in PROVIDER_FACTORIES {
        if cfg.provider_type == type_name {
            return factory(name.to_string(), cfg);
        }
    }
    Box::new(OpenAiCompatProvider::new(name.to_string(), cfg))
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
    sem.clone().acquire_owned().await.map_err(|e| {
        AgentError::ProviderTyped(crate::provider::error::ProviderError::Other {
            status: 0,
            body: format!("semaphore closed: {e}"),
        })
    })
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

/// Grouped research subsystem state.
/// Re-export from services.
pub(crate) use services::research::ResearchState;

/// Grouped search/scrape key pools.
/// Re-export from services.
pub(crate) use services::search::SearchState;

/// Grouped catalog/extension state: MCP, agents, skills, tools.
pub(crate) struct CatalogState {
    pub mcp_registry: Arc<RwLock<McpRegistry>>,
    pub session_mcp: RwLock<HashMap<String, Vec<Arc<McpServer>>>>,
    pub agent_registry: AgentRegistry,
    pub agent_store: Arc<agent_store::AgentStore>,
    pub extra_tool_factories: RwLock<ExtraToolFactories>,
    pub remote_ctx: tool::remote::RemoteContext,
    pub hooks: hooks::HookRegistry,
}

/// Grouped state for tools that persist across sessions (todo list, plans).
/// Keeps `AgentCore` from leaking tool-specific types as public fields.
pub struct SharedToolState {
    pub(crate) todo_list: tool::todo_tool::TodoList,
    pub(crate) plan_state: tool::plan_tool::PlanState,
}

impl SharedToolState {
    fn new() -> Self {
        Self {
            todo_list: tool::todo_tool::TodoList::new(),
            plan_state: tool::plan_tool::PlanState::new(),
        }
    }
}

pub struct AgentCore {
    config: arc_swap::ArcSwap<Config>,
    /// Catalog & extensions: MCP, agents, skills, remote, hooks.
    pub(crate) catalog: CatalogState,

    research: ResearchState,
    self_ref: std::sync::RwLock<Option<Weak<AgentCore>>>,

    search: SearchState,
    /// Step 4: Provider service (owns provider, cache, health).
    /// Gradually replacing direct access to `provider`, `provider_cache`, `model_health`.
    pub(crate) provider_svc: Arc<services::ProviderService>,
    /// Session state (owns sessions, cancels, store, senders).
    /// Methods on SessionState replace direct field access.
    pub(crate) ss: Arc<services::session_state::SessionState>,
    /// Per-agent token tracker — shared across all sessions.
    pub token_tracker: token_tracker::TokenTracker,
    /// Shared todo list — persists across turns.
    /// Shared tool state — persists across all sessions and turns.
    pub shared_tools: SharedToolState,
}

impl AgentCore {
    /// Load current config snapshot. Cheap (Arc clone).
    pub fn config(&self) -> arc_swap::Guard<std::sync::Arc<Config>> {
        self.config.load()
    }

    /// Hot-reload config from a new value.
    pub fn reload_config(&self, new_config: Config) {
        self.config.store(std::sync::Arc::new(new_config));
    }

    pub fn new(config: Config, provider: Box<dyn Provider>) -> Self {
        let session_dir = config.session_dir_abs();
        let store = Arc::new(JsonlSessionStore::new(session_dir.clone()));
        let research_root = config
            .research
            .storage_dir
            .clone()
            .unwrap_or_else(research::store::research_root);
        let research_store: Arc<dyn ResearchStore> = Arc::new(FsResearchStore::new(research_root));
        let max_concurrent = config.research.max_concurrent_runs.max(1);

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

        let search_state = crate::services::search::SearchState::from_config(&config.exa_api_keys);

        // Clone for ProviderService before moving into Self.
        let provider_arc: Arc<dyn Provider> = Arc::from(provider);
        let config_arc = Arc::new(config.clone());

        Self {
            config: arc_swap::ArcSwap::from_pointee(config),
            catalog: CatalogState {
                mcp_registry: Arc::new(RwLock::new(McpRegistry::new())),
                session_mcp: RwLock::new(HashMap::new()),
                agent_registry: AgentRegistry::new(),
                agent_store,
                extra_tool_factories: RwLock::new(Vec::new()),
                remote_ctx: tool::remote::RemoteContext::new(),
                hooks: hooks::HookRegistry::new(),
            },
            self_ref: std::sync::RwLock::new(None),
            research: ResearchState::new(research_store, max_concurrent),
            search: search_state,
            provider_svc: Arc::new(services::ProviderService::new(
                provider_arc,
                model_health,
                config_arc,
            )),
            ss: Arc::new(services::session_state::SessionState::new(store)),
            token_tracker: token_tracker::TokenTracker::new(),
            shared_tools: SharedToolState::new(),
        }
    }

    /// B7: Get the remote execution context.
    pub fn remote_context(&self) -> &tool::remote::RemoteContext {
        &self.catalog.remote_ctx
    }

    /// Load session IDs that were mid-turn when the process crashed.
    /// Filters to sessions updated within `recency` to avoid spamming
    /// stale topics that accumulated in `.active_sessions` across restarts.
    pub async fn drain_interrupted_sessions(&self, recency: chrono::Duration) -> Vec<String> {
        let all = self.ss.store.drain_interrupted().await;
        if all.is_empty() {
            return all;
        }
        let cutoff = chrono::Utc::now() - recency;
        let sessions = self.ss.sessions.read().await;
        let mut recent = Vec::new();
        for sid in &all {
            if let Some(s) = sessions.get(sid.as_str()) {
                if s.updated_at >= cutoff {
                    recent.push(sid.clone());
                } else {
                    tracing::debug!(
                        session = %sid,
                        updated = %s.updated_at,
                        "skip stale interrupted session"
                    );
                }
            }
        }
        tracing::info!(
            "{} in .active_sessions, {} recent (within {}s)",
            all.len(),
            recent.len(),
            recency.num_seconds(),
        );
        recent
    }

    /// B6: Get the hook registry.
    pub fn hooks(&self) -> &hooks::HookRegistry {
        &self.catalog.hooks
    }

    /// Shared runtime health tracker. Exposed so telemetry surfaces
    /// (Phase 3 Prometheus exporter, `/model health` CLI) can query
    /// rolling counters without round-tripping through the coordinator.
    pub fn model_health(&self) -> Arc<crate::model_catalog::ModelHealth> {
        self.provider_svc.health()
    }

    /// Shared catalog of disk-loaded agent roles. Use
    /// `naked_core::agent_store::resolve_role(name, &core.agent_store(),
    /// &config.agent_roles)` to pick a role with overrides applied.
    pub fn agent_store(&self) -> Arc<agent_store::AgentStore> {
        self.catalog.agent_store.clone()
    }

    /// Register an extra tool factory. Each factory is called once per
    /// session turn to produce a fresh tool instance. Use this to inject
    /// tools from the embedding binary (e.g. `telegram_attach` from
    /// `naked-tg`) without coupling naked-core to Telegram.
    pub async fn register_extra_tool<F>(&self, factory: F)
    where
        F: Fn() -> Box<dyn tool::Tool> + Send + Sync + 'static,
    {
        self.catalog
            .extra_tool_factories
            .write()
            .await
            .push(Arc::new(factory));
    }

    /// Must be called once after wrapping in `Arc` so orchestration tools can
    /// obtain a reference back to the core (e.g. `research_launch`).
    pub fn init_self_ref(self: &Arc<Self>) {
        *write_or_recover(&self.self_ref) = Some(Arc::downgrade(self));
    }

    /// Expose the underlying provider so out-of-loop consumers (e.g. the
    /// [`crate::agent_validator::GatekeeperValidator`]) can issue one-shot
    /// completions without spinning up a session. Returns the same `Arc`
    /// the agent uses for its own loop, so token budgets and rate limits
    /// stay shared.
    pub fn provider(&self) -> Arc<dyn Provider> {
        self.provider_svc.default_provider()
    }
}

// ---------------------------------------------------------------------------
// Trait impls — compile-time contracts (plan-solid-core-v4, Step 1)
// ---------------------------------------------------------------------------

#[async_trait::async_trait]
impl services::ProviderResolver for AgentCore {
    async fn resolve_provider(&self, name: &str) -> Arc<dyn Provider> {
        self.provider_for(name).await
    }
    fn default_provider_model(&self) -> (String, String) {
        self.default_provider_model()
    }
}

#[async_trait::async_trait]
impl services::SessionLifecycle for AgentCore {
    async fn create_session(&self, workspace: &Path) -> String {
        self.create_session(workspace).await
    }
    async fn send_prompt(&self, session_id: &str, text: &str) -> Result<AgentHandle> {
        self.send_prompt(session_id, text).await
    }
    async fn is_session_active(&self, session_id: &str) -> bool {
        self.is_session_active(session_id).await
    }
}

#[async_trait::async_trait]
impl services::SessionControl for AgentCore {
    async fn abort(&self, session_id: &str) {
        self.abort(session_id).await
    }
    async fn list_sessions(&self) -> Vec<session::SessionSummary> {
        self.list_sessions().await
    }
}

#[async_trait::async_trait]
impl services::SessionDiagnostics for AgentCore {
    async fn session_total_usage(&self, session_id: &str) -> types::TurnUsage {
        self.session_total_usage(session_id).await
    }
    async fn session_provider_model(&self, session_id: &str) -> (String, String) {
        self.session_provider_model(session_id).await
    }
}

impl services::SessionManager for AgentCore {}

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
        self.core.research.context.set_id(Some(spec.id.clone()));
        self.core
            .research
            .context
            .set_run_id(Some(run_id.to_string()));
        self.core.research.context.reset_saves();

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
            .unwrap_or_else(|| self.core.config().default_provider.clone());
        let effective_model = model
            .clone()
            .unwrap_or_else(|| self.core.config().default_model.clone());

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

        self.core.research.context.set_id(None);
        self.core.research.context.set_run_id(None);
        self.core.research.context.reset_saves();
    }
}

// Re-export from research submodules for backward compatibility.
pub use research::patch::{PatchField, ResearchPatch, apply_research_patch};
pub use research::runlog::write_research_memory_link_for;
