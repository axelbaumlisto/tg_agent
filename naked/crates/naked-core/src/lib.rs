#![cfg_attr(not(test), warn(clippy::unwrap_used))]

// ── Core engine ─────────────────────────────────────────────────────────────
pub mod config;
pub mod cycle_archiver;
pub mod error;
pub mod error_taxonomy;
#[path = "history_mod/mod.rs"]
pub mod history;
pub mod hooks;
pub mod lifecycle_hooks;
pub mod liveness;
pub mod loop_;
pub mod loop_guard;
pub mod loop_observer;
pub mod lsp;
pub mod prompt;
pub mod services;
pub mod session;
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
pub mod permissions;
pub mod retry;
pub mod schema_migration;
pub mod stream_filter;
pub mod token_tracker;
pub mod types;
pub mod util;

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
use config::{Config, EffectiveSessionConfig};
use error::{AgentError, Result};
use mcp::client::{McpRegistry, McpServer};
use provider::Provider;
// Re-export factory machinery moved to `provider/factory.rs` (T4 of
// PLAN_CORE_HARDENING_v2): keeps `naked_core::create_provider` and
// `naked_core::build_provider_from_config` working for embedders.
pub use provider::factory::{build_provider_from_config, create_provider};
use research::{FsResearchStore, ResearchStore};
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
    config: Arc<arc_swap::ArcSwap<Config>>,
    /// Catalog & extensions: MCP, agents, skills, remote, hooks.
    pub(crate) catalog: CatalogState,

    pub(crate) research: Arc<ResearchState>,
    /// Pure store/registry operations on the research subsystem.
    /// AgentCore retains 1-line delegates for back-compat (T1 of
    /// PLAN_CORE_HARDENING_v2). Coordinator-bound methods (`run_research*`,
    /// `ask_research`) stay on AgentCore because they need `Arc<Self>`.
    pub(crate) research_svc: Arc<services::research::ResearchService>,
    self_ref: std::sync::RwLock<Option<Weak<AgentCore>>>,

    search: SearchState,
    /// Step 4: Provider service (owns provider, cache, health).
    /// Gradually replacing direct access to `provider`, `provider_cache`, `model_health`.
    pub(crate) provider_svc: Arc<services::ProviderService>,
    /// Session configuration helpers (provider/model overrides, reasoning, allow-list).
    pub(crate) session_config: Arc<services::SessionConfigService>,
    /// Session state (owns sessions, cancels, store, senders).
    /// Methods on SessionState replace direct field access.
    pub(crate) ss: Arc<services::session_state::SessionState>,
    /// Per-agent token tracker — shared across all sessions.
    pub token_tracker: token_tracker::TokenTracker,
    /// Shared todo list — persists across turns.
    /// Shared tool state — persists across all sessions and turns.
    pub shared_tools: SharedToolState,
    /// PLAN_QUALITY_v1 wiring: optional LSP manager. naked-tg's
    /// wiring.rs constructs an `LspManager` from `[lsp]` config and
    /// installs it via [`AgentCore::set_lsp`]. Default `None` →
    /// zero overhead.
    pub(crate) lsp: std::sync::RwLock<Option<Arc<crate::lsp::LspManager>>>,
    /// PLAN_QUALITY_v1 wiring: optional lifecycle hook runner. The
    /// bot loads `~/.naked/hooks.json` once at boot and installs
    /// the runner via [`AgentCore::set_lifecycle_hooks`].
    pub(crate) lifecycle_hooks:
        std::sync::RwLock<Option<Arc<crate::lifecycle_hooks::LifecycleHookRunner>>>,
    /// PLAN_QUALITY_v1 wiring: optional permission ruleset. Loaded
    /// from `~/.naked/permissions.json` and shared across the
    /// process.
    pub(crate) permissions:
        std::sync::RwLock<Option<Arc<tokio::sync::RwLock<crate::permissions::Ruleset>>>>,
}

// ---------------------------------------------------------------------------
// AgentCore::new helpers (T10 of PLAN_CORE_HARDENING_v2)
// ---------------------------------------------------------------------------
//
// `new` used to be a 79-LOC inline constructor. Splitting the construction
// into named helpers gives every chunk a single responsibility and makes
// the top-level orchestration readable at a glance.

fn build_research_store(config: &Config) -> Arc<dyn ResearchStore> {
    let root = config
        .research
        .storage_dir
        .clone()
        .unwrap_or_else(research::store::research_root);
    Arc::new(FsResearchStore::new(root))
}

/// Load agent roles from disk, falling back to an empty store.
///
/// A failure here is intentionally non-fatal: the bot would rather
/// start with no roles than refuse to boot. Missing roots are not
/// errors. Logged at `error!` so an operator can spot a bad path
/// in `journalctl`.
fn load_agent_store(agent_dirs: &[std::path::PathBuf]) -> Arc<agent_store::AgentStore> {
    match agent_store::AgentStore::load_dirs(agent_dirs) {
        Ok(s) => Arc::new(s),
        Err(e) => {
            tracing::error!(error = %e, "agent_store: load failed, continuing with empty catalog");
            Arc::new(agent_store::AgentStore::empty())
        }
    }
}

fn build_model_health(config: &Config) -> Arc<crate::model_catalog::ModelHealth> {
    let h = Arc::new(crate::model_catalog::ModelHealth::load(
        config.model_health.clone(),
    ));
    if let Some(observer) = crate::model_catalog::ObservationRecorder::default_location() {
        h.attach_observer(Arc::new(observer));
    }
    h
}

fn build_catalog_state(agent_store: Arc<agent_store::AgentStore>) -> CatalogState {
    CatalogState {
        mcp_registry: Arc::new(RwLock::new(McpRegistry::new())),
        session_mcp: RwLock::new(HashMap::new()),
        agent_registry: AgentRegistry::new(),
        agent_store,
        extra_tool_factories: RwLock::new(Vec::new()),
        remote_ctx: tool::remote::RemoteContext::new(),
        hooks: hooks::HookRegistry::new(),
    }
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
        let store = Arc::new(JsonlSessionStore::new(config.session_dir_abs()));
        let research_store = build_research_store(&config);
        let max_concurrent = config.research.max_concurrent_runs.max(1);
        let agent_store = load_agent_store(&config.agent_dirs);
        let model_health = build_model_health(&config);
        let search_state = crate::services::search::SearchState::from_config(&config.exa_api_keys);

        let provider_arc: Arc<dyn Provider> = Arc::from(provider);
        let config_arc = Arc::new(config.clone());
        let config_swap = Arc::new(arc_swap::ArcSwap::from_pointee(config));
        let provider_svc = Arc::new(services::ProviderService::new(
            provider_arc,
            model_health,
            config_arc,
        ));
        let session_state = Arc::new(services::session_state::SessionState::new(store));
        let session_config = Arc::new(services::SessionConfigService::new(
            session_state.clone(),
            provider_svc.clone(),
        ));
        let research_state = Arc::new(ResearchState::new(research_store, max_concurrent));
        let research_svc = Arc::new(services::research::ResearchService::new(
            research_state.clone(),
            config_swap.clone(),
        ));

        Self {
            config: config_swap,
            catalog: build_catalog_state(agent_store),
            self_ref: std::sync::RwLock::new(None),
            research: research_state,
            research_svc,
            search: search_state,
            provider_svc,
            session_config,
            ss: session_state,
            token_tracker: token_tracker::TokenTracker::new(),
            shared_tools: SharedToolState::new(),
            lsp: std::sync::RwLock::new(None),
            lifecycle_hooks: std::sync::RwLock::new(None),
            permissions: std::sync::RwLock::new(None),
        }
    }

    /// PLAN_QUALITY_v1 setter — install LSP manager.
    pub fn set_lsp(&self, lsp: Arc<crate::lsp::LspManager>) {
        if let Ok(mut g) = self.lsp.write() {
            *g = Some(lsp);
        }
    }

    /// PLAN_QUALITY_v1 setter — install lifecycle hook runner.
    pub fn set_lifecycle_hooks(&self, runner: Arc<crate::lifecycle_hooks::LifecycleHookRunner>) {
        if let Ok(mut g) = self.lifecycle_hooks.write() {
            *g = Some(runner);
        }
    }

    /// PLAN_QUALITY_v1 setter — install permission ruleset.
    pub fn set_permissions(&self, ruleset: Arc<tokio::sync::RwLock<crate::permissions::Ruleset>>) {
        if let Ok(mut g) = self.permissions.write() {
            *g = Some(ruleset);
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
    /// obtain a reference back to the core (e.g. `research_run`).
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

    pub async fn set_session_provider(
        &self,
        session_id: &str,
        provider: Option<&str>,
        model: Option<&str>,
    ) -> Result<()> {
        let cfg_guard = self.config();
        self.session_config
            .set_session_provider(Arc::clone(&cfg_guard), session_id, provider, model)
            .await
    }

    pub async fn set_session_reasoning(&self, session_id: &str, reasoning: &str) -> Result<()> {
        self.session_config
            .set_session_reasoning(session_id, reasoning)
            .await
    }

    pub async fn set_session_yolo(&self, session_id: &str, enabled_at: Option<i64>) -> Result<()> {
        self.session_config
            .set_session_yolo(session_id, enabled_at)
            .await
    }

    pub async fn set_session_allow_list(&self, session_id: &str, tools: &[String]) -> Result<()> {
        self.session_config
            .set_session_allow_list(session_id, tools)
            .await
    }

    pub async fn session_reasoning(&self, session_id: &str) -> Option<String> {
        self.session_config.session_reasoning(session_id).await
    }

    pub async fn set_session_channel_id(&self, session_id: &str, channel_id: &str) {
        self.session_config
            .set_session_channel_id(session_id, channel_id)
            .await
    }

    pub async fn channel_session_mappings(&self) -> Vec<(String, String)> {
        self.session_config.channel_session_mappings().await
    }

    pub async fn session_total_usage(&self, session_id: &str) -> types::TurnUsage {
        self.session_config.session_total_usage(session_id).await
    }

    pub async fn session_file_stats(&self, session_id: &str) -> (Vec<String>, Vec<String>) {
        self.session_config.session_file_stats(session_id).await
    }

    pub async fn session_context_usage(&self, session_id: &str) -> Option<(usize, u32)> {
        self.session_config.session_context_usage(session_id).await
    }

    pub async fn session_provider_model(&self, session_id: &str) -> (String, String) {
        let cfg_guard = self.config();
        self.session_config
            .session_provider_model(Arc::clone(&cfg_guard), session_id)
            .await
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

pub(crate) use services::research_adapter::AgentCoreResearchRunner;

// Re-export from research submodules for backward compatibility.
pub use research::patch::{PatchField, ResearchPatch, apply_research_patch};
pub use research::runlog::write_research_memory_link_for;
