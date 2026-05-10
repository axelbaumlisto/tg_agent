use crate::loop_observer::{LoopObserver, TracingObserver};
use crate::provider::Provider;
use crate::tool::registry::ToolRegistry;

// Re-imports made visible to the companion test file via `use super::*`.
// These are referenced from `loop__tests.rs` and must be in this module's
// namespace; they are only consumed in test builds.
#[cfg(test)]
use crate::error::AgentError;
#[cfg(test)]
use crate::history::ConversationHistory;
#[cfg(test)]
use crate::types::{AgentEvent, StreamChunk, TurnUsage};
#[cfg(test)]
use tokio::sync::mpsc;
#[cfg(test)]
use tokio_util::sync::CancellationToken;

// Sub-modules (private impls of AgentLoop are split across files)
mod budget;
mod cycle;
pub mod lsp_hooks;
mod permission;
mod run;
mod steers;
mod stream_turn;
mod tools;

const MAX_STREAM_RETRIES: usize = 3;
const BASE_RETRY_DELAY_MS: u64 = 1000;
const HEARTBEAT_INTERVAL: std::time::Duration = std::time::Duration::from_secs(5);

/// How many times to retry the *same* turn after the provider closes the
/// stream with zero text, zero reasoning and zero tool calls. Some
/// providers — notably `glm-5-turbo` and occasionally Alibaba routes —
/// hiccup like this for a single second under load. A bounded retry lets
/// the agent recover transparently instead of failing the turn and asking
/// the operator to re-issue the prompt.
const MAX_EMPTY_CONTENT_RETRIES: usize = 2;
/// Base back-off for empty-content retries. Kept short (250 ms) because
/// the symptom is a *closed-but-empty* stream, not rate-limit, so we want
/// to come back fast. Doubles per attempt (250 → 500).
const EMPTY_CONTENT_BASE_DELAY_MS: u64 = 250;

pub struct LoopConfig {
    pub max_iterations: usize,
    pub cwd: std::path::PathBuf,
    pub model: String,
    pub max_tokens: u32,
    pub temperature: Option<f32>,
    pub reasoning: Option<String>,
    /// Provider id the `model` lives under (e.g. `"zai"`, `"kimi-code"`).
    /// Required by the Phase 3 [`crate::model_catalog::ModelHealth`]
    /// tracker to key per-pair counters. Default `""` — when empty, the
    /// loop skips health recording (keeps tests with a raw
    /// `LoopConfig::default()` working unchanged).
    pub provider: String,
    /// Optional health tracker handle. When attached, the loop reports
    /// Success / Empty / Error events at the three termination branches
    /// so the selector can quarantine misbehaving pairs across
    /// restarts. `None` disables recording entirely.
    pub health: Option<std::sync::Arc<crate::model_catalog::ModelHealth>>,
    /// Optional token tracker — records (model, in, out) per turn.
    pub token_tracker: Option<crate::token_tracker::TokenTracker>,
    /// Optional audit directory — logs tool calls to JSONL.
    pub audit_dir: Option<std::path::PathBuf>,
    /// Checkpoint-restart cycle configuration.
    pub cycle_config: Option<crate::session::cycle::CycleConfig>,
    /// Session id for cycle archive naming.
    pub session_id: Option<String>,
    /// Data directory for cycle archives (e.g. state/data).
    pub data_dir: Option<std::path::PathBuf>,
    /// Shared working set for file tracking across turns.
    pub working_set: Option<std::sync::Arc<std::sync::Mutex<crate::working_set::WorkingSet>>>,
    /// Observer for business events (DIP). Defaults to [`TracingObserver`].
    pub observer: std::sync::Arc<dyn LoopObserver>,
    /// Archiver for completed cycle messages (DIP). Defaults to [`NoopCycleArchiver`].
    pub cycle_archiver: std::sync::Arc<dyn crate::cycle_archiver::CycleArchiver>,
    /// Optional LSP manager. When attached, the loop calls
    /// `diagnostics_for(workspace, path)` after every successful
    /// edit_file/apply_patch/write_file and pushes the rendered
    /// block as a synthetic system message before the next request.
    /// `None` disables the post-edit hook (zero overhead). Wiring
    /// for T2 of `PLAN_QUALITY_v1.md`.
    pub lsp: Option<std::sync::Arc<crate::lsp::LspManager>>,
    /// Optional lifecycle hook runner. When attached, the loop
    /// fires PreToolUse/PostToolUse/PermissionRequest events to
    /// any user-configured shell-out hooks. T6 of
    /// `PLAN_QUALITY_v1.md`.
    pub lifecycle_hooks: Option<std::sync::Arc<crate::lifecycle_hooks::LifecycleHookRunner>>,
    /// Optional permission ruleset. When attached, the loop's
    /// approval flow consults `Ruleset::evaluate(tool, target)`
    /// before surfacing a UI prompt. T5 of `PLAN_QUALITY_v1.md`.
    pub permissions: Option<std::sync::Arc<tokio::sync::RwLock<crate::permissions::Ruleset>>>,
}

impl Default for LoopConfig {
    fn default() -> Self {
        Self {
            max_iterations: 0,
            cwd: std::env::current_dir().unwrap_or_default(),
            model: String::new(),
            max_tokens: 16384,
            temperature: None,
            reasoning: None,
            provider: String::new(),
            health: None,
            token_tracker: None,
            audit_dir: None,
            cycle_config: None,
            session_id: None,
            data_dir: None,
            working_set: None,
            observer: std::sync::Arc::new(TracingObserver),
            cycle_archiver: std::sync::Arc::new(crate::cycle_archiver::NoopCycleArchiver),
            lsp: None,
            lifecycle_hooks: None,
            permissions: None,
        }
    }
}

impl LoopConfig {
    /// Record one health event iff a tracker is attached and the
    /// provider id is non-empty. Centralised so the three call sites
    /// don't repeat the same guard.
    pub(crate) fn record_health(
        &self,
        kind: crate::model_catalog::HealthEventKind,
        latency_ms: Option<u64>,
        detail: Option<String>,
    ) {
        if self.provider.is_empty() {
            return;
        }
        if let Some(h) = &self.health {
            h.record(&self.provider, &self.model, kind, latency_ms, detail);
        }
    }
}

pub struct AgentLoop {
    provider: Box<dyn Provider>,
    tools: ToolRegistry,
    config: LoopConfig,
    policy: Box<dyn crate::tool::policy::ToolPolicy>,
    approval_cache: crate::tool::approval_cache::ApprovalCache,
}

impl AgentLoop {
    pub fn new(provider: Box<dyn Provider>, tools: ToolRegistry, config: LoopConfig) -> Self {
        let own_source = Self::detect_own_source_dir(&config.cwd);
        Self {
            provider,
            tools,
            config,
            policy: Box::new(crate::tool::policy::default_pipeline(own_source)),
            approval_cache: crate::tool::approval_cache::ApprovalCache::new(),
        }
    }

    /// Detect the bot's own source directory from the workspace path.
    /// If cwd contains a `crates/naked-core/` directory, that's our source tree.
    fn detect_own_source_dir(cwd: &std::path::Path) -> Option<std::path::PathBuf> {
        // Walk up from cwd looking for our Cargo workspace with naked-core
        let mut dir = cwd.to_path_buf();
        for _ in 0..5 {
            if dir.join("crates/naked-core/src").is_dir() {
                return Some(dir.join("crates"));
            }
            if !dir.pop() {
                break;
            }
        }
        None
    }

    /// Create with a custom tool policy.
    pub fn with_policy(
        provider: Box<dyn Provider>,
        tools: ToolRegistry,
        config: LoopConfig,
        policy: Box<dyn crate::tool::policy::ToolPolicy>,
    ) -> Self {
        Self {
            provider,
            tools,
            config,
            policy,
            approval_cache: crate::tool::approval_cache::ApprovalCache::new(),
        }
    }
}

#[cfg(test)]
#[path = "../loop__tests.rs"]
mod tests;
