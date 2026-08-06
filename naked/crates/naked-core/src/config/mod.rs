mod loader;
mod memory;
mod provider;
mod research;
mod validate;
pub use loader::filter_dead_keys_from_json;
pub use memory::*;
mod media;
pub use media::*;
mod session;
pub use research::*;
pub use session::*;

use std::collections::HashMap;
use std::path::PathBuf;

use serde::{Deserialize, Serialize};

use crate::error::{AgentError, Result};

/// Telegram-specific configuration. Grouped to keep Config clean.
/// JSON fields remain flat thanks to `#[serde(flatten)]`.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct TelegramConfig {
    #[serde(default)]
    pub telegram_bot_token: Option<String>,
    /// Telegram chat IDs allowed to interact with the bot (empty = deny all)
    #[serde(default)]
    pub allowed_chat_ids: Vec<i64>,
    /// Prefix group messages with the sender's `@username:` for multi-author
    /// chats.
    #[serde(default = "default_sender_attribution")]
    pub tg_sender_attribution: bool,
    /// Debounce window for coalescing client-split plain-text bursts, in ms.
    /// `0` disables text coalescing (safe default; operators opt in).
    #[serde(default = "default_coalesce_text_ms")]
    pub coalesce_text_ms: u64,
    /// PLAN_TG_LONG_ANSWERS_v2 S7 rollout flag. Because TelegramConfig is
    /// flattened into [`Config`], this key must live at the root of
    /// `naked.json`/`state/naked.json`; a nested `.telegram` object is ignored.
    #[serde(default)]
    pub tg_long_answer_fix_enabled: bool,
}

/// Top-level agent configuration. Loaded from JSON, env vars override.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Config {
    #[serde(default)]
    pub providers: HashMap<String, ProviderConfig>,
    /// B105 — retired-provider aliases: `old_name → live_name` or
    /// `old_name → live_name/live_model`.
    ///
    /// Sessions persist their provider/model pin in
    /// `sessions/<id>/config.json`, and that pin OUTLIVES the removal of a
    /// provider from `providers`. Before this existed, such a session
    /// resolved to the default provider via a silent fallback while its
    /// span, `/model` output, and `model_health.jsonl` all kept reporting
    /// the dead pair — e.g. 229 sessions pinned to `qwen` still logged
    /// `provider=qwen model=qwen3.6-plus kind=success` months after the
    /// DashScope account was retired, so health data could not be trusted.
    ///
    /// An alias makes the redirect explicit and self-documenting instead of
    /// requiring a migration pass over session files: the retired name is
    /// rewritten to a live one at merge time, so every downstream consumer
    /// (provider resolution, health records, tracing spans, `/model`) sees
    /// the same truthful pair. Aliases are chased transitively with a small
    /// bound and are ignored when the name still exists in `providers`, so
    /// restoring a provider automatically wins over its alias.
    ///
    /// Model handling: the `provider/model` form pins BOTH, which is the
    /// normal case for a retirement (the old model does not exist upstream
    /// anymore). The bare `provider` form keeps the session's model string,
    /// which only makes sense when the target provider serves that same
    /// model id.
    #[serde(default)]
    pub provider_aliases: HashMap<String, String>,
    #[serde(default)]
    pub default_provider: String,
    #[serde(default)]
    pub default_model: String,
    #[serde(default = "default_workspace")]
    pub workspace: PathBuf,
    #[serde(default = "default_max_iterations")]
    pub max_iterations: usize,
    #[serde(default = "default_max_tokens")]
    pub max_tokens: u32,
    #[serde(default)]
    pub temperature: Option<f32>,
    /// Context window in tokens. Converted to chars (~4 chars/token) for history compaction.
    #[serde(default)]
    pub context_window: Option<u32>,
    #[serde(default = "default_tool_timeout")]
    pub tool_timeout_secs: u64,
    #[serde(default)]
    pub fallback: Vec<String>,
    /// Glob patterns to filter the /model menu (e.g. `["anthropic/*", "openai/gpt-4*"]`).
    /// Empty = show all models. Patterns match `provider/model`.
    #[serde(default)]
    pub model_scope: Vec<String>,
    #[serde(default)]
    pub system_prompt_path: Option<PathBuf>,
    #[serde(default, rename = "mcpServers")]
    pub mcp_servers: HashMap<String, McpServerConfig>,
    #[serde(default)]
    pub skill_roots: Vec<PathBuf>,
    /// Directories searched for agent definitions. Each entry should
    /// contain `<name>/role.json` (+ `prompt.md`, optional
    /// `criteria.txt`) directories, one per agent role. Resolved in
    /// declaration order — earlier entries shadow later ones, so a
    /// project-local `./agents` wins over `~/.naked/agents`.
    ///
    /// When the field is absent or empty in `naked.json`, the loader
    /// installs the same default pair the skills system uses
    /// (`./agents`, `~/.naked/agents`) so a fresh checkout works
    /// without configuration. Tilde expansion is applied at load time.
    #[serde(default)]
    pub agent_dirs: Vec<PathBuf>,
    #[serde(default = "default_session_dir")]
    pub session_dir: PathBuf,
    /// Telegram bot settings (token, allowed chats, sender attribution).
    #[serde(flatten)]
    pub telegram: TelegramConfig,
    /// CYCLE 2 rollout flag for run-id keyed multi-stream control plane.
    /// Default false preserves the current single-run-per-thread behaviour.
    #[serde(default)]
    pub run_registry_multi_stream_enabled: bool,
    /// PLAN_BACKEND_HARDENING_v1 S2b rollout flag for the coarse provider/stream
    /// turn-body backstop. Default false preserves legacy stream behavior exactly.
    #[serde(default)]
    pub turn_deadline_backstop_enabled: bool,
    /// Coarse provider/stream turn-body backstop ceiling in seconds when
    /// `turn_deadline_backstop_enabled` is true. This must stay above full
    /// fallback-chain latency; it is a hang backstop, not an SLA.
    #[serde(default = "default_turn_deadline_secs")]
    pub turn_deadline_secs: u64,
    /// PLAN_FAST_BACKEND_v2 Step B rollout flag for explicit stale-edit preconditions.
    /// Default false preserves legacy edit_file behaviour.
    #[serde(default)]
    pub stale_edit_guard_enabled: bool,
    /// PLAN_FAST_BACKEND_v2 Step C rollout flag for hashline/content-anchored edit mode.
    /// Default false preserves legacy edit_file behaviour.
    #[serde(default)]
    pub hashline_edit_enabled: bool,
    /// PLAN_FAST_BACKEND_v2 Step D rollout flag for bounded read_file/file_snapshot cache.
    /// Default false preserves legacy direct filesystem reads.
    #[serde(default)]
    pub fs_cache_enabled: bool,
    /// PLAN_FAST_BACKEND_v2 Step E rollout flag for session-scoped persistent bash.
    /// Default false preserves legacy per-call bash process spawning.
    #[serde(default)]
    pub persistent_bash_enabled: bool,
    /// PLAN_SNAPSHOTS_DISABLE_FLAG_v1: master switch for per-turn workspace
    /// snapshots (legacy git-stash + side-git SnapshotRepo at
    /// ~/.naked/snapshots/<hash>/.git). Default false = DISABLED — no capture
    /// spawns, no side repo. Set true to restore the pre-turn safety-net snapshot
    /// feeding /undo and the revert_turn tool.
    #[serde(default)]
    pub snapshots_enabled: bool,
    /// Process-wide byte budget for the bounded read_file/file_snapshot cache.
    #[serde(default = "default_fs_cache_max_bytes")]
    pub fs_cache_max_bytes: u64,
    /// PLAN_FAST_BACKEND_v1 Step 2b rollout flag for fff's long-lived fast index.
    /// Default false preserves the legacy per-turn FilePicker path.
    #[serde(default)]
    pub fff_fast_index_enabled: bool,
    /// Max canonical workspaces allowed to hold long-lived fff watchers/indexes.
    #[serde(default = "default_fff_fast_index_max_workspaces")]
    pub fff_fast_index_max_workspaces: usize,
    /// Hard mmap/content cache byte cap per indexed workspace.
    #[serde(default = "default_fff_fast_index_cache_max_bytes")]
    pub fff_fast_index_cache_max_bytes: u64,
    /// Exa.ai API keys for web search (round-robin rotation)
    #[serde(default)]
    pub exa_api_keys: Vec<String>,
    /// Telegram media ingestion (voice / photo / document). Optional — when
    /// absent, media messages are saved to `workspace/artifacts/` and the path
    /// is relayed to the agent without transcription or vision.
    #[serde(default)]
    pub tg_media: TgMediaConfig,
    /// Long-running research subsystem (perfection-plan v5). All fields are
    /// optional; missing sections fall back to the global provider/model.
    #[serde(default)]
    pub research: ResearchConfig,
    /// Lightweight memory enrichment subsystem (daily digest, pre-compaction
    /// flush, scoring-based promotion). All fields are optional; defaults
    /// keep the system working out-of-the-box without touching `naked.json`.
    #[serde(default)]
    pub memory: MemoryConfig,
    /// Per-name overrides for built-in agent roles
    /// ([`crate::agent_role::builtin_roles`]). Keyed by role name. Each
    /// entry can override `model`, `max_iters`, and `default_skills`
    /// without recompiling. Unknown role names are silently ignored
    /// (forward-compat — pinning a config for a role that ships in a
    /// later release should not crash older binaries).
    #[serde(default)]
    pub agent_roles: HashMap<String, crate::agent_role::AgentRoleOverride>,
    /// Default reasoning/thinking level applied to every new session that
    /// doesn't override it via `/reasoning` or `SessionConfig.reasoning`.
    /// Accepts `"off"`, `"low"`, `"medium"`, `"high"`. `None` ⇒ providers
    /// receive no reasoning hint (legacy behaviour).
    ///
    /// Plumbed into the OpenAI-compatible `reasoning_effort` parameter for
    /// providers on the allowlist (`kimi.com`, `moonshot.cn/.ai`,
    /// `openai.com`, `fireworks`, `openrouter`, `api.deepseek.com`) and
    /// into Anthropic-style extended-thinking budgets for Anthropic
    /// providers; Dashscope/Aliyun (qwen) gets `enable_thinking: true`.
    /// See [`crate::provider::openai_compat::apply_reasoning_params`] and
    /// [`crate::provider::anthropic`] for the per-host gating rules.
    ///
    /// `research.reasoning` takes precedence inside research turns; this
    /// field controls everything else (interactive chat, sub-agents, etc.).
    #[serde(default)]
    pub default_reasoning: Option<String>,
    /// Per-Telegram-chat persona overrides. Keyed by `chat_id` (negative for
    /// groups/supergroups, positive for private DMs). Each persona swaps the
    /// session's `workspace` to a dedicated directory so the system prompt,
    /// project memory (`MEMORY.md` / drafts / `DREAMS.md`), per-project
    /// `CLAUDE.md`/`AGENTS.md` walk, and the default `bash` cwd all become
    /// physically isolated from other chats — without spawning a separate
    /// bot process. Unmapped chats keep using `Config.workspace` and the
    /// global system prompt as before.
    ///
    /// See [`ChatPersona`] for the available knobs and
    /// `crates/naked-tg/src/main.rs::get_or_create_session` for how the
    /// override is applied at session creation time.
    #[serde(default)]
    pub chat_personas: HashMap<i64, ChatPersona>,
    /// Hard-enforce the [`crate::model_catalog`] capability filter at every
    /// model selection site (research fallback chain, chat session
    /// creation, digest, classify). When `true` (current default), the
    /// selector filters out mismatches before dispatch — quarantined
    /// pairs, deprecated models, and task-fit violations never reach
    /// the provider. Unknown (provider, model) pairs stay permissive so
    /// newly-added models keep booting without a catalog entry.
    ///
    /// Legacy configs that prefer the warn-only behaviour can opt out
    /// with `{ "enforce_model_capabilities": false }` in `naked.json`;
    /// [`Config::validate_and_warn`] still emits the same `warn!` lines
    /// regardless of this flag.
    #[serde(default = "default_enforce_model_capabilities")]
    pub enforce_model_capabilities: bool,

    /// Runtime health tracker policy (Phase 3). Quarantine thresholds,
    /// rolling window horizon, and the log path. All fields optional
    /// with sensible defaults — missing `naked.json` blocks still get
    /// the default `ModelHealthConfig`.
    #[serde(default)]
    pub model_health: crate::model_catalog::ModelHealthConfig,
}

/// Per-chat overrides that turn one Telegram chat into a self-contained
/// "persona" inside the same bot process. The minimum-viable set right now:
///
/// - dedicated `workspace` (drives prompt resolution, project memory path,
///   instruction-file walk, and tool cwd defaults);
/// - opt-out for slash-commands (so the chat behaves like a pure
///   natural-language conversation — feature toggles happen by asking
///   the agent in plain language, not by typing `/yolo`/`/model`/etc.).
///
/// Deliberately tiny on purpose — extra knobs (per-chat model, skill_roots,
/// MCP servers) are easy to add later behind concrete use cases per the
/// "no speculative config" rule from `AGENTS.md`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatPersona {
    /// Short slug used in logs and as a memory namespace hint. Should be
    /// stable (changing it relocates the persona's memory directory).
    pub name: String,
    /// Workspace directory passed to [`crate::AgentCore::create_session_with_channel`]
    /// for any session bound to this chat. The runtime expands a leading
    /// `~/` to `$HOME` and creates the directory at startup (idempotent).
    /// All persona-scoped state lives under this path: the system prompt is
    /// resolved as `<workspace>/.naked/system_prompt.md`, project memory
    /// goes to `~/.naked/projects/<workspace-slug>/memory/`, etc.
    pub workspace: PathBuf,
    /// When `false` (default), Telegram messages starting with `/` are
    /// silently ignored in this chat — the bot replies once per chat with
    /// a hint that the persona is natural-language only and then drops
    /// further slash messages without dispatch. Set to `true` to keep the
    /// usual command surface (`/new`, `/yolo`, `/model`, …).
    #[serde(default)]
    pub allow_slash_commands: bool,
}

impl ChatPersona {
    /// Resolve `~/` in the workspace path against the current `$HOME`.
    pub fn workspace_expanded(&self) -> PathBuf {
        expand_tilde(&self.workspace)
    }
}

fn default_sender_attribution() -> bool {
    true
}

fn default_coalesce_text_ms() -> u64 {
    0
}

fn default_enforce_model_capabilities() -> bool {
    true
}

fn default_fff_fast_index_max_workspaces() -> usize {
    4
}

fn default_fff_fast_index_cache_max_bytes() -> u64 {
    256 * 1024 * 1024
}

pub use provider::{ProviderConfig, ResolvedProvider};

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct McpServerConfig {
    #[serde(default)]
    pub name: String,
    #[serde(default)]
    pub transport: McpTransportType,
    #[serde(default)]
    pub command: String,
    #[serde(default)]
    pub args: Vec<String>,
    #[serde(default)]
    pub env: HashMap<String, String>,
    #[serde(default)]
    pub url: Option<String>,
    #[serde(default)]
    pub headers: HashMap<String, String>,
    #[serde(default)]
    pub tool_timeout_secs: Option<u64>,
}

#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum McpTransportType {
    #[default]
    Stdio,
    Http,
    Sse,
}

fn default_workspace() -> PathBuf {
    std::env::current_dir().unwrap_or_else(|_| PathBuf::from("."))
}
fn default_max_iterations() -> usize {
    0
}
fn default_max_tokens() -> u32 {
    16384
}
fn default_tool_timeout() -> u64 {
    120
}
fn default_turn_deadline_secs() -> u64 {
    900
}
fn default_session_dir() -> PathBuf {
    PathBuf::from(".naked/sessions")
}
fn default_fs_cache_max_bytes() -> u64 {
    crate::tool::fs_cache::DEFAULT_FS_CACHE_MAX_BYTES
}
pub(crate) fn dirs_home() -> PathBuf {
    std::env::var("HOME")
        .map(PathBuf::from)
        .unwrap_or_else(|_| PathBuf::from("."))
}

impl Default for Config {
    fn default() -> Self {
        Self {
            providers: HashMap::new(),
            provider_aliases: HashMap::new(),
            default_provider: String::new(),
            default_model: String::new(),
            workspace: default_workspace(),
            max_iterations: default_max_iterations(),
            max_tokens: default_max_tokens(),
            temperature: None,
            context_window: None,
            tool_timeout_secs: default_tool_timeout(),
            fallback: Vec::new(),
            system_prompt_path: None,
            mcp_servers: HashMap::new(),
            skill_roots: Vec::new(),
            agent_dirs: Vec::new(),
            session_dir: default_session_dir(),
            telegram: TelegramConfig::default(),
            run_registry_multi_stream_enabled: false,
            turn_deadline_backstop_enabled: false,
            turn_deadline_secs: default_turn_deadline_secs(),
            stale_edit_guard_enabled: false,
            hashline_edit_enabled: false,
            fs_cache_enabled: false,
            persistent_bash_enabled: false,
            snapshots_enabled: false,
            fs_cache_max_bytes: default_fs_cache_max_bytes(),
            fff_fast_index_enabled: false,
            fff_fast_index_max_workspaces: default_fff_fast_index_max_workspaces(),
            fff_fast_index_cache_max_bytes: default_fff_fast_index_cache_max_bytes(),
            exa_api_keys: Vec::new(),
            tg_media: TgMediaConfig::default(),
            research: ResearchConfig::default(),
            memory: MemoryConfig::default(),
            agent_roles: HashMap::new(),
            default_reasoning: None,
            chat_personas: HashMap::new(),
            enforce_model_capabilities: default_enforce_model_capabilities(),
            model_health: crate::model_catalog::ModelHealthConfig::default(),
            model_scope: Vec::new(),
        }
    }
}

/// Split one `provider_aliases` value into `(provider, optional model)`.
///
/// Accepts `"live"` (redirect only) and `"live/model"` (pin both). Returns
/// `None` for shapes that name no provider (`""`, `"/model"`), so the caller
/// stops instead of chasing an empty name — a malformed alias must be a
/// bounded no-op, never a hang or a jump to a blank provider.
fn parse_alias_target(target: &str) -> Option<(String, Option<String>)> {
    match target.split_once('/') {
        Some((p, m)) if !p.is_empty() && !m.is_empty() => {
            Some((p.to_string(), Some(m.to_string())))
        }
        Some((p, _)) if !p.is_empty() => Some((p.to_string(), None)),
        Some(_) => None,
        None if !target.is_empty() => Some((target.to_string(), None)),
        None => None,
    }
}

/// Max alias hops chased by [`Config::resolve_provider_alias`] before the
/// chain is declared broken. Small on purpose: legitimate retirement chains
/// are 1-2 hops (`ali_cp -> qwen -> anthropic`), and a low bound turns a
/// mis-edited config into a bounded no-op instead of a hang.
const MAX_PROVIDER_ALIAS_HOPS: usize = 8;

impl Config {
    /// B105 — resolve a possibly-retired provider name (and optional model)
    /// through `provider_aliases`.
    ///
    /// Returns `(provider, model_override)`. `model_override` is `Some` only
    /// when an alias pinned a model via the `live_provider/live_model` form;
    /// callers keep the session's own model when it is `None`.
    ///
    /// Rules, in order:
    /// 1. A name that still exists in `providers` is returned untouched —
    ///    restoring a retired provider silently disables its alias, so
    ///    reviving one is a pure config edit with no code change.
    /// 2. Otherwise the alias chain is followed until it lands on a live
    ///    provider, runs out, or trips the hop/cycle guard.
    /// 3. A chain that cannot reach a live provider returns the LAST name it
    ///    reached rather than the original, so the WARN downstream names the
    ///    end of the broken chain. A cycle logs once and stops.
    ///
    /// The first pinned model wins: in `a -> b/m1` then `b -> c/m2` the
    /// result is `(c, Some(m1))`, because the alias closest to what the user
    /// actually pinned is the most specific intent.
    pub fn resolve_provider_alias(&self, name: &str) -> (String, Option<String>) {
        if name.is_empty() || self.providers.contains_key(name) {
            return (name.to_string(), None);
        }

        let mut current = name.to_string();
        let mut model_override: Option<String> = None;
        let mut seen: Vec<String> = vec![current.clone()];

        for _ in 0..MAX_PROVIDER_ALIAS_HOPS {
            let Some(target) = self.provider_aliases.get(&current) else {
                break;
            };
            let Some((next_provider, next_model)) = parse_alias_target(target) else {
                break;
            };
            // First pinned model wins — closest to the user's own pin.
            if model_override.is_none() {
                model_override = next_model;
            }
            if seen.iter().any(|s| s == &next_provider) {
                tracing::warn!(
                    "provider_aliases cycle detected: {} -> {next_provider}; stopping",
                    seen.join(" -> ")
                );
                return (current, model_override);
            }
            current = next_provider.clone();
            seen.push(next_provider);
            if self.providers.contains_key(&current) {
                return (current, model_override);
            }
        }

        (current, model_override)
    }

    /// Load config: JSON file -> env var overrides.
    ///
    /// Search order for JSON:
    fn resolve_mcp_names(&mut self) {
        for (name, server) in &mut self.mcp_servers {
            if server.name.is_empty() {
                server.name = name.clone();
            }
        }
    }

    /// Get provider config by name.
    pub fn provider_config(&self, name: &str) -> Option<&ProviderConfig> {
        self.providers.get(name)
    }

    /// Resolve the default provider, returning its resolved config.
    pub fn resolve_default_provider(&self) -> Result<(String, ResolvedProvider)> {
        let name = if self.default_provider.is_empty() {
            self.providers
                .keys()
                .next()
                .ok_or_else(|| AgentError::ProviderNotConfigured("(default)".to_string()))?
                .clone()
        } else {
            self.default_provider.clone()
        };

        let pc = self
            .providers
            .get(&name)
            .ok_or_else(|| AgentError::ProviderNotConfigured(name.to_string()))?;

        Ok((name, pc.resolved()?))
    }

    /// Parse fallback list: "provider_name/model_id" entries.
    pub fn fallback_providers(&self) -> Vec<(String, String)> {
        self.fallback
            .iter()
            .filter_map(|entry| {
                let (p, m) = entry.split_once('/')?;
                Some((p.to_string(), m.to_string()))
            })
            .collect()
    }

    /// MCP servers as a flat Vec (for McpRegistry).
    pub fn mcp_server_list(&self) -> Vec<McpServerConfig> {
        self.mcp_servers.values().cloned().collect()
    }

    /// Effective max_tokens: per-provider override > global config.
    pub fn effective_max_tokens(&self, provider_name: &str) -> u32 {
        self.providers
            .get(provider_name)
            .and_then(|pc| pc.max_tokens)
            .unwrap_or(self.max_tokens)
    }

    /// Effective temperature: per-provider override > global config.
    pub fn effective_temperature(&self, provider_name: &str) -> Option<f32> {
        self.providers
            .get(provider_name)
            .and_then(|pc| pc.temperature)
            .or(self.temperature)
    }

    pub fn session_dir_abs(&self) -> PathBuf {
        if self.session_dir.is_absolute() {
            self.session_dir.clone()
        } else {
            self.workspace.join(&self.session_dir)
        }
    }
}

#[cfg(test)]
#[path = "tests.rs"]
mod tests;
