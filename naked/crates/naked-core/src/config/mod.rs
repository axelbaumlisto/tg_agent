mod loader;
mod memory;
mod provider;
mod research;
mod validate;
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
}

/// Top-level agent configuration. Loaded from JSON, env vars override.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Config {
    #[serde(default)]
    pub providers: HashMap<String, ProviderConfig>,
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

fn default_enforce_model_capabilities() -> bool {
    true
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
fn default_session_dir() -> PathBuf {
    PathBuf::from(".naked/sessions")
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

impl Config {
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
