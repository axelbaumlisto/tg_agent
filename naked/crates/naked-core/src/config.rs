use std::collections::HashMap;
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

use crate::error::{AgentError, Result};

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
    #[serde(default)]
    pub telegram_bot_token: Option<String>,
    /// Telegram chat IDs allowed to interact with the bot (empty = deny all)
    #[serde(default)]
    pub allowed_chat_ids: Vec<i64>,
    /// Prefix group messages with the sender's `@username:` for multi-author
    /// chats. Ignored in private (1-on-1) chats where there is no ambiguity.
    #[serde(default = "default_sender_attribution")]
    pub tg_sender_attribution: bool,
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

/// Runtime settings for Telegram media processing. All sub-sections are
/// optional — a missing vision/audio config degrades gracefully to "path only".
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TgMediaConfig {
    /// Transcribe incoming voice / audio via an OpenAI-compatible Whisper API
    /// (Groq `whisper-large-v3` is the recommended default).
    #[serde(default)]
    pub audio: Option<AudioProviderCfg>,
    /// Caption / describe incoming photos via an OpenAI-compatible chat API
    /// that supports `image_url` content (xAI `grok-2-vision`, OpenAI `gpt-4o-mini`).
    #[serde(default)]
    pub vision: Option<VisionProviderCfg>,
    /// Hard size caps per media kind (applied after Telegram's 20 MB limit).
    #[serde(default)]
    pub limits: MediaLimits,
    /// Days to keep files under `workspace/.naked/artifacts/`; older dirs are
    /// swept at bot startup. `0` disables sweeping.
    #[serde(default = "default_artifact_retention_days")]
    pub artifact_retention_days: u64,
    /// Maximum document size to inline into the prompt (bytes). Larger docs
    /// are saved to disk and only the path is relayed.
    #[serde(default = "default_docs_inline_max")]
    pub docs_inline_max_bytes: u64,
    /// When true, photos/stickers are passed to the **main** model as native
    /// image content blocks (Anthropic `image.source.base64`, OpenAI/Groq/xAI
    /// `image_url.url=data:`) IF the active model is known to support vision.
    /// When false (or the model is text-only), falls back to running the
    /// `vision` describer above and inlining the **text** description into the
    /// prompt — the legacy behaviour. Default: `true`.
    #[serde(default = "default_native_image_context")]
    pub native_image_context: bool,
    /// Hard cap on bytes per image when `native_image_context` is on. Larger
    /// photos are described instead of being uploaded as base64 (avoids
    /// blowing up context windows). Default: 4 MB (matches Groq's hard cap).
    #[serde(default = "default_native_image_max_bytes")]
    pub native_image_max_bytes: u64,
    /// Optional substring matchers added on top of the built-in vision-model
    /// allowlist (`is_vision_capable_model`). Useful for new model IDs we
    /// haven't hard-coded yet — e.g. `["llama-4", "qwen-vl"]`.
    #[serde(default)]
    pub vision_model_extras: Vec<String>,
    /// Per-model vision capability override map. **Key is a substring** matched
    /// case-insensitively against the model id; value forces vision on (`true`)
    /// or off (`false`) regardless of needles or `vision_model_extras`. Useful
    /// for OpenRouter-style model ids where one provider hosts both
    /// vision-capable and text-only variants under similar names. Resolution
    /// order: this map → provider override → built-in needles → extras.
    #[serde(default)]
    pub model_vision_overrides: HashMap<String, bool>,
    /// Default OpenAI-style `image_url.detail` knob. Accepted values:
    /// `"auto"` (default — provider decides), `"low"` (~85 tokens, ~512×512
    /// downsample), `"high"` (tile-grid, full resolution). Anthropic, Gemini,
    /// xAI ignore the field. Set this to `"low"` to slash token spend on noisy
    /// chat photos; set to `"high"` for screenshots / OCR scenarios where
    /// detail matters more than cost.
    #[serde(default)]
    pub image_detail: Option<crate::types::ImageDetail>,
}

impl Default for TgMediaConfig {
    fn default() -> Self {
        Self {
            audio: None,
            vision: None,
            limits: MediaLimits::default(),
            artifact_retention_days: default_artifact_retention_days(),
            docs_inline_max_bytes: default_docs_inline_max(),
            native_image_context: default_native_image_context(),
            native_image_max_bytes: default_native_image_max_bytes(),
            vision_model_extras: Vec::new(),
            model_vision_overrides: HashMap::new(),
            image_detail: None,
        }
    }
}

impl TgMediaConfig {
    /// True if `model` is known to accept inline images. Combines a built-in
    /// allowlist (Anthropic Claude 3+, GPT-4o family, Groq Llama-4 + Maverick,
    /// xAI grok-2-vision, Gemini, Qwen-VL) with `vision_model_extras` from
    /// config. Matching is case-insensitive substring.
    pub fn is_vision_capable_model(&self, model: &str) -> bool {
        self.is_vision_capable_with_provider(model, None)
    }

    /// Check vision capability with optional provider-level override.
    ///
    /// Resolution order (first hit wins):
    /// 1. `tg_media.model_vision_overrides[<substring>]` — most specific.
    /// 2. `provider.supports_vision = Some(true|false)` — provider-wide override.
    /// 3. `BUILTIN_VISION_MODEL_NEEDLES` substring match.
    /// 4. `tg_media.vision_model_extras` user-supplied substring match.
    pub fn is_vision_capable_with_provider(
        &self,
        model: &str,
        provider: Option<&ProviderConfig>,
    ) -> bool {
        let m = model.to_ascii_lowercase();
        for (key, &force) in &self.model_vision_overrides {
            if !key.is_empty() && m.contains(&key.to_ascii_lowercase()) {
                return force;
            }
        }
        if let Some(pc) = provider
            && let Some(force) = pc.supports_vision
        {
            return force;
        }
        for needle in BUILTIN_VISION_MODEL_NEEDLES {
            if m.contains(needle) {
                return true;
            }
        }
        for extra in &self.vision_model_extras {
            if !extra.is_empty() && m.contains(&extra.to_ascii_lowercase()) {
                return true;
            }
        }
        false
    }

    /// Per-provider hard cap (decoded bytes) for inline image payloads. The
    /// global `native_image_max_bytes` is the floor; provider-specific limits
    /// further trim it down where the upstream API enforces a stricter ceiling.
    /// Conservative values from public API docs (2026-04):
    /// * Anthropic — 5 MB per image, max 100 per request.
    /// * OpenAI — 20 MB per image (gpt-4o family).
    /// * Groq — 4 MB per image (llama-4 vision).
    /// * xAI Grok — 10 MB per image.
    /// * Gemini — 7 MB inline; larger via the File API.
    /// * Anything unknown — fall back to `native_image_max_bytes`.
    pub fn provider_image_cap(&self, provider_type: &str, base_url: Option<&str>) -> u64 {
        let global = self.native_image_max_bytes;
        let cap_for = |bytes: u64| global.min(bytes);
        let url_lc = base_url.unwrap_or("").to_ascii_lowercase();
        match provider_type {
            "anthropic" => cap_for(5 * 1024 * 1024),
            "copilot" => cap_for(5 * 1024 * 1024), // routes through Anthropic/OpenAI; pick stricter floor.
            _ if url_lc.contains("groq.com") => cap_for(4 * 1024 * 1024),
            _ if url_lc.contains("api.openai.com") => cap_for(20 * 1024 * 1024),
            _ if url_lc.contains("api.x.ai") => cap_for(10 * 1024 * 1024),
            _ if url_lc.contains("googleapis.com") || url_lc.contains("generativelanguage") => {
                cap_for(7 * 1024 * 1024)
            }
            _ if url_lc.contains("openrouter.ai") => cap_for(20 * 1024 * 1024),
            _ => global,
        }
    }
}

/// Hard-coded substrings of model IDs that accept vision input today.
/// Conservative — when in doubt, leave out and let users add to
/// `tg_media.vision_model_extras`.
const BUILTIN_VISION_MODEL_NEEDLES: &[&str] = &[
    // Anthropic — every Claude 3+ model is multimodal.
    "claude-3",
    "claude-sonnet-4",
    "claude-opus-4",
    "claude-haiku-4",
    "claude-4",
    // OpenAI
    "gpt-4o",
    "gpt-4-turbo",
    "gpt-4-vision",
    "gpt-5",
    "o1",
    "o3",
    "o4",
    // Groq multimodal
    "llama-4-scout",
    "llama-4-maverick",
    "llama-3.2-11b-vision",
    "llama-3.2-90b-vision",
    // xAI
    "grok-2-vision",
    "grok-3-vision",
    "grok-4",
    // Google
    "gemini-1.5",
    "gemini-2",
    "gemini-pro-vision",
    // Qwen / others
    "qwen-vl",
    "qwen2-vl",
    "qwen2.5-vl",
    "pixtral",
    "minicpm-v",
];

/// OpenAI-compatible Whisper endpoint for audio transcription.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AudioProviderCfg {
    /// e.g. `https://api.groq.com/openai/v1/audio/transcriptions`
    pub api_url: String,
    /// Direct key, or `$ENV` reference.
    pub api_key: String,
    /// e.g. `whisper-large-v3`
    pub model: String,
    /// Optional ISO-639-1 language hint (`ru`, `en`, ...). `None` → auto.
    #[serde(default)]
    pub language: Option<String>,
}

/// Vision describer. Uses OpenAI-compatible chat-completions with
/// `image_url: data:...;base64,...` content parts.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VisionProviderCfg {
    /// e.g. `https://api.x.ai/v1/chat/completions`
    pub api_url: String,
    pub api_key: String,
    /// e.g. `grok-2-vision`, `gpt-4o-mini`
    pub model: String,
    /// Max tokens for the description. Default 400 is enough for ~200 words.
    #[serde(default = "default_vision_max_tokens")]
    pub max_tokens: u32,
    /// Override the default describer prompt. Placeholders: `{caption}`.
    #[serde(default)]
    pub prompt_override: Option<String>,
}

/// Strict per-kind byte caps. Defaults: photo 5 MB, audio 20 MB, doc 10 MB.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MediaLimits {
    #[serde(default = "default_photo_max_bytes")]
    pub photo_max_bytes: u64,
    #[serde(default = "default_audio_max_bytes")]
    pub audio_max_bytes: u64,
    #[serde(default = "default_doc_max_bytes")]
    pub doc_max_bytes: u64,
}

impl Default for MediaLimits {
    fn default() -> Self {
        Self {
            photo_max_bytes: default_photo_max_bytes(),
            audio_max_bytes: default_audio_max_bytes(),
            doc_max_bytes: default_doc_max_bytes(),
        }
    }
}

fn default_photo_max_bytes() -> u64 {
    5 * 1024 * 1024
}
fn default_audio_max_bytes() -> u64 {
    20 * 1024 * 1024
}
fn default_doc_max_bytes() -> u64 {
    10 * 1024 * 1024
}
fn default_artifact_retention_days() -> u64 {
    7
}
fn default_docs_inline_max() -> u64 {
    128 * 1024
}
fn default_vision_max_tokens() -> u32 {
    400
}
fn default_native_image_context() -> bool {
    true
}
fn default_native_image_max_bytes() -> u64 {
    4 * 1024 * 1024
}

/// Configuration for the research subsystem. Defaults are safe but conservative:
/// 30 navigations per run, 20-minute wall-clock cap. Uses the same model family
/// as the main assistant (Qwen3.6-Plus by default) for best tool-use quality;
/// override with a cheaper model via `research.model` if needed.
///
/// Every field is optional. When the whole section is absent from JSON, the
/// default is "enabled, global provider/model, 30 navs, 1200 s". Unknown
/// `provider` or a `model` not listed under its provider surfaces as a
/// `tracing::warn!` via `Config::validate_and_warn` — not a hard error, so
/// existing configs keep booting unchanged.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ResearchConfig {
    /// When `false`, `/research *` commands return "research disabled" and
    /// the background runner is never spawned. Default: `true`.
    #[serde(default = "default_research_enabled")]
    pub enabled: bool,
    /// Override provider for research turns. `None` → `default_provider`.
    #[serde(default)]
    pub provider: Option<String>,
    /// Override model for research turns. `None` → `default_model`.
    #[serde(default)]
    pub model: Option<String>,
    /// Fallback models tried in order when the primary model is rejected by
    /// all provider keys (HTTP 400 "model not supported"). Empty = no fallback.
    #[serde(default)]
    pub fallback_models: Vec<String>,
    /// Max agent iterations per run. Each iteration = one LLM ↔ tool loop.
    #[serde(default = "default_research_max_iterations")]
    pub max_iterations: u32,
    /// Wall-clock budget per run, in seconds. Prevents runaway runs when a
    /// site hangs or a tool loops. Default: 1200 (20 min).
    #[serde(default = "default_research_max_wall_seconds")]
    pub max_wall_seconds: u64,
    /// Optional seed sources to pre-populate newly created specs. E.g.
    /// `["https://www.chotot.com/mua-ban-oto"]` for VN car research.
    #[serde(default)]
    pub default_sources: Vec<String>,
    /// Storage root. `None` → `$NAKED_HOME/research` → `~/.naked/research`.
    #[serde(default)]
    pub storage_dir: Option<PathBuf>,
    /// Tools the research agent is allowed to use during a run. `None` →
    /// registry default. Useful to force a tight loop that only uses
    /// `web_fetch` + `web_search` + `research_save`, forbidding shell.
    #[serde(default)]
    pub allowed_tools: Option<Vec<String>>,
    /// Allow `/research schedule` to manage systemd timers. Default: `true`.
    #[serde(default = "default_true")]
    pub schedule_enabled: bool,
    /// Default interval for scheduled runs when the spec doesn't override.
    /// Only used when creating a new timer. Default: 21600 (6 hours).
    #[serde(default = "default_schedule_interval")]
    pub default_interval_seconds: u64,
    /// Post a summary of new findings to the spec's `chat_id` after a
    /// scheduled run if `new_findings > 0`. Default: `true`.
    #[serde(default = "default_true")]
    pub notify_on_new_findings: bool,
    /// Run gatekeeper verification automatically for runs launched from
    /// Telegram (`/research run`) and the `research_launch` agent tool.
    /// Uses `gatekeeper.max_rounds` for the round budget.
    /// CLI `naked research run --verify` is unaffected (always explicit).
    /// Default: `true`.
    #[serde(default = "default_true")]
    pub verify_by_default: bool,
    /// Reasoning/thinking level applied to every research session via
    /// [`AgentCore::set_session_reasoning`]. Accepts `"off"`, `"low"`,
    /// `"medium"`, `"high"`. `None` ⇒ inherit from session/global default.
    ///
    /// Plumbed into the OpenAI-compatible `reasoning_effort` parameter for
    /// providers that accept it (kimi.com / moonshot.* / openai.com /
    /// fireworks / openrouter / deepseek) and into Anthropic-style
    /// extended-thinking budgets for Anthropic providers. See
    /// [`crate::provider::openai_compat::apply_reasoning_params`] and
    /// [`crate::provider::anthropic`] for the per-host gating rules.
    ///
    /// Recommended for `kimi-for-coding`: `"medium"` — same default as the
    /// official Roo Code integration guide.
    #[serde(default)]
    pub reasoning: Option<String>,
    /// Gatekeeper verification rules. Controls quality checks applied after
    /// each research run when using `--verify` / `run_verified()`.
    #[serde(default)]
    pub gatekeeper: GatekeeperConfig,
    /// Maximum number of research runs that may execute concurrently in this
    /// process. Applies to every entry point — manual `/research run`, the
    /// `research_launch` LLM tool, and the in-process scheduler. Default
    /// `5`: the scheduler subsystem treats this as the parallelism budget
    /// for at-time / cron / interval triggered runs. Set to `1` if your
    /// runs share a singleton (e.g. a single Playwright browser).
    #[serde(default = "default_max_concurrent_runs")]
    pub max_concurrent_runs: usize,
    /// Wall-clock cap for a single scheduler-launched run (seconds). When
    /// the cap is hit the scheduler cancels the run, increments the failure
    /// counter for that spec, and on the next tick may dispatch it again
    /// (subject to `max_retries_before_alert`). Per-spec override:
    /// `ResearchSpec.task_timeout_seconds`.
    #[serde(default = "default_task_timeout_seconds")]
    pub task_timeout_seconds: u64,
    /// Number of consecutive run failures (errors *or* timeouts) tolerated
    /// before the scheduler posts an alert into the spec's chat and stops
    /// retrying until reset. `0` disables the alert path entirely (legacy
    /// behaviour: keep retrying forever).
    #[serde(default = "default_max_retries_before_alert")]
    pub max_retries_before_alert: u32,
}

/// Quality gate configuration for the research verification loop.
///
/// All thresholds and prompts are configurable via JSON so you can tune
/// the quality bar and feedback wording without recompiling.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct GatekeeperConfig {
    /// Max verification rounds before accepting remaining issues.
    #[serde(default = "default_gk_max_rounds")]
    pub max_rounds: u32,
    /// Minimum excerpt length (chars) to be considered actionable.
    #[serde(default = "default_gk_min_excerpt")]
    pub min_excerpt_chars: usize,
    /// Minimum source_content length (chars).
    #[serde(default = "default_gk_min_source")]
    pub min_source_content_chars: usize,
    /// Maximum listing age in days. Older findings are removed.
    #[serde(default = "default_gk_max_age")]
    pub max_listing_age_days: i64,
    /// Require `listing_date` on every finding. When `true`, findings
    /// without a date trigger a feedback re-run.
    #[serde(default = "default_true")]
    pub require_listing_date: bool,
    /// Require `source_content` on every finding.
    #[serde(default = "default_true")]
    pub require_source_content: bool,
    /// Detect and remove semantic duplicates (same title+price, different URL).
    #[serde(default = "default_true")]
    pub detect_semantic_duplicates: bool,
    /// Stop the feedback loop when issue count doesn't decrease between rounds.
    #[serde(default = "default_true")]
    pub stop_on_stagnation: bool,
    /// HTTP timeout for URL liveness checks (seconds).
    #[serde(default = "default_gk_url_timeout")]
    pub url_check_timeout_secs: u64,

    // --- Prompt templates (use {placeholders}) ---
    /// Quality warnings returned inline by `research_save` when data is incomplete.
    /// Each entry is a condition→message pair checked at save time.
    /// Available placeholders: `{min_excerpt}`, `{min_source}`.
    #[serde(default)]
    pub save_warnings: GatekeeperSaveWarnings,

    /// Feedback prompt header sent to the agent when quality issues are found.
    /// Available placeholders: `{id}`, `{topic}`, `{today}`,
    /// `{dead_count}`, `{remediation_count}`, `{dead_list}`,
    /// `{quality_sections}`, `{dedup_list}`, `{min_excerpt}`, `{min_source}`.
    #[serde(default = "default_gk_feedback_header")]
    pub feedback_prompt_header: String,

    /// Section template for missing dates.
    /// Placeholders: `{count}`, `{today}`, `{list}`.
    #[serde(default = "default_gk_missing_date_section")]
    pub missing_date_section: String,

    /// Section template for missing source_content.
    /// Placeholders: `{count}`, `{min_source}`, `{list}`.
    #[serde(default = "default_gk_missing_source_section")]
    pub missing_source_section: String,

    /// Section template for short excerpts.
    /// Placeholders: `{count}`, `{min_excerpt}`, `{list}`.
    #[serde(default = "default_gk_short_excerpt_section")]
    pub short_excerpt_section: String,
}

/// Per-field quality warnings emitted by `research_save` so the agent
/// gets immediate feedback even during the first research pass.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct GatekeeperSaveWarnings {
    #[serde(default = "default_gk_warn_no_date")]
    pub no_listing_date: String,
    #[serde(default = "default_gk_warn_no_source")]
    pub no_source_content: String,
    #[serde(default = "default_gk_warn_short_excerpt")]
    pub short_excerpt: String,
}

impl Default for GatekeeperSaveWarnings {
    fn default() -> Self {
        Self {
            no_listing_date: default_gk_warn_no_date(),
            no_source_content: default_gk_warn_no_source(),
            short_excerpt: default_gk_warn_short_excerpt(),
        }
    }
}

impl Default for GatekeeperConfig {
    fn default() -> Self {
        Self {
            max_rounds: default_gk_max_rounds(),
            min_excerpt_chars: default_gk_min_excerpt(),
            min_source_content_chars: default_gk_min_source(),
            max_listing_age_days: default_gk_max_age(),
            require_listing_date: true,
            require_source_content: true,
            detect_semantic_duplicates: true,
            stop_on_stagnation: true,
            url_check_timeout_secs: default_gk_url_timeout(),
            save_warnings: GatekeeperSaveWarnings::default(),
            feedback_prompt_header: default_gk_feedback_header(),
            missing_date_section: default_gk_missing_date_section(),
            missing_source_section: default_gk_missing_source_section(),
            short_excerpt_section: default_gk_short_excerpt_section(),
        }
    }
}

fn default_gk_max_rounds() -> u32 {
    3
}
fn default_gk_min_excerpt() -> usize {
    300
}
fn default_gk_min_source() -> usize {
    100
}
fn default_gk_max_age() -> i64 {
    90
}
fn default_gk_url_timeout() -> u64 {
    10
}

fn default_gk_warn_no_date() -> String {
    "missing listing_date — look for ngày đăng/cập nhật on the page".into()
}
fn default_gk_warn_no_source() -> String {
    "missing source_content — paste the main page text (up to 8000 chars)".into()
}
fn default_gk_warn_short_excerpt() -> String {
    "excerpt too short (need ≥{min_excerpt} chars) — include contacts/phone, area m², price terms, deposit, address, condition".into()
}

fn default_gk_feedback_header() -> String {
    "You are an autonomous researcher. Research id: `{id}`.\n\
     \n\
     # GATEKEEPER FEEDBACK — verification round\n\
     \n\
     The previous research run has been verified. Some findings FAILED \
     quality checks. Your job now:\n\
     1. Find REPLACEMENTS for {dead_count} removed dead/stale links — \
        search for NEW listings using `web_search_exa` with different query \
        variations. Try at least 3 new search queries. Always OPEN the URL \
        with web_fetch BEFORE saving — verify it loads a real listing page.\n\
     2. FIX quality issues on {remediation_count} existing findings by \
        re-fetching each URL and calling `research_save` again with complete data.\n\
     \n\
     ## Removed findings (dead/stale — {dead_count})\n{dead_list}\n\
     \n\
     {remediation_section}\
     ## Quality issues breakdown\n\
     {quality_sections}\
     ## Rules\n\
     1. Every `research_save` call MUST include ALL of: title, price, \
        listing_date (today = {today}), excerpt (≥{min_excerpt} chars with contacts, \
        area, terms), source_content (≥{min_source} chars of page text).\n\
     2. To UPDATE an existing finding, call `research_save` with the SAME URL — \
        the system will overwrite the old data.\n\
     3. Do NOT re-save URLs that already have good data unless you are \
        specifically fixing an issue listed above.\n\
     4. For each remediation URL, actually OPEN the page (web_fetch or browser) \
        and re-extract the data. Do NOT guess or copy from memory.\n\
     5. If a page hides contacts behind login, note in excerpt: \
        \"Contacts hidden — requires site registration\".\n\
     6. Focus on the same topic: {topic}\n\
     \n\
     # Known findings (do NOT duplicate unless fixing)\n{dedup_list}\n"
        .into()
}

fn default_gk_missing_date_section() -> String {
    "### Missing listing_date ({count} findings)\n\
     Re-visit each URL below and look for the publication/update date. \
     Check: `ngày đăng`, `cập nhật`, `đăng ngày`, breadcrumbs, sidebar, \
     page footer near listing ID. Convert relative dates: `hôm nay` → {today}, \
     `hôm qua` → yesterday, `N ngày trước` → today minus N. \
     If truly absent, use `\"listing_date\": \"unknown\"`.\n{list}\n\n"
        .into()
}

fn default_gk_missing_source_section() -> String {
    "### Missing source_content ({count} findings)\n\
     Re-fetch each URL and paste the main page text (stripped of nav/ads/JS) \
     into the `source_content` field (up to 8000 chars). This is MANDATORY.\n{list}\n\n"
        .into()
}

fn default_gk_short_excerpt_section() -> String {
    "### Too-short excerpts ({count} findings)\n\
     Re-fetch each URL and expand the `excerpt` to ≥{min_excerpt} chars with ALL actionable \
     details: contacts (phone, Zalo, WhatsApp), area m², floor, conditions, \
     deposit, contract terms, amenities, neighbourhood.\n{list}\n\n"
        .into()
}

impl Default for ResearchConfig {
    fn default() -> Self {
        Self {
            enabled: default_research_enabled(),
            provider: None,
            model: None,
            fallback_models: Vec::new(),
            max_iterations: default_research_max_iterations(),
            max_wall_seconds: default_research_max_wall_seconds(),
            default_sources: Vec::new(),
            storage_dir: None,
            allowed_tools: None,
            schedule_enabled: true,
            default_interval_seconds: default_schedule_interval(),
            notify_on_new_findings: true,
            verify_by_default: true,
            reasoning: None,
            gatekeeper: GatekeeperConfig::default(),
            max_concurrent_runs: default_max_concurrent_runs(),
            task_timeout_seconds: default_task_timeout_seconds(),
            max_retries_before_alert: default_max_retries_before_alert(),
        }
    }
}

fn default_max_concurrent_runs() -> usize {
    5
}

/// Configuration for the lightweight memory-enrichment subsystem.
///
/// Two-tier storage: `MEMORY.md` holds active rules (shipped to every
/// system prompt), and per-day `memory/YYYY-MM-DD.md` files act as
/// short-lived drafts. A daily digest job merges yesterday's drafts
/// into `MEMORY.md` (LLM-summarized + scoring-based promotion), and
/// rejected candidates land in `DREAMS.md` for human review.
///
/// All fields are optional; defaults are tuned for "drop-in, no
/// config changes required" operation. Set `daily_enabled=false` to
/// disable the background job entirely.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct MemoryConfig {
    /// Master switch. `false` = no daily digest, no pre-compaction
    /// flush, no session-close summaries. The synchronous read APIs
    /// (`load_rules_for`, etc.) keep working unchanged.
    #[serde(default = "default_true")]
    pub daily_enabled: bool,
    /// Cron expression for the daily digest job. Defaults to `"0 4 * * *"`
    /// — 04:00 UTC, when most users are asleep.
    #[serde(default = "default_daily_cron")]
    pub daily_cron: String,
    /// What the digest does:
    /// * `"summarize_only"` — write a summary into `DREAMS.md`, never
    ///   touch `MEMORY.md`. Useful for inspecting the system before
    ///   trusting it with promotions.
    /// * `"summarize_and_promote"` (default) — also promote scoring
    ///   winners into `MEMORY.md`.
    #[serde(default = "default_daily_mode")]
    pub daily_mode: String,
    /// Provider/model used for digest LLM calls. `"<provider>/<model>"`
    /// or just `"<model>"` to use the default provider. `None` → use
    /// the global default. Pick a cheap model — the digest only sees
    /// short markdown lines.
    #[serde(default)]
    pub digest_provider: Option<String>,
    /// Maximum chars passed to the digest LLM in one call. Larger
    /// daily files are tail-truncated. Default: 8 KB.
    #[serde(default = "default_digest_max_chars")]
    pub digest_max_chars: usize,
    /// Maximum chars of the "Recent shift" block injected into every
    /// new turn's system prompt. Larger blobs are tail-truncated.
    #[serde(default = "default_recent_shift_max_chars")]
    pub recent_shift_max_chars: usize,
    /// How many days back the "Recent shift" block looks at. Default:
    /// 2 (today + yesterday).
    #[serde(default = "default_recent_shift_days")]
    pub recent_shift_days: u32,
    /// How many days of `memory/YYYY-MM-DD.md` files to keep on disk
    /// before the digest job rotates them out. Default: 30.
    #[serde(default = "default_daily_retention_days")]
    pub daily_retention_days: u32,
    /// How many days of `DREAMS.md` entries to keep before rotation.
    /// Default: 90.
    #[serde(default = "default_dreams_retention_days")]
    pub dreams_retention_days: u32,
    /// When `true`, `Session::reset/new` triggers a fire-and-forget
    /// summary into the current day's draft file.
    #[serde(default = "default_true")]
    pub session_close_summary: bool,
    /// Idle minutes before a session is treated as "closed" by the
    /// background sweep. `0` disables idle close. Default: 60.
    #[serde(default = "default_session_idle_close_minutes")]
    pub session_idle_close_minutes: u32,
    /// When `true`, `MemoryService::store` skips writes whose content
    /// hash already exists in the same scope. Default: `true`.
    #[serde(default = "default_true")]
    pub dedup_on_store: bool,
    /// When `true`, the conversation-history compactor performs a
    /// "silent turn" before sending the summary prompt to the LLM,
    /// asking it to flush rules-of-thumb / corrections to that day's
    /// draft file. Default: `true`.
    #[serde(default = "default_true")]
    pub pre_compaction_flush: bool,
    /// When `true`, the per-message memory classifier writes hits into
    /// today's daily draft file instead of `MEMORY.md`. The daily digest
    /// then decides whether to promote them based on
    /// `promote_min_repeat_days`. This prevents one-off mis-classifications
    /// from polluting durable rules. Default: `true`.
    ///
    /// Set to `false` to restore legacy behaviour (classifier writes
    /// straight to `MEMORY.md` — fast feedback, no scoring gate).
    #[serde(default = "default_true")]
    pub auto_classify_to_drafts: bool,
    /// "Forgetting" via compaction: when a `MEMORY.md` section grows
    /// past this many entries, the digest LLM is asked to merge the
    /// `compact_window` oldest items into a single summary line. The
    /// merged entry replaces them, with `source:compaction` and a
    /// `merged_from:` metadata trail. `0` disables compaction (default
    /// `12` — generous so small projects never hit it).
    #[serde(default = "default_compact_threshold")]
    pub compact_threshold_per_section: u32,
    /// How many of the oldest entries to roll into one when compaction
    /// fires. The current section must contain at least
    /// `compact_threshold_per_section` items. Default `5`.
    #[serde(default = "default_compact_window")]
    pub compact_window: u32,
    /// Hard upper bound on the LLM-generated merge summary. Anything
    /// longer is truncated server-side. Keep small so MEMORY.md never
    /// grows back faster than it shrinks. Default 240 chars.
    #[serde(default = "default_compact_max_chars")]
    pub compact_max_chars: usize,
    /// Don't compact entries younger than this many days — fresh rules
    /// are still earning their keep. Default 14.
    #[serde(default = "default_compact_min_age_days")]
    pub compact_min_age_days: u32,
    /// Spare entries from compaction if their **effective** recall
    /// score is at least this. The effective score applies an
    /// exponential half-life decay over `last_recalled_at` (see
    /// `compact_recall_half_life_days`) so a memory recalled 3 times
    /// last week is more protected than one recalled 3 times a year
    /// ago. Recently-used rules survive even when old. Default `2`.
    #[serde(default = "default_compact_spare_recall")]
    pub compact_spare_recall: u32,
    /// Half-life (in days) for the exponential decay applied to
    /// `recall_count` during compaction scoring. Operationally:
    /// `effective_recall = recall_count * 0.5^(age_in_days / half_life)`
    /// where `age_in_days = today - last_recalled_at` (or `today -
    /// created_at` if the entry was never recalled). `0` disables the
    /// decay entirely (back to the legacy "raw `recall_count`"
    /// behaviour). Default `30` days — a memory recalled once today
    /// fully counts; one recalled 30 days ago contributes 0.5 of its
    /// recall count; one recalled 90 days ago contributes 0.125.
    #[serde(default = "default_compact_recall_half_life_days")]
    pub compact_recall_half_life_days: u32,
    /// Promotion gate: a draft entry must appear on at least this many
    /// distinct daily files before it becomes a promotion candidate.
    /// Default: `2` (i.e. it stuck around for ≥2 days).
    #[serde(default = "default_promote_min_repeat_days")]
    pub promote_min_repeat_days: u32,
    /// Promotion gate: a draft entry must have been recalled (matched
    /// during context injection) at least this many times. `0` disables
    /// the gate (recall counts ignored). Default: `0` — recall
    /// tracking is best-effort and we don't want to block promotion
    /// on a rarely-instrumented signal.
    #[serde(default = "default_promote_min_recall_count")]
    pub promote_min_recall_count: u32,
}

impl Default for MemoryConfig {
    fn default() -> Self {
        Self {
            daily_enabled: true,
            daily_cron: default_daily_cron(),
            daily_mode: default_daily_mode(),
            digest_provider: None,
            digest_max_chars: default_digest_max_chars(),
            recent_shift_max_chars: default_recent_shift_max_chars(),
            recent_shift_days: default_recent_shift_days(),
            daily_retention_days: default_daily_retention_days(),
            dreams_retention_days: default_dreams_retention_days(),
            session_close_summary: true,
            session_idle_close_minutes: default_session_idle_close_minutes(),
            dedup_on_store: true,
            pre_compaction_flush: true,
            auto_classify_to_drafts: true,
            compact_threshold_per_section: default_compact_threshold(),
            compact_window: default_compact_window(),
            compact_max_chars: default_compact_max_chars(),
            compact_min_age_days: default_compact_min_age_days(),
            compact_spare_recall: default_compact_spare_recall(),
            compact_recall_half_life_days: default_compact_recall_half_life_days(),
            promote_min_repeat_days: default_promote_min_repeat_days(),
            promote_min_recall_count: default_promote_min_recall_count(),
        }
    }
}

fn default_daily_cron() -> String {
    "0 4 * * *".to_string()
}
fn default_daily_mode() -> String {
    "summarize_and_promote".to_string()
}
fn default_digest_max_chars() -> usize {
    8000
}
fn default_recent_shift_max_chars() -> usize {
    1500
}
fn default_recent_shift_days() -> u32 {
    2
}
fn default_daily_retention_days() -> u32 {
    30
}
fn default_dreams_retention_days() -> u32 {
    90
}
fn default_session_idle_close_minutes() -> u32 {
    60
}
fn default_promote_min_repeat_days() -> u32 {
    2
}
fn default_promote_min_recall_count() -> u32 {
    0
}
fn default_compact_threshold() -> u32 {
    12
}
fn default_compact_window() -> u32 {
    5
}
fn default_compact_max_chars() -> usize {
    240
}
fn default_compact_min_age_days() -> u32 {
    14
}
fn default_compact_spare_recall() -> u32 {
    2
}
fn default_compact_recall_half_life_days() -> u32 {
    30
}

fn default_task_timeout_seconds() -> u64 {
    1800
}

fn default_max_retries_before_alert() -> u32 {
    3
}

fn default_research_enabled() -> bool {
    true
}
fn default_research_max_iterations() -> u32 {
    30
}
fn default_research_max_wall_seconds() -> u64 {
    1200
}
fn default_true() -> bool {
    true
}
fn default_schedule_interval() -> u64 {
    21600
}

impl AudioProviderCfg {
    /// Resolve the API key: `$VAR` → env lookup, otherwise literal.
    pub fn resolved_api_key(&self) -> Result<String> {
        expand_env(&self.api_key)
    }
}

impl VisionProviderCfg {
    pub fn resolved_api_key(&self) -> Result<String> {
        expand_env(&self.api_key)
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProviderConfig {
    /// "anthropic" or "openai_compat" -- selects wire format
    #[serde(rename = "type")]
    pub provider_type: String,
    /// Primary API key, or `$ENV_VAR` to read from environment
    pub api_key: String,
    /// Additional keys for automatic rotation on failure
    #[serde(default)]
    pub api_keys: Vec<String>,
    #[serde(default)]
    pub base_url: Option<String>,
    #[serde(default)]
    pub models: Vec<String>,
    /// Per-provider max_tokens override (0 = use global default)
    #[serde(default)]
    pub max_tokens: Option<u32>,
    /// Per-provider temperature override
    #[serde(default)]
    pub temperature: Option<f32>,
    /// Per-provider context window override in tokens
    #[serde(default)]
    pub context_window: Option<u32>,
    /// Extra HTTP headers (e.g. User-Agent for Kimi Code)
    #[serde(default)]
    pub headers: HashMap<String, String>,
    /// Model name aliases: `alias → real_model_id`. The alias is what the
    /// user sees in `/model` menus and what `request.model` carries through
    /// the bot session; the real ID is what gets sent in the JSON body to
    /// the upstream API. Useful when the upstream model ID is opaque or
    /// differs from how the model is publicly known (e.g.
    /// `kimi-k2.6 → kimi-for-coding` because the Kimi For Coding endpoint
    /// is the K2.6 release, but the API only accepts the literal
    /// `kimi-for-coding` string).
    ///
    /// Aliases are appended to `models()` so they show up alongside real
    /// IDs in selection menus. If both the alias and the real ID are listed
    /// in `models`, that's fine — both will route to the same upstream
    /// model.
    #[serde(default)]
    pub model_aliases: HashMap<String, String>,
    /// Force vision-capability classification for this provider's models. When
    /// `Some(true)`, every model under this provider is treated as vision-
    /// capable (overrides the built-in needle list). When `Some(false)`, every
    /// model is forced text-only. When `None` (default), classification falls
    /// back to `BUILTIN_VISION_MODEL_NEEDLES`.
    ///
    /// Use this to opt-in new vision-capable models without recompiling the
    /// binary, or to explicitly disable native multimodal routing for a
    /// provider whose vision tier you don't want to pay for.
    #[serde(default)]
    pub supports_vision: Option<bool>,
    /// Structured per-model capability catalog. Keyed by model id (post-alias
    /// resolution) — entries for alias keys are accepted but the canonical
    /// location is the real model id. Missing entries default to
    /// [`crate::model_catalog::ModelCapabilities::unknown`], which is
    /// permissive so legacy configs keep booting unchanged. See the
    /// [`crate::model_catalog`] module for the schema.
    #[serde(default)]
    pub capabilities:
        HashMap<String, crate::model_catalog::ModelCapabilities>,
}

impl ProviderConfig {
    /// Resolve api_key: if starts with `$`, read from env var.
    pub fn resolved_api_key(&self) -> Result<String> {
        expand_env(&self.api_key)
    }

    /// All resolved keys: primary `api_key` + any `api_keys`, deduplicated.
    pub fn resolved_all_keys(&self) -> Vec<String> {
        let mut keys = Vec::new();
        if let Ok(k) = expand_env(&self.api_key)
            && !k.is_empty()
            && !k.starts_with('$')
        {
            keys.push(k);
        }
        for raw in &self.api_keys {
            if let Ok(k) = expand_env(raw)
                && !k.is_empty()
                && !k.starts_with('$')
                && !keys.contains(&k)
            {
                keys.push(k);
            }
        }
        keys
    }

    /// Return a copy with the api_key resolved from env.
    pub fn resolved(&self) -> Result<ResolvedProvider> {
        let all_keys = self.resolved_all_keys();
        Ok(ResolvedProvider {
            provider_type: self.provider_type.clone(),
            api_key: self.resolved_api_key()?,
            all_keys,
            base_url: self.base_url.clone(),
            models: self.models.clone(),
            max_tokens: self.max_tokens,
            temperature: self.temperature,
            headers: self.headers.clone(),
            model_aliases: self.model_aliases.clone(),
        })
    }

    /// Resolve a model name through `model_aliases`. If `model` matches an
    /// alias key, return the real upstream model ID; otherwise return the
    /// input unchanged. Used by every provider implementation right before
    /// it serializes the chat-completions request body.
    pub fn resolve_model_alias<'a>(&'a self, model: &'a str) -> &'a str {
        self.model_aliases
            .get(model)
            .map(String::as_str)
            .unwrap_or(model)
    }

    /// Look up the capability block for `model`, trying the literal id
    /// first and falling back to the alias-resolved id if the literal
    /// wasn't registered. Returns a cloned permissive default
    /// ([`crate::model_catalog::ModelCapabilities::unknown`]) when neither
    /// is present — callers never need to special-case `Option`.
    pub fn capabilities_for(
        &self,
        model: &str,
    ) -> crate::model_catalog::ModelCapabilities {
        if let Some(caps) = self.capabilities.get(model) {
            return caps.clone();
        }
        let resolved = self.resolve_model_alias(model);
        if resolved != model {
            if let Some(caps) = self.capabilities.get(resolved) {
                return caps.clone();
            }
        }
        crate::model_catalog::ModelCapabilities::unknown()
    }

    /// Combined model list: real `models` plus alias keys. Aliases that
    /// duplicate an entry already in `models` are skipped so the menu
    /// stays clean. Order: real models first (in declaration order), then
    /// aliases (alphabetical for stability).
    pub fn models_with_aliases(&self) -> Vec<String> {
        let mut out = self.models.clone();
        let mut alias_keys: Vec<&String> = self.model_aliases.keys().collect();
        alias_keys.sort();
        for k in alias_keys {
            if !out.iter().any(|m| m == k) {
                out.push(k.clone());
            }
        }
        out
    }
}

/// Provider config with api_key already resolved (no `$VAR` references).
#[derive(Debug, Clone)]
pub struct ResolvedProvider {
    pub provider_type: String,
    pub api_key: String,
    /// All valid keys (primary + extras), already resolved and deduplicated.
    pub all_keys: Vec<String>,
    pub base_url: Option<String>,
    pub models: Vec<String>,
    pub max_tokens: Option<u32>,
    pub temperature: Option<f32>,
    pub headers: HashMap<String, String>,
    /// See [`ProviderConfig::model_aliases`].
    pub model_aliases: HashMap<String, String>,
}

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

/// Emit `tracing::warn!` lines when a referenced `(provider, model)`
/// pair conflicts with its capability block. Shared by every selection
/// site in [`Config::validate_capabilities`]. Deliberately does nothing
/// for `status=active`/`fits(task)=true` pairs — silence is the happy
/// path.
///
/// `site` is a short human-readable label ("config.default_{provider,model}",
/// "config.research.fallback_models[]", ...) surfaced as a log field so
/// operators can grep to the offending config key.
fn warn_capability_mismatch(
    provider_name: &str,
    model: &str,
    task: crate::model_catalog::TaskKind,
    site: &'static str,
    pc: &ProviderConfig,
) {
    use crate::model_catalog::ModelStatus;

    // Only walk the alias chain if we actually have a caps block — no
    // sense hiding "unknown" pairs behind the permissive default.
    let resolved_model = pc.resolve_model_alias(model);
    let caps = match pc
        .capabilities
        .get(model)
        .or_else(|| pc.capabilities.get(resolved_model))
    {
        Some(c) => c,
        None => return, // No caps block → permissive unknown(), stay silent.
    };

    match caps.status {
        ModelStatus::Deprecated => {
            tracing::warn!(
                site = %site,
                provider = %provider_name,
                model = %model,
                task = %task,
                status = "deprecated",
                known_failure_modes = ?caps.known_failure_modes,
                notes = ?caps.notes,
                "model is deprecated; selector will refuse once enforce_model_capabilities=true"
            );
            return;
        }
        ModelStatus::Experimental => {
            tracing::warn!(
                site = %site,
                provider = %provider_name,
                model = %model,
                task = %task,
                status = "experimental",
                "model is marked experimental; pin it explicitly in new sessions only"
            );
            return;
        }
        ModelStatus::Degraded | ModelStatus::Active => {}
    }

    if !caps.fits(task) {
        tracing::warn!(
            site = %site,
            provider = %provider_name,
            model = %model,
            task = %task,
            task_fit = ?caps.task_fit,
            "model is not marked fit for this task in the capability catalog"
        );
    }

    if matches!(caps.status, ModelStatus::Degraded)
        && matches!(
            task,
            crate::model_catalog::TaskKind::Research
                | crate::model_catalog::TaskKind::Coding
        )
    {
        tracing::warn!(
            site = %site,
            provider = %provider_name,
            model = %model,
            task = %task,
            known_failure_modes = ?caps.known_failure_modes,
            "model is Degraded but referenced from a high-stakes task; \
             consider swapping to an Active alternative"
        );
    }
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
            telegram_bot_token: None,
            allowed_chat_ids: Vec::new(),
            tg_sender_attribution: default_sender_attribution(),
            exa_api_keys: Vec::new(),
            tg_media: TgMediaConfig::default(),
            research: ResearchConfig::default(),
            memory: MemoryConfig::default(),
            agent_roles: HashMap::new(),
            default_reasoning: None,
            chat_personas: HashMap::new(),
            enforce_model_capabilities: default_enforce_model_capabilities(),
            model_health: crate::model_catalog::ModelHealthConfig::default(),
        }
    }
}

impl Config {
    /// Load config: JSON file -> env var overrides.
    ///
    /// Search order for JSON:
    /// 1. `NAKED_CONFIG` env var
    /// 2. `.naked/config.json` in workspace
    /// 3. `naked.json` in workspace
    /// 4. `~/.naked/config.json`
    pub fn load() -> Result<Self> {
        Self::load_dotenv();

        let mut cfg = if let Ok(path) = std::env::var("NAKED_CONFIG") {
            Self::from_json_file(Path::new(&path))?
        } else {
            Self::discover_json()?
        };

        cfg.apply_env_overrides();
        cfg.resolve_mcp_names();

        cfg.skill_roots = cfg
            .skill_roots
            .into_iter()
            .map(|p| expand_tilde(&p))
            .collect();

        if cfg.skill_roots.is_empty() {
            let home = dirs_home();
            cfg.skill_roots = vec![home.join(".naked/skills"), home.join(".agents/skills")];
        }

        cfg.agent_dirs = cfg
            .agent_dirs
            .into_iter()
            .map(|p| expand_tilde(&p))
            .collect();

        if cfg.agent_dirs.is_empty() {
            let home = dirs_home();
            cfg.agent_dirs = vec![PathBuf::from("./agents"), home.join(".naked/agents")];
        }

        for persona in cfg.chat_personas.values_mut() {
            persona.workspace = expand_tilde(&persona.workspace);
        }

        cfg.validate_and_warn();

        Ok(cfg)
    }

    /// Sanity-check the loaded config and emit `tracing::warn!` for the
    /// common silent-failure cases: `default_provider` not present in
    /// `providers`, referenced provider missing `api_key`, default model
    /// not in the provider's `models` list. Never fails — warns only.
    ///
    /// Rationale: previously a typo in `default_provider` would silently
    /// make the bot use whatever provider happened to be first in the map.
    /// The bot would boot, accept messages, and hallucinate errors later
    /// when it hit the non-existent key. A 2-line warn at startup is
    /// dramatically cheaper to diagnose.
    pub fn validate_and_warn(&self) {
        if !self.default_provider.is_empty() && !self.providers.contains_key(&self.default_provider)
        {
            tracing::warn!(
                provider = %self.default_provider,
                known = ?self.providers.keys().collect::<Vec<_>>(),
                "config.default_provider is not in `providers` — requests will fail"
            );
        }

        if let Some(pc) = self.providers.get(&self.default_provider)
            && !self.default_model.is_empty()
            && !pc.models.is_empty()
            && !pc.models.iter().any(|m| m == &self.default_model)
        {
            tracing::warn!(
                model = %self.default_model,
                provider = %self.default_provider,
                known = ?pc.models,
                "config.default_model is not listed under provider's `models` — may be rejected"
            );
        }

        for (name, pc) in &self.providers {
            if pc.api_key.is_empty() && pc.provider_type != "copilot" {
                tracing::warn!(
                    provider = %name,
                    "provider has empty api_key and is not copilot (auto-login); \
                     requests will 401 when routed here"
                );
            }
        }

        // Research subsystem: warn if the override points at a provider or model
        // that won't actually work at runtime. Typos here silently fell back to
        // the main provider before v5 — which made research burn the very
        // tokens the user was trying to save.
        if self.research.enabled {
            if let Some(p) = self.research.provider.as_deref()
                && !p.is_empty()
                && !self.providers.contains_key(p)
            {
                tracing::warn!(
                    research_provider = %p,
                    known = ?self.providers.keys().collect::<Vec<_>>(),
                    "config.research.provider is not in `providers` — research will fall back to default"
                );
            }
            if let Some(p) = self.research.provider.as_deref()
                && let Some(pc) = self.providers.get(p)
                && let Some(m) = self.research.model.as_deref()
                && !m.is_empty()
                && !pc.models.is_empty()
                && !pc.models.iter().any(|model| model == m)
            {
                tracing::warn!(
                    research_model = %m,
                    research_provider = %p,
                    known = ?pc.models,
                    "config.research.model is not listed under its provider — may be rejected"
                );
            }
        }

        // Capability-catalog checks. Soft (warn-only) under the default
        // `enforce_model_capabilities=false`; phase 2 wires them into the
        // selector so warnings become hard filters. We still emit the
        // warning in enforce mode so operators see why a model was
        // filtered out.
        self.validate_capabilities();
    }

    /// Walk every selection site and compare the (provider, model) pair
    /// against the structured capability catalog
    /// ([`crate::model_catalog::ModelCapabilities`]).
    ///
    /// Three classes of warning are emitted:
    /// 1. `status=deprecated` — the selector will eventually refuse to
    ///    pick this; fix the config now.
    /// 2. `status=degraded` with a known-bad task fit — fine for
    ///    operator-pinned chat, but a lurking surprise for research.
    /// 3. `fits(task)=false` — the task isn't in the model's `task_fit`
    ///    list; structured selectors will skip it.
    ///
    /// Pairs with no capability block default to
    /// [`crate::model_catalog::ModelCapabilities::unknown`] and pass
    /// through silently (back-compat with legacy configs that haven't
    /// been seeded yet).
    fn validate_capabilities(&self) {
        use crate::model_catalog::{ModelStatus, TaskKind};

        // --- default chat pair ---------------------------------------
        if !self.default_provider.is_empty()
            && !self.default_model.is_empty()
            && let Some(pc) = self.providers.get(&self.default_provider)
        {
            warn_capability_mismatch(
                &self.default_provider,
                &self.default_model,
                TaskKind::Chat,
                "config.default_{provider,model}",
                pc,
            );
        }

        // --- global fallback chain -----------------------------------
        // Entries are "provider/model" strings; the chat turn consumes
        // them when the primary provider fails (see
        // `AgentConfig::fallback_providers`).
        for entry in &self.fallback {
            let Some((p, m)) = crate::research::parse_provider_model_pair(entry)
            else {
                continue;
            };
            if let Some(pc) = self.providers.get(&p) {
                warn_capability_mismatch(
                    &p,
                    &m,
                    TaskKind::Chat,
                    "config.fallback[]",
                    pc,
                );
            }
        }

        // --- research primary ---------------------------------------
        if self.research.enabled
            && let Some(p) = self
                .research
                .provider
                .as_deref()
                .filter(|s| !s.is_empty())
                .or_else(|| {
                    if self.default_provider.is_empty() {
                        None
                    } else {
                        Some(self.default_provider.as_str())
                    }
                })
            && let Some(pc) = self.providers.get(p)
            && let Some(m) = self
                .research
                .model
                .as_deref()
                .filter(|s| !s.is_empty())
                .or_else(|| {
                    if self.default_model.is_empty() {
                        None
                    } else {
                        Some(self.default_model.as_str())
                    }
                })
        {
            warn_capability_mismatch(
                p,
                m,
                TaskKind::Research,
                "config.research.{provider,model}",
                pc,
            );
        }

        // --- research fallback chain ---------------------------------
        if self.research.enabled {
            for entry in &self.research.fallback_models {
                // Entries may be bare model ids (resolved against
                // research.provider / default_provider) or
                // "provider/model" pairs.
                let (p, m) = match crate::research::parse_provider_model_pair(entry) {
                    Some(pair) => pair,
                    None => {
                        // Bare model — pin to research.provider or default.
                        let p = self
                            .research
                            .provider
                            .clone()
                            .filter(|s| !s.is_empty())
                            .unwrap_or_else(|| self.default_provider.clone());
                        if p.is_empty() {
                            continue;
                        }
                        (p, entry.clone())
                    }
                };
                if let Some(pc) = self.providers.get(&p) {
                    warn_capability_mismatch(
                        &p,
                        &m,
                        TaskKind::Research,
                        "config.research.fallback_models[]",
                        pc,
                    );
                }
            }
        }

        // --- memory digest provider ---------------------------------
        // The memory module parses `memory.digest_provider` as either
        // "<model>" (inherits default provider) or "<provider>/<model>".
        if let Some(entry) = self.memory.digest_provider.as_deref()
            && !entry.is_empty()
        {
            let (p, m) = match crate::research::parse_provider_model_pair(entry) {
                Some(pair) => pair,
                None => {
                    if self.default_provider.is_empty() {
                        return;
                    }
                    (self.default_provider.clone(), entry.to_string())
                }
            };
            if let Some(pc) = self.providers.get(&p) {
                warn_capability_mismatch(
                    &p,
                    &m,
                    TaskKind::Digest,
                    "config.memory.digest_provider",
                    pc,
                );
            }
        }

        // --- hard-deprecated model final sweep -----------------------
        // Also walk every explicitly-catalogued (provider, model) pair
        // and shout about `status=deprecated` ones — useful for pairs
        // that aren't referenced by any selection site yet but are
        // still in `providers[x].models[]`.
        for (pname, pc) in &self.providers {
            for model in &pc.models {
                if let Some(caps) = pc.capabilities.get(model)
                    && matches!(caps.status, ModelStatus::Deprecated)
                {
                    tracing::warn!(
                        provider = %pname,
                        model = %model,
                        "declared model is marked status=deprecated in capabilities; \
                         remove it from `providers.{}.models` or flip its status"
                         , pname
                    );
                }
            }
        }
    }

    /// Load from a specific JSON file.
    pub fn from_json_file(path: &Path) -> Result<Self> {
        if !path.exists() {
            return Err(AgentError::Config(format!(
                "config file not found: {}",
                path.display()
            )));
        }
        let data = std::fs::read_to_string(path)?;
        Self::from_json_str(&data)
    }

    /// Parse from JSON string.
    pub fn from_json_str(json: &str) -> Result<Self> {
        serde_json::from_str(json).map_err(|e| AgentError::Config(format!("JSON parse error: {e}")))
    }

    /// Search standard locations for config JSON.
    /// Walks up parent directories (like git) to find config.
    fn discover_json() -> Result<Self> {
        let cwd = std::env::current_dir().unwrap_or_else(|_| PathBuf::from("."));

        let mut dir = cwd.as_path();
        loop {
            let names = [".naked/config.json", "naked.json"];
            for name in &names {
                let path = dir.join(name);
                if path.exists() {
                    tracing::info!("Loading config from {}", path.display());
                    return Self::from_json_file(&path);
                }
            }
            match dir.parent() {
                Some(p) => dir = p,
                None => break,
            }
        }

        let home = dirs_home().join(".naked/config.json");
        if home.exists() {
            tracing::info!("Loading config from {}", home.display());
            return Self::from_json_file(&home);
        }

        Ok(Self::default())
    }

    /// Walk up from cwd to find `.env`, like `discover_json`.
    fn load_dotenv() {
        let cwd = std::env::current_dir().unwrap_or_else(|_| PathBuf::from("."));
        let mut dir = cwd.as_path();
        loop {
            let path = dir.join(".env");
            if path.exists() {
                let _ = dotenvy::from_path(&path);
                return;
            }
            match dir.parent() {
                Some(p) => dir = p,
                None => break,
            }
        }
        let _ = dotenvy::dotenv();
    }

    /// Env vars override JSON values (for CI, Docker, etc).
    fn apply_env_overrides(&mut self) {
        if let Ok(v) = std::env::var("NAKED_PROVIDER") {
            self.default_provider = v;
        }
        if let Ok(v) = std::env::var("NAKED_MODEL") {
            self.default_model = v;
        }
        if let Ok(v) = std::env::var("NAKED_WORKSPACE") {
            self.workspace = PathBuf::from(v);
        }
        if let Ok(v) = std::env::var("NAKED_MAX_ITERATIONS")
            && let Ok(n) = v.parse()
        {
            self.max_iterations = n;
        }
        if let Ok(v) = std::env::var("NAKED_MAX_TOKENS")
            && let Ok(n) = v.parse()
        {
            self.max_tokens = n;
        }
        if let Ok(v) = std::env::var("NAKED_TEMPERATURE")
            && let Ok(n) = v.parse()
        {
            self.temperature = Some(n);
        }
        if let Ok(v) = std::env::var("NAKED_TOOL_TIMEOUT")
            && let Ok(n) = v.parse()
        {
            self.tool_timeout_secs = n;
        }
        if let Ok(v) = std::env::var("NAKED_SESSION_DIR") {
            self.session_dir = PathBuf::from(v);
        }
        if let Ok(v) = std::env::var("NAKED_TELEGRAM_TOKEN") {
            self.telegram_bot_token = Some(v);
        }
        if let Ok(v) = std::env::var("NAKED_ALLOWED_CHAT_IDS") {
            self.allowed_chat_ids = v
                .split(',')
                .filter_map(|s| s.trim().parse::<i64>().ok())
                .collect();
        }
        if self.exa_api_keys.is_empty() {
            if let Ok(v) = std::env::var("EXA_API_KEYS") {
                self.exa_api_keys = v
                    .split(',')
                    .map(|s| s.trim().to_string())
                    .filter(|s| !s.is_empty())
                    .collect();
            } else if let Ok(v) = std::env::var("EXA_API_KEY")
                && !v.is_empty()
            {
                self.exa_api_keys = vec![v];
            }
        }
    }

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
                .ok_or_else(|| AgentError::Config("no providers configured".into()))?
                .clone()
        } else {
            self.default_provider.clone()
        };

        let pc = self
            .providers
            .get(&name)
            .ok_or_else(|| AgentError::Config(format!("provider '{name}' not found in config")))?;

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

/// Per-session config override. All fields are optional — missing fields
/// fall back to the global `Config`. Placed in `sessions/{id}/config.json`.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct SessionConfig {
    #[serde(default)]
    pub default_provider: Option<String>,
    #[serde(default)]
    pub default_model: Option<String>,
    #[serde(default)]
    pub max_tokens: Option<u32>,
    #[serde(default)]
    pub temperature: Option<f32>,
    #[serde(default)]
    pub context_window: Option<u32>,
    #[serde(default)]
    pub max_iterations: Option<usize>,
    /// Reasoning/thinking level: "off", "low", "medium", "high"
    #[serde(default)]
    pub reasoning: Option<String>,
    #[serde(default, rename = "mcpServers")]
    pub mcp_servers: Option<HashMap<String, McpServerConfig>>,
    #[serde(default)]
    pub skill_roots: Option<Vec<PathBuf>>,
    #[serde(default)]
    pub system_prompt_path: Option<PathBuf>,
    /// Unix timestamp (secs) when yolo was enabled. Expires after 72h.
    #[serde(default)]
    pub yolo_enabled_at: Option<i64>,
    /// Per-session tool allow-list (persisted across restarts).
    #[serde(default)]
    pub allow_list: Option<Vec<String>>,
}

impl SessionConfig {
    pub fn from_file(path: &Path) -> Result<Self> {
        let data = std::fs::read_to_string(path)?;
        serde_json::from_str(&data)
            .map_err(|e| AgentError::Config(format!("session config parse error: {e}")))
    }
}

/// Effective config for a single session: global merged with per-session overrides.
#[derive(Debug, Clone)]
pub struct EffectiveSessionConfig {
    pub provider: String,
    pub model: String,
    pub max_tokens: u32,
    pub temperature: Option<f32>,
    /// Context window in tokens (None = default 100k).
    pub context_window: Option<u32>,
    pub max_iterations: usize,
    /// Reasoning/thinking level: "off", "low", "medium", "high"
    pub reasoning: Option<String>,
    pub mcp_servers: HashMap<String, McpServerConfig>,
    pub skill_roots: Vec<PathBuf>,
    pub system_prompt_path: Option<PathBuf>,
}

impl Config {
    /// Merge global config with per-session overrides.
    /// Session values win; missing session fields fall back to global.
    /// MCP servers are additive: session servers are merged on top of global
    /// (session overrides global if same name).
    pub fn merge_session(&self, session: &SessionConfig) -> EffectiveSessionConfig {
        let mut mcp_servers = self.mcp_servers.clone();
        if let Some(extra) = &session.mcp_servers {
            for (name, cfg) in extra {
                mcp_servers.insert(name.clone(), cfg.clone());
            }
        }

        let skill_roots = session
            .skill_roots
            .clone()
            .unwrap_or_else(|| self.skill_roots.clone());

        EffectiveSessionConfig {
            provider: session
                .default_provider
                .clone()
                .unwrap_or_else(|| self.default_provider.clone()),
            model: session
                .default_model
                .clone()
                .unwrap_or_else(|| self.default_model.clone()),
            max_tokens: session.max_tokens.unwrap_or(self.max_tokens),
            temperature: session.temperature.or(self.temperature),
            context_window: session.context_window.or(self.context_window),
            max_iterations: session.max_iterations.unwrap_or(self.max_iterations),
            // Session reasoning wins; otherwise inherit `Config.default_reasoning`
            // so e.g. the bot launches every chat at the configured global
            // thinking level without the user re-typing `/reasoning medium`.
            reasoning: session
                .reasoning
                .clone()
                .or_else(|| self.default_reasoning.clone()),
            mcp_servers,
            skill_roots,
            system_prompt_path: session
                .system_prompt_path
                .clone()
                .or_else(|| self.system_prompt_path.clone()),
        }
    }

    /// Produce an effective config with no overrides (all global values).
    pub fn default_effective(&self) -> EffectiveSessionConfig {
        self.merge_session(&SessionConfig::default())
    }
}

/// Expand `$VAR` or `${VAR}` references in a string from env.
pub fn expand_env(s: &str) -> Result<String> {
    if let Some(var_name) = s.strip_prefix('$') {
        let var_name = var_name.trim_start_matches('{').trim_end_matches('}');
        std::env::var(var_name).map_err(|_| {
            AgentError::Config(format!(
                "env var '{var_name}' not set (referenced in config)"
            ))
        })
    } else {
        Ok(s.to_string())
    }
}

/// Expand leading `~` or `~/` to the user's home directory.
pub fn expand_tilde(p: &Path) -> PathBuf {
    let s = p.to_string_lossy();
    if s == "~" {
        dirs_home()
    } else if let Some(rest) = s.strip_prefix("~/") {
        dirs_home().join(rest)
    } else {
        p.to_path_buf()
    }
}

fn dirs_home() -> PathBuf {
    std::env::var("HOME")
        .map(PathBuf::from)
        .unwrap_or_else(|_| PathBuf::from("."))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn vision_capable_model_builtin_allowlist() {
        let cfg = TgMediaConfig::default();
        // Anthropic Claude 3+
        assert!(cfg.is_vision_capable_model("claude-sonnet-4-20250514"));
        assert!(cfg.is_vision_capable_model("claude-3-5-sonnet-20241022"));
        assert!(cfg.is_vision_capable_model("claude-haiku-4-5-20251001"));
        // OpenAI
        assert!(cfg.is_vision_capable_model("gpt-4o"));
        assert!(cfg.is_vision_capable_model("gpt-4o-mini"));
        // Groq Llama-4
        assert!(cfg.is_vision_capable_model("meta-llama/llama-4-scout-17b-16e-instruct"));
        assert!(cfg.is_vision_capable_model("meta-llama/Llama-4-Maverick-17B-128E-Instruct"));
        // xAI
        assert!(cfg.is_vision_capable_model("grok-2-vision-latest"));
        // Gemini
        assert!(cfg.is_vision_capable_model("gemini-1.5-pro"));
        // Text-only models — NOT vision
        assert!(!cfg.is_vision_capable_model("llama-3.3-70b-versatile"));
        assert!(!cfg.is_vision_capable_model("glm-5-turbo"));
        assert!(!cfg.is_vision_capable_model("MiniMax-Text-01"));
        assert!(!cfg.is_vision_capable_model("deepseek-chat"));
        assert!(!cfg.is_vision_capable_model("kimi-k2"));
    }

    #[test]
    fn provider_supports_vision_override_force_true() {
        let cfg = TgMediaConfig::default();
        let mut pc = test_pc("k");
        pc.supports_vision = Some(true);
        // Random unknown text-only model name → still treated as vision-capable.
        assert!(cfg.is_vision_capable_with_provider("brand-new-2099-omni", Some(&pc)));
    }

    #[test]
    fn provider_supports_vision_override_force_false() {
        let cfg = TgMediaConfig::default();
        let mut pc = test_pc("k");
        pc.supports_vision = Some(false);
        // claude-3 normally hits the built-in needles → forced off here.
        assert!(!cfg.is_vision_capable_with_provider("claude-3-5-sonnet-20240620", Some(&pc)));
    }

    #[test]
    fn model_vision_overrides_take_precedence() {
        let mut cfg = TgMediaConfig::default();
        cfg.model_vision_overrides
            .insert("gpt-4o-mini".into(), false); // force off even though needles say yes
        cfg.model_vision_overrides
            .insert("custom-llama-vision".into(), true);
        let mut pc = test_pc("k");
        pc.supports_vision = Some(true);
        // Per-model false beats provider-wide true and built-in needle.
        assert!(!cfg.is_vision_capable_with_provider("gpt-4o-mini", Some(&pc)));
        // Per-model true lights up an unknown model regardless of provider config.
        assert!(cfg.is_vision_capable_with_provider("custom-llama-vision-7b", None));
    }

    #[test]
    fn provider_image_cap_per_provider_floors() {
        let cfg = TgMediaConfig {
            native_image_max_bytes: 50 * 1024 * 1024,
            ..TgMediaConfig::default()
        };
        // Anthropic floor → 5 MB, even with global 50 MB.
        assert_eq!(cfg.provider_image_cap("anthropic", None), 5 * 1024 * 1024);
        assert_eq!(
            cfg.provider_image_cap("openai_compat", Some("https://api.groq.com/openai/v1")),
            4 * 1024 * 1024
        );
        assert_eq!(
            cfg.provider_image_cap("openai_compat", Some("https://api.openai.com/v1")),
            20 * 1024 * 1024
        );
        // Unknown base URL — falls back to global.
        assert_eq!(
            cfg.provider_image_cap("openai_compat", Some("https://example.com")),
            50 * 1024 * 1024
        );
    }

    #[test]
    fn provider_image_cap_respects_global_below_provider_limit() {
        let cfg = TgMediaConfig {
            native_image_max_bytes: 1024 * 1024, // 1 MB global
            ..TgMediaConfig::default()
        };
        // Global is the floor when stricter than provider's own cap.
        assert_eq!(
            cfg.provider_image_cap("openai_compat", Some("https://api.openai.com/v1")),
            1024 * 1024
        );
    }

    #[test]
    fn provider_supports_vision_none_falls_back_to_needles() {
        let cfg = TgMediaConfig::default();
        let pc = test_pc("k");
        // None → fall through to needle list.
        assert!(cfg.is_vision_capable_with_provider("claude-haiku-4-5-20251001", Some(&pc)));
        assert!(!cfg.is_vision_capable_with_provider("llama-3.3-70b-versatile", Some(&pc)));
    }

    #[test]
    fn vision_capable_model_extras_extend_allowlist() {
        let mut cfg = TgMediaConfig::default();
        assert!(!cfg.is_vision_capable_model("internlm-xcomposer2.5-7b"));
        cfg.vision_model_extras = vec!["internlm-xcomposer".into(), "qwen3-vl".into()];
        assert!(cfg.is_vision_capable_model("internlm-xcomposer2.5-7b"));
        assert!(cfg.is_vision_capable_model("Qwen3-VL-72B"));
        // Empty entries are ignored.
        cfg.vision_model_extras = vec!["".into()];
        assert!(!cfg.is_vision_capable_model("anything"));
    }

    #[test]
    fn parse_minimal_json() {
        let json = r#"{"providers": {}, "default_provider": "", "default_model": ""}"#;
        let cfg = Config::from_json_str(json).unwrap();
        assert!(cfg.providers.is_empty());
        assert_eq!(cfg.max_iterations, 0);
    }

    #[test]
    fn parse_full_json() {
        let json = r#"{
            "providers": {
                "claude": {
                    "type": "anthropic",
                    "api_key": "test-api-key",
                    "models": ["claude-sonnet-4"]
                },
                "groq": {
                    "type": "openai_compat",
                    "api_key": "$GROQ_API_KEY",
                    "base_url": "https://api.groq.com/openai/v1",
                    "models": ["llama-3.3-70b"]
                }
            },
            "default_provider": "claude",
            "default_model": "claude-sonnet-4",
            "fallback": ["groq/llama-3.3-70b"],
            "max_iterations": 30,
            "mcpServers": {
                "fs": {"command": "mcp-fs", "args": ["--root", "/"]}
            }
        }"#;
        let cfg = Config::from_json_str(json).unwrap();
        assert_eq!(cfg.providers.len(), 2);
        assert_eq!(cfg.default_provider, "claude");
        assert_eq!(cfg.max_iterations, 30);
        assert_eq!(cfg.mcp_server_list().len(), 1);
        assert_eq!(cfg.fallback_providers().len(), 1);
        assert_eq!(
            cfg.fallback_providers()[0],
            ("groq".into(), "llama-3.3-70b".into())
        );
    }

    fn test_pc(api_key: &str) -> ProviderConfig {
        ProviderConfig {
            provider_type: "openai_compat".into(),
            api_key: api_key.into(),
            api_keys: Vec::new(),
            base_url: None,
            models: Vec::new(),
            max_tokens: None,
            temperature: None,
            context_window: None,
            headers: Default::default(),
            supports_vision: None,
            model_aliases: Default::default(),
            capabilities: Default::default(),
        }
    }

    #[test]
    fn provider_resolved_direct_key() {
        let mut pc = test_pc("test-direct-key");
        pc.provider_type = "anthropic".into();
        let resolved = pc.resolved().unwrap();
        assert_eq!(resolved.api_key, "test-direct-key");
    }

    #[test]
    fn resolved_all_keys_single() {
        let keys = test_pc("key-a").resolved_all_keys();
        assert_eq!(keys, vec!["key-a"]);
    }

    #[test]
    fn resolved_all_keys_multiple() {
        let mut pc = test_pc("key-a");
        pc.api_keys = vec!["key-b".into(), "key-c".into()];
        assert_eq!(pc.resolved_all_keys(), vec!["key-a", "key-b", "key-c"]);
    }

    #[test]
    fn resolved_all_keys_deduplicates() {
        let mut pc = test_pc("key-a");
        pc.api_keys = vec!["key-a".into(), "key-b".into()];
        assert_eq!(pc.resolved_all_keys(), vec!["key-a", "key-b"]);
    }

    #[test]
    fn resolved_all_keys_skips_unresolved_env() {
        let mut pc = test_pc("key-a");
        pc.api_keys = vec!["$NONEXISTENT_KEY_FOR_TEST_XYZ".into(), "key-b".into()];
        assert_eq!(pc.resolved_all_keys(), vec!["key-a", "key-b"]);
    }

    #[test]
    fn resolved_provider_has_all_keys() {
        let mut pc = test_pc("key-1");
        pc.provider_type = "anthropic".into();
        pc.api_keys = vec!["key-2".into()];
        let resolved = pc.resolved().unwrap();
        assert_eq!(resolved.all_keys, vec!["key-1", "key-2"]);
    }

    #[test]
    fn resolved_provider_carries_max_tokens_and_temperature() {
        let mut pc = test_pc("key-x");
        pc.max_tokens = Some(4096);
        pc.temperature = Some(0.7);
        let resolved = pc.resolved().unwrap();
        assert_eq!(resolved.max_tokens, Some(4096));
        assert_eq!(resolved.temperature, Some(0.7));
    }

    #[test]
    fn effective_max_tokens_global_vs_provider() {
        let json = r#"{
            "providers": {
                "fast": {"type": "openai_compat", "api_key": "k", "max_tokens": 4096},
                "big": {"type": "openai_compat", "api_key": "k"}
            },
            "max_tokens": 8192
        }"#;
        let cfg = Config::from_json_str(json).unwrap();
        assert_eq!(cfg.effective_max_tokens("fast"), 4096);
        assert_eq!(cfg.effective_max_tokens("big"), 8192);
        assert_eq!(cfg.effective_max_tokens("missing"), 8192);
    }

    #[test]
    fn effective_temperature_global_vs_provider() {
        let json = r#"{
            "providers": {
                "creative": {"type": "openai_compat", "api_key": "k", "temperature": 0.9},
                "default": {"type": "openai_compat", "api_key": "k"}
            },
            "temperature": 0.3
        }"#;
        let cfg = Config::from_json_str(json).unwrap();
        assert_eq!(cfg.effective_temperature("creative"), Some(0.9));
        assert_eq!(cfg.effective_temperature("default"), Some(0.3));
        assert_eq!(cfg.effective_temperature("missing"), Some(0.3));
    }

    #[test]
    fn temperature_none_when_unset() {
        let json = r#"{"providers": {"p": {"type": "openai_compat", "api_key": "k"}}}"#;
        let cfg = Config::from_json_str(json).unwrap();
        assert_eq!(cfg.effective_temperature("p"), None);
    }

    #[test]
    fn multi_key_json_roundtrip() {
        let json = r#"{
            "type": "openai_compat",
            "api_key": "primary",
            "api_keys": ["backup1", "backup2"],
            "models": ["gpt-4o"]
        }"#;
        let pc: ProviderConfig = serde_json::from_str(json).unwrap();
        assert_eq!(pc.api_key, "primary");
        assert_eq!(pc.api_keys, vec!["backup1", "backup2"]);
        let keys = pc.resolved_all_keys();
        assert_eq!(keys, vec!["primary", "backup1", "backup2"]);
    }

    #[test]
    fn expand_env_direct_value() {
        assert_eq!(expand_env("plain-key").unwrap(), "plain-key");
    }

    #[test]
    fn expand_env_dollar_var() {
        // Test with a var we know exists in any Unix environment
        let home = std::env::var("HOME").unwrap_or_default();
        if !home.is_empty() {
            assert_eq!(expand_env("$HOME").unwrap(), home);
        }
    }

    #[test]
    fn expand_env_braces() {
        let home = std::env::var("HOME").unwrap_or_default();
        if !home.is_empty() {
            assert_eq!(expand_env("${HOME}").unwrap(), home);
        }
    }

    #[test]
    fn expand_env_missing_var_is_error() {
        assert!(expand_env("$DEFINITELY_NOT_SET_ABCXYZ").is_err());
    }

    #[test]
    fn session_dir_abs_relative() {
        let cfg = Config {
            workspace: PathBuf::from("/home/user/project"),
            session_dir: PathBuf::from(".naked/sessions"),
            ..Default::default()
        };
        assert_eq!(
            cfg.session_dir_abs(),
            PathBuf::from("/home/user/project/.naked/sessions")
        );
    }

    #[test]
    fn session_dir_abs_absolute() {
        let cfg = Config {
            session_dir: PathBuf::from("/tmp/sessions"),
            ..Default::default()
        };
        assert_eq!(cfg.session_dir_abs(), PathBuf::from("/tmp/sessions"));
    }

    #[test]
    fn default_config_empty() {
        let cfg = Config::default();
        assert!(cfg.providers.is_empty());
        assert!(cfg.default_provider.is_empty());
        assert_eq!(cfg.max_iterations, 0);
    }

    #[test]
    fn mcp_servers_get_names() {
        let json = r#"{"mcpServers": {"my-server": {"command": "test"}}}"#;
        let mut cfg = Config::from_json_str(json).unwrap();
        cfg.resolve_mcp_names();
        let servers = cfg.mcp_server_list();
        assert_eq!(servers[0].name, "my-server");
    }

    #[test]
    fn resolve_default_provider_picks_first() {
        let json = r#"{"providers": {"only": {"type": "openai_compat", "api_key": "k"}}}"#;
        let cfg = Config::from_json_str(json).unwrap();
        let (name, resolved) = cfg.resolve_default_provider().unwrap();
        assert_eq!(name, "only");
        assert_eq!(resolved.api_key, "k");
    }

    #[test]
    fn parse_full_opencode_catalog() {
        let json = include_str!("../../../config.example.json");
        let cfg = Config::from_json_str(json).unwrap();

        // All providers from OpenCode/ZeroClaw
        assert!(
            cfg.providers.len() >= 40,
            "expected 40+ providers, got {}",
            cfg.providers.len()
        );

        // Spot-check key providers
        let check = |name: &str, expected_type: &str, has_url: bool| {
            let pc = cfg
                .providers
                .get(name)
                .unwrap_or_else(|| panic!("missing provider: {name}"));
            assert_eq!(pc.provider_type, expected_type, "wrong type for {name}");
            if has_url {
                assert!(pc.base_url.is_some(), "missing base_url for {name}");
            }
        };

        check("anthropic", "anthropic", false);
        check("openai", "openai_compat", true);
        check("groq", "openai_compat", true);
        check("fireworks", "openai_compat", true);
        check("deepseek", "openai_compat", true);
        check("mistral", "openai_compat", true);
        check("xai", "openai_compat", true);
        check("together", "openai_compat", true);
        check("openrouter", "openai_compat", true);
        check("gemini", "openai_compat", true);
        check("minimax", "openai_compat", true);
        check("glm", "openai_compat", true);
        check("moonshot", "openai_compat", true);
        check("kimi-code", "openai_compat", true);
        check("qwen", "openai_compat", true);
        check("ollama", "openai_compat", true);
        check("nvidia", "openai_compat", true);
        check("perplexity", "openai_compat", true);
        check("cohere", "openai_compat", true);
        check("cerebras", "openai_compat", true);
        check("siliconflow", "openai_compat", true);
        check("telnyx", "openai_compat", true);
        check("azure-openai", "openai_compat", true);

        // Defaults
        assert_eq!(cfg.default_provider, "anthropic");
        assert_eq!(cfg.default_model, "claude-sonnet-4-20250514");

        // Fallbacks parsed
        let fb = cfg.fallback_providers();
        assert_eq!(fb.len(), 3);
        assert_eq!(fb[0].0, "groq");
        assert_eq!(fb[1].0, "deepseek");
        assert_eq!(fb[2].0, "openai");

        // Model lists
        let anthropic_models = &cfg.providers["anthropic"].models;
        assert!(anthropic_models.contains(&"claude-sonnet-4-20250514".to_string()));
        assert!(anthropic_models.contains(&"claude-opus-4-20250514".to_string()));

        let openai_models = &cfg.providers["openai"].models;
        assert!(openai_models.contains(&"gpt-4o".to_string()));
        assert!(openai_models.contains(&"o3".to_string()));

        let groq_models = &cfg.providers["groq"].models;
        assert!(groq_models.contains(&"llama-3.3-70b-versatile".to_string()));

        // Every provider can resolve with a direct key
        for (name, pc) in &cfg.providers {
            if !pc.api_key.starts_with('$') {
                let resolved = pc.resolved().unwrap();
                assert!(!resolved.api_key.is_empty(), "empty key for {name}");
            }
        }
    }

    #[test]
    fn all_provider_types_valid() {
        let json = include_str!("../../../config.example.json");
        let cfg = Config::from_json_str(json).unwrap();

        let valid_types = ["anthropic", "openai_compat"];
        for (name, pc) in &cfg.providers {
            assert!(
                valid_types.contains(&pc.provider_type.as_str()),
                "invalid provider_type '{}' for provider '{name}'",
                pc.provider_type
            );
        }
    }

    #[test]
    fn expand_tilde_home() {
        let home = dirs_home();
        assert_eq!(expand_tilde(Path::new("~")), home);
        assert_eq!(expand_tilde(Path::new("~/foo/bar")), home.join("foo/bar"));
        assert_eq!(
            expand_tilde(Path::new("/abs/path")),
            PathBuf::from("/abs/path")
        );
        assert_eq!(
            expand_tilde(Path::new("relative")),
            PathBuf::from("relative")
        );
    }

    #[test]
    fn skill_roots_tilde_expanded() {
        let json = r#"{"skill_roots": ["~/.zeroclaw/workspace/skills", "/absolute/path"]}"#;
        let mut cfg = Config::from_json_str(json).unwrap();
        cfg.skill_roots = cfg
            .skill_roots
            .into_iter()
            .map(|p| expand_tilde(&p))
            .collect();
        let home = dirs_home();
        assert_eq!(cfg.skill_roots[0], home.join(".zeroclaw/workspace/skills"));
        assert_eq!(cfg.skill_roots[1], PathBuf::from("/absolute/path"));
    }

    #[test]
    fn base_urls_are_valid() {
        let json = include_str!("../../../config.example.json");
        let cfg = Config::from_json_str(json).unwrap();

        for (name, pc) in &cfg.providers {
            if let Some(url) = &pc.base_url {
                assert!(
                    url.starts_with("http://") || url.starts_with("https://"),
                    "invalid base_url '{url}' for provider '{name}'"
                );
            }
        }
    }

    // ── SessionConfig + merge tests ─────────────────────────────────────

    #[test]
    fn session_config_parse_empty() {
        let sc: SessionConfig = serde_json::from_str("{}").unwrap();
        assert!(sc.default_provider.is_none());
        assert!(sc.default_model.is_none());
        assert!(sc.max_tokens.is_none());
        assert!(sc.temperature.is_none());
        assert!(sc.max_iterations.is_none());
        assert!(sc.mcp_servers.is_none());
        assert!(sc.skill_roots.is_none());
        assert!(sc.system_prompt_path.is_none());
    }

    #[test]
    fn session_config_parse_full() {
        let json = r#"{
            "default_provider": "anthropic",
            "default_model": "claude-sonnet-4",
            "max_tokens": 4096,
            "temperature": 0.5,
            "max_iterations": 20,
            "mcpServers": {"db": {"command": "db-server"}},
            "skill_roots": ["/my/skills"],
            "system_prompt_path": "./custom.md"
        }"#;
        let sc: SessionConfig = serde_json::from_str(json).unwrap();
        assert_eq!(sc.default_provider.as_deref(), Some("anthropic"));
        assert_eq!(sc.default_model.as_deref(), Some("claude-sonnet-4"));
        assert_eq!(sc.max_tokens, Some(4096));
        assert_eq!(sc.temperature, Some(0.5));
        assert_eq!(sc.max_iterations, Some(20));
        assert_eq!(sc.mcp_servers.as_ref().unwrap().len(), 1);
        assert_eq!(sc.skill_roots.as_ref().unwrap().len(), 1);
    }

    #[test]
    fn merge_empty_session_returns_global() {
        let cfg = Config {
            default_provider: "groq".into(),
            default_model: "llama".into(),
            max_tokens: 8192,
            temperature: Some(0.3),
            max_iterations: 30,
            ..Default::default()
        };
        let eff = cfg.merge_session(&SessionConfig::default());
        assert_eq!(eff.provider, "groq");
        assert_eq!(eff.model, "llama");
        assert_eq!(eff.max_tokens, 8192);
        assert_eq!(eff.temperature, Some(0.3));
        assert_eq!(eff.max_iterations, 30);
    }

    #[test]
    fn merge_session_overrides_fields() {
        let cfg = Config {
            default_provider: "groq".into(),
            default_model: "llama".into(),
            max_tokens: 8192,
            temperature: Some(0.3),
            max_iterations: 30,
            ..Default::default()
        };
        let sc = SessionConfig {
            default_provider: Some("anthropic".into()),
            default_model: Some("claude".into()),
            max_tokens: Some(2048),
            temperature: Some(0.9),
            max_iterations: Some(10),
            ..Default::default()
        };
        let eff = cfg.merge_session(&sc);
        assert_eq!(eff.provider, "anthropic");
        assert_eq!(eff.model, "claude");
        assert_eq!(eff.max_tokens, 2048);
        assert_eq!(eff.temperature, Some(0.9));
        assert_eq!(eff.max_iterations, 10);
    }

    #[test]
    fn merge_session_partial_override() {
        let cfg = Config {
            default_provider: "groq".into(),
            default_model: "llama".into(),
            max_tokens: 8192,
            max_iterations: 30,
            ..Default::default()
        };
        let sc = SessionConfig {
            default_model: Some("mixtral".into()),
            ..Default::default()
        };
        let eff = cfg.merge_session(&sc);
        assert_eq!(eff.provider, "groq");
        assert_eq!(eff.model, "mixtral");
        assert_eq!(eff.max_tokens, 8192);
        assert_eq!(eff.max_iterations, 30);
    }

    #[test]
    fn merge_mcp_servers_additive() {
        let mut global_mcp = HashMap::new();
        global_mcp.insert(
            "echo".into(),
            McpServerConfig {
                name: "echo".into(),
                command: "echo-server".into(),
                ..Default::default()
            },
        );
        let cfg = Config {
            mcp_servers: global_mcp,
            ..Default::default()
        };

        let mut session_mcp = HashMap::new();
        session_mcp.insert(
            "db".into(),
            McpServerConfig {
                name: "db".into(),
                command: "db-server".into(),
                ..Default::default()
            },
        );
        let sc = SessionConfig {
            mcp_servers: Some(session_mcp),
            ..Default::default()
        };

        let eff = cfg.merge_session(&sc);
        assert_eq!(eff.mcp_servers.len(), 2);
        assert!(eff.mcp_servers.contains_key("echo"));
        assert!(eff.mcp_servers.contains_key("db"));
    }

    #[test]
    fn merge_mcp_session_overrides_same_name() {
        let mut global_mcp = HashMap::new();
        global_mcp.insert(
            "server".into(),
            McpServerConfig {
                name: "server".into(),
                command: "global-cmd".into(),
                ..Default::default()
            },
        );
        let cfg = Config {
            mcp_servers: global_mcp,
            ..Default::default()
        };

        let mut session_mcp = HashMap::new();
        session_mcp.insert(
            "server".into(),
            McpServerConfig {
                name: "server".into(),
                command: "session-cmd".into(),
                ..Default::default()
            },
        );
        let sc = SessionConfig {
            mcp_servers: Some(session_mcp),
            ..Default::default()
        };

        let eff = cfg.merge_session(&sc);
        assert_eq!(eff.mcp_servers.len(), 1);
        assert_eq!(eff.mcp_servers["server"].command, "session-cmd");
    }

    #[test]
    fn merge_skill_roots_session_wins() {
        let cfg = Config {
            skill_roots: vec![PathBuf::from("/global/skills")],
            ..Default::default()
        };
        let sc = SessionConfig {
            skill_roots: Some(vec![PathBuf::from("/session/skills")]),
            ..Default::default()
        };
        let eff = cfg.merge_session(&sc);
        assert_eq!(eff.skill_roots, vec![PathBuf::from("/session/skills")]);
    }

    #[test]
    fn merge_skill_roots_fallback_to_global() {
        let cfg = Config {
            skill_roots: vec![PathBuf::from("/global/skills")],
            ..Default::default()
        };
        let eff = cfg.merge_session(&SessionConfig::default());
        assert_eq!(eff.skill_roots, vec![PathBuf::from("/global/skills")]);
    }

    #[test]
    fn session_config_from_file() {
        let tmp = tempfile::tempdir().unwrap();
        let path = tmp.path().join("config.json");
        std::fs::write(&path, r#"{"default_model": "gpt-4o", "max_tokens": 1024}"#).unwrap();
        let sc = SessionConfig::from_file(&path).unwrap();
        assert_eq!(sc.default_model.as_deref(), Some("gpt-4o"));
        assert_eq!(sc.max_tokens, Some(1024));
        assert!(sc.default_provider.is_none());
    }

    #[test]
    fn default_effective_matches_global() {
        let cfg = Config {
            default_provider: "test".into(),
            default_model: "m1".into(),
            max_tokens: 4000,
            temperature: Some(0.5),
            max_iterations: 25,
            ..Default::default()
        };
        let eff = cfg.default_effective();
        assert_eq!(eff.provider, "test");
        assert_eq!(eff.model, "m1");
        assert_eq!(eff.max_tokens, 4000);
        assert_eq!(eff.temperature, Some(0.5));
        assert_eq!(eff.max_iterations, 25);
    }

    #[test]
    fn validate_and_warn_never_panics_on_empty() {
        // Pure smoke test: a blank default-Config must not crash validation.
        // Warnings go to tracing; we just assert the function returns.
        let cfg = Config::default();
        cfg.validate_and_warn();
    }

    #[test]
    fn validate_and_warn_mismatched_default_provider_is_tolerated() {
        // Wrong `default_provider` must log and return, not panic.
        let mut providers = HashMap::new();
        providers.insert("real".to_string(), test_pc("sk"));
        let cfg = Config {
            providers,
            default_provider: "typo".into(),
            default_model: "m1".into(),
            ..Default::default()
        };
        cfg.validate_and_warn();
    }

    #[test]
    fn research_config_defaults_sane() {
        let rc = ResearchConfig::default();
        assert!(rc.enabled);
        assert_eq!(rc.max_iterations, 30);
        assert_eq!(rc.max_wall_seconds, 1200);
        assert!(rc.provider.is_none());
        assert!(rc.model.is_none());
        assert!(rc.default_sources.is_empty());
        assert!(rc.allowed_tools.is_none());
    }

    #[test]
    fn research_config_parses_from_json() {
        let json = r#"{
            "providers": {
                "qwen": {"type": "openai_compat", "api_key": "k", "models": ["qwen3.6-plus"]}
            },
            "research": {
                "enabled": true,
                "provider": "qwen",
                "model": "qwen3.6-plus",
                "max_iterations": 20,
                "max_wall_seconds": 900,
                "default_sources": ["https://chotot.com"]
            }
        }"#;
        let cfg = Config::from_json_str(json).unwrap();
        assert_eq!(cfg.research.provider.as_deref(), Some("qwen"));
        assert_eq!(cfg.research.model.as_deref(), Some("qwen3.6-plus"));
        assert_eq!(cfg.research.max_iterations, 20);
        assert_eq!(cfg.research.max_wall_seconds, 900);
        assert_eq!(cfg.research.default_sources.len(), 1);
    }

    #[test]
    fn research_config_missing_section_uses_defaults() {
        let cfg = Config::from_json_str(r#"{"providers": {}}"#).unwrap();
        assert!(cfg.research.enabled);
        assert_eq!(cfg.research.max_iterations, 30);
    }

    #[test]
    fn validate_and_warn_mismatched_research_provider_is_tolerated() {
        let mut providers = HashMap::new();
        providers.insert("qwen".to_string(), test_pc("k"));
        let cfg = Config {
            providers,
            default_provider: "qwen".into(),
            default_model: "qwen3.6-plus".into(),
            research: ResearchConfig {
                enabled: true,
                provider: Some("typo-provider".into()),
                model: Some("qwen3.6-plus".into()),
                ..Default::default()
            },
            ..Default::default()
        };
        // Must not panic — warns and returns.
        cfg.validate_and_warn();
    }

    #[test]
    fn validate_and_warn_mismatched_research_model_is_tolerated() {
        let mut pc = test_pc("k");
        pc.models = vec!["qwen3.6-plus".into()];
        let mut providers = HashMap::new();
        providers.insert("qwen".to_string(), pc);
        let cfg = Config {
            providers,
            default_provider: "qwen".into(),
            default_model: "qwen3.6-plus".into(),
            research: ResearchConfig {
                enabled: true,
                provider: Some("qwen".into()),
                model: Some("qwen-typo".into()),
                ..Default::default()
            },
            ..Default::default()
        };
        cfg.validate_and_warn();
    }

    #[test]
    fn validate_and_warn_missing_api_key_for_non_copilot_is_tolerated() {
        let mut providers = HashMap::new();
        providers.insert("openai".to_string(), test_pc(""));
        let cfg = Config {
            providers,
            default_provider: "openai".into(),
            default_model: "gpt-4o".into(),
            ..Default::default()
        };
        cfg.validate_and_warn();
    }

    // --- capability-aware validator tests -------------------------------
    //
    // These exercise `Config::validate_capabilities` which never panics and
    // only emits `tracing::warn!`. We can't inspect the tracing output
    // without pulling in a subscriber, so the tests focus on "does not
    // panic" + "does not misbehave across a variety of input shapes". The
    // hard guarantees (which specific warning fires) are locked in by the
    // integration tests in `tests/capabilities_validator.rs`, which capture
    // real tracing output.

    fn pc_with_caps(
        api_key: &str,
        models: Vec<&str>,
        caps: Vec<(&str, crate::model_catalog::ModelCapabilities)>,
    ) -> ProviderConfig {
        let mut pc = test_pc(api_key);
        pc.models = models.into_iter().map(String::from).collect();
        pc.capabilities = caps
            .into_iter()
            .map(|(k, v)| (k.to_string(), v))
            .collect();
        pc
    }

    #[test]
    fn validate_capabilities_silent_when_unknown_pair() {
        // No capabilities block for the referenced model → permissive
        // unknown() fallback, no warning.
        let pc = pc_with_caps("k", vec!["gpt-4o"], vec![]);
        let mut providers = HashMap::new();
        providers.insert("openai".into(), pc);
        let cfg = Config {
            providers,
            default_provider: "openai".into(),
            default_model: "gpt-4o".into(),
            ..Default::default()
        };
        cfg.validate_and_warn();
    }

    #[test]
    fn validate_capabilities_fires_on_deprecated_default_model() {
        use crate::model_catalog::{ModelCapabilities, ModelStatus};
        let mut caps = ModelCapabilities::unknown();
        caps.status = ModelStatus::Deprecated;
        let pc = pc_with_caps("k", vec!["qwen3.6-plus"], vec![("qwen3.6-plus", caps)]);
        let mut providers = HashMap::new();
        providers.insert("qwen".into(), pc);
        let cfg = Config {
            providers,
            default_provider: "qwen".into(),
            default_model: "qwen3.6-plus".into(),
            ..Default::default()
        };
        cfg.validate_and_warn();
    }

    #[test]
    fn validate_capabilities_fires_on_task_fit_miss() {
        use crate::model_catalog::{ModelCapabilities, TaskKind};
        let mut caps = ModelCapabilities::unknown();
        // Mark as chat-only, then reference from research.
        caps.task_fit = vec![TaskKind::Chat, TaskKind::Classify];
        let pc = pc_with_caps(
            "k",
            vec!["kimi-for-classify-only"],
            vec![("kimi-for-classify-only", caps)],
        );
        let mut providers = HashMap::new();
        providers.insert("kimi-code".into(), pc);
        let cfg = Config {
            providers,
            default_provider: "kimi-code".into(),
            default_model: "kimi-for-classify-only".into(),
            research: ResearchConfig {
                enabled: true,
                provider: Some("kimi-code".into()),
                model: Some("kimi-for-classify-only".into()),
                ..Default::default()
            },
            ..Default::default()
        };
        cfg.validate_and_warn();
    }

    #[test]
    fn validate_capabilities_fallback_chain_with_deprecated() {
        use crate::model_catalog::{ModelCapabilities, ModelStatus};
        let mut dead = ModelCapabilities::unknown();
        dead.status = ModelStatus::Deprecated;
        let pc = pc_with_caps(
            "k",
            vec!["live", "zombie"],
            vec![("zombie", dead)],
        );
        let mut providers = HashMap::new();
        providers.insert("myprov".into(), pc);
        let cfg = Config {
            providers,
            default_provider: "myprov".into(),
            default_model: "live".into(),
            fallback: vec!["myprov/zombie".into()],
            research: ResearchConfig {
                enabled: true,
                provider: Some("myprov".into()),
                model: Some("live".into()),
                fallback_models: vec!["myprov/zombie".into(), "zombie".into()],
                ..Default::default()
            },
            ..Default::default()
        };
        cfg.validate_and_warn();
    }

    #[test]
    fn validate_capabilities_degraded_model_referenced_from_research() {
        // The canonical case this whole phase exists to prevent:
        // `glm-5-turbo` pinned as a research backend.
        use crate::model_catalog::{ModelCapabilities, ModelStatus, TaskKind, ToolUseLevel};
        let caps = ModelCapabilities {
            status: ModelStatus::Degraded,
            task_fit: vec![TaskKind::Chat, TaskKind::Classify],
            tool_use: ToolUseLevel::TextOnly,
            known_failure_modes: vec!["empty_content".into()],
            ..ModelCapabilities::unknown()
        };
        let pc = pc_with_caps(
            "k",
            vec!["glm-5-turbo"],
            vec![("glm-5-turbo", caps)],
        );
        let mut providers = HashMap::new();
        providers.insert("zai".into(), pc);
        let cfg = Config {
            providers,
            default_provider: "zai".into(),
            default_model: "glm-5-turbo".into(),
            research: ResearchConfig {
                enabled: true,
                provider: Some("zai".into()),
                model: Some("glm-5-turbo".into()),
                ..Default::default()
            },
            ..Default::default()
        };
        cfg.validate_and_warn();
    }

    #[test]
    fn validate_capabilities_digest_provider_string_parses() {
        use crate::model_catalog::{ModelCapabilities, ModelStatus};
        let mut dead = ModelCapabilities::unknown();
        dead.status = ModelStatus::Deprecated;
        let pc = pc_with_caps(
            "k",
            vec!["gpt-3.5"],
            vec![("gpt-3.5", dead)],
        );
        let mut providers = HashMap::new();
        providers.insert("openai".into(), pc);
        let cfg = Config {
            providers,
            default_provider: "openai".into(),
            default_model: "gpt-3.5".into(),
            memory: MemoryConfig {
                digest_provider: Some("openai/gpt-3.5".into()),
                ..Default::default()
            },
            ..Default::default()
        };
        cfg.validate_and_warn();
    }

    #[test]
    fn gatekeeper_config_defaults_are_sane() {
        let gk = GatekeeperConfig::default();
        assert_eq!(gk.max_rounds, 3);
        assert_eq!(gk.min_excerpt_chars, 300);
        assert_eq!(gk.min_source_content_chars, 100);
        assert_eq!(gk.max_listing_age_days, 90);
        assert!(gk.require_listing_date);
        assert!(gk.require_source_content);
        assert!(gk.detect_semantic_duplicates);
        assert!(gk.stop_on_stagnation);
        assert_eq!(gk.url_check_timeout_secs, 10);
        assert!(gk.feedback_prompt_header.contains("{id}"));
        assert!(gk.feedback_prompt_header.contains("{topic}"));
    }

    #[test]
    fn gatekeeper_config_parses_from_json() {
        let json = r#"{
            "providers": {},
            "research": {
                "gatekeeper": {
                    "max_rounds": 5,
                    "min_excerpt_chars": 300,
                    "min_source_content_chars": 200,
                    "max_listing_age_days": 30,
                    "require_listing_date": false,
                    "require_source_content": false,
                    "detect_semantic_duplicates": false,
                    "stop_on_stagnation": false,
                    "url_check_timeout_secs": 15,
                    "save_warnings": {
                        "no_listing_date": "custom: add date",
                        "no_source_content": "custom: add source",
                        "short_excerpt": "custom: excerpt too short (need {min_excerpt})"
                    }
                }
            }
        }"#;
        let cfg = Config::from_json_str(json).unwrap();
        let gk = &cfg.research.gatekeeper;
        assert_eq!(gk.max_rounds, 5);
        assert_eq!(gk.min_excerpt_chars, 300);
        assert_eq!(gk.min_source_content_chars, 200);
        assert_eq!(gk.max_listing_age_days, 30);
        assert!(!gk.require_listing_date);
        assert!(!gk.require_source_content);
        assert!(!gk.detect_semantic_duplicates);
        assert!(!gk.stop_on_stagnation);
        assert_eq!(gk.url_check_timeout_secs, 15);
        assert_eq!(gk.save_warnings.no_listing_date, "custom: add date");
    }

    #[test]
    fn gatekeeper_config_missing_uses_defaults() {
        let cfg = Config::from_json_str(r#"{"providers": {}}"#).unwrap();
        let gk = &cfg.research.gatekeeper;
        assert_eq!(gk.max_rounds, 3);
        assert_eq!(gk.min_excerpt_chars, 300);
        assert!(gk.require_listing_date);
    }

    #[test]
    fn verify_by_default_default_is_true() {
        let cfg = Config::from_json_str(r#"{"providers": {}}"#).unwrap();
        assert!(cfg.research.verify_by_default);
        assert!(ResearchConfig::default().verify_by_default);
    }

    #[test]
    fn verify_by_default_can_be_disabled() {
        let json = r#"{
            "providers": {},
            "research": {
                "verify_by_default": false
            }
        }"#;
        let cfg = Config::from_json_str(json).unwrap();
        assert!(!cfg.research.verify_by_default);
        assert_eq!(cfg.research.gatekeeper.max_rounds, 3);
    }

    /// Contract test against the shipped `naked.json` at the repo root.
    /// Locks in the kimi-for-coding + reasoning=medium primary research
    /// configuration so a careless edit can't silently downgrade quality.
    /// Skips gracefully when the file isn't present (downstream consumers
    /// of naked-core may not ship naked.json).
    #[test]
    fn shipped_naked_json_research_uses_kimi_for_coding_with_reasoning() {
        let manifest = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        let candidate = manifest
            .parent()
            .and_then(|p| p.parent())
            .map(|p| p.join("naked.json"));
        let Some(path) = candidate.filter(|p| p.is_file()) else {
            eprintln!("skipping: naked.json not present at expected path");
            return;
        };
        let raw = std::fs::read_to_string(&path).expect("read naked.json");
        let cfg = Config::from_json_str(&raw).expect("parse naked.json");

        // Research must run on kimi-for-coding via kimi-code provider, with
        // reasoning_effort=medium (per the official Roo Code recipe).
        assert_eq!(cfg.research.provider.as_deref(), Some("kimi-code"));
        assert_eq!(cfg.research.model.as_deref(), Some("kimi-for-coding"));
        assert_eq!(cfg.research.reasoning.as_deref(), Some("medium"));

        // Cross-provider fallback chain must include qwen for resilience
        // when the kimi-code endpoint is throttled / down.
        let chain = &cfg.research.fallback_models;
        assert!(
            chain.iter().any(|m| m.starts_with("qwen/")),
            "fallback chain must include a qwen/* entry — got {chain:?}"
        );

        // The kimi-code provider must list kimi-for-coding among its models
        // so model-validation in `validate_and_warn` doesn't flag it.
        let kimi = cfg
            .providers
            .get("kimi-code")
            .expect("kimi-code provider must be configured");
        assert!(
            kimi.models.iter().any(|m| m == "kimi-for-coding"),
            "kimi-code provider must declare `kimi-for-coding` model"
        );
        assert_eq!(
            kimi.headers.get("User-Agent").map(|s| s.as_str()),
            Some("claude-code/1.0"),
            "kimi-code requires User-Agent: claude-code/1.0 — without it \
             api.kimi.com refuses kimi-for-coding with 403 \
             \"Kimi For Coding is currently only available for Coding Agents\""
        );
    }

    #[test]
    fn research_reasoning_default_is_none() {
        let cfg = Config::from_json_str(r#"{"providers": {}}"#).unwrap();
        assert!(cfg.research.reasoning.is_none());
        assert!(ResearchConfig::default().reasoning.is_none());
    }

    #[test]
    fn default_reasoning_cascades_into_session() {
        // Global default present, session leaves it unset → session inherits.
        let cfg =
            Config::from_json_str(r#"{"providers": {}, "default_reasoning": "medium"}"#).unwrap();
        let eff = cfg.merge_session(&SessionConfig::default());
        assert_eq!(
            eff.reasoning.as_deref(),
            Some("medium"),
            "session must inherit Config.default_reasoning when SessionConfig.reasoning is None"
        );
    }

    #[test]
    fn session_reasoning_overrides_global_default() {
        let cfg =
            Config::from_json_str(r#"{"providers": {}, "default_reasoning": "medium"}"#).unwrap();
        let session = SessionConfig {
            reasoning: Some("high".into()),
            ..SessionConfig::default()
        };
        let eff = cfg.merge_session(&session);
        assert_eq!(
            eff.reasoning.as_deref(),
            Some("high"),
            "explicit SessionConfig.reasoning must win over Config.default_reasoning"
        );
    }

    #[test]
    fn default_reasoning_absent_means_no_reasoning() {
        let cfg = Config::from_json_str(r#"{"providers": {}}"#).unwrap();
        let eff = cfg.merge_session(&SessionConfig::default());
        assert!(
            eff.reasoning.is_none(),
            "without Config.default_reasoning and without session override, \
             the effective config must carry no reasoning hint (legacy behaviour)"
        );
    }

    #[test]
    fn shipped_naked_json_sets_default_reasoning_medium() {
        // Contract test against the file we actually ship: every provider
        // we use (kimi, moonshot, qwen, ali_cp, openrouter, fireworks,
        // anthropic) must receive `reasoning="medium"` for *all* sessions
        // — not just research turns. Without this, the bot's interactive
        // chat would silently downgrade to non-thinking despite the user
        // configuring kimi-for-coding everywhere.
        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../naked.json");
        if !path.exists() {
            return;
        }
        let raw = std::fs::read_to_string(&path).unwrap();
        let cfg = Config::from_json_str(&raw).unwrap();
        assert_eq!(
            cfg.default_reasoning.as_deref(),
            Some("medium"),
            "naked.json must set default_reasoning=\"medium\" so qwen / \
             moonshot / kimi-for-coding all stream reasoning_content by \
             default — see the user request in 42189a5b about \"процесс \
             размышлений тоже выводим\""
        );
    }

    #[test]
    fn research_reasoning_can_be_set() {
        let json = r#"{
            "providers": {},
            "research": {
                "reasoning": "medium"
            }
        }"#;
        let cfg = Config::from_json_str(json).unwrap();
        assert_eq!(cfg.research.reasoning.as_deref(), Some("medium"));
    }

    #[test]
    fn chat_personas_absent_means_empty_map() {
        let cfg = Config::from_json_str(r#"{"providers": {}}"#).unwrap();
        assert!(
            cfg.chat_personas.is_empty(),
            "missing chat_personas in JSON must default to an empty map (legacy bots keep \
             behaving exactly like before — no persona, no overrides)"
        );
    }

    #[test]
    fn chat_personas_parses_negative_chat_id_keys() {
        // serde_json deserialises object keys as strings; ours are i64 — make
        // sure the negative supergroup id (most common case for the Income
        // chat) round-trips through the map.
        let json = r#"{
            "providers": {},
            "chat_personas": {
                "-5084292206": {
                    "name": "income",
                    "workspace": "/home/operator/.naked/channels/income"
                }
            }
        }"#;
        let cfg = Config::from_json_str(json).unwrap();
        let persona = cfg.chat_personas.get(&-5084292206_i64).expect(
            "negative i64 chat_id key from JSON string must be deserialised into HashMap<i64,_>",
        );
        assert_eq!(persona.name, "income");
        assert_eq!(
            persona.workspace,
            std::path::PathBuf::from("/home/operator/.naked/channels/income")
        );
        assert!(
            !persona.allow_slash_commands,
            "allow_slash_commands defaults to false so personas behave as natural-language-only \
             chats unless explicitly opted in"
        );
    }

    #[test]
    fn chat_persona_allow_slash_commands_opt_in() {
        let json = r#"{
            "providers": {},
            "chat_personas": {
                "12345": {
                    "name": "ops",
                    "workspace": "/tmp/ops",
                    "allow_slash_commands": true
                }
            }
        }"#;
        let cfg = Config::from_json_str(json).unwrap();
        let persona = cfg.chat_personas.get(&12345_i64).unwrap();
        assert!(persona.allow_slash_commands);
    }

    #[test]
    fn chat_persona_workspace_expanded_resolves_tilde() {
        // Mirrors `skill_roots_tilde_expanded` style: read the current
        // `$HOME` via `dirs_home()` instead of mutating process env (the
        // crate is `#![forbid(unsafe_code)]`, so `std::env::set_var`
        // is off-limits even inside tests).
        let home = dirs_home();
        let persona = ChatPersona {
            name: "x".into(),
            workspace: PathBuf::from("~/.naked/channels/x"),
            allow_slash_commands: false,
        };
        assert_eq!(persona.workspace_expanded(), home.join(".naked/channels/x"));
        // Absolute paths must pass through untouched.
        let abs = ChatPersona {
            name: "y".into(),
            workspace: PathBuf::from("/var/lib/y"),
            allow_slash_commands: false,
        };
        assert_eq!(abs.workspace_expanded(), PathBuf::from("/var/lib/y"));
    }

    #[test]
    fn config_load_expands_tilde_in_persona_workspaces() {
        // `Config::load()` runs `expand_tilde` over every persona workspace
        // exactly once (mirroring the `skill_roots`/`agent_dirs` treatment).
        // We exercise the same expansion path via direct manipulation
        // instead of touching the on-disk loader.
        let json = r#"{
            "providers": {},
            "chat_personas": {
                "-1": {
                    "name": "a",
                    "workspace": "~/.naked/channels/a"
                },
                "2": {
                    "name": "b",
                    "workspace": "/var/lib/b"
                }
            }
        }"#;
        let mut cfg = Config::from_json_str(json).unwrap();
        for persona in cfg.chat_personas.values_mut() {
            persona.workspace = expand_tilde(&persona.workspace);
        }
        let home = dirs_home();
        assert_eq!(
            cfg.chat_personas[&-1_i64].workspace,
            home.join(".naked/channels/a")
        );
        assert_eq!(
            cfg.chat_personas[&2_i64].workspace,
            PathBuf::from("/var/lib/b")
        );
    }

    #[test]
    fn gatekeeper_config_partial_override() {
        let json = r#"{
            "providers": {},
            "research": {
                "gatekeeper": {
                    "min_excerpt_chars": 500,
                    "require_listing_date": false
                }
            }
        }"#;
        let cfg = Config::from_json_str(json).unwrap();
        let gk = &cfg.research.gatekeeper;
        assert_eq!(gk.min_excerpt_chars, 500);
        assert!(!gk.require_listing_date);
        // other fields keep defaults
        assert_eq!(gk.max_rounds, 3);
        assert!(gk.require_source_content);
        assert!(gk.detect_semantic_duplicates);
    }
}
