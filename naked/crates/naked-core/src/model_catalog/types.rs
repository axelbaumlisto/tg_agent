//! Structured capability catalog for every `(provider, model)` pair
//! declared in `naked.json`. Lives alongside [`crate::config::ProviderConfig`]
//! but intentionally split out so the schema can grow (health tracking,
//! ranking, auto-enrichment) without turning `config.rs` into a 4000-line
//! god-file.
//!
//! Design principles:
//!
//! 1. **Per-pair, not per-model.** The same model id (`glm-5`, `claude-sonnet-4`)
//!    can live under several providers (`zai` native vs `openrouter` aggregator
//!    vs `copilot` proxy) with radically different latency / tool-use fidelity /
//!    failure modes. Capabilities therefore key off the `(provider, model)`
//!    tuple, stored inside `ProviderConfig.capabilities`.
//! 2. **Everything optional.** Legacy `naked.json` without capability blocks
//!    still parses. Missing entries fall back to `ModelCapabilities::unknown()`
//!    which is deliberately permissive ("allow for any task, unknown tier") so
//!    validation / selection degrades gracefully rather than locking every
//!    model out.
//! 3. **JSON ergonomics.** All enums serialize as lowercase `snake_case`
//!    strings. Operators type `"status": "degraded"`, not
//!    `"status": "Degraded"`.
//!
//! ```json
//! "providers": {
//!   "zai": {
//!     "type": "anthropic",
//!     "models": ["glm-5", "glm-5-turbo"],
//!     "capabilities": {
//!       "glm-5": {
//!         "status": "active",
//!         "task_fit": ["chat", "digest", "classify"],
//!         "tool_use": "full",
//!         "quality_tier": "a",
//!         "latency_tier": "medium",
//!         "cost_tier": "cheap"
//!       },
//!       "glm-5-turbo": {
//!         "status": "degraded",
//!         "task_fit": ["chat", "classify"],
//!         "tool_use": "text_only",
//!         "known_failure_modes": ["empty_content"],
//!         "latency_tier": "fast",
//!         "cost_tier": "cheap",
//!         "notes": "Closes stream with zero content under load; avoid for research"
//!       }
//!     }
//!   }
//! }
//! ```

use serde::{Deserialize, Serialize};

/// Deployment status of a `(provider, model)` pair. The selector consults
/// this first — `Deprecated` and `Experimental` are never auto-selected;
/// `Degraded` is only picked when the caller explicitly allows it via the
/// `Budget.allow_degraded` flag.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ModelStatus {
    /// Default. Healthy, use freely.
    #[default]
    Active,
    /// Known issues that don't kill the model but should bias selection
    /// away from it when equivalent alternatives exist (e.g. `glm-5-turbo`
    /// empty-content). Still eligible if caller opts in.
    Degraded,
    /// Not to be used — removed upstream or replaced. The `/model` menu
    /// hides these; selection refuses them even if pinned.
    Deprecated,
    /// Newly added, not yet field-tested. Allowed but never chosen
    /// automatically; operator must pin explicitly.
    Experimental,
}

/// Task categories the agent picks a model for. Stored as a `Vec` on each
/// capability entry — a model that only fits `[Chat, Classify]` is filtered
/// out of research / digest chains before the first request is sent.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TaskKind {
    /// Interactive chat turn in Telegram / CLI. Default surface.
    Chat,
    /// `/research` long-running agent loop with tool-use pressure.
    Research,
    /// Memory-digest summarization job (cheap, short input, needs
    /// reasonable summarization quality — no tool use required).
    Digest,
    /// Auto-classify of user messages into memory entries. Very cheap,
    /// structured-output, no tool use.
    Classify,
    /// Multi-file coding / refactor tasks (Aider-style agent loops).
    Coding,
    /// Vision input (photos, screenshots). Requires `supports_vision=true`.
    Vision,
}

impl TaskKind {
    /// Every known task kind. Used by [`ModelCapabilities::unknown`] to
    /// populate a maximally-permissive default when no caps block was
    /// supplied.
    pub const ALL: &'static [TaskKind] = &[
        TaskKind::Chat,
        TaskKind::Research,
        TaskKind::Digest,
        TaskKind::Classify,
        TaskKind::Coding,
        TaskKind::Vision,
    ];
}

impl std::fmt::Display for TaskKind {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let s = match self {
            TaskKind::Chat => "chat",
            TaskKind::Research => "research",
            TaskKind::Digest => "digest",
            TaskKind::Classify => "classify",
            TaskKind::Coding => "coding",
            TaskKind::Vision => "vision",
        };
        f.write_str(s)
    }
}

/// How rich the model's tool-calling surface is. Research / coding loops
/// demand `Full`; digest / classify are fine with `None`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ToolUseLevel {
    /// Native function-calling that correctly interleaves thought, tool
    /// calls and their results. Required for research/coding.
    #[default]
    Full,
    /// Accepts tool schemas but frequently skips calls, returns them as
    /// prose, or breaks on multi-turn tool dialogues. OK for a single
    /// one-shot tool bloop, not for an agent loop.
    TextOnly,
    /// Plain text only — cannot call tools at all. Fine for digest /
    /// classify / summarize.
    None,
}

/// Reasoning / "thinking" capability. Maps onto
/// [`crate::provider::openai_compat::apply_reasoning_params`] and
/// Anthropic extended-thinking budgets. Higher levels cost more tokens.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ReasoningLevel {
    /// No `reasoning_effort` hint sent.
    Off,
    /// `reasoning_effort="low"`.
    Low,
    /// `reasoning_effort="medium"` — the default for most production tasks.
    #[default]
    Medium,
    /// `reasoning_effort="high"` — reserve for planning / hard research.
    High,
}

/// Coarse quality band — used by the selector to rank ties. Numeric
/// ordering is `S > A > B > C`. `None` = unknown / not yet calibrated.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum QualityTier {
    /// Best reasoning available. Claude Opus 4+, Claude Sonnet 4, GPT-5.
    S,
    /// Production-grade. `kimi-for-coding`, `qwen3.5-plus`, `glm-5`.
    A,
    /// Workhorse / commoditized. `glm-5-turbo`, `kimi-k2.5`.
    B,
    /// Legacy or ultra-cheap. `qwen-turbo`, free OpenRouter models.
    C,
}

/// Coarse latency band — time-to-first-token under normal load.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum LatencyTier {
    /// Sub-second TTFT (GLM-turbo, Qwen-turbo, Groq, Claude Haiku).
    Fast,
    /// 1-3 s TTFT (typical chat models).
    Medium,
    /// >3 s TTFT or very variable (thinking models, long-ctx on cold
    /// infra, OpenRouter routing hops).
    Slow,
}

/// Coarse cost band. Selector downranks `Premium` unless the caller's
/// budget permits it.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CostTier {
    /// No per-request charge (e.g. OpenRouter `:free` routes under their
    /// free tier, Copilot proxy via the IDE subscription).
    Free,
    /// Roughly ≤ $0.50 per million input tokens.
    Cheap,
    /// $0.50 – $3.00 / Mtok input.
    Medium,
    /// > $3.00 / Mtok input (Claude Opus, GPT-4.5, Gemini Ultra).
    Premium,
}

/// Everything we know about a single `(provider, model)` pair, keyed by
/// model id under [`crate::config::ProviderConfig::capabilities`].
///
/// All fields are `#[serde(default)]` / `Option` so a sparse entry like
/// `{ "status": "deprecated" }` parses fine and every omitted field keeps
/// the permissive default from [`ModelCapabilities::unknown`].
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelCapabilities {
    #[serde(default)]
    pub status: ModelStatus,

    /// Which [`TaskKind`]s this model is approved for. When empty on
    /// disk, deserialized as "all" via [`ModelCapabilities::unknown`] —
    /// `#[serde(default = "TaskKind_all")]` would be equivalent but less
    /// explicit.
    #[serde(default = "task_fit_all")]
    pub task_fit: Vec<TaskKind>,

    #[serde(default)]
    pub tool_use: ToolUseLevel,

    #[serde(default)]
    pub reasoning: ReasoningLevel,

    #[serde(default)]
    pub supports_thinking: bool,

    /// Per-model vision capability override. `Some(true)` wins over
    /// [`crate::config::ProviderConfig::supports_vision`]; `None` delegates
    /// to the provider-wide flag + built-in needle list in
    /// [`crate::config::TgMediaConfig::is_vision_capable_with_provider`].
    #[serde(default)]
    pub supports_vision: Option<bool>,

    /// Per-model context-window override in tokens. `None` delegates to
    /// `ProviderConfig.context_window`, then the global fallback.
    #[serde(default)]
    pub context_window: Option<u32>,

    /// Per-model max output tokens. `None` delegates to `provider.max_tokens`
    /// then `Config.max_tokens`.
    #[serde(default)]
    pub max_output_tokens: Option<u32>,

    /// `None` = not yet calibrated. Selector treats unknown tiers as
    /// middling (between `A` and `B`).
    #[serde(default)]
    pub quality_tier: Option<QualityTier>,

    #[serde(default)]
    pub latency_tier: Option<LatencyTier>,

    #[serde(default)]
    pub cost_tier: Option<CostTier>,

    /// Free-form tags like `"empty_content"`, `"401_key_expired"`,
    /// `"content_filter_chinese"`, `"truncates_over_16k"`. Surfaced in
    /// startup warns and selector reasoning logs, never consumed by the
    /// selector itself (kept free-form so operators can invent new
    /// labels without schema churn).
    #[serde(default)]
    pub known_failure_modes: Vec<String>,

    /// Human-readable prose for operators skimming the catalog. Preserved
    /// verbatim by the exporter into generated skill sections.
    #[serde(default)]
    pub notes: Option<String>,
}

fn task_fit_all() -> Vec<TaskKind> {
    TaskKind::ALL.to_vec()
}

impl Default for ModelCapabilities {
    fn default() -> Self {
        Self::unknown()
    }
}

impl ModelCapabilities {
    /// Permissive default used for pairs that have no capability block in
    /// `naked.json`. Status `Active`, every task allowed, `Full` tool use,
    /// no tier information. Equivalent to "we don't know anything about
    /// this model, so don't filter it out".
    pub fn unknown() -> Self {
        Self {
            status: ModelStatus::Active,
            task_fit: TaskKind::ALL.to_vec(),
            tool_use: ToolUseLevel::Full,
            reasoning: ReasoningLevel::Medium,
            supports_thinking: false,
            supports_vision: None,
            context_window: None,
            max_output_tokens: None,
            quality_tier: None,
            latency_tier: None,
            cost_tier: None,
            known_failure_modes: Vec::new(),
            notes: None,
        }
    }

    /// True if this pair is eligible for the given task kind based on
    /// `status` and `task_fit`. Does NOT consult runtime health (that
    /// lives in [`crate::model_catalog::health`], phase 3).
    pub fn fits(&self, task: TaskKind) -> bool {
        match self.status {
            ModelStatus::Deprecated => false,
            ModelStatus::Experimental => {
                // Experimental never auto-selected; caller must pin it,
                // in which case `fits` isn't consulted. We conservatively
                // return false here so the selector doesn't rank it into
                // fallback chains.
                false
            }
            ModelStatus::Active | ModelStatus::Degraded => {
                self.task_fit.iter().any(|t| *t == task)
            }
        }
    }

    /// True if the model has at least `TextOnly` tool support (useful for
    /// classify where we accept structured-output in prose).
    pub fn can_use_tools_at_all(&self) -> bool {
        matches!(self.tool_use, ToolUseLevel::Full | ToolUseLevel::TextOnly)
    }

    /// True if the model can participate in an agent loop (research /
    /// coding / interactive chat with real tools).
    pub fn supports_agent_loop(&self) -> bool {
        matches!(self.tool_use, ToolUseLevel::Full)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn unknown_default_is_permissive() {
        let caps = ModelCapabilities::unknown();
        assert_eq!(caps.status, ModelStatus::Active);
        assert_eq!(caps.task_fit.len(), TaskKind::ALL.len());
        assert!(caps.fits(TaskKind::Research));
        assert!(caps.fits(TaskKind::Chat));
        assert!(caps.fits(TaskKind::Vision));
        assert!(caps.supports_agent_loop());
    }

    #[test]
    fn deprecated_fits_nothing() {
        let mut caps = ModelCapabilities::unknown();
        caps.status = ModelStatus::Deprecated;
        for task in TaskKind::ALL {
            assert!(!caps.fits(*task), "deprecated should fail for {task}");
        }
    }

    #[test]
    fn experimental_fits_nothing_auto() {
        let mut caps = ModelCapabilities::unknown();
        caps.status = ModelStatus::Experimental;
        for task in TaskKind::ALL {
            assert!(!caps.fits(*task));
        }
    }

    #[test]
    fn task_fit_restricts_eligibility() {
        let mut caps = ModelCapabilities::unknown();
        caps.task_fit = vec![TaskKind::Chat, TaskKind::Classify];
        assert!(caps.fits(TaskKind::Chat));
        assert!(caps.fits(TaskKind::Classify));
        assert!(!caps.fits(TaskKind::Research));
        assert!(!caps.fits(TaskKind::Digest));
    }

    #[test]
    fn serde_roundtrip_full() {
        let caps = ModelCapabilities {
            status: ModelStatus::Degraded,
            task_fit: vec![TaskKind::Chat],
            tool_use: ToolUseLevel::TextOnly,
            reasoning: ReasoningLevel::Low,
            supports_thinking: true,
            supports_vision: Some(false),
            context_window: Some(200_000),
            max_output_tokens: Some(4096),
            quality_tier: Some(QualityTier::B),
            latency_tier: Some(LatencyTier::Fast),
            cost_tier: Some(CostTier::Cheap),
            known_failure_modes: vec!["empty_content".into(), "trunc_16k".into()],
            notes: Some("Only for throwaway turns".into()),
        };
        let j = serde_json::to_string(&caps).unwrap();
        let back: ModelCapabilities = serde_json::from_str(&j).unwrap();
        assert_eq!(back.status, caps.status);
        assert_eq!(back.task_fit, caps.task_fit);
        assert_eq!(back.tool_use, caps.tool_use);
        assert_eq!(back.known_failure_modes, caps.known_failure_modes);
        assert_eq!(back.quality_tier, caps.quality_tier);
    }

    #[test]
    fn sparse_entry_gets_permissive_defaults() {
        // Only `status` specified. Everything else should default
        // to "unknown" i.e. permissive.
        let j = json!({ "status": "deprecated" });
        let caps: ModelCapabilities = serde_json::from_value(j).unwrap();
        assert_eq!(caps.status, ModelStatus::Deprecated);
        assert_eq!(caps.task_fit.len(), TaskKind::ALL.len());
        assert_eq!(caps.tool_use, ToolUseLevel::Full);
        assert!(caps.known_failure_modes.is_empty());
    }

    #[test]
    fn enum_json_shape_is_snake_case() {
        let caps = ModelCapabilities {
            status: ModelStatus::Degraded,
            tool_use: ToolUseLevel::TextOnly,
            ..ModelCapabilities::unknown()
        };
        let v = serde_json::to_value(&caps).unwrap();
        assert_eq!(v["status"], "degraded");
        assert_eq!(v["tool_use"], "text_only");
    }

    #[test]
    fn task_fit_deserializes_snake_case() {
        let j = json!({
            "status": "active",
            "task_fit": ["chat", "research", "classify"],
        });
        let caps: ModelCapabilities = serde_json::from_value(j).unwrap();
        assert!(caps.fits(TaskKind::Chat));
        assert!(caps.fits(TaskKind::Research));
        assert!(!caps.fits(TaskKind::Digest));
        assert!(!caps.fits(TaskKind::Coding));
    }

    #[test]
    fn quality_tier_ordering() {
        assert!(QualityTier::S < QualityTier::A);
        assert!(QualityTier::A < QualityTier::B);
        assert!(QualityTier::B < QualityTier::C);
    }
}
