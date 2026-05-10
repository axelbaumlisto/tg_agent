//! Generic agent-role + task + batch primitives.
//!
//! Goal: keep one mechanism for "drive an LLM agent on a focused task,
//! with a chosen tool subset, then validate the output". Captcha bypass,
//! contact extraction, fact-checking, summarisation — all become
//! configurations of these primitives, never bespoke Rust code.
//!
//! Concepts (smallest possible API):
//!
//! * [`AgentRole`] — recipe: name, system prompt, model override,
//!   default skills, max iterations, tool filter.
//! * [`Task`] — instance: id + role + prompt + per-task wall-clock cap.
//! * [`TaskOutput`] — result: stop reason, elapsed, drained text,
//!   per-tool call counts, captcha-marker hits, skill loads, errors.
//! * [`Validator`] — trait: takes a [`TaskOutput`] and returns a
//!   [`ValidationVerdict`] (passed + reasons).
//! * [`run_batch`] — runs a `Vec<Task>` in parallel, bounded by the
//!   process-wide research-run semaphore (so we never exceed the
//!   concurrent-browser cap).
//!
//! What is *NOT* here: pipelines, phases, retries, DAGs. Those live
//! one layer up (still TBD). MVP keeps everything flat — a batch is
//! just `Vec<Task>` and validation is a separate post-processing
//! step. We grow it when there's a real second use case.

use std::collections::BTreeMap;
use std::path::PathBuf;

use async_trait::async_trait;
use serde::{Deserialize, Serialize};

/// Tool inclusion policy for a role. The host (currently `AgentCore`)
/// owns the full tool registry; the role just declares which subset
/// the LLM is allowed to call. Default is `AllowAll` so a role with
/// no opinions still works.
///
/// Why an enum and not a `Vec<String>`: most roles want either "all"
/// (orchestrator) or "all except dangerous ones" (read-only research).
/// Spelling out the full whitelist for every role would be DRY-hostile
/// and force every new tool to update every role config.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum ToolFilter {
    /// Expose every tool the host has registered. Use for orchestrator
    /// roles (`web_researcher`) where the LLM should pick freely.
    #[default]
    AllowAll,
    /// Expose only the named tools. Anything else is invisible to the
    /// LLM. Use for narrow roles (`browser_extractor` = browser_* +
    /// web_fetch + research_save only).
    Allow { tools: Vec<String> },
    /// Expose every tool **except** the named ones. Use for read-only
    /// roles that should not be able to write files / run shell.
    Deny { tools: Vec<String> },
}

impl ToolFilter {
    /// Returns `true` iff this filter would expose `tool_name` to the
    /// LLM. Pure function — host calls it once per tool when building
    /// the per-task `ToolRegistry`.
    pub fn allows(&self, tool_name: &str) -> bool {
        match self {
            Self::AllowAll => true,
            Self::Allow { tools } => tools.iter().any(|t| t == tool_name),
            Self::Deny { tools } => !tools.iter().any(|t| t == tool_name),
        }
    }
}

/// Browser-runtime knobs that turn the difference between "captcha
/// every page" and "no captcha at all". Both fields are optional and
/// default to "use whatever the playwright MCP defaults to". When
/// either is set the role-runner overrides the MCP env at spawn time.
///
/// Backed by external services:
///
/// * `proxy` → residential / mobile proxy URL (Webshare, Proxyon,
///   BrightData). Reddit consensus: this is the bigger lever — most
///   captchas are triggered by datacenter-IP fingerprinting, not by
///   actual bot detection. Fixing the IP makes captchas rarely
///   appear in the first place.
/// * `extensions` → Chromium `--load-extension` paths (e.g. local
///   CapSolver extension dir). Solves the captchas that *do* appear
///   by injecting solver-API tokens into the page invisibly.
/// * `extension_env` → env vars exported to the Chromium process
///   (e.g. `CAPSOLVER_API_KEY=...`). Kept as a `BTreeMap` so the
///   serialised form is deterministic.
///
/// All of this is *configuration*. The agent-role machinery does not
/// know what CapSolver is — it just plumbs paths and env into the
/// browser MCP. New solvers / proxy providers do not require Rust
/// changes.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct BrowserRuntime {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub proxy: Option<String>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub extensions: Vec<String>,
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub extension_env: BTreeMap<String, String>,
}

impl BrowserRuntime {
    /// Returns `true` iff any field is set. Used by the host to
    /// decide whether to bother applying overrides at all.
    pub fn is_configured(&self) -> bool {
        self.proxy.is_some() || !self.extensions.is_empty() || !self.extension_env.is_empty()
    }
}

/// A reusable agent recipe. Built once (typically from `naked.json`
/// or via a constructor in [`crate::agent_role::registry`]) and shared
/// across many tasks via `Arc<AgentRole>`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentRole {
    /// Stable identifier surfaced in CLI / logs / metrics. Lower-snake-case.
    pub name: String,
    /// One-line summary for `naked agent list` / docs. Optional.
    #[serde(default)]
    pub description: String,
    /// Path (relative to the role's directory or absolute) to the
    /// markdown file containing the system prompt body. The
    /// [`store::AgentStore`] loader reads this and populates the
    /// runtime `system` field below.
    ///
    /// Optional so the in-memory `AgentRole::new(name, system)` builder
    /// path keeps working for tests / programmatic construction; data-
    /// driven role definitions ship the prompt as a sibling
    /// `prompt.md` and reference it from `role.json`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub prompt_file: Option<PathBuf>,
    /// Path to the default acceptance-criteria template (with
    /// `{topic}` / other placeholders). Read at load time into
    /// `default_criteria` so the gatekeeper validator can pick it up
    /// without the CLI having to ship a hardcoded template.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub criteria_file: Option<PathBuf>,
    /// Full system prompt. Either set inline by `AgentRole::new` or
    /// hydrated from `prompt_file` by the loader. May contain
    /// `{topic}` / `{urls}` placeholders that the host expands per
    /// task. We keep the templating dumb on purpose — single-pass
    /// `String::replace`, no Handlebars.
    #[serde(default)]
    pub system: String,
    /// Default acceptance criteria template, hydrated from
    /// `criteria_file` at load time. The CLI / coordinator picks this
    /// up when no `--criteria` override is supplied. Skipped from
    /// serialisation: it's a derived runtime field, not a config knob
    /// (the source of truth is the `criteria_file` text).
    #[serde(skip)]
    pub default_criteria: Option<String>,
    /// Optional model override. `None` ⇒ use the global default model
    /// (or the [`crate::config::Config`]-level role override if present).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub model: Option<String>,
    /// Skills auto-loaded into the prompt before the first turn. Names
    /// must be resolvable by the active [`crate::skill::resolver::SkillResolver`].
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub default_skills: Vec<String>,
    /// Validators applied to the [`TaskOutput`] when the operator
    /// doesn't explicitly pick a set. Lives on the role on purpose:
    /// validation is part of the role's acceptance contract (DRY —
    /// CLI / TG / scheduler all pick this up automatically), and
    /// orchestrators stay dumb (`if --validators absent { use
    /// role.default_validators }`). Empty ⇒ host fallback (currently
    /// `["gatekeeper"]`).
    ///
    /// Names must match registered validator ids: `gatekeeper`,
    /// `phone-vn`, `no-fake-contact`, `no-captcha`. Unknown names
    /// abort with a clear error so a typo here doesn't silently
    /// disable validation.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub default_validators: Vec<String>,
    /// Hard cap on tool-use loops. `0` ⇒ host default (currently 30).
    #[serde(default)]
    pub max_iters: usize,
    /// Tool subset policy.
    #[serde(default)]
    pub tool_filter: ToolFilter,
    /// Browser-runtime knobs (proxy / extensions / env). Default is empty.
    #[serde(
        default,
        skip_serializing_if = "BrowserRuntime::is_configured_serde_skip"
    )]
    pub browser_runtime: BrowserRuntime,
}

impl BrowserRuntime {
    // serde-callable wrapper around `is_configured` that returns true
    // when the field should be SKIPPED — i.e. when it is empty. Keeps
    // the JSON output free of `{"browser_runtime":{"extension_env":{}}}`
    // noise on the 99% of roles that don't configure a browser.
    #[doc(hidden)]
    pub fn is_configured_serde_skip(b: &BrowserRuntime) -> bool {
        !b.is_configured()
    }
}

impl AgentRole {
    /// Construct a minimal role with sensible defaults. Use builder
    /// methods to override.
    ///
    /// ```
    /// use naked_core::agent_role::AgentRole;
    /// let r = AgentRole::new("hello", "You are a friendly bot.");
    /// assert_eq!(r.name, "hello");
    /// assert!(r.default_skills.is_empty());
    /// ```
    pub fn new(name: impl Into<String>, system: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            description: String::new(),
            prompt_file: None,
            criteria_file: None,
            system: system.into(),
            default_criteria: None,
            model: None,
            default_skills: Vec::new(),
            default_validators: Vec::new(),
            max_iters: 0,
            tool_filter: ToolFilter::AllowAll,
            browser_runtime: BrowserRuntime::default(),
        }
    }

    pub fn with_description(mut self, desc: impl Into<String>) -> Self {
        self.description = desc.into();
        self
    }

    pub fn with_model(mut self, model: impl Into<String>) -> Self {
        self.model = Some(model.into());
        self
    }

    pub fn with_skills(mut self, skills: Vec<String>) -> Self {
        self.default_skills = skills;
        self
    }

    pub fn with_validators(mut self, validators: Vec<String>) -> Self {
        self.default_validators = validators;
        self
    }

    pub fn with_max_iters(mut self, n: usize) -> Self {
        self.max_iters = n;
        self
    }

    pub fn with_tool_filter(mut self, f: ToolFilter) -> Self {
        self.tool_filter = f;
        self
    }

    pub fn with_browser_runtime(mut self, br: BrowserRuntime) -> Self {
        self.browser_runtime = br;
        self
    }
}

/// One concrete unit of work for an [`AgentRole`].
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Task {
    /// Caller-supplied id, surfaced in [`TaskOutput::task_id`] and logs.
    /// If empty, the host generates a UUID-based one.
    #[serde(default)]
    pub id: String,
    /// Role name to dispatch to. The host resolves this against the
    /// role registry at run time — keeping the JSON schema cheap.
    pub role: String,
    /// User prompt (the "what to do"). Concatenated after the role's
    /// `system` field; placeholders in `system` are expanded against
    /// `context` first.
    pub prompt: String,
    /// Free-form structured context: URLs, hints, prior outputs.
    /// Available to the role-runner via placeholder expansion (see
    /// [`expand_placeholders`]).
    #[serde(default)]
    pub context: serde_json::Value,
    /// Optional wall-clock cap. `None` ⇒ host default
    /// (CoordinatorConfig::default_max_wall_seconds — currently 600s).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_wall_secs: Option<u64>,
}

impl Task {
    pub fn new(role: impl Into<String>, prompt: impl Into<String>) -> Self {
        Self {
            id: String::new(),
            role: role.into(),
            prompt: prompt.into(),
            context: serde_json::Value::Null,
            max_wall_secs: None,
        }
    }

    pub fn with_id(mut self, id: impl Into<String>) -> Self {
        self.id = id.into();
        self
    }

    pub fn with_context(mut self, ctx: serde_json::Value) -> Self {
        self.context = ctx;
        self
    }

    pub fn with_max_wall(mut self, secs: u64) -> Self {
        self.max_wall_secs = Some(secs);
        self
    }
}

/// Why the agent stopped. Mirrors `research::StopReason` so
/// downstream consumers (gatekeeper, probe summary, batch report)
/// can reason about it uniformly.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum StopReason {
    /// Agent emitted no new content for one full turn — natural
    /// completion.
    AgentIdle,
    /// Wall-clock cap (`Task::max_wall_secs`) elapsed before the agent
    /// went idle.
    Timeout,
    /// The provider stream closed unexpectedly.
    StreamClosed,
    /// External cancellation (Ctrl-C, parent gave up).
    Cancelled,
    /// Provider / tool / serialization error.
    Error,
}

/// Per-task observability bag. Produced by the host's drain loop;
/// consumed by validators, the batch-report printer, and tests.
///
/// Field choices are deliberately the same shape as the ad-hoc
/// `DrainStats` in `research/coordinator.rs` so we can converge on a
/// single struct in a follow-up.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct TaskStats {
    /// `tool_name → invocation count`. BTreeMap so Debug / serde
    /// output is deterministic (test-friendly).
    pub tools: BTreeMap<String, u32>,
    /// Number of tool outputs that contained any anti-bot marker
    /// (`xac-thuc`, `Just a moment`, `Enable JavaScript`, etc.).
    /// Surface metric for "is the captcha mitigation working".
    pub captcha_hits: u32,
    /// Number of times the agent invoked the `Skill` tool. A non-zero
    /// value tells operators the role's `default_skills` worked OR
    /// the agent autonomously loaded one.
    pub skill_loads: u32,
    /// Total streamed text deltas. Rough proxy for "did the agent
    /// say anything at all".
    pub text_deltas: u32,
    /// Tool calls whose state was Error.
    pub errors: u32,
}

impl TaskStats {
    /// Produce the same single-line summary the research probe uses,
    /// so existing greppers keep working unchanged.
    pub fn summary_line(&self) -> String {
        let tools = self
            .tools
            .iter()
            .map(|(k, v)| format!("{k}={v}"))
            .collect::<Vec<_>>()
            .join(",");
        format!(
            "[task-summary] tools={{{tools}}} captcha_hits={} skill_loads={} text_deltas={} errors={}",
            self.captcha_hits, self.skill_loads, self.text_deltas, self.errors,
        )
    }
}

/// What a single [`Task`] produced.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TaskOutput {
    pub task_id: String,
    pub role_name: String,
    pub stop_reason: StopReason,
    /// Wall-clock seconds the host actually spent, including model
    /// thinking + tool roundtrips + streaming.
    pub elapsed_secs: f64,
    /// All streamed assistant text concatenated, in order. May be
    /// empty if the agent only used tools and never spoke.
    pub text: String,
    pub stats: TaskStats,
    /// Free-form structured side-output. The host populates this
    /// from anything role-specific (e.g. saved findings, extracted
    /// JSON blobs). Default `Null`.
    #[serde(default)]
    pub artifacts: serde_json::Value,
}

impl TaskOutput {
    /// Convenience constructor for tests / mock runners.
    pub fn skeleton(task_id: impl Into<String>, role: impl Into<String>) -> Self {
        Self {
            task_id: task_id.into(),
            role_name: role.into(),
            stop_reason: StopReason::AgentIdle,
            elapsed_secs: 0.0,
            text: String::new(),
            stats: TaskStats::default(),
            artifacts: serde_json::Value::Null,
        }
    }
}

/// Decision a [`Validator`] returns about a single output.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ValidationVerdict {
    /// Validator name (so callers can attribute reasons when chaining
    /// multiple validators).
    pub validator: String,
    pub passed: bool,
    /// Per-validator human-readable reasons. Empty when `passed` is
    /// true and the validator has nothing to say.
    #[serde(default)]
    pub reasons: Vec<String>,
}

impl ValidationVerdict {
    pub fn pass(validator: impl Into<String>) -> Self {
        Self {
            validator: validator.into(),
            passed: true,
            reasons: Vec::new(),
        }
    }

    pub fn fail(validator: impl Into<String>, reason: impl Into<String>) -> Self {
        Self {
            validator: validator.into(),
            passed: false,
            reasons: vec![reason.into()],
        }
    }
}

/// Strategy for inspecting a [`TaskOutput`] and accepting / rejecting it.
/// Implementations are stateless — feel free to share via `Arc`.
#[async_trait]
pub trait Validator: Send + Sync {
    /// Stable identifier surfaced in [`ValidationVerdict::validator`].
    fn name(&self) -> &str;
    /// Inspect the output and return a verdict. Pure function in the
    /// sense that it doesn't mutate the output — but it MAY perform
    /// I/O (e.g. fetch a URL to confirm it's reachable). Hosts must
    /// budget for this when running validators in batch.
    async fn validate(&self, output: &TaskOutput) -> ValidationVerdict;
}

/// Expand `{key}` placeholders in `template` against `context`. Only
/// top-level scalar fields of a JSON object are substituted; arrays
/// and nested objects are JSON-stringified. Missing keys are left
/// as-is so a typo in the template is visible in the agent's prompt
/// rather than silently expanding to "".
pub fn expand_placeholders(template: &str, context: &serde_json::Value) -> String {
    let Some(obj) = context.as_object() else {
        return template.to_string();
    };
    let mut out = template.to_string();
    for (k, v) in obj {
        let placeholder = format!("{{{k}}}");
        if !out.contains(&placeholder) {
            continue;
        }
        let replacement = match v {
            serde_json::Value::String(s) => s.clone(),
            other => other.to_string(),
        };
        out = out.replace(&placeholder, &replacement);
    }
    out
}

/// Partial override applied on top of a built-in [`AgentRole`].
///
/// Keyed by role name in [`crate::config::Config::agent_roles`] so
/// operators can edit `naked.json` to swap models / lengthen budgets
/// / add skills without recompiling. Every field is optional —
/// `None` means "keep the built-in value".
///
/// Field choices are deliberately the smallest set that matters in
/// practice: model swap (cost / capability tuning), iteration budget
/// (cost cap), skills (loadout). Tool filter / browser_runtime are
/// intentionally NOT overrideable from JSON yet — they're security-
/// adjacent and changing them at runtime should require a deliberate
/// code change (and a code review).
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct AgentRoleOverride {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub model: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_iters: Option<usize>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub default_skills: Option<Vec<String>>,
    /// Override of [`AgentRole::default_validators`]. `None` keeps the
    /// shipped list; `Some(vec![])` explicitly clears validation
    /// (operator opt-out for debugging only — production should leave
    /// at least `gatekeeper` in place).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub default_validators: Option<Vec<String>>,
}

/// Apply an override to a role, returning the merged result. Pure;
/// the input is consumed by value to discourage callers from sharing
/// the pre-override role around — the merged one is what should flow
/// through the runtime.
pub fn apply_role_override(mut role: AgentRole, ov: &AgentRoleOverride) -> AgentRole {
    if let Some(m) = &ov.model {
        role.model = Some(m.clone());
    }
    if let Some(n) = ov.max_iters {
        role.max_iters = n;
    }
    if let Some(s) = &ov.default_skills {
        role.default_skills = s.clone();
    }
    if let Some(v) = &ov.default_validators {
        role.default_validators = v.clone();
    }
    role
}

// ── Built-in role bodies REMOVED in v0.2 ─────────────────────────────
//
// Role definitions (system prompt, model, tool filter, max_iters,
// browser_runtime, default_criteria) now live on disk under
// `naked/agents/<name>/{role.json, prompt.md, criteria.txt}` and are
// loaded by [`crate::agent_store::AgentStore`] from `Config::agent_dirs`
// at startup.
//
// Fetching a role by name:
//
// ```
// let store = core.agent_store();
// let role = naked_core::agent_store::resolve_role(
//     "web_researcher",
//     &store,
//     &config.agent_roles,
// );
// ```
//
// Adding a new role no longer requires a Rust change: drop a new
// directory under `agents/` and restart the binary. Previous Rust
// constructors (`browser_extractor`, `web_researcher`,
// `builtin_roles`, `resolve_role`) were deleted; their prompts are
// preserved verbatim under `naked/agents/<name>/prompt.md`.
//
// `apply_role_override` is still here (used by the store / resolver).

#[cfg(test)]
#[path = "agent_role_tests.rs"]
mod tests;

// ── T10 of PLAN_QUALITY_v1: canonical role taxonomy ──────────────
//
// Six well-known roles + `custom`. Each has a tuned system-prompt
// prefix; the parent picks the right kind of helper for the work
// instead of dispatching ad-hoc roles. Aliases (case-insensitive)
// match DeepSeek TUI's table so users coming from there don't have
// to relearn the spelling.

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum CanonicalRole {
    General,
    Explore,
    Plan,
    Review,
    Implementer,
    Verifier,
    Custom,
}

impl CanonicalRole {
    #[must_use]
    pub fn label(self) -> &'static str {
        match self {
            Self::General => "general",
            Self::Explore => "explore",
            Self::Plan => "plan",
            Self::Review => "review",
            Self::Implementer => "implementer",
            Self::Verifier => "verifier",
            Self::Custom => "custom",
        }
    }
}

/// Map any case-insensitive alias to a [`CanonicalRole`]. Unknown
/// inputs return `None` so the caller can decide whether to surface
/// an error.
#[must_use]
pub fn canonicalize_role(s: &str) -> Option<CanonicalRole> {
    let lower = s.trim().to_ascii_lowercase();
    Some(match lower.as_str() {
        "general" | "worker" | "default" | "general-purpose" => CanonicalRole::General,
        "explore" | "explorer" | "exploration" => CanonicalRole::Explore,
        "plan" | "planning" | "awaiter" => CanonicalRole::Plan,
        "review" | "reviewer" | "code-review" => CanonicalRole::Review,
        "implementer" | "implement" | "implementation" | "builder" => CanonicalRole::Implementer,
        "verifier" | "verify" | "verification" | "validator" | "tester" => CanonicalRole::Verifier,
        "custom" => CanonicalRole::Custom,
        _ => return None,
    })
}

/// Build a freshly-configured [`AgentRole`] for a canonical role.
/// Tool-filter enforcement still flows through the loop's permission
/// gate; the role's intent is set here so the model receives a
/// posture-appropriate system prompt prefix.
#[must_use]
pub fn role_for_canonical(role: CanonicalRole) -> AgentRole {
    let (name, system) = match role {
        CanonicalRole::General => (
            "general",
            "You are a focused sub-agent. Carry out the parent's task end-to-end. \
             Prefer minimum-edit solutions. Hand back a brief structured summary.",
        ),
        CanonicalRole::Explore => (
            "explore",
            "You are a READ-ONLY explorer. Map the requested code/topic FAST. \
             You may run shell to grep / list / read; you MUST NOT write to disk \
             or apply patches. Hand back a structured map with file paths + line ranges.",
        ),
        CanonicalRole::Plan => (
            "plan",
            "You are a planner. Produce an executable strategy: numbered steps, \
             explicit dependencies, acceptance criteria. You may write plan files \
             but MUST NOT apply code edits. Don't carry out the plan; the parent \
             dispatches an implementer for that.",
        ),
        CanonicalRole::Review => (
            "review",
            "You are a reviewer. Read + grade the change. Severity scores: \
             critical / high / medium / low / nit. NEVER write or patch. \
             Describe each fix as a finding; the parent decides whether to dispatch \
             an implementer.",
        ),
        CanonicalRole::Implementer => (
            "implementer",
            "You are an implementer. Land the specified change with the minimum \
             edit. No drive-by refactors. Run a quick verification (cargo check / \
             pytest etc) and hand back a structured outcome.",
        ),
        CanonicalRole::Verifier => (
            "verifier",
            "You are a verifier. Run the requested test/validation suite, report \
             pass/fail with the failing assertion + stack. Do NOT fix failures; \
             record fix candidates under RISKS in the output.",
        ),
        CanonicalRole::Custom => (
            "custom",
            "You are a custom-scope sub-agent. The parent has supplied an explicit \
             `allowed_tools` list; you may only use those. Stay within the requested \
             narrow scope.",
        ),
    };
    AgentRole::new(name, system)
}

#[cfg(test)]
mod canonical_tests {
    use super::*;

    #[test]
    fn aliases_resolve() {
        assert_eq!(canonicalize_role("explorer"), Some(CanonicalRole::Explore));
        assert_eq!(canonicalize_role("WORKER"), Some(CanonicalRole::General));
        assert_eq!(
            canonicalize_role("Code-Review"),
            Some(CanonicalRole::Review)
        );
        assert_eq!(canonicalize_role("tester"), Some(CanonicalRole::Verifier));
        assert_eq!(canonicalize_role("nope"), None);
    }

    #[test]
    fn build_returns_role_with_expected_name() {
        let r = role_for_canonical(CanonicalRole::Explore);
        assert_eq!(r.name, "explore");
    }

    #[test]
    fn every_role_has_distinct_label() {
        let labels: Vec<&str> = [
            CanonicalRole::General,
            CanonicalRole::Explore,
            CanonicalRole::Plan,
            CanonicalRole::Review,
            CanonicalRole::Implementer,
            CanonicalRole::Verifier,
            CanonicalRole::Custom,
        ]
        .iter()
        .map(|&r| r.label())
        .collect();
        assert_eq!(
            labels
                .iter()
                .collect::<std::collections::HashSet<_>>()
                .len(),
            7
        );
    }
}
