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
mod tests {
    use super::*;

    #[test]
    fn tool_filter_allow_all_lets_everything_through() {
        let f = ToolFilter::default();
        assert!(f.allows("web_fetch"));
        assert!(f.allows("bash"));
        assert!(f.allows("anything"));
    }

    #[test]
    fn tool_filter_allow_lists_only_named_tools() {
        let f = ToolFilter::Allow {
            tools: vec!["web_fetch".into(), "research_save".into()],
        };
        assert!(f.allows("web_fetch"));
        assert!(f.allows("research_save"));
        assert!(!f.allows("bash"));
        assert!(!f.allows("write_file"));
    }

    #[test]
    fn tool_filter_deny_excludes_named_tools() {
        let f = ToolFilter::Deny {
            tools: vec!["bash".into(), "write_file".into()],
        };
        assert!(f.allows("web_fetch"));
        assert!(f.allows("research_save"));
        assert!(!f.allows("bash"));
        assert!(!f.allows("write_file"));
    }

    #[test]
    fn agent_role_builder_chain() {
        // Use a placeholder model id — we are testing the builder
        // wiring, not asserting any real provider has this model.
        // Hard-coding e.g. `qwen3.6-plus` would create a hidden
        // assumption that whoever runs the test stack also has that
        // provider configured.
        let r = AgentRole::new("test", "you are a tester")
            .with_description("for tests")
            .with_model("placeholder-model-id")
            .with_skills(vec!["web-browser-playbook".into()])
            .with_max_iters(15)
            .with_tool_filter(ToolFilter::Allow {
                tools: vec!["web_fetch".into()],
            });

        assert_eq!(r.name, "test");
        assert_eq!(r.description, "for tests");
        assert_eq!(r.model.as_deref(), Some("placeholder-model-id"));
        assert_eq!(r.default_skills, vec!["web-browser-playbook".to_string()]);
        assert_eq!(r.max_iters, 15);
        assert!(r.tool_filter.allows("web_fetch"));
        assert!(!r.tool_filter.allows("bash"));
    }

    #[test]
    fn agent_role_serialises_compactly_when_defaults() {
        // Default role should produce minimal JSON — no empty maps,
        // no `null`s. Otherwise naked.json gets noisy when listing
        // many roles.
        let r = AgentRole::new("min", "sys");
        let json = serde_json::to_string(&r).unwrap();
        // No `model`, no `default_skills`, no `browser_runtime`.
        assert!(!json.contains("\"model\""), "json: {json}");
        assert!(!json.contains("\"default_skills\""), "json: {json}");
        assert!(!json.contains("\"browser_runtime\""), "json: {json}");
        // tool_filter MUST be present (it's tagged so we always know
        // the policy explicitly even when default).
        assert!(json.contains("\"tool_filter\""), "json: {json}");
        assert!(json.contains("\"allow_all\""), "json: {json}");
    }

    #[test]
    fn agent_role_round_trips_through_json() {
        let r = AgentRole::new("rt", "sys")
            .with_model("m1")
            .with_skills(vec!["s1".into(), "s2".into()])
            .with_browser_runtime(BrowserRuntime {
                proxy: Some("http://proxy:8080".into()),
                extensions: vec!["./ext/capsolver".into()],
                extension_env: [("CAPSOLVER_API_KEY".to_string(), "abc".to_string())]
                    .into_iter()
                    .collect(),
            });
        let json = serde_json::to_string(&r).unwrap();
        let back: AgentRole = serde_json::from_str(&json).unwrap();
        assert_eq!(back.name, r.name);
        assert_eq!(back.model, r.model);
        assert_eq!(back.default_skills, r.default_skills);
        assert_eq!(back.browser_runtime.proxy, r.browser_runtime.proxy);
        assert_eq!(
            back.browser_runtime.extension_env.get("CAPSOLVER_API_KEY"),
            Some(&"abc".to_string())
        );
    }

    #[test]
    fn task_builder_chain() {
        let t = Task::new("browser_extractor", "open and extract")
            .with_id("t-1")
            .with_max_wall(60)
            .with_context(serde_json::json!({"url": "https://x.com"}));
        assert_eq!(t.id, "t-1");
        assert_eq!(t.role, "browser_extractor");
        assert_eq!(t.max_wall_secs, Some(60));
        assert_eq!(t.context["url"], "https://x.com");
    }

    #[test]
    fn task_stats_summary_line_is_grep_compatible() {
        // Pin the on-wire shape so probe / batch greppers keep
        // working.
        let mut s = TaskStats::default();
        s.tools.insert("Skill".into(), 1);
        s.tools.insert("browser_navigate".into(), 2);
        s.captcha_hits = 1;
        s.skill_loads = 1;
        s.text_deltas = 7;

        let line = s.summary_line();
        assert!(line.starts_with("[task-summary] "));
        assert!(line.contains("Skill=1"));
        assert!(line.contains("browser_navigate=2"));
        assert!(line.contains("captcha_hits=1"));
        assert!(line.contains("skill_loads=1"));
        assert!(line.contains("text_deltas=7"));
        assert!(line.contains("errors=0"));
    }

    #[test]
    fn validation_verdict_helpers() {
        let p = ValidationVerdict::pass("phone-vn");
        assert!(p.passed);
        assert!(p.reasons.is_empty());
        assert_eq!(p.validator, "phone-vn");

        let f = ValidationVerdict::fail("phone-vn", "no digits");
        assert!(!f.passed);
        assert_eq!(f.reasons, vec!["no digits".to_string()]);
    }

    #[test]
    fn expand_placeholders_substitutes_top_level_strings() {
        let ctx = serde_json::json!({
            "topic": "real-estate",
            "url":   "https://example.com",
            "n":     5,
        });
        let out = expand_placeholders("topic={topic}, url={url}, n={n}", &ctx);
        assert_eq!(out, "topic=real-estate, url=https://example.com, n=5");
    }

    #[test]
    fn expand_placeholders_leaves_unknown_keys_alone() {
        let ctx = serde_json::json!({"topic": "x"});
        let out = expand_placeholders("{topic} but not {missing}", &ctx);
        assert_eq!(out, "x but not {missing}");
    }

    #[test]
    fn expand_placeholders_handles_non_object_context() {
        let out = expand_placeholders("hello {name}", &serde_json::Value::Null);
        assert_eq!(out, "hello {name}");
    }

    #[test]
    fn task_output_skeleton_has_idle_default() {
        let o = TaskOutput::skeleton("t-1", "browser_extractor");
        assert_eq!(o.task_id, "t-1");
        assert_eq!(o.role_name, "browser_extractor");
        assert_eq!(o.stop_reason, StopReason::AgentIdle);
    }

    // A trivially-implementable Validator used to verify the trait
    // signature compiles + returns the right shape.
    struct AlwaysPass;
    #[async_trait]
    impl Validator for AlwaysPass {
        fn name(&self) -> &str {
            "always-pass"
        }
        async fn validate(&self, _o: &TaskOutput) -> ValidationVerdict {
            ValidationVerdict::pass(self.name())
        }
    }

    // Note: the contract tests for the built-in roles
    // (`browser_extractor` / `web_researcher` shape, universality of
    // `web_researcher`'s prompt) now live in
    // `agent_store::tests` — they load the shipped data files
    // (`naked/agents/<name>/`) via `AgentStore::load_dirs` so the
    // assertions stay anchored to the real disk artefacts an
    // operator can edit, not to a stale Rust-side copy.

    #[test]
    fn agent_role_default_validators_round_trips() {
        let r = AgentRole::new("v", "sys")
            .with_validators(vec!["gatekeeper".into(), "phone-vn".into()]);
        assert_eq!(
            r.default_validators,
            vec!["gatekeeper".to_string(), "phone-vn".to_string()]
        );
        let json = serde_json::to_string(&r).unwrap();
        assert!(json.contains("\"default_validators\""), "json: {json}");
        let back: AgentRole = serde_json::from_str(&json).unwrap();
        assert_eq!(back.default_validators, r.default_validators);
    }

    #[test]
    fn agent_role_default_validators_empty_omitted_in_json() {
        let r = AgentRole::new("v", "sys");
        let json = serde_json::to_string(&r).unwrap();
        assert!(
            !json.contains("\"default_validators\""),
            "empty list must be skipped: {json}"
        );
    }

    #[test]
    fn agent_role_override_validators_replaces_list() {
        let base = AgentRole::new("rt", "sys")
            .with_validators(vec!["gatekeeper".into(), "phone-vn".into()]);
        let ov = AgentRoleOverride {
            default_validators: Some(vec!["gatekeeper".into()]),
            ..Default::default()
        };
        let merged = apply_role_override(base.clone(), &ov);
        assert_eq!(merged.default_validators, vec!["gatekeeper".to_string()]);

        // None-override leaves the list intact (no surprise wipe).
        let merged = apply_role_override(base.clone(), &AgentRoleOverride::default());
        assert_eq!(merged.default_validators, base.default_validators);

        // Explicit Some(vec![]) is the documented opt-out.
        let ov_empty = AgentRoleOverride {
            default_validators: Some(vec![]),
            ..Default::default()
        };
        let merged = apply_role_override(base, &ov_empty);
        assert!(merged.default_validators.is_empty());
    }

    #[test]
    fn agent_role_override_apply_only_touches_set_fields() {
        let base = AgentRole::new("rt", "sys")
            .with_model("default-m")
            .with_max_iters(20);
        let o = AgentRoleOverride {
            model: Some("override-m".into()),
            ..Default::default()
        };
        let merged = apply_role_override(base.clone(), &o);
        assert_eq!(merged.model.as_deref(), Some("override-m"));
        assert_eq!(merged.max_iters, 20, "untouched field must survive");

        let o = AgentRoleOverride {
            max_iters: Some(99),
            default_skills: Some(vec!["only-this".into()]),
            ..Default::default()
        };
        let merged = apply_role_override(base.clone(), &o);
        assert_eq!(merged.model.as_deref(), Some("default-m"));
        assert_eq!(merged.max_iters, 99);
        assert_eq!(merged.default_skills, vec!["only-this".to_string()]);
    }

    #[tokio::test]
    async fn validator_trait_is_object_safe_and_callable() {
        // The whole point of `dyn Validator` is to let us stack
        // heterogeneous validators in a `Vec<Arc<dyn Validator>>`.
        // This test is the contract: if it stops compiling we've
        // accidentally broken object-safety on the trait.
        let v: std::sync::Arc<dyn Validator> = std::sync::Arc::new(AlwaysPass);
        let out = TaskOutput::skeleton("t", "r");
        let verdict = v.validate(&out).await;
        assert!(verdict.passed);
        assert_eq!(verdict.validator, "always-pass");
    }
}
