//! High-level research orchestration tools available in every chat turn.
//!
//! Unlike the low-level `research_save` / `research_list` tools (which require
//! `ResearchContext` set by the coordinator), these operate on explicit spec ids
//! and let the LLM agent propose & execute research workflows conversationally:
//!
//! - `research_create`     — create a new spec
//! - `research_list_specs` — enumerate existing specs with status
//! - `research_findings`   — query findings for any spec
//! - `research_launch`     — kick off a full background run via the coordinator
//! - `research_set_target` — enter inline research mode (sets context for
//!   `research_save` in the current turn)

use std::path::Path;
use std::sync::{Arc, Weak};

use async_trait::async_trait;
use chrono::Utc;
use serde_json::{Value, json};

use crate::config::ResearchConfig;
use crate::tool::Tool;
use crate::types::{Permission, ToolResult, ToolSpec};
use crate::{AgentCore, ResearchPatch};

use super::spec::{ResearchSpec, new_research_id};
use super::store::ResearchStore;
use super::tool::{ResearchContext, parse_listing_date};

fn ok(msg: impl Into<String>) -> ToolResult {
    ToolResult {
        output: msg.into(),
        is_error: false,
    }
}
fn err(msg: impl Into<String>) -> ToolResult {
    ToolResult {
        output: msg.into(),
        is_error: true,
    }
}

// ─── research_create ────────────────────────────────────────────────────────

pub struct ResearchCreateTool {
    store: Arc<dyn ResearchStore>,
    defaults: ResearchConfig,
}

impl ResearchCreateTool {
    pub fn new(store: Arc<dyn ResearchStore>, defaults: ResearchConfig) -> Self {
        Self { store, defaults }
    }
}

#[async_trait]
impl Tool for ResearchCreateTool {
    fn spec(&self) -> ToolSpec {
        ToolSpec {
            name: "research_create".into(),
            description: "Create a new research spec for a given topic. Returns the spec id \
                          which can be passed to research_launch (deep background run) or \
                          research_set_target (inline research in the current chat)."
                .into(),
            parameters: json!({
                "type": "object",
                "properties": {
                    "topic": {
                        "type": "string",
                        "description": "Free-text description of what to research, e.g. \
                            'commercial rental in Da Nang under $200/month'"
                    },
                    "sources": {
                        "type": "array",
                        "items": { "type": "string" },
                        "description": "Optional seed URLs to start from. If omitted, \
                            the configured default_sources are used."
                    }
                },
                "required": ["topic"]
            }),
            permission: Permission::WorkspaceWrite,
        }
    }

    async fn execute(&self, input: Value, _cwd: &Path) -> ToolResult {
        let topic = match input.get("topic").and_then(|v| v.as_str()) {
            Some(t) if !t.trim().is_empty() => t.trim(),
            _ => return err("missing or empty `topic`"),
        };
        let sources: Vec<String> = input
            .get("sources")
            .and_then(|v| v.as_array())
            .map(|arr| {
                arr.iter()
                    .filter_map(|v| v.as_str().map(String::from))
                    .collect()
            })
            .unwrap_or_else(|| self.defaults.default_sources.clone());

        let spec = ResearchSpec {
            id: new_research_id(topic),
            topic: topic.to_string(),
            sources: sources.clone(),
            interval_seconds: None,
            run_at: None,
            cron: None,
            task_timeout_seconds: None,
            session_id: None,
            chat_id: None,
            thread_id: None,
            provider: None,
            model: None,
            max_iterations: None,
            max_wall_seconds: None,
            created_at: Utc::now(),
            paused: false,
            pause_reason: None,
        };

        match self.store.create_spec(&spec).await {
            Ok(()) => ok(json!({
                "id": spec.id,
                "topic": spec.topic,
                "sources": sources,
                "hint": "Use research_launch to start a deep background run, or \
                         research_set_target to research inline in this chat."
            })
            .to_string()),
            Err(e) => err(format!("failed to create spec: {e}")),
        }
    }
}

// ─── research_list_specs ────────────────────────────────────────────────────

pub struct ResearchListSpecsTool {
    store: Arc<dyn ResearchStore>,
    /// Snapshot of the global research config taken at registration time —
    /// surfaced in tool output so the LLM can see e.g. whether verification
    /// is enabled by default. The actual run-time behaviour still consults
    /// the live `AgentCore::config()`; this is purely informational.
    config: ResearchConfig,
}

impl ResearchListSpecsTool {
    pub fn new(store: Arc<dyn ResearchStore>, config: ResearchConfig) -> Self {
        Self { store, config }
    }
}

#[async_trait]
impl Tool for ResearchListSpecsTool {
    fn spec(&self) -> ToolSpec {
        ToolSpec {
            name: "research_list_specs".into(),
            description: "List all research specs with their status, schedule, finding \
                          counts, and last-run metrics (incl. gatekeeper verification \
                          stats when available)."
                .into(),
            parameters: json!({
                "type": "object",
                "properties": {},
            }),
            permission: Permission::ReadOnly,
        }
    }

    async fn execute(&self, _input: Value, _cwd: &Path) -> ToolResult {
        let specs = match self.store.list_specs().await {
            Ok(s) => s,
            Err(e) => return err(format!("failed to list specs: {e}")),
        };
        if specs.is_empty() {
            return ok("No research specs defined. Use research_create to start one.");
        }

        let mut items = Vec::with_capacity(specs.len());
        for s in &specs {
            let total = self.store.count_findings(&s.id).await.unwrap_or(0);
            // Pull a small window so we can prefer a record that carries
            // verification stats (the trailing `<run>-verified` summary)
            // when one exists, and fall back to the most recent otherwise.
            let runs = self
                .store
                .list_runs(&s.id, Some(20))
                .await
                .unwrap_or_default();
            let last_run = runs
                .iter()
                .find(|r| r.verification_rounds.is_some())
                .or_else(|| runs.first())
                .map(|r| {
                    json!({
                        "run_id": r.run_id,
                        "at": r.finished_at.to_rfc3339(),
                        "new": r.new_findings,
                        "total": r.total_findings_after,
                        "stop_reason": r.stop_reason,
                        "provider": r.provider,
                        "model": r.model,
                        "elapsed_secs": r.elapsed_secs,
                        "verification_rounds": r.verification_rounds,
                        "dead_removed": r.dead_removed,
                        "replacements_found": r.replacements_found,
                        "remaining_issues": r.remaining_issues,
                    })
                });
            items.push(json!({
                "id": s.id,
                "topic": s.topic,
                "paused": s.paused,
                "interval_seconds": s.interval_seconds,
                "verify_by_default": self.config.verify_by_default,
                "sources": s.sources,
                "total_findings": total,
                "last_run": last_run,
                "created_at": s.created_at.to_rfc3339(),
            }));
        }
        ok(Value::Array(items).to_string())
    }
}

// ─── research_metrics ───────────────────────────────────────────────────────

/// Detailed view of a single spec: schedule, dead-finding lifetime totals,
/// the last `runs_limit` `RunRecord`s, plus the head of `report.md` so the
/// LLM can answer "расскажи метрики последнего исследования" without
/// pulling the whole report.
pub struct ResearchMetricsTool {
    store: Arc<dyn ResearchStore>,
    runs_limit_default: usize,
}

impl ResearchMetricsTool {
    pub fn new(store: Arc<dyn ResearchStore>) -> Self {
        Self {
            store,
            runs_limit_default: 5,
        }
    }
}

#[async_trait]
impl Tool for ResearchMetricsTool {
    fn spec(&self) -> ToolSpec {
        ToolSpec {
            name: "research_metrics".into(),
            description: "Return detailed metrics for a research spec: schedule, total/fresh \
                          finding counts, the last N run records (incl. gatekeeper stats), \
                          and the head of report.md. Use this to answer natural-language \
                          questions like 'how is research X going?' or 'tell me about the \
                          last run'."
                .into(),
            parameters: json!({
                "type": "object",
                "properties": {
                    "spec_id": {
                        "type": "string",
                        "description": "Research spec id (from research_list_specs)"
                    },
                    "runs_limit": {
                        "type": "integer",
                        "description": "Max number of recent run records to return. Default: 5."
                    }
                },
                "required": ["spec_id"]
            }),
            permission: Permission::ReadOnly,
        }
    }

    async fn execute(&self, input: Value, _cwd: &Path) -> ToolResult {
        let spec_id = match input.get("spec_id").and_then(|v| v.as_str()) {
            Some(id) if !id.is_empty() => id,
            _ => return err("missing `spec_id`"),
        };
        let runs_limit = input
            .get("runs_limit")
            .and_then(|v| v.as_u64())
            .map(|v| v as usize)
            .unwrap_or(self.runs_limit_default)
            .max(1);

        let spec = match self.store.load_spec(spec_id).await {
            Ok(s) => s,
            Err(e) => return err(format!("spec `{spec_id}` not found: {e}")),
        };
        let total = self.store.count_findings(spec_id).await.unwrap_or(0);
        let runs = self
            .store
            .list_runs(spec_id, Some(runs_limit))
            .await
            .unwrap_or_default();

        // Findings filtered to the most-recent run id are "fresh".
        let fresh_findings_count = if let Some(latest) = runs.first() {
            let all = self
                .store
                .list_findings(spec_id, None)
                .await
                .unwrap_or_default();
            all.iter().filter(|f| f.run_id == latest.run_id).count() as u32
        } else {
            0
        };

        let report_excerpt = self
            .store
            .read_report(spec_id)
            .await
            .ok()
            .flatten()
            .map(|s| s.chars().take(800).collect::<String>());

        let runs_json: Vec<Value> = runs
            .iter()
            .map(|r| {
                json!({
                    "run_id": r.run_id,
                    "started_at": r.started_at.to_rfc3339(),
                    "finished_at": r.finished_at.to_rfc3339(),
                    "new_findings": r.new_findings,
                    "total_findings_after": r.total_findings_after,
                    "stop_reason": r.stop_reason,
                    "provider": r.provider,
                    "model": r.model,
                    "elapsed_secs": r.elapsed_secs,
                    "verification_rounds": r.verification_rounds,
                    "dead_removed": r.dead_removed,
                    "replacements_found": r.replacements_found,
                    "remaining_issues": r.remaining_issues,
                })
            })
            .collect();

        ok(json!({
            "spec_id": spec.id,
            "topic": spec.topic,
            "paused": spec.paused,
            "sources": spec.sources,
            "schedule": {
                "interval_seconds": spec.interval_seconds,
            },
            "total_findings": total,
            "fresh_findings_count": fresh_findings_count,
            "runs": runs_json,
            "report_excerpt": report_excerpt,
        })
        .to_string())
    }
}

// ─── research_findings ──────────────────────────────────────────────────────

pub struct ResearchFindingsTool {
    store: Arc<dyn ResearchStore>,
}

impl ResearchFindingsTool {
    pub fn new(store: Arc<dyn ResearchStore>) -> Self {
        Self { store }
    }
}

#[async_trait]
impl Tool for ResearchFindingsTool {
    fn spec(&self) -> ToolSpec {
        ToolSpec {
            name: "research_findings".into(),
            description: "Show findings for a research spec. Use fresh_only=true to see \
                          only the latest run's results."
                .into(),
            parameters: json!({
                "type": "object",
                "properties": {
                    "spec_id": {
                        "type": "string",
                        "description": "Research spec id (from research_list_specs or research_create)"
                    },
                    "fresh_only": {
                        "type": "boolean",
                        "description": "If true, only show findings from the most recent run. Default: false."
                    },
                    "max_age_days": {
                        "type": "integer",
                        "description": "Hide findings with listing_date older than N days. Default: 90. Set to 0 to show all."
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Max findings to return. Default: 30."
                    }
                },
                "required": ["spec_id"]
            }),
            permission: Permission::ReadOnly,
        }
    }

    async fn execute(&self, input: Value, _cwd: &Path) -> ToolResult {
        let spec_id = match input.get("spec_id").and_then(|v| v.as_str()) {
            Some(id) if !id.is_empty() => id,
            _ => return err("missing `spec_id`"),
        };
        let fresh_only = input
            .get("fresh_only")
            .and_then(|v| v.as_bool())
            .unwrap_or(false);
        let max_age_days = input
            .get("max_age_days")
            .and_then(|v| v.as_i64())
            .unwrap_or(90);
        let limit = input.get("limit").and_then(|v| v.as_u64()).unwrap_or(30) as usize;

        if self.store.load_spec(spec_id).await.is_err() {
            return err(format!("spec `{spec_id}` not found"));
        }

        let all = match self.store.list_findings(spec_id, None).await {
            Ok(f) => f,
            Err(e) => return err(format!("failed to load findings: {e}")),
        };

        let latest_run_id = if fresh_only {
            let runs = self
                .store
                .list_runs(spec_id, Some(1))
                .await
                .unwrap_or_default();
            runs.first().map(|r| r.run_id.clone())
        } else {
            None
        };

        let today = Utc::now().date_naive();
        let mut skipped_stale = 0u32;

        let filtered: Vec<_> = all
            .iter()
            .filter(|f| {
                if let Some(ref rid) = latest_run_id
                    && &f.run_id != rid
                {
                    return false;
                }
                // Filter out stale listings if max_age_days > 0
                if max_age_days > 0
                    && let Some(ref date_str) = f.listing_date
                    && let Some(d) = parse_listing_date(date_str)
                {
                    let age = (today - d).num_days();
                    if age > max_age_days {
                        skipped_stale += 1;
                        return false;
                    }
                }
                true
            })
            .take(limit)
            .map(|f| {
                json!({
                    "title": f.title,
                    "price": f.price,
                    "url": f.url,
                    "listing_date": f.listing_date,
                    "excerpt": f.excerpt,
                    "has_source_content": f.source_content.is_some(),
                })
            })
            .collect();

        let total = all.len();
        let shown = filtered.len();
        let fresh_count = latest_run_id
            .as_ref()
            .map(|rid| all.iter().filter(|f| f.run_id == *rid).count());

        ok(json!({
            "spec_id": spec_id,
            "total_findings": total,
            "fresh_findings": fresh_count,
            "showing": shown,
            "skipped_stale": skipped_stale,
            "max_age_days": max_age_days,
            "findings": filtered,
        })
        .to_string())
    }
}

// ─── research_launch ────────────────────────────────────────────────────────

pub struct ResearchLaunchTool {
    core: Weak<AgentCore>,
}

impl ResearchLaunchTool {
    pub fn new(core: Weak<AgentCore>) -> Self {
        Self { core }
    }
}

#[async_trait]
impl Tool for ResearchLaunchTool {
    fn spec(&self) -> ToolSpec {
        ToolSpec {
            name: "research_launch".into(),
            description: "Launch a deep background research run for a spec. The run uses a \
                          dedicated agent session with web_fetch and research_save, and \
                          typically takes 10-20 minutes. Runs gatekeeper verification by \
                          default (configurable via `research.verify_by_default`): dead \
                          findings are removed and the agent re-runs to find replacements. \
                          Returns immediately; use research_findings to check results later."
                .into(),
            parameters: json!({
                "type": "object",
                "properties": {
                    "spec_id": {
                        "type": "string",
                        "description": "Research spec id to run"
                    }
                },
                "required": ["spec_id"]
            }),
            permission: Permission::Dangerous,
        }
    }

    async fn execute(&self, input: Value, _cwd: &Path) -> ToolResult {
        let spec_id = match input.get("spec_id").and_then(|v| v.as_str()) {
            Some(id) if !id.is_empty() => id.to_string(),
            _ => return err("missing `spec_id`"),
        };

        let core = match self.core.upgrade() {
            Some(c) => c,
            None => return err("agent core is no longer available"),
        };

        if let Err(e) = core.load_research(&spec_id).await {
            return err(format!("spec `{spec_id}` not found: {e}"));
        }

        let id = spec_id.clone();
        let verify = core.config().research.verify_by_default;
        let max_rounds = core.config().research.gatekeeper.max_rounds;
        tokio::spawn(async move {
            if verify {
                match core.run_research_verified(&id, max_rounds).await {
                    Ok(vr) => {
                        let r = &vr.last_run;
                        tracing::info!(
                            spec_id = %id,
                            new = r.new_findings,
                            total = r.total_findings_after,
                            rounds = vr.verification_rounds,
                            dead_removed = vr.dead_removed,
                            replacements = vr.replacements_found,
                            final_findings = vr.final_findings,
                            "research_launch background run (verified) finished"
                        );
                    }
                    Err(e) => {
                        tracing::error!(
                            spec_id = %id,
                            ?e,
                            "research_launch background run (verified) failed"
                        );
                    }
                }
            } else {
                match core.run_research(&id).await {
                    Ok(r) => {
                        tracing::info!(
                            spec_id = %id,
                            new = r.new_findings,
                            total = r.total_findings_after,
                            "research_launch background run finished"
                        );
                    }
                    Err(e) => {
                        tracing::error!(
                            spec_id = %id,
                            ?e,
                            "research_launch background run failed"
                        );
                    }
                }
            }
        });

        let hint = if verify {
            "The run is executing in the background (10-20 min, plus gatekeeper \
             verification rounds). Use research_findings to check results when done."
        } else {
            "The run is executing in the background (10-20 min). \
             Use research_findings to check results when done."
        };
        ok(json!({
            "status": "launched",
            "spec_id": spec_id,
            "verified": verify,
            "hint": hint,
        })
        .to_string())
    }
}

// ─── research_update_spec ───────────────────────────────────────────────────

/// LLM-facing partial-update tool. Maps to [`AgentCore::update_research`].
/// Every parameter is optional; absent ones leave the spec unchanged.
pub struct ResearchUpdateSpecTool {
    core: Weak<AgentCore>,
}

impl ResearchUpdateSpecTool {
    pub fn new(core: Weak<AgentCore>) -> Self {
        Self { core }
    }
}

#[async_trait]
impl Tool for ResearchUpdateSpecTool {
    fn spec(&self) -> ToolSpec {
        ToolSpec {
            name: "research_update_spec".into(),
            description: "Apply a partial update to a research spec — change topic, add/replace \
                          sources, set or clear schedule, override provider/model, etc. Pass \
                          only the fields you want to change. Use `interval_seconds: null` \
                          (or omit and pass `clear_schedule: true`) to remove the schedule. \
                          Sources passed via `sources_add` are appended (deduplicated); \
                          `sources_set` replaces the entire list."
                .into(),
            parameters: json!({
                "type": "object",
                "properties": {
                    "spec_id": { "type": "string", "description": "Spec id to update" },
                    "topic": { "type": "string" },
                    "sources_set": {
                        "type": "array", "items": {"type": "string"},
                        "description": "Replace the entire sources list."
                    },
                    "sources_add": {
                        "type": "array", "items": {"type": "string"},
                        "description": "Append URLs to the sources list (dedup)."
                    },
                    "interval_seconds": {
                        "type": ["integer", "null"],
                        "description": "Polling interval in seconds. Use null to clear."
                    },
                    "clear_schedule": {
                        "type": "boolean",
                        "description": "Convenience flag — equivalent to interval_seconds=null."
                    },
                    "provider": { "type": "string" },
                    "model": { "type": "string" },
                    "max_iterations": { "type": ["integer", "null"] },
                    "max_wall_seconds": { "type": ["integer", "null"] },
                    "paused": {
                        "type": "boolean",
                        "description": "Pause/resume the spec. Affects scheduled runs only."
                    }
                },
                "required": ["spec_id"]
            }),
            permission: Permission::WorkspaceWrite,
        }
    }

    async fn execute(&self, input: Value, _cwd: &Path) -> ToolResult {
        let spec_id = match input.get("spec_id").and_then(|v| v.as_str()) {
            Some(id) if !id.is_empty() => id.to_string(),
            _ => return err("missing `spec_id`"),
        };
        let core = match self.core.upgrade() {
            Some(c) => c,
            None => return err("agent core is no longer available"),
        };

        let mut patch = ResearchPatch::default();
        if let Some(t) = input.get("topic").and_then(|v| v.as_str()) {
            patch.topic = Some(t.to_string());
        }
        if let Some(arr) = input.get("sources_set").and_then(|v| v.as_array()) {
            patch.sources_replace = Some(
                arr.iter()
                    .filter_map(|v| v.as_str().map(String::from))
                    .collect(),
            );
        }
        if let Some(arr) = input.get("sources_add").and_then(|v| v.as_array()) {
            patch.sources_add = Some(
                arr.iter()
                    .filter_map(|v| v.as_str().map(String::from))
                    .collect(),
            );
        }
        // interval_seconds: missing → no-op, null → clear, integer → set
        if input.get("clear_schedule").and_then(|v| v.as_bool()) == Some(true) {
            patch.interval_seconds = Some(None);
        } else if let Some(v) = input.get("interval_seconds") {
            if v.is_null() {
                patch.interval_seconds = Some(None);
            } else if let Some(n) = v.as_u64() {
                patch.interval_seconds = Some(Some(n));
            } else {
                return err("`interval_seconds` must be an integer or null");
            }
        }
        if let Some(p) = input.get("provider").and_then(|v| v.as_str()) {
            patch.provider = Some(p.to_string());
        }
        if let Some(m) = input.get("model").and_then(|v| v.as_str()) {
            patch.model = Some(m.to_string());
        }
        if let Some(v) = input.get("max_iterations") {
            if v.is_null() {
                patch.max_iterations = Some(None);
            } else if let Some(n) = v.as_u64() {
                patch.max_iterations = Some(Some(n as u32));
            }
        }
        if let Some(v) = input.get("max_wall_seconds") {
            if v.is_null() {
                patch.max_wall_seconds = Some(None);
            } else if let Some(n) = v.as_u64() {
                patch.max_wall_seconds = Some(Some(n));
            }
        }

        let updated = match core.update_research(&spec_id, patch).await {
            Ok(s) => s,
            Err(e) => return err(format!("failed to update spec: {e}")),
        };

        if let Some(p) = input.get("paused").and_then(|v| v.as_bool())
            && let Err(e) = core.set_research_paused(&spec_id, p).await
        {
            return err(format!("paused flag updated failed: {e}"));
        }

        // Re-load to reflect the paused flag too.
        let final_spec = core.load_research(&spec_id).await.ok().unwrap_or(updated);

        ok(json!({
            "id": final_spec.id,
            "topic": final_spec.topic,
            "sources": final_spec.sources,
            "interval_seconds": final_spec.interval_seconds,
            "provider": final_spec.provider,
            "model": final_spec.model,
            "max_iterations": final_spec.max_iterations,
            "max_wall_seconds": final_spec.max_wall_seconds,
            "paused": final_spec.paused,
        })
        .to_string())
    }
}

// ─── research_pause / research_resume ───────────────────────────────────────

/// Pause a research spec. Paused specs are skipped by the in-process
/// scheduler but `/research run` and `research_launch` still work — pausing
/// only affects automatic background runs.
pub struct ResearchPauseTool {
    core: Weak<AgentCore>,
}

impl ResearchPauseTool {
    pub fn new(core: Weak<AgentCore>) -> Self {
        Self { core }
    }
}

#[async_trait]
impl Tool for ResearchPauseTool {
    fn spec(&self) -> ToolSpec {
        ToolSpec {
            name: "research_pause".into(),
            description: "Pause a research spec — the background scheduler \
                          will stop launching it on its interval. Manual runs \
                          via /research run or research_launch still work."
                .into(),
            parameters: json!({
                "type": "object",
                "properties": {
                    "spec_id": { "type": "string", "description": "Spec id to pause" }
                },
                "required": ["spec_id"]
            }),
            permission: Permission::WorkspaceWrite,
        }
    }

    async fn execute(&self, input: Value, _cwd: &Path) -> ToolResult {
        let spec_id = match input.get("spec_id").and_then(|v| v.as_str()) {
            Some(id) if !id.is_empty() => id.to_string(),
            _ => return err("missing `spec_id`"),
        };
        let core = match self.core.upgrade() {
            Some(c) => c,
            None => return err("agent core is no longer available"),
        };
        if let Err(e) = core.set_research_paused(&spec_id, true).await {
            return err(format!("failed to pause: {e}"));
        }
        ok(json!({ "spec_id": spec_id, "paused": true }).to_string())
    }
}

/// Resume a paused research spec — re-arms the scheduler immediately.
pub struct ResearchResumeTool {
    core: Weak<AgentCore>,
}

impl ResearchResumeTool {
    pub fn new(core: Weak<AgentCore>) -> Self {
        Self { core }
    }
}

#[async_trait]
impl Tool for ResearchResumeTool {
    fn spec(&self) -> ToolSpec {
        ToolSpec {
            name: "research_resume".into(),
            description: "Resume a paused research spec — re-arms the \
                          background scheduler immediately. No-op if the spec \
                          was not paused."
                .into(),
            parameters: json!({
                "type": "object",
                "properties": {
                    "spec_id": { "type": "string", "description": "Spec id to resume" }
                },
                "required": ["spec_id"]
            }),
            permission: Permission::WorkspaceWrite,
        }
    }

    async fn execute(&self, input: Value, _cwd: &Path) -> ToolResult {
        let spec_id = match input.get("spec_id").and_then(|v| v.as_str()) {
            Some(id) if !id.is_empty() => id.to_string(),
            _ => return err("missing `spec_id`"),
        };
        let core = match self.core.upgrade() {
            Some(c) => c,
            None => return err("agent core is no longer available"),
        };
        if let Err(e) = core.set_research_paused(&spec_id, false).await {
            return err(format!("failed to resume: {e}"));
        }
        ok(json!({ "spec_id": spec_id, "paused": false }).to_string())
    }
}

// ─── research_set_schedule ──────────────────────────────────────────────────

/// Thin tool dedicated to schedule/pause changes — exists alongside
/// [`ResearchUpdateSpecTool`] to give the LLM a more discoverable, narrower
/// surface for the common "change schedule" / "pause for now" intents.
///
/// Semantics:
/// - `interval_seconds: integer` — set polling interval to N seconds.
/// - `interval_seconds: null`    — clear schedule (one-shot only).
/// - `interval_seconds` omitted  — leave schedule unchanged.
/// - `enabled: true`             — unpause (resume scheduling).
/// - `enabled: false`            — pause (skip scheduled runs).
/// - `enabled` omitted           — leave pause flag unchanged.
///
/// At least one of `interval_seconds` / `enabled` must be present.
pub struct ResearchSetScheduleTool {
    core: Weak<AgentCore>,
}

impl ResearchSetScheduleTool {
    pub fn new(core: Weak<AgentCore>) -> Self {
        Self { core }
    }
}

#[async_trait]
impl Tool for ResearchSetScheduleTool {
    fn spec(&self) -> ToolSpec {
        ToolSpec {
            name: "research_set_schedule".into(),
            description: "Set or clear the schedule of a research spec, and/or toggle \
                          whether the scheduler runs it. `interval_seconds: null` clears \
                          the schedule; `enabled: false` pauses; `enabled: true` resumes. \
                          The in-process scheduler is re-armed immediately after the change."
                .into(),
            parameters: json!({
                "type": "object",
                "properties": {
                    "spec_id": { "type": "string", "description": "Spec id to update" },
                    "interval_seconds": {
                        "type": ["integer", "null"],
                        "description": "Polling interval in seconds. Use null to clear; \
                                        omit to leave schedule unchanged."
                    },
                    "enabled": {
                        "type": "boolean",
                        "description": "true → unpause / resume; false → pause. Omit to \
                                        leave the pause flag unchanged."
                    }
                },
                "required": ["spec_id"]
            }),
            permission: Permission::WorkspaceWrite,
        }
    }

    async fn execute(&self, input: Value, _cwd: &Path) -> ToolResult {
        let spec_id = match input.get("spec_id").and_then(|v| v.as_str()) {
            Some(id) if !id.is_empty() => id.to_string(),
            _ => return err("missing `spec_id`"),
        };
        let core = match self.core.upgrade() {
            Some(c) => c,
            None => return err("agent core is no longer available"),
        };

        let interval_field = input.get("interval_seconds");
        let enabled = input.get("enabled").and_then(|v| v.as_bool());

        if interval_field.is_none() && enabled.is_none() {
            return err("at least one of `interval_seconds` or `enabled` must be provided");
        }

        // Apply schedule first (if requested).
        if let Some(v) = interval_field {
            let interval_patch = if v.is_null() {
                Some(None)
            } else if let Some(n) = v.as_u64() {
                Some(Some(n))
            } else {
                return err("`interval_seconds` must be an integer or null");
            };
            let patch = ResearchPatch {
                interval_seconds: interval_patch,
                ..Default::default()
            };
            if let Err(e) = core.update_research(&spec_id, patch).await {
                return err(format!("failed to update schedule: {e}"));
            }
        }

        // Then apply pause flag (if requested).
        if let Some(en) = enabled
            && let Err(e) = core.set_research_paused(&spec_id, !en).await
        {
            return err(format!("failed to toggle pause flag: {e}"));
        }

        let final_spec = match core.load_research(&spec_id).await {
            Ok(s) => s,
            Err(e) => return err(format!("failed to reload spec: {e}")),
        };
        ok(json!({
            "spec_id": final_spec.id,
            "interval_seconds": final_spec.interval_seconds,
            "paused": final_spec.paused,
        })
        .to_string())
    }
}

// ─── research_set_target ────────────────────────────────────────────────────

pub struct ResearchSetTargetTool {
    store: Arc<dyn ResearchStore>,
    context: ResearchContext,
}

impl ResearchSetTargetTool {
    pub fn new(store: Arc<dyn ResearchStore>, context: ResearchContext) -> Self {
        Self { store, context }
    }
}

#[async_trait]
impl Tool for ResearchSetTargetTool {
    fn spec(&self) -> ToolSpec {
        ToolSpec {
            name: "research_set_target".into(),
            description: "Enter inline research mode: sets the active research context so \
                          that subsequent research_save calls in this chat turn persist \
                          findings to the given spec. Pass an empty spec_id to clear. \
                          A clear with zero saves in the current run is REJECTED unless \
                          you pass `force: true` — this prevents the most common idle \
                          failure where the agent visits pages but never persists data. \
                          If genuinely nothing fits the spec (rare), pass force=true with \
                          a brief note explaining why nothing qualified."
                .into(),
            parameters: json!({
                "type": "object",
                "properties": {
                    "spec_id": {
                        "type": "string",
                        "description": "Research spec id to target, or empty string to clear"
                    },
                    "force": {
                        "type": "boolean",
                        "default": false,
                        "description": "Bypass the zero-saves clear guard. Use only when \
                            the page truly contained no qualifying listings."
                    },
                    "note": {
                        "type": "string",
                        "description": "Optional one-line explanation for an empty clear \
                            (logged to the waterfall as a Note event)."
                    }
                },
                "required": ["spec_id"]
            }),
            permission: Permission::WorkspaceWrite,
        }
    }

    async fn execute(&self, input: Value, _cwd: &Path) -> ToolResult {
        let spec_id = input
            .get("spec_id")
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .trim();
        let force = input
            .get("force")
            .and_then(|v| v.as_bool())
            .unwrap_or(false);
        let note = input
            .get("note")
            .and_then(|v| v.as_str())
            .map(str::trim)
            .filter(|s| !s.is_empty());

        if spec_id.is_empty() {
            // Guard against the most common idle-failure mode: agent
            // burns its tool budget on web_fetch / browser_navigate,
            // then "wraps up" by clearing the target without ever
            // calling `research_save`. Refuse, unless `force` is set.
            let saves = self.context.save_count();
            if saves == 0 && !force {
                let active_id = self.context.id().unwrap_or_default();
                if let Some(reg) = self.context.run_events() {
                    reg.push(
                        &active_id,
                        super::run_events::RunEvent::new(
                            super::run_events::EventKind::Note,
                            "research_set_target(clear) refused: zero saves so far".to_string(),
                        ),
                    )
                    .await;
                }
                return err("Cannot clear research context with 0 findings saved. \
                     You browsed pages but never called `research_save`. \
                     Save findings now from the pages you've already opened: \
                     for each qualifying listing call \
                     `research_save({\"url\":..., \"title\":..., \"price\":..., \
                     \"listing_date\":..., \"excerpt\":..., \"source_content\":...})`. \
                     If you genuinely found nothing qualifying (rare — re-check), \
                     retry with `{\"spec_id\":\"\", \"force\": true, \
                     \"note\": \"why nothing matched\"}`."
                    .to_string());
            }
            // Mirror the explicit empty-clear into the waterfall so the
            // operator can see WHY a run produced no findings.
            if let Some(reg) = self.context.run_events() {
                let active_id = self.context.id().unwrap_or_default();
                let label = if let Some(n) = note {
                    format!("agent cleared target (force, saves={saves}): {n}")
                } else {
                    format!("agent cleared target (saves={saves})")
                };
                reg.push(
                    &active_id,
                    super::run_events::RunEvent::new(super::run_events::EventKind::Note, label),
                )
                .await;
            }
            self.context.set_id(None);
            self.context.set_run_id(None);
            self.context.reset_saves();
            return ok("Research context cleared. research_save is no longer active.");
        }

        if let Err(e) = self.store.load_spec(spec_id).await {
            return err(format!("spec `{spec_id}` not found: {e}"));
        }

        let total = self.store.count_findings(spec_id).await.unwrap_or(0);

        let run_id = uuid::Uuid::new_v4().simple().to_string()[..12].to_string();
        self.context.set_id(Some(spec_id.to_string()));
        self.context.set_run_id(Some(run_id.clone()));
        // Fresh bind → fresh save counter. The caller of this tool
        // explicitly opens a new logical run, so previous successes
        // shouldn't satisfy the clear-guard later.
        self.context.reset_saves();

        ok(json!({
            "status": "active",
            "spec_id": spec_id,
            "run_id": run_id,
            "existing_findings": total,
            "hint": "Now use web_fetch to browse pages, then research_save to persist \
                     each finding. Call research_set_target with empty spec_id when done. \
                     The clear is refused if you call it with 0 saves — pass force=true \
                     ONLY when nothing on the visited pages qualified for the spec."
        })
        .to_string())
    }
}

// ─── research_help ──────────────────────────────────────────────────────────

/// Read-only tool that returns a structured markdown explanation of how the
/// research subsystem works (scheduler, semaphore, verification, storage,
/// every other research tool). Use when the user asks "how does research
/// work" / "as it works" / "explain scheduling". For state of a *specific*
/// research prefer [`ResearchMetricsTool`] — `research_help` is meant for
/// the architecture-level question, not the per-spec status one.
pub struct ResearchHelpTool {
    config: ResearchConfig,
}

impl ResearchHelpTool {
    pub fn new(config: ResearchConfig) -> Self {
        Self { config }
    }
}

#[async_trait]
impl Tool for ResearchHelpTool {
    fn spec(&self) -> ToolSpec {
        ToolSpec {
            name: "research_help".into(),
            description: "Return a structured markdown guide explaining how the research \
                          subsystem works: scheduling, concurrency cap, gatekeeper verification, \
                          storage layout, and every research_* tool. Use ONLY for general \
                          'how does the system work' questions — for the current state of a \
                          specific spec, use research_metrics instead."
                .into(),
            parameters: json!({
                "type": "object",
                "properties": {},
                "additionalProperties": false
            }),
            permission: Permission::ReadOnly,
        }
    }

    async fn execute(&self, _input: Value, _cwd: &Path) -> ToolResult {
        ok(super::briefing::full(&self.config))
    }
}

#[cfg(test)]
mod tests {
    //! Integration-style tests for the research-ops tool surface. These
    //! pin the system-level safety guards we rely on to keep the agent
    //! honest — most importantly: `research_set_target` must refuse a
    //! "clear without saving anything" cleanup so we never again ship a
    //! run where the model burned 30 tool calls and produced 0 findings.
    use super::*;
    use crate::research::store::FsResearchStore;
    use chrono::Utc;
    use tempfile::tempdir;

    fn make_spec(id: &str) -> super::super::spec::ResearchSpec {
        super::super::spec::ResearchSpec {
            id: id.into(),
            topic: "topic".into(),
            sources: vec![],
            interval_seconds: None,
            run_at: None,
            cron: None,
            task_timeout_seconds: None,
            session_id: None,
            chat_id: None,
            thread_id: None,
            provider: None,
            model: None,
            max_iterations: None,
            max_wall_seconds: None,
            created_at: Utc::now(),
            paused: false,
            pause_reason: None,
        }
    }

    fn cwd() -> std::path::PathBuf {
        std::env::current_dir().unwrap()
    }

    async fn setup() -> (
        tempfile::TempDir,
        std::sync::Arc<dyn ResearchStore>,
        ResearchContext,
    ) {
        let tmp = tempdir().unwrap();
        let store: std::sync::Arc<dyn ResearchStore> =
            std::sync::Arc::new(FsResearchStore::new(tmp.path().to_path_buf()));
        let ctx = ResearchContext::new();
        (tmp, store, ctx)
    }

    #[tokio::test]
    async fn set_target_bind_resets_save_count_and_returns_active() {
        let (_tmp, store, ctx) = setup().await;
        store.create_spec(&make_spec("s1")).await.unwrap();
        // Pretend a previous run had saves; rebinding must reset.
        ctx.note_save();
        ctx.note_save();
        assert_eq!(ctx.save_count(), 2);

        let tool = ResearchSetTargetTool::new(store.clone(), ctx.clone());
        let out = tool
            .execute(serde_json::json!({"spec_id":"s1"}), &cwd())
            .await;
        assert!(!out.is_error, "got: {}", out.output);
        assert_eq!(
            ctx.save_count(),
            0,
            "fresh bind must reset the save counter so the clear-guard \
             starts from zero for this run"
        );
        assert_eq!(ctx.id().as_deref(), Some("s1"));
    }

    #[tokio::test]
    async fn set_target_clear_with_zero_saves_is_rejected() {
        let (_tmp, store, ctx) = setup().await;
        store.create_spec(&make_spec("s1")).await.unwrap();
        ctx.set_id(Some("s1".into()));
        ctx.set_run_id(Some("run1".into()));
        ctx.reset_saves();

        let tool = ResearchSetTargetTool::new(store.clone(), ctx.clone());
        let out = tool
            .execute(serde_json::json!({"spec_id":""}), &cwd())
            .await;
        assert!(
            out.is_error,
            "clear with 0 saves must error, but got success: {}",
            out.output
        );
        assert!(
            out.output.contains("research_save"),
            "error must point the agent at the save tool: {}",
            out.output
        );
        // Context must remain bound so the agent can recover.
        assert_eq!(ctx.id().as_deref(), Some("s1"));
        assert_eq!(ctx.run_id().as_deref(), Some("run1"));
    }

    #[tokio::test]
    async fn set_target_clear_with_force_succeeds_even_on_zero_saves() {
        let (_tmp, store, ctx) = setup().await;
        store.create_spec(&make_spec("s1")).await.unwrap();
        ctx.set_id(Some("s1".into()));
        ctx.set_run_id(Some("run1".into()));
        ctx.reset_saves();

        let tool = ResearchSetTargetTool::new(store.clone(), ctx.clone());
        let out = tool
            .execute(
                serde_json::json!({
                    "spec_id":"",
                    "force": true,
                    "note": "no listings matched the strict spec on visited pages"
                }),
                &cwd(),
            )
            .await;
        assert!(!out.is_error, "force-clear must succeed: {}", out.output);
        assert_eq!(ctx.id(), None);
        assert_eq!(ctx.run_id(), None);
        assert_eq!(ctx.save_count(), 0);
    }

    #[tokio::test]
    async fn set_target_clear_with_at_least_one_save_succeeds() {
        let (_tmp, store, ctx) = setup().await;
        store.create_spec(&make_spec("s1")).await.unwrap();
        ctx.set_id(Some("s1".into()));
        ctx.set_run_id(Some("run1".into()));
        ctx.reset_saves();
        // Simulate a successful research_save during the run.
        ctx.note_save();

        let tool = ResearchSetTargetTool::new(store.clone(), ctx.clone());
        let out = tool
            .execute(serde_json::json!({"spec_id":""}), &cwd())
            .await;
        assert!(
            !out.is_error,
            "clear after a real save must succeed, got: {}",
            out.output
        );
        assert_eq!(ctx.id(), None);
    }
}
