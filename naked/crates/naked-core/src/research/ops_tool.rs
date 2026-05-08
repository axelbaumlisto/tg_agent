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
use crate::{PatchField, ResearchPatch};

use super::spec::{ResearchSpec, new_research_id};
use super::store::ResearchStore;
use super::tool::parse_listing_date;

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
            _ => return ToolResult::err("missing or empty `topic`"),
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
            Ok(()) => ToolResult::ok(
                json!({
                    "id": spec.id,
                    "topic": spec.topic,
                    "sources": sources,
                    "hint": "Use research_launch to start a deep background run, or \
                             research_set_target to research inline in this chat."
                })
                .to_string(),
            ),
            Err(e) => ToolResult::err(format!("failed to create spec: {e}")),
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
            Err(e) => return ToolResult::err(format!("failed to list specs: {e}")),
        };
        if specs.is_empty() {
            return ToolResult::ok("No research specs defined. Use research_create to start one.");
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
        ToolResult::ok(Value::Array(items).to_string())
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
            _ => return ToolResult::err("missing `spec_id`"),
        };
        let runs_limit = input
            .get("runs_limit")
            .and_then(|v| v.as_u64())
            .map(|v| v as usize)
            .unwrap_or(self.runs_limit_default)
            .max(1);

        let spec = match self.store.load_spec(spec_id).await {
            Ok(s) => s,
            Err(e) => return ToolResult::err(format!("spec `{spec_id}` not found: {e}")),
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

        ToolResult::ok(
            json!({
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
            .to_string(),
        )
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
            _ => return ToolResult::err("missing `spec_id`"),
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
            return ToolResult::err(format!("spec `{spec_id}` not found"));
        }

        let all = match self.store.list_findings(spec_id, None).await {
            Ok(f) => f,
            Err(e) => return ToolResult::err(format!("failed to load findings: {e}")),
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

        ToolResult::ok(
            json!({
                "spec_id": spec_id,
                "total_findings": total,
                "fresh_findings": fresh_count,
                "showing": shown,
                "skipped_stale": skipped_stale,
                "max_age_days": max_age_days,
                "findings": filtered,
            })
            .to_string(),
        )
    }
}

// ─── research_launch ────────────────────────────────────────────────────────

pub struct ResearchLaunchTool {
    runner: Weak<dyn super::ResearchRunner>,
}

impl ResearchLaunchTool {
    pub fn new(runner: Weak<dyn super::ResearchRunner>) -> Self {
        Self { runner }
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
            _ => return ToolResult::err("missing `spec_id`"),
        };

        let runner = match self.runner.upgrade() {
            Some(c) => c,
            None => return ToolResult::err("agent core is no longer available"),
        };

        if let Err(e) = runner.load_research(&spec_id).await {
            return ToolResult::err(format!("spec `{spec_id}` not found: {e}"));
        }

        let id = spec_id.clone();
        let (verify, max_rounds) = runner.research_verify_config();
        tokio::spawn(async move {
            if verify {
                match runner.run_research_verified(&id, max_rounds).await {
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
                match runner.run_research(&id).await {
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
        ToolResult::ok(
            json!({
                "status": "launched",
                "spec_id": spec_id,
                "verified": verify,
                "hint": hint,
            })
            .to_string(),
        )
    }
}

// ─── research_update_spec ───────────────────────────────────────────────────

/// LLM-facing partial-update tool. Maps to [`AgentCore::update_research`].
/// Every parameter is optional; absent ones leave the spec unchanged.
pub struct ResearchUpdateSpecTool {
    runner: Weak<dyn super::ResearchRunner>,
}

impl ResearchUpdateSpecTool {
    pub fn new(runner: Weak<dyn super::ResearchRunner>) -> Self {
        Self { runner }
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
            _ => return ToolResult::err("missing `spec_id`"),
        };
        let runner = match self.runner.upgrade() {
            Some(c) => c,
            None => return ToolResult::err("agent core is no longer available"),
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
            patch.interval_seconds = PatchField::Clear;
        } else if let Some(v) = input.get("interval_seconds") {
            if v.is_null() {
                patch.interval_seconds = PatchField::Clear;
            } else if let Some(n) = v.as_u64() {
                patch.interval_seconds = PatchField::Set(n);
            } else {
                return ToolResult::err("`interval_seconds` must be an integer or null");
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
                patch.max_iterations = PatchField::Clear;
            } else if let Some(n) = v.as_u64() {
                patch.max_iterations = PatchField::Set(n as u32);
            }
        }
        if let Some(v) = input.get("max_wall_seconds") {
            if v.is_null() {
                patch.max_wall_seconds = PatchField::Clear;
            } else if let Some(n) = v.as_u64() {
                patch.max_wall_seconds = PatchField::Set(n);
            }
        }

        let updated = match runner.update_research(&spec_id, patch).await {
            Ok(s) => s,
            Err(e) => return ToolResult::err(format!("failed to update spec: {e}")),
        };

        if let Some(p) = input.get("paused").and_then(|v| v.as_bool())
            && let Err(e) = runner.set_research_paused(&spec_id, p).await
        {
            return ToolResult::err(format!("paused flag updated failed: {e}"));
        }

        // Re-load to reflect the paused flag too.
        let final_spec = runner.load_research(&spec_id).await.ok().unwrap_or(updated);

        ToolResult::ok(
            json!({
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
            .to_string(),
        )
    }
}
