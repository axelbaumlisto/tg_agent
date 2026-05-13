//! `research_metrics` and `research_findings` tool implementations.

use std::path::Path;
use std::sync::Arc;

use async_trait::async_trait;
use chrono::Utc;
use serde_json::{Value, json};

use crate::tool::Tool;
use crate::types::{Permission, ToolResult, ToolSpec};

use super::super::store::ResearchStore;
use super::super::tool::output::parse_listing_date;

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
