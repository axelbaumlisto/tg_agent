//! `research_launch` and `research_update_spec` tool implementations.

use std::path::Path;
use std::sync::Weak;

use async_trait::async_trait;
use serde_json::{Value, json};

use crate::tool::Tool;
use crate::types::{Permission, ToolResult, ToolSpec};
use crate::{PatchField, ResearchPatch};

use super::super::ResearchRunner;

// ─── research_launch ────────────────────────────────────────────────────────

pub struct ResearchLaunchTool {
    runner: Weak<dyn ResearchRunner>,
}

impl ResearchLaunchTool {
    pub fn new(runner: Weak<dyn ResearchRunner>) -> Self {
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
    runner: Weak<dyn ResearchRunner>,
}

impl ResearchUpdateSpecTool {
    pub fn new(runner: Weak<dyn ResearchRunner>) -> Self {
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
