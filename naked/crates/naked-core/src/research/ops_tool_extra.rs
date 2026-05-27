//! Additional research operation tools (pause, resume, schedule, set-target, help).

use std::path::Path;
use std::sync::{Arc, Weak};

use async_trait::async_trait;
use serde_json::{Value, json};

use super::context::ResearchContext;
use super::store::ResearchStore;
use crate::config::ResearchConfig;
use crate::tool::Tool;
use crate::types::{Permission, ToolResult, ToolSpec};
use crate::{PatchField, ResearchPatch};

pub struct ResearchPauseTool {
    runner: Weak<dyn super::ResearchRunner>,
}

impl ResearchPauseTool {
    pub fn new(runner: Weak<dyn super::ResearchRunner>) -> Self {
        Self { runner }
    }
}

#[async_trait]
impl Tool for ResearchPauseTool {
    fn spec(&self) -> ToolSpec {
        ToolSpec {
            name: "research_pause".into(),
            description: "Pause a research spec — the background scheduler \
                          will stop launching it on its interval. Manual runs \
                          via /research run or research_run still work."
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
            _ => return ToolResult::err("missing `spec_id`"),
        };
        let runner = match self.runner.upgrade() {
            Some(c) => c,
            None => return ToolResult::err("agent core is no longer available"),
        };
        if let Err(e) = runner.set_research_paused(&spec_id, true).await {
            return ToolResult::err(format!("failed to pause: {e}"));
        }
        ToolResult::ok(json!({ "spec_id": spec_id, "paused": true }).to_string())
    }
}

/// Resume a paused research spec — re-arms the scheduler immediately.
pub struct ResearchResumeTool {
    runner: Weak<dyn super::ResearchRunner>,
}

impl ResearchResumeTool {
    pub fn new(runner: Weak<dyn super::ResearchRunner>) -> Self {
        Self { runner }
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
            _ => return ToolResult::err("missing `spec_id`"),
        };
        let runner = match self.runner.upgrade() {
            Some(c) => c,
            None => return ToolResult::err("agent core is no longer available"),
        };
        if let Err(e) = runner.set_research_paused(&spec_id, false).await {
            return ToolResult::err(format!("failed to resume: {e}"));
        }
        ToolResult::ok(json!({ "spec_id": spec_id, "paused": false }).to_string())
    }
}

// ─── research_delete ─────────────────────────────────────────────────────

pub struct ResearchDeleteTool {
    runner: Weak<dyn super::ResearchRunner>,
}

impl ResearchDeleteTool {
    pub fn new(runner: Weak<dyn super::ResearchRunner>) -> Self {
        Self { runner }
    }
}

#[async_trait]
impl Tool for ResearchDeleteTool {
    fn spec(&self) -> ToolSpec {
        ToolSpec {
            name: "research_delete".into(),
            description: "Delete a research spec and all its data (findings, runs, cursors). \
                          Irreversible."
                .into(),
            parameters: json!({
                "type": "object",
                "properties": {
                    "spec_id": {
                        "type": "string",
                        "description": "Research spec ID to delete."
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
        // Verify exists
        if let Err(e) = runner.load_research(&spec_id).await {
            return ToolResult::err(format!("spec `{spec_id}` not found: {e}"));
        }
        if let Err(e) = runner.delete_research(&spec_id).await {
            return ToolResult::err(format!("failed to delete `{spec_id}`: {e}"));
        }
        ToolResult::ok(format!("deleted `{spec_id}` and all its data"))
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
    runner: Weak<dyn super::ResearchRunner>,
}

impl ResearchSetScheduleTool {
    pub fn new(runner: Weak<dyn super::ResearchRunner>) -> Self {
        Self { runner }
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
            _ => return ToolResult::err("missing `spec_id`"),
        };
        let runner = match self.runner.upgrade() {
            Some(c) => c,
            None => return ToolResult::err("agent core is no longer available"),
        };

        let interval_field = input.get("interval_seconds");
        let enabled = input.get("enabled").and_then(|v| v.as_bool());

        if interval_field.is_none() && enabled.is_none() {
            return ToolResult::err(
                "at least one of `interval_seconds` or `enabled` must be provided",
            );
        }

        // Apply schedule first (if requested).
        if let Some(v) = interval_field {
            let interval_patch = if v.is_null() {
                PatchField::Clear
            } else if let Some(n) = v.as_u64() {
                PatchField::Set(n)
            } else {
                return ToolResult::err("`interval_seconds` must be an integer or null");
            };
            let patch = ResearchPatch {
                interval_seconds: interval_patch,
                ..Default::default()
            };
            if let Err(e) = runner.update_research(&spec_id, patch).await {
                return ToolResult::err(format!("failed to update schedule: {e}"));
            }
        }

        // Then apply pause flag (if requested).
        if let Some(en) = enabled
            && let Err(e) = runner.set_research_paused(&spec_id, !en).await
        {
            return ToolResult::err(format!("failed to toggle pause flag: {e}"));
        }

        let final_spec = match runner.load_research(&spec_id).await {
            Ok(s) => s,
            Err(e) => return ToolResult::err(format!("failed to reload spec: {e}")),
        };
        ToolResult::ok(
            json!({
                "spec_id": final_spec.id,
                "interval_seconds": final_spec.interval_seconds,
                "paused": final_spec.paused,
            })
            .to_string(),
        )
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
            permission: Permission::ReadOnly,
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
                return ToolResult::err(
                    "Cannot clear research context with 0 findings saved. \
                     You browsed pages but never called `research_save`. \
                     Save findings now from the pages you've already opened: \
                     for each qualifying listing call \
                     `research_save({\"url\":..., \"title\":..., \"price\":..., \
                     \"listing_date\":..., \"excerpt\":..., \"source_content\":...})`. \
                     If you genuinely found nothing qualifying (rare — re-check), \
                     retry with `{\"spec_id\":\"\", \"force\": true, \
                     \"note\": \"why nothing matched\"}`."
                        .to_string(),
                );
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
            return ToolResult::ok("Research context cleared. research_save is no longer active.");
        }

        if let Err(e) = self.store.load_spec(spec_id).await {
            return ToolResult::err(format!("spec `{spec_id}` not found: {e}"));
        }

        let total = self.store.count_findings(spec_id).await.unwrap_or(0);

        let run_id = uuid::Uuid::new_v4().simple().to_string()[..12].to_string();
        self.context.set_id(Some(spec_id.to_string()));
        self.context.set_run_id(Some(run_id.clone()));
        // Fresh bind → fresh save counter. The caller of this tool
        // explicitly opens a new logical run, so previous successes
        // shouldn't satisfy the clear-guard later.
        self.context.reset_saves();

        ToolResult::ok(
            json!({
                "status": "active",
                "spec_id": spec_id,
                "run_id": run_id,
                "existing_findings": total,
                "hint": "Now use web_fetch to browse pages, then research_save to persist \
                         each finding. Call research_set_target with empty spec_id when done. \
                         The clear is refused if you call it with 0 saves — pass force=true \
                         ONLY when nothing on the visited pages qualified for the spec."
            })
            .to_string(),
        )
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
        ToolResult::ok(super::briefing::full(&self.config))
    }
}

#[cfg(test)]
#[path = "ops_tool_tests.rs"]
mod tests;
