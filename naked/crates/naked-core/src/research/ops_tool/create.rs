//! `research_create` and `research_list_specs` tool implementations.

use std::path::Path;
use std::sync::Arc;

use async_trait::async_trait;
use chrono::Utc;
use serde_json::{Value, json};

use crate::config::ResearchConfig;
use crate::tool::Tool;
use crate::types::{Permission, ToolResult, ToolSpec};

use super::super::spec::{ResearchSpec, new_research_id};
use super::super::store::ResearchStore;

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
