//! Long-running research subsystem: user posts a topic, agent walks the web in
//! the background, accumulates deduplicated findings, emits a rolling report,
//! and can be resumed/repeated on a schedule.
//!
//! Shape: each research has its own directory under `$NAKED_HOME/research/<id>/`
//! with `spec.json`, `findings.jsonl`, `cursor.json`, `report.md`, `runs.jsonl`.
//! Dedup happens on `blake3(normalized_url)` (see `spec::canonicalize_url`).
//! Scheduling is handled out-of-band by a systemd templated unit that invokes
//! `naked research run <id>` — the coordinator itself is intentionally
//! stateless between runs.

pub mod briefing;
pub mod context;
#[path = "coordinator_mod/mod.rs"]
pub mod coordinator;
pub mod filter_rules;
pub mod inflight;
pub mod launch;
pub mod memory_diff;
pub mod ops_tool;
pub mod ops_tool_extra;
pub mod patch;
pub mod quality_assessor;
pub mod reconciler;
pub mod run_events;
pub mod runlog;
pub mod scheduler_hook;
pub mod spec;
pub mod state_view;
pub mod store;
pub mod store_fs;
pub mod tool;
pub mod validators;

pub use context::ResearchContext; // tests/e2e_*.rs
pub use coordinator::{
    AgentRunner, CoordinatorConfig, ResearchCoordinator, RunReport, StopReason, VerifiedRunReport,
    parse_provider_model_pair,
};
pub use inflight::{Inflight, RunState};
pub use ops_tool::{
    ResearchCreateTool, ResearchFindingsTool, ResearchListSpecsTool, ResearchMetricsTool,
    ResearchUpdateSpecTool,
};
pub use ops_tool_extra::{
    ResearchHelpTool, ResearchPauseTool, ResearchResumeTool, ResearchSetScheduleTool,
    ResearchSetTargetTool,
};
pub use patch::{PatchField, ResearchPatch, apply_research_patch};
pub(crate) use run_events::RunEventRegistry; // used by services/research + research_ops
pub use run_events::{EventKind, RunEvent};
pub use runlog::write_research_memory_link_for;
pub(crate) use scheduler_hook::noop_hook; // internal: services/research
pub use scheduler_hook::{NoopSchedulerHook, SchedulerEvent, SchedulerHook};
pub(crate) use spec::new_research_id; // used by research_ops / services
pub use spec::{Finding, ResearchSpec, RunRecord, canonicalize_url, dedup_hash};
pub use state_view::{StateView, render_state};
pub use store::{
    FindingStore, InflightStore, ReportStore, ResearchStore, RunStore, SpecStore, research_root,
    research_runlog_path,
};
// ArtifactStore stays pub(crate) at definition site (no external consumers).
pub use store_fs::FsResearchStore;
pub use tool::{ResearchListTool, ResearchSaveCursorTool, ResearchSaveTool, ResearchStatusTool};
// scan_and_redact is pub(crate); callers use `tool::redact::scan_and_redact`.

// ── ResearchRunner trait (ISP: tools see only what they need) ────────────

/// Minimal interface for research tools that need to trigger runs
/// or modify specs. Decouples tool structs from AgentCore.
#[async_trait::async_trait]
pub trait ResearchRunner: Send + Sync {
    async fn load_research(&self, id: &str) -> crate::error::Result<ResearchSpec>;
    async fn set_research_paused(&self, id: &str, paused: bool) -> crate::error::Result<()>;
    async fn update_research(
        &self,
        id: &str,
        patch: crate::research::patch::ResearchPatch,
    ) -> crate::error::Result<ResearchSpec>;
    async fn run_research(&self, id: &str) -> crate::error::Result<RunReport>;
    async fn run_research_verified(
        &self,
        id: &str,
        max_rounds: u32,
    ) -> crate::error::Result<VerifiedRunReport>;
    fn research_verify_config(&self) -> (bool, u32);
    async fn delete_research(&self, id: &str) -> crate::error::Result<()>;
}
