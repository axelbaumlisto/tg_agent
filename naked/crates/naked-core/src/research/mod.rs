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
pub mod coordinator;
pub mod filter_rules;
pub mod inflight;
pub mod launch;
pub mod memory_diff;
pub mod ops_tool;
pub mod quality_assessor;
pub mod reconciler;
pub mod run_events;
pub mod scheduler_hook;
pub mod spec;
pub mod state_view;
pub mod store;
pub mod tool;
pub mod validators;

pub use coordinator::{
    AgentRunner, CoordinatorConfig, ResearchCoordinator, RunReport, StopReason, VerifiedRunReport,
    parse_provider_model_pair,
};
pub use inflight::{Inflight, RunState};
pub use ops_tool::{
    ResearchCreateTool, ResearchFindingsTool, ResearchHelpTool, ResearchLaunchTool,
    ResearchListSpecsTool, ResearchMetricsTool, ResearchPauseTool, ResearchResumeTool,
    ResearchSetScheduleTool, ResearchSetTargetTool, ResearchUpdateSpecTool,
};
pub use run_events::{EventKind, RunEvent, RunEventRegistry};
pub use scheduler_hook::{NoopSchedulerHook, SchedulerEvent, SchedulerHook, noop_hook};
pub use spec::{
    Cursor, Finding, ResearchSpec, RunRecord, canonicalize_url, dedup_hash, new_research_id,
    normalize_title_for_similarity, titles_are_similar,
};
pub use state_view::{StateView, render_state};
pub use store::{FsResearchStore, ResearchStore, research_root};
pub use tool::{
    ResearchContext, ResearchListTool, ResearchSaveCursorTool, ResearchSaveTool,
    ResearchStatusTool, scan_and_redact,
};
