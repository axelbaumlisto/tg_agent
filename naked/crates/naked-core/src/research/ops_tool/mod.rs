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

pub mod create;
pub mod metrics;
pub mod runner;

pub use create::{ResearchCreateTool, ResearchListSpecsTool};
pub use metrics::{ResearchFindingsTool, ResearchMetricsTool};
pub use runner::ResearchUpdateSpecTool;
// ResearchLaunchTool removed (PLAN_UNIFIED_TURN_v1) — use research_run instead.
