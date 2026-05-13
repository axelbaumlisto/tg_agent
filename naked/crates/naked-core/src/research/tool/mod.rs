// REGISTRY-WAIVE: B45 (dead-code revealed by Phase D' pub→pub(crate) flip).
// PLAN_SKILL_VS_CORE_v1 audit found 55 items in research/ never used inside
// the crate; they were hidden by `pub` visibility (dead_code lint exempts
// pub items). Audit + delete is queued as separate B45 cleanup task.
// Until then, this allow keeps the clippy gate green.
#![allow(dead_code)]

//! Tools the research agent calls during a run.
//!
//! - `research_save` — persist a new `Finding`. Idempotent thanks to
//!   `dedup_hash(url)`; duplicates return `duplicate:true` without an error.
//! - `research_list` — dedup-list of already-stored URLs so the model can skip
//!   them in the current turn (the coordinator prompt already carries 50, but
//!   an explicit tool call is useful for long runs).
//! - `research_status` — explicit-id lookup used by the `/research ask`
//!   conversational flow in Telegram: the model is given the research id and
//!   pulls spec + last findings + last run in one call.
//! - `research_save_cursor` — opaque JSON state for resumable pagination.
//!
//! All four mirror the ambient-context pattern from `tool::memory::MemoryTool`
//! (`ResearchContext::id` is set by the coordinator right before it kicks off
//! the turn, and cleared afterwards). Without a context set, save/list/cursor
//! return an error — never guess which research to write to.

use std::sync::Arc;

use super::context::ResearchContext;
use super::store::ResearchStore;

// These three imports are only needed for the test module (`tool_tests.rs`)
// which accesses them via `use super::*;`.  They are not referenced in
// production code within this module, so we gate them to `#[cfg(test)]` and
// suppress the "unused import" lint that fires because mod.rs itself never
// names them directly.
#[cfg(test)]
use crate::tool::Tool;
#[cfg(test)]
use chrono::Utc;
#[cfg(test)]
use serde_json::json;

mod handlers;
pub(crate) mod output;
pub(crate) mod redact;

// `parse_listing_date` accessed via `crate::research::tool::output::X`,
// `scan_and_redact` accessed via `super::redact::scan_and_redact` in tests.
// No module-level re-exports needed.
// Used only by the test module via `use super::*;`.
#[cfg(test)]
pub(crate) use output::strip_source_attribution;

const MAX_LISTING_AGE_DAYS: i64 = 90;

/// `research_save` — save or update a finding, with configurable quality warnings.
pub struct ResearchSaveTool {
    store: Arc<dyn ResearchStore>,
    context: ResearchContext,
    gk: crate::config::GatekeeperConfig,
}

impl ResearchSaveTool {
    pub fn new(
        store: Arc<dyn ResearchStore>,
        context: ResearchContext,
        gk: crate::config::GatekeeperConfig,
    ) -> Self {
        Self { store, context, gk }
    }
}

/// `research_list` — return already-known URLs so the agent can self-dedup
/// mid-run (the coordinator prompt carries the first 50, this tool is for
/// when the research grows beyond that).
pub struct ResearchListTool {
    store: Arc<dyn ResearchStore>,
    context: ResearchContext,
}

impl ResearchListTool {
    pub fn new(store: Arc<dyn ResearchStore>, context: ResearchContext) -> Self {
        Self { store, context }
    }
}

/// `research_save_cursor` — opaque state blob the agent writes to resume
/// pagination between runs. Replaces the previous cursor atomically.
pub struct ResearchSaveCursorTool {
    store: Arc<dyn ResearchStore>,
    context: ResearchContext,
}

impl ResearchSaveCursorTool {
    pub fn new(store: Arc<dyn ResearchStore>, context: ResearchContext) -> Self {
        Self { store, context }
    }
}

/// `research_status` — explicit-id lookup. Unlike the three tools above it does
/// NOT need an ambient context, so the Telegram `/research ask` flow can hand
/// a specific id to the agent and ask for a reasoned reply. Returns a compact
/// markdown block: topic, counts, last 10 findings, last 3 runs.
pub struct ResearchStatusTool {
    store: Arc<dyn ResearchStore>,
}

impl ResearchStatusTool {
    pub fn new(store: Arc<dyn ResearchStore>) -> Self {
        Self { store }
    }
}

#[cfg(test)]
#[path = "../tool_tests.rs"]
mod tests;
