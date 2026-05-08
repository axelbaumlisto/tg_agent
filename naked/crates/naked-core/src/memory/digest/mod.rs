//! Daily-memory digest: turns yesterday's draft entries into a
//! `DigestPlan` (promote / reject / summarize) and applies it
//! atomically to `MEMORY.md` + `DREAMS.md`.
//!
//! Pipeline (one scope per call):
//!
//! 1. Read all daily-draft files in the lookback window
//!    (`memory/YYYY-MM-DD.md`).
//! 2. Compute deterministic scoring hints per draft entry:
//!    `repeat_days` (how many distinct daily files contain a matching
//!    `content_hash`), `recall_count` (best-effort, reserved for
//!    future instrumentation — currently always 0), `age_days`,
//!    `source_diversity` (number of distinct `source` strings).
//! 3. Optionally call the LLM with the draft list + scores asking for
//!    a one-paragraph human summary. Falls back to a deterministic
//!    summary on LLM failure — the digest must never silently break.
//! 4. Decide promotions (`scoring_promote`) using config gates
//!    (`promote_min_repeat_days`, `promote_min_recall_count`).
//! 5. Build a `DigestPlan` and apply it: append a `DreamEntry`,
//!    optionally append promoted entries to `MEMORY.md` (when
//!    `daily_mode == "summarize_and_promote"`).

pub mod compact;
pub mod extract;
pub mod format;
pub mod score;

pub use compact::{CompactionResult, compact_memory, compact_memory_at};
pub use extract::{
    collect_window, extract_and_append, flush_recall_to_disk, pre_compaction_flush, session_close,
};
pub use format::{apply_plan, deterministic_summary, last_digest_summary, summarize_with_llm};
pub use score::{effective_recall, partition_candidates, pick_compaction_victims};

use std::path::Path;

use chrono::NaiveDate;

use super::store::MarkdownMemoryStore;
use super::types::{MemoryEntry, MemoryScope};
use crate::config::MemoryConfig;
use crate::provider::Provider;

/// Mode of the daily digest, mirrors `MemoryConfig.daily_mode`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DigestMode {
    /// Only write a `DreamEntry`. `MEMORY.md` is never touched.
    SummarizeOnly,
    /// Write a `DreamEntry` AND promote scoring winners to `MEMORY.md`.
    SummarizeAndPromote,
}

impl DigestMode {
    pub fn from_str_or_default(s: &str) -> Self {
        match s {
            "summarize_only" => Self::SummarizeOnly,
            _ => Self::SummarizeAndPromote,
        }
    }
}

/// Deterministic scoring hints attached to a draft entry. Computed
/// from the draft files alone — no LLM, no recall instrumentation
/// required.
#[derive(Debug, Clone, Default)]
pub struct ScoringHints {
    /// Number of distinct daily files in the lookback window that
    /// contain a matching `content_hash`. Higher = the rule kept
    /// reappearing across days.
    pub repeat_days: u32,
    /// Number of distinct `source` strings across the matching
    /// occurrences. Higher = the rule was suggested by multiple
    /// independent paths (user, classifier, model).
    pub source_diversity: u32,
    /// Days since the entry was first seen.
    pub age_days: u32,
    /// Best-effort recall count. Reserved for future instrumentation;
    /// always 0 today.
    pub recall_count: u32,
    /// How many raw occurrences (including wording variants) mapped
    /// to this fingerprint. Higher = user keeps repeating this rule.
    pub reinforcements: u32,
}

/// One promotion candidate emitted by `scoring_promote`.
#[derive(Debug, Clone)]
pub struct PromoteEntry {
    pub entry: MemoryEntry,
    pub hints: ScoringHints,
}

/// One rejected candidate emitted by `scoring_promote`.
#[derive(Debug, Clone)]
pub struct RejectedEntry {
    pub entry: MemoryEntry,
    pub hints: ScoringHints,
    pub reason: String,
}

/// Final plan handed to `apply_plan`.
#[derive(Debug, Clone)]
pub struct DigestPlan {
    pub scope: MemoryScope,
    pub mode: DigestMode,
    /// Human-readable summary (LLM-generated when available, otherwise
    /// a deterministic fallback).
    pub summary: String,
    pub promoted: Vec<PromoteEntry>,
    pub rejected: Vec<RejectedEntry>,
}

/// Outcome of `apply_plan`. Returned for logging/observability.
#[derive(Debug, Default)]
pub struct ApplyOutcome {
    pub promoted_written: usize,
    pub dreams_appended: bool,
    /// Recall counters merged into MEMORY.md from in-memory hits.
    /// Bumped lazily inside `run_daily` (not `apply_plan`) — kept
    /// here so observers see all per-run side-effects in one struct.
    pub recall_persisted: usize,
    /// How many entries were folded into a compaction summary.
    pub compacted_in: usize,
    /// How many compaction summaries were written (one per fired
    /// section).
    pub compacted_out: usize,
}

/// Convenience: resolve the `MEMORY.md` path for a scope without
/// re-implementing the scope path logic at every call site.
pub(crate) fn scope_memory_path(workspace: &Path, scope: &MemoryScope) -> std::path::PathBuf {
    match scope {
        MemoryScope::Global => MarkdownMemoryStore::global_memory_path(),
        MemoryScope::Project => MarkdownMemoryStore::project_memory_path(workspace),
        MemoryScope::User(id) => MarkdownMemoryStore::user_memory_path(id),
    }
}

/// Last-resort merge when the LLM is unavailable or returns garbage:
/// concatenate the bullets, truncate to `max_chars` bytes. Kept in
/// `mod.rs` so the companion test suite can reach it without crossing
/// module boundaries (tests use `use super::*` which sees private
/// items of the direct parent).
fn deterministic_merge(victims: &[MemoryEntry], max_chars: usize) -> String {
    let joined = victims
        .iter()
        .map(|e| e.content.trim())
        .collect::<Vec<_>>()
        .join("; ");
    let summary = format!("[merged] {}", joined);
    compact::truncate_with_ellipsis(summary, max_chars)
}

/// Build a `DigestPlan` end-to-end (read drafts → score → partition →
/// summarize). Pulled out of `run_daily` so it can be unit-tested
/// without disk I/O for the LLM step.
pub async fn build_plan(
    workspace: &Path,
    scope: &MemoryScope,
    cfg: &MemoryConfig,
    today: NaiveDate,
    llm: Option<(&dyn Provider, &str)>,
) -> DigestPlan {
    let candidates = collect_window(workspace, scope, today, cfg.recent_shift_days.max(1));
    let (promoted, rejected) = partition_candidates(candidates, cfg);

    let summary = match llm {
        Some((provider, model)) => {
            match summarize_with_llm(provider, model, &promoted, &rejected, cfg).await {
                Ok(s) => s,
                Err(e) => {
                    tracing::warn!(
                        "digest LLM summary failed: {e:#}; using deterministic fallback"
                    );
                    deterministic_summary(&promoted, &rejected)
                }
            }
        }
        None => deterministic_summary(&promoted, &rejected),
    };

    DigestPlan {
        scope: scope.clone(),
        mode: DigestMode::from_str_or_default(&cfg.daily_mode),
        summary,
        promoted,
        rejected,
    }
}

// Test-only imports exposed to the child `tests` module via
// `use super::*`. Child modules see all items (including private) in
// their direct parent's namespace.
#[cfg(test)]
use {super::types::MemoryType, chrono::Utc};

#[cfg(test)]
#[path = "digest_tests.rs"]
mod tests;
