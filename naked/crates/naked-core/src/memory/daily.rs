//! Daily-digest orchestration. The cron scheduler in `naked-tg` calls
//! `run_daily(...)` at the configured time; everything below is pure
//! `naked-core` so unit tests can exercise it without spinning up the
//! Telegram bot.
//!
//! Idempotence: `should_run_today` reads a small lock-file
//! (`memory/.last_digest`) so duplicate triggers in the same UTC day
//! are no-ops. The lock is written *after* a successful run.
//!
//! Recall-count instrumentation: `record_recall` is exposed for
//! callers who want to bump the in-memory recall counter for an
//! entry. Today the counter is best-effort and lives in a process-
//! local map; it survives restarts only via the `recall:N` metadata
//! tag we attach to entries on next save. Promotion gates can ignore
//! the count entirely (`promote_min_recall_count = 0`, the default).

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Mutex;
use std::sync::OnceLock;

use chrono::{NaiveDate, Utc};

use super::digest::{self, ApplyOutcome, DigestPlan};
use super::dreams;
use super::store::{MarkdownMemoryStore, scope_memory_dir};
use super::types::MemoryScope;
use crate::config::MemoryConfig;
use crate::provider::Provider;

/// Lock-file name written after a successful digest run.
const LOCK_FILENAME: &str = ".last_digest";

/// Process-local, best-effort recall counter. Maps `(scope_str,
/// content_hash)` → recent recall counts. Cleared on restart — the
/// daily digest is always free to fall back to `repeat_days` alone.
static RECALL_COUNTERS: OnceLock<Mutex<HashMap<String, u32>>> = OnceLock::new();

fn recall_map() -> &'static Mutex<HashMap<String, u32>> {
    RECALL_COUNTERS.get_or_init(|| Mutex::new(HashMap::new()))
}

fn recall_key(scope: &MemoryScope, content: &str) -> String {
    use super::store::content_hash;
    format!("{}:{:016x}", scope, content_hash(content))
}

/// Bump the in-memory recall counter for `content` in `scope`. Called
/// from the prompt-injection path each time an entry is included in
/// the system prompt.
pub fn record_recall(scope: &MemoryScope, content: &str) {
    let mut m = recall_map().lock().expect("recall map poisoned");
    *m.entry(recall_key(scope, content)).or_insert(0) += 1;
}

/// Read and clear the in-memory recall counter for one entry. Used by
/// `run_daily` when assembling scoring hints.
pub fn take_recall(scope: &MemoryScope, content: &str) -> u32 {
    let mut m = recall_map().lock().expect("recall map poisoned");
    m.remove(&recall_key(scope, content)).unwrap_or(0)
}

/// Path of the `.last_digest` lock file for `scope`.
fn lock_path(workspace: &Path, scope: &MemoryScope) -> PathBuf {
    scope_memory_dir(workspace, scope).join(LOCK_FILENAME)
}

/// Returns `true` when no successful digest has run today (UTC) yet.
pub fn should_run_today(workspace: &Path, scope: &MemoryScope) -> bool {
    let path = lock_path(workspace, scope);
    // REGISTRY-WAIVE: intentional fallback: missing path → empty result
    let Ok(raw) = std::fs::read_to_string(&path) else {
        return true;
    };
    let Some(last) = raw
        .lines()
        .next()
        .and_then(|s| NaiveDate::parse_from_str(s.trim(), "%Y-%m-%d").ok())
    else {
        return true;
    };
    last < Utc::now().date_naive()
}

/// Mark today as digested for `scope`.
pub fn mark_ran_today(workspace: &Path, scope: &MemoryScope) -> std::io::Result<()> {
    let path = lock_path(workspace, scope);
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let body = format!("{}\n", Utc::now().date_naive().format("%Y-%m-%d"));
    let tmp = path.with_extension("tmp");
    std::fs::write(&tmp, body)?;
    std::fs::rename(&tmp, &path)?;
    Ok(())
}

/// Top-level entry point used by the scheduler. Runs the digest
/// pipeline once for `scope`, applies the plan, rotates old daily
/// files, and updates the lock file. Returns `Ok(None)` when the
/// digest was skipped because it already ran today.
pub async fn run_daily(
    workspace: &Path,
    scope: &MemoryScope,
    cfg: &MemoryConfig,
    llm: Option<(&dyn Provider, &str)>,
) -> std::io::Result<Option<(DigestPlan, ApplyOutcome)>> {
    if !cfg.daily_enabled {
        tracing::debug!(scope = %scope, "memory daily disabled by config; skipping");
        return Ok(None);
    }
    if !should_run_today(workspace, scope) {
        tracing::debug!(scope = %scope, "memory daily already ran today; skipping");
        return Ok(None);
    }

    let today = Utc::now().date_naive();

    // Step 0: persist any pending in-memory recall counters into
    // `MEMORY.md` so promotion/forgetting decisions can use them.
    let recall_persisted = match digest::flush_recall_to_disk(workspace, scope) {
        Ok(n) => n,
        Err(e) => {
            tracing::warn!(scope = %scope, "flush_recall_to_disk failed: {e:#}");
            0
        }
    };

    let plan = digest::build_plan(workspace, scope, cfg, today, llm).await;

    let mut outcome = digest::apply_plan(workspace, &plan, cfg)?;
    outcome.recall_persisted = recall_persisted;

    // Step 4: forgetting via compaction. Old, low-recall entries
    // get folded 5→1. Best-effort: any error is logged, not fatal.
    match digest::compact_memory(workspace, scope, cfg, today, llm).await {
        Ok(results) if !results.is_empty() => {
            let in_n: usize = results.iter().map(|r| r.victims.len()).sum();
            let out_n = results.iter().filter(|r| r.summary.is_some()).count();
            outcome.compacted_in = in_n;
            outcome.compacted_out = out_n;
            // Append a single dream entry summarising the compaction
            // so the operator can audit it from `/memory dreams`.
            let summary_text = format!(
                "[compaction] folded {} entries into {} summary line(s)",
                in_n, out_n,
            );
            let promoted_text: Vec<String> = results
                .iter()
                .filter_map(|r| r.summary.as_ref().map(|s| s.content.clone()))
                .collect();
            let rejected_items: Vec<dreams::RejectedItem> = results
                .iter()
                .flat_map(|r| {
                    r.victims.iter().map(|v| dreams::RejectedItem {
                        content: v.content.clone(),
                        reason: Some(format!(
                            "compacted into {}",
                            r.summary.as_ref().map(|s| s.id.clone()).unwrap_or_default()
                        )),
                    })
                })
                .collect();
            let dream =
                dreams::build_entry(scope.clone(), &summary_text, promoted_text, rejected_items);
            if let Err(e) = dreams::append_dream(workspace, &dream, cfg.dreams_retention_days) {
                tracing::warn!(scope = %scope, "compaction dream append failed: {e:#}");
            }
            tracing::info!(
                scope = %scope,
                folded_in = in_n,
                summaries = out_n,
                "memory compaction applied"
            );
        }
        Ok(_) => {}
        Err(e) => tracing::warn!(scope = %scope, "compact_memory failed: {e:#}"),
    }

    // Rotate old daily files (best-effort).
    match MarkdownMemoryStore::rotate_daily_files(workspace, scope, cfg.daily_retention_days) {
        Ok(n) if n > 0 => tracing::info!(scope = %scope, removed = n, "rotated old daily files"),
        Ok(_) => {}
        Err(e) => tracing::warn!(scope = %scope, "daily rotation failed: {e:#}"),
    }

    if let Err(e) = mark_ran_today(workspace, scope) {
        tracing::warn!(scope = %scope, "failed to update digest lock file: {e:#}");
    }

    tracing::info!(
        scope = %scope,
        promoted = outcome.promoted_written,
        rejected = plan.rejected.len(),
        "memory daily digest applied"
    );
    Ok(Some((plan, outcome)))
}

/// Build a "Recent shift" string for prompt injection — concatenation
/// of the last `cfg.recent_shift_days` daily-draft files for `scope`,
/// trimmed to `cfg.recent_shift_max_chars`. Returns `None` when there
/// is nothing recent to show.
pub fn recent_shift_block(
    workspace: &Path,
    scope: &MemoryScope,
    cfg: &MemoryConfig,
) -> Option<String> {
    let today = Utc::now().date_naive();
    let mut out = String::from("[Memory — recent shift]\n");
    let mut wrote_any = false;
    for delta in 0..cfg.recent_shift_days.max(1) {
        let date = today - chrono::Duration::days(delta as i64);
        let entries = MarkdownMemoryStore::read_daily(workspace, scope, date);
        if entries.is_empty() {
            continue;
        }
        out.push_str(&format!("\n{}:\n", date));
        for e in entries {
            out.push_str(&format!("- {}\n", e.content));
            // Bump recall counter: the prompt is about to include this
            // line so we treat it as "recalled once".
            record_recall(scope, &e.content);
            wrote_any = true;
        }
    }
    if !wrote_any {
        return None;
    }
    if out.len() > cfg.recent_shift_max_chars {
        out.truncate(out.floor_char_boundary(cfg.recent_shift_max_chars));
        out.push_str("\n…[truncated]");
    }
    Some(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn proj() -> MemoryScope {
        MemoryScope::Project
    }

    #[test]
    fn recall_counter_round_trips() {
        let scope = MemoryScope::User("test_recall_user".into());
        record_recall(&scope, "rule X");
        record_recall(&scope, "rule X");
        assert_eq!(take_recall(&scope, "rule X"), 2);
        // Subsequent take returns 0 (we cleared on read).
        assert_eq!(take_recall(&scope, "rule X"), 0);
    }

    #[test]
    fn should_run_today_when_lock_missing() {
        let dir = tempfile::tempdir().unwrap();
        let workspace = dir.path();
        let scope = proj();
        let mem_dir = scope_memory_dir(workspace, &scope);
        // Skip when the resolved path is outside the tempdir (real
        // NAKED_HOME is in play and we can't safely poke at it).
        if !mem_dir.starts_with(dir.path()) {
            return;
        }
        assert!(should_run_today(workspace, &scope));
    }

    #[test]
    fn mark_then_should_not_run() {
        let dir = tempfile::tempdir().unwrap();
        let workspace = dir.path();
        let scope = proj();
        let mem_dir = scope_memory_dir(workspace, &scope);
        if !mem_dir.starts_with(dir.path()) {
            return;
        }
        std::fs::create_dir_all(&mem_dir).unwrap();
        mark_ran_today(workspace, &scope).unwrap();
        assert!(!should_run_today(workspace, &scope));
    }
}
