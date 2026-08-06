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
use std::sync::MutexGuard;
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

/// Gauge sentinel for scopes whose digest marker is missing or unparsable.
/// This must be visibly stale rather than 0/fresh, otherwise the B97 detector
/// would miss a digest that never wrote its persisted marker.
pub const MISSING_DIGEST_DAYS_SENTINEL: u64 = 9_999;

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

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum RecallCounterError {
    Poisoned,
}

impl std::fmt::Display for RecallCounterError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Poisoned => f.write_str("recall map mutex poisoned"),
        }
    }
}

fn recall_map_lock() -> Result<MutexGuard<'static, HashMap<String, u32>>, RecallCounterError> {
    recall_map().lock().map_err(|e| {
        drop(e.into_inner());
        RecallCounterError::Poisoned
    })
}

/// Bump the in-memory recall counter for `content` in `scope`. Called
/// from the prompt-injection path each time an entry is included in
/// the system prompt.
pub fn record_recall(scope: &MemoryScope, content: &str) {
    if let Err(e) = try_record_recall(scope, content) {
        tracing::debug!(scope = %scope, error = %e, "memory recall counter skipped");
    }
}

pub(crate) fn try_record_recall(
    scope: &MemoryScope,
    content: &str,
) -> Result<(), RecallCounterError> {
    let mut m = recall_map_lock()?;
    *m.entry(recall_key(scope, content)).or_insert(0) += 1;
    Ok(())
}

/// Read and clear the in-memory recall counter for one entry. Used by
/// `run_daily` when assembling scoring hints.
pub fn take_recall(scope: &MemoryScope, content: &str) -> u32 {
    let mut m = recall_map().lock().unwrap_or_else(|e| e.into_inner());
    m.remove(&recall_key(scope, content)).unwrap_or(0)
}

/// Path of the `.last_digest` lock file for `scope`.
fn lock_path(workspace: &Path, scope: &MemoryScope) -> PathBuf {
    scope_memory_dir(workspace, scope).join(LOCK_FILENAME)
}

/// Returns `true` when no successful digest has run today (UTC) yet.
pub fn should_run_today(workspace: &Path, scope: &MemoryScope) -> bool {
    match last_digest_date(workspace, scope) {
        Some(last) => last < Utc::now().date_naive(),
        None => true,
    }
}

fn last_digest_date(workspace: &Path, scope: &MemoryScope) -> Option<NaiveDate> {
    let path = lock_path(workspace, scope);
    let raw = std::fs::read_to_string(&path).ok()?;
    raw.lines()
        .next()
        .and_then(|s| NaiveDate::parse_from_str(s.trim(), "%Y-%m-%d").ok())
}

/// Days since the persisted `.last_digest` marker for one scope.
/// Missing/invalid markers return [`MISSING_DIGEST_DAYS_SENTINEL`] so a
/// restart or corrupt marker cannot make a stale digest look fresh.
pub fn days_since_last_digest_at(workspace: &Path, scope: &MemoryScope, today: NaiveDate) -> u64 {
    match last_digest_date(workspace, scope) {
        Some(last) => (today - last).num_days().max(0) as u64,
        None => MISSING_DIGEST_DAYS_SENTINEL,
    }
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct DigestStalenessSnapshot {
    pub global: u64,
    pub project: u64,
    /// Maximum staleness across discovered user scopes. `0` means no user
    /// memory directories exist, not that a user digest necessarily ran today.
    pub user: u64,
}

pub fn digest_staleness_snapshot_at(workspace: &Path, today: NaiveDate) -> DigestStalenessSnapshot {
    let user = super::store::list_user_scopes()
        .iter()
        .map(|scope| days_since_last_digest_at(workspace, scope, today))
        .max()
        .unwrap_or(0);
    DigestStalenessSnapshot {
        global: days_since_last_digest_at(workspace, &MemoryScope::Global, today),
        project: days_since_last_digest_at(workspace, &MemoryScope::Project, today),
        user,
    }
}

pub fn digest_staleness_snapshot(workspace: &Path) -> DigestStalenessSnapshot {
    digest_staleness_snapshot_at(workspace, Utc::now().date_naive())
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
    record_memory_digest_metrics(&plan, cfg);
    tracing::info!("{}", counterfactual_log_line(scope, &plan));

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

fn record_memory_digest_metrics(plan: &DigestPlan, cfg: &MemoryConfig) {
    use std::sync::atomic::Ordering;

    crate::types::MEMORY_CANDIDATES_COUNT
        .fetch_add(u64::from(plan.counterfactual.candidates), Ordering::Relaxed);
    let roads = digest::promoted_road_counts(&plan.promoted, cfg);
    crate::types::MEMORY_PROMOTED_REPEAT_DAYS_COUNT
        .fetch_add(u64::from(roads.repeat_days), Ordering::Relaxed);
    crate::types::MEMORY_PROMOTED_REINFORCE_COUNT
        .fetch_add(u64::from(roads.reinforce), Ordering::Relaxed);
    crate::types::MEMORY_PROMOTED_REOBS_COUNT.fetch_add(u64::from(roads.reobs), Ordering::Relaxed);
}

pub(crate) fn counterfactual_log_line(scope: &MemoryScope, plan: &DigestPlan) -> String {
    let stats = plan.counterfactual;
    format!(
        "memory digest counterfactual scope={} road_counts=independent_may_overlap candidates={} would_promote_repeat_days={} would_promote_reinforce={} would_promote_reobs={} promoted={} rejected={}",
        scope,
        stats.candidates,
        stats.would_promote.repeat_days,
        stats.would_promote.reinforce,
        stats.would_promote.reobs,
        plan.promoted.len(),
        plan.rejected.len(),
    )
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
    match try_recent_shift_block(workspace, scope, cfg) {
        Ok(block) => block,
        Err(e) => {
            crate::memory::record_injection_failed("recent_shift", e);
            None
        }
    }
}

pub(crate) fn try_recent_shift_block(
    workspace: &Path,
    scope: &MemoryScope,
    cfg: &MemoryConfig,
) -> Result<Option<String>, RecallCounterError> {
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
            // Bump recall counter before mutating `out`: if the counter is
            // poisoned, the whole shift injection degrades to "inject nothing"
            // rather than partially appending context and then failing.
            try_record_recall(scope, &e.content)?;
            out.push_str(&format!("- {}\n", e.content));
            wrote_any = true;
        }
    }
    if !wrote_any {
        return Ok(None);
    }
    if out.len() > cfg.recent_shift_max_chars {
        out.truncate(out.floor_char_boundary(cfg.recent_shift_max_chars));
        out.push_str("\n…[truncated]");
    }
    Ok(Some(out))
}

#[cfg(test)]
mod tests {
    use super::*;

    use std::sync::atomic::Ordering;

    use crate::memory::store::MarkdownMemoryStore;
    use crate::memory::types::{MemoryEntry, MemoryType};

    fn proj() -> MemoryScope {
        MemoryScope::Project
    }

    fn reset_recall_map_for_test() {
        let map = recall_map();
        let mut guard = map.lock().unwrap_or_else(|e| e.into_inner());
        guard.clear();
        map.clear_poison();
    }

    fn poison_recall_map_for_test() {
        let _ = std::thread::spawn(|| {
            let _guard = recall_map().lock().unwrap_or_else(|e| e.into_inner());
            panic!("poison recall map for S8 regression test");
        })
        .join();
    }

    #[test]
    fn recall_counter_round_trips() {
        let _lock = crate::memory::MEMORY_INJECTION_TEST_LOCK
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        reset_recall_map_for_test();
        let scope = MemoryScope::User("test_recall_user".into());
        record_recall(&scope, "rule X");
        record_recall(&scope, "rule X");
        assert_eq!(take_recall(&scope, "rule X"), 2);
        // Subsequent take returns 0 (we cleared on read).
        assert_eq!(take_recall(&scope, "rule X"), 0);
        reset_recall_map_for_test();
    }

    #[test]
    fn s8_poisoned_recall_mutex_degrades_recent_shift_to_empty_and_counted() {
        let _lock = crate::memory::MEMORY_INJECTION_TEST_LOCK
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        reset_recall_map_for_test();
        let before = crate::types::MEMORY_INJECTION_FAILED_COUNT.load(Ordering::Relaxed);
        let dir = tempfile::tempdir().unwrap();
        let _root = crate::memory::store::MemoryPaths::set_test_root(dir.path());
        let workspace = dir.path().join("workspace");
        let scope = proj();
        let entry = MemoryEntry::new(
            MemoryType::Preference,
            "fresh daily rule".to_string(),
            "test",
            scope.clone(),
        );
        let path = MarkdownMemoryStore::daily_path(&workspace, &scope, Utc::now().date_naive());
        MarkdownMemoryStore::append_dedup(&path, &entry, false).unwrap();

        poison_recall_map_for_test();
        let block = recent_shift_block(&workspace, &scope, &MemoryConfig::default());
        let after = crate::types::MEMORY_INJECTION_FAILED_COUNT.load(Ordering::Relaxed);

        assert!(block.is_none(), "poisoned recall map must inject nothing");
        assert_eq!(after, before + 1, "poison failure must be counted once");
        reset_recall_map_for_test();
    }

    #[test]
    fn s8_recent_shift_happy_path_returns_block_without_counter_bump() {
        let _lock = crate::memory::MEMORY_INJECTION_TEST_LOCK
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        reset_recall_map_for_test();
        let before = crate::types::MEMORY_INJECTION_FAILED_COUNT.load(Ordering::Relaxed);
        let dir = tempfile::tempdir().unwrap();
        let _root = crate::memory::store::MemoryPaths::set_test_root(dir.path());
        let workspace = dir.path().join("workspace");
        let scope = proj();
        let entry = MemoryEntry::new(
            MemoryType::Preference,
            "healthy daily rule".to_string(),
            "test",
            scope.clone(),
        );
        let path = MarkdownMemoryStore::daily_path(&workspace, &scope, Utc::now().date_naive());
        MarkdownMemoryStore::append_dedup(&path, &entry, false).unwrap();

        let block = recent_shift_block(&workspace, &scope, &MemoryConfig::default())
            .expect("healthy recall map should render recent shift");
        let after = crate::types::MEMORY_INJECTION_FAILED_COUNT.load(Ordering::Relaxed);

        assert!(block.contains("healthy daily rule"));
        assert_eq!(
            after, before,
            "healthy recent shift must not bump failure counter"
        );
        reset_recall_map_for_test();
    }

    #[test]
    fn should_run_today_when_lock_missing() {
        let dir = tempfile::tempdir().unwrap();
        let _root = crate::memory::store::MemoryPaths::set_test_root(dir.path());
        let workspace = dir.path();
        let scope = proj();
        let mem_dir = scope_memory_dir(workspace, &scope);
        assert!(
            mem_dir.starts_with(dir.path()),
            "memory dir must stay inside temp root: {}",
            mem_dir.display()
        );
        assert!(should_run_today(workspace, &scope));
    }

    #[test]
    fn mark_then_should_not_run() {
        let dir = tempfile::tempdir().unwrap();
        let _root = crate::memory::store::MemoryPaths::set_test_root(dir.path());
        let workspace = dir.path();
        let scope = proj();
        let mem_dir = scope_memory_dir(workspace, &scope);
        assert!(
            mem_dir.starts_with(dir.path()),
            "memory dir must stay inside temp root: {}",
            mem_dir.display()
        );
        std::fs::create_dir_all(&mem_dir).unwrap();
        mark_ran_today(workspace, &scope).unwrap();
        assert!(!should_run_today(workspace, &scope));
    }

    #[test]
    fn days_since_last_digest_reads_persisted_marker_after_restart() {
        let dir = tempfile::tempdir().unwrap();
        let workspace = dir.path().join("workspace");
        let scope = proj();
        let today = NaiveDate::from_ymd_opt(2026, 4, 24).unwrap();
        let stale = today - chrono::Duration::days(3);

        {
            let _root = crate::memory::store::MemoryPaths::set_test_root(dir.path());
            let path = lock_path(&workspace, &scope);
            let parent = path.parent().expect("lock path has parent");
            std::fs::create_dir_all(parent).unwrap();
            std::fs::write(&path, format!("{stale}\n")).unwrap();
            assert_eq!(days_since_last_digest_at(&workspace, &scope, today), 3);
        }

        let _fresh_root = crate::memory::store::MemoryPaths::set_test_root(dir.path());
        let fresh_state = digest_staleness_snapshot_at(&workspace, today);
        assert_eq!(fresh_state.project, 3);
        assert_eq!(fresh_state.global, MISSING_DIGEST_DAYS_SENTINEL);
    }
}
