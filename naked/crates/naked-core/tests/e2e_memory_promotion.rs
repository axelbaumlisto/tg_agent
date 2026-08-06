//! Hermetic E2E test for the v0.5 memory re-observation promotion loop.
//!
//! This intentionally does not require `NAKED_HOME`: the regression being
//! guarded is that memory e2e coverage must fail loudly rather than silently
//! returning when a precondition is absent.

use std::path::{Path, PathBuf};
use std::sync::atomic::Ordering;

use chrono::Duration;
use naked_core::config::MemoryConfig;
use naked_core::memory::daily::run_daily;
use naked_core::memory::store::{MarkdownMemoryStore, MemoryPaths};
use naked_core::memory::types::{MemoryEntry, MemoryScope, MemoryType};
use naked_core::types::{MEMORY_CANDIDATES_COUNT, MEMORY_PROMOTED_REOBS_COUNT};

const PROMOTED_RULE: &str =
    "When validating memory promotion, require two independent re-observations.";

#[tokio::test(flavor = "current_thread")]
async fn repeated_rule_reobservation_promotes_only_when_flag_enabled() {
    let temp = tempfile::tempdir().expect("tempdir must be creatable");
    let memory_root = temp.path().join("naked-home");
    let _root = MemoryPaths::set_test_root(&memory_root);

    assert_memory_root_is_isolated(temp.path(), &memory_root);

    let candidates_before = MEMORY_CANDIDATES_COUNT.load(Ordering::Relaxed);
    let promoted_reobs_before = MEMORY_PROMOTED_REOBS_COUNT.load(Ordering::Relaxed);

    let flag_off = run_case(&memory_root, "flag-off", false).await;
    assert_eq!(
        flag_off.plan.counterfactual.candidates, 1,
        "the repeated daily drafts should deduplicate into one scored candidate"
    );
    assert_eq!(
        flag_off.plan.counterfactual.would_promote.reobs, 1,
        "S4 counterfactual diagnostics must expose the reobs road even while the feature flag is off"
    );
    assert!(
        flag_off.plan.promoted.is_empty(),
        "default-off contract: reobserved rule must not promote while memory_reobservation_promote_enabled=false"
    );
    assert_eq!(
        flag_off.outcome.promoted_written, 0,
        "flag-off digest must not write the rule to MEMORY.md"
    );
    assert_memory_absent_or_missing_rule(&flag_off.memory_path, PROMOTED_RULE);
    assert_eq!(
        MEMORY_PROMOTED_REOBS_COUNT.load(Ordering::Relaxed),
        promoted_reobs_before,
        "flag-off counterfactual reobs must not increment the actual promoted reobs metric"
    );

    let flag_on = run_case(&memory_root, "flag-on", true).await;
    assert_eq!(flag_on.plan.counterfactual.candidates, 1);
    assert_eq!(
        flag_on.plan.counterfactual.would_promote.reobs, 1,
        "S4 counterfactual diagnostics should still report the reobs road when it actually promotes"
    );
    assert_eq!(
        flag_on.plan.counterfactual.would_promote.repeat_days, 0,
        "fixture must not accidentally qualify through the repeat-days road"
    );
    assert_eq!(
        flag_on.plan.counterfactual.would_promote.reinforce, 0,
        "fixture must not accidentally qualify through the raw-reinforcement road"
    );
    assert_eq!(
        flag_on.plan.promoted.len(),
        1,
        "two persisted re-observations should promote via the S3 road when enabled"
    );
    assert_eq!(
        flag_on.plan.promoted[0].hints.repeat_days, 2,
        "the rule must be observed on two distinct daily draft files"
    );
    assert_eq!(
        flag_on.plan.promoted[0].hints.recall_count, 2,
        "collect_window must feed persisted recall_count into ScoringHints"
    );
    assert_eq!(
        flag_on.outcome.promoted_written, 1,
        "run_daily(.., None) should apply a promotion without an LLM"
    );
    assert_memory_contains_rule(&flag_on.memory_path, PROMOTED_RULE);

    assert!(
        MEMORY_CANDIDATES_COUNT.load(Ordering::Relaxed) >= candidates_before + 2,
        "each digest run should increment the S4 candidate counter"
    );
    assert!(
        MEMORY_PROMOTED_REOBS_COUNT.load(Ordering::Relaxed) > promoted_reobs_before,
        "the enabled digest should increment the actual promoted reobs metric"
    );
}

struct CaseResult {
    plan: naked_core::memory::digest::DigestPlan,
    outcome: naked_core::memory::digest::ApplyOutcome,
    memory_path: PathBuf,
}

async fn run_case(memory_root: &Path, workspace_slug: &str, reobs_enabled: bool) -> CaseResult {
    let workspace = memory_root.join("workspaces").join(workspace_slug);
    std::fs::create_dir_all(&workspace).expect("workspace directory must be creatable");
    let scope = MemoryScope::Project;

    write_two_day_reobserved_rule(&workspace, &scope, PROMOTED_RULE);

    let cfg = MemoryConfig {
        memory_reobservation_promote_enabled: reobs_enabled,
        promote_min_reobservations: 2,
        // The fixture has exactly 2 repeat-days and 2 raw appended occurrences;
        // requiring 3 days ensures only the re-observation road can promote.
        promote_min_repeat_days: 3,
        promote_min_recall_count: 0,
        compact_threshold_per_section: 0,
        ..MemoryConfig::default()
    };

    let (plan, outcome) = run_daily(&workspace, &scope, &cfg, None)
        .await
        .expect("run_daily should not fail")
        .expect("fresh hermetic workspace should not be skipped by the daily lock");

    CaseResult {
        plan,
        outcome,
        memory_path: MarkdownMemoryStore::project_memory_path(&workspace),
    }
}

fn write_two_day_reobserved_rule(workspace: &Path, scope: &MemoryScope, content: &str) {
    let today_entry = entry(content, scope);
    assert!(
        MarkdownMemoryStore::append_daily(workspace, &today_entry, true)
            .expect("append_daily should create today's draft entry"),
        "first append_daily call should create the draft entry"
    );
    assert!(
        !MarkdownMemoryStore::append_daily(workspace, &entry(content, scope), true)
            .expect("append_daily dedup hit should persist re-observation"),
        "second append_daily call should be a dedup hit"
    );

    let dates = MarkdownMemoryStore::list_daily(workspace, scope);
    let today = *dates
        .first()
        .expect("append_daily must create at least one dated draft file");
    let previous_day = today - Duration::days(1);
    let previous_path = MarkdownMemoryStore::daily_path(workspace, scope, previous_day);

    assert!(
        MarkdownMemoryStore::append_dedup(&previous_path, &entry(content, scope), true)
            .expect("append_dedup should create previous day's draft entry"),
        "first previous-day append_dedup call should create the draft entry"
    );
    assert!(
        !MarkdownMemoryStore::append_dedup(&previous_path, &entry(content, scope), true)
            .expect("append_dedup dedup hit should persist previous-day re-observation"),
        "second previous-day append_dedup call should be a dedup hit"
    );

    assert_daily_file_has_one_reobservation(workspace, scope, today);
    assert_daily_file_has_one_reobservation(workspace, scope, previous_day);
}

fn entry(content: &str, scope: &MemoryScope) -> MemoryEntry {
    MemoryEntry::new(
        MemoryType::ProjectKnowledge,
        content.to_string(),
        "e2e_memory_promotion",
        scope.clone(),
    )
}

fn assert_daily_file_has_one_reobservation(
    workspace: &Path,
    scope: &MemoryScope,
    date: chrono::NaiveDate,
) {
    let entries = MarkdownMemoryStore::read_daily(workspace, scope, date);
    assert_eq!(
        entries.len(),
        1,
        "daily draft should contain one deduped rule"
    );
    assert_eq!(
        entries[0].recall_count, 1,
        "dedup hit should persist exactly one re-observation in the daily draft"
    );
    assert!(
        entries[0].last_recalled_at.is_some(),
        "dedup hit should persist last_recalled_at"
    );
}

fn assert_memory_root_is_isolated(temp_root: &Path, memory_root: &Path) {
    let resolved = MarkdownMemoryStore::naked_home();
    assert_eq!(
        resolved, memory_root,
        "test must use the injected hermetic memory root"
    );
    assert!(
        resolved.starts_with(temp_root),
        "resolved memory root must stay inside tempdir: {} not under {}",
        resolved.display(),
        temp_root.display()
    );
    assert!(
        !resolved.ends_with(".naked"),
        "test must never resolve to the operator's real ~/.naked"
    );
}

fn assert_memory_contains_rule(path: &Path, rule: &str) {
    let content = std::fs::read_to_string(path).expect("MEMORY.md should be written");
    assert!(
        content.contains(rule),
        "promoted rule missing from {}:\n{}",
        path.display(),
        content
    );
}

fn assert_memory_absent_or_missing_rule(path: &Path, rule: &str) {
    let content = std::fs::read_to_string(path).unwrap_or_default();
    assert!(
        !content.contains(rule),
        "flag-off run unexpectedly wrote promoted rule to {}:\n{}",
        path.display(),
        content
    );
}
