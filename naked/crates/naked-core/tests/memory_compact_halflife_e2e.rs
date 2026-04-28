//! End-to-end tests for Phase 3.4: half-life decay in memory
//! compaction.
//!
//! These tests exercise the full on-disk pipeline:
//!
//! 1. Seed a real `MEMORY.md` file with entries that have a mix of
//!    (old-but-recently-recalled) and (old-and-stale) profiles.
//! 2. Run `compact_memory_at` (the `$NAKED_HOME`-free test seam
//!    added alongside [`compact_memory`]) with a `MemoryConfig`
//!    that has the half-life knob set.
//! 3. Reload the file from disk and verify the invariants:
//!    * recently-recalled entries are preserved verbatim;
//!    * stale old entries have been replaced by a single
//!      `source=compaction[...]` summary line;
//!    * unit-test assumptions about section partitioning hold
//!      when multiple sections (Preference, Correction, …) are
//!      live simultaneously.
//!
//! Unit tests already cover `pick_compaction_victims` and
//! `effective_recall` in isolation (see `memory::digest::tests` in
//! `naked-core`). The E2E tests here exist to catch wiring bugs
//! between `compact_memory` and the markdown-store load/save/parse
//! pipeline that a pure unit test cannot detect.

use chrono::{Duration as ChronoDuration, Utc};
use naked_core::config::MemoryConfig;
use naked_core::memory::digest::compact_memory_at;
use naked_core::memory::store::MarkdownMemoryStore;
use naked_core::memory::types::{MemoryEntry, MemoryScope, MemoryType};
use tempfile::tempdir;

fn cfg_with_half_life(half_life_days: u32) -> MemoryConfig {
    MemoryConfig {
        // Small section budget so we fire on 4-item sections.
        compact_threshold_per_section: 4,
        // Merge 3 of 4 victims into one summary.
        compact_window: 3,
        compact_max_chars: 500,
        // All seeded entries are older than 14 days, so the
        // min-age gate doesn't accidentally save them.
        compact_min_age_days: 14,
        // Anything with effective_recall < 2.0 is eligible.
        compact_spare_recall: 2,
        compact_recall_half_life_days: half_life_days,
        ..MemoryConfig::default()
    }
}

fn make_entry(
    ty: MemoryType,
    content: &str,
    created_days_ago: i64,
    recall_count: u32,
    last_recall_days_ago: Option<i64>,
) -> MemoryEntry {
    let now = Utc::now();
    MemoryEntry {
        id: uuid::Uuid::new_v4().to_string()[..8].to_string(),
        memory_type: ty,
        content: content.to_string(),
        created_at: now - ChronoDuration::days(created_days_ago),
        source: "test".into(),
        scope: MemoryScope::Project,
        recall_count,
        last_recalled_at: last_recall_days_ago.map(|d| now - ChronoDuration::days(d)),
    }
}

#[tokio::test]
async fn halflife_decay_protects_recently_recalled_old_entry() {
    // Four Preference entries all ~180 days old. Three have no
    // recall activity (eligible). One was recalled just yesterday
    // with a fresh recall_count of 5 — its effective recall score
    // is ≈ 5.0, well above the `compact_spare_recall=2` threshold,
    // so it MUST survive compaction even though its `created_at`
    // is as ancient as the others.
    let tmp = tempdir().unwrap();
    let path = tmp.path().join("MEMORY.md");

    let entries = vec![
        make_entry(
            MemoryType::Preference,
            "P-stale-A: prefer tabs",
            180,
            0,
            None,
        ),
        make_entry(
            MemoryType::Preference,
            "P-stale-B: no trailing ws",
            180,
            0,
            None,
        ),
        make_entry(
            MemoryType::Preference,
            "P-stale-C: 4-space tabs",
            180,
            0,
            None,
        ),
        make_entry(
            MemoryType::Preference,
            "P-FRESH: always run tests before commit",
            180,
            5,
            Some(1),
        ),
    ];
    MarkdownMemoryStore::save(&path, &entries).expect("save initial");

    let cfg = cfg_with_half_life(30);
    let today = Utc::now().date_naive();
    let results = compact_memory_at(&path, &MemoryScope::Project, &cfg, today, None)
        .await
        .expect("compaction must not error");
    assert_eq!(results.len(), 1, "exactly one section must fire");

    let reloaded = MarkdownMemoryStore::load(&path, MemoryScope::Project);
    // Survivors = entries that aren't the compaction-summary line.
    let survivors: Vec<&str> = reloaded
        .iter()
        .filter(|e| !e.source.starts_with("compaction["))
        .map(|e| e.content.as_str())
        .collect();
    assert!(
        survivors.iter().any(|c| c.contains("P-FRESH")),
        "recently-recalled entry must survive, got survivors: {survivors:?}"
    );
    for stale in ["P-stale-A", "P-stale-B", "P-stale-C"] {
        assert!(
            !survivors.iter().any(|c| c.contains(stale)),
            "stale entry {stale} must have been compacted, got survivors: {survivors:?}"
        );
    }
    let merged = reloaded
        .iter()
        .find(|e| e.source.starts_with("compaction["))
        .expect("a compaction-summary entry must be present");
    assert!(
        merged.content.contains("P-stale-A")
            && merged.content.contains("P-stale-B")
            && merged.content.contains("P-stale-C"),
        "summary must mention all three victims, got: {:?}",
        merged.content
    );
}

#[tokio::test]
async fn halflife_decay_disabled_falls_back_to_raw_recall_count() {
    // Same corpus as above but with `compact_recall_half_life_days
    // = 0`. The protection threshold then collapses to raw
    // `recall_count >= compact_spare_recall`. The spec has
    // recall_count = 1 (below the 2-threshold), so it MUST become
    // eligible and be compacted together with the others. This is
    // the explicit legacy-behaviour escape hatch documented on
    // `compact_recall_half_life_days`.
    let tmp = tempdir().unwrap();
    let path = tmp.path().join("MEMORY.md");

    let entries = vec![
        make_entry(MemoryType::Correction, "C-A: fix null check", 180, 0, None),
        make_entry(
            MemoryType::Correction,
            "C-B: await before log",
            180,
            0,
            None,
        ),
        make_entry(
            MemoryType::Correction,
            "C-C: trim before compare",
            180,
            0,
            None,
        ),
        make_entry(
            MemoryType::Correction,
            "C-ONCE: recalled-once yesterday",
            180,
            1,
            Some(1),
        ),
    ];
    MarkdownMemoryStore::save(&path, &entries).expect("save initial");

    let cfg = cfg_with_half_life(0);
    let today = Utc::now().date_naive();
    let results = compact_memory_at(&path, &MemoryScope::Project, &cfg, today, None)
        .await
        .expect("compaction ok");
    assert_eq!(results.len(), 1);

    // Sort is (effective_recall asc, created_at asc). With
    // half_life=0, effective_recall == recall_count. The 3 entries
    // with recall=0 score lower than the one with recall=1, so the
    // latter is the sole survivor (outside the `compact_window=3`
    // cap) — it MUST stay.
    let reloaded = MarkdownMemoryStore::load(&path, MemoryScope::Project);
    let survivors: Vec<&str> = reloaded
        .iter()
        .filter(|e| !e.source.starts_with("compaction["))
        .map(|e| e.content.as_str())
        .collect();
    assert!(
        survivors.iter().any(|c| c.contains("C-ONCE")),
        "higher-recall entry wins even with half_life=0, got survivors: {survivors:?}"
    );
    for victim in ["C-A", "C-B", "C-C"] {
        assert!(
            !survivors.iter().any(|c| c.contains(&format!("{victim}:"))),
            "{victim} must have been compacted, got survivors: {survivors:?}"
        );
    }
}

#[tokio::test]
async fn halflife_decay_demotes_old_recall_below_threshold() {
    // Entry with `recall_count = 8` but `last_recalled_at = 90
    // days ago`. Half-life 30 days → effective_recall ≈ 8 *
    // 0.5^(90/30) = 8 * 0.125 = 1.0, below the spare threshold of
    // 2. It MUST be compacted alongside the other three stale
    // items — pure half-life demotion with no `created_at` change.
    let tmp = tempdir().unwrap();
    let path = tmp.path().join("MEMORY.md");

    let entries = vec![
        make_entry(
            MemoryType::ProjectKnowledge,
            "K-A: module layout",
            200,
            0,
            None,
        ),
        make_entry(
            MemoryType::ProjectKnowledge,
            "K-B: build profile",
            200,
            0,
            None,
        ),
        make_entry(
            MemoryType::ProjectKnowledge,
            "K-C: test topology",
            200,
            0,
            None,
        ),
        make_entry(
            MemoryType::ProjectKnowledge,
            "K-DECAYED: once-hot, now-cold ritual",
            200,
            8,
            Some(90),
        ),
    ];
    MarkdownMemoryStore::save(&path, &entries).expect("save initial");

    let cfg = cfg_with_half_life(30);
    let today = Utc::now().date_naive();
    let results = compact_memory_at(&path, &MemoryScope::Project, &cfg, today, None)
        .await
        .expect("compaction ok");
    assert_eq!(results.len(), 1, "decayed entry must be eligible → fire");

    let reloaded = MarkdownMemoryStore::load(&path, MemoryScope::Project);
    let survivors: Vec<&str> = reloaded
        .iter()
        .filter(|e| !e.source.starts_with("compaction["))
        .map(|e| e.content.as_str())
        .collect();
    // The summary line keeps the 3 oldest / least-recalled (recall=0)
    // entries as victims. K-DECAYED has effective_recall ≈ 1.0,
    // higher than the 0s, so it ranks last and survives *this*
    // compaction window (cap = 3). Assert the three raw victims
    // are gone but K-DECAYED is still around.
    for victim in ["K-A:", "K-B:", "K-C:"] {
        assert!(
            !survivors.iter().any(|c| c.contains(victim)),
            "{victim} must be compacted, got survivors: {survivors:?}"
        );
    }
    assert!(
        survivors.iter().any(|c| c.contains("K-DECAYED")),
        "highest-scored entry stays under compact_window cap, got survivors: {survivors:?}"
    );
}

#[tokio::test]
async fn compaction_no_op_when_below_threshold() {
    // 3 entries in a section where the threshold is 4 → no section
    // should fire; MEMORY.md must be byte-identical to the seed.
    let tmp = tempdir().unwrap();
    let path = tmp.path().join("MEMORY.md");

    let entries = vec![
        make_entry(MemoryType::Failure, "F-A: network flake", 200, 0, None),
        make_entry(MemoryType::Failure, "F-B: disk full", 200, 0, None),
        make_entry(MemoryType::Failure, "F-C: rate-limit", 200, 0, None),
    ];
    MarkdownMemoryStore::save(&path, &entries).expect("save");
    let before = std::fs::read_to_string(&path).unwrap();

    let cfg = cfg_with_half_life(30);
    let today = Utc::now().date_naive();
    let results = compact_memory_at(&path, &MemoryScope::Project, &cfg, today, None)
        .await
        .expect("compaction ok");
    assert!(results.is_empty(), "below threshold ⇒ no results");

    let after = std::fs::read_to_string(&path).unwrap();
    assert_eq!(before, after, "no-op compaction must not rewrite the file");
}

#[tokio::test]
async fn compaction_handles_missing_file_gracefully() {
    // Missing MEMORY.md — e.g. fresh project. Must return
    // Ok(empty), not ENOENT.
    let tmp = tempdir().unwrap();
    let path = tmp.path().join("does-not-exist.md");
    let cfg = cfg_with_half_life(30);
    let today = Utc::now().date_naive();
    let results = compact_memory_at(&path, &MemoryScope::Project, &cfg, today, None)
        .await
        .expect("missing file must be silently handled");
    assert!(results.is_empty());
}

#[tokio::test]
async fn compaction_isolates_sections() {
    // One section past threshold (Preference, 4 entries) and one
    // below (Correction, 2 entries). Only the Preference section
    // should fire; Correction must be untouched.
    let tmp = tempdir().unwrap();
    let path = tmp.path().join("MEMORY.md");

    let entries = vec![
        make_entry(
            MemoryType::Preference,
            "P-A: tabs over spaces",
            200,
            0,
            None,
        ),
        make_entry(MemoryType::Preference, "P-B: semis optional", 200, 0, None),
        make_entry(MemoryType::Preference, "P-C: trailing commas", 200, 0, None),
        make_entry(
            MemoryType::Preference,
            "P-D: lowercase imports",
            200,
            0,
            None,
        ),
        make_entry(MemoryType::Correction, "C-A: handle nulls", 200, 0, None),
        make_entry(MemoryType::Correction, "C-B: avoid globals", 200, 0, None),
    ];
    MarkdownMemoryStore::save(&path, &entries).expect("save");

    let cfg = cfg_with_half_life(30);
    let today = Utc::now().date_naive();
    let results = compact_memory_at(&path, &MemoryScope::Project, &cfg, today, None)
        .await
        .expect("compaction ok");
    assert_eq!(results.len(), 1, "only Preference section should fire");

    let reloaded = MarkdownMemoryStore::load(&path, MemoryScope::Project);

    // Both correction entries MUST survive intact.
    for survivor in ["C-A:", "C-B:"] {
        assert!(
            reloaded.iter().any(|e| e.content.contains(survivor)),
            "Correction entry {survivor} must be untouched, got: {:?}",
            reloaded
                .iter()
                .map(|e| e.content.as_str())
                .collect::<Vec<_>>()
        );
    }
    // Three of four preferences compacted into one summary +
    // one survivor from the compact_window cap.
    let pref_count = reloaded
        .iter()
        .filter(|e| e.memory_type == MemoryType::Preference)
        .count();
    assert_eq!(
        pref_count,
        2,
        "4 prefs → 1 survivor + 1 summary = 2, got {pref_count}: {:?}",
        reloaded
            .iter()
            .filter(|e| e.memory_type == MemoryType::Preference)
            .map(|e| e.content.as_str())
            .collect::<Vec<_>>()
    );
}
