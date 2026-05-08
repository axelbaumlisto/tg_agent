//! Scoring, decay, and promotion-victim selection for the memory digest.
//!
//! This module is a pure decision layer: no disk I/O, no LLM calls.
//! All functions are deterministic and easy to unit-test in isolation.

use chrono::NaiveDate;

use crate::config::MemoryConfig;
use crate::memory::types::{MemoryEntry, MemoryType};

use super::{PromoteEntry, RejectedEntry, ScoringHints};

/// Pure decision layer: split candidates into promotions and
/// rejections based on `MemoryConfig` gates. The LLM is NOT consulted
/// here so the rule is deterministic and easy to test.
pub fn partition_candidates(
    candidates: Vec<(MemoryEntry, ScoringHints)>,
    cfg: &MemoryConfig,
) -> (Vec<PromoteEntry>, Vec<RejectedEntry>) {
    let mut promoted = Vec::new();
    let mut rejected = Vec::new();
    for (entry, hints) in candidates {
        let mut reasons: Vec<String> = Vec::new();

        // Reinforcement boost: 3+ raw occurrences (user keeps repeating)
        // counts as if it appeared on one extra day. Strong reinforcement
        // (5+) skips the repeat_days gate entirely — traumatic/important
        // memories that the user emphasizes get promoted immediately.
        // Negative memories (corrections, failures) stick faster —
        // one less day required, like traumatic memory in humans.
        let is_negative = matches!(
            entry.memory_type,
            MemoryType::Correction | MemoryType::Failure
        );
        let day_boost = u32::from(is_negative); // +1 day for corrections/failures

        let effective_days = if hints.reinforcements >= 5 {
            cfg.promote_min_repeat_days // auto-qualify: user repeated 5+ times
        } else if hints.reinforcements >= 3 {
            hints.repeat_days + 1 + day_boost
        } else {
            hints.repeat_days + day_boost
        };

        if effective_days < cfg.promote_min_repeat_days {
            reasons.push(format!(
                "appeared on only {} day(s) (need ≥{}){}",
                hints.repeat_days,
                cfg.promote_min_repeat_days,
                if hints.reinforcements > 1 {
                    format!(" [{} reinforcements]", hints.reinforcements)
                } else {
                    String::new()
                }
            ));
        }
        if cfg.promote_min_recall_count > 0 && hints.recall_count < cfg.promote_min_recall_count {
            reasons.push(format!(
                "recalled only {} time(s) (need ≥{})",
                hints.recall_count, cfg.promote_min_recall_count
            ));
        }
        if reasons.is_empty() {
            promoted.push(PromoteEntry { entry, hints });
        } else {
            rejected.push(RejectedEntry {
                entry,
                hints,
                reason: reasons.join("; "),
            });
        }
    }
    (promoted, rejected)
}

/// Effective recall score for an entry under exponential half-life
/// decay. Pure function — no I/O.
///
/// `effective_recall = recall_count * 0.5^(age_days / half_life_days)`
///
/// where `age_days` is `today - last_recalled_at` (or `today -
/// created_at` when the entry has never been recalled). When
/// `half_life_days == 0` the decay is disabled and the raw
/// `recall_count` is returned (legacy behaviour).
///
/// Returned as `f32` so the score-vs-threshold comparison can be
/// fractional (`recall_count = 3, age = half_life ⇒ score = 1.5`).
pub fn effective_recall(entry: &MemoryEntry, today: NaiveDate, half_life_days: u32) -> f32 {
    let raw = entry.recall_count as f32;
    if half_life_days == 0 || raw == 0.0 {
        return raw;
    }
    let anchor = entry
        .last_recalled_at
        .map(|dt| dt.date_naive())
        .unwrap_or_else(|| entry.created_at.date_naive());
    let age_days = (today - anchor).num_days().max(0) as f32;
    let factor = 0.5f32.powf(age_days / half_life_days as f32);
    raw * factor
}

/// Pick eligible victims from `entries` (already in section order),
/// returning the indices to merge. Pure function — no I/O, easy to
/// test.
///
/// Filtering rules:
/// 1. Created on or before `today - compact_min_age_days` (don't
///    compact fresh rules).
/// 2. `effective_recall(entry) < compact_spare_recall` — protects
///    recently-recalled entries even when they are old. The
///    half-life is governed by `compact_recall_half_life_days`.
/// 3. `content.len() <= compact_max_chars` — never replace a long
///    entry with a shorter summary that loses operational detail.
///
/// Ordering: oldest **and least-recalled** first. We sort by
/// `(effective_recall ascending, created_at ascending)` so that two
/// equally-stale entries are broken by age (oldest wins).
pub fn pick_compaction_victims(
    entries: &[MemoryEntry],
    cfg: &MemoryConfig,
    today: NaiveDate,
) -> Vec<usize> {
    if cfg.compact_threshold_per_section == 0 || cfg.compact_window < 2 {
        return Vec::new();
    }
    if entries.len() < cfg.compact_threshold_per_section as usize {
        return Vec::new();
    }
    let min_age = chrono::Duration::days(cfg.compact_min_age_days as i64);
    let cutoff_date = today - min_age;
    let half_life = cfg.compact_recall_half_life_days;
    let spare = cfg.compact_spare_recall as f32;
    let mut eligible: Vec<usize> = entries
        .iter()
        .enumerate()
        .filter(|(_, e)| {
            e.created_at.date_naive() <= cutoff_date
                && effective_recall(e, today, half_life) < spare
                && e.content.len() <= cfg.compact_max_chars
        })
        .map(|(i, _)| i)
        .collect();
    if eligible.len() < cfg.compact_window as usize {
        return Vec::new();
    }
    // Sort by `(effective_recall ascending, created_at ascending)`
    // so least-useful entries bubble to the top. f32 doesn't
    // implement Ord, so we wrap with `total_cmp` for a stable order
    // even when scores collide.
    eligible.sort_by(|&a, &b| {
        let sa = effective_recall(&entries[a], today, half_life);
        let sb = effective_recall(&entries[b], today, half_life);
        sa.total_cmp(&sb)
            .then_with(|| entries[a].created_at.cmp(&entries[b].created_at))
    });
    eligible.truncate(cfg.compact_window as usize);
    eligible.sort_unstable();
    eligible
}
