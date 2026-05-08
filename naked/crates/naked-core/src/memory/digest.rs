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

use std::collections::HashMap;
use std::path::Path;

use chrono::{DateTime, NaiveDate, Utc};

use super::dreams::{self, RejectedItem};
use super::store::MarkdownMemoryStore;
use super::types::{MemoryEntry, MemoryScope, MemoryType};
use crate::config::MemoryConfig;
use crate::error::Result;
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

/// Aggregate every draft entry in the window into one flat vector,
/// computing scoring hints in the same pass. Called by `run_daily`.
///
/// Window is `lookback_days` going back from `today` (inclusive).
/// Today's drafts ARE included so the digest picks up activity that
/// happened in the same UTC day as the run. Entries with the same
/// `content_hash` across multiple days are deduplicated — we keep the
/// oldest occurrence and aggregate scoring info on top.
pub fn collect_window(
    workspace: &Path,
    scope: &MemoryScope,
    today: NaiveDate,
    lookback_days: u32,
) -> Vec<(MemoryEntry, ScoringHints)> {
    use super::store::content_fingerprint;

    // Group by semantic fingerprint (more aggressive than content_hash).
    // This merges "greet the user briefly" and "Greet the user briefly in Turn 7."
    // into one candidate with repeat_days counting both.
    let mut by_fp: HashMap<u64, (MemoryEntry, Vec<NaiveDate>, Vec<String>, u32)> = HashMap::new();
    let max_back = lookback_days.max(1);

    for delta in 0..max_back {
        let date = today - chrono::Duration::days(delta as i64);
        for entry in MarkdownMemoryStore::read_daily(workspace, scope, date) {
            let fp = content_fingerprint(&entry.content);
            let bucket = by_fp
                .entry(fp)
                .or_insert_with(|| (entry.clone(), Vec::new(), Vec::new(), 0));
            // Keep the shortest/cleanest content as canonical.
            if entry.content.len() < bucket.0.content.len() {
                bucket.0.content.clone_from(&entry.content);
            }
            // Keep the oldest timestamp.
            if entry.created_at < bucket.0.created_at {
                bucket.0.created_at = entry.created_at;
            }
            if !bucket.1.contains(&date) {
                bucket.1.push(date);
            }
            if !bucket.2.contains(&entry.source) {
                bucket.2.push(entry.source.clone());
            }
            // Reinforcement: every occurrence strengthens the memory.
            bucket.3 += 1;
        }
    }

    let now = Utc::now();
    by_fp
        .into_values()
        .map(|(entry, dates, sources, reinforcements)| {
            let hints = ScoringHints {
                repeat_days: dates.len() as u32,
                source_diversity: sources.len() as u32,
                age_days: (now - entry.created_at).num_days().max(0) as u32,
                recall_count: 0,
                reinforcements,
            };
            (entry, hints)
        })
        .collect()
}

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
            super::types::MemoryType::Correction | super::types::MemoryType::Failure
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

/// Build a deterministic one-paragraph summary of the day's activity
/// — used as fallback when the LLM call fails or is disabled.
pub fn deterministic_summary(promoted: &[PromoteEntry], rejected: &[RejectedEntry]) -> String {
    format!(
        "Digest: {} promoted, {} rejected, {} total candidates seen this window.",
        promoted.len(),
        rejected.len(),
        promoted.len() + rejected.len()
    )
}

/// Build the LLM prompt body for `summarize_with_llm`. Plain text, no
/// JSON — we only ask the model for a paragraph, never for structure.
fn build_summary_prompt(
    promoted: &[PromoteEntry],
    rejected: &[RejectedEntry],
    max_chars: usize,
) -> String {
    let mut s = String::new();
    s.push_str("Promoted entries:\n");
    for p in promoted {
        s.push_str(&format!(
            "- [{}] {} (repeat_days={}, sources={})\n",
            p.entry.memory_type, p.entry.content, p.hints.repeat_days, p.hints.source_diversity
        ));
    }
    s.push_str("\nRejected entries:\n");
    for r in rejected {
        s.push_str(&format!(
            "- [{}] {} — reason: {}\n",
            r.entry.memory_type, r.entry.content, r.reason
        ));
    }
    if s.len() > max_chars {
        s.truncate(max_chars);
    }
    s
}

/// Call the LLM to produce a one-paragraph human summary. Returns
/// `Err` on transport / empty-output failures so callers can fall
/// back to `deterministic_summary` and never block the daily run.
pub async fn summarize_with_llm(
    provider: &dyn Provider,
    model: &str,
    promoted: &[PromoteEntry],
    rejected: &[RejectedEntry],
    cfg: &MemoryConfig,
) -> Result<String> {
    use crate::provider::ChatRequest;
    use crate::types::StreamChunk;
    use tokio_stream::StreamExt;

    let body = build_summary_prompt(promoted, rejected, cfg.digest_max_chars);
    let system = "You are summarizing a day of memory drafts. Produce ONE compact paragraph \
        (≤400 chars) that captures the theme, key promoted rules, and notable rejections. \
        Plain prose, no markdown lists, no headings.";
    let req = ChatRequest {
        model: model.to_string(),
        system: system.to_string(),
        messages: vec![serde_json::json!({ "role": "user", "content": body })],
        tools: vec![],
        max_tokens: 512,
        temperature: Some(0.0),
        reasoning: None,
    };

    let mut stream = provider.stream_chat(req).await.map_err(|e| {
        crate::error::AgentError::ProviderTyped(crate::provider::error::ProviderError::Other {
            status: 0,
            body: format!("digest summary stream: {e}"),
        })
    })?;
    let mut out = String::new();
    while let Some(chunk) = stream.next().await {
        match chunk {
            StreamChunk::Text(t) => out.push_str(&t),
            StreamChunk::Done => break,
            StreamChunk::Error(e) => {
                return Err(crate::error::AgentError::ProviderTyped(
                    crate::provider::error::ProviderError::Other {
                        status: 0,
                        body: format!("digest summary stream: {e}"),
                    },
                ));
            }
            _ => {}
        }
    }
    if out.trim().is_empty() {
        return Err(crate::error::AgentError::ProviderTyped(
            crate::provider::error::ProviderError::Other {
                status: 0,
                body: "digest summary returned empty".into(),
            },
        ));
    }
    Ok(out.trim().to_string())
}

/// Apply a digest plan: write the dream entry, optionally append
/// promotions to `MEMORY.md`, and (always) backup `MEMORY.md` first.
pub fn apply_plan(
    workspace: &Path,
    plan: &DigestPlan,
    cfg: &MemoryConfig,
) -> std::io::Result<ApplyOutcome> {
    let mut outcome = ApplyOutcome::default();

    // 1. Backup MEMORY.md (best-effort) before any writes.
    let memory_path = scope_memory_path(workspace, &plan.scope);
    if memory_path.exists() {
        let backup = memory_path.with_extension("md.bak");
        let _ = std::fs::copy(&memory_path, backup);
    }

    // 2. Promote when mode allows.
    if plan.mode == DigestMode::SummarizeAndPromote {
        for p in &plan.promoted {
            // Force `dedup=true` so re-runs are idempotent.
            let written = MarkdownMemoryStore::append_dedup(&memory_path, &p.entry, true)?;
            if written {
                outcome.promoted_written += 1;
            }
        }
    }

    // 3. Append the dream entry.
    let rejected_items: Vec<RejectedItem> = plan
        .rejected
        .iter()
        .map(|r| RejectedItem {
            content: r.entry.content.clone(),
            reason: Some(r.reason.clone()),
        })
        .collect();
    let promoted_text: Vec<String> = plan
        .promoted
        .iter()
        .map(|p| p.entry.content.clone())
        .collect();
    let dream = dreams::build_entry(
        plan.scope.clone(),
        &plan.summary,
        promoted_text,
        rejected_items,
    );
    if dreams::append_dream(workspace, &dream, cfg.dreams_retention_days).is_ok() {
        outcome.dreams_appended = true;
    }

    Ok(outcome)
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

/// Drain the in-memory `RECALL_COUNTERS` for `scope` into the
/// persistent `MEMORY.md` so the counters survive restarts.
///
/// For each entry whose content matches a draining bucket we bump
/// `recall_count` and refresh `last_recalled_at`. Unmatched buckets
/// are silently dropped (the entry was promoted-then-deleted, or
/// it's a draft-only string that never landed in MEMORY.md).
///
/// Returns the number of entries that picked up at least one bump.
pub fn flush_recall_to_disk(workspace: &Path, scope: &MemoryScope) -> std::io::Result<usize> {
    use super::daily;

    let path = scope_memory_path(workspace, scope);
    if !path.exists() {
        return Ok(0);
    }
    let mut entries = MarkdownMemoryStore::load(&path, scope.clone());
    if entries.is_empty() {
        return Ok(0);
    }

    let today = Utc::now();
    let mut bumped = 0usize;
    for e in &mut entries {
        let n = daily::take_recall(scope, &e.content);
        if n > 0 {
            e.recall_count = e.recall_count.saturating_add(n);
            e.last_recalled_at = Some(today);
            bumped += 1;
        }
    }
    if bumped > 0 {
        MarkdownMemoryStore::save(&path, &entries)?;
    }
    Ok(bumped)
}

/// One section of MEMORY.md that the digest decided to compact: the
/// list of victims plus the resulting one-line summary. Returned by
/// `compact_section` for logging, DREAMS.md, and tests.
#[derive(Debug, Clone, Default)]
pub struct CompactionResult {
    pub victims: Vec<MemoryEntry>,
    pub summary: Option<MemoryEntry>,
}

const COMPACT_PROMPT: &str = "You will receive a JSON array of memory rules from the same project section. \
     Merge them into ONE single-sentence rule that preserves the operational meaning \
     of all of them. Strict rules: \n\
     - Output exactly one sentence, plain text, no bullets, no numbering, no markdown.\n\
     - Do not invent details that aren't in the input.\n\
     - If the input is contradictory, prefer the most recent rule.\n\
     - Maximum length is enforced by the caller — be concise.";

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

/// Run compaction over MEMORY.md for `scope` once. For each
/// eligible section, ask the LLM to merge `cfg.compact_window` old
/// entries into one summary line and replace them in place.
///
/// Returns one `CompactionResult` per fired section. Best-effort:
/// LLM failures fall back to no-op (the section is left untouched).
pub async fn compact_memory(
    workspace: &Path,
    scope: &MemoryScope,
    cfg: &MemoryConfig,
    today: NaiveDate,
    llm: Option<(&dyn Provider, &str)>,
) -> std::io::Result<Vec<CompactionResult>> {
    let path = scope_memory_path(workspace, scope);
    compact_memory_at(&path, scope, cfg, today, llm).await
}

/// Same as [`compact_memory`] but takes an explicit `MEMORY.md`
/// path instead of resolving it through `$NAKED_HOME`. E2E tests
/// use this to exercise the full on-disk compaction pipeline
/// without having to mutate the process-global `NAKED_HOME` env
/// var (forbidden by the workspace `unsafe_code = "forbid"`
/// lint). Production callers should keep using [`compact_memory`].
pub async fn compact_memory_at(
    path: &Path,
    scope: &MemoryScope,
    cfg: &MemoryConfig,
    today: NaiveDate,
    llm: Option<(&dyn Provider, &str)>,
) -> std::io::Result<Vec<CompactionResult>> {
    if cfg.compact_threshold_per_section == 0 {
        return Ok(Vec::new());
    }
    if !path.exists() {
        return Ok(Vec::new());
    }
    let mut all = MarkdownMemoryStore::load(path, scope.clone());
    if all.is_empty() {
        return Ok(Vec::new());
    }

    let mut results: Vec<CompactionResult> = Vec::new();
    for &ty in MemoryType::ALL {
        // Indices in `all` that belong to this section.
        let section: Vec<usize> = all
            .iter()
            .enumerate()
            .filter(|(_, e)| e.memory_type == ty)
            .map(|(i, _)| i)
            .collect();
        if section.is_empty() {
            continue;
        }
        let section_entries: Vec<MemoryEntry> = section.iter().map(|&i| all[i].clone()).collect();

        let local_victims = pick_compaction_victims(&section_entries, cfg, today);
        if local_victims.is_empty() {
            continue;
        }
        // Translate local section indices back to indices into `all`.
        let global_victim_idx: Vec<usize> = local_victims.iter().map(|&i| section[i]).collect();
        let victim_entries: Vec<MemoryEntry> = local_victims
            .iter()
            .map(|&i| section_entries[i].clone())
            .collect();

        // Ask the LLM (or fall back to a deterministic merge).
        let summary_text = if let Some((provider, model)) = llm {
            llm_compact_merge(provider, model, &victim_entries, cfg.compact_max_chars).await
        } else {
            None
        }
        .unwrap_or_else(|| deterministic_merge(&victim_entries, cfg.compact_max_chars));

        if summary_text.trim().is_empty() {
            continue;
        }

        let merged_ids: Vec<String> = victim_entries.iter().map(|e| e.id.clone()).collect();
        let mut merged = MemoryEntry::new(ty, summary_text, "compaction", scope.clone());
        // Embed merged-from list directly in the source field — the
        // markdown comment doesn't have a dedicated slot for it and
        // we want it visible in `git diff` of MEMORY.md.
        merged.source = format!("compaction[{}]", merged_ids.join(","));

        // Remove victims from `all` (in reverse index order to keep
        // remaining indices stable), then append the summary.
        let mut sorted_idx = global_victim_idx.clone();
        sorted_idx.sort_unstable_by(|a, b| b.cmp(a));
        for i in sorted_idx {
            all.remove(i);
        }
        all.push(merged.clone());

        results.push(CompactionResult {
            victims: victim_entries,
            summary: Some(merged),
        });
    }

    if !results.is_empty() {
        MarkdownMemoryStore::save(path, &all)?;
    }
    Ok(results)
}

/// Last-resort merge when the LLM is unavailable or returns garbage:
/// concatenate the bullets, truncate to `max_chars` bytes. Not
/// pretty, but it preserves information.
fn deterministic_merge(victims: &[MemoryEntry], max_chars: usize) -> String {
    let joined = victims
        .iter()
        .map(|e| e.content.trim())
        .collect::<Vec<_>>()
        .join("; ");
    let summary = format!("[merged] {}", joined);
    truncate_with_ellipsis(summary, max_chars)
}

/// Trim `s` to at most `max_chars` bytes, appending `…` when a cut
/// happened. Respects UTF-8 char boundaries.
pub(crate) fn truncate_with_ellipsis(s: String, max_chars: usize) -> String {
    if s.len() <= max_chars {
        return s;
    }
    const ELLIPSIS: &str = "…"; // 3 bytes
    let budget = max_chars.saturating_sub(ELLIPSIS.len());
    let mut cut = budget;
    while cut > 0 && !s.is_char_boundary(cut) {
        cut -= 1;
    }
    let mut out = String::with_capacity(cut + ELLIPSIS.len());
    out.push_str(&s[..cut]);
    out.push_str(ELLIPSIS);
    out
}

/// LLM-driven merge. Returns `None` on stream/parse failure so the
/// caller falls back to `deterministic_merge`.
async fn llm_compact_merge(
    provider: &dyn Provider,
    model: &str,
    victims: &[MemoryEntry],
    max_chars: usize,
) -> Option<String> {
    use crate::types::StreamChunk;
    use tokio_stream::StreamExt;

    let bullets: Vec<String> = victims.iter().map(|e| e.content.clone()).collect();
    let payload = serde_json::to_string(&bullets).ok()?;
    let req = crate::provider::ChatRequest {
        model: model.to_string(),
        system: COMPACT_PROMPT.to_string(),
        messages: vec![serde_json::json!({
            "role": "user",
            "content": format!("Merge into one rule (≤{} chars):\n{}", max_chars, payload),
        })],
        tools: vec![],
        max_tokens: 256,
        temperature: Some(0.0),
        reasoning: None,
    };
    let mut stream = provider.stream_chat(req).await.ok()?;
    let mut out = String::new();
    while let Some(chunk) = stream.next().await {
        match chunk {
            StreamChunk::Text(t) => out.push_str(&t),
            StreamChunk::Done => break,
            StreamChunk::Error(_) => return None,
            _ => {}
        }
    }
    let trimmed = out.trim().trim_matches('"').trim().to_string();
    if trimmed.is_empty() {
        return None;
    }
    Some(truncate_with_ellipsis(trimmed, max_chars))
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

// ── Session-close + pre-compaction flush ──────────────────────────────────
//
// Both helpers are short LLM turns that translate raw conversation text
// into bullet-point candidate rules, then append them into today's
// daily-draft file as `MemoryType::ProjectKnowledge`. The daily digest
// later decides whether to promote any of them.
//
// They are best-effort and never propagate errors to the caller —
// session-close happens on `/new` and pre-compaction happens inside the
// hot path.

const FLUSH_PROMPT: &str = "From the conversation excerpt below, extract up to 5 SHORT, ACTIONABLE rules-of-thumb \
     that would be useful to remember next time. Each rule must be a single sentence ≤140 chars. \
     Skip narrative summary, skip personal asides. \
     Output one rule per line, plain text, no numbering, no bullets, no markdown. \
     If nothing rule-worthy stands out, output exactly: NONE";

/// LLM-driven flush: extract candidate rules and write them to today's
/// daily-draft file. Used by session-close and pre-compaction hooks.
///
/// Errors are swallowed and logged — the caller (interactive turn) must
/// not fail because the digest LLM hiccupped.
pub async fn extract_and_append(
    provider: &dyn Provider,
    model: &str,
    workspace: &Path,
    scope: &MemoryScope,
    conversation_text: &str,
    source_label: &str,
) {
    let snippet_max = 8_000;
    let snippet = if conversation_text.len() > snippet_max {
        &conversation_text[conversation_text.len() - snippet_max..]
    } else {
        conversation_text
    };

    let req = crate::provider::ChatRequest {
        model: model.to_string(),
        system: FLUSH_PROMPT.to_string(),
        messages: vec![serde_json::json!({ "role": "user", "content": snippet })],
        tools: vec![],
        max_tokens: 512,
        temperature: Some(0.0),
        reasoning: None,
    };
    let mut stream = match provider.stream_chat(req).await {
        Ok(s) => s,
        Err(e) => {
            tracing::warn!(
                source = source_label,
                "memory flush LLM failed to start: {e:#}"
            );
            return;
        }
    };
    let mut out = String::new();
    use crate::types::StreamChunk;
    use tokio_stream::StreamExt;
    while let Some(chunk) = stream.next().await {
        match chunk {
            StreamChunk::Text(t) => out.push_str(&t),
            StreamChunk::Done => break,
            StreamChunk::Error(e) => {
                tracing::warn!(source = source_label, "memory flush stream error: {e:#}");
                return;
            }
            _ => {}
        }
    }
    let trimmed = out.trim();
    if trimmed.is_empty() || trimmed.eq_ignore_ascii_case("NONE") {
        tracing::debug!(source = source_label, "memory flush produced no rules");
        return;
    }

    let mut written = 0;
    for line in trimmed.lines() {
        let s = line.trim().trim_start_matches('-').trim();
        if s.is_empty() || s.eq_ignore_ascii_case("none") || s.len() < 8 {
            continue;
        }
        let entry = MemoryEntry::new(
            MemoryType::ProjectKnowledge,
            s.to_string(),
            source_label,
            scope.clone(),
        );
        if let Ok(true) = MarkdownMemoryStore::append_daily(workspace, &entry, true) {
            written += 1;
        }
    }
    if written > 0 {
        tracing::info!(
            scope = %scope,
            source = source_label,
            written,
            "memory flush appended draft rules"
        );
    }
}

/// Convenience wrapper: session-close flush. Called from `Session::reset`
/// or `/new` handlers.
pub async fn session_close(
    provider: &dyn Provider,
    model: &str,
    workspace: &Path,
    scope: &MemoryScope,
    transcript: &str,
) {
    extract_and_append(
        provider,
        model,
        workspace,
        scope,
        transcript,
        "session_close",
    )
    .await;
}

/// Convenience wrapper: pre-compaction flush. Called from the
/// conversation-history compactor right before the LLM summary call.
pub async fn pre_compaction_flush(
    provider: &dyn Provider,
    model: &str,
    workspace: &Path,
    scope: &MemoryScope,
    conversation_text: &str,
) {
    extract_and_append(
        provider,
        model,
        workspace,
        scope,
        conversation_text,
        "pre_compaction",
    )
    .await;
}

/// Helper used by the prompt builder: returns the latest digest
/// summary written for `scope` (cross-checked by `since`). Currently
/// unused but kept around as a public API for future "remind me what
/// the digest decided yesterday" surfaces.
pub fn last_digest_summary(
    workspace: &Path,
    scope: &MemoryScope,
    since: DateTime<Utc>,
) -> Option<String> {
    let dreams = dreams::read_dreams(workspace, scope);
    let cutoff = since.date_naive();
    dreams
        .into_iter()
        .rfind(|d| d.date >= cutoff)
        .map(|d| d.summary)
}

#[cfg(test)]
#[path = "digest_tests.rs"]
mod tests;
