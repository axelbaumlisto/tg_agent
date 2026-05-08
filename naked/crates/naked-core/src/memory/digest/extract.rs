//! Draft collection and conversation extraction for the memory digest.
//!
//! This module handles two related concerns:
//!
//! * **Collection** — reading and deduplicating draft entries from the
//!   daily-draft files in the lookback window (`collect_window`,
//!   `flush_recall_to_disk`).
//! * **Extraction** — LLM-driven extraction of rule candidates from a
//!   raw conversation transcript, written back to today's draft file
//!   (`extract_and_append`, `session_close`, `pre_compaction_flush`).

use std::collections::HashMap;
use std::path::Path;

use chrono::{NaiveDate, Utc};

use crate::memory::store::{MarkdownMemoryStore, content_fingerprint};
use crate::memory::types::{MemoryEntry, MemoryScope, MemoryType};
use crate::provider::Provider;

use super::ScoringHints;

const FLUSH_PROMPT: &str = "From the conversation excerpt below, extract up to 5 SHORT, ACTIONABLE rules-of-thumb \
     that would be useful to remember next time. Each rule must be a single sentence ≤140 chars. \
     Skip narrative summary, skip personal asides. \
     Output one rule per line, plain text, no numbering, no bullets, no markdown. \
     If nothing rule-worthy stands out, output exactly: NONE";

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
    use crate::memory::daily;

    let path = super::scope_memory_path(workspace, scope);
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
