//! Memory compaction pipeline: merges stale entries within a section
//! into a single summary rule via LLM (or a deterministic fallback).
//!
//! Entry points:
//! * [`compact_memory`]    — resolves the path from scope, then delegates.
//! * [`compact_memory_at`] — path-explicit seam used by e2e tests.

use std::path::Path;

use chrono::NaiveDate;

use crate::config::MemoryConfig;
use crate::memory::store::MarkdownMemoryStore;
use crate::memory::types::{MemoryEntry, MemoryScope, MemoryType};
use crate::provider::Provider;

/// One section of MEMORY.md that the digest decided to compact: the
/// list of victims plus the resulting one-line summary. Returned by
/// `compact_memory_at` for logging, DREAMS.md, and tests.
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
    let path = super::scope_memory_path(workspace, scope);
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

        let local_victims = super::pick_compaction_victims(&section_entries, cfg, today);
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
        .unwrap_or_else(|| super::deterministic_merge(&victim_entries, cfg.compact_max_chars));

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
