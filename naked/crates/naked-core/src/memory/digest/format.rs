//! Summary rendering and plan application for the memory digest.
//!
//! This module produces human-readable output from a `DigestPlan` and
//! writes the final artefacts to disk:
//!
//! * [`deterministic_summary`] — fallback one-liner (no LLM needed).
//! * [`summarize_with_llm`]   — LLM-generated paragraph summary.
//! * [`apply_plan`]           — write promotions + dream entry to disk.
//! * [`last_digest_summary`]  — read back the most-recent DREAMS.md entry.

use std::path::Path;

use chrono::{DateTime, Utc};

use crate::config::MemoryConfig;
use crate::error::Result;
use crate::memory::dreams::{self, RejectedItem};
use crate::memory::store::MarkdownMemoryStore;
use crate::memory::types::MemoryScope;
use crate::provider::Provider;

use super::{ApplyOutcome, DigestMode, DigestPlan, PromoteEntry, RejectedEntry};

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
        s.truncate(s.floor_char_boundary(max_chars));
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
    let memory_path = super::scope_memory_path(workspace, &plan.scope);
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
