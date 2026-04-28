//! DREAMS.md — human-readable journal of every daily-digest run.
//!
//! Modeled after `openclaw`'s "dreams" surface: a separate Markdown
//! file the user can open to review what the digest decided to keep,
//! drop, or merge. The agent never reads this file — it is purely an
//! audit log so a human can sanity-check the LLM-driven promotion
//! pipeline without diffing `MEMORY.md`.
//!
//! Format:
//!
//! ```text
//! # Dreams
//!
//! ## 2026-04-20 (project)
//!
//! Summary: …short LLM-generated digest of the day's drafts…
//!
//! Promoted (3):
//! - rule one
//! - rule two
//! - rule three
//!
//! Rejected (5):
//! - junk one — reason: too vague
//! - junk two — reason: contradicts existing rule X
//! - …
//! ```
//!
//! Rotation: entries older than `MemoryConfig.dreams_retention_days`
//! are pruned at write time.

use std::path::{Path, PathBuf};

use chrono::{NaiveDate, Utc};
use serde::{Deserialize, Serialize};

use super::store::scope_memory_dir;
use super::types::MemoryScope;

/// One entry to append to `DREAMS.md`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DreamEntry {
    /// Date the digest ran (UTC).
    pub date: NaiveDate,
    /// Scope this digest covered (one DreamEntry per scope).
    pub scope: MemoryScope,
    /// LLM-generated summary of the day's draft activity.
    pub summary: String,
    /// Rules promoted into `MEMORY.md` this run (markdown list items,
    /// without the leading `- `).
    pub promoted: Vec<String>,
    /// Rules rejected by the digest (with optional reasons).
    pub rejected: Vec<RejectedItem>,
}

/// One rejected candidate. `reason` is optional — the digest LLM is
/// asked for a one-line justification but tolerates omission.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RejectedItem {
    pub content: String,
    #[serde(default)]
    pub reason: Option<String>,
}

/// Path of the `DREAMS.md` file for `scope`.
pub fn dreams_path(workspace: &Path, scope: &MemoryScope) -> PathBuf {
    scope_memory_dir(workspace, scope).join("DREAMS.md")
}

/// Append a new entry to the scope's `DREAMS.md`. Creates the file if
/// missing. After append, prunes entries older than `retention_days`.
pub fn append_dream(
    workspace: &Path,
    entry: &DreamEntry,
    retention_days: u32,
) -> std::io::Result<()> {
    let path = dreams_path(workspace, &entry.scope);
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }

    let existing = std::fs::read_to_string(&path).unwrap_or_default();
    let mut entries = parse_dreams(&existing);
    entries.push(entry.clone());
    entries = prune_old(entries, retention_days);

    let body = format_dreams(&entries);
    let tmp = path.with_extension("md.tmp");
    std::fs::write(&tmp, body)?;
    std::fs::rename(&tmp, &path)?;
    Ok(())
}

/// Read all dream entries from the scope's `DREAMS.md`. Returns an
/// empty vector when the file doesn't exist.
pub fn read_dreams(workspace: &Path, scope: &MemoryScope) -> Vec<DreamEntry> {
    let path = dreams_path(workspace, scope);
    let raw = match std::fs::read_to_string(&path) {
        Ok(s) => s,
        Err(_) => return Vec::new(),
    };
    parse_dreams(&raw)
}

/// Drop entries older than `retention_days`. `0` keeps everything.
fn prune_old(entries: Vec<DreamEntry>, retention_days: u32) -> Vec<DreamEntry> {
    if retention_days == 0 {
        return entries;
    }
    let cutoff = Utc::now().date_naive() - chrono::Duration::days(retention_days as i64);
    entries.into_iter().filter(|e| e.date >= cutoff).collect()
}

/// Format entries back into a Markdown document.
fn format_dreams(entries: &[DreamEntry]) -> String {
    let mut out = String::from("# Dreams\n");
    out.push_str(
        "<!-- Daily digest journal. Auto-generated; safe to read but \
         do not edit by hand — the next digest run will rewrite it. -->\n",
    );
    for e in entries {
        out.push_str(&format!("\n## {} ({})\n\n", e.date, e.scope));
        if !e.summary.trim().is_empty() {
            out.push_str(&format!("Summary: {}\n\n", e.summary.trim()));
        }
        out.push_str(&format!("Promoted ({}):\n", e.promoted.len()));
        for p in &e.promoted {
            out.push_str(&format!("- {}\n", p.trim()));
        }
        out.push_str(&format!("\nRejected ({}):\n", e.rejected.len()));
        for r in &e.rejected {
            match &r.reason {
                Some(reason) => {
                    out.push_str(&format!("- {} — reason: {}\n", r.content.trim(), reason.trim()))
                }
                None => out.push_str(&format!("- {}\n", r.content.trim())),
            }
        }
    }
    out
}

/// Parse a `DREAMS.md` file back into entries. Best-effort: malformed
/// blocks are skipped silently, never panics. The agent never reads
/// dreams so failure modes are purely cosmetic.
fn parse_dreams(content: &str) -> Vec<DreamEntry> {
    let mut out = Vec::new();
    let mut cur: Option<DreamEntry> = None;
    let mut mode = ParseMode::None;

    for line in content.lines() {
        let trimmed = line.trim();
        if let Some(header) = trimmed.strip_prefix("## ") {
            // Push the in-progress entry.
            if let Some(e) = cur.take() {
                out.push(e);
            }
            // Header form: "YYYY-MM-DD (scope_str)"
            if let Some((date_str, rest)) = header.split_once(' ')
                && let Ok(date) = NaiveDate::parse_from_str(date_str, "%Y-%m-%d")
            {
                let scope_str = rest.trim_start_matches('(').trim_end_matches(')');
                if let Ok(scope) = scope_str.parse::<MemoryScope>() {
                    cur = Some(DreamEntry {
                        date,
                        scope,
                        summary: String::new(),
                        promoted: Vec::new(),
                        rejected: Vec::new(),
                    });
                    mode = ParseMode::None;
                }
            }
            continue;
        }

        let Some(e) = cur.as_mut() else { continue };

        if let Some(rest) = trimmed.strip_prefix("Summary:") {
            e.summary = rest.trim().to_string();
            mode = ParseMode::None;
        } else if trimmed.starts_with("Promoted") {
            mode = ParseMode::Promoted;
        } else if trimmed.starts_with("Rejected") {
            mode = ParseMode::Rejected;
        } else if let Some(item) = trimmed.strip_prefix("- ") {
            match mode {
                ParseMode::Promoted => e.promoted.push(item.to_string()),
                ParseMode::Rejected => {
                    let (content, reason) = match item.split_once(" — reason: ") {
                        Some((c, r)) => (c.trim().to_string(), Some(r.trim().to_string())),
                        None => (item.to_string(), None),
                    };
                    e.rejected.push(RejectedItem { content, reason });
                }
                ParseMode::None => {}
            }
        }
    }
    if let Some(e) = cur.take() {
        out.push(e);
    }
    out
}

#[derive(Clone, Copy)]
enum ParseMode {
    None,
    Promoted,
    Rejected,
}

/// Convenience constructor used by the digest job.
pub fn build_entry(
    scope: MemoryScope,
    summary: impl Into<String>,
    promoted: Vec<String>,
    rejected: Vec<RejectedItem>,
) -> DreamEntry {
    DreamEntry {
        date: Utc::now().date_naive(),
        scope,
        summary: summary.into(),
        promoted,
        rejected,
    }
}

/// Convenience: drop the file entirely. Used by the test suite.
#[cfg(test)]
pub fn clear_dreams(workspace: &Path, scope: &MemoryScope) -> std::io::Result<()> {
    let p = dreams_path(workspace, scope);
    if p.exists() {
        std::fs::remove_file(p)?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn proj() -> MemoryScope {
        MemoryScope::Project
    }

    #[test]
    fn round_trip_one_entry() {
        let entries = vec![DreamEntry {
            date: NaiveDate::from_ymd_opt(2026, 4, 20).unwrap(),
            scope: proj(),
            summary: "Quiet day, two new rules".into(),
            promoted: vec!["use tabs".into(), "no unwrap".into()],
            rejected: vec![RejectedItem {
                content: "vague hunch".into(),
                reason: Some("not actionable".into()),
            }],
        }];
        let formatted = format_dreams(&entries);
        let parsed = parse_dreams(&formatted);
        assert_eq!(parsed.len(), 1);
        assert_eq!(parsed[0].summary, "Quiet day, two new rules");
        assert_eq!(parsed[0].promoted, vec!["use tabs", "no unwrap"]);
        assert_eq!(parsed[0].rejected.len(), 1);
        assert_eq!(parsed[0].rejected[0].content, "vague hunch");
        assert_eq!(parsed[0].rejected[0].reason.as_deref(), Some("not actionable"));
    }

    #[test]
    fn parse_rejects_without_reason() {
        let md = "# Dreams\n\n## 2026-04-20 (project)\n\nSummary: x\n\nPromoted (0):\n\nRejected (1):\n- foo\n";
        let parsed = parse_dreams(md);
        assert_eq!(parsed.len(), 1);
        assert_eq!(parsed[0].rejected.len(), 1);
        assert!(parsed[0].rejected[0].reason.is_none());
    }

    #[test]
    fn prune_drops_old_entries() {
        let old = NaiveDate::from_ymd_opt(2000, 1, 1).unwrap();
        let new = Utc::now().date_naive();
        let entries = vec![
            DreamEntry {
                date: old,
                scope: proj(),
                summary: "old".into(),
                promoted: vec![],
                rejected: vec![],
            },
            DreamEntry {
                date: new,
                scope: proj(),
                summary: "new".into(),
                promoted: vec![],
                rejected: vec![],
            },
        ];
        let pruned = prune_old(entries, 30);
        assert_eq!(pruned.len(), 1);
        assert_eq!(pruned[0].summary, "new");
    }

    #[test]
    fn prune_zero_keeps_all() {
        let old = NaiveDate::from_ymd_opt(1999, 1, 1).unwrap();
        let entries = vec![DreamEntry {
            date: old,
            scope: proj(),
            summary: "ancient".into(),
            promoted: vec![],
            rejected: vec![],
        }];
        let pruned = prune_old(entries, 0);
        assert_eq!(pruned.len(), 1);
    }

    #[test]
    fn build_entry_uses_today() {
        let e = build_entry(proj(), "x", vec![], vec![]);
        assert_eq!(e.date, Utc::now().date_naive());
    }
}
