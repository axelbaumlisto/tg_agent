use std::path::{Path, PathBuf};

use chrono::{DateTime, NaiveDate, Utc};

use super::types::{MemoryEntry, MemoryScope, MemoryType};

pub const MAX_ENTRIES_PER_FILE: usize = 100;
pub const MAX_ENTRY_CHARS: usize = 500;
pub const MAX_INJECTION_CHARS: usize = 4000;

/// Compute a stable, case/whitespace-insensitive hash of memory content.
/// Used by `append_dedup` to detect duplicates without depending on the
/// exact UUID of an existing entry. Collisions are extremely unlikely
/// for short markdown lines and the dedup is best-effort anyway —
/// false positives only suppress an identical line.
pub fn content_hash(content: &str) -> u64 {
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};
    let mut h = DefaultHasher::new();
    content
        .trim()
        .to_lowercase()
        .split_whitespace()
        .collect::<Vec<_>>()
        .join(" ")
        .hash(&mut h);
    h.finish()
}

/// Reads and writes MEMORY.md files in a structured markdown format.
pub struct MarkdownMemoryStore;

impl MarkdownMemoryStore {
    /// Convert a workspace path to a slug: `/home/spex/work` -> `-home-spex-work`
    pub fn project_slug(workspace: &Path) -> String {
        let s = workspace.to_string_lossy();
        let slug = s.replace('/', "-");
        if slug.starts_with('-') {
            slug
        } else {
            format!("-{slug}")
        }
    }

    pub fn naked_home() -> PathBuf {
        if let Ok(v) = std::env::var("NAKED_HOME") {
            return PathBuf::from(v);
        }
        dirs_home().join(".naked")
    }

    pub fn global_memory_path() -> PathBuf {
        Self::naked_home().join("memory/MEMORY.md")
    }

    pub fn project_memory_path(workspace: &Path) -> PathBuf {
        let slug = Self::project_slug(workspace);
        Self::naked_home()
            .join("projects")
            .join(slug)
            .join("memory/MEMORY.md")
    }

    /// Per-user memory file. `user_id` is sanitized: non-alphanumeric chars become `_`.
    pub fn user_memory_path(user_id: &str) -> PathBuf {
        let slug = sanitize_user_id(user_id);
        Self::naked_home()
            .join("users")
            .join(slug)
            .join("memory/MEMORY.md")
    }

    /// Parse a MEMORY.md file into entries.
    pub fn load(path: &Path, scope: MemoryScope) -> Vec<MemoryEntry> {
        let content = match std::fs::read_to_string(path) {
            Ok(c) => c,
            Err(_) => return Vec::new(),
        };
        parse_memory_md(&content, scope)
    }

    /// Write entries to a MEMORY.md file (atomic: .tmp + rename).
    pub fn save(path: &Path, entries: &[MemoryEntry]) -> std::io::Result<()> {
        let content = format_memory_md(entries);
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        let tmp = path.with_extension("md.tmp");
        std::fs::write(&tmp, &content)?;
        std::fs::rename(&tmp, path)?;
        Ok(())
    }

    /// Append a single entry. Enforces MAX_ENTRIES_PER_FILE by evicting oldest.
    /// Returns `false` if duplicate content already exists.
    pub fn append(path: &Path, entry: &MemoryEntry) -> std::io::Result<bool> {
        Self::append_dedup(path, entry, true)
    }

    /// Append with explicit dedup control.
    ///
    /// When `dedup=true` (the default for `append`), an entry whose
    /// `content_hash` matches an existing one is skipped — but the
    /// existing entry's `created_at` is **refreshed to today** so
    /// repeated mentions keep the rule "young" in retention metrics.
    /// Returns `false` if dedup hit, `true` if a new entry was written.
    ///
    /// When `dedup=false`, the entry is always appended (useful for
    /// daily-draft files where we want to count repetitions).
    pub fn append_dedup(
        path: &Path,
        entry: &MemoryEntry,
        dedup: bool,
    ) -> std::io::Result<bool> {
        let mut entries = Self::load(path, entry.scope.clone());

        if dedup {
            let new_hash = content_hash(&entry.content);
            if let Some(existing) = entries
                .iter_mut()
                .find(|e| content_hash(&e.content) == new_hash)
            {
                // Refresh the timestamp so repeated drafts keep the
                // rule fresh in retention/scoring windows. We
                // deliberately do not bump the `id` so external
                // references stay valid.
                existing.created_at = Utc::now();
                Self::save(path, &entries)?;
                return Ok(false);
            }
        }

        entries.push(entry.clone());

        while entries.len() > MAX_ENTRIES_PER_FILE {
            entries.remove(0);
        }

        Self::save(path, &entries)?;
        Ok(true)
    }

    // ── Daily-draft files ──────────────────────────────────────────────────
    //
    // The two-tier memory system writes per-day "drafts" to
    // `<scope>/memory/YYYY-MM-DD.md`. The daily digest job (see
    // `memory/daily.rs`) reads yesterday's draft, scores entries, and
    // promotes survivors into `MEMORY.md`. Old daily files are pruned
    // by `rotate_daily_files`.

    /// Path of the daily-draft file for `scope` on `date` (UTC).
    pub fn daily_path(workspace: &Path, scope: &MemoryScope, date: NaiveDate) -> PathBuf {
        let dir = scope_memory_dir(workspace, scope);
        dir.join(format!("{}.md", date.format("%Y-%m-%d")))
    }

    /// Append `entry` to today's daily-draft file (UTC). Honours
    /// `dedup` semantics from `append_dedup`. Returns `true` when a
    /// new entry was created.
    pub fn append_daily(
        workspace: &Path,
        entry: &MemoryEntry,
        dedup: bool,
    ) -> std::io::Result<bool> {
        let today = Utc::now().date_naive();
        let path = Self::daily_path(workspace, &entry.scope, today);
        Self::append_dedup(&path, entry, dedup)
    }

    /// Read a single daily file. Returns `Vec::new()` if the file is
    /// missing.
    pub fn read_daily(
        workspace: &Path,
        scope: &MemoryScope,
        date: NaiveDate,
    ) -> Vec<MemoryEntry> {
        Self::load(&Self::daily_path(workspace, scope, date), scope.clone())
    }

    /// List daily-draft files present on disk for `scope`, newest
    /// first. Returns the parsed dates only — call `read_daily` to
    /// load each file's contents on demand.
    pub fn list_daily(workspace: &Path, scope: &MemoryScope) -> Vec<NaiveDate> {
        let dir = scope_memory_dir(workspace, scope);
        let Ok(entries) = std::fs::read_dir(&dir) else {
            return Vec::new();
        };
        let mut dates: Vec<NaiveDate> = entries
            .filter_map(|e| e.ok())
            .filter_map(|e| {
                let name = e.file_name();
                let s = name.to_str()?;
                let stem = s.strip_suffix(".md")?;
                NaiveDate::parse_from_str(stem, "%Y-%m-%d").ok()
            })
            .collect();
        dates.sort();
        dates.reverse();
        dates
    }

    /// Delete daily-draft files older than `keep_days` (UTC). Returns
    /// the number of files removed.
    pub fn rotate_daily_files(
        workspace: &Path,
        scope: &MemoryScope,
        keep_days: u32,
    ) -> std::io::Result<usize> {
        if keep_days == 0 {
            return Ok(0);
        }
        let cutoff = Utc::now().date_naive() - chrono::Duration::days(keep_days as i64);
        let mut removed = 0usize;
        for date in Self::list_daily(workspace, scope) {
            if date < cutoff {
                let p = Self::daily_path(workspace, scope, date);
                if std::fs::remove_file(&p).is_ok() {
                    removed += 1;
                }
            }
        }
        Ok(removed)
    }

    /// Most-recent N daily-draft dates for `scope` (newest first).
    /// Convenience wrapper around `list_daily` that caps the result.
    pub fn recent_daily_dates(
        workspace: &Path,
        scope: &MemoryScope,
        n: usize,
    ) -> Vec<NaiveDate> {
        let mut v = Self::list_daily(workspace, scope);
        v.truncate(n);
        v
    }

    /// `created_at` of the newest entry across all daily-draft files
    /// for `scope`. Used by the daily-digest scheduler to decide
    /// whether anything happened "today" worth digesting.
    pub fn last_daily_activity(
        workspace: &Path,
        scope: &MemoryScope,
    ) -> Option<DateTime<Utc>> {
        let mut latest: Option<DateTime<Utc>> = None;
        for date in Self::list_daily(workspace, scope) {
            for e in Self::read_daily(workspace, scope, date) {
                latest = Some(latest.map_or(e.created_at, |x| x.max(e.created_at)));
            }
        }
        latest
    }

    /// Remove an entry by id. Returns true if found and removed.
    pub fn remove(path: &Path, id: &str, scope: MemoryScope) -> std::io::Result<bool> {
        let mut entries = Self::load(path, scope);
        let before = entries.len();
        entries.retain(|e| e.id != id);
        if entries.len() == before {
            return Ok(false);
        }
        Self::save(path, &entries)?;
        Ok(true)
    }

    /// Remove all entries from a file.
    pub fn clear(path: &Path) -> std::io::Result<()> {
        Self::save(path, &[])
    }
}

/// Parse a MEMORY.md document into entries.
fn parse_memory_md(content: &str, scope: MemoryScope) -> Vec<MemoryEntry> {
    let mut entries = Vec::new();
    let mut current_type: Option<MemoryType> = None;

    for line in content.lines() {
        let trimmed = line.trim();

        if let Some(heading) = trimmed.strip_prefix("## ") {
            current_type = MemoryType::from_heading(heading);
            continue;
        }

        if trimmed.starts_with("# ") || trimmed.is_empty() {
            continue;
        }

        if let Some(item) = trimmed.strip_prefix("- ") {
            let memory_type = current_type.unwrap_or(MemoryType::Preference);
            if let Some(entry) = parse_entry_line(item, memory_type, scope.clone()) {
                entries.push(entry);
            }
        }
    }

    entries
}

/// Sanitize a user id for use as a path slug: keep alphanumerics, `_`, `-`,
/// replace everything else with `_`.
fn sanitize_user_id(raw: &str) -> String {
    let s: String = raw
        .chars()
        .map(|c| {
            if c.is_ascii_alphanumeric() || c == '_' || c == '-' {
                c
            } else {
                '_'
            }
        })
        .collect();
    if s.is_empty() { "_".to_string() } else { s }
}

/// Parse `content text <!-- id:xxx created:yyyy-mm-dd source:zzz -->`
fn parse_entry_line(
    line: &str,
    memory_type: MemoryType,
    scope: MemoryScope,
) -> Option<MemoryEntry> {
    let (content, meta) = if let Some(idx) = line.find("<!--") {
        let end = line.find("-->").unwrap_or(line.len());
        let meta_str = &line[idx + 4..end].trim();
        (line[..idx].trim().to_string(), Some(meta_str.to_string()))
    } else {
        (line.trim().to_string(), None)
    };

    if content.is_empty() {
        return None;
    }

    let mut id = uuid::Uuid::new_v4().to_string()[..8].to_string();
    let mut created_at = Utc::now();
    let mut source = "unknown".to_string();
    let mut recall_count: u32 = 0;
    let mut last_recalled_at: Option<chrono::DateTime<Utc>> = None;

    if let Some(meta) = meta {
        for part in meta.split_whitespace() {
            if let Some(val) = part.strip_prefix("id:") {
                id = val.to_string();
            } else if let Some(val) = part.strip_prefix("created:") {
                if let Ok(dt) = chrono::NaiveDate::parse_from_str(val, "%Y-%m-%d") {
                    created_at = dt.and_hms_opt(0, 0, 0).unwrap().and_utc();
                }
            } else if let Some(val) = part.strip_prefix("source:") {
                source = val.to_string();
            } else if let Some(val) = part.strip_prefix("recall:") {
                if let Ok(n) = val.parse::<u32>() {
                    recall_count = n;
                }
            } else if let Some(val) = part.strip_prefix("last_recall:")
                && let Ok(dt) = chrono::NaiveDate::parse_from_str(val, "%Y-%m-%d")
            {
                last_recalled_at = Some(dt.and_hms_opt(0, 0, 0).unwrap().and_utc());
            }
        }
    }

    Some(MemoryEntry {
        id,
        memory_type,
        content,
        created_at,
        source,
        scope,
        recall_count,
        last_recalled_at,
    })
}

/// Format entries as a MEMORY.md document.
fn format_memory_md(entries: &[MemoryEntry]) -> String {
    let mut out = String::from("# Memory\n");

    for &ty in MemoryType::ALL {
        let items: Vec<&MemoryEntry> = entries.iter().filter(|e| e.memory_type == ty).collect();
        if items.is_empty() {
            continue;
        }
        out.push_str(&format!("\n## {}\n", ty.section_heading()));
        for entry in items {
            out.push_str(&entry.to_markdown_line());
            out.push('\n');
        }
    }

    out
}

fn dirs_home() -> PathBuf {
    std::env::var("HOME")
        .map(PathBuf::from)
        .unwrap_or_else(|_| PathBuf::from("."))
}

/// Directory that holds `MEMORY.md` and per-day draft files for the
/// given scope. Mirrors the resolution rules in
/// `MarkdownMemoryStore::{global,project,user}_memory_path` but without
/// the trailing `MEMORY.md` segment.
/// Enumerate every user-id (sanitized) that has a `memory/` directory
/// under `<NAKED_HOME>/users/<id>/memory`. Returns `MemoryScope::User`
/// instances ready for iteration by the daily-digest scheduler. The
/// `id` returned is the on-disk slug (already sanitized), not the raw
/// Telegram sender id; that's fine because `MemoryScope::User(slug)`
/// re-sanitizes deterministically.
pub fn list_user_scopes() -> Vec<MemoryScope> {
    let users_dir = MarkdownMemoryStore::naked_home().join("users");
    let Ok(read) = std::fs::read_dir(&users_dir) else {
        return Vec::new();
    };
    let mut out = Vec::new();
    for entry in read.flatten() {
        let path = entry.path();
        if !path.join("memory").is_dir() {
            continue;
        }
        let Some(id) = path.file_name().and_then(|s| s.to_str()) else {
            continue;
        };
        out.push(MemoryScope::User(id.to_string()));
    }
    out
}

pub fn scope_memory_dir(workspace: &Path, scope: &MemoryScope) -> PathBuf {
    let memory_md = match scope {
        MemoryScope::Global => MarkdownMemoryStore::global_memory_path(),
        MemoryScope::Project => MarkdownMemoryStore::project_memory_path(workspace),
        MemoryScope::User(id) => MarkdownMemoryStore::user_memory_path(id),
    };
    memory_md
        .parent()
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("."))
}

#[cfg(test)]
mod tests {
    use super::*;

    const SAMPLE_MD: &str = "\
# Memory

## Preferences
- Use tabs not spaces <!-- id:a1b2c3 created:2026-04-08 source:user -->
- Write tests first <!-- id:d4e5f6 created:2026-04-08 source:auto -->

## Corrections
- Don't use unwrap() <!-- id:g7h8i9 created:2026-04-09 source:auto -->

## Project Knowledge
- API endpoint is /api/v1/ <!-- id:j0k1l2 created:2026-04-08 source:model -->
";

    #[test]
    fn parse_sample_md() {
        let entries = parse_memory_md(SAMPLE_MD, MemoryScope::Project);
        assert_eq!(entries.len(), 4);
        assert_eq!(entries[0].id, "a1b2c3");
        assert_eq!(entries[0].content, "Use tabs not spaces");
        assert_eq!(entries[0].memory_type, MemoryType::Preference);
        assert_eq!(entries[0].source, "user");

        assert_eq!(entries[2].memory_type, MemoryType::Correction);
        assert_eq!(entries[3].memory_type, MemoryType::ProjectKnowledge);
    }

    #[test]
    fn format_roundtrip() {
        let entries = parse_memory_md(SAMPLE_MD, MemoryScope::Project);
        let formatted = format_memory_md(&entries);
        let reparsed = parse_memory_md(&formatted, MemoryScope::Project);
        assert_eq!(entries.len(), reparsed.len());
        for (a, b) in entries.iter().zip(reparsed.iter()) {
            assert_eq!(a.id, b.id);
            assert_eq!(a.content, b.content);
            assert_eq!(a.memory_type, b.memory_type);
        }
    }

    #[test]
    fn project_slug_conversion() {
        assert_eq!(
            MarkdownMemoryStore::project_slug(Path::new("/home/spex/work/erp")),
            "-home-spex-work-erp"
        );
        assert_eq!(
            MarkdownMemoryStore::project_slug(Path::new("relative/path")),
            "-relative-path"
        );
    }

    #[test]
    fn save_and_load_file() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("MEMORY.md");

        let entries = vec![
            MemoryEntry::new(
                MemoryType::Preference,
                "Use Rust".into(),
                "user",
                MemoryScope::Project,
            ),
            MemoryEntry::new(
                MemoryType::Correction,
                "No unwrap".into(),
                "auto",
                MemoryScope::Project,
            ),
        ];

        MarkdownMemoryStore::save(&path, &entries).unwrap();
        let loaded = MarkdownMemoryStore::load(&path, MemoryScope::Project);
        assert_eq!(loaded.len(), 2);
        assert_eq!(loaded[0].content, "Use Rust");
        assert_eq!(loaded[1].content, "No unwrap");
    }

    #[test]
    fn append_deduplicates() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("MEMORY.md");

        let entry = MemoryEntry::new(
            MemoryType::Preference,
            "Use tabs".into(),
            "user",
            MemoryScope::Project,
        );
        assert!(MarkdownMemoryStore::append(&path, &entry).unwrap());
        assert!(!MarkdownMemoryStore::append(&path, &entry).unwrap());

        let loaded = MarkdownMemoryStore::load(&path, MemoryScope::Project);
        assert_eq!(loaded.len(), 1);
    }

    #[test]
    fn append_evicts_oldest_when_full() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("MEMORY.md");

        let mut entries = Vec::new();
        for i in 0..MAX_ENTRIES_PER_FILE {
            entries.push(MemoryEntry::new(
                MemoryType::Preference,
                format!("rule {i}"),
                "auto",
                MemoryScope::Project,
            ));
        }
        MarkdownMemoryStore::save(&path, &entries).unwrap();

        let new = MemoryEntry::new(
            MemoryType::Preference,
            "rule new".into(),
            "auto",
            MemoryScope::Project,
        );
        MarkdownMemoryStore::append(&path, &new).unwrap();

        let loaded = MarkdownMemoryStore::load(&path, MemoryScope::Project);
        assert_eq!(loaded.len(), MAX_ENTRIES_PER_FILE);
        assert_eq!(loaded[0].content, "rule 1");
        assert_eq!(loaded.last().unwrap().content, "rule new");
    }

    #[test]
    fn remove_entry_by_id() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("MEMORY.md");

        let e1 = MemoryEntry::new(
            MemoryType::Preference,
            "keep me".into(),
            "user",
            MemoryScope::Project,
        );
        let e2 = MemoryEntry::new(
            MemoryType::Correction,
            "delete me".into(),
            "auto",
            MemoryScope::Project,
        );
        MarkdownMemoryStore::save(&path, &[e1, e2.clone()]).unwrap();

        assert!(MarkdownMemoryStore::remove(&path, &e2.id, MemoryScope::Project).unwrap());
        let loaded = MarkdownMemoryStore::load(&path, MemoryScope::Project);
        assert_eq!(loaded.len(), 1);
        assert_eq!(loaded[0].content, "keep me");
    }

    #[test]
    fn clear_removes_all() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("MEMORY.md");

        let entry = MemoryEntry::new(
            MemoryType::Failure,
            "test fail".into(),
            "auto",
            MemoryScope::Global,
        );
        MarkdownMemoryStore::append(&path, &entry).unwrap();
        MarkdownMemoryStore::clear(&path).unwrap();

        let loaded = MarkdownMemoryStore::load(&path, MemoryScope::Global);
        assert!(loaded.is_empty());
    }

    #[test]
    fn parse_entry_without_metadata() {
        let md = "# Memory\n\n## Preferences\n- Simple rule without metadata\n";
        let entries = parse_memory_md(md, MemoryScope::Global);
        assert_eq!(entries.len(), 1);
        assert_eq!(entries[0].content, "Simple rule without metadata");
        assert_eq!(entries[0].source, "unknown");
    }

    #[test]
    fn content_hash_is_normalized() {
        assert_eq!(content_hash("Use Tabs"), content_hash("  use   tabs  "));
        assert_ne!(content_hash("Use tabs"), content_hash("Use spaces"));
    }

    #[test]
    fn append_dedup_refreshes_timestamp_on_duplicate() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("MEMORY.md");

        let mut e = MemoryEntry::new(
            MemoryType::Preference,
            "Use tabs".into(),
            "user",
            MemoryScope::Project,
        );
        // Pretend the original entry was created a year ago.
        e.created_at = Utc::now() - chrono::Duration::days(365);
        MarkdownMemoryStore::save(&path, std::slice::from_ref(&e)).unwrap();

        let dup = MemoryEntry::new(
            MemoryType::Preference,
            "  USE   tabs  ".into(),
            "auto",
            MemoryScope::Project,
        );
        // dedup=true → returns false (no new entry), but refreshes
        // the existing entry's `created_at` to "now".
        assert!(!MarkdownMemoryStore::append_dedup(&path, &dup, true).unwrap());
        let loaded = MarkdownMemoryStore::load(&path, MemoryScope::Project);
        assert_eq!(loaded.len(), 1);
        let age_days = (Utc::now() - loaded[0].created_at).num_days();
        assert!(age_days < 2, "expected refreshed timestamp, got {age_days}d old");
    }

    #[test]
    fn append_dedup_false_always_appends() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("MEMORY.md");

        let e = MemoryEntry::new(
            MemoryType::Preference,
            "Use tabs".into(),
            "user",
            MemoryScope::Project,
        );
        assert!(MarkdownMemoryStore::append_dedup(&path, &e, false).unwrap());
        assert!(MarkdownMemoryStore::append_dedup(&path, &e, false).unwrap());
        assert_eq!(MarkdownMemoryStore::load(&path, MemoryScope::Project).len(), 2);
    }

    /// Daily-file format round trip directly via the underlying
    /// load/save plumbing — exercises parsing of the
    /// `memory/YYYY-MM-DD.md` filename pattern without touching
    /// `NAKED_HOME` (which would require `unsafe` env mutation under
    /// `forbid(unsafe_code)`).
    #[test]
    fn daily_file_format_round_trip_via_path() {
        let dir = tempfile::tempdir().unwrap();
        let today = Utc::now().date_naive();
        let path = dir.path().join(format!("{}.md", today.format("%Y-%m-%d")));

        let entry = MemoryEntry::new(
            MemoryType::Preference,
            "draft rule".into(),
            "auto",
            MemoryScope::Project,
        );
        assert!(MarkdownMemoryStore::append_dedup(&path, &entry, true).unwrap());

        let loaded = MarkdownMemoryStore::load(&path, MemoryScope::Project);
        assert_eq!(loaded.len(), 1);
        assert_eq!(loaded[0].content, "draft rule");
    }
}
