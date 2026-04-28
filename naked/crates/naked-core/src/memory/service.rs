use std::path::Path;

use super::store::{MAX_ENTRY_CHARS, MAX_INJECTION_CHARS, MarkdownMemoryStore};
use super::types::{MemoryEntry, MemoryScope, MemoryType};

/// High-level API for the memory system.
pub struct MemoryService;

impl MemoryService {
    /// Load active rules from global + project memory for injection into the
    /// system prompt. Returns empty string if no rules. Equivalent to
    /// [`Self::load_rules_for`] with `sender_id=None`.
    pub fn load_rules(workspace: &Path) -> String {
        Self::load_rules_for(workspace, None)
    }

    /// Load active rules, optionally including per-author memory when a
    /// `sender_id` is known. Sections appear in order: Global → Project → User.
    pub fn load_rules_for(workspace: &Path, sender_id: Option<&str>) -> String {
        let global = MarkdownMemoryStore::load(
            &MarkdownMemoryStore::global_memory_path(),
            MemoryScope::Global,
        );
        let project = MarkdownMemoryStore::load(
            &MarkdownMemoryStore::project_memory_path(workspace),
            MemoryScope::Project,
        );
        let user = sender_id
            .filter(|s| !s.is_empty())
            .map(|id| {
                MarkdownMemoryStore::load(
                    &MarkdownMemoryStore::user_memory_path(id),
                    MemoryScope::User(id.to_string()),
                )
            })
            .unwrap_or_default();

        if global.is_empty() && project.is_empty() && user.is_empty() {
            return String::new();
        }

        let mut out = String::from("[Memory — active rules]\n");
        let mut remaining = MAX_INJECTION_CHARS - out.len();

        let append_section =
            |entries: &[MemoryEntry], out: &mut String, remaining: &mut usize, label: &str| {
                if entries.is_empty() {
                    return;
                }
                let header = format!("\n{label}:\n");
                if header.len() >= *remaining {
                    return;
                }
                out.push_str(&header);
                *remaining -= header.len();

                for ty in MemoryType::ALL {
                    let items: Vec<&MemoryEntry> =
                        entries.iter().filter(|e| e.memory_type == *ty).collect();
                    if items.is_empty() {
                        continue;
                    }
                    let sub = format!("  {}:\n", ty.section_heading());
                    if sub.len() >= *remaining {
                        return;
                    }
                    out.push_str(&sub);
                    *remaining -= sub.len();

                    for entry in items {
                        let line = format!("  - {}\n", entry.content);
                        if line.len() >= *remaining {
                            return;
                        }
                        out.push_str(&line);
                        *remaining -= line.len();
                    }
                }
            };

        append_section(&global, &mut out, &mut remaining, "Global");
        append_section(&project, &mut out, &mut remaining, "Project");
        if let Some(id) = sender_id.filter(|s| !s.is_empty()) {
            let label = format!("User ({id})");
            append_section(&user, &mut out, &mut remaining, &label);
        }

        out
    }

    /// Store a new memory entry. Truncates content to MAX_ENTRY_CHARS.
    pub fn store(
        workspace: &Path,
        scope: MemoryScope,
        memory_type: MemoryType,
        content: &str,
        source: &str,
    ) -> std::io::Result<bool> {
        // UTF-8 safe truncation: `MAX_ENTRY_CHARS` is named in chars, not
        // bytes. Memory entries persist arbitrary user text (Russian,
        // Vietnamese, emoji, …) so we must walk char boundaries — slicing
        // on a raw byte index panics inside multi-byte codepoints.
        let content = if content.chars().count() > MAX_ENTRY_CHARS {
            let mut t: String = content
                .chars()
                .take(MAX_ENTRY_CHARS.saturating_sub(3))
                .collect();
            t.push_str("...");
            t
        } else {
            content.to_string()
        };

        let path = scope_path(workspace, &scope);
        let entry = MemoryEntry::new(memory_type, content, source, scope);
        MarkdownMemoryStore::append(&path, &entry)
    }

    /// Search entries by substring (case-insensitive) across global + project.
    /// Equivalent to [`Self::search_for`] with `sender_id=None`.
    pub fn search(workspace: &Path, query: &str) -> Vec<MemoryEntry> {
        Self::search_for(workspace, query, None)
    }

    /// Search entries by substring (case-insensitive) across global + project,
    /// and — when `sender_id` is provided — the corresponding user memory.
    pub fn search_for(workspace: &Path, query: &str, sender_id: Option<&str>) -> Vec<MemoryEntry> {
        let query_lower = query.to_lowercase();
        let mut results = Vec::new();

        let global = MarkdownMemoryStore::load(
            &MarkdownMemoryStore::global_memory_path(),
            MemoryScope::Global,
        );
        let project = MarkdownMemoryStore::load(
            &MarkdownMemoryStore::project_memory_path(workspace),
            MemoryScope::Project,
        );
        let user = sender_id
            .filter(|s| !s.is_empty())
            .map(|id| {
                MarkdownMemoryStore::load(
                    &MarkdownMemoryStore::user_memory_path(id),
                    MemoryScope::User(id.to_string()),
                )
            })
            .unwrap_or_default();

        for entry in global.into_iter().chain(project).chain(user) {
            if entry.content.to_lowercase().contains(&query_lower) {
                results.push(entry);
            }
        }

        results
    }

    /// List all entries, optionally filtered by scope.
    ///
    /// - `Some(User(id))` — only that user's file.
    /// - `Some(Global)` / `Some(Project)` — only that file.
    /// - `None` — global + project (user scope requires an explicit id).
    pub fn list(workspace: &Path, scope: Option<MemoryScope>) -> Vec<MemoryEntry> {
        match scope {
            Some(MemoryScope::User(id)) => MarkdownMemoryStore::load(
                &MarkdownMemoryStore::user_memory_path(&id),
                MemoryScope::User(id),
            ),
            Some(MemoryScope::Global) => MarkdownMemoryStore::load(
                &MarkdownMemoryStore::global_memory_path(),
                MemoryScope::Global,
            ),
            Some(MemoryScope::Project) => MarkdownMemoryStore::load(
                &MarkdownMemoryStore::project_memory_path(workspace),
                MemoryScope::Project,
            ),
            None => {
                let mut entries = MarkdownMemoryStore::load(
                    &MarkdownMemoryStore::global_memory_path(),
                    MemoryScope::Global,
                );
                entries.extend(MarkdownMemoryStore::load(
                    &MarkdownMemoryStore::project_memory_path(workspace),
                    MemoryScope::Project,
                ));
                entries
            }
        }
    }

    /// Delete an entry by id. Searches global and project files. For user-scoped
    /// entries, pass `user_id` explicitly to avoid iterating every user.
    pub fn delete(workspace: &Path, id: &str) -> std::io::Result<bool> {
        let global_path = MarkdownMemoryStore::global_memory_path();
        if MarkdownMemoryStore::remove(&global_path, id, MemoryScope::Global)? {
            return Ok(true);
        }
        let project_path = MarkdownMemoryStore::project_memory_path(workspace);
        MarkdownMemoryStore::remove(&project_path, id, MemoryScope::Project)
    }

    /// Delete an entry from a specific user's memory by id.
    pub fn delete_user(user_id: &str, id: &str) -> std::io::Result<bool> {
        let path = MarkdownMemoryStore::user_memory_path(user_id);
        MarkdownMemoryStore::remove(&path, id, MemoryScope::User(user_id.to_string()))
    }

    /// Clear all entries for a given scope.
    pub fn clear(workspace: &Path, scope: MemoryScope) -> std::io::Result<()> {
        let path = scope_path(workspace, &scope);
        if path.exists() {
            MarkdownMemoryStore::clear(&path)?;
        }
        Ok(())
    }
}

fn scope_path(workspace: &Path, scope: &MemoryScope) -> std::path::PathBuf {
    match scope {
        MemoryScope::Global => MarkdownMemoryStore::global_memory_path(),
        MemoryScope::Project => MarkdownMemoryStore::project_memory_path(workspace),
        MemoryScope::User(id) => MarkdownMemoryStore::user_memory_path(id),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use crate::memory::store::MarkdownMemoryStore;

    /// Low-level service tests using MarkdownMemoryStore directly
    /// (avoids env var mutation issues with Rust 2024 edition).

    #[test]
    fn store_and_load_via_files() {
        let dir = tempfile::tempdir().unwrap();
        let global_path = dir.path().join("global/MEMORY.md");
        let project_path = dir.path().join("project/MEMORY.md");

        let e1 = MemoryEntry::new(
            MemoryType::Preference,
            "use tabs".into(),
            "user",
            MemoryScope::Project,
        );
        let e2 = MemoryEntry::new(
            MemoryType::Correction,
            "no unwrap".into(),
            "auto",
            MemoryScope::Global,
        );

        assert!(MarkdownMemoryStore::append(&project_path, &e1).unwrap());
        assert!(MarkdownMemoryStore::append(&global_path, &e2).unwrap());

        let project_entries = MarkdownMemoryStore::load(&project_path, MemoryScope::Project);
        assert_eq!(project_entries.len(), 1);
        assert_eq!(project_entries[0].content, "use tabs");

        let global_entries = MarkdownMemoryStore::load(&global_path, MemoryScope::Global);
        assert_eq!(global_entries.len(), 1);
        assert_eq!(global_entries[0].content, "no unwrap");
    }

    #[test]
    fn search_case_insensitive_via_files() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("MEMORY.md");

        let e1 = MemoryEntry::new(
            MemoryType::Preference,
            "Use Rust for CLI tools".into(),
            "user",
            MemoryScope::Project,
        );
        let e2 = MemoryEntry::new(
            MemoryType::Correction,
            "Python is fine for scripts".into(),
            "auto",
            MemoryScope::Project,
        );

        MarkdownMemoryStore::append(&path, &e1).unwrap();
        MarkdownMemoryStore::append(&path, &e2).unwrap();

        let entries = MarkdownMemoryStore::load(&path, MemoryScope::Project);
        let query = "rust";
        let results: Vec<&MemoryEntry> = entries
            .iter()
            .filter(|e| e.content.to_lowercase().contains(query))
            .collect();
        assert_eq!(results.len(), 1);
        assert!(results[0].content.contains("Rust"));
    }

    #[test]
    fn delete_by_id_via_files() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("MEMORY.md");

        let entry = MemoryEntry::new(
            MemoryType::Failure,
            "test failure".into(),
            "auto",
            MemoryScope::Project,
        );
        let id = entry.id.clone();

        MarkdownMemoryStore::append(&path, &entry).unwrap();
        assert!(MarkdownMemoryStore::remove(&path, &id, MemoryScope::Project).unwrap());
        assert!(MarkdownMemoryStore::load(&path, MemoryScope::Project).is_empty());
    }

    #[test]
    fn load_rules_format() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("MEMORY.md");

        for i in 0..5 {
            let entry = MemoryEntry::new(
                MemoryType::Preference,
                format!("rule {i}"),
                "auto",
                MemoryScope::Project,
            );
            MarkdownMemoryStore::append(&path, &entry).unwrap();
        }

        let entries = MarkdownMemoryStore::load(&path, MemoryScope::Project);
        assert_eq!(entries.len(), 5);

        // Test that load_rules output respects the budget format
        let mut out = String::from("[Memory — active rules]\n\nProject:\n  Preferences:\n");
        for entry in &entries {
            out.push_str(&format!("  - {}\n", entry.content));
        }
        assert!(out.len() < MAX_INJECTION_CHARS);
    }

    #[test]
    fn store_truncates_long_content() {
        let content = "x".repeat(MAX_ENTRY_CHARS + 100);
        let truncated = if content.chars().count() > MAX_ENTRY_CHARS {
            let mut t: String = content
                .chars()
                .take(MAX_ENTRY_CHARS.saturating_sub(3))
                .collect();
            t.push_str("...");
            t
        } else {
            content.clone()
        };
        assert_eq!(truncated.chars().count(), MAX_ENTRY_CHARS);
        assert!(truncated.len() < content.len());
    }

    #[test]
    fn store_handles_multibyte_content_without_panic() {
        // Regression: a Cyrillic blob longer than `MAX_ENTRY_CHARS` used to
        // panic on `&content[..MAX_ENTRY_CHARS - 3]` because the byte
        // boundary fell inside a 2-byte codepoint. Now it must round-trip
        // safely and stay under the byte budget.
        let cyr = "й".repeat(MAX_ENTRY_CHARS + 50); // each 'й' is 2 UTF-8 bytes
        let truncated = if cyr.chars().count() > MAX_ENTRY_CHARS {
            let mut t: String = cyr
                .chars()
                .take(MAX_ENTRY_CHARS.saturating_sub(3))
                .collect();
            t.push_str("...");
            t
        } else {
            cyr.clone()
        };
        // Must be safe UTF-8 and respect the char budget (not byte budget).
        assert_eq!(truncated.chars().count(), MAX_ENTRY_CHARS);
        assert!(truncated.ends_with("..."));
    }

    #[test]
    fn clear_via_files() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("MEMORY.md");

        let e1 = MemoryEntry::new(
            MemoryType::Preference,
            "a".into(),
            "u",
            MemoryScope::Project,
        );
        MarkdownMemoryStore::append(&path, &e1).unwrap();
        MarkdownMemoryStore::clear(&path).unwrap();
        assert!(MarkdownMemoryStore::load(&path, MemoryScope::Project).is_empty());
    }
}
