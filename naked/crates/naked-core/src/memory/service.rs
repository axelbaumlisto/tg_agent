use std::path::Path;

use super::store::{MAX_ENTRY_CHARS, MAX_INJECTION_CHARS, MarkdownMemoryStore};
use super::types::{MemoryEntry, MemoryScope, MemoryType};
use crate::config::MemoryConfig;

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

    /// Load active rules with memory config available to the per-turn hot path.
    /// Default-off `memory_scope_priority_injection_enabled=false` calls the
    /// legacy renderer above directly. When enabled, all Corrections from all
    /// scopes render first (User → Project → Global, newest first), then the
    /// remaining types render by User → Project → Global and by type.
    pub fn load_rules_for_with_config(
        workspace: &Path,
        sender_id: Option<&str>,
        config: &MemoryConfig,
    ) -> String {
        if !config.memory_scope_priority_injection_enabled {
            return Self::load_rules_for(workspace, sender_id);
        }

        let loaded = LoadedMemories::load(workspace, sender_id);
        let (rendered, dropped) = render_scope_priority(&loaded, sender_id);
        record_injection_dropped(dropped);
        rendered
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

#[derive(Debug, Default)]
struct LoadedMemories {
    global: Vec<MemoryEntry>,
    project: Vec<MemoryEntry>,
    user: Vec<MemoryEntry>,
}

impl LoadedMemories {
    fn load(workspace: &Path, sender_id: Option<&str>) -> Self {
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
        Self {
            global,
            project,
            user,
        }
    }

    fn is_empty(&self) -> bool {
        self.global.is_empty() && self.project.is_empty() && self.user.is_empty()
    }
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
struct DroppedByScope {
    global: usize,
    project: usize,
    user: usize,
}

impl DroppedByScope {
    fn add(&mut self, scope: ScopeBucket, count: usize) {
        match scope {
            ScopeBucket::Global => self.global += count,
            ScopeBucket::Project => self.project += count,
            ScopeBucket::User => self.user += count,
        }
    }

    fn total(self) -> usize {
        self.global + self.project + self.user
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ScopeBucket {
    Global,
    Project,
    User,
}

const NON_CORRECTION_TYPES: [MemoryType; 3] = [
    MemoryType::Preference,
    MemoryType::ProjectKnowledge,
    MemoryType::Failure,
];

#[derive(Debug)]
struct PriorityGroup<'a> {
    scope: ScopeBucket,
    label: String,
    memory_type: MemoryType,
    entries: Vec<&'a MemoryEntry>,
}

fn render_scope_priority(
    loaded: &LoadedMemories,
    sender_id: Option<&str>,
) -> (String, DroppedByScope) {
    if loaded.is_empty() {
        return (String::new(), DroppedByScope::default());
    }

    let mut groups = Vec::new();
    let user_label = sender_id
        .filter(|s| !s.is_empty())
        .map(|id| format!("User ({id})"));

    if let Some(label) = &user_label {
        push_priority_group(
            &mut groups,
            &loaded.user,
            label.clone(),
            ScopeBucket::User,
            MemoryType::Correction,
        );
    }
    push_priority_group(
        &mut groups,
        &loaded.project,
        "Project".to_string(),
        ScopeBucket::Project,
        MemoryType::Correction,
    );
    push_priority_group(
        &mut groups,
        &loaded.global,
        "Global".to_string(),
        ScopeBucket::Global,
        MemoryType::Correction,
    );

    if let Some(label) = &user_label {
        for memory_type in NON_CORRECTION_TYPES {
            push_priority_group(
                &mut groups,
                &loaded.user,
                label.clone(),
                ScopeBucket::User,
                memory_type,
            );
        }
    }
    for memory_type in NON_CORRECTION_TYPES {
        push_priority_group(
            &mut groups,
            &loaded.project,
            "Project".to_string(),
            ScopeBucket::Project,
            memory_type,
        );
    }
    for memory_type in NON_CORRECTION_TYPES {
        push_priority_group(
            &mut groups,
            &loaded.global,
            "Global".to_string(),
            ScopeBucket::Global,
            memory_type,
        );
    }

    append_priority_groups(&groups)
}

fn push_priority_group<'a>(
    groups: &mut Vec<PriorityGroup<'a>>,
    entries: &'a [MemoryEntry],
    label: String,
    scope: ScopeBucket,
    memory_type: MemoryType,
) {
    let mut items: Vec<&MemoryEntry> = entries
        .iter()
        .filter(|entry| entry.memory_type == memory_type)
        .collect();
    if items.is_empty() {
        return;
    }
    items.sort_by(|a, b| {
        b.created_at
            .cmp(&a.created_at)
            .then_with(|| a.id.cmp(&b.id))
    });
    groups.push(PriorityGroup {
        scope,
        label,
        memory_type,
        entries: items,
    });
}

fn append_priority_groups(groups: &[PriorityGroup<'_>]) -> (String, DroppedByScope) {
    let mut out = String::from(
        "[Memory — active rules]
",
    );
    let mut remaining = MAX_INJECTION_CHARS - out.len();
    let mut dropped = DroppedByScope::default();

    for (group_idx, group) in groups.iter().enumerate() {
        let header = format!(
            "
{}:
",
            group.label
        );
        if header.len() >= remaining {
            count_remaining_groups(groups, group_idx, 0, &mut dropped);
            return (out, dropped);
        }
        out.push_str(&header);
        remaining -= header.len();

        let sub = format!(
            "  {}:
",
            group.memory_type.section_heading()
        );
        if sub.len() >= remaining {
            count_remaining_groups(groups, group_idx, 0, &mut dropped);
            return (out, dropped);
        }
        out.push_str(&sub);
        remaining -= sub.len();

        for (entry_idx, entry) in group.entries.iter().enumerate() {
            let line = format!(
                "  - {}
",
                entry.content
            );
            if line.len() >= remaining {
                count_remaining_groups(groups, group_idx, entry_idx, &mut dropped);
                return (out, dropped);
            }
            out.push_str(&line);
            remaining -= line.len();
        }
    }

    (out, dropped)
}

fn count_remaining_groups(
    groups: &[PriorityGroup<'_>],
    group_idx: usize,
    entry_idx: usize,
    dropped: &mut DroppedByScope,
) {
    for (idx, group) in groups.iter().enumerate().skip(group_idx) {
        let start = if idx == group_idx { entry_idx } else { 0 };
        if start < group.entries.len() {
            dropped.add(group.scope, group.entries.len() - start);
        }
    }
}

fn record_injection_dropped(dropped: DroppedByScope) {
    if dropped.total() == 0 {
        return;
    }

    use std::sync::atomic::Ordering;

    if dropped.global > 0 {
        crate::types::MEMORY_INJECTION_DROPPED_GLOBAL_COUNT
            .fetch_add(dropped.global as u64, Ordering::Relaxed);
    }
    if dropped.project > 0 {
        crate::types::MEMORY_INJECTION_DROPPED_PROJECT_COUNT
            .fetch_add(dropped.project as u64, Ordering::Relaxed);
    }
    if dropped.user > 0 {
        crate::types::MEMORY_INJECTION_DROPPED_USER_COUNT
            .fetch_add(dropped.user as u64, Ordering::Relaxed);
    }

    tracing::warn!(
        global = dropped.global,
        project = dropped.project,
        user = dropped.user,
        max_chars = MAX_INJECTION_CHARS,
        "memory injection truncated: dropped entries due to prompt budget"
    );
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

    use std::io::Write;
    use std::sync::{Arc, Mutex};

    use chrono::TimeZone;

    use crate::memory::store::{MarkdownMemoryStore, MemoryPaths};

    static S5_TEST_LOCK: Mutex<()> = Mutex::new(());

    #[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
    struct DroppedSnapshot {
        global: u64,
        project: u64,
        user: u64,
    }

    impl DroppedSnapshot {
        fn current() -> Self {
            use std::sync::atomic::Ordering;

            Self {
                global: crate::types::MEMORY_INJECTION_DROPPED_GLOBAL_COUNT.load(Ordering::Relaxed),
                project: crate::types::MEMORY_INJECTION_DROPPED_PROJECT_COUNT
                    .load(Ordering::Relaxed),
                user: crate::types::MEMORY_INJECTION_DROPPED_USER_COUNT.load(Ordering::Relaxed),
            }
        }

        fn delta_since(self, before: Self) -> Self {
            // Deltas over PROCESS-GLOBAL counters: another test finishing
            // between the two snapshots can only ever raise them, but a future
            // reset path would make a plain `-` panic in debug. Saturating is
            // the same value in the normal case and a soft 0 in the pathological
            // one.
            Self {
                global: self.global.saturating_sub(before.global),
                project: self.project.saturating_sub(before.project),
                user: self.user.saturating_sub(before.user),
            }
        }
    }

    #[derive(Clone)]
    struct SharedLog(Arc<Mutex<Vec<u8>>>);

    impl Write for SharedLog {
        fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
            self.0.lock().unwrap().extend_from_slice(buf);
            Ok(buf.len())
        }

        fn flush(&mut self) -> std::io::Result<()> {
            Ok(())
        }
    }

    fn capture_warn_logs<R>(f: impl FnOnce() -> R) -> (R, String) {
        let log_bytes = Arc::new(Mutex::new(Vec::new()));
        let make_writer = {
            let log_bytes = log_bytes.clone();
            move || SharedLog(log_bytes.clone())
        };
        let subscriber = tracing_subscriber::fmt()
            .with_max_level(tracing::Level::WARN)
            .with_ansi(false)
            .with_writer(make_writer)
            .finish();
        let _subscriber_guard = tracing::subscriber::set_default(subscriber);
        let result = f();
        let logs = String::from_utf8(log_bytes.lock().unwrap().clone()).unwrap();
        (result, logs)
    }

    fn priority_cfg(enabled: bool) -> MemoryConfig {
        MemoryConfig {
            memory_scope_priority_injection_enabled: enabled,
            ..Default::default()
        }
    }

    fn fixed_entry(
        scope: MemoryScope,
        memory_type: MemoryType,
        content: impl Into<String>,
        day: u32,
    ) -> MemoryEntry {
        let content = content.into();
        let mut entry = MemoryEntry::new(memory_type, content.clone(), "test", scope);
        entry.id = format!("id{day:02}{:016x}", content.len());
        entry.created_at = chrono::Utc
            .with_ymd_and_hms(2026, 8, day, 12, 0, 0)
            .unwrap();
        entry
    }

    fn save_scope(workspace: &Path, scope: MemoryScope, entries: &[MemoryEntry]) {
        let path = scope_path(workspace, &scope);
        MarkdownMemoryStore::save(&path, entries).unwrap();
    }

    fn long_content(prefix: &str, width: usize) -> String {
        format!("{prefix}_{}", "x".repeat(width))
    }

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
    fn s5_flag_off_matches_pre_s5_golden_on_overflowing_corpus() {
        let _lock = S5_TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let dir = tempfile::tempdir().unwrap();
        let _root = MemoryPaths::set_test_root(dir.path());
        let workspace = dir.path().join("workspace");
        let user_id = "alice";

        let globals: Vec<MemoryEntry> = (0..8)
            .map(|i| {
                fixed_entry(
                    MemoryScope::Global,
                    MemoryType::Preference,
                    long_content(&format!("GLOBAL_LEGACY_{i:02}"), 520),
                    i + 1,
                )
            })
            .collect();
        save_scope(&workspace, MemoryScope::Global, &globals);
        save_scope(
            &workspace,
            MemoryScope::Project,
            &[fixed_entry(
                MemoryScope::Project,
                MemoryType::Preference,
                "PROJECT_WOULD_BE_DROPPED_LEGACY",
                20,
            )],
        );
        save_scope(
            &workspace,
            MemoryScope::User(user_id.to_string()),
            &[fixed_entry(
                MemoryScope::User(user_id.to_string()),
                MemoryType::Preference,
                "USER_WOULD_BE_DROPPED_LEGACY",
                21,
            )],
        );

        // Frozen from `git show 0df6425~1:./crates/naked-core/src/memory/service.rs`:
        // Global → Project → User, type order from `MemoryType::ALL`, stop before
        // the first line that would exceed `MAX_INJECTION_CHARS`.
        let expected = r#"[Memory — active rules]

Global:
  Preferences:
  - GLOBAL_LEGACY_00_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
  - GLOBAL_LEGACY_01_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
  - GLOBAL_LEGACY_02_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
  - GLOBAL_LEGACY_03_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
  - GLOBAL_LEGACY_04_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
  - GLOBAL_LEGACY_05_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
  - GLOBAL_LEGACY_06_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

Project:
  Preferences:
  - PROJECT_WOULD_BE_DROPPED_LEGACY

User (alice):
  Preferences:
  - USER_WOULD_BE_DROPPED_LEGACY
"#;
        let flag_off = MemoryService::load_rules_for_with_config(
            &workspace,
            Some(user_id),
            &priority_cfg(false),
        );

        assert_eq!(flag_off, expected);
        assert!(flag_off.len() <= MAX_INJECTION_CHARS);
        assert!(flag_off.contains("GLOBAL_LEGACY_06"));
        assert!(!flag_off.contains("GLOBAL_LEGACY_07"));
        assert!(flag_off.contains("PROJECT_WOULD_BE_DROPPED_LEGACY"));
        assert!(flag_off.contains("USER_WOULD_BE_DROPPED_LEGACY"));
    }

    #[test]
    fn s5_flag_on_prioritizes_user_project_and_counts_exact_global_drops() {
        let _lock = S5_TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let dir = tempfile::tempdir().unwrap();
        let _root = MemoryPaths::set_test_root(dir.path());
        let workspace = dir.path().join("workspace");
        let user_id = "alice";
        let global_count = 14usize;

        let globals: Vec<MemoryEntry> = (0..global_count)
            .map(|i| {
                fixed_entry(
                    MemoryScope::Global,
                    MemoryType::Preference,
                    long_content(&format!("GLOBAL_OVERFLOW_{i:02}"), 420),
                    (i + 1) as u32,
                )
            })
            .collect();
        save_scope(&workspace, MemoryScope::Global, &globals);
        save_scope(
            &workspace,
            MemoryScope::Project,
            &[fixed_entry(
                MemoryScope::Project,
                MemoryType::Correction,
                "PROJECT_KEEP_CORRECTION",
                20,
            )],
        );
        save_scope(
            &workspace,
            MemoryScope::User(user_id.to_string()),
            &[fixed_entry(
                MemoryScope::User(user_id.to_string()),
                MemoryType::Preference,
                "USER_KEEP_PREFERENCE",
                21,
            )],
        );

        let before = DroppedSnapshot::current();
        let rules = MemoryService::load_rules_for_with_config(
            &workspace,
            Some(user_id),
            &priority_cfg(true),
        );
        let delta = DroppedSnapshot::current().delta_since(before);
        let rendered_globals = (0..global_count)
            .filter(|i| rules.contains(&format!("GLOBAL_OVERFLOW_{i:02}")))
            .count();
        let expected_global_drops = (global_count - rendered_globals) as u64;

        assert!(rules.contains("USER_KEEP_PREFERENCE"));
        assert!(rules.contains("PROJECT_KEEP_CORRECTION"));
        assert!(
            expected_global_drops > 0,
            "fixture must overflow global scope"
        );
        assert_eq!(delta.global, expected_global_drops);
        assert_eq!(delta.project, 0);
        assert_eq!(delta.user, 0);
        assert!(rules.len() <= MAX_INJECTION_CHARS);
    }

    #[test]
    fn s5_flag_on_renders_corrections_before_other_types_and_newest_first() {
        let _lock = S5_TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let dir = tempfile::tempdir().unwrap();
        let _root = MemoryPaths::set_test_root(dir.path());
        let workspace = dir.path().join("workspace");
        save_scope(
            &workspace,
            MemoryScope::Project,
            &[
                fixed_entry(
                    MemoryScope::Project,
                    MemoryType::Preference,
                    "PREFERENCE_NEWER_THAN_CORRECTIONS",
                    25,
                ),
                fixed_entry(
                    MemoryScope::Project,
                    MemoryType::Correction,
                    "CORRECTION_OLDER",
                    10,
                ),
                fixed_entry(
                    MemoryScope::Project,
                    MemoryType::Correction,
                    "CORRECTION_NEWER",
                    20,
                ),
            ],
        );

        let rules =
            MemoryService::load_rules_for_with_config(&workspace, None, &priority_cfg(true));
        let corrections_heading = rules.find("  Corrections:").unwrap();
        let preferences_heading = rules.find("  Preferences:").unwrap();
        let newer = rules.find("CORRECTION_NEWER").unwrap();
        let older = rules.find("CORRECTION_OLDER").unwrap();

        assert!(corrections_heading < preferences_heading);
        assert!(
            newer < older,
            "newer correction must render before older correction"
        );
        assert!(older < preferences_heading);
    }

    #[test]
    fn s5_flag_on_keeps_global_correction_when_project_overflows_live_shape() {
        let _lock = S5_TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let dir = tempfile::tempdir().unwrap();
        let _root = MemoryPaths::set_test_root(dir.path());
        let workspace = dir.path().join("workspace");
        const GLOBAL_CORRECTION: &str = "Never share code or secrets outside the repo.";

        save_scope(
            &workspace,
            MemoryScope::Global,
            &[
                fixed_entry(
                    MemoryScope::Global,
                    MemoryType::Correction,
                    GLOBAL_CORRECTION,
                    1,
                ),
                fixed_entry(
                    MemoryScope::Global,
                    MemoryType::Preference,
                    "GLOBAL_PREFERENCE_CAN_BE_DROPPED_AFTER_PROJECT_OVERFLOW",
                    2,
                ),
            ],
        );
        let project_entries: Vec<MemoryEntry> = (0..48)
            .map(|i| {
                fixed_entry(
                    MemoryScope::Project,
                    MemoryType::Preference,
                    long_content(&format!("PROJECT_LIVE_OVERFLOW_{i:02}"), 180),
                    (i % 28) as u32 + 1,
                )
            })
            .collect();
        save_scope(&workspace, MemoryScope::Project, &project_entries);

        let before = DroppedSnapshot::current();
        let rules =
            MemoryService::load_rules_for_with_config(&workspace, None, &priority_cfg(true));
        let delta = DroppedSnapshot::current().delta_since(before);

        assert!(
            rules.contains(GLOBAL_CORRECTION),
            "global Correction must survive even when Project overflows: {rules}"
        );
        assert!(
            delta.project > 0,
            "live-shaped fixture must drop some project entries"
        );
        assert_eq!(delta.user, 0);
        assert!(rules.len() <= MAX_INJECTION_CHARS);
    }

    #[test]
    fn s5_rendered_block_never_exceeds_budget_in_either_mode() {
        let _lock = S5_TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let dir = tempfile::tempdir().unwrap();
        let _root = MemoryPaths::set_test_root(dir.path());
        let workspace = dir.path().join("workspace");
        let globals: Vec<MemoryEntry> = (0..30)
            .map(|i| {
                fixed_entry(
                    MemoryScope::Global,
                    MemoryType::Preference,
                    long_content(&format!("BUDGET_GLOBAL_{i:02}"), 420),
                    (i % 28) as u32 + 1,
                )
            })
            .collect();
        save_scope(&workspace, MemoryScope::Global, &globals);

        let off = MemoryService::load_rules_for_with_config(&workspace, None, &priority_cfg(false));
        let on = MemoryService::load_rules_for_with_config(&workspace, None, &priority_cfg(true));

        assert!(off.len() <= MAX_INJECTION_CHARS);
        assert!(on.len() <= MAX_INJECTION_CHARS);
    }

    #[test]
    fn s5_small_corpus_identical_and_no_warn_or_counter() {
        let _lock = S5_TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let dir = tempfile::tempdir().unwrap();
        let _root = MemoryPaths::set_test_root(dir.path());
        let workspace = dir.path().join("workspace");
        save_scope(
            &workspace,
            MemoryScope::Global,
            &[fixed_entry(
                MemoryScope::Global,
                MemoryType::Preference,
                "SMALL_GLOBAL_ONLY",
                1,
            )],
        );

        let before = DroppedSnapshot::current();
        let ((legacy, priority), logs) = capture_warn_logs(|| {
            (
                MemoryService::load_rules_for_with_config(&workspace, None, &priority_cfg(false)),
                MemoryService::load_rules_for_with_config(&workspace, None, &priority_cfg(true)),
            )
        });
        let delta = DroppedSnapshot::current().delta_since(before);

        assert_eq!(priority, legacy);
        assert_eq!(delta, DroppedSnapshot::default());
        assert!(!logs.contains("memory injection truncated"));
    }

    #[test]
    fn s5_warns_once_per_load_not_once_per_dropped_entry() {
        let _lock = S5_TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let dir = tempfile::tempdir().unwrap();
        let _root = MemoryPaths::set_test_root(dir.path());
        let workspace = dir.path().join("workspace");
        let globals: Vec<MemoryEntry> = (0..48)
            .map(|i| {
                fixed_entry(
                    MemoryScope::Global,
                    MemoryType::Preference,
                    long_content(&format!("WARN_DROP_{i:02}"), 480),
                    (i % 28) as u32 + 1,
                )
            })
            .collect();
        save_scope(&workspace, MemoryScope::Global, &globals);

        let (rules, logs) = capture_warn_logs(|| {
            MemoryService::load_rules_for_with_config(&workspace, None, &priority_cfg(true))
        });

        assert!(rules.len() <= MAX_INJECTION_CHARS);
        assert_eq!(logs.matches("memory injection truncated").count(), 1);
        assert!(
            logs.contains("global="),
            "warn must include per-scope breakdown: {logs}"
        );
        assert!(
            logs.contains("project=0"),
            "warn must include project scope: {logs}"
        );
        assert!(
            logs.contains("user=0"),
            "warn must include user scope: {logs}"
        );
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
