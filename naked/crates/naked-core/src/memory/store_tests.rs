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
    assert!(
        age_days < 2,
        "expected refreshed timestamp, got {age_days}d old"
    );
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
    assert_eq!(
        MarkdownMemoryStore::load(&path, MemoryScope::Project).len(),
        2
    );
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
