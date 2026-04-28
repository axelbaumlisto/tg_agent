use std::fmt;

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MemoryType {
    Preference,
    Correction,
    ProjectKnowledge,
    Failure,
}

impl MemoryType {
    pub fn section_heading(&self) -> &'static str {
        match self {
            Self::Preference => "Preferences",
            Self::Correction => "Corrections",
            Self::ProjectKnowledge => "Project Knowledge",
            Self::Failure => "Failures",
        }
    }

    pub fn from_heading(heading: &str) -> Option<Self> {
        match heading.trim().to_ascii_lowercase().as_str() {
            "preferences" => Some(Self::Preference),
            "corrections" => Some(Self::Correction),
            "project knowledge" => Some(Self::ProjectKnowledge),
            "failures" => Some(Self::Failure),
            _ => None,
        }
    }

    pub const ALL: &'static [MemoryType] = &[
        Self::Preference,
        Self::Correction,
        Self::ProjectKnowledge,
        Self::Failure,
    ];
}

impl fmt::Display for MemoryType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Preference => write!(f, "preference"),
            Self::Correction => write!(f, "correction"),
            Self::ProjectKnowledge => write!(f, "project_knowledge"),
            Self::Failure => write!(f, "failure"),
        }
    }
}

impl std::str::FromStr for MemoryType {
    type Err = String;
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_ascii_lowercase().as_str() {
            "preference" => Ok(Self::Preference),
            "correction" => Ok(Self::Correction),
            "project_knowledge" => Ok(Self::ProjectKnowledge),
            "failure" => Ok(Self::Failure),
            _ => Err(format!("unknown memory type: {s}")),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MemoryScope {
    Project,
    Global,
    /// Per-user scope. Identifier is typically a Telegram `user.id` (stringified)
    /// or a CLI `$USER`. Stored at `~/.naked/users/{sanitized_id}/memory/MEMORY.md`.
    User(String),
}

impl fmt::Display for MemoryScope {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Project => write!(f, "project"),
            Self::Global => write!(f, "global"),
            Self::User(id) => write!(f, "user:{id}"),
        }
    }
}

impl std::str::FromStr for MemoryScope {
    type Err = String;
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let lower = s.to_ascii_lowercase();
        match lower.as_str() {
            "project" => Ok(Self::Project),
            "global" => Ok(Self::Global),
            _ => {
                if let Some(id) = lower.strip_prefix("user:") {
                    let id = id.trim();
                    if id.is_empty() {
                        Err("user scope requires an id: 'user:<id>'".to_string())
                    } else {
                        Ok(Self::User(id.to_string()))
                    }
                } else {
                    Err(format!("unknown scope: {s}"))
                }
            }
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryEntry {
    pub id: String,
    pub memory_type: MemoryType,
    pub content: String,
    pub created_at: DateTime<Utc>,
    pub source: String,
    pub scope: MemoryScope,
    /// How many times the entry has been recalled (matched against the
    /// system prompt or returned by `memory.search`). Persisted so the
    /// counter survives restarts; the digest uses it for promotion and
    /// for the TTL/forgetting sweep. Default: 0.
    #[serde(default)]
    pub recall_count: u32,
    /// Last UTC date the entry was recalled. Used by the forgetting
    /// sweep to spare entries that are old but still in active use.
    /// Default: `None` (never recalled).
    #[serde(default)]
    pub last_recalled_at: Option<DateTime<Utc>>,
}

impl MemoryEntry {
    pub fn new(memory_type: MemoryType, content: String, source: &str, scope: MemoryScope) -> Self {
        let id = uuid::Uuid::new_v4().to_string()[..8].to_string();
        Self {
            id,
            memory_type,
            content,
            created_at: Utc::now(),
            source: source.to_string(),
            scope,
            recall_count: 0,
            last_recalled_at: None,
        }
    }

    /// Format as a markdown list item with metadata comment.
    ///
    /// `recall:N` and `last_recall:YYYY-MM-DD` are emitted only when
    /// non-default so old MEMORY.md files stay byte-identical until
    /// the entry actually gets recalled at least once.
    pub fn to_markdown_line(&self) -> String {
        let date = self.created_at.format("%Y-%m-%d");
        let mut tail = String::new();
        if self.recall_count > 0 {
            tail.push_str(&format!(" recall:{}", self.recall_count));
        }
        if let Some(when) = self.last_recalled_at {
            tail.push_str(&format!(" last_recall:{}", when.format("%Y-%m-%d")));
        }
        format!(
            "- {} <!-- id:{} created:{} source:{}{} -->",
            self.content, self.id, date, self.source, tail,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn memory_type_roundtrip() {
        for ty in MemoryType::ALL {
            let s = ty.to_string();
            let parsed: MemoryType = s.parse().unwrap();
            assert_eq!(*ty, parsed);
        }
    }

    #[test]
    fn memory_type_from_heading() {
        assert_eq!(
            MemoryType::from_heading("Preferences"),
            Some(MemoryType::Preference)
        );
        assert_eq!(
            MemoryType::from_heading("Project Knowledge"),
            Some(MemoryType::ProjectKnowledge)
        );
        assert_eq!(MemoryType::from_heading("Unknown"), None);
    }

    #[test]
    fn memory_scope_roundtrip() {
        let scopes = [
            MemoryScope::Project,
            MemoryScope::Global,
            MemoryScope::User("12345".into()),
        ];
        for s in &scopes {
            let text = s.to_string();
            let parsed: MemoryScope = text.parse().unwrap();
            assert_eq!(*s, parsed);
        }
    }

    #[test]
    fn memory_scope_user_empty_id_rejected() {
        let err = "user:".parse::<MemoryScope>().unwrap_err();
        assert!(err.contains("requires an id"));
    }

    #[test]
    fn memory_scope_display_user() {
        assert_eq!(MemoryScope::User("alice".into()).to_string(), "user:alice");
    }

    #[test]
    fn memory_entry_markdown_line() {
        let entry = MemoryEntry {
            id: "abc123".into(),
            memory_type: MemoryType::Preference,
            content: "Use tabs".into(),
            created_at: chrono::DateTime::parse_from_rfc3339("2026-04-08T00:00:00Z")
                .unwrap()
                .with_timezone(&Utc),
            source: "user".into(),
            scope: MemoryScope::Project,
            recall_count: 0,
            last_recalled_at: None,
        };
        let line = entry.to_markdown_line();
        assert!(line.starts_with("- Use tabs"));
        assert!(line.contains("id:abc123"));
        assert!(line.contains("created:2026-04-08"));
        assert!(line.contains("source:user"));
        // recall fields stay absent when defaulted, preserving
        // byte-compat with pre-recall MEMORY.md files.
        assert!(!line.contains("recall:"));
        assert!(!line.contains("last_recall:"));
    }

    #[test]
    fn memory_entry_markdown_line_includes_recall_when_set() {
        let mut entry = MemoryEntry::new(
            MemoryType::Preference,
            "Use tabs".into(),
            "user",
            MemoryScope::Project,
        );
        entry.id = "abc123".into();
        entry.recall_count = 3;
        entry.last_recalled_at = Some(
            chrono::DateTime::parse_from_rfc3339("2026-04-21T12:00:00Z")
                .unwrap()
                .with_timezone(&Utc),
        );
        let line = entry.to_markdown_line();
        assert!(line.contains("recall:3"));
        assert!(line.contains("last_recall:2026-04-21"));
    }

    #[test]
    fn memory_entry_new_generates_id() {
        let e = MemoryEntry::new(
            MemoryType::Correction,
            "test content".into(),
            "auto",
            MemoryScope::Global,
        );
        assert_eq!(e.id.len(), 8);
        assert_eq!(e.source, "auto");
        assert_eq!(e.scope, MemoryScope::Global);
    }
}
