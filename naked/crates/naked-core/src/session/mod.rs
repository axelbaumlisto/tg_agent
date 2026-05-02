pub mod budget;
pub mod jsonl_store;
pub mod store;
pub mod turn_invariants;
pub mod usage;

use std::collections::HashSet;

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

use crate::history::ConversationHistory;

use self::usage::UsageTracker;

/// Tracks which files were accessed during a session.
/// Used by compaction to include `<read-files>` and `<modified-files>`
/// in the summary so context survives across compaction boundaries.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct FileTracker {
    pub read: HashSet<String>,
    pub written: HashSet<String>,
    pub edited: HashSet<String>,
}

impl FileTracker {
    /// Record a file operation from a tool call.
    pub fn record_tool(&mut self, tool_name: &str, input: &serde_json::Value) {
        let path = input
            .get("file_path")
            .or_else(|| input.get("path"))
            .and_then(|v| v.as_str())
            .map(|s| s.to_string());
        match (tool_name, path) {
            ("read_file", Some(p)) => { self.read.insert(p); }
            ("write_file", Some(p)) => { self.written.insert(p); }
            ("edit_file", Some(p)) => { self.edited.insert(p); }
            _ => {}
        }
    }

    /// Files only read (not modified).
    pub fn read_only(&self) -> Vec<&str> {
        self.read
            .iter()
            .filter(|p| !self.written.contains(*p) && !self.edited.contains(*p))
            .map(|s| s.as_str())
            .collect()
    }

    /// All modified files (written + edited).
    pub fn modified(&self) -> Vec<&str> {
        self.written
            .union(&self.edited)
            .map(|s| s.as_str())
            .collect()
    }

    pub fn is_empty(&self) -> bool {
        self.read.is_empty() && self.written.is_empty() && self.edited.is_empty()
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SessionState {
    Active,
    Idle,
    Sleeping,
    Closed,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ForkInfo {
    pub parent_session_id: String,
    pub branch_name: Option<String>,
    pub forked_at: DateTime<Utc>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SessionMetadata {
    pub name: Option<String>,
    pub provider: String,
    pub model: String,
    pub channel: String,
    /// Opaque channel-specific ID for restoring channel→session mapping.
    /// Telegram: "tg:{chat_id}:{thread_id}"
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub channel_id: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SessionSummary {
    pub id: String,
    pub name: Option<String>,
    pub provider: Option<String>,
    pub model: Option<String>,
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
    pub message_count: usize,
    pub state: SessionState,
    /// Total bytes occupied by externalized image artifacts in this session
    /// (sum of file sizes under `<session>/artifacts/img_*.*`). Populated by
    /// `Session::summary_with_root()` when the caller knows the sessions root;
    /// the no-arg `summary()` leaves it `None` because it has no filesystem
    /// context. Useful for surfacing per-session disk usage in `/sessions list`
    /// and for the future `vacuum-sessions` command.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub image_artifacts_total_bytes: Option<u64>,
}

pub struct Session {
    pub id: String,
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
    pub workspace: std::path::PathBuf,
    pub history: ConversationHistory,
    pub usage: UsageTracker,
    pub fork_info: Option<ForkInfo>,
    pub state: SessionState,
    pub metadata: SessionMetadata,
    /// B3: Track files read/written/edited for compaction summaries.
    pub files: FileTracker,
    /// Number of messages last written to disk. Used for incremental save:
    /// if message_count > persisted_len → append-only (fast).
    /// if message_count <= persisted_len → full rewrite (compaction happened).
    pub(crate) persisted_msg_count: usize,
}

impl Session {
    pub fn new(
        workspace: std::path::PathBuf,
        system_prompt: String,
        metadata: SessionMetadata,
    ) -> Self {
        let now = Utc::now();
        Self {
            id: uuid::Uuid::new_v4().to_string(),
            created_at: now,
            updated_at: now,
            workspace,
            history: ConversationHistory::new(system_prompt),
            usage: UsageTracker::default(),
            fork_info: None,
            state: SessionState::Idle,
            metadata,
            files: FileTracker::default(),
            persisted_msg_count: 0,
        }
    }

    pub fn fork(&self, branch_name: Option<String>) -> Self {
        let now = Utc::now();
        Self {
            id: uuid::Uuid::new_v4().to_string(),
            created_at: now,
            updated_at: now,
            workspace: self.workspace.clone(),
            history: self.history.clone(),
            usage: UsageTracker::default(),
            fork_info: Some(ForkInfo {
                parent_session_id: self.id.clone(),
                branch_name,
                forked_at: now,
            }),
            state: SessionState::Idle,
            metadata: self.metadata.clone(),
            files: self.files.clone(),
            persisted_msg_count: 0,
        }
    }

    /// Directory for this session inside the sessions root.
    /// Layout: `{sessions_root}/{id}/`
    pub fn session_dir(&self, sessions_root: &std::path::Path) -> std::path::PathBuf {
        sessions_root.join(&self.id)
    }

    /// Isolated workspace for artifacts created during this dialog.
    /// Layout: `{sessions_root}/{id}/artifacts/`
    pub fn artifacts_dir(&self, sessions_root: &std::path::Path) -> std::path::PathBuf {
        sessions_root.join(&self.id).join("artifacts")
    }

    /// Per-session prompt override (if exists).
    /// Layout: `{sessions_root}/{id}/prompt.md`
    pub fn prompt_path(&self, sessions_root: &std::path::Path) -> std::path::PathBuf {
        sessions_root.join(&self.id).join("prompt.md")
    }

    /// Per-session skills directory.
    /// Layout: `{sessions_root}/{id}/skills/`
    pub fn skills_dir(&self, sessions_root: &std::path::Path) -> std::path::PathBuf {
        sessions_root.join(&self.id).join("skills")
    }

    pub fn summary(&self) -> SessionSummary {
        SessionSummary {
            id: self.id.clone(),
            name: self.metadata.name.clone(),
            provider: Some(self.metadata.provider.clone()),
            model: Some(self.metadata.model.clone()),
            created_at: self.created_at,
            updated_at: self.updated_at,
            message_count: self.history.message_count(),
            state: self.state.clone(),
            image_artifacts_total_bytes: None,
        }
    }

    /// Like [`summary`], but also walks `<sessions_root>/<id>/artifacts/` to
    /// fill in `image_artifacts_total_bytes`. Errors reading the directory
    /// degrade silently to `None` — disk-usage telemetry must never fail a
    /// session listing.
    pub fn summary_with_root(&self, sessions_root: &std::path::Path) -> SessionSummary {
        let mut s = self.summary();
        s.image_artifacts_total_bytes = artifact_dir_size(&self.artifacts_dir(sessions_root));
        s
    }
}

/// Sum the byte size of every `img_*` file in `dir`. Returns `None` if the
/// directory is missing or unreadable. Non-image artifacts (anything not
/// matching the `img_` prefix from `extern_image_blocks`) are ignored — that
/// keeps the metric focused on multimodal-context cost rather than ad-hoc
/// session attachments dropped here by other code paths.
fn artifact_dir_size(dir: &std::path::Path) -> Option<u64> {
    if !dir.is_dir() {
        return Some(0);
    }
    let entries = std::fs::read_dir(dir).ok()?;
    let mut total: u64 = 0;
    for e in entries.flatten() {
        let name = e.file_name();
        let name_s = name.to_string_lossy();
        if !name_s.starts_with("img_") {
            continue;
        }
        if let Ok(meta) = e.metadata() {
            total = total.saturating_add(meta.len());
        }
    }
    Some(total)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_metadata() -> SessionMetadata {
        SessionMetadata {
            name: Some("test".into()),
            provider: "anthropic".into(),
            model: "claude-sonnet-4".into(),
            channel: "cli".into(),
            channel_id: None,
        }
    }

    #[test]
    fn new_session_has_idle_state() {
        let s = Session::new("/tmp".into(), "prompt".into(), test_metadata());
        assert_eq!(s.state, SessionState::Idle);
        assert!(s.fork_info.is_none());
        assert_eq!(s.history.system_prompt(), "prompt");
    }

    #[test]
    fn new_session_has_uuid_id() {
        let s = Session::new("/tmp".into(), "p".into(), test_metadata());
        assert_eq!(s.id.len(), 36); // UUID v4 format
    }

    #[test]
    fn fork_preserves_history() {
        let mut s = Session::new("/tmp".into(), "prompt".into(), test_metadata());
        s.history.push_user("hello");
        let forked = s.fork(Some("branch1".into()));
        assert_ne!(s.id, forked.id);
        assert_eq!(forked.history.message_count(), 1);
        assert_eq!(forked.fork_info.as_ref().unwrap().parent_session_id, s.id);
        assert_eq!(
            forked.fork_info.as_ref().unwrap().branch_name.as_deref(),
            Some("branch1")
        );
        assert_eq!(forked.state, SessionState::Idle);
    }

    #[test]
    fn fork_resets_usage() {
        let mut s = Session::new("/tmp".into(), "p".into(), test_metadata());
        s.usage.record_turn(&crate::types::TurnUsage {
            input_tokens: 100,
            output_tokens: 50,
            ..Default::default()
        });
        let forked = s.fork(None);
        assert_eq!(forked.usage.turn_count, 0);
        assert_eq!(forked.usage.total_input_tokens, 0);
    }

    #[test]
    fn summary_reflects_session() {
        let mut s = Session::new("/tmp".into(), "p".into(), test_metadata());
        s.history.push_user("msg1");
        s.history.push_user("msg2");
        let sum = s.summary();
        assert_eq!(sum.id, s.id);
        assert_eq!(sum.message_count, 2);
        assert_eq!(sum.state, SessionState::Idle);
        assert_eq!(sum.name, Some("test".into()));
    }

    #[test]
    fn summary_with_root_reports_artifact_size() {
        let dir = tempfile::tempdir().unwrap();
        let mut s = Session::new(dir.path().into(), "p".into(), test_metadata());
        s.history.push_user("msg1");
        let artifacts = s.artifacts_dir(dir.path());
        std::fs::create_dir_all(&artifacts).unwrap();
        std::fs::write(artifacts.join("img_aaaaaaaaaaaaaaaa.png"), vec![0u8; 1234]).unwrap();
        std::fs::write(artifacts.join("img_bbbbbbbbbbbbbbbb.jpg"), vec![0u8; 100]).unwrap();
        // Non-img file ignored.
        std::fs::write(artifacts.join("notes.txt"), "hello").unwrap();
        let sum = s.summary_with_root(dir.path());
        assert_eq!(sum.image_artifacts_total_bytes, Some(1334));
    }

    #[test]
    fn summary_with_root_zero_when_no_artifacts() {
        let dir = tempfile::tempdir().unwrap();
        let s = Session::new(dir.path().into(), "p".into(), test_metadata());
        let sum = s.summary_with_root(dir.path());
        assert_eq!(sum.image_artifacts_total_bytes, Some(0));
    }

    #[test]
    fn session_state_serde() {
        let json = serde_json::to_string(&SessionState::Active).unwrap();
        assert_eq!(json, "\"active\"");
        let parsed: SessionState = serde_json::from_str("\"sleeping\"").unwrap();
        assert_eq!(parsed, SessionState::Sleeping);
    }
}

#[cfg(test)]
mod file_tracker_tests {
    use super::*;

    #[test]
    fn record_and_classify() {
        let mut ft = FileTracker::default();
        ft.record_tool("read_file", &serde_json::json!({"file_path": "src/main.rs"}));
        ft.record_tool("edit_file", &serde_json::json!({"file_path": "src/lib.rs"}));
        ft.record_tool("write_file", &serde_json::json!({"file_path": "new.rs"}));
        ft.record_tool("read_file", &serde_json::json!({"file_path": "src/lib.rs"}));
        ft.record_tool("bash", &serde_json::json!({"command": "ls"}));

        assert_eq!(ft.read.len(), 2); // main.rs, lib.rs
        assert_eq!(ft.edited.len(), 1); // lib.rs
        assert_eq!(ft.written.len(), 1); // new.rs

        // read_only excludes modified
        let ro = ft.read_only();
        assert!(ro.contains(&"src/main.rs"));
        assert!(!ro.contains(&"src/lib.rs")); // also edited

        // modified = written + edited
        let modified = ft.modified();
        assert!(modified.contains(&"src/lib.rs"));
        assert!(modified.contains(&"new.rs"));
        assert!(!modified.contains(&"src/main.rs"));
    }

    #[test]
    fn empty_tracker() {
        let ft = FileTracker::default();
        assert!(ft.is_empty());
        assert!(ft.read_only().is_empty());
        assert!(ft.modified().is_empty());
    }

    #[test]
    fn path_field_variants() {
        let mut ft = FileTracker::default();
        // Some tools use "path" instead of "file_path"
        ft.record_tool("read_file", &serde_json::json!({"path": "readme.md"}));
        assert!(ft.read.contains("readme.md"));
    }
}
