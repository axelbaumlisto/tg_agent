//! Track sessions with in-flight turns. Persisted to disk so
//! after a crash we know which chats had interrupted turns.

use std::collections::HashSet;
use std::path::{Path, PathBuf};
use tokio::sync::RwLock;

pub struct ActiveTurns {
    path: PathBuf,
    active: RwLock<HashSet<String>>,
}

impl ActiveTurns {
    pub fn new(dir: &Path) -> Self {
        let path = dir.join(".active_turns");
        Self {
            path,
            active: RwLock::new(HashSet::new()),
        }
    }

    /// Path to the active turns file (for passing to spawned tasks).
    pub fn path(&self) -> &Path {
        &self.path
    }

    /// Mark a session as having an active turn. Persists to disk.
    pub async fn mark_active(&self, session_id: &str) {
        self.active.write().await.insert(session_id.to_string());
        self.flush().await;
    }

    /// Mark a session as idle (turn finished). Persists to disk.
    pub async fn mark_idle(&self, session_id: &str) {
        self.active.write().await.remove(session_id);
        self.flush().await;
    }

    /// Get sessions that were active when the process last exited.
    /// These are the "dirty" sessions from a crash.
    /// Clears the file after reading (one-shot).
    pub fn load_crashed(&self) -> Vec<String> {
        let content = std::fs::read_to_string(&self.path).unwrap_or_default();
        let _ = std::fs::remove_file(&self.path);
        content
            .lines()
            .filter(|l| !l.is_empty())
            .map(|l| l.to_string())
            .collect()
    }

    async fn flush(&self) {
        let active = self.active.read().await;
        let content: String = active.iter().map(|s| format!("{s}\n")).collect();
        let _ = tokio::fs::write(&self.path, &content).await;
    }
    /// Static: remove a session from the active-turns file without needing &self.
    /// Used from spawned tasks that don't have access to the ActiveTurns instance.
    pub fn remove_from_file(path: &Path, session_id: &str) {
        let content = std::fs::read_to_string(path).unwrap_or_default();
        let filtered: String = content
            .lines()
            .filter(|l| l.trim() != session_id)
            .map(|l| format!("{l}\n"))
            .collect();
        let _ = std::fs::write(path, filtered);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn track_active_and_idle() {
        let dir = tempfile::tempdir().unwrap();
        let tracker = ActiveTurns::new(dir.path());

        tracker.mark_active("sess-1").await;
        tracker.mark_active("sess-2").await;

        // Simulate crash — load without mark_idle
        let tracker2 = ActiveTurns::new(dir.path());
        let crashed = tracker2.load_crashed();
        assert_eq!(crashed.len(), 2);
        assert!(crashed.contains(&"sess-1".to_string()));
        assert!(crashed.contains(&"sess-2".to_string()));

        // After load_crashed, file is cleared
        let crashed2 = tracker2.load_crashed();
        assert!(crashed2.is_empty());
    }

    #[tokio::test]
    async fn idle_removes_from_file() {
        let dir = tempfile::tempdir().unwrap();
        let tracker = ActiveTurns::new(dir.path());

        tracker.mark_active("sess-1").await;
        tracker.mark_active("sess-2").await;
        tracker.mark_idle("sess-1").await;

        let tracker2 = ActiveTurns::new(dir.path());
        let crashed = tracker2.load_crashed();
        assert_eq!(crashed, vec!["sess-2"]);
    }
}
