//! Turn-level workspace snapshots via git stash.
//!
//! Before each turn that may edit files, the agent stashes uncommitted
//! changes. After the turn, the stash is labeled. Users can `/undo` to
//! pop the last stash, or `/restore N` to apply an older one.
//!
//! Non-git workspaces are silently skipped (no crash, no stash).

use std::path::Path;
use std::process::Stdio;

/// Take a pre-turn snapshot (git stash push).
/// Returns the stash message if successful, None if not a git repo or clean.
pub async fn pre_turn_snapshot(cwd: &Path, turn_seq: u64) -> Option<String> {
    if !is_git_repo(cwd).await {
        return None;
    }
    // Check if there are changes to stash:
    if is_clean(cwd).await {
        return None;
    }
    let msg = format!("naked:pre-turn:{turn_seq}");
    let ok = git_cmd(cwd, &["stash", "push", "-m", &msg, "--include-untracked"])
        .await
        .is_some();
    if ok { Some(msg) } else { None }
}

/// Restore (undo) the last turn's changes.
/// Pops the most recent naked stash.
pub async fn undo_last(cwd: &Path) -> Result<String, String> {
    if !is_git_repo(cwd).await {
        return Err("not a git repository".into());
    }
    // Find the latest naked stash:
    let list = stash_list(cwd).await;
    let entry = list
        .iter()
        .find(|s| s.message.starts_with("naked:pre-turn:"));
    match entry {
        Some(s) => {
            let idx = format!("stash@{{{}}}", s.index);
            match git_cmd(cwd, &["stash", "pop", &idx]).await {
                Some(out) => Ok(format!("Restored: {} ({})", s.message, out.trim())),
                None => Err("git stash pop failed".into()),
            }
        }
        None => Err("no naked snapshots found".into()),
    }
}

/// List all naked snapshots.
pub async fn list_snapshots(cwd: &Path) -> Vec<StashEntry> {
    stash_list(cwd)
        .await
        .into_iter()
        .filter(|s| s.message.starts_with("naked:"))
        .collect()
}

/// A parsed git stash entry.
#[derive(Debug, Clone)]
pub struct StashEntry {
    pub index: usize,
    pub message: String,
}

// ── Internal helpers ──────────────────────────────────────────────

async fn is_git_repo(cwd: &Path) -> bool {
    git_cmd(cwd, &["rev-parse", "--is-inside-work-tree"])
        .await
        .map(|s| s.trim() == "true")
        .unwrap_or(false)
}

async fn is_clean(cwd: &Path) -> bool {
    git_cmd(cwd, &["status", "--porcelain"])
        .await
        .map(|s| s.trim().is_empty())
        .unwrap_or(true)
}

async fn stash_list(cwd: &Path) -> Vec<StashEntry> {
    let output = match git_cmd(cwd, &["stash", "list", "--format=%gd %s"]).await {
        Some(s) => s,
        None => return Vec::new(),
    };
    output
        .lines()
        .filter_map(|line| {
            // Format: "stash@{0} On main: naked:pre-turn:5"
            let (idx_part, rest) = line.split_once(' ')?;
            let idx: usize = idx_part
                .trim_start_matches("stash@{")
                .trim_end_matches('}')
                .parse()
                .ok()?;
            // Message is after "On <branch>: " or just the message
            let msg = rest.split_once(": ").map(|(_, m)| m).unwrap_or(rest);
            Some(StashEntry {
                index: idx,
                message: msg.to_string(),
            })
        })
        .collect()
}

async fn git_cmd(cwd: &Path, args: &[&str]) -> Option<String> {
    let output = tokio::process::Command::new("git")
        .args(args)
        .current_dir(cwd)
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .output()
        .await
        .ok()?;
    if output.status.success() {
        Some(String::from_utf8_lossy(&output.stdout).into_owned())
    } else {
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    async fn init_git_repo() -> TempDir {
        let tmp = TempDir::new().unwrap();
        git_cmd(tmp.path(), &["init"]).await.unwrap();
        git_cmd(tmp.path(), &["config", "user.email", "test@test.com"])
            .await
            .unwrap();
        git_cmd(tmp.path(), &["config", "user.name", "Test"])
            .await
            .unwrap();
        // Initial commit so stash works:
        tokio::fs::write(tmp.path().join("README.md"), "# test\n")
            .await
            .unwrap();
        git_cmd(tmp.path(), &["add", "."]).await.unwrap();
        git_cmd(tmp.path(), &["commit", "-m", "init"])
            .await
            .unwrap();
        tmp
    }

    #[tokio::test]
    async fn snapshot_on_clean_repo_returns_none() {
        let tmp = init_git_repo().await;
        let result = pre_turn_snapshot(tmp.path(), 1).await;
        assert!(result.is_none(), "clean repo should not stash");
    }

    #[tokio::test]
    async fn snapshot_on_dirty_repo_stashes() {
        let tmp = init_git_repo().await;
        tokio::fs::write(tmp.path().join("new.txt"), "dirty\n")
            .await
            .unwrap();
        let result = pre_turn_snapshot(tmp.path(), 42).await;
        assert!(result.is_some());
        assert!(result.unwrap().contains("42"));
        // File should be gone after stash:
        assert!(!tmp.path().join("new.txt").exists());
    }

    #[tokio::test]
    async fn undo_restores_stashed_changes() {
        let tmp = init_git_repo().await;
        tokio::fs::write(tmp.path().join("edit.txt"), "changes\n")
            .await
            .unwrap();
        pre_turn_snapshot(tmp.path(), 1).await;
        assert!(!tmp.path().join("edit.txt").exists());

        let result = undo_last(tmp.path()).await;
        assert!(result.is_ok());
        assert!(tmp.path().join("edit.txt").exists());
    }

    #[tokio::test]
    async fn undo_on_empty_stash_errors() {
        let tmp = init_git_repo().await;
        let result = undo_last(tmp.path()).await;
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("no naked snapshots"));
    }

    #[tokio::test]
    async fn list_snapshots_filters_naked() {
        let tmp = init_git_repo().await;
        // Create two stashes:
        tokio::fs::write(tmp.path().join("a.txt"), "a\n")
            .await
            .unwrap();
        pre_turn_snapshot(tmp.path(), 1).await;
        tokio::fs::write(tmp.path().join("b.txt"), "b\n")
            .await
            .unwrap();
        pre_turn_snapshot(tmp.path(), 2).await;

        let list = list_snapshots(tmp.path()).await;
        assert_eq!(list.len(), 2);
        assert!(list[0].message.contains("pre-turn:2"));
        assert!(list[1].message.contains("pre-turn:1"));
    }

    #[tokio::test]
    async fn non_git_dir_returns_none() {
        let tmp = TempDir::new().unwrap();
        assert!(pre_turn_snapshot(tmp.path(), 1).await.is_none());
        assert!(undo_last(tmp.path()).await.is_err());
    }
}
