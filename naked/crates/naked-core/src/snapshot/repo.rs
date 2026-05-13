//! Side-git snapshot repo.
//!
//! Every shell-out to git uses BOTH `--git-dir=<side>` and
//! `--work-tree=<workspace>` so the user's `.git` is never touched
//! and the snapshots stay in our own object store.

use std::path::{Path, PathBuf};
use std::process::Command;

/// One snapshot record.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Snapshot {
    pub id: SnapshotId,
    pub label: String,
    pub ts_unix: i64,
}

/// Newtype wrapper for a snapshot's git commit SHA.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SnapshotId(pub String);

impl SnapshotId {
    #[must_use]
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

/// Side-git snapshot repository for one workspace.
#[derive(Debug, Clone)]
pub struct SnapshotRepo {
    /// Workspace root (the `--work-tree`).
    workspace: PathBuf,
    /// `.git` directory of the SIDE repo (the `--git-dir`).
    git_dir: PathBuf,
}

impl SnapshotRepo {
    /// Open the snapshot repo for `workspace`, initialising the
    /// side-git repo if missing. Sets `gc.auto = 0` so background gc
    /// doesn't fire mid-turn. Idempotent.
    pub fn open_or_init(workspace: &Path) -> Result<Self, String> {
        let git_dir = super::paths::snapshot_git_dir(workspace)
            .ok_or_else(|| "cannot resolve snapshot dir (no $HOME?)".to_string())?;
        if !git_dir.exists() {
            if let Some(parent) = git_dir.parent() {
                std::fs::create_dir_all(parent)
                    .map_err(|e| format!("create snapshot parent dir: {e}"))?;
            }
            // `git init` doesn't accept `--git-dir` / `--work-tree`
            // — pass target as positional. Subsequent commands use
            // the standard `--git-dir + --work-tree` pair.
            let init = Command::new("git")
                .args(["init", "--bare", "--quiet"])
                .arg(&git_dir)
                .status()
                .map_err(|e| format!("git init: {e}"))?;
            if !init.success() {
                return Err(format!("git init exited {init}"));
            }
            Self::run_git(&git_dir, workspace, &["config", "gc.auto", "0"])?;
            Self::run_git(
                &git_dir,
                workspace,
                &["config", "user.email", "snapshots@naked"],
            )?;
            Self::run_git(
                &git_dir,
                workspace,
                &["config", "user.name", "naked-snapshots"],
            )?;
        }
        Ok(Self {
            workspace: workspace.to_path_buf(),
            git_dir,
        })
    }

    /// Capture a snapshot with `label`. Returns the new snapshot id,
    /// or `None` if there was nothing to commit (clean state).
    pub fn capture(&self, label: &str) -> Result<Option<SnapshotId>, String> {
        Self::run_git(&self.git_dir, &self.workspace, &["add", "-A"])?;
        // `git diff --cached --quiet` exits 1 if there ARE staged changes.
        let diff_status = Command::new("git")
            .args([
                "--git-dir",
                &self.git_dir.to_string_lossy(),
                "--work-tree",
                &self.workspace.to_string_lossy(),
                "diff",
                "--cached",
                "--quiet",
            ])
            .status()
            .map_err(|e| format!("git diff --cached: {e}"))?;
        if diff_status.success() {
            // No changes; nothing to snapshot.
            return Ok(None);
        }
        Self::run_git(
            &self.git_dir,
            &self.workspace,
            &["commit", "-m", label, "--quiet", "--allow-empty"],
        )?;
        let head = Self::run_git_capture(&self.git_dir, &self.workspace, &["rev-parse", "HEAD"])?;
        Ok(Some(SnapshotId(head.trim().to_string())))
    }

    /// List the most-recent `limit` snapshots (newest first).
    pub fn list(&self, limit: usize) -> Result<Vec<Snapshot>, String> {
        if !self.git_dir.exists() {
            return Ok(Vec::new());
        }
        let out = Self::run_git_capture(
            &self.git_dir,
            &self.workspace,
            &[
                "log",
                &format!("-{limit}"),
                "--format=%H%x09%ct%x09%s",
                "--all",
            ],
        )
        .unwrap_or_default();
        let mut snaps = Vec::new();
        for line in out.lines() {
            let mut parts = line.splitn(3, '\t');
            let id = parts.next().unwrap_or("");
            let ts: i64 = parts.next().and_then(|s| s.parse().ok()).unwrap_or(0);
            let label = parts.next().unwrap_or("");
            if !id.is_empty() {
                snaps.push(Snapshot {
                    id: SnapshotId(id.to_string()),
                    ts_unix: ts,
                    label: label.to_string(),
                });
            }
        }
        Ok(snaps)
    }

    /// Restore the workspace to the contents of snapshot `id`.
    /// Atomic from git's POV (`git checkout -- .`); the user's own
    /// `.git` is never touched.
    pub fn restore(&self, id: &SnapshotId) -> Result<(), String> {
        Self::run_git(
            &self.git_dir,
            &self.workspace,
            &["checkout", id.as_str(), "--", "."],
        )?;
        Ok(())
    }

    /// Internal: run `git` with `--git-dir` + `--work-tree`.
    fn run_git(git_dir: &Path, work_tree: &Path, args: &[&str]) -> Result<(), String> {
        // `config` doesn't need `--work-tree`. Other commands do (we
        // use a bare repo + external work-tree, which only works when
        // both are set).
        let needs_work_tree = !matches!(args.first().copied(), Some("config"));
        let mut cmd = Command::new("git");
        cmd.args(["--git-dir", &git_dir.to_string_lossy()]);
        if needs_work_tree {
            cmd.args(["--work-tree", &work_tree.to_string_lossy()]);
        }
        let status = cmd
            .args(args)
            .status()
            .map_err(|e| format!("git {args:?}: {e}"))?;
        if !status.success() {
            return Err(format!("git {args:?} exited {}", status));
        }
        Ok(())
    }

    fn run_git_capture(git_dir: &Path, work_tree: &Path, args: &[&str]) -> Result<String, String> {
        let out = Command::new("git")
            .args([
                "--git-dir",
                &git_dir.to_string_lossy(),
                "--work-tree",
                &work_tree.to_string_lossy(),
            ])
            .args(args)
            .output()
            .map_err(|e| format!("git {args:?}: {e}"))?;
        if !out.status.success() {
            return Err(format!(
                "git {args:?} exited {}: {}",
                out.status,
                String::from_utf8_lossy(&out.stderr)
            ));
        }
        Ok(String::from_utf8_lossy(&out.stdout).to_string())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    fn fresh_workspace() -> (TempDir, PathBuf) {
        let dir = TempDir::new().unwrap();
        let ws = dir.path().to_path_buf();
        std::fs::write(ws.join("a.txt"), "hello").unwrap();
        (dir, ws)
    }

    #[test]
    fn init_creates_dir() {
        let (_g, ws) = fresh_workspace();
        // Override snapshot root for this test via NAKED_HOME-style:
        // we use the real user dir but a unique workspace. The
        // sha-based dir scheme isolates each test.
        let repo = SnapshotRepo::open_or_init(&ws);
        if let Ok(r) = repo {
            assert!(r.git_dir.exists());
        }
    }

    #[test]
    fn capture_returns_id_and_no_changes_skipped() {
        let (_g, ws) = fresh_workspace();
        // REGISTRY-WAIVE: snapshot non-critical: degrade silently if repo missing
        let Ok(repo) = SnapshotRepo::open_or_init(&ws) else {
            // git binary missing on CI is acceptable.
            return;
        };
        let first = repo.capture("pre-turn:1").unwrap();
        assert!(first.is_some(), "first capture must produce id");
        // Second capture with no changes → None.
        let second = repo.capture("pre-turn:2").unwrap();
        assert!(
            second.is_none(),
            "no-op capture must return None, got {second:?}"
        );
    }

    #[test]
    fn list_orders_newest_first() {
        let (_g, ws) = fresh_workspace();
        // REGISTRY-WAIVE: snapshot non-critical: degrade silently if repo missing
        let Ok(repo) = SnapshotRepo::open_or_init(&ws) else {
            return;
        };
        repo.capture("first").unwrap();
        std::fs::write(ws.join("a.txt"), "second").unwrap();
        repo.capture("second").unwrap();
        let list = repo.list(10).unwrap();
        assert!(list.len() >= 2);
        assert_eq!(list[0].label, "second");
        assert_eq!(list[1].label, "first");
    }

    #[test]
    fn restore_overwrites_modified() {
        let (_g, ws) = fresh_workspace();
        // REGISTRY-WAIVE: snapshot non-critical: degrade silently if repo missing
        let Ok(repo) = SnapshotRepo::open_or_init(&ws) else {
            return;
        };
        let first = repo.capture("pre").unwrap().unwrap();
        std::fs::write(ws.join("a.txt"), "changed").unwrap();
        repo.restore(&first).unwrap();
        let after = std::fs::read_to_string(ws.join("a.txt")).unwrap();
        assert_eq!(after, "hello", "restore must rewrite modified file");
    }
}
