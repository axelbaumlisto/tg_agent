//! Snapshot retention.
//!
//! `prune_older_than` walks `~/.naked/snapshots/<hash>/.git` for
//! every workspace and runs `git gc --prune=<duration>` on each.
//! Cheap; safe to call on every bot boot.

use std::path::Path;
use std::process::Command;
use std::time::Duration;

/// Default retention before snapshots are eligible for `git gc`.
pub const DEFAULT_MAX_AGE: Duration = Duration::from_secs(7 * 24 * 60 * 60);

/// Run `git gc --prune=<formatted>` on every snapshot side-repo
/// under `~/.naked/snapshots/`. Returns the number of repos pruned
/// (best-effort; missing/inaccessible repos are silently skipped).
///
/// Also clears any `tmp_pack_*` files left by an interrupted
/// `git pack` operation — those can otherwise accumulate
/// indefinitely.
pub fn prune_older_than(max_age: Duration) -> usize {
    let Some(root) = super::paths::snapshots_root() else {
        return 0;
    };
    // REGISTRY-WAIVE: intentional fallback: missing path → empty result
    let Ok(entries) = std::fs::read_dir(&root) else {
        return 0;
    };
    let prune_arg = format!("--prune={}", format_duration(max_age));
    let mut count = 0;
    for entry in entries.flatten() {
        let git_dir = entry.path().join(".git");
        if !git_dir.exists() {
            continue;
        }
        purge_tmp_pack_files(&git_dir);
        let _ = Command::new("git")
            .args([
                "--git-dir",
                &git_dir.to_string_lossy(),
                "gc",
                &prune_arg,
                "--quiet",
            ])
            .status();
        count += 1;
    }
    count
}

/// B156: default per-workspace cap on the number of snapshot directories.
///
/// Each directory under `~/.naked/snapshots/` is ONE workspace (the name is a
/// hash of the canonicalised path), so this is a cap on how many distinct
/// workspaces keep undo history at once, not a cap per workspace.
pub const DEFAULT_MAX_WORKSPACES: usize = 200;

/// Keep the `max_workspaces` most recently modified snapshot directories and
/// remove the rest. Returns how many were removed.
///
/// Why count-and-mtime rather than age or disk use: the directory name is a
/// one-way hash of the workspace path (`paths::snapshot_dir_for`), so we cannot
/// ask whether the originating workspace still exists. Recency is the only
/// signal available, and a hard count is the only bound that cannot drift.
///
/// Ordering by mtime means an actively used workspace survives regardless of
/// how old it is, while one untouched for months falls off as newer ones
/// arrive. Snapshots are a convenience for undoing a turn, not a backup:
/// losing an old one costs a rollback, not data.
pub fn prune_to_workspace_cap(max_workspaces: usize) -> usize {
    let Some(root) = super::paths::snapshots_root() else {
        return 0;
    };
    prune_dir_to_cap(&root, max_workspaces)
}

/// Cap logic against an explicit root, so it is testable without mocking
/// `$HOME` — the crate forbids `unsafe`, so tests cannot set env vars, and the
/// pre-existing test admitted as much ("We can't easily mock $HOME").
pub(crate) fn prune_dir_to_cap(root: &Path, max_workspaces: usize) -> usize {
    // REGISTRY-WAIVE: intentional fallback: missing path → empty result
    let Ok(entries) = std::fs::read_dir(root) else {
        return 0;
    };

    let mut dirs: Vec<(std::time::SystemTime, std::path::PathBuf)> = entries
        .flatten()
        .filter(|e| e.path().is_dir())
        .filter_map(|e| {
            let modified = e.metadata().and_then(|m| m.modified()).ok()?;
            Some((modified, e.path()))
        })
        .collect();

    if dirs.len() <= max_workspaces {
        return 0;
    }

    // Newest first, then drop everything past the cap.
    dirs.sort_by_key(|(modified, _)| std::cmp::Reverse(*modified));
    let mut removed = 0;
    for (_, path) in dirs.into_iter().skip(max_workspaces) {
        if std::fs::remove_dir_all(&path).is_ok() {
            removed += 1;
        }
    }
    if removed > 0 {
        tracing::info!(
            removed,
            cap = max_workspaces,
            "snapshot retention: removed oldest workspace snapshot dirs"
        );
    }
    removed
}

fn format_duration(d: Duration) -> String {
    // Git's --prune= accepts "<N>.days.ago" and similar; we use a
    // wall-clock UTC timestamp for unambiguous semantics.
    let now = std::time::SystemTime::now();
    let cutoff = now.checked_sub(d).unwrap_or(now);
    let secs = cutoff
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);
    format!("@{{{secs}}}")
}

fn purge_tmp_pack_files(git_dir: &Path) {
    let pack_dir = git_dir.join("objects").join("pack");
    // REGISTRY-WAIVE: intentional fallback: missing path → empty result
    let Ok(entries) = std::fs::read_dir(&pack_dir) else {
        return;
    };
    for entry in entries.flatten() {
        let name = entry.file_name();
        if name.to_string_lossy().starts_with("tmp_pack_") {
            let _ = std::fs::remove_file(entry.path());
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn prune_no_root_returns_zero() {
        // We can't easily mock $HOME; just ensure the function
        // doesn't panic on a missing/empty root.
        let _ = prune_older_than(DEFAULT_MAX_AGE);
    }

    #[test]
    fn format_duration_emits_unix_timestamp() {
        let s = format_duration(Duration::from_secs(0));
        assert!(s.starts_with("@{"));
        assert!(s.ends_with('}'));
    }

    /// B156: the cap keeps the N most recently modified workspaces and drops
    /// the rest. Ordering matters as much as the count — an actively used
    /// workspace must survive regardless of its age.
    #[test]
    fn cap_keeps_newest_and_removes_the_rest() {
        use std::time::{Duration as Dur, SystemTime};
        let tmp = tempfile::TempDir::new().unwrap();
        let root = tmp.path();

        // Five workspaces, oldest first by mtime.
        let names = ["oldest", "older", "middle", "newer", "newest"];
        for (i, n) in names.iter().enumerate() {
            let d = root.join(n);
            std::fs::create_dir_all(d.join(".git")).unwrap();
            let when = SystemTime::now() - Dur::from_secs((names.len() - i) as u64 * 3600);
            filetime::set_file_mtime(&d, filetime::FileTime::from_system_time(when)).unwrap();
        }

        let removed = prune_dir_to_cap(root, 2);
        assert_eq!(removed, 3, "5 workspaces capped at 2 must remove 3");
        assert!(root.join("newest").exists(), "newest must survive");
        assert!(root.join("newer").exists(), "second newest must survive");
        assert!(!root.join("middle").exists());
        assert!(!root.join("older").exists());
        assert!(!root.join("oldest").exists(), "oldest must go first");
    }

    /// Under the cap nothing is touched — retention must not be destructive
    /// on a healthy host.
    #[test]
    fn cap_is_a_noop_when_under_the_limit() {
        let tmp = tempfile::TempDir::new().unwrap();
        for n in ["a", "b"] {
            std::fs::create_dir_all(tmp.path().join(n)).unwrap();
        }
        assert_eq!(prune_dir_to_cap(tmp.path(), 200), 0);
        assert!(tmp.path().join("a").exists());
        assert!(tmp.path().join("b").exists());
    }

    /// A missing root must be silent, not a panic: the directory does not
    /// exist until the first snapshot is taken.
    #[test]
    fn cap_on_missing_root_returns_zero() {
        let tmp = tempfile::TempDir::new().unwrap();
        assert_eq!(prune_dir_to_cap(&tmp.path().join("nope"), 10), 0);
    }
}
