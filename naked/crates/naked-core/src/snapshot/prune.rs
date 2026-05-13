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
}
