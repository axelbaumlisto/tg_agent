//! Path helpers for the snapshot side-repo layout.

use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};
use std::path::{Path, PathBuf};

/// Root for all snapshot repos: `~/.naked/snapshots/`.
pub fn snapshots_root() -> Option<PathBuf> {
    std::env::var_os("HOME")
        .map(PathBuf::from)
        .map(|h| h.join(".naked").join("snapshots"))
}

/// Stable directory for one workspace: `~/.naked/snapshots/<u64-hex>/`.
///
/// We hash the canonicalised path with the std DefaultHasher and
/// format the resulting u64 as hex. 8 bytes of entropy is
/// collision-safe for the few thousand workspaces any one host will
/// ever see; using std avoids pulling in `sha2` + `hex` for one
/// directory name.
#[must_use]
pub fn snapshot_dir_for(workspace: &Path) -> Option<PathBuf> {
    let canonical = workspace
        .canonicalize()
        .unwrap_or_else(|_| workspace.to_path_buf());
    let mut hasher = DefaultHasher::new();
    canonical.to_string_lossy().as_bytes().hash(&mut hasher);
    let hash = format!("{:016x}", hasher.finish());
    snapshots_root().map(|r| r.join(hash))
}

/// `.git` dir inside the workspace's snapshot directory.
#[must_use]
pub fn snapshot_git_dir(workspace: &Path) -> Option<PathBuf> {
    snapshot_dir_for(workspace).map(|d| d.join(".git"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn path_stable_across_runs() {
        let p = std::path::PathBuf::from("/tmp/test-naked-stable");
        let _ = std::fs::create_dir_all(&p);
        let a = snapshot_dir_for(&p);
        let b = snapshot_dir_for(&p);
        assert_eq!(a, b);
    }

    #[test]
    fn different_ws_different_hashes() {
        let p1 = std::path::PathBuf::from("/tmp/test-naked-a");
        let p2 = std::path::PathBuf::from("/tmp/test-naked-b");
        let _ = std::fs::create_dir_all(&p1);
        let _ = std::fs::create_dir_all(&p2);
        let a = snapshot_dir_for(&p1).unwrap();
        let b = snapshot_dir_for(&p2).unwrap();
        assert_ne!(a, b);
    }
}
