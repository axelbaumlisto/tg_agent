//! Workspace snapshots — pre/post-turn safety net (T1 of `PLAN_QUALITY_v1.md`).
//!
//! Each turn the agent takes a `pre-turn:<seq>` snapshot of the
//! workspace into a side git repo at
//! `~/.naked/snapshots/<workspace-hash>/.git`, and a matching
//! `post-turn:<seq>` snapshot when the turn finishes. Users can
//! roll back via `/restore N` (slash command) or, when the model
//! recognises an "undo my last edit" intent, the `revert_turn`
//! tool.
//!
//! Why a side repo? The user's own `.git` is never touched.
//! `--git-dir` and `--work-tree` are *always* set together when we
//! shell out to git; that single invariant keeps snapshots and the
//! user's repo completely independent. Workspaces without `.git`
//! still get snapshots.
//!
//! Failure model: pre/post-turn snapshot calls are **non-fatal**.
//! If `git` is missing, the disk is full, or the workspace is
//! read-only, the turn proceeds and the engine logs a warning.
//! The snapshot is a safety net, not a correctness gate.

pub mod paths;
pub mod prune;
pub mod repo;

// Legacy git-stash based snapshot path (pre-T1). Kept for the
// existing `pre_turn_snapshot` / `undo_last` / `/restore` callers in
// session_ops::turn and naked-tg::commands::ops. The new T1
// SnapshotRepo (in `repo.rs`) is the side-git replacement and will
// take over once both wiring sites are migrated.
mod legacy_stash;
pub use legacy_stash::{StashEntry, list_snapshots, pre_turn_snapshot, undo_last};

pub use paths::{snapshot_dir_for, snapshot_git_dir};
pub use prune::{DEFAULT_MAX_AGE, prune_older_than};
pub use repo::{Snapshot, SnapshotId, SnapshotRepo};
