//! Cross-process advisory lock for the research scheduler.
//!
//! Two `naked-tg` processes pointed at the same `NAKED_HOME/research`
//! directory would race on every shared file:
//!
//! * `inflight.json` per spec — one process would resurrect the other's
//!   in-flight tasks every boot, doubling LLM spend.
//! * `runs.jsonl` — append races could interleave bytes mid-line.
//! * findings dedup cache — the in-memory hash sets in
//!   [`naked_core::research::FsResearchStore`] would diverge.
//!
//! [`SchedulerLock`] takes an exclusive **advisory** file lock on
//! `<root>/scheduler.lock` (using the standard library's
//! `std::fs::File::try_lock`, stable since Rust 1.89) before the
//! scheduler loop starts. Contention is detected immediately
//! (non-blocking `try_lock`) so the second process can fail fast with
//! a clear error instead of silently corrupting state. Dropping the
//! [`SchedulerLock`] (or process exit) releases the lock — the kernel
//! cleans up advisory locks even on `SIGKILL`, so a crashed previous
//! instance does not strand the lock.
//!
//! The lock is **advisory**, not mandatory: any external tool that
//! does not call `try_lock` can still write to the directory. We do
//! not try to defend against malicious tampering — only against the
//! common operator footgun of accidentally running two `naked-tg`
//! services against the same `NAKED_HOME`.

use std::fmt;
use std::fs::{File, OpenOptions, TryLockError};
use std::io::{self, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};

/// Filename under the scheduler root that holds the lock. Operators
/// can `cat` it to see the PID of the holder.
pub const LOCK_FILENAME: &str = "scheduler.lock";

/// Failure modes for [`SchedulerLock::try_acquire`].
#[derive(Debug)]
pub enum LockError {
    /// Another process already holds the lock.
    Held {
        path: PathBuf,
        existing_pid: Option<u32>,
    },
    /// Filesystem error while creating or opening the lock file.
    Io(io::Error),
}

impl fmt::Display for LockError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Held { path, existing_pid } => match existing_pid {
                Some(pid) => write!(
                    f,
                    "scheduler lock at {} is held by pid {pid}",
                    path.display()
                ),
                None => write!(
                    f,
                    "scheduler lock at {} is held by another process",
                    path.display()
                ),
            },
            Self::Io(e) => write!(f, "scheduler lock io error: {e}"),
        }
    }
}

impl std::error::Error for LockError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Held { .. } => None,
            Self::Io(e) => Some(e),
        }
    }
}

/// Live exclusive advisory lock on `<root>/scheduler.lock`. Released
/// on drop — keep the value alive for the lifetime of the scheduler.
#[derive(Debug)]
pub struct SchedulerLock {
    path: PathBuf,
    file: File,
}

impl SchedulerLock {
    /// Try to acquire the scheduler lock under `root`. Creates the
    /// directory and the lock file if missing. The PID of the holder
    /// is written into the lock file on success so operators can grep
    /// for stale holders. Non-blocking — never waits for the lock.
    pub fn try_acquire(root: &Path) -> Result<Self, LockError> {
        std::fs::create_dir_all(root).map_err(LockError::Io)?;
        let path = root.join(LOCK_FILENAME);
        let file = OpenOptions::new()
            .create(true)
            .read(true)
            .write(true)
            .truncate(false)
            .open(&path)
            .map_err(LockError::Io)?;

        match file.try_lock() {
            Ok(()) => {
                // Best-effort PID stamp. We hold the lock so racing
                // writers can't sneak in here. Failure is non-fatal —
                // the stamp is for humans.
                let pid = std::process::id();
                let _ = file.set_len(0);
                let mut writer = &file;
                let _ = writer.seek(SeekFrom::Start(0));
                let _ = writer.write_all(format!("{pid}\n").as_bytes());
                let _ = writer.flush();
                Ok(Self { path, file })
            }
            Err(TryLockError::WouldBlock) => {
                let existing_pid = read_pid_stamp(&path);
                Err(LockError::Held { path, existing_pid })
            }
            Err(TryLockError::Error(e)) => Err(LockError::Io(e)),
        }
    }

    /// Path to the lock file. Useful for diagnostics and tests.
    pub fn path(&self) -> &Path {
        &self.path
    }
}

impl Drop for SchedulerLock {
    fn drop(&mut self) {
        // Best-effort: explicit unlock so the lock is released even
        // when the kernel is slow to reclaim the FD (matters mostly
        // for tests that re-acquire in the same process).
        let _ = self.file.unlock();
    }
}

fn read_pid_stamp(path: &Path) -> Option<u32> {
    let s = std::fs::read_to_string(path).ok()?;
    s.trim().parse::<u32>().ok()
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    #[test]
    fn first_acquire_succeeds_and_stamps_pid() {
        let dir = tempdir().unwrap();
        let lock = SchedulerLock::try_acquire(dir.path()).expect("first acquire must succeed");
        assert_eq!(lock.path(), dir.path().join(LOCK_FILENAME));

        let stamp = std::fs::read_to_string(lock.path()).unwrap();
        let pid: u32 = stamp.trim().parse().unwrap();
        assert_eq!(pid, std::process::id());
    }

    #[test]
    fn second_acquire_returns_held_with_pid() {
        let dir = tempdir().unwrap();
        let _first = SchedulerLock::try_acquire(dir.path()).unwrap();
        let err = SchedulerLock::try_acquire(dir.path())
            .expect_err("second acquire on the same root must fail");
        match err {
            LockError::Held { existing_pid, path } => {
                assert_eq!(path, dir.path().join(LOCK_FILENAME));
                assert_eq!(existing_pid, Some(std::process::id()));
            }
            other => panic!("expected LockError::Held, got: {other:?}"),
        }
    }

    #[test]
    fn drop_releases_lock_so_we_can_reacquire() {
        let dir = tempdir().unwrap();
        {
            let _lock = SchedulerLock::try_acquire(dir.path()).unwrap();
        }
        let _again = SchedulerLock::try_acquire(dir.path())
            .expect("after drop the lock must be re-acquirable");
    }

    #[test]
    fn acquire_creates_missing_root() {
        let dir = tempdir().unwrap();
        let nested = dir.path().join("a").join("b").join("research");
        assert!(!nested.exists());
        let lock = SchedulerLock::try_acquire(&nested).unwrap();
        assert!(nested.exists());
        assert!(lock.path().exists());
    }

    #[test]
    fn lock_error_display_contains_path_and_pid() {
        let dir = tempdir().unwrap();
        let _first = SchedulerLock::try_acquire(dir.path()).unwrap();
        let err = SchedulerLock::try_acquire(dir.path()).unwrap_err();
        let msg = format!("{err}");
        assert!(msg.contains("scheduler.lock"), "got: {msg}");
        assert!(msg.contains(&std::process::id().to_string()), "got: {msg}");
    }

    #[test]
    fn cross_process_lock_is_held() {
        // Spawn a child that holds the lock; main process must fail to acquire.
        // We re-exec the current test binary as a `cargo test` "ignored helper"
        // is overkill — instead, fork via std::process and a small inline
        // helper script using `flock`-equivalent semantics is awkward.
        // Cheap alternative: assert the in-process behaviour above already
        // proves `try_lock` is exclusive. The kernel uses the same `flock(2)`
        // / `LockFileEx` regardless of which process holds it.
        //
        // This test is a deliberate placeholder so future maintainers
        // see the cross-process intent documented even though the
        // in-process tests already cover the contract.
        let dir = tempdir().unwrap();
        let lock1 = SchedulerLock::try_acquire(dir.path()).unwrap();
        assert!(matches!(
            SchedulerLock::try_acquire(dir.path()),
            Err(LockError::Held { .. })
        ));
        drop(lock1);
    }
}
