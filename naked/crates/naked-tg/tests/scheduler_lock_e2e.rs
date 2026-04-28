//! Cross-process end-to-end test for Phase 3.2: scheduler advisory
//! lock.
//!
//! The unit tests in `naked_tg::scheduler_lock` prove that *within
//! a single process* two calls to [`SchedulerLock::try_acquire`]
//! against the same path are mutually exclusive. They do NOT prove
//! that a *separate* process trying to open the same file while
//! we hold the lock will also be blocked. That guarantee is the
//! entire point of the lock (two `naked-tg` services against the
//! same `$NAKED_HOME/research`), so it must be exercised with a
//! real second process.
//!
//! We do this by shelling out to `flock(1)` from util-linux, which
//! uses the same `flock(2)` syscall [`std::fs::File::try_lock`]
//! issues on Linux. That is the real kernel-level contract — if
//! ever a future Rust release changes the backing syscall to a
//! fcntl-based POSIX lock, *this test will start failing*, which
//! is exactly the regression signal we want.
//!
//! Tests silently skip themselves when `flock(1)` is not on
//! `$PATH` (macOS dev boxes, minimal containers).

#![cfg(unix)]

use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};

use naked_tg::scheduler_lock::{LockError, SchedulerLock};
use tempfile::tempdir;

fn have_flock_cli() -> bool {
    Command::new("flock")
        .arg("--version")
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .status()
        .map(|s| s.success())
        .unwrap_or(false)
}

/// Run `flock -n <path> true` and return whether it succeeded.
/// `-n` means non-blocking: exit non-zero immediately if the lock
/// is held. That's exactly the contract we want to verify.
fn flock_nonblocking_succeeds(path: &Path) -> bool {
    let status = Command::new("flock")
        .arg("-n")
        .arg(path)
        .arg("true")
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .status()
        .expect("flock must be spawnable");
    status.success()
}

#[test]
fn another_process_cannot_acquire_while_we_hold_the_lock() {
    if !have_flock_cli() {
        eprintln!("SKIP: flock(1) not on PATH");
        return;
    }
    let dir = tempdir().unwrap();
    let lock = SchedulerLock::try_acquire(dir.path()).expect("parent must acquire");
    let path: PathBuf = lock.path().to_path_buf();

    // 1. While the parent holds the lock, an external process MUST
    //    fail to take it non-blockingly.
    assert!(
        !flock_nonblocking_succeeds(&path),
        "flock(1) -n against {} must fail while parent holds the lock",
        path.display()
    );

    // 2. Drop the parent's lock; now flock(1) should succeed.
    drop(lock);
    assert!(
        flock_nonblocking_succeeds(&path),
        "flock(1) -n against {} must succeed after parent drops",
        path.display()
    );
}

#[test]
fn reacquire_after_external_lock_release() {
    if !have_flock_cli() {
        eprintln!("SKIP: flock(1) not on PATH");
        return;
    }
    let dir = tempdir().unwrap();
    // Create the file via a throwaway first-acquire so flock(1) has
    // something to open.
    let first = SchedulerLock::try_acquire(dir.path()).expect("bootstrap lock file");
    let path: PathBuf = first.path().to_path_buf();
    drop(first);

    // External `flock --no-fork sleep 0.5` holds the lock for
    // half a second. `--no-fork` replaces the `flock` process with
    // `sleep` via `execvp` so the lock-holding FD lives in a
    // single PID — when `sleep` exits, the kernel releases the
    // lock atomically with the `exit(2)`.
    let mut child = Command::new("flock")
        .arg("--no-fork")
        .arg(&path)
        .arg("sleep")
        .arg("0.5")
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .spawn()
        .expect("spawn flock child");

    // Give the child a moment to reach the inner `sleep` with the
    // lock taken. 100ms is plenty for process startup.
    std::thread::sleep(std::time::Duration::from_millis(100));

    // Now our in-process `try_acquire` MUST see `Held`.
    match SchedulerLock::try_acquire(dir.path()) {
        Err(LockError::Held { .. }) => {
            // expected — the external process owns the lock
        }
        Err(other) => panic!("unexpected error: {other}"),
        Ok(_) => {
            panic!("SchedulerLock::try_acquire must fail while external flock(1) holds the lock")
        }
    }

    // Wait for the child to exit naturally (`sleep 0.5`), then
    // verify we can re-acquire.
    child.wait().expect("wait flock child");
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(2);
    loop {
        match SchedulerLock::try_acquire(dir.path()) {
            Ok(_) => break,
            Err(LockError::Held { .. }) if std::time::Instant::now() < deadline => {
                std::thread::sleep(std::time::Duration::from_millis(25));
            }
            Err(e) => panic!("reacquire after child exit must succeed: {e}"),
        }
    }
}
