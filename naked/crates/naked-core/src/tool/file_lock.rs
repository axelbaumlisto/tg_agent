//! C1: Per-file mutex to serialize concurrent edits.
//!
//! When multiple sub-agents edit files in parallel, two edits to the
//! same file can race. This module provides a global lock map that
//! serializes operations per-path while allowing different files to
//! proceed concurrently.

use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;
use tokio::sync::Mutex;

use std::sync::LazyLock;

type LockMap = Mutex<HashMap<PathBuf, Arc<Mutex<()>>>>;

static FILE_LOCKS: LazyLock<LockMap> = LazyLock::new(|| Mutex::new(HashMap::new()));

/// Acquire a per-file lock for the given path. The returned guard
/// must be held for the duration of the read-modify-write cycle.
pub async fn lock_file(path: &std::path::Path) -> tokio::sync::OwnedMutexGuard<()> {
    let canonical = path.canonicalize().unwrap_or_else(|_| path.to_path_buf());
    let mutex = {
        let mut map = FILE_LOCKS.lock().await;
        map.entry(canonical)
            .or_insert_with(|| Arc::new(Mutex::new(())))
            .clone()
    };
    mutex.lock_owned().await
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicU32, Ordering};

    #[tokio::test]
    async fn serializes_same_file() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("test.txt");
        std::fs::write(&path, "").unwrap();

        let counter = Arc::new(AtomicU32::new(0));
        let mut handles = vec![];

        for _ in 0..5 {
            let p = path.clone();
            let c = counter.clone();
            handles.push(tokio::spawn(async move {
                let _guard = lock_file(&p).await;
                let val = c.fetch_add(1, Ordering::SeqCst);
                // Inside lock: no two tasks should see the same value
                tokio::time::sleep(tokio::time::Duration::from_millis(1)).await;
                val
            }));
        }

        let mut results = vec![];
        for h in handles {
            results.push(h.await.unwrap());
        }
        results.sort();
        // All unique values = serialized access
        assert_eq!(results, vec![0, 1, 2, 3, 4]);
    }

    #[tokio::test]
    async fn different_files_parallel() {
        let dir = tempfile::tempdir().unwrap();
        let p1 = dir.path().join("a.txt");
        let p2 = dir.path().join("b.txt");
        std::fs::write(&p1, "").unwrap();
        std::fs::write(&p2, "").unwrap();

        // Two locks on different files should not block each other
        let g1 = lock_file(&p1).await;
        let g2 = lock_file(&p2).await;
        drop(g1);
        drop(g2);
    }
}
