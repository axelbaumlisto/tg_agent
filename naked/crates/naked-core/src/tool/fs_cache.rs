//! Bounded in-memory cache for small text file reads.
//!
//! PLAN_FAST_BACKEND_v2 Step D: this cache is deliberately scoped to
//! `read_file` / `file_snapshot`. It is never used for grep (fff owns search
//! indexes) and never as the authoritative source for edits.
//!
//! Residual risk: the cache key intentionally uses dev/inode/mtime_ns/len, not a
//! content hash. A filesystem or external writer that preserves all four values
//! can still create a stale hit; authoritative edit/apply paths bypass and
//! invalidate this cache.

use std::collections::{HashMap, VecDeque};
use std::future::Future;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use parking_lot::Mutex;
use tokio::sync::Notify;

#[cfg(unix)]
use std::os::unix::fs::MetadataExt;

use crate::types::{
    FS_CACHE_HIT_COUNT, FS_CACHE_INVALIDATE_COUNT, FS_CACHE_MISS_COUNT,
    FS_CACHE_STALE_BYPASS_COUNT, FS_CACHE_TOO_LARGE_COUNT,
};

pub const DEFAULT_FS_CACHE_MAX_BYTES: u64 = 64 * 1024 * 1024;
pub const MAX_CACHEABLE_TEXT_BYTES: u64 = 256 * 1024;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct FsCacheKey {
    pub dev: u64,
    pub inode: u64,
    pub mtime_ns: i128,
    pub len: u64,
}

impl FsCacheKey {
    #[cfg(unix)]
    pub fn from_metadata(meta: &std::fs::Metadata) -> Self {
        Self {
            dev: meta.dev(),
            inode: meta.ino(),
            mtime_ns: i128::from(meta.mtime()) * 1_000_000_000i128 + i128::from(meta.mtime_nsec()),
            len: meta.len(),
        }
    }
}

#[derive(Debug, Clone)]
pub struct CachedContent {
    content: Arc<str>,
    key: FsCacheKey,
}

impl CachedContent {
    pub fn as_str(&self) -> &str {
        &self.content
    }

    pub fn key(&self) -> FsCacheKey {
        self.key
    }
}

#[derive(Debug)]
struct CacheEntry {
    content: Arc<str>,
    path: PathBuf,
    bytes: u64,
}

#[derive(Debug, Default)]
struct CacheState {
    entries: HashMap<FsCacheKey, CacheEntry>,
    lru: VecDeque<FsCacheKey>,
    in_flight: HashMap<FsCacheKey, Arc<Notify>>,
    total_bytes: u64,
}

#[derive(Debug)]
pub struct FsCache {
    max_bytes: AtomicU64,
    state: Mutex<CacheState>,
}

impl FsCache {
    pub fn new(max_bytes: u64) -> Self {
        Self {
            max_bytes: AtomicU64::new(max_bytes),
            state: Mutex::new(CacheState::default()),
        }
    }

    pub fn set_max_bytes(&self, max_bytes: u64) {
        self.max_bytes.store(max_bytes, Ordering::Relaxed);
        let mut state = self.state.lock();
        self.evict_locked(&mut state);
    }

    pub async fn get_or_load<F, Fut>(
        &self,
        path: &Path,
        loader: F,
    ) -> std::io::Result<CachedContent>
    where
        F: FnOnce() -> Fut,
        Fut: Future<Output = std::io::Result<String>>,
    {
        let canonical_path = canonicalize_for_index(path).await;
        let mut loader = Some(loader);

        loop {
            let metadata_before = tokio::fs::metadata(path).await?;
            let key_before = FsCacheKey::from_metadata(&metadata_before);
            let max_bytes = self.max_bytes.load(Ordering::Relaxed);
            if metadata_before.len() > MAX_CACHEABLE_TEXT_BYTES || metadata_before.len() > max_bytes
            {
                FS_CACHE_TOO_LARGE_COUNT.fetch_add(1, Ordering::Relaxed);
                let Some(loader) = loader.take() else {
                    let content = tokio::fs::read_to_string(path).await?;
                    return Ok(CachedContent {
                        content: Arc::from(content),
                        key: key_before,
                    });
                };
                let content = loader().await?;
                return Ok(CachedContent {
                    content: Arc::from(content),
                    key: key_before,
                });
            }

            let mut wait_for = None;
            {
                let mut state = self.state.lock();
                if let Some(content) = self.hit_locked(&mut state, key_before) {
                    FS_CACHE_HIT_COUNT.fetch_add(1, Ordering::Relaxed);
                    return Ok(CachedContent {
                        content,
                        key: key_before,
                    });
                }
                if let Some(notify) = state.in_flight.get(&key_before) {
                    let mut notified = Box::pin(notify.clone().notified_owned());
                    notified.as_mut().enable();
                    wait_for = Some(notified);
                } else {
                    state.in_flight.insert(key_before, Arc::new(Notify::new()));
                }
            }

            #[cfg(test)]
            if wait_for.is_some() {
                pause_waiter_gap_for_test().await;
            }

            if let Some(notified) = wait_for {
                notified.await;
                continue;
            }

            let Some(loader) = loader.take() else {
                continue;
            };
            FS_CACHE_MISS_COUNT.fetch_add(1, Ordering::Relaxed);
            let loaded = loader().await;
            return self
                .finish_load(path, canonical_path, key_before, loaded)
                .await;
        }
    }

    pub async fn invalidate(&self, path: &Path) {
        let canonical_path = canonicalize_for_index(path).await;
        let mut state = self.state.lock();
        let keys: Vec<FsCacheKey> = state
            .entries
            .iter()
            .filter_map(|(key, entry)| (entry.path == canonical_path).then_some(*key))
            .collect();
        for key in keys {
            if let Some(entry) = state.entries.remove(&key) {
                state.total_bytes = state.total_bytes.saturating_sub(entry.bytes);
                remove_lru_key(&mut state.lru, key);
            }
        }
        FS_CACHE_INVALIDATE_COUNT.fetch_add(1, Ordering::Relaxed);
    }

    #[cfg(test)]
    pub fn entry_count(&self) -> usize {
        self.state.lock().entries.len()
    }

    #[cfg(test)]
    pub fn contains_key(&self, key: FsCacheKey) -> bool {
        self.state.lock().entries.contains_key(&key)
    }

    fn hit_locked(&self, state: &mut CacheState, key: FsCacheKey) -> Option<Arc<str>> {
        let content = state.entries.get(&key)?.content.clone();
        remove_lru_key(&mut state.lru, key);
        state.lru.push_back(key);
        Some(content)
    }

    async fn finish_load(
        &self,
        path: &Path,
        canonical_path: PathBuf,
        key_before: FsCacheKey,
        loaded: std::io::Result<String>,
    ) -> std::io::Result<CachedContent> {
        let content = match loaded {
            Ok(content) => content,
            Err(err) => {
                self.finish_in_flight(key_before);
                return Err(err);
            }
        };

        let metadata_after = match tokio::fs::metadata(path).await {
            Ok(meta) => meta,
            Err(_) => {
                self.finish_in_flight(key_before);
                FS_CACHE_STALE_BYPASS_COUNT.fetch_add(1, Ordering::Relaxed);
                return Ok(CachedContent {
                    content: Arc::from(content),
                    key: key_before,
                });
            }
        };
        let key_after = FsCacheKey::from_metadata(&metadata_after);
        if key_after != key_before || content.as_bytes().contains(&0) {
            self.finish_in_flight(key_before);
            FS_CACHE_STALE_BYPASS_COUNT.fetch_add(1, Ordering::Relaxed);
            return Ok(CachedContent {
                content: Arc::from(content),
                key: key_after,
            });
        }
        let max_bytes = self.max_bytes.load(Ordering::Relaxed);
        if content.len() as u64 > MAX_CACHEABLE_TEXT_BYTES || content.len() as u64 > max_bytes {
            self.finish_in_flight(key_before);
            FS_CACHE_TOO_LARGE_COUNT.fetch_add(1, Ordering::Relaxed);
            return Ok(CachedContent {
                content: Arc::from(content),
                key: key_after,
            });
        }

        let content: Arc<str> = Arc::from(content);
        let bytes = content.len() as u64;
        let mut state = self.state.lock();
        let notify = state.in_flight.remove(&key_before);
        if let Some(entry) = state.entries.remove(&key_before) {
            state.total_bytes = state.total_bytes.saturating_sub(entry.bytes);
            remove_lru_key(&mut state.lru, key_before);
        }
        state.entries.insert(
            key_before,
            CacheEntry {
                content: content.clone(),
                path: canonical_path,
                bytes,
            },
        );
        state.total_bytes = state.total_bytes.saturating_add(bytes);
        state.lru.push_back(key_before);
        self.evict_locked(&mut state);
        if let Some(notify) = notify {
            notify.notify_waiters();
        }
        Ok(CachedContent {
            content,
            key: key_before,
        })
    }

    fn finish_in_flight(&self, key: FsCacheKey) {
        let notify = self.state.lock().in_flight.remove(&key);
        if let Some(notify) = notify {
            notify.notify_waiters();
        }
    }

    fn evict_locked(&self, state: &mut CacheState) {
        let max_bytes = self.max_bytes.load(Ordering::Relaxed);
        while state.total_bytes > max_bytes {
            let Some(key) = state.lru.pop_front() else {
                break;
            };
            if let Some(entry) = state.entries.remove(&key) {
                state.total_bytes = state.total_bytes.saturating_sub(entry.bytes);
            }
        }
    }
}

async fn canonicalize_for_index(path: &Path) -> PathBuf {
    match tokio::fs::canonicalize(path).await {
        Ok(path) => path,
        Err(_) if path.is_absolute() => path.to_path_buf(),
        Err(_) => std::env::current_dir()
            .map(|cwd| cwd.join(path))
            .unwrap_or_else(|_| path.to_path_buf()),
    }
}

fn remove_lru_key(lru: &mut VecDeque<FsCacheKey>, key: FsCacheKey) {
    lru.retain(|candidate| *candidate != key);
}

#[cfg(test)]
#[derive(Clone)]
struct FsCacheWaiterGapHook {
    reached: Arc<Notify>,
    release: Arc<Notify>,
}

#[cfg(test)]
static FS_CACHE_WAITER_GAP_HOOK: std::sync::OnceLock<Mutex<Option<FsCacheWaiterGapHook>>> =
    std::sync::OnceLock::new();

#[cfg(test)]
async fn pause_waiter_gap_for_test() {
    let hook = FS_CACHE_WAITER_GAP_HOOK
        .get_or_init(|| Mutex::new(None))
        .lock()
        .clone();
    if let Some(hook) = hook {
        hook.reached.notify_waiters();
        hook.release.notified().await;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering as AtomicOrdering};
    use std::time::Duration;

    use filetime::{FileTime, set_file_mtime};
    use tokio::sync::Barrier;

    #[tokio::test]
    async fn fs_cache_key_includes_dev_inode_mtime_len() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("a.txt");
        std::fs::write(&path, "one").unwrap();
        let cache = FsCache::new(1024 * 1024);

        let first = cache
            .get_or_load(&path, || async { tokio::fs::read_to_string(&path).await })
            .await
            .unwrap();
        let first_key = first.key();
        let second = cache
            .get_or_load(&path, || async {
                panic!("unchanged file should hit cache")
            })
            .await
            .unwrap();
        assert_eq!(second.key(), first_key);

        let bumped = FileTime::from_unix_time(1_893_456_789, 123_456_789);
        set_file_mtime(&path, bumped).unwrap();
        let mtime_changed = cache
            .get_or_load(&path, || async { tokio::fs::read_to_string(&path).await })
            .await
            .unwrap();
        assert_ne!(mtime_changed.key(), first_key);
        assert_eq!(mtime_changed.key().dev, first_key.dev);
        assert_eq!(mtime_changed.key().inode, first_key.inode);
        assert_eq!(mtime_changed.key().len, first_key.len);
        assert!(mtime_changed.key().dev > 0);
        assert!(mtime_changed.key().inode > 0);

        std::fs::write(&path, "longer").unwrap();
        let len_changed = cache
            .get_or_load(&path, || async { tokio::fs::read_to_string(&path).await })
            .await
            .unwrap();
        assert_ne!(len_changed.key().len, mtime_changed.key().len);
    }

    #[tokio::test]
    async fn fs_cache_hit_avoids_second_load() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("a.txt");
        std::fs::write(&path, "hello").unwrap();
        let cache = FsCache::new(1024 * 1024);
        let calls = Arc::new(AtomicUsize::new(0));

        for _ in 0..2 {
            let calls = calls.clone();
            let path_for_loader = path.clone();
            let content = cache
                .get_or_load(&path, move || async move {
                    calls.fetch_add(1, AtomicOrdering::SeqCst);
                    tokio::fs::read_to_string(path_for_loader).await
                })
                .await
                .unwrap();
            assert_eq!(content.as_str(), "hello");
        }
        assert_eq!(calls.load(AtomicOrdering::SeqCst), 1);
    }

    #[tokio::test]
    async fn fs_cache_external_writer_key_miss_reloads() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("a.txt");
        std::fs::write(&path, "old").unwrap();
        let cache = FsCache::new(1024 * 1024);

        let first = cache
            .get_or_load(&path, || async { tokio::fs::read_to_string(&path).await })
            .await
            .unwrap();
        assert_eq!(first.as_str(), "old");

        std::thread::sleep(Duration::from_millis(5));
        std::fs::write(&path, "new value").unwrap();
        let calls = Arc::new(AtomicUsize::new(0));
        let calls_for_loader = calls.clone();
        let path_for_loader = path.clone();
        let second = cache
            .get_or_load(&path, move || async move {
                calls_for_loader.fetch_add(1, AtomicOrdering::SeqCst);
                tokio::fs::read_to_string(path_for_loader).await
            })
            .await
            .unwrap();
        assert_eq!(second.as_str(), "new value");
        assert_eq!(calls.load(AtomicOrdering::SeqCst), 1);
    }

    #[tokio::test]
    async fn fs_cache_loader_mutates_file_not_cached() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("a.txt");
        std::fs::write(&path, "old").unwrap();
        let cache = FsCache::new(1024 * 1024);
        let calls = Arc::new(AtomicUsize::new(0));

        let calls_for_loader = calls.clone();
        let path_for_loader = path.clone();
        let first = cache
            .get_or_load(&path, move || async move {
                calls_for_loader.fetch_add(1, AtomicOrdering::SeqCst);
                tokio::fs::write(&path_for_loader, "new value").await?;
                Ok("old snapshot".to_string())
            })
            .await
            .unwrap();
        assert_eq!(first.as_str(), "old snapshot");
        assert_eq!(cache.entry_count(), 0);

        let calls_for_loader = calls.clone();
        let path_for_loader = path.clone();
        let second = cache
            .get_or_load(&path, move || async move {
                calls_for_loader.fetch_add(1, AtomicOrdering::SeqCst);
                tokio::fs::read_to_string(path_for_loader).await
            })
            .await
            .unwrap();
        assert_eq!(second.as_str(), "new value");
        assert_eq!(calls.load(AtomicOrdering::SeqCst), 2);
    }

    #[tokio::test]
    async fn fs_cache_evicts_lru_over_budget() {
        let dir = tempfile::tempdir().unwrap();
        let p1 = dir.path().join("one.txt");
        let p2 = dir.path().join("two.txt");
        std::fs::write(&p1, "1111").unwrap();
        std::fs::write(&p2, "2222").unwrap();
        let cache = FsCache::new(6);

        let k1 = cache
            .get_or_load(&p1, || async { tokio::fs::read_to_string(&p1).await })
            .await
            .unwrap()
            .key();
        let k2 = cache
            .get_or_load(&p2, || async { tokio::fs::read_to_string(&p2).await })
            .await
            .unwrap()
            .key();

        assert!(!cache.contains_key(k1));
        assert!(cache.contains_key(k2));
    }

    #[tokio::test]
    async fn fs_cache_too_large_file_not_cached() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("big.txt");
        std::fs::write(&path, "abcdef").unwrap();
        let cache = FsCache::new(4);
        let calls = Arc::new(AtomicUsize::new(0));

        for _ in 0..2 {
            let calls = calls.clone();
            let path_for_loader = path.clone();
            let content = cache
                .get_or_load(&path, move || async move {
                    calls.fetch_add(1, AtomicOrdering::SeqCst);
                    tokio::fs::read_to_string(path_for_loader).await
                })
                .await
                .unwrap();
            assert_eq!(content.as_str(), "abcdef");
        }
        assert_eq!(calls.load(AtomicOrdering::SeqCst), 2);
        assert_eq!(cache.entry_count(), 0);
    }

    #[tokio::test]
    async fn fs_cache_concurrent_same_file_singleflight() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("a.txt");
        std::fs::write(&path, "hello").unwrap();
        let cache = Arc::new(FsCache::new(1024 * 1024));
        let barrier = Arc::new(Barrier::new(8));
        let calls = Arc::new(AtomicUsize::new(0));
        let mut tasks = Vec::new();

        for _ in 0..8 {
            let cache = cache.clone();
            let barrier = barrier.clone();
            let calls = calls.clone();
            let path = path.clone();
            tasks.push(tokio::spawn(async move {
                barrier.wait().await;
                let loader_path = path.clone();
                cache
                    .get_or_load(&path, move || async move {
                        calls.fetch_add(1, AtomicOrdering::SeqCst);
                        tokio::time::sleep(Duration::from_millis(25)).await;
                        tokio::fs::read_to_string(loader_path).await
                    })
                    .await
                    .unwrap()
                    .as_str()
                    .to_string()
            }));
        }

        for task in tasks {
            assert_eq!(task.await.unwrap(), "hello");
        }
        assert_eq!(calls.load(AtomicOrdering::SeqCst), 1);
    }

    #[tokio::test]
    async fn fs_cache_singleflight_no_lost_wakeup() {
        struct HookGuard;
        impl Drop for HookGuard {
            fn drop(&mut self) {
                *FS_CACHE_WAITER_GAP_HOOK
                    .get_or_init(|| Mutex::new(None))
                    .lock() = None;
            }
        }

        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("a.txt");
        std::fs::write(&path, "hello").unwrap();

        let cache = Arc::new(FsCache::new(1024 * 1024));
        let loader_entered = Arc::new(Notify::new());
        let finish_loader = Arc::new(Notify::new());
        let leader_cache = cache.clone();
        let leader_path = path.clone();
        let leader_loader_entered = loader_entered.clone();
        let leader_finish_loader = finish_loader.clone();
        let leader = tokio::spawn(async move {
            leader_cache
                .get_or_load(&leader_path, move || async move {
                    leader_loader_entered.notify_waiters();
                    leader_finish_loader.notified().await;
                    Ok("hello".to_string())
                })
                .await
        });

        tokio::time::timeout(Duration::from_secs(1), loader_entered.notified())
            .await
            .expect("leader loader should be parked with an in-flight entry");

        let hook = FsCacheWaiterGapHook {
            reached: Arc::new(Notify::new()),
            release: Arc::new(Notify::new()),
        };
        *FS_CACHE_WAITER_GAP_HOOK
            .get_or_init(|| Mutex::new(None))
            .lock() = Some(hook.clone());
        let _hook_guard = HookGuard;

        let waiter_cache = cache.clone();
        let waiter_path = path.clone();
        let waiter = tokio::spawn(async move {
            waiter_cache
                .get_or_load(&waiter_path, || async {
                    Err(std::io::Error::other("waiter loader must not run"))
                })
                .await
        });

        tokio::time::timeout(Duration::from_secs(1), hook.reached.notified())
            .await
            .expect("waiter should reach the real get_or_load in-flight branch");

        finish_loader.notify_waiters();
        let leader_value = tokio::time::timeout(Duration::from_secs(1), leader)
            .await
            .expect("leader should complete while waiter is paused in the gap")
            .unwrap()
            .unwrap();
        assert_eq!(leader_value.as_str(), "hello");

        *FS_CACHE_WAITER_GAP_HOOK
            .get_or_init(|| Mutex::new(None))
            .lock() = None;
        hook.release.notify_waiters();
        let waiter_value = tokio::time::timeout(Duration::from_secs(1), waiter)
            .await
            .expect("waiter should observe the leader notification instead of hanging")
            .unwrap()
            .unwrap();
        assert_eq!(waiter_value.as_str(), "hello");
    }
}
