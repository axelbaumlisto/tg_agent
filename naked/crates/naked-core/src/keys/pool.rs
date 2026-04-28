//! [`KeyPool`] — composes one or more [`KeyProvider`] sources, dedups the
//! returned keys, and serves them via a round-robin cursor with TTL refresh.
//!
//! Designed for the search/scrape engines: each engine holds an `Arc<KeyPool>`
//! for its own provider type and calls [`KeyPool::next`] per request. On
//! `401`/`403` the engine should mark the key as dead via
//! [`KeyPool::mark_dead`] (in-process only — persistent dead-marking is the
//! importer's job) and immediately retry with [`KeyPool::next`].

use std::collections::HashSet;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, RwLock};
use std::time::{Duration, Instant};

use super::KeyProvider;

pub struct KeyPool {
    providers: Vec<Arc<dyn KeyProvider>>,
    key_type: String,
    inner: RwLock<PoolInner>,
    ttl: Duration,
    cursor: AtomicUsize,
}

struct PoolInner {
    keys: Vec<String>,
    dead: HashSet<String>,
    refreshed_at: Instant,
}

impl KeyPool {
    pub fn new(
        providers: Vec<Arc<dyn KeyProvider>>,
        key_type: impl Into<String>,
        ttl: Duration,
    ) -> Self {
        Self {
            providers,
            key_type: key_type.into(),
            inner: RwLock::new(PoolInner {
                keys: Vec::new(),
                dead: HashSet::new(),
                refreshed_at: Instant::now()
                    .checked_sub(Duration::from_secs(86_400))
                    .unwrap_or_else(Instant::now),
            }),
            ttl,
            cursor: AtomicUsize::new(0),
        }
    }

    /// Constructor for the common case of an empty pool (no providers wired).
    pub fn empty(key_type: impl Into<String>) -> Self {
        Self::new(Vec::new(), key_type, Duration::from_secs(3600))
    }

    fn refresh_if_stale(&self) {
        let stale = {
            let g = self.inner.read().unwrap();
            g.refreshed_at.elapsed() > self.ttl || g.keys.is_empty()
        };
        if !stale {
            return;
        }
        let mut all: Vec<String> = Vec::new();
        for p in &self.providers {
            match p.fetch(&self.key_type) {
                Ok(mut k) => all.append(&mut k),
                Err(e) => tracing::warn!(
                    key_type = %self.key_type,
                    "key provider error: {e}"
                ),
            }
        }
        all.sort();
        all.dedup();
        let mut g = self.inner.write().unwrap();
        // Keep the dead-set across refreshes so a transient 401 doesn't
        // re-introduce a known-dead key on the next TTL tick.
        all.retain(|k| !g.dead.contains(k));
        g.keys = all;
        g.refreshed_at = Instant::now();
    }

    /// Round-robin one key out. Returns `None` when the pool is empty (after
    /// refresh) — callers should treat this as "this engine is unusable now".
    pub fn next(&self) -> Option<String> {
        self.refresh_if_stale();
        let g = self.inner.read().unwrap();
        if g.keys.is_empty() {
            return None;
        }
        let i = self.cursor.fetch_add(1, Ordering::Relaxed) % g.keys.len();
        Some(g.keys[i].clone())
    }

    /// Number of currently-live keys.
    pub fn size(&self) -> usize {
        self.refresh_if_stale();
        self.inner.read().unwrap().keys.len()
    }

    /// Mark a key as dead for the lifetime of this process. The next refresh
    /// will skip it; the cursor adjusts naturally on the next call.
    pub fn mark_dead(&self, key: &str) {
        let mut g = self.inner.write().unwrap();
        g.dead.insert(key.to_string());
        g.keys.retain(|k| k != key);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    struct Fake(Vec<String>);
    impl KeyProvider for Fake {
        fn fetch(&self, _: &str) -> Result<Vec<String>, String> {
            Ok(self.0.clone())
        }
    }

    #[test]
    fn empty_pool_returns_none() {
        let p = KeyPool::empty("x");
        assert!(p.next().is_none());
        assert_eq!(p.size(), 0);
    }

    #[test]
    fn round_robin_cycles_through_keys() {
        let pool = KeyPool::new(
            vec![Arc::new(Fake(vec!["A".into(), "B".into(), "C".into()]))],
            "x",
            Duration::from_secs(60),
        );
        let seen: Vec<_> = (0..6).map(|_| pool.next().unwrap()).collect();
        assert_eq!(seen, vec!["A", "B", "C", "A", "B", "C"]);
    }

    #[test]
    fn dedups_across_providers() {
        let pool = KeyPool::new(
            vec![
                Arc::new(Fake(vec!["A".into(), "B".into()])),
                Arc::new(Fake(vec!["B".into(), "C".into()])),
            ],
            "x",
            Duration::from_secs(60),
        );
        assert_eq!(pool.size(), 3);
    }

    #[test]
    fn mark_dead_excludes_key_immediately() {
        let pool = KeyPool::new(
            vec![Arc::new(Fake(vec!["A".into(), "B".into(), "C".into()]))],
            "x",
            Duration::from_secs(60),
        );
        pool.mark_dead("B");
        assert_eq!(pool.size(), 2);
        let seen: Vec<_> = (0..4).map(|_| pool.next().unwrap()).collect();
        assert!(seen.iter().all(|k| k == "A" || k == "C"));
    }
}
