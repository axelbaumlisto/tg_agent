//! Per-source heartbeat registry — the liveness contract.
//!
//! Solves the «watchdog врёт» class of bugs (incident 2026-05-10 12:47):
//! systemd's `WatchdogTimestamp` only confirms the process exists, not
//! that it does meaningful work. A polling loop stuck on `resp.json().await`
//! looks alive to the kernel but is functionally dead.
//!
//! `LivenessRegistry` provides per-named-source timestamps; a downstream
//! arbiter (in `naked-tg::watchdog`) decides whether to forward
//! sd_notify pings based on freshness of *every* required source. If any
//! lags beyond its budget, the arbiter stops pinging — systemd's
//! `WatchdogSec=30s` then prunes the process and `Restart=always`
//! brings it back.
//!
//! Plan: naked/docs/PLAN_LIVENESS_v1.md §4 T1.

use std::collections::HashMap;
use std::sync::RwLock;
use std::sync::atomic::{AtomicI64, Ordering};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

/// Per-source last-seen registry.
///
/// Cheap to clone via `Arc<LivenessRegistry>`. Reads (`stale_sources`,
/// `last_beat`) take a short shared lock; writes (`beat` for a known
/// source) only do an atomic store while holding the read lock; first
/// `beat` for a new source takes the write lock to insert.
///
/// All sources are keyed by `&'static str` because heartbeat names are
/// always compile-time literals (`"tg_polling.tick"`, `"scheduler.tick"`,
/// …). This keeps the map allocation-free and lookup branch-free.
#[derive(Debug, Default)]
pub struct LivenessRegistry {
    /// `name → last beat time as Unix epoch ms`.
    /// `0` means "never beat yet" (post-`register`, pre-`beat`).
    sources: RwLock<HashMap<&'static str, AtomicI64>>,
}

impl LivenessRegistry {
    /// New empty registry.
    pub fn new() -> Self {
        Self::default()
    }

    /// Pre-register a source so it shows up in `last_beat` even before
    /// its first `beat`. Idempotent — re-registering the same name is
    /// a no-op (existing AtomicI64 is preserved).
    ///
    /// Useful for arbiters that want to assert "every required source
    /// has been seen at least once" during a startup grace period.
    pub fn register(&self, source: &'static str) {
        let mut g = crate::write_or_recover(&self.sources);
        g.entry(source).or_insert_with(|| AtomicI64::new(0));
    }

    /// Record a heartbeat from `source`. Lock-free for the hot path
    /// (after the first call) — only an atomic store under a shared
    /// read lock.
    pub fn beat(&self, source: &'static str) {
        let now = now_ms();
        // Fast path: known source.
        {
            let g = crate::read_or_recover(&self.sources);
            if let Some(slot) = g.get(source) {
                slot.store(now, Ordering::Relaxed);
                return;
            }
        }
        // Slow path: first beat for this name; insert under write lock.
        let mut g = crate::write_or_recover(&self.sources);
        g.entry(source)
            .or_insert_with(|| AtomicI64::new(now))
            .store(now, Ordering::Relaxed);
    }

    /// How long ago `source` last beat. `None` if it was never registered
    /// or never beaten. `Some(very_large)` if registered but never beaten
    /// (the slot value is `0` ms since epoch — i.e. 1970).
    pub fn last_beat(&self, source: &'static str) -> Option<Duration> {
        let g = crate::read_or_recover(&self.sources);
        let last = g.get(source)?.load(Ordering::Relaxed);
        if last == 0 {
            return Some(Duration::from_secs(u64::MAX / 2));
        }
        let delta = now_ms().saturating_sub(last);
        Some(Duration::from_millis(delta as u64))
    }

    /// Names whose last beat is older than `max_silence`. Each entry
    /// is `(name, age)` so callers can render diagnostics without a
    /// second call.
    ///
    /// Sources that were `register`ed but never `beat`en count as stale
    /// (their slot is `0` since epoch, which is always older than any
    /// finite `max_silence`).
    pub fn stale_sources(&self, max_silence: Duration) -> Vec<(&'static str, Duration)> {
        let now = now_ms();
        let cutoff_ms = now.saturating_sub(max_silence.as_millis() as i64);
        let g = crate::read_or_recover(&self.sources);
        let mut out: Vec<(&'static str, Duration)> = g
            .iter()
            .filter_map(|(name, slot)| {
                let last = slot.load(Ordering::Relaxed);
                if last < cutoff_ms {
                    let age = if last == 0 {
                        Duration::from_secs(u64::MAX / 2)
                    } else {
                        Duration::from_millis(now.saturating_sub(last) as u64)
                    };
                    Some((*name, age))
                } else {
                    None
                }
            })
            .collect();
        // Stable order so log lines + tests are deterministic.
        out.sort_by_key(|(n, _)| *n);
        out
    }

    /// Snapshot of every known source for diagnostics. `(name, age)`
    /// where `age == None` means "registered but never beaten".
    /// Sorted by name.
    pub fn snapshot(&self) -> Vec<(&'static str, Option<Duration>)> {
        let now = now_ms();
        let g = crate::read_or_recover(&self.sources);
        let mut out: Vec<_> = g
            .iter()
            .map(|(name, slot)| {
                let last = slot.load(Ordering::Relaxed);
                let age = if last == 0 {
                    None
                } else {
                    Some(Duration::from_millis(now.saturating_sub(last) as u64))
                };
                (*name, age)
            })
            .collect();
        out.sort_by_key(|(n, _)| *n);
        out
    }
}

#[inline]
fn now_ms() -> i64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_millis() as i64)
        .unwrap_or(0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;
    use std::thread;

    #[test]
    fn empty_registry_has_no_stale() {
        let r = LivenessRegistry::new();
        assert!(r.stale_sources(Duration::from_millis(1)).is_empty());
        assert_eq!(r.last_beat("anything"), None);
    }

    #[test]
    fn beat_then_immediately_fresh() {
        let r = LivenessRegistry::new();
        r.beat("polling");
        // Just-beaten source is not in stale(d) for any d > 0.
        assert!(
            r.stale_sources(Duration::from_secs(60)).is_empty(),
            "freshly-beaten source should not be stale"
        );
        let age = r.last_beat("polling").expect("should have beat");
        assert!(age < Duration::from_secs(1));
    }

    #[test]
    fn stale_after_sleep_smaller_than_threshold() {
        let r = LivenessRegistry::new();
        r.beat("source");
        std::thread::sleep(Duration::from_millis(50));
        // 1ms threshold → 50ms-old beat IS stale.
        let stale = r.stale_sources(Duration::from_millis(1));
        assert_eq!(stale.len(), 1);
        assert_eq!(stale[0].0, "source");
        assert!(stale[0].1 >= Duration::from_millis(50));
    }

    #[test]
    fn registered_but_never_beat_is_always_stale() {
        let r = LivenessRegistry::new();
        r.register("scheduler");
        let stale = r.stale_sources(Duration::from_secs(1));
        assert_eq!(stale.len(), 1);
        assert_eq!(stale[0].0, "scheduler");
        assert_eq!(
            r.last_beat("scheduler"),
            Some(Duration::from_secs(u64::MAX / 2))
        );
    }

    #[test]
    fn beat_overrides_register() {
        let r = LivenessRegistry::new();
        r.register("polling");
        r.beat("polling");
        assert!(r.stale_sources(Duration::from_secs(60)).is_empty());
    }

    #[test]
    fn multiple_sources_sorted() {
        let r = LivenessRegistry::new();
        r.beat("zeta");
        r.beat("alpha");
        r.beat("mid");
        std::thread::sleep(Duration::from_millis(20));
        let stale = r.stale_sources(Duration::from_millis(1));
        let names: Vec<_> = stale.iter().map(|(n, _)| *n).collect();
        assert_eq!(names, vec!["alpha", "mid", "zeta"]);
    }

    #[test]
    fn snapshot_distinguishes_registered_from_beaten() {
        let r = LivenessRegistry::new();
        r.register("never");
        r.beat("seen");
        let snap = r.snapshot();
        assert_eq!(snap.len(), 2);
        // sorted: "never" < "seen"
        assert_eq!(snap[0].0, "never");
        assert_eq!(snap[0].1, None);
        assert_eq!(snap[1].0, "seen");
        assert!(snap[1].1.is_some());
    }

    #[test]
    fn concurrent_beats_no_corruption() {
        let r = Arc::new(LivenessRegistry::new());
        let mut handles = Vec::new();
        for i in 0..16 {
            let r = Arc::clone(&r);
            handles.push(thread::spawn(move || {
                let names: &[&'static str] = &["a", "b", "c", "d"];
                for _ in 0..1000 {
                    r.beat(names[i % names.len()]);
                }
            }));
        }
        for h in handles {
            h.join().unwrap();
        }
        // All 4 names should be present and fresh.
        assert_eq!(r.snapshot().len(), 4);
        assert!(r.stale_sources(Duration::from_secs(60)).is_empty());
    }

    #[test]
    fn re_register_is_idempotent() {
        let r = LivenessRegistry::new();
        r.beat("polling");
        let first_beat = r.last_beat("polling").unwrap();
        // re-register must NOT reset the timestamp
        r.register("polling");
        let after = r.last_beat("polling").unwrap();
        // After should be >= first_beat (only forward time)
        assert!(after >= first_beat);
        // and the source is still fresh
        assert!(r.stale_sources(Duration::from_secs(60)).is_empty());
    }
}

#[cfg(test)]
mod proptests {
    //! Property-based tests for [`LivenessRegistry`] (T8 of
    //! PLAN_LIVENESS_v1).
    //!
    //! Properties:
    //!   1. **Just-beaten source is fresh**: `beat(s); stale_sources(d>1ms)`
    //!      never contains `s`.
    //!   2. **Stale-set monotone in d**: `stale_sources(d1) ⊆ stale_sources(d2)`
    //!      whenever `d1 ≥ d2`.
    //!   3. **Snapshot covers everyone**: every `register`-ed and every
    //!      `beat`-en name appears in `snapshot()` exactly once.

    use super::*;
    use proptest::prelude::*;

    /// 5–8 names from a small fixed pool, with possible repeats.
    fn arb_names() -> impl Strategy<Value = Vec<&'static str>> {
        let pool: &[&'static str] = &[
            "alpha", "bravo", "charlie", "delta", "echo", "foxtrot", "golf", "hotel",
        ];
        proptest::sample::select(pool.to_vec()).prop_flat_map(move |_| {
            proptest::collection::vec(proptest::sample::select(pool.to_vec()), 1..6usize)
        })
    }

    proptest! {
        #![proptest_config(ProptestConfig {
            cases: 200,
            ..ProptestConfig::default()
        })]

        #[test]
        fn just_beaten_is_fresh(name in proptest::sample::select(
            vec!["alpha", "bravo", "charlie", "delta", "echo"]
        )) {
            let r = LivenessRegistry::new();
            r.beat(name);
            // Wait at most 0; even d=10s wouldn't make a just-beaten
            // source stale.
            let stale = r.stale_sources(Duration::from_secs(10));
            prop_assert!(
                !stale.iter().any(|(n, _)| *n == name),
                "just-beaten {name} appeared in stale list: {stale:?}"
            );
        }

        #[test]
        fn stale_monotone_in_threshold(
            beats in arb_names(),
            d_ms in 0u64..2000,
        ) {
            let r = LivenessRegistry::new();
            for n in &beats {
                r.beat(n);
            }
            // Wait so that d_ms can credibly mark some as stale.
            std::thread::sleep(Duration::from_millis(50));
            let small = r.stale_sources(Duration::from_millis(d_ms));
            let large = r.stale_sources(Duration::from_millis(d_ms.saturating_add(50)));
            // Anything stale at the larger threshold must also be stale
            // at the smaller threshold (older-than-d2 is stricter when d2 > d1).
            // Wait — actually the inverse: stale(d) returns sources OLDER than d.
            // For d_large > d_small → stale(d_large) ⊆ stale(d_small).
            for (name, _) in &large {
                prop_assert!(
                    small.iter().any(|(n, _)| n == name),
                    "{} stale at d={}ms but not at d={}ms",
                    name,
                    d_ms.saturating_add(50),
                    d_ms,
                );
            }
        }

        #[test]
        fn snapshot_includes_all(
            registered in arb_names(),
            beaten in arb_names(),
        ) {
            let r = LivenessRegistry::new();
            for n in &registered { r.register(n); }
            for n in &beaten { r.beat(n); }
            let snap = r.snapshot();
            // every unique name in registered ∪ beaten must be in snapshot
            let mut expected: std::collections::HashSet<&str> =
                registered.iter().copied().collect();
            for n in &beaten { expected.insert(n); }
            prop_assert_eq!(
                snap.len(),
                expected.len(),
                "snapshot count mismatch: snap={:?} expected={:?}",
                snap,
                expected,
            );
            for (n, _) in &snap {
                prop_assert!(expected.contains(n), "unexpected name {n} in snapshot");
            }
        }
    }
}
