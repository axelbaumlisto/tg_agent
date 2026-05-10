//! Golden tests for the liveness arbiter — DO NOT DELETE.
//!
//! These tests pin the contract that fixes the silent-polling-death
//! failure mode of incident 2026-05-10 12:47 (full diagnosis in
//! `naked/docs/PLAN_LIVENESS_v1.md`):
//!
//!   * `WatchdogTimestamp` MUST stop ticking when the polling loop
//!     stops beating, even if the runtime is otherwise alive.
//!   * The grace window MUST cover boot wiring without false trips.
//!   * Multi-source requirements MUST behave AND-shaped (any-stale
//!     suppresses).
//!   * Shutdown MUST be prompt regardless of arbiter state.
//!
//! Renaming or removing any test here requires a paired removal in
//! `PLAN_LIVENESS_v1.md` § 4 T7. The golden anchor convention follows
//! `loop_golden.rs` (G1..G8) and `scheduler_e2e.rs`.

use std::sync::{
    Arc, Mutex,
    atomic::{AtomicUsize, Ordering},
};
use std::time::Duration;

use async_trait::async_trait;
use tokio::sync::Notify;

use naked_core::liveness::LivenessRegistry;
use naked_tg::watchdog::{LivenessRequirement, WatchdogNotifier, spawn_watchdog_with_liveness};

#[derive(Default, Debug)]
struct CountingNotifier {
    ready: AtomicUsize,
    alive: AtomicUsize,
    stopping: AtomicUsize,
    interval: Mutex<Option<Duration>>,
}

impl CountingNotifier {
    fn with_interval(d: Duration) -> Arc<Self> {
        let n = CountingNotifier::default();
        *n.interval.lock().unwrap() = Some(d);
        Arc::new(n)
    }
}

#[async_trait]
impl WatchdogNotifier for CountingNotifier {
    async fn notify_ready(&self) {
        self.ready.fetch_add(1, Ordering::SeqCst);
    }
    async fn notify_alive(&self) {
        self.alive.fetch_add(1, Ordering::SeqCst);
    }
    async fn notify_stopping(&self) {
        self.stopping.fetch_add(1, Ordering::SeqCst);
    }
    fn interval(&self) -> Option<Duration> {
        *self.interval.lock().unwrap()
    }
}

// ─── L1: healthy polling forwards sd_notify ──────────────────────────

#[tokio::test]
async fn l1_healthy_polling_forwards_pings() {
    let liveness = Arc::new(LivenessRegistry::new());
    let n = CountingNotifier::with_interval(Duration::from_millis(200));
    let shutdown = Arc::new(Notify::new());
    let beater_kill = Arc::new(Notify::new());

    let lc = liveness.clone();
    let bk = beater_kill.clone();
    let beater = tokio::spawn(async move {
        loop {
            tokio::select! {
                _ = bk.notified() => return,
                _ = tokio::time::sleep(Duration::from_millis(30)) => {
                    lc.beat("tg_polling.tick");
                }
            }
        }
    });

    let h = spawn_watchdog_with_liveness(
        n.clone(),
        liveness,
        vec![LivenessRequirement {
            source: "tg_polling.tick",
            max_silence: Duration::from_millis(150),
        }],
        Duration::from_millis(50), // tiny grace
        shutdown.clone(),
    )
    .expect("ticker spawned");

    tokio::time::sleep(Duration::from_millis(500)).await;
    shutdown.notify_one();
    beater_kill.notify_one();
    let _ = h.await;
    let _ = beater.await;

    let alive = n.alive.load(Ordering::SeqCst);
    assert!(
        alive >= 3,
        "L1: healthy polling at 30ms cadence with 100ms ticks should yield ≥ 3 pings, got {alive}"
    );
}

// ─── L2: silent polling stops sd_notify ──────────────────────────────

#[tokio::test]
async fn l2_silent_polling_stops_pings() {
    // Reproduces incident 2026-05-10 12:47:
    //   - polling task wedged on resp.json().await
    //   - watchdog runtime task continues
    //   - PRE-FIX: sd_notify keeps ticking → systemd thinks all is well
    //   - POST-FIX: sd_notify stops within max_silence
    let liveness = Arc::new(LivenessRegistry::new());
    liveness.register("tg_polling.tick"); // never beat
    let n = CountingNotifier::with_interval(Duration::from_millis(200));
    let shutdown = Arc::new(Notify::new());

    let h = spawn_watchdog_with_liveness(
        n.clone(),
        liveness,
        vec![LivenessRequirement {
            source: "tg_polling.tick",
            max_silence: Duration::from_millis(50),
        }],
        Duration::from_millis(20), // grace expires almost immediately
        shutdown.clone(),
    )
    .expect("ticker spawned");

    tokio::time::sleep(Duration::from_millis(500)).await;
    shutdown.notify_one();
    let _ = h.await;

    let alive = n.alive.load(Ordering::SeqCst);
    assert_eq!(
        alive, 0,
        "L2: silent polling MUST suppress ALL sd_notify pings (this is the incident fix), got {alive}"
    );
}

// ─── L3: grace period covers boot wiring ─────────────────────────────

#[tokio::test]
async fn l3_grace_period_covers_boot_wiring() {
    // During boot, sources may not have beaten yet but we still want
    // sd_notify forwarded so systemd doesn't kill us before
    // initialization completes.
    let liveness = Arc::new(LivenessRegistry::new());
    liveness.register("tg_polling.tick"); // never beat — never seen
    let n = CountingNotifier::with_interval(Duration::from_millis(100));
    let shutdown = Arc::new(Notify::new());

    let h = spawn_watchdog_with_liveness(
        n.clone(),
        liveness,
        vec![LivenessRequirement {
            source: "tg_polling.tick",
            max_silence: Duration::from_millis(20),
        }],
        Duration::from_millis(500), // generous grace
        shutdown.clone(),
    )
    .expect("ticker spawned");

    tokio::time::sleep(Duration::from_millis(250)).await;
    shutdown.notify_one();
    let _ = h.await;

    let alive = n.alive.load(Ordering::SeqCst);
    assert!(
        alive >= 2,
        "L3: grace period must allow pings even when sources are stale, got {alive}"
    );
}

// ─── L4: AND-shape across multiple sources ───────────────────────────

#[tokio::test]
async fn l4_any_one_stale_suppresses_all_pings() {
    // Two sources required; only one beats. The arbiter must NOT
    // ping — this is the AND-shape that prevents one healthy
    // subsystem from masking the death of another.
    let liveness = Arc::new(LivenessRegistry::new());
    liveness.register("tg_polling.tick"); // beats below
    liveness.register("scheduler.tick"); // never beats
    let n = CountingNotifier::with_interval(Duration::from_millis(200));
    let shutdown = Arc::new(Notify::new());
    let beater_kill = Arc::new(Notify::new());

    let lc = liveness.clone();
    let bk = beater_kill.clone();
    let beater = tokio::spawn(async move {
        loop {
            tokio::select! {
                _ = bk.notified() => return,
                _ = tokio::time::sleep(Duration::from_millis(20)) => {
                    lc.beat("tg_polling.tick");
                }
            }
        }
    });

    let h = spawn_watchdog_with_liveness(
        n.clone(),
        liveness,
        vec![
            LivenessRequirement {
                source: "tg_polling.tick",
                max_silence: Duration::from_millis(100),
            },
            LivenessRequirement {
                source: "scheduler.tick",
                max_silence: Duration::from_millis(100),
            },
        ],
        Duration::from_millis(20), // negligible grace
        shutdown.clone(),
    )
    .expect("ticker spawned");

    tokio::time::sleep(Duration::from_millis(500)).await;
    shutdown.notify_one();
    beater_kill.notify_one();
    let _ = h.await;
    let _ = beater.await;

    let alive = n.alive.load(Ordering::SeqCst);
    assert_eq!(
        alive, 0,
        "L4: any-stale-source must suppress all pings (AND-shape), got {alive}"
    );
}

// ─── L5: shutdown is prompt regardless of state ──────────────────────

#[tokio::test]
async fn l5_shutdown_is_prompt() {
    // Watchdog interval is long; shutdown must short-circuit it.
    let liveness = Arc::new(LivenessRegistry::new());
    let n = CountingNotifier::with_interval(Duration::from_secs(10));
    let shutdown = Arc::new(Notify::new());

    let h = spawn_watchdog_with_liveness(
        n,
        liveness,
        vec![],
        Duration::from_secs(0),
        shutdown.clone(),
    )
    .expect("ticker spawned");

    tokio::time::sleep(Duration::from_millis(20)).await;
    let start = std::time::Instant::now();
    shutdown.notify_one();
    let _ = h.await;
    let elapsed = start.elapsed();

    assert!(
        elapsed < Duration::from_millis(200),
        "L5: shutdown took {elapsed:?} — must be <200ms regardless of interval"
    );
}
