//! End-to-end test for Phase 3.6: systemd watchdog full lifecycle.
//!
//! The unit tests in `naked_tg::watchdog::tests` exercise the trait
//! contract and each `notify_*` method in isolation. What they do
//! NOT cover is the *composition* of:
//!
//!   1. a real `SystemdWatchdog` pointed at a live `UnixDatagram`,
//!   2. the `spawn_watchdog_ticks` coordinator that calls
//!      `notify_alive` on a half-interval timer,
//!   3. the `Notify`-driven shutdown that `main.rs` uses to stop
//!      the ticker gracefully.
//!
//! This test wires all three exactly as `main.rs` does and proves
//! that the full sequence `READY=1` → repeated `WATCHDOG=1` →
//! `STOPPING=1` arrives on the wire in the expected order. A
//! regression that silently disables any one of those three would
//! put the bot into a state where systemd's `WatchdogSec=` would
//! kill and restart us periodically — exactly the silent wedge
//! the E2E is here to catch.

#![cfg(unix)]

use std::os::unix::net::UnixDatagram;
use std::sync::Arc;
use std::time::{Duration, Instant};

use naked_tg::watchdog::{SystemdWatchdog, WatchdogNotifier, spawn_watchdog_ticks};
use tempfile::tempdir;
use tokio::sync::Notify;

/// Drain `buf.len()`-worth of datagrams until either `expected`
/// messages have arrived or `deadline` elapses.
async fn drain_up_to(server: &UnixDatagram, expected: usize, deadline: Instant) -> Vec<String> {
    let mut out: Vec<String> = Vec::new();
    let mut buf = [0u8; 512];
    while out.len() < expected && Instant::now() < deadline {
        match server.recv(&mut buf) {
            Ok(n) => out.push(String::from_utf8_lossy(&buf[..n]).to_string()),
            Err(e) if e.kind() == std::io::ErrorKind::WouldBlock => {
                tokio::time::sleep(Duration::from_millis(10)).await;
            }
            Err(e) => panic!("recv error: {e}"),
        }
    }
    out
}

#[tokio::test]
async fn full_lifecycle_ready_alive_stopping_roundtrips_over_unix_socket() {
    // 1. Bind a receiving socket that stands in for systemd's
    //    `$NOTIFY_SOCKET`.
    let dir = tempdir().unwrap();
    let sock_path = dir.path().join("notify.sock");
    let server = UnixDatagram::bind(&sock_path).unwrap();
    server.set_nonblocking(true).unwrap();

    // 2. Boot the watchdog exactly like `main.rs` does: detect →
    //    ready → spawn_ticks → (run) → shutdown → stopping.
    let interval = Duration::from_millis(200); // tick = 100ms
    let wd = SystemdWatchdog::for_test(sock_path.clone(), interval);
    assert_eq!(wd.interval(), Some(interval));
    let wd_arc: Arc<dyn WatchdogNotifier> = Arc::new(wd);

    // READY first.
    wd_arc.notify_ready().await;

    // Periodic ticker.
    let shutdown = Arc::new(Notify::new());
    let handle = spawn_watchdog_ticks(wd_arc.clone(), shutdown.clone())
        .expect("interval is Some → ticker must spawn");

    // 3. Let ~3.5 tick windows elapse → expect ≥2 `WATCHDOG=1`s.
    tokio::time::sleep(Duration::from_millis(350)).await;

    // 4. Graceful shutdown: stop ticks, then STOPPING=1.
    shutdown.notify_waiters();
    tokio::time::timeout(Duration::from_millis(300), handle)
        .await
        .expect("ticker must exit promptly after shutdown")
        .expect("ticker task must not panic");
    wd_arc.notify_stopping().await;

    // 5. Drain up to 1 (READY) + 5 (tolerated alive slack) + 1
    //    (STOPPING) = 7 messages with a generous recv timeout.
    let deadline = Instant::now() + Duration::from_millis(500);
    let msgs = drain_up_to(&server, 7, deadline).await;

    // 6. Invariants: READY came first, STOPPING came last, at least
    //    2 WATCHDOG=1s in between.
    assert!(
        msgs.first()
            .map(|m| m.starts_with("READY=1"))
            .unwrap_or(false),
        "first message must be READY=1, got: {msgs:?}"
    );
    assert!(
        msgs.first()
            .map(|m| m.contains(&format!("MAINPID={}", std::process::id())))
            .unwrap_or(false),
        "READY must carry MAINPID=<this pid>, got: {msgs:?}"
    );
    assert!(
        msgs.last()
            .map(|m| m.trim() == "STOPPING=1")
            .unwrap_or(false),
        "last message must be STOPPING=1, got: {msgs:?}"
    );
    let alive_count = msgs.iter().filter(|m| m.trim() == "WATCHDOG=1").count();
    assert!(
        alive_count >= 2,
        "expected ≥2 WATCHDOG=1 pings in 350ms with 200ms interval, got {alive_count}: {msgs:?}"
    );
    // And the alive pings must be strictly between READY and
    // STOPPING — no leak after shutdown.
    let stopping_idx = msgs
        .iter()
        .position(|m| m.trim() == "STOPPING=1")
        .expect("STOPPING asserted above");
    for (i, m) in msgs.iter().enumerate() {
        if m.trim() == "WATCHDOG=1" {
            assert!(
                i > 0 && i < stopping_idx,
                "WATCHDOG at {i} must be between READY(0) and STOPPING({stopping_idx}), got: {msgs:?}"
            );
        }
    }
}

#[tokio::test]
async fn ticker_stops_immediately_on_shutdown_and_emits_no_alive() {
    // A tighter safety check: if we shutdown *before* the first
    // half-interval ticks, `main.rs` must see 0 `WATCHDOG=1`
    // messages — no leak of an early ping, no hang.
    let dir = tempdir().unwrap();
    let sock_path = dir.path().join("notify.sock");
    let server = UnixDatagram::bind(&sock_path).unwrap();
    server.set_nonblocking(true).unwrap();

    let wd: Arc<dyn WatchdogNotifier> = Arc::new(SystemdWatchdog::for_test(
        sock_path,
        Duration::from_secs(10), // tick = 5s → well past our shutdown
    ));
    wd.notify_ready().await;

    let shutdown = Arc::new(Notify::new());
    let handle = spawn_watchdog_ticks(wd.clone(), shutdown.clone()).expect("ticker");

    tokio::time::sleep(Duration::from_millis(20)).await;
    shutdown.notify_waiters();
    tokio::time::timeout(Duration::from_millis(200), handle)
        .await
        .expect("ticker must drop on shutdown")
        .ok();

    wd.notify_stopping().await;

    let deadline = Instant::now() + Duration::from_millis(200);
    let msgs = drain_up_to(&server, 2, deadline).await;
    let alive = msgs.iter().filter(|m| m.trim() == "WATCHDOG=1").count();
    assert_eq!(
        alive, 0,
        "no alive ping should fire when shutdown precedes first tick, got: {msgs:?}"
    );
    assert!(
        msgs.iter().any(|m| m.starts_with("READY=1")),
        "READY must still have landed, got: {msgs:?}"
    );
    assert!(
        msgs.iter().any(|m| m.trim() == "STOPPING=1"),
        "STOPPING must still have landed, got: {msgs:?}"
    );
}
