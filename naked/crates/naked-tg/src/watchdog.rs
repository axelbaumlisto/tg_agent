//! Process-supervisor liveness reporting for `naked-tg`.
//!
//! systemd (and other supervisors that speak the `sd_notify(3)`
//! protocol) can be configured with `WatchdogSec=` to kill and
//! restart a service that goes silent. That's exactly the safety
//! net we want for the bot: the in-process panic supervisor
//! ([`crate::research_scheduler::supervised_run_loop`]) catches
//! tokio-task panics, but it cannot catch a wedge in the *runtime*
//! (deadlock, allocator stuck, blocked-on-syscall) — only an
//! external watchdog can.
//!
//! Design (SOLID + DRY):
//!
//! * [`WatchdogNotifier`] trait defines the three lifecycle events
//!   we need: `notify_ready`, `notify_alive`, `notify_stopping`.
//!   Production wires [`SystemdWatchdog`]; tests plug in a
//!   `Vec<Event>`-recording fake; environments without a supervisor
//!   get [`NoopWatchdog`] (the default).
//!
//! * [`spawn_watchdog_ticks`] is the single periodic-tick primitive.
//!   It owns the timer + shutdown wiring so individual notifier
//!   impls don't reinvent it.
//!
//! * [`SystemdWatchdog::detect_from_env`] reads `$NOTIFY_SOCKET` and
//!   `$WATCHDOG_USEC`; either being absent disables the integration
//!   gracefully (returns `None`), so the same binary runs identically
//!   under `cargo run`, `tmux`, and `systemctl --user start`.
//!
//! ## Lifecycle
//!
//! ```text
//!     boot ──notify_ready()──► READY=1\nMAINPID=<pid>
//!                                ▼
//!     loop: every (interval/2) ──notify_alive()──► WATCHDOG=1
//!                                ▼
//!     shutdown ──notify_stopping()──► STOPPING=1
//! ```
//!
//! Half-the-interval ticks are the systemd-recommended cadence
//! (`man sd_watchdog_enabled`): it gives us one full retry budget
//! in case a single tick is delayed by GC / IO without tripping the
//! kill-and-restart.

use async_trait::async_trait;
use std::env;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Duration;
use tokio::sync::Notify;
use tokio::time::{Instant, sleep};

#[cfg(unix)]
use std::os::unix::net::UnixDatagram;

/// Liveness sink. All methods are idempotent so callers can be
/// sloppy about ordering (e.g. `notify_stopping` after a `Drop`-only
/// path is harmless).
#[async_trait]
pub trait WatchdogNotifier: Send + Sync {
    /// Called once after the bot has finished booting and is ready
    /// to accept work. Sends `READY=1` to systemd.
    async fn notify_ready(&self);

    /// Periodic liveness ping. Sends `WATCHDOG=1`.
    async fn notify_alive(&self);

    /// Called once on graceful shutdown. Sends `STOPPING=1` so
    /// systemd does not flag the exit as a crash.
    async fn notify_stopping(&self);

    /// Recommended ping interval. Returning `None` disables the
    /// periodic ticker entirely (used by the no-op impl).
    fn interval(&self) -> Option<Duration> {
        None
    }
}

/// No-op watchdog. The default for environments without a
/// supervisor (interactive `cargo run`, dev tmux session, CI).
/// Implements `WatchdogNotifier` so the bot wiring is the same in
/// every environment — Open/Closed.
#[derive(Debug, Default, Clone, Copy)]
pub struct NoopWatchdog;

#[async_trait]
impl WatchdogNotifier for NoopWatchdog {
    async fn notify_ready(&self) {}
    async fn notify_alive(&self) {}
    async fn notify_stopping(&self) {}
    fn interval(&self) -> Option<Duration> {
        None
    }
}

/// Spawns a tokio task that ticks `notify.notify_alive()` every
/// `notify.interval() / 2` until `shutdown` is fired. Returns the
/// `JoinHandle` for tests; production drops it (the task self-stops
/// on shutdown).
///
/// Returns `None` immediately when `notify.interval()` is `None` —
/// callers can wire this unconditionally without an `if`.
pub fn spawn_watchdog_ticks(
    notifier: Arc<dyn WatchdogNotifier>,
    shutdown: Arc<Notify>,
) -> Option<tokio::task::JoinHandle<()>> {
    let interval = notifier.interval()?;
    if interval == Duration::ZERO {
        return None;
    }
    // systemd man page: ping at *half* the configured interval to
    // leave headroom for one missed tick.
    let tick = interval / 2;
    let tick = tick.max(Duration::from_millis(50));
    Some(tokio::spawn(async move {
        loop {
            let next = Instant::now() + tick;
            tokio::select! {
                _ = shutdown.notified() => return,
                _ = sleep_until(next) => {
                    notifier.notify_alive().await;
                }
            }
        }
    }))
}

async fn sleep_until(deadline: Instant) {
    sleep(deadline.saturating_duration_since(Instant::now())).await;
}

/// systemd-compatible watchdog. Talks the `sd_notify(3)` protocol
/// directly over a `UnixDatagram` — no external dependency, no FFI.
///
/// Construction goes through [`SystemdWatchdog::detect_from_env`]
/// which respects the standard `$NOTIFY_SOCKET` and
/// `$WATCHDOG_USEC` env contract. Returns `None` when either is
/// missing (or invalid) so the caller can fall back to
/// [`NoopWatchdog`] without branching on the platform.
#[cfg(unix)]
pub struct SystemdWatchdog {
    socket_path: PathBuf,
    interval: Duration,
    /// Cached client socket. Created lazily because tests often
    /// construct the struct without a live `$NOTIFY_SOCKET` and we
    /// don't want construction to fail just because of that.
    socket: std::sync::Mutex<Option<UnixDatagram>>,
}

#[cfg(unix)]
impl SystemdWatchdog {
    /// Build a watchdog from the systemd-set environment. Returns
    /// `None` (gracefully) when:
    ///
    /// * `$NOTIFY_SOCKET` is unset → no supervisor.
    /// * `$WATCHDOG_USEC` is unset or unparsable → supervisor is
    ///   present but watchdog is off; we still send `READY=1` /
    ///   `STOPPING=1` via [`Self::with_socket_only`] (used directly
    ///   from `main`), but the periodic ticker is a no-op.
    pub fn detect_from_env() -> Option<Self> {
        Self::detect_from(|k| env::var(k).ok())
    }

    /// Test seam: same as [`Self::detect_from_env`] but the env
    /// reader is injected. Lets tests cover the `None` edges
    /// without mutating the real process environment (which is
    /// forbidden by the workspace lint and unsafe in multi-threaded
    /// tests anyway).
    pub fn detect_from<F>(read_env: F) -> Option<Self>
    where
        F: Fn(&str) -> Option<String>,
    {
        let socket = read_env("NOTIFY_SOCKET")?;
        if socket.is_empty() {
            return None;
        }
        let usec: u64 = read_env("WATCHDOG_USEC")?.parse().ok()?;
        if usec == 0 {
            return None;
        }
        Some(Self {
            socket_path: PathBuf::from(socket),
            interval: Duration::from_micros(usec),
            socket: std::sync::Mutex::new(None),
        })
    }

    /// Construct a watchdog that only emits `READY=1` and
    /// `STOPPING=1` — no periodic pings. Used when systemd has set
    /// `Type=notify` but no `WatchdogSec=`.
    pub fn with_socket_only(socket_path: PathBuf) -> Self {
        Self {
            socket_path,
            interval: Duration::ZERO,
            socket: std::sync::Mutex::new(None),
        }
    }

    /// Test seam: build directly with both fields explicit. Not
    /// part of the public stable surface.
    #[doc(hidden)]
    pub fn for_test(socket_path: PathBuf, interval: Duration) -> Self {
        Self {
            socket_path,
            interval,
            socket: std::sync::Mutex::new(None),
        }
    }

    fn send(&self, payload: &[u8]) {
        // Best-effort: a missing socket / reset supervisor is *not*
        // fatal. We log at debug because in dev (no supervisor) the
        // socket may legitimately be absent.
        let res = self.send_inner(payload);
        if let Err(e) = res {
            tracing::debug!(socket = %self.socket_path.display(), "sd_notify send failed: {e}");
        }
    }

    fn send_inner(&self, payload: &[u8]) -> std::io::Result<()> {
        let mut guard = self
            .socket
            .lock()
            .map_err(|e| std::io::Error::other(format!("socket mutex poisoned: {e}")))?;
        if guard.is_none() {
            *guard = Some(UnixDatagram::unbound()?);
        }
        let sock = guard.as_ref().expect("just initialised");
        let path = resolve_socket_path(&self.socket_path);
        sock.send_to(payload, &path)?;
        Ok(())
    }
}

#[cfg(unix)]
#[async_trait]
impl WatchdogNotifier for SystemdWatchdog {
    async fn notify_ready(&self) {
        let pid = std::process::id();
        self.send(format!("READY=1\nMAINPID={pid}\n").as_bytes());
    }

    async fn notify_alive(&self) {
        self.send(b"WATCHDOG=1\n");
    }

    async fn notify_stopping(&self) {
        self.send(b"STOPPING=1\n");
    }

    fn interval(&self) -> Option<Duration> {
        if self.interval == Duration::ZERO {
            None
        } else {
            Some(self.interval)
        }
    }
}

/// Translate the systemd `$NOTIFY_SOCKET` value into a path the
/// `UnixDatagram` API can address. systemd uses two formats:
///
/// * `/path/to/socket` — a regular filesystem path.
/// * `@abstract-name`  — Linux abstract namespace; the leading `@`
///   stands for a NUL byte. We pass the path through unchanged
///   because `UnixDatagram::send_to` handles abstract sockets via
///   `Path` on Linux when the first byte is `\0`.
///
/// Until Rust adds first-class abstract-socket support to `Path`,
/// the `@`-style address is only usable through `socket(2)` /
/// `bind(2)` directly. We keep the conversion in one place so
/// future patches need only touch this function.
#[cfg(unix)]
fn resolve_socket_path(p: &std::path::Path) -> PathBuf {
    p.to_path_buf()
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Mutex;

    #[derive(Default, Debug)]
    struct RecordingNotifier {
        events: Mutex<Vec<&'static str>>,
        interval: Option<Duration>,
    }

    #[async_trait]
    impl WatchdogNotifier for RecordingNotifier {
        async fn notify_ready(&self) {
            self.events.lock().unwrap().push("ready");
        }
        async fn notify_alive(&self) {
            self.events.lock().unwrap().push("alive");
        }
        async fn notify_stopping(&self) {
            self.events.lock().unwrap().push("stopping");
        }
        fn interval(&self) -> Option<Duration> {
            self.interval
        }
    }

    #[tokio::test]
    async fn noop_watchdog_reports_no_interval_and_swallows_calls() {
        let n = NoopWatchdog;
        assert!(n.interval().is_none());
        // Must not panic.
        n.notify_ready().await;
        n.notify_alive().await;
        n.notify_stopping().await;
    }

    #[tokio::test]
    async fn spawn_returns_none_when_interval_is_none() {
        let n: Arc<dyn WatchdogNotifier> = Arc::new(NoopWatchdog);
        let shutdown = Arc::new(Notify::new());
        assert!(spawn_watchdog_ticks(n, shutdown).is_none());
    }

    #[tokio::test]
    async fn spawn_returns_none_when_interval_is_zero() {
        let n: Arc<RecordingNotifier> = Arc::new(RecordingNotifier {
            events: Default::default(),
            interval: Some(Duration::ZERO),
        });
        let dyn_n: Arc<dyn WatchdogNotifier> = n.clone();
        let shutdown = Arc::new(Notify::new());
        assert!(spawn_watchdog_ticks(dyn_n, shutdown).is_none());
        assert!(n.events.lock().unwrap().is_empty());
    }

    #[tokio::test]
    async fn ticker_emits_alive_at_half_the_interval() {
        let n: Arc<RecordingNotifier> = Arc::new(RecordingNotifier {
            events: Default::default(),
            interval: Some(Duration::from_millis(200)),
        });
        let dyn_n: Arc<dyn WatchdogNotifier> = n.clone();
        let shutdown = Arc::new(Notify::new());
        let handle = spawn_watchdog_ticks(dyn_n, shutdown.clone()).expect("ticker");
        // 200ms interval → 100ms tick. After ~350ms we should see
        // at least 2 pings (and definitely no more than ~5 to leave
        // room for jitter on slow CI runners).
        tokio::time::sleep(Duration::from_millis(350)).await;
        shutdown.notify_waiters();
        let _ = handle.await;
        let n_alive = n
            .events
            .lock()
            .unwrap()
            .iter()
            .filter(|e| **e == "alive")
            .count();
        assert!(
            (2..=5).contains(&n_alive),
            "expected 2..=5 alive pings in 350ms with 100ms tick, got {n_alive}"
        );
    }

    #[tokio::test]
    async fn ticker_stops_promptly_on_shutdown() {
        let n: Arc<RecordingNotifier> = Arc::new(RecordingNotifier {
            events: Default::default(),
            interval: Some(Duration::from_millis(500)),
        });
        let dyn_n: Arc<dyn WatchdogNotifier> = n.clone();
        let shutdown = Arc::new(Notify::new());
        let handle = spawn_watchdog_ticks(dyn_n, shutdown.clone()).expect("ticker");
        // Fire shutdown almost immediately — the join should
        // resolve well before the first tick (250ms) would fire.
        tokio::time::sleep(Duration::from_millis(20)).await;
        shutdown.notify_waiters();
        tokio::time::timeout(Duration::from_millis(150), handle)
            .await
            .expect("ticker must drop on shutdown well before first tick")
            .ok();
        // No `alive` should have fired this fast.
        let events = n.events.lock().unwrap().clone();
        assert!(
            events.iter().all(|e| *e != "alive"),
            "expected no alive pings before first half-interval, got {events:?}"
        );
    }

    #[cfg(unix)]
    #[test]
    fn detect_from_returns_none_when_socket_unset() {
        let env = |_: &str| -> Option<String> { None };
        assert!(SystemdWatchdog::detect_from(env).is_none());
    }

    #[cfg(unix)]
    #[test]
    fn detect_from_returns_none_when_socket_empty() {
        let env = |k: &str| -> Option<String> {
            if k == "NOTIFY_SOCKET" { Some(String::new()) } else { None }
        };
        assert!(SystemdWatchdog::detect_from(env).is_none());
    }

    #[cfg(unix)]
    #[test]
    fn detect_from_returns_none_when_usec_missing() {
        let env = |k: &str| -> Option<String> {
            if k == "NOTIFY_SOCKET" {
                Some("/tmp/never-exists.sock".into())
            } else {
                None
            }
        };
        assert!(SystemdWatchdog::detect_from(env).is_none());
    }

    #[cfg(unix)]
    #[test]
    fn detect_from_returns_none_when_usec_zero_or_garbage() {
        let env_zero = |k: &str| -> Option<String> {
            match k {
                "NOTIFY_SOCKET" => Some("/tmp/x.sock".into()),
                "WATCHDOG_USEC" => Some("0".into()),
                _ => None,
            }
        };
        assert!(SystemdWatchdog::detect_from(env_zero).is_none());

        let env_garbage = |k: &str| -> Option<String> {
            match k {
                "NOTIFY_SOCKET" => Some("/tmp/x.sock".into()),
                "WATCHDOG_USEC" => Some("not-a-number".into()),
                _ => None,
            }
        };
        assert!(SystemdWatchdog::detect_from(env_garbage).is_none());
    }

    #[cfg(unix)]
    #[test]
    fn detect_from_returns_some_with_proper_env() {
        let env = |k: &str| -> Option<String> {
            match k {
                "NOTIFY_SOCKET" => Some("/tmp/notify.sock".into()),
                "WATCHDOG_USEC" => Some("30000000".into()),
                _ => None,
            }
        };
        let wd = SystemdWatchdog::detect_from(env).expect("must build");
        assert_eq!(wd.interval(), Some(Duration::from_secs(30)));
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn systemd_watchdog_writes_to_unix_socket() {
        // Bind a real datagram socket in tempdir, point a watchdog
        // at it, fire READY/WATCHDOG/STOPPING, and verify all three
        // payloads land verbatim.
        let dir = tempfile::tempdir().unwrap();
        let sock_path = dir.path().join("notify.sock");
        let server = UnixDatagram::bind(&sock_path).unwrap();
        server.set_nonblocking(true).unwrap();

        let wd = SystemdWatchdog::for_test(sock_path.clone(), Duration::from_secs(30));
        wd.notify_ready().await;
        wd.notify_alive().await;
        wd.notify_stopping().await;

        let mut received: Vec<String> = Vec::new();
        let deadline = std::time::Instant::now() + Duration::from_millis(500);
        let mut buf = [0u8; 256];
        while received.len() < 3 && std::time::Instant::now() < deadline {
            match server.recv(&mut buf) {
                Ok(n) => received.push(String::from_utf8_lossy(&buf[..n]).to_string()),
                Err(e) if e.kind() == std::io::ErrorKind::WouldBlock => {
                    tokio::time::sleep(Duration::from_millis(10)).await;
                }
                Err(e) => panic!("recv error: {e}"),
            }
        }
        assert_eq!(received.len(), 3, "got: {received:?}");
        assert!(received[0].starts_with("READY=1"), "got: {:?}", received[0]);
        assert!(
            received[0].contains(&format!("MAINPID={}", std::process::id())),
            "got: {:?}",
            received[0]
        );
        assert_eq!(received[1].trim(), "WATCHDOG=1");
        assert_eq!(received[2].trim(), "STOPPING=1");
        assert_eq!(wd.interval(), Some(Duration::from_secs(30)));
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn systemd_watchdog_send_to_missing_socket_is_swallowed() {
        // No bound socket at this path — `send_to` will EAGAIN /
        // ENOENT. Must not panic, must not propagate.
        let wd = SystemdWatchdog::for_test(
            PathBuf::from("/tmp/naked-tg-watchdog-nonexistent.sock"),
            Duration::from_secs(30),
        );
        wd.notify_ready().await;
        wd.notify_alive().await;
        wd.notify_stopping().await;
    }

    #[cfg(unix)]
    #[test]
    fn with_socket_only_disables_periodic_ticker() {
        let wd = SystemdWatchdog::with_socket_only(PathBuf::from("/tmp/x.sock"));
        assert!(
            wd.interval().is_none(),
            "no WatchdogSec → no periodic ticks, but READY/STOPPING still send"
        );
    }
}
