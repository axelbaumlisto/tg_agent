//! Supervised long-running tasks.
//!
//! Wraps `tokio::spawn` with three guarantees the bare primitive lacks:
//!
//! 1. **Panic survival.** A panic inside the task body is logged at
//!    `error!` and the task is restarted (modulo backoff). The runtime
//!    is unaffected. Without this wrapper, a panic in a detached task
//!    silently dies — exactly the failure mode of the polling loop in
//!    incident 2026-05-10 12:47.
//!
//! 2. **Error → restart.** A `Result::Err` return triggers the same
//!    restart path as a panic. Useful for tasks that surface
//!    “liveness exceeded” / network / parser errors and want to be
//!    re-spun rather than declared dead.
//!
//! 3. **Cancellation-respecting.** Holds a `CancellationToken`. When
//!    cancelled, the supervisor stops re-spinning the inner task and
//!    awaits the in-flight body. Backoff sleeps are themselves
//!    `select!`'d against the token so shutdown is bounded.
//!
//! Plan: naked/docs/PLAN_LIVENESS_v1.md §4 T3.

use std::future::Future;
use std::sync::Arc;
use std::time::Duration;

use async_trait::async_trait;
use tokio::task::JoinHandle;
use tokio_util::sync::CancellationToken;

/// Optional panic-observation hook for [`spawn_supervised_with_opts`].
///
/// Every panic restart in the supervisor produces one call to
/// [`PanicHook::on_panic`]. The default `spawn_supervised` does not
/// install a hook (matches old behaviour); the research scheduler
/// installs an adapter around its [`TaskNotifier`] so operators get
/// the same alert that the legacy `supervised_run_loop` produced.
///
/// SOLID/DRY anchor: this trait is the only point of contact between
/// the supervisor primitive and any panic-notification subsystem.
/// The supervisor does not depend on scheduler types.
#[async_trait]
pub trait PanicHook: Send + Sync {
    /// Called once per panic-detected restart.
    async fn on_panic(&self, task_name: &str, attempt: u32, details: &str);
}

/// Full configuration for [`spawn_supervised_with_opts`].
///
/// The legacy [`spawn_supervised`] builds a default options struct;
/// callers that need a panic hook or a restart cap construct this
/// directly. The struct is `#[non_exhaustive]`-equivalent in spirit
/// (every field is `pub` and named) so future options (e.g.
/// `panic_window_size`) can be added without breaking call sites.
pub struct SupervisorOptions {
    /// Static label used in tracing spans + the panic hook payload.
    pub name: &'static str,
    /// Backoff curve for restart sleeps.
    pub backoff: Backoff,
    /// External cancel signal. The supervisor breaks out of its
    /// restart loop when fired and propagates the same token to the
    /// inner task body.
    pub shutdown: CancellationToken,
    /// `Some(N)` to give up after N attempts. `None` = restart
    /// forever (production polling / scheduler default).
    pub max_attempts: Option<u32>,
    /// Panic observer. `None` to disable.
    pub panic_hook: Option<Arc<dyn PanicHook>>,
}

/// Reusable backoff schedule for the supervisor's restart loop.
#[derive(Debug, Clone, Copy)]
pub struct Backoff {
    /// First retry delay (e.g. 250 ms).
    pub initial: Duration,
    /// Doubling cap (e.g. 30 s). Subsequent retries plateau here.
    pub max: Duration,
    /// `current = current * 2` after every restart, capped at `max`.
    pub multiplier: u32,
}

impl Backoff {
    pub const fn default_polling() -> Self {
        Self {
            initial: Duration::from_millis(500),
            max: Duration::from_secs(30),
            multiplier: 2,
        }
    }

    /// Pure helper — given the current delay, compute the next delay.
    /// Public so unit tests can verify the curve without poking at the
    /// supervisor internals.
    #[must_use]
    pub fn next(&self, current: Duration) -> Duration {
        let next_ms = current
            .as_millis()
            .saturating_mul(u128::from(self.multiplier))
            .min(self.max.as_millis());
        Duration::from_millis(next_ms as u64)
    }
}

/// Spawn a long-running, supervised task.
///
/// `factory` is called once per attempt — it must be `Clone`-friendly
/// or carry only owned state. The supervisor itself runs as a single
/// detached `tokio::spawn`; it returns the supervisor's own
/// `JoinHandle`. Aborting that handle signals everything to wind down
/// (the factory's most recent body sees its `CancellationToken`
/// cancelled via the same token the supervisor holds).
///
/// Per-attempt outcomes:
///   - `Ok(Ok(()))`: clean exit, supervisor stops.
///   - `Ok(Err(e))`: factory returned Err; logged, restart after backoff.
///   - panic: caught by `JoinHandle::await`, logged, restart after backoff.
///   - cancellation: the inner task observes the same token and
///     returns; the supervisor breaks out of the restart loop.
pub fn spawn_supervised<F, Fut>(
    name: &'static str,
    backoff: Backoff,
    shutdown: CancellationToken,
    factory: F,
) -> JoinHandle<()>
where
    F: FnMut(CancellationToken) -> Fut + Send + 'static,
    Fut: Future<Output = anyhow::Result<()>> + Send + 'static,
{
    spawn_supervised_with_opts(
        SupervisorOptions {
            name,
            backoff,
            shutdown,
            max_attempts: None,
            panic_hook: None,
        },
        factory,
    )
}

/// Full-control supervisor. F1 of `PLAN_NEXT_SESSION.md`: this is
/// the single supervision primitive in the codebase. The legacy
/// `spawn_supervised` and the scheduler's old `supervised_run_loop`
/// both delegate here.
///
/// Per-attempt outcomes are unchanged from `spawn_supervised`. The
/// extra options:
///   * `panic_hook`: invoked on every panic-restart with `(name,
///     attempt, details)`. Errors are logged but not piped through
///     the hook — the hook is panic-specific (matches the old
///     `notify_supervisor_panic` semantics).
///   * `max_attempts: Some(n)`: after `n` panics, the supervisor
///     gives up (logs at `error!` and exits). Errors do NOT consume
///     the attempt budget — only panics, matching the legacy
///     `supervised_run_loop_capped` semantics tests assert.
pub fn spawn_supervised_with_opts<F, Fut>(opts: SupervisorOptions, mut factory: F) -> JoinHandle<()>
where
    F: FnMut(CancellationToken) -> Fut + Send + 'static,
    Fut: Future<Output = anyhow::Result<()>> + Send + 'static,
{
    let SupervisorOptions {
        name,
        backoff,
        shutdown,
        max_attempts,
        panic_hook,
    } = opts;
    tokio::spawn(async move {
        let mut delay = backoff.initial;
        let mut panic_attempts: u32 = 0;
        loop {
            if shutdown.is_cancelled() {
                tracing::debug!(task = name, "supervised: shutdown observed before spawn");
                break;
            }

            let attempt_token = shutdown.clone();
            let inner = tokio::spawn(factory(attempt_token));

            match inner.await {
                Ok(Ok(())) => {
                    tracing::info!(task = name, "supervised task exited Ok; not restarting");
                    break;
                }
                Ok(Err(e)) => {
                    tracing::error!(task = name, "supervised task error: {e:#}");
                }
                Err(join_err) if join_err.is_panic() => {
                    panic_attempts = panic_attempts.saturating_add(1);
                    let msg = panic_payload(join_err.into_panic());
                    let details = format!(
                        "task '{name}' panicked (attempt {panic_attempts}/{cap}): {msg}",
                        cap = match max_attempts {
                            Some(n) => n.to_string(),
                            None => "∞".to_string(),
                        }
                    );
                    tracing::error!("{details}");
                    if let Some(hook) = panic_hook.as_ref() {
                        hook.on_panic(name, panic_attempts, &details).await;
                    }
                    if let Some(cap) = max_attempts
                        && panic_attempts >= cap
                    {
                        tracing::error!(
                            task = name,
                            attempts = panic_attempts,
                            "supervised task reached panic cap; giving up"
                        );
                        break;
                    }
                }
                Err(_cancelled) => {
                    tracing::info!(
                        task = name,
                        "supervised task cancelled by JoinHandle::abort"
                    );
                    break;
                }
            }

            if shutdown.is_cancelled() {
                tracing::debug!(task = name, "supervised: shutdown observed before backoff");
                break;
            }

            tracing::warn!(
                task = name,
                delay_ms = delay.as_millis() as u64,
                "supervised: backing off before restart"
            );
            tokio::select! {
                _ = shutdown.cancelled() => {
                    tracing::debug!(task = name, "supervised: shutdown during backoff");
                    break;
                }
                _ = tokio::time::sleep(delay) => {}
            }
            delay = backoff.next(delay);
        }
        tracing::info!(task = name, "supervised loop done");
    })
}

fn panic_payload(payload: Box<dyn std::any::Any + Send>) -> String {
    if let Some(s) = payload.downcast_ref::<&'static str>() {
        (*s).to_string()
    } else if let Some(s) = payload.downcast_ref::<String>() {
        s.clone()
    } else {
        "<non-string panic payload>".to_string()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;
    use std::sync::atomic::{AtomicUsize, Ordering};

    fn ts(ms: u64) -> Duration {
        Duration::from_millis(ms)
    }

    #[test]
    fn backoff_curve_doubles_then_caps() {
        let b = Backoff {
            initial: ts(100),
            max: ts(800),
            multiplier: 2,
        };
        assert_eq!(b.next(ts(100)), ts(200));
        assert_eq!(b.next(ts(200)), ts(400));
        assert_eq!(b.next(ts(400)), ts(800));
        assert_eq!(b.next(ts(800)), ts(800), "cap holds");
        assert_eq!(b.next(ts(2000)), ts(800), "above-cap input clamps to cap");
    }

    #[tokio::test]
    async fn ok_exit_does_not_restart() {
        let calls = Arc::new(AtomicUsize::new(0));
        let calls_clone = Arc::clone(&calls);
        let shutdown = CancellationToken::new();

        let h = spawn_supervised(
            "test_ok",
            Backoff {
                initial: ts(10),
                max: ts(50),
                multiplier: 2,
            },
            shutdown.clone(),
            move |_tok| {
                let calls = Arc::clone(&calls_clone);
                async move {
                    calls.fetch_add(1, Ordering::SeqCst);
                    Ok(())
                }
            },
        );

        h.await.unwrap();
        assert_eq!(calls.load(Ordering::SeqCst), 1);
    }

    #[tokio::test]
    async fn err_triggers_restart() {
        let calls = Arc::new(AtomicUsize::new(0));
        let calls_clone = Arc::clone(&calls);
        let shutdown = CancellationToken::new();

        let h = spawn_supervised(
            "test_err",
            Backoff {
                initial: ts(5),
                max: ts(20),
                multiplier: 2,
            },
            shutdown.clone(),
            move |_tok| {
                let calls = Arc::clone(&calls_clone);
                async move {
                    let n = calls.fetch_add(1, Ordering::SeqCst) + 1;
                    if n < 3 {
                        Err(anyhow::anyhow!("simulated failure {n}"))
                    } else {
                        Ok(())
                    }
                }
            },
        );

        h.await.unwrap();
        // 2 errors then 1 success = 3 calls total
        assert_eq!(calls.load(Ordering::SeqCst), 3);
    }

    #[tokio::test]
    async fn panic_triggers_restart() {
        let calls = Arc::new(AtomicUsize::new(0));
        let calls_clone = Arc::clone(&calls);
        let shutdown = CancellationToken::new();

        let h = spawn_supervised(
            "test_panic",
            Backoff {
                initial: ts(5),
                max: ts(20),
                multiplier: 2,
            },
            shutdown.clone(),
            move |_tok| {
                let calls = Arc::clone(&calls_clone);
                async move {
                    let n = calls.fetch_add(1, Ordering::SeqCst) + 1;
                    if n < 2 {
                        panic!("intentional panic in supervised test");
                    }
                    Ok(())
                }
            },
        );

        h.await.unwrap();
        assert_eq!(calls.load(Ordering::SeqCst), 2);
    }

    #[tokio::test]
    async fn shutdown_during_backoff_breaks_immediately() {
        let calls = Arc::new(AtomicUsize::new(0));
        let calls_clone = Arc::clone(&calls);
        let shutdown = CancellationToken::new();

        let h = spawn_supervised(
            "test_shutdown_backoff",
            Backoff {
                initial: ts(5_000), // 5s sleep — shutdown must short-circuit
                max: ts(30_000),
                multiplier: 2,
            },
            shutdown.clone(),
            move |_tok| {
                let calls = Arc::clone(&calls_clone);
                async move {
                    calls.fetch_add(1, Ordering::SeqCst);
                    Err(anyhow::anyhow!("force backoff"))
                }
            },
        );

        // Let one Err fire, then shutdown immediately — supervisor must
        // break out of its 5s sleep, not wait it out.
        tokio::time::sleep(ts(50)).await;
        shutdown.cancel();

        let start = std::time::Instant::now();
        h.await.unwrap();
        let elapsed = start.elapsed();
        assert!(
            elapsed < ts(500),
            "supervisor took {elapsed:?} after shutdown — should be <500ms"
        );
        // Only one attempt happened (subsequent restart was cancelled in backoff).
        assert_eq!(calls.load(Ordering::SeqCst), 1);
    }

    #[tokio::test]
    async fn task_observes_shutdown_token() {
        // Inner task must see cancellation through the token it was
        // handed; supervisor closes cleanly when inner returns Ok via
        // shutdown branch.
        let shutdown = CancellationToken::new();
        let calls = Arc::new(AtomicUsize::new(0));
        let calls_clone = Arc::clone(&calls);

        let h = spawn_supervised(
            "test_inner_obeys_shutdown",
            Backoff {
                initial: ts(10),
                max: ts(50),
                multiplier: 2,
            },
            shutdown.clone(),
            move |tok| {
                let calls = Arc::clone(&calls_clone);
                async move {
                    calls.fetch_add(1, Ordering::SeqCst);
                    tokio::select! {
                        _ = tok.cancelled() => Ok(()),
                        _ = tokio::time::sleep(ts(60_000)) => unreachable!(),
                    }
                }
            },
        );

        tokio::time::sleep(ts(30)).await;
        shutdown.cancel();
        let start = std::time::Instant::now();
        h.await.unwrap();
        let elapsed = start.elapsed();
        assert!(elapsed < ts(500));
        assert_eq!(calls.load(Ordering::SeqCst), 1);
    }
}
