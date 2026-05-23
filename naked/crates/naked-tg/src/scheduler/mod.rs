//! In-process research scheduler: wakes periodically (and on
//! [`naked_core::research::SchedulerEvent`] notifications), inspects every
//! spec, and launches research runs for those whose schedule is due.
//!
//! ## Trigger semantics (priority, highest first)
//!
//! Each [`ResearchSpec`] may carry up to four scheduling fields. They are
//! evaluated in this order; the first one that says "due now" wins:
//!
//! 1. `paused`            — short-circuit: never due.
//! 2. `run_at`            — one-shot at-time: fires once when `now >= run_at`,
//!    then the field is cleared back to disk.
//! 3. `cron`              — recurring 5-field cron expression.
//! 4. `interval_seconds`  — legacy "every N seconds" trigger, still honoured
//!    for backward-compat with existing JSONL.
//!
//! ## Concurrency, tracking, and timeouts
//!
//! * Up to `SchedulerConfig.max_concurrent_runs` runs may be in flight at
//!   once (default `5`). The cap is enforced by counting entries in the
//!   in-memory `running` map — _not_ by the AgentCore semaphore — so the
//!   scheduler can keep dispatching new ones the moment a slot opens.
//! * Each tick performs a "sweep" before dispatch: every running task is
//!   checked for completion (`JoinHandle::is_finished()`) or timeout
//!   (`started_at.elapsed() > task_timeout`). Completed slots are freed;
//!   timed-out tasks are flagged as failures and their slot is released
//!   (the underlying run keeps draining naturally — the scheduler does NOT
//!   force-cancel because that would require a cancel-token plumbed through
//!   the entire research coordinator).
//! * Consecutive failures (errors + timeouts) per spec are tracked in
//!   memory. When the counter reaches
//!   `SchedulerConfig.max_retries_before_alert` (default `3`), the
//!   scheduler calls [`TaskNotifier::notify_failure`] (which the bot wires
//!   to a Telegram message into `spec.chat_id`) and resets the counter.

use std::collections::{HashMap, HashSet};
use std::sync::{Arc, Weak};
use std::time::Duration;

use async_trait::async_trait;
use chrono::{DateTime, Utc};
use naked_core::AgentCore;
use naked_core::research::{
    Inflight, ResearchSpec, ResearchStore, RunState, SchedulerEvent, SchedulerHook, StopReason,
};
use naked_core::{PatchField, ResearchPatch};
use tokio::sync::{Mutex, Notify, Semaphore};
use tokio_util::sync::CancellationToken;

use crate::cron_util::next_cron_after;
use tokio::task::JoinHandle;
use tokio::time::{Instant, sleep};

mod clock;
mod dispatch;
mod lifecycle;
mod tasks;

// Public re-exports for external tests and other crates
pub use clock::{Clock, RealClock};
pub use dispatch::{is_due, plan_dispatches};

/// Configuration knobs for [`ResearchScheduler`]. Defaults match the prod TG
/// bot: scan every 30s, allow five runs at a time, time out individual runs
/// at 10 minutes, alert after 3 consecutive failures.
///
/// **NOTE**: `Debug` is intentionally NOT derived because `dispatch_fn`
/// (T6.3 — PLAN_RESEARCH_AGENT_FLOW_v1) wraps an opaque async closure that
/// can't be Debug-printed. Manual impl below renders the closure as a
/// presence/absence marker so logging is still useful.
#[derive(Clone)]
pub struct SchedulerConfig {
    /// Maximum interval between background scans even when no events arrive.
    pub tick_interval: Duration,
    /// Hard cap on concurrent runs the scheduler will keep in flight.
    /// Mirrors `ResearchConfig.max_concurrent_runs` in production.
    pub max_concurrent_runs: usize,
    /// Whether to use the gatekeeper-verified loop. Mirrors
    /// `ResearchConfig.verify_by_default`.
    pub verify_by_default: bool,
    /// Max gatekeeper rounds when `verify_by_default` is true.
    pub max_verification_rounds: u32,
    /// Default per-task wall-clock timeout. Per-spec override:
    /// `ResearchSpec.task_timeout_seconds`.
    pub task_timeout: Duration,
    /// Number of consecutive failures (errors + timeouts) at which a
    /// *first* alert is emitted via [`TaskNotifier`]. The counter is
    /// **not** reset by the alert (so `auto_pause_after_failures` can
    /// keep counting); the next alert for the same streak only fires
    /// when the spec is auto-paused. `0` disables alerts entirely.
    pub max_retries_before_alert: u32,
    /// After this many consecutive failures the scheduler patches
    /// `spec.paused = true` on disk and stops touching the spec
    /// until an operator runs `/research resume`. This is the loop
    /// breaker: prevents a deterministic-failure spec from burning
    /// LLM budget forever via cron re-fires. MUST be >=
    /// `max_retries_before_alert` to be useful (otherwise auto-pause
    /// fires before the operator hears about it). `0` disables.
    /// Default `5`.
    pub auto_pause_after_failures: u32,
    /// How many times an inflight task may be resurrected after the
    /// process crashed mid-run before being declared permanently
    /// `Failed`. `0` disables resurrection entirely. Default `3`.
    pub max_resurrection_attempts: u32,
    /// Heartbeat budget for running tasks. If a task's
    /// `last_heartbeat` is older than this, the sweep treats it as
    /// stuck and finalises it as `Failed`. The budget MUST exceed
    /// `tick_interval` so a healthy in-flight job is never killed by
    /// transient lock contention. Default = 4 × `tick_interval`.
    pub heartbeat_budget: Duration,
    /// Pause between consecutive resurrection spawns at boot.
    /// Avoids a thundering-herd of LLM calls when many specs were
    /// running at the moment the previous process died (e.g. a
    /// kernel OOM that took out 20 in-flight researches).
    /// Default `100ms`.
    pub resurrection_stagger: Duration,
    /// Two-stage cancellation grace window. When a task exceeds its
    /// `task_timeout`, the sweep first issues a *cooperative* cancel
    /// via [`tokio_util::sync::CancellationToken`] — letting the
    /// research coordinator drop in-flight HTTP/LLM streams,
    /// persist a `Cancelled` `RunRecord`, and exit cleanly. Only if
    /// the worker is still running after `cancel_grace_period` does
    /// the sweep escalate to a hard `JoinHandle::abort()`. Default
    /// `30s`. Set to `Duration::ZERO` to skip cooperative cancel and
    /// abort immediately (legacy behaviour, useful for tests).
    pub cancel_grace_period: Duration,
    /// How long terminal `Inflight` records (`Completed`/`Failed`) are
    /// retained on disk after their `finished_at` timestamp before the
    /// periodic sweep deletes them via
    /// [`naked_core::research::ResearchStore::purge_terminal_inflight`].
    /// Operators rely on these records for `/research state` ("what
    /// was the last attempt for this spec?"), so the default keeps a
    /// generous **14 day** window. `Duration::ZERO` disables the
    /// periodic purge entirely.
    pub inflight_terminal_retention: Duration,
    /// How often the periodic sweep runs the terminal-inflight purge.
    /// Independent of `tick_interval` because the sweep itself is
    /// cheap (one `read_dir` + a few `stat`s per spec) and operators
    /// don't need second-by-second reaction time. Default `1h`.
    /// `Duration::ZERO` (or `inflight_terminal_retention == ZERO`)
    /// disables the periodic purge entirely.
    pub inflight_purge_interval: Duration,
    /// Time source used by the scheduler. Defaults to [`RealClock`] (wall
    /// clock). Override with a `MockClock` in tests for deterministic time.
    pub clock: std::sync::Arc<dyn Clock>,
    /// Optional liveness registry. When set, the scheduler beats
    /// `"scheduler.tick"` at the start of every tick so the watchdog
    /// arbiter (`spawn_watchdog_with_liveness`) can detect a frozen
    /// scheduler with the same mechanism it uses for the polling
    /// loop. F2 of `PLAN_NEXT_SESSION.md`.
    pub liveness: Option<std::sync::Arc<naked_core::liveness::LivenessRegistry>>,
    /// Optional synthetic-message dispatcher. T6.3 (PLAN_RESEARCH_AGENT_FLOW_v1):
    /// when set AND `spec.chat_id` is configured, the scheduler injects a
    /// synthetic message into the chat thread instead of calling
    /// `run_research_verified_with_cancel` directly. This binds the
    /// scheduler-launched run to a real chat session so the operator
    /// can `/abort` it like any other turn (B57 mitigation).
    ///
    /// When `None` (default) OR when `spec.chat_id` is `None`, the
    /// scheduler falls back to the legacy direct-run path (T6.4 fallback).
    /// Wired by `wiring.rs` at boot.
    pub dispatch_fn: Option<crate::synthetic::SyntheticDispatchFn>,
}

impl std::fmt::Debug for SchedulerConfig {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SchedulerConfig")
            .field("tick_interval", &self.tick_interval)
            .field("max_concurrent_runs", &self.max_concurrent_runs)
            .field("verify_by_default", &self.verify_by_default)
            .field("max_verification_rounds", &self.max_verification_rounds)
            .field("task_timeout", &self.task_timeout)
            .field("max_retries_before_alert", &self.max_retries_before_alert)
            .field("auto_pause_after_failures", &self.auto_pause_after_failures)
            .field("max_resurrection_attempts", &self.max_resurrection_attempts)
            .field("heartbeat_budget", &self.heartbeat_budget)
            .field("resurrection_stagger", &self.resurrection_stagger)
            .field("cancel_grace_period", &self.cancel_grace_period)
            .field(
                "inflight_terminal_retention",
                &self.inflight_terminal_retention,
            )
            .field("inflight_purge_interval", &self.inflight_purge_interval)
            .field("liveness", &self.liveness.is_some())
            .field("dispatch_fn", &self.dispatch_fn.is_some())
            .finish()
    }
}

impl Default for SchedulerConfig {
    fn default() -> Self {
        let tick = Duration::from_secs(30);
        Self {
            tick_interval: tick,
            max_concurrent_runs: 5,
            verify_by_default: true,
            max_verification_rounds: 3,
            task_timeout: Duration::from_secs(1800),
            max_retries_before_alert: 3,
            auto_pause_after_failures: 5,
            max_resurrection_attempts: 3,
            heartbeat_budget: tick * 4,
            resurrection_stagger: Duration::from_millis(100),
            cancel_grace_period: Duration::from_secs(30),
            inflight_terminal_retention: Duration::from_secs(14 * 24 * 60 * 60),
            inflight_purge_interval: Duration::from_secs(60 * 60),
            clock: std::sync::Arc::new(RealClock),
            liveness: None,
            dispatch_fn: None,
        }
    }
}

/// Outcome of a single scheduler-launched run, used internally to update
/// per-spec failure counters and decide whether to alert.
#[derive(Debug)]
enum RunOutcome {
    Success,
    Failure(String),
}

/// Tracking record for an in-flight scheduler run. Stored in the `running`
/// map; the sweep phase of every tick consults this to free slots and detect
/// stuck tasks.
struct RunningHandle {
    started_at: Instant,
    timeout: Duration,
    handle: JoinHandle<RunOutcome>,
    /// Cooperative-cancellation handle plumbed all the way down to
    /// `coordinator::run_once_with_cancel`. The sweep timeout uses
    /// `cancel.cancel()` first (so the worker can drop its HTTP/LLM
    /// streams gracefully and persist a `Cancelled` RunRecord) and
    /// only falls back to `JoinHandle::abort()` if the worker is
    /// still around after the cooperative grace window.
    cancel: CancellationToken,
    /// `Some(t)` once the sweep has issued the cooperative cancel
    /// at instant `t`. The next sweep tick uses
    /// `cancel_requested_at.elapsed() > cancel_grace_period` to
    /// decide when to escalate to a hard abort. `None` for live,
    /// healthy tasks.
    cancel_requested_at: Option<Instant>,
    /// `attempt_id` of the `Inflight` record that owns this run.
    /// Logged on slot release for correlation with the on-disk
    /// ledger; otherwise unused at runtime.
    #[allow(dead_code)]
    attempt_id: String,
}

/// Notification sink for terminal scheduler events. The production bot
/// implements this to push a Telegram message into `spec.chat_id`; tests use
/// `NoopNotifier` (default) which discards everything.
///
/// `notify_failure` is the only required method — it fires when the failure
/// streak for a spec reaches `max_retries_before_alert`. `notify_success` is
/// optional and currently unused by production code, but reserved for future
/// "first successful run after N failures" hooks.
#[async_trait]
pub trait TaskNotifier: Send + Sync {
    async fn notify_failure(
        &self,
        spec: &ResearchSpec,
        consecutive_failures: u32,
        last_error: &str,
    );
    async fn notify_success(&self, _spec: &ResearchSpec, _run_id: &str) {}

    /// Fired by [`supervised_run_loop`] when the scheduler's main loop panics.
    /// Default implementation is a no-op so existing notifier impls (e.g.
    /// [`NoopNotifier`], legacy bot wirings) keep compiling unchanged
    /// (Open/Closed). The supervisor *also* logs at `error!` level so
    /// observability never depends solely on this hook.
    async fn notify_supervisor_panic(&self, _details: &str) {}
}

/// Default no-op notifier. Used when the bot doesn't wire a real one — keeps
/// every test path and headless deployment hassle-free.
pub struct NoopNotifier;

#[async_trait]
impl TaskNotifier for NoopNotifier {
    async fn notify_failure(
        &self,
        _spec: &ResearchSpec,
        _consecutive_failures: u32,
        _last_error: &str,
    ) {
    }
}

/// Adapter: bridge a [`TaskNotifier`] to the supervisor's
/// [`crate::supervised::PanicHook`] interface. F1 of
/// `PLAN_NEXT_SESSION.md` removed the bespoke
/// `lifecycle::supervised_run_loop`; this is the only piece of glue
/// needed to keep the existing `notify_supervisor_panic`
/// observability path intact when the research scheduler runs under
/// the unified primitive.
pub(crate) struct NotifierPanicHook(pub(crate) Arc<dyn TaskNotifier>);

#[async_trait]
impl crate::supervised::PanicHook for NotifierPanicHook {
    async fn on_panic(&self, _task_name: &str, _attempt: u32, details: &str) {
        // The legacy hook signature is `(details: &str)`. We pass
        // through a pre-formatted string that already contains the
        // task name + attempt number (built by
        // `spawn_supervised_with_opts`).
        self.0.notify_supervisor_panic(details).await;
    }
}

/// The hook end of the scheduler — implements
/// [`naked_core::research::SchedulerHook`] and pokes the scheduler loop on
/// every spec mutation so reschedules are immediate (no need to wait for the
/// next tick).
pub struct ResearchSchedulerHook {
    notify: Arc<Notify>,
    /// Live handle to the same `SchedulerState` the loop mutates.
    /// Lets `failure_snapshot()` return the current consecutive-fail
    /// counter and alert flag without spawning a side-channel —
    /// readers just take the lock for a microsecond and copy two
    /// `u32`s. `Weak` so the hook can outlive the scheduler thread
    /// during shutdown without leaking the state map.
    state: std::sync::Weak<Mutex<SchedulerState>>,
}

#[async_trait]
impl SchedulerHook for ResearchSchedulerHook {
    async fn notify(&self, event: SchedulerEvent) {
        tracing::debug!(?event, "scheduler hook fired");
        self.notify.notify_one();
    }

    async fn failure_snapshot(&self, spec_id: &str) -> Option<(u32, bool)> {
        let state = self.state.upgrade()?;
        let s = state.lock().await;
        let count = s.failures.get(spec_id).copied().unwrap_or(0);
        let alerted = s.alerted.contains(spec_id);
        Some((count, alerted))
    }

    async fn reset_failures(&self, spec_id: &str) {
        if let Some(state) = self.state.upgrade() {
            let mut s = state.lock().await;
            s.failures.remove(spec_id);
            s.alerted.remove(spec_id);
        }
    }
}

/// Public handle for the bot. Owns the background task; dropping it does NOT
/// stop the loop — call [`ResearchScheduler::shutdown`] for that.
#[allow(dead_code)] // some methods only used by tests / future surfaces
pub struct ResearchScheduler {
    notify: Arc<Notify>,
    shutdown: Arc<Notify>,
    config: SchedulerConfig,
}

#[allow(dead_code)]
impl ResearchScheduler {
    pub fn config(&self) -> &SchedulerConfig {
        &self.config
    }

    /// Start the scheduler. Returns the scheduler handle and the
    /// [`SchedulerHook`] that callers must register on the [`AgentCore`] via
    /// [`AgentCore::set_scheduler_hook`].
    pub fn start(
        core: Weak<AgentCore>,
        config: SchedulerConfig,
    ) -> (Arc<Self>, Arc<dyn SchedulerHook>) {
        Self::start_with_notifier(core, config, Arc::new(NoopNotifier))
    }

    /// Same as [`Self::start`] but lets the caller plug in a [`TaskNotifier`]
    /// so failure alerts can be pushed (e.g. into a Telegram chat). Production
    /// uses this entry point; tests use [`Self::start`].
    pub fn start_with_notifier(
        core: Weak<AgentCore>,
        config: SchedulerConfig,
        notifier: Arc<dyn TaskNotifier>,
    ) -> (Arc<Self>, Arc<dyn SchedulerHook>) {
        let notify = Arc::new(Notify::new());
        let shutdown = Arc::new(Notify::new());

        // We keep the AgentCore semaphore connection (reused for manual /
        // LLM-driven runs) but the *scheduling* concurrency cap is the
        // hard ceiling on `running` map size.
        let semaphore: Arc<Semaphore> = match core.upgrade() {
            Some(c) => c.research_run_permits(),
            None => Arc::new(Semaphore::new(config.max_concurrent_runs.max(1))),
        };

        let state = Arc::new(Mutex::new(SchedulerState::default()));

        let scheduler = Arc::new(Self {
            notify: notify.clone(),
            shutdown: shutdown.clone(),
            config: config.clone(),
        });

        let hook: Arc<dyn SchedulerHook> = Arc::new(ResearchSchedulerHook {
            notify: notify.clone(),
            state: Arc::downgrade(&state),
        });

        let loop_notify = notify.clone();
        let loop_shutdown = shutdown.clone();
        // SAFETY against silent loop death: wrap `run_loop` in a panic-catching
        // supervisor. Without this, a `tokio::spawn(run_loop)` whose body
        // panics would log via tokio and detach — the bot would keep handling
        // /commands but no research would ever fire again, and nobody would
        // know until users complained. The supervisor restarts the loop and
        // sends a `notify_supervisor_panic` so operators see the incident.
        // F1: single supervisor primitive. Wrap `run_loop` in
        // `spawn_supervised_with_opts` so a panic restarts the loop
        // (with backoff) instead of silently detaching, and route
        // panic events through the existing TaskNotifier via a
        // tiny adapter. Constant 5s backoff matches the previous
        // `supervised_run_loop` behaviour.
        let panic_hook: std::sync::Arc<dyn crate::supervised::PanicHook> =
            std::sync::Arc::new(NotifierPanicHook(notifier.clone()));
        let _supervisor = crate::supervised::spawn_supervised_with_opts(
            crate::supervised::SupervisorOptions {
                name: "research",
                backoff: crate::supervised::Backoff {
                    initial: Duration::from_secs(5),
                    max: Duration::from_secs(5),
                    multiplier: 1,
                },
                // Scheduler-internal shutdown is signalled via the
                // `Arc<Notify>` already passed to `run_loop`; when
                // it fires, `run_loop` returns Ok and our supervisor
                // sees a clean exit. So an externally-never-cancelled
                // token is the right choice here — we don't want the
                // supervisor to abort `run_loop` mid-iteration; the
                // loop's own graceful-shutdown sweep MUST run.
                shutdown: tokio_util::sync::CancellationToken::new(),
                max_attempts: None,
                panic_hook: Some(panic_hook),
            },
            move |_inner_token| {
                let core = core.clone();
                let config = config.clone();
                let semaphore = semaphore.clone();
                let state = state.clone();
                let notifier = notifier.clone();
                let loop_notify = loop_notify.clone();
                let loop_shutdown = loop_shutdown.clone();
                async move {
                    lifecycle::run_loop(
                        core,
                        config,
                        semaphore,
                        state,
                        notifier,
                        loop_notify,
                        loop_shutdown,
                    )
                    .await;
                    Ok::<(), anyhow::Error>(())
                }
            },
        );

        (scheduler, hook)
    }

    /// Wake the scheduler loop immediately (used by tests).
    pub fn poke(&self) {
        self.notify.notify_one();
    }

    /// Stop the loop gracefully. Idempotent.
    pub fn shutdown(&self) {
        self.shutdown.notify_waiters();
    }
}

#[derive(Default)]
struct SchedulerState {
    /// Last *finished* run timestamp per spec (cache populated lazily from
    /// `runs.jsonl` on first sighting, then maintained in-memory).
    last_runs: HashMap<String, DateTime<Utc>>,
    /// In-flight runs. Bounded by `SchedulerConfig.max_concurrent_runs`.
    running: HashMap<String, RunningHandle>,
    /// Consecutive failure streak per spec. Reset on success and on
    /// auto-pause; *not* reset on alert (so the auto-pause threshold
    /// can keep counting).
    failures: HashMap<String, u32>,
    /// Spec ids for which we've already emitted a `notify_failure`
    /// alert during the current failure streak. Cleared on success
    /// or auto-pause. Prevents alert spam when alerts and auto-pause
    /// thresholds differ (alert at 3, pause at 5 → one alert at the
    /// 3rd failure, then silence until pause at the 5th).
    alerted: HashSet<String>,
    /// One-shot `run_at` triggers we already fired in this process and
    /// for which we've cleared the field on disk. Prevents a re-fire if
    /// the disk write is racing with the next sweep.
    one_shot_fired: HashSet<String>,
}

#[cfg(test)]
#[path = "tests.rs"]
mod tests;
