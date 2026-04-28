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
use naked_core::ResearchPatch;
use naked_core::AgentCore;
use naked_core::research::{
    Inflight, ResearchSpec, ResearchStore, RunState, SchedulerEvent, SchedulerHook, StopReason,
};
use tokio::sync::{Mutex, Notify, Semaphore};
use tokio_util::sync::CancellationToken;

use crate::cron_util::next_cron_after;
use tokio::task::JoinHandle;
use tokio::time::{Instant, sleep};

/// Configuration knobs for [`ResearchScheduler`]. Defaults match the prod TG
/// bot: scan every 30s, allow five runs at a time, time out individual runs
/// at 10 minutes, alert after 3 consecutive failures.
#[derive(Debug, Clone)]
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
    async fn notify_failure(&self, spec: &ResearchSpec, consecutive_failures: u32, last_error: &str);
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
        let supervisor_notifier = notifier.clone();
        let supervisor_backoff = Duration::from_secs(5);
        tokio::spawn(async move {
            supervised_run_loop("research", supervisor_notifier, supervisor_backoff, move || {
                let core = core.clone();
                let config = config.clone();
                let semaphore = semaphore.clone();
                let state = state.clone();
                let notifier = notifier.clone();
                let loop_notify = loop_notify.clone();
                let loop_shutdown = loop_shutdown.clone();
                async move {
                    run_loop(
                        core,
                        config,
                        semaphore,
                        state,
                        notifier,
                        loop_notify,
                        loop_shutdown,
                    )
                    .await;
                }
            })
            .await;
        });

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

async fn run_loop(
    core: Weak<AgentCore>,
    config: SchedulerConfig,
    semaphore: Arc<Semaphore>,
    state: Arc<Mutex<SchedulerState>>,
    notifier: Arc<dyn TaskNotifier>,
    notify: Arc<Notify>,
    shutdown: Arc<Notify>,
) {
    tracing::info!(
        tick_secs = config.tick_interval.as_secs(),
        max_concurrent = config.max_concurrent_runs,
        timeout_secs = config.task_timeout.as_secs(),
        max_retries = config.max_retries_before_alert,
        max_resurrect = config.max_resurrection_attempts,
        heartbeat_budget_secs = config.heartbeat_budget.as_secs(),
        verify = config.verify_by_default,
        "research scheduler started"
    );

    // Boot pass: clean up stale `*.tmp` files from interrupted
    // atomic-writes, then resurrect or finalise tasks that were
    // running when the previous process died. Done once;
    // subsequent ticks deal only with newly-stuck tasks via the
    // sweep.
    if let Some(core_arc) = core.upgrade() {
        purge_stale_tmp_files(&core_arc.research_store()).await;
        if let Err(e) = resurrect_at_boot(&core_arc, &state, &notifier, &config).await {
            tracing::warn!("scheduler boot resurrection failed: {e:#}");
        }
        // Boot-time purge of stale terminal inflights. Cheap and gives
        // operators a clean slate immediately after a long downtime
        // (when many terminal records may have aged past the retention
        // window). Subsequent purges happen inside the main loop on
        // `inflight_purge_interval`.
        purge_terminal_inflight_now(&core_arc.research_store(), &config).await;
    }

    let mut last_inflight_purge = Instant::now();
    loop {
        let woke_for_shutdown = tokio::select! {
            _ = sleep_until_next(config.tick_interval) => false,
            _ = notify.notified() => false,
            _ = shutdown.notified() => true,
        };
        if woke_for_shutdown {
            tracing::info!("research scheduler shutting down");
            // Best-effort: abort every in-flight worker and flip its
            // ledger to `Failed("shutdown")` so the next process'
            // resurrection pass picks them up instead of leaving
            // them as `Running` forever.
            if let Some(core_arc) = core.upgrade() {
                let store = core_arc.research_store();
                let aborted = shutdown_running_tasks(&state, &store).await;
                if !aborted.is_empty() {
                    tracing::info!(
                        count = aborted.len(),
                        ids = ?aborted,
                        "scheduler shutdown: aborted in-flight workers"
                    );
                }
            }
            return;
        }

        let Some(core_arc) = core.upgrade() else {
            tracing::info!("agent core dropped; scheduler exiting");
            // Same cleanup path as graceful shutdown: if the core was
            // dropped while runs were live, mark them Failed so we don't
            // strand them as `Running` on disk.
            let dummy_store_lookup: Option<Arc<dyn ResearchStore>> = None;
            if let Some(store) = dummy_store_lookup {
                let _ = shutdown_running_tasks(&state, &store).await;
            } else {
                // No store handle available — at least abort the
                // futures so they stop burning CPU/network.
                let mut s = state.lock().await;
                for (_, h) in s.running.drain() {
                    h.handle.abort();
                }
            }
            return;
        };
        if let Err(e) = scan_and_dispatch(&core_arc, &semaphore, &state, &notifier, &config).await {
            tracing::warn!("scheduler scan failed: {e:#}");
        }

        // Periodic purge of stale terminal inflights. Disabled when
        // either knob is `Duration::ZERO`. We piggy-back on the existing
        // tick instead of a separate task to keep the lifetime story
        // simple (one cancel point on shutdown).
        if config.inflight_purge_interval > Duration::ZERO
            && config.inflight_terminal_retention > Duration::ZERO
            && last_inflight_purge.elapsed() >= config.inflight_purge_interval
        {
            purge_terminal_inflight_now(&core_arc.research_store(), &config).await;
            last_inflight_purge = Instant::now();
        }
    }
}

/// Run a single pass of [`ResearchStore::purge_terminal_inflight`]
/// using the configured retention. Logs the count at debug level.
/// Errors are downgraded to a warn — a failed purge is never fatal
/// (the on-disk records are inert).
async fn purge_terminal_inflight_now(
    store: &Arc<dyn ResearchStore>,
    config: &SchedulerConfig,
) {
    if config.inflight_terminal_retention == Duration::ZERO {
        return;
    }
    let retention = match chrono::Duration::from_std(config.inflight_terminal_retention) {
        Ok(d) => d,
        Err(e) => {
            tracing::warn!(
                "inflight_terminal_retention out of chrono::Duration range: {e}; skipping purge"
            );
            return;
        }
    };
    match store.purge_terminal_inflight(chrono::Utc::now(), retention).await {
        Ok(0) => {}
        Ok(n) => tracing::debug!(removed = n, "purged stale terminal inflight records"),
        Err(e) => tracing::warn!("purge_terminal_inflight failed: {e:#}"),
    }
}

/// On startup, walk every spec's `inflight.json`. For each
/// non-terminal record we either:
/// * mark `Failed` (paused spec, or resurrection cap reached), or
/// * re-tag as `Scheduled + scheduled_after_resurrection = true` so
///   the next dispatch tick picks it up under the same
///   `max_concurrent_runs` cap as fresh runs.
///
/// Pull-model: NEVER spawns tasks here. That guarantees a 50-spec
/// crash-recovery can't punch through the concurrency cap and
/// flood the upstream LLM API.
async fn resurrect_at_boot(
    core: &Arc<AgentCore>,
    _state: &Arc<Mutex<SchedulerState>>,
    notifier: &Arc<dyn TaskNotifier>,
    config: &SchedulerConfig,
) -> anyhow::Result<()> {
    let store = core.research_store();
    rebuild_resurrection_queue(&store, notifier, config).await
}

/// Pull-model implementation, unit-testable against any
/// `Arc<dyn ResearchStore>`. Walks all non-terminal inflight records:
/// re-tags healthy ones for drain, finalises paused / cap-hit ones as
/// `Failed`, and emits an alert for the latter.
async fn rebuild_resurrection_queue(
    store: &Arc<dyn ResearchStore>,
    notifier: &Arc<dyn TaskNotifier>,
    config: &SchedulerConfig,
) -> anyhow::Result<()> {
    let stale = store.list_nonterminal_inflight().await?;
    if stale.is_empty() {
        return Ok(());
    }
    tracing::info!(
        count = stale.len(),
        "scheduler resurrection: found non-terminal inflight records"
    );
    for infl in stale {
        let spec = match store.load_spec(&infl.spec_id).await {
            Ok(s) => s,
            Err(e) => {
                tracing::warn!(spec = %infl.spec_id, "spec missing during resurrect: {e:#}; clearing inflight");
                let _ = store.clear_inflight(&infl.spec_id).await;
                continue;
            }
        };
        if spec.paused {
            tracing::info!(spec = %spec.id, "skipping resurrection: spec is paused");
            let mut final_inf = infl.clone();
            final_inf.mark_failed("paused before resurrection — pending attempt dropped");
            let _ = store.save_inflight(&spec.id, &final_inf).await;
            continue;
        }
        if config.max_resurrection_attempts == 0
            || infl.attempt >= config.max_resurrection_attempts
        {
            tracing::warn!(
                spec = %spec.id,
                attempt = infl.attempt,
                cap = config.max_resurrection_attempts,
                "resurrection cap hit; finalising as Failed"
            );
            let mut final_inf = infl.clone();
            final_inf.mark_failed(format!(
                "resurrection cap reached after {} attempt(s)",
                infl.attempt
            ));
            let _ = store.save_inflight(&spec.id, &final_inf).await;
            notifier
                .notify_failure(
                    &spec,
                    config.max_retries_before_alert.max(1),
                    final_inf.error.as_deref().unwrap_or("resurrection failed"),
                )
                .await;
            continue;
        }
        tracing::info!(
            spec = %spec.id,
            prev_attempt = infl.attempt,
            prev_state = ?infl.state,
            "tagging interrupted attempt for capacity-aware resurrection"
        );
        let next_attempt = infl.attempt.saturating_add(1);
        let mut next = infl.clone();
        next.mark_scheduled_for_resurrection(next_attempt);
        if let Err(e) = store.save_inflight(&spec.id, &next).await {
            tracing::warn!(spec = %spec.id, "failed to persist resurrection tag: {e:#}");
        }
    }
    Ok(())
}

/// Pure planner: from the set of non-terminal inflight records, pick
/// the spec ids that should drain through `spawn_task` *this tick*.
///
/// Selection rules:
/// * Only entries whose state is `Scheduled` AND
///   `scheduled_after_resurrection` is true.
/// * Skip specs already in `running` (in-memory map).
/// * Sort by `spec_id` for deterministic order (stable logs / tests).
/// * Cap at `available_slots`.
///
/// Returns `(spec_id, attempt)` pairs so the caller passes the
/// current attempt count into `spawn_task`.
fn plan_resurrection_drains(
    nonterminal: &[Inflight],
    running: &HashSet<String>,
    available_slots: usize,
) -> Vec<(String, u32)> {
    if available_slots == 0 {
        return Vec::new();
    }
    let mut candidates: Vec<(String, u32)> = nonterminal
        .iter()
        .filter(|i| i.state == RunState::Scheduled && i.scheduled_after_resurrection)
        .filter(|i| !running.contains(&i.spec_id))
        .map(|i| (i.spec_id.clone(), i.attempt))
        .collect();
    candidates.sort_by(|a, b| a.0.cmp(&b.0));
    candidates.truncate(available_slots);
    candidates
}

/// Tick-time orchestrator: drain the resurrection queue under the same
/// concurrency cap as fresh dispatches. Calls `spawn_task` for each
/// drained spec; staggers consecutive spawns to avoid a thundering
/// herd of LLM calls. Returns the number of slots consumed.
async fn drain_resurrection_queue(
    core: &Arc<AgentCore>,
    state: &Arc<Mutex<SchedulerState>>,
    notifier: &Arc<dyn TaskNotifier>,
    config: &SchedulerConfig,
) -> usize {
    let store = core.research_store();
    let nonterminal = match store.list_nonterminal_inflight().await {
        Ok(v) => v,
        Err(e) => {
            tracing::warn!("resurrection drain: list failed: {e:#}");
            return 0;
        }
    };
    if nonterminal.is_empty() {
        return 0;
    }
    let (running_ids, available_slots) = {
        let s = state.lock().await;
        let cap = config.max_concurrent_runs.saturating_sub(s.running.len());
        let ids: HashSet<String> = s.running.keys().cloned().collect();
        (ids, cap)
    };
    let plan = plan_resurrection_drains(&nonterminal, &running_ids, available_slots);
    let mut consumed = 0;
    for (id, attempt) in plan {
        let spec = match store.load_spec(&id).await {
            Ok(s) if !s.paused => s,
            Ok(_) => {
                tracing::debug!(spec = %id, "skipping drain: spec is paused");
                continue;
            }
            Err(e) => {
                tracing::warn!(spec = %id, "drain: load_spec failed: {e:#}");
                continue;
            }
        };
        tracing::info!(
            spec = %spec.id,
            attempt,
            "draining resurrection-tagged inflight under concurrency cap"
        );
        spawn_task(core, state, notifier, config, &spec, attempt).await;
        consumed += 1;
        if config.resurrection_stagger > Duration::ZERO {
            sleep(config.resurrection_stagger).await;
        }
    }
    consumed
}

/// Wrap a long-running async loop in a panic-catching supervisor.
///
/// Spawns the inner loop on a fresh tokio task; when that task panics, fires
/// `notifier.notify_supervisor_panic`, sleeps `backoff`, and re-spawns. Returns
/// only when the inner future completes cleanly or the inner task is cancelled
/// (we don't want to fight `JoinHandle::abort` from the outside).
///
/// The supervisor also logs every panic at `error!` level so observability
/// never depends solely on the notifier hook.
///
/// `make_loop_body` must be `FnMut` because we re-invoke it for every
/// restart (each restart needs a fresh future).
///
/// SOLID: this is the single supervision primitive. `run_loop` plugs into it
/// without knowing supervision details; tests plug in fake bodies the same way.
pub async fn supervised_run_loop<F, Fut>(
    name: &'static str,
    notifier: Arc<dyn TaskNotifier>,
    backoff: Duration,
    make_loop_body: F,
) where
    F: FnMut() -> Fut + Send + 'static,
    Fut: std::future::Future<Output = ()> + Send + 'static,
{
    supervised_run_loop_capped(name, notifier, backoff, u32::MAX, make_loop_body).await
}

/// Same as [`supervised_run_loop`] but with a hard cap on restart attempts.
/// Used by tests to keep pathological-panic scenarios bounded; production
/// uses the uncapped variant.
pub async fn supervised_run_loop_capped<F, Fut>(
    name: &'static str,
    notifier: Arc<dyn TaskNotifier>,
    backoff: Duration,
    max_attempts: u32,
    mut make_loop_body: F,
) where
    F: FnMut() -> Fut + Send + 'static,
    Fut: std::future::Future<Output = ()> + Send + 'static,
{
    let mut attempt: u32 = 0;
    loop {
        attempt += 1;
        let body = make_loop_body();
        let handle = tokio::spawn(body);
        match handle.await {
            Ok(()) => return,
            Err(je) if je.is_cancelled() => {
                tracing::info!(loop_name = %name, "supervised loop cancelled");
                return;
            }
            Err(je) => {
                let panic_msg = format_panic_payload(&je);
                let details = format!(
                    "scheduler '{name}' loop panicked (attempt {attempt}/{cap}): {panic_msg}",
                    cap = if max_attempts == u32::MAX {
                        "∞".to_string()
                    } else {
                        max_attempts.to_string()
                    }
                );
                tracing::error!("{details}");
                notifier.notify_supervisor_panic(&details).await;
                if attempt >= max_attempts {
                    tracing::error!(
                        loop_name = %name,
                        attempts = attempt,
                        "supervised loop reached restart cap; giving up"
                    );
                    return;
                }
                if backoff > Duration::ZERO {
                    tokio::time::sleep(backoff).await;
                }
            }
        }
    }
}

fn format_panic_payload(je: &tokio::task::JoinError) -> String {
    // `JoinError::into_panic` would consume; we want a borrowed inspection.
    // Fall back to Debug — for `panic!("msg")` this includes the message.
    format!("{je:?}")
}

/// Graceful-shutdown sweep: abort every entry in the in-memory
/// `running` map and flip the corresponding on-disk inflight ledger
/// to `Failed("shutdown")` so the next process boot resurrects the
/// task instead of leaving it as `Running` forever.
///
/// Skips inflight records that already reached a terminal state on
/// disk — this can happen if the worker future crossed the finish
/// line in the same millisecond shutdown was requested. We trust
/// the worker's own write in that case.
///
/// Returns the spec ids actually aborted (for logging/test).
async fn shutdown_running_tasks(
    state: &Arc<Mutex<SchedulerState>>,
    store: &Arc<dyn ResearchStore>,
) -> Vec<String> {
    let drained: Vec<(String, RunningHandle)> = {
        let mut s = state.lock().await;
        s.running.drain().collect()
    };
    let mut aborted_ids = Vec::with_capacity(drained.len());
    for (id, handle) in drained {
        // Issue cooperative cancel first; abort is the safety net
        // for anything not yet at an `await` point.
        handle.cancel.cancel();
        handle.handle.abort();
        match store.load_inflight(&id).await {
            Ok(Some(mut infl)) if !infl.state.is_terminal() => {
                infl.mark_failed(
                    "scheduler shutdown — worker aborted before completion (will be \
                     resurrected on next boot)",
                );
                if let Err(e) = store.save_inflight(&id, &infl).await {
                    tracing::warn!(
                        spec = %id,
                        "shutdown sweep: failed to persist Failed inflight: {e:#}"
                    );
                }
            }
            Ok(Some(_terminal)) => {
                tracing::debug!(
                    spec = %id,
                    "shutdown sweep: inflight already terminal, leaving as-is"
                );
            }
            Ok(None) => {
                tracing::debug!(
                    spec = %id,
                    "shutdown sweep: no inflight record on disk to update"
                );
            }
            Err(e) => {
                tracing::warn!(
                    spec = %id,
                    "shutdown sweep: failed to load inflight: {e:#}"
                );
            }
        }
        aborted_ids.push(id);
    }
    aborted_ids
}

/// Best-effort cleanup of half-written `*.tmp` files left over
/// from a `tmp + rename` atomic-write that was interrupted by a
/// process kill. These can otherwise pile up in research
/// directories indefinitely.
async fn purge_stale_tmp_files(store: &Arc<dyn ResearchStore>) {
    let Some(root) = store.fs_root() else {
        // Non-FS backend (RAM-only test fake) → nothing to do.
        return;
    };
    let mut count = 0u32;
    let mut total_bytes = 0u64;
    let Ok(mut dir) = tokio::fs::read_dir(root).await else {
        return;
    };
    while let Ok(Some(entry)) = dir.next_entry().await {
        let Ok(ft) = entry.file_type().await else {
            continue;
        };
        let path = entry.path();
        if ft.is_file() {
            // Root-level `*.tmp` files. `FsResearchStore::atomic_write`
            // can leave these behind for `scheduler.lock`-adjacent
            // writes (e.g. a future `root.json` summary): a crash
            // between `fs::write(tmp)` and `fs::rename(tmp, final)`
            // strands the half-written `.tmp`. We sweep them here
            // rather than letting them accumulate; never touch the
            // lock file itself (no `.tmp` extension) or anything
            // without `.tmp`.
            if path.extension().is_some_and(|ext| ext == "tmp")
                && let Some(removed_bytes) = try_remove_tmp(&path).await
            {
                count += 1;
                total_bytes = total_bytes.saturating_add(removed_bytes);
            }
            continue;
        }
        if !ft.is_dir() {
            continue;
        }
        let spec_dir = path;
        let Ok(mut sub) = tokio::fs::read_dir(&spec_dir).await else {
            continue;
        };
        while let Ok(Some(file)) = sub.next_entry().await {
            let path = file.path();
            if path.extension().is_some_and(|ext| ext == "tmp")
                && let Some(removed_bytes) = try_remove_tmp(&path).await
            {
                count += 1;
                total_bytes = total_bytes.saturating_add(removed_bytes);
            }
        }
    }
    if count > 0 {
        tracing::info!(
            removed = count,
            bytes = total_bytes,
            "scheduler boot: purged stale *.tmp files from previous crash"
        );
    }
}

/// Best-effort delete of a `*.tmp` file. Returns `Some(bytes)` when
/// the file was successfully removed (and we know its size), `None`
/// on any error. Logging is intentionally `debug!` because a stale
/// tmp from a previous boot is the *normal* case here.
async fn try_remove_tmp(path: &std::path::Path) -> Option<u64> {
    let bytes = tokio::fs::metadata(path).await.ok().map(|m| m.len()).unwrap_or(0);
    match tokio::fs::remove_file(path).await {
        Ok(()) => Some(bytes),
        Err(e) => {
            tracing::debug!(?path, "purge_stale_tmp_files: {e:#}");
            None
        }
    }
}

async fn sleep_until_next(d: Duration) {
    let deadline = Instant::now() + d;
    sleep(deadline.saturating_duration_since(Instant::now())).await;
}

/// Pure decision predicate: should the scheduler launch a run for this spec
/// right now? Honours the four-tier priority documented at the top of this
/// module.
///
/// `last_run` is the timestamp of the most recent *finished* run for this
/// spec (any outcome). `now` is the current wall clock.
pub fn is_due(spec: &ResearchSpec, last_run: Option<DateTime<Utc>>, now: DateTime<Utc>) -> bool {
    if spec.paused {
        return false;
    }

    // 1. One-shot at-time trigger.
    if let Some(at) = spec.run_at
        && now >= at
    {
        // Did we already run *after* the run_at moment? If yes, we've
        // fired this one-shot already (the cleanup write to disk may
        // not have landed yet, hence the in-memory `one_shot_fired`
        // tracking inside `scan_and_dispatch`).
        match last_run {
            Some(t) if t >= at => {} // already fired, fall through to other triggers
            _ => return true,
        }
    }

    // 2. Recurring cron trigger.
    if let Some(expr) = spec.cron.as_deref() {
        let anchor = last_run.unwrap_or_else(|| now - chrono::Duration::days(365 * 10));
        if let Some(next) = next_cron_after(expr, anchor)
            && next <= now
        {
            return true;
        }
    }

    // 3. Legacy interval trigger.
    if let Some(interval) = spec.interval_seconds {
        if interval == 0 {
            return false;
        }
        return match last_run {
            Some(t) => (now - t).num_seconds() >= interval as i64,
            None => true,
        };
    }

    false
}

/// Pure scheduling decision. Returns the list of spec ids that should be
/// dispatched on this tick, capped at `available_slots`. Specs already
/// running are excluded.
pub fn plan_dispatches(
    specs: &[ResearchSpec],
    last_runs: &HashMap<String, DateTime<Utc>>,
    running: &HashSet<String>,
    available_slots: usize,
    now: DateTime<Utc>,
) -> Vec<String> {
    let mut out = Vec::new();
    if available_slots == 0 {
        return out;
    }
    for spec in specs {
        if out.len() >= available_slots {
            break;
        }
        if running.contains(&spec.id) {
            continue;
        }
        let last = last_runs.get(&spec.id).copied();
        if !is_due(spec, last, now) {
            continue;
        }
        out.push(spec.id.clone());
    }
    out
}

async fn scan_and_dispatch(
    core: &Arc<AgentCore>,
    _semaphore: &Arc<Semaphore>,
    state: &Arc<Mutex<SchedulerState>>,
    notifier: &Arc<dyn TaskNotifier>,
    config: &SchedulerConfig,
) -> anyhow::Result<()> {
    let store = core.research_store();
    let specs = store.list_specs().await?;
    let now = Utc::now();

    // ── Sweep phase: free slots from completed / timed-out runs ────────────
    sweep_running(state, notifier, &specs, config, &store, Some(core)).await;

    // ── Resurrection drain (pull-model) ───────────────────────────────────
    // Drain re-queued inflights BEFORE planning fresh dispatches so a
    // crash-recovered job has the same priority as a freshly-due one,
    // and so resurrection respects the same `max_concurrent_runs` cap.
    let _drained = drain_resurrection_queue(core, state, notifier, config).await;

    // ── Hydrate last-run cache ────────────────────────────────────────────
    {
        let mut s = state.lock().await;
        for spec in &specs {
            if !s.last_runs.contains_key(&spec.id)
                && let Some(t) = lookup_last_run_on_disk(&store, &spec.id).await
            {
                s.last_runs.insert(spec.id.clone(), t);
            }
        }
    }

    // ── Plan ───────────────────────────────────────────────────────────────
    let plan = {
        let s = state.lock().await;
        let available_slots = config
            .max_concurrent_runs
            .saturating_sub(s.running.len());
        let running_set: HashSet<String> = s.running.keys().cloned().collect();
        plan_dispatches(&specs, &s.last_runs, &running_set, available_slots, now)
    };
    if plan.is_empty() {
        return Ok(());
    }

    // ── Dispatch ───────────────────────────────────────────────────────────
    for id in plan {
        let spec = match specs.iter().find(|s| s.id == id) {
            Some(s) => s.clone(),
            None => continue,
        };

        // If this dispatch was triggered by a one-shot `run_at`, clear the
        // field on disk *before* spawning so a crash mid-run doesn't cause a
        // re-fire on next boot.
        if let Some(at) = spec.run_at
            && now >= at
        {
            let patch = ResearchPatch {
                run_at: Some(None),
                ..Default::default()
            };
            if let Err(e) = core.update_research(&spec.id, patch).await {
                tracing::warn!(spec = %spec.id, "failed to clear run_at: {e:#}");
            }
            state.lock().await.one_shot_fired.insert(spec.id.clone());
        }

        // Pre-set last_run so other planning passes wait one full cycle.
        state.lock().await.last_runs.insert(spec.id.clone(), now);

        spawn_task(core, state, notifier, config, &spec, 1).await;
    }
    Ok(())
}

/// Spawn the worker for one scheduling attempt. Owns the on-disk
/// state-machine: writes `Scheduled` before spawn, has the worker
/// flip to `Running`, and finalises to `Completed` / `Failed` from
/// inside the spawned future.
///
/// `attempt` is `1` for a fresh dispatch and `prev_attempt + 1`
/// when called from `resurrect_at_boot`.
async fn spawn_task(
    core: &Arc<AgentCore>,
    state: &Arc<Mutex<SchedulerState>>,
    notifier: &Arc<dyn TaskNotifier>,
    config: &SchedulerConfig,
    spec: &ResearchSpec,
    attempt: u32,
) {
    let store = core.research_store();
    let timeout = spec
        .task_timeout_seconds
        .map(Duration::from_secs)
        .unwrap_or(config.task_timeout);
    let started = Instant::now();

    // Phase 1: persist `Scheduled` so we survive a crash *before*
    // the worker future actually starts running.
    let mut infl = Inflight::scheduled(spec.id.clone(), attempt);
    let attempt_id = infl.attempt_id.clone();
    if let Err(e) = store.save_inflight(&spec.id, &infl).await {
        tracing::warn!(spec = %spec.id, "failed to persist Scheduled inflight: {e:#}");
    }

    let core_clone = core.clone();
    let state_clone = state.clone();
    let notifier_clone = notifier.clone();
    let config_clone = config.clone();
    let spec_for_task = spec.clone();
    let id_for_task = spec.id.clone();
    let attempt_id_for_task = attempt_id.clone();
    // Cooperative cancel handle. Stored in the `RunningHandle` for
    // the sweep loop and a clone is moved into the worker future so
    // the coordinator can observe it via `run_once_with_cancel`.
    let cancel = CancellationToken::new();
    let cancel_for_task = cancel.clone();
    let handle = tokio::spawn(async move {
        // Phase 2: flip ledger to `Running` immediately.
        let store_inner = core_clone.research_store();
        infl.mark_running();
        if let Err(e) = store_inner.save_inflight(&id_for_task, &infl).await {
            tracing::warn!(spec = %id_for_task, "failed to persist Running inflight: {e:#}");
        }

        tracing::info!(
            spec = %id_for_task,
            attempt = attempt,
            attempt_id = %attempt_id_for_task,
            verify = config_clone.verify_by_default,
            "scheduler launching research run"
        );
        // Keep the full RunReport so we can tell a `Cancelled`
        // stop reason apart from a real success — the scheduler
        // treats cancellation as a *failure* (it always means the
        // sweep timeout fired), not as a normal completion.
        let result: Result<(String, StopReason), _> = if config_clone.verify_by_default {
            core_clone
                .clone()
                .run_research_verified_with_cancel(
                    &id_for_task,
                    config_clone.max_verification_rounds,
                    cancel_for_task,
                )
                .await
                .map(|v| (v.last_run.run_id, v.last_run.stop_reason))
        } else {
            core_clone
                .clone()
                .run_research_with_cancel(&id_for_task, cancel_for_task)
                .await
                .map(|r| (r.run_id, r.stop_reason))
        };
        // Phase 3: stamp terminal state on the ledger.
        match &result {
            Ok((run_id, stop)) if !matches!(stop, StopReason::Cancelled) => {
                tracing::info!(spec = %id_for_task, %run_id, ?stop, "scheduler run complete");
                infl.mark_completed(Some(run_id.clone()));
            }
            Ok((run_id, _cancelled)) => {
                tracing::warn!(
                    spec = %id_for_task,
                    %run_id,
                    "scheduler run cancelled by sweep timeout — recording as Failed"
                );
                infl.run_id = Some(run_id.clone());
                infl.mark_failed("cancelled by sweep timeout");
            }
            Err(e) => {
                tracing::warn!(spec = %id_for_task, "scheduler run failed: {e:#}");
                infl.mark_failed(format!("{e:#}"));
            }
        }
        if let Err(e) = store_inner.save_inflight(&id_for_task, &infl).await {
            tracing::warn!(spec = %id_for_task, "failed to persist terminal inflight: {e:#}");
        }
        let outcome = match &result {
            Ok((_, stop)) if !matches!(stop, StopReason::Cancelled) => RunOutcome::Success,
            Ok((_, _)) => RunOutcome::Failure("cancelled by sweep timeout".into()),
            Err(e) => RunOutcome::Failure(format!("{e:#}")),
        };
        apply_outcome(
            &state_clone,
            &notifier_clone,
            Some(&core_clone),
            &spec_for_task,
            &outcome,
            &config_clone,
        )
        .await;
        outcome
    });
    state.lock().await.running.insert(
        spec.id.clone(),
        RunningHandle {
            started_at: started,
            timeout,
            handle,
            cancel,
            cancel_requested_at: None,
            attempt_id,
        },
    );
}

/// Walk the `running` map: drop entries whose JoinHandle finished, treat
/// over-budget entries as timed-out failures (drop the slot, let the underlying
/// task finish naturally), refresh the heartbeat on the on-disk inflight
/// ledger for live entries, and update the failure counters accordingly.
async fn sweep_running(
    state: &Arc<Mutex<SchedulerState>>,
    notifier: &Arc<dyn TaskNotifier>,
    specs: &[ResearchSpec],
    config: &SchedulerConfig,
    store: &Arc<dyn ResearchStore>,
    core: Option<&Arc<AgentCore>>,
) {
    // Two-stage timeout handling:
    //   * `to_drop`            — JoinHandle naturally finished, free slot.
    //   * `to_request_cancel`  — over budget for the *first* time; issue
    //                            cooperative `CancellationToken::cancel()`
    //                            but KEEP the slot reserved so the next
    //                            cron tick doesn't immediately spawn a
    //                            fresh duplicate while the worker is
    //                            still draining.
    //   * `to_hard_abort`      — cooperative cancel was issued more than
    //                            `cancel_grace_period` ago and the worker
    //                            is still alive; escalate to
    //                            `JoinHandle::abort()` and free the slot.
    let mut to_drop: Vec<String> = Vec::new();
    let mut to_request_cancel: Vec<String> = Vec::new();
    let mut to_hard_abort: Vec<String> = Vec::new();
    let mut to_heartbeat: Vec<String> = Vec::new();
    {
        let s = state.lock().await;
        for (id, h) in s.running.iter() {
            if h.handle.is_finished() {
                to_drop.push(id.clone());
            } else if let Some(req_at) = h.cancel_requested_at {
                if req_at.elapsed() > config.cancel_grace_period {
                    to_hard_abort.push(id.clone());
                }
                // else: still in grace window — leave slot reserved,
                // refresh heartbeat NOT needed (worker is winding down).
            } else if h.started_at.elapsed() > h.timeout {
                to_request_cancel.push(id.clone());
            } else {
                to_heartbeat.push(id.clone());
            }
        }
    }
    // Refresh heartbeats for healthy in-flight runs so external
    // observers can tell live tasks from frozen ones.
    for id in to_heartbeat {
        if let Ok(Some(mut infl)) = store.load_inflight(&id).await
            && infl.state == RunState::Running
        {
            infl.heartbeat();
            let _ = store.save_inflight(&id, &infl).await;
        }
    }
    // Stage 1: cooperative cancel for newly over-budget runs. We
    // re-acquire the lock as `&mut` so we can stamp
    // `cancel_requested_at` on the handle in place; the slot is NOT
    // released yet — the worker still owns it until either it
    // finishes on its own or the grace window expires.
    if !to_request_cancel.is_empty() && config.cancel_grace_period > Duration::ZERO {
        let mut s = state.lock().await;
        for id in &to_request_cancel {
            if let Some(h) = s.running.get_mut(id) {
                h.cancel.cancel();
                h.cancel_requested_at = Some(Instant::now());
                tracing::warn!(
                    spec = %id,
                    timeout_secs = h.timeout.as_secs(),
                    grace_secs = config.cancel_grace_period.as_secs(),
                    "scheduler task exceeded timeout — issued cooperative cancel; will hard-abort if worker doesn't honour it within grace window"
                );
            }
        }
    } else if !to_request_cancel.is_empty() {
        // Grace window disabled — escalate immediately.
        to_hard_abort.extend(to_request_cancel.into_iter());
    }
    // Stage 2: hard-abort (after grace OR with grace disabled).
    for id in to_hard_abort {
        let removed = state.lock().await.running.remove(&id);
        let timeout_secs = removed
            .as_ref()
            .map(|h| h.timeout.as_secs())
            .unwrap_or_else(|| config.task_timeout.as_secs());
        let grace_secs = config.cancel_grace_period.as_secs();
        // CRITICAL loop-prevention: hard-abort the runaway
        // JoinHandle. Without this, dropping the handle
        // detaches the future — it keeps consuming HTTP/LLM
        // budget while the next tick happily spawns a fresh
        // attempt for the same spec (because the in-memory
        // slot is already free), creating a 1 → 2 → 4 →
        // worker-fan-out death-spiral. abort() schedules a
        // cancellation point at the next .await; since the
        // research coordinator is fully async (HTTP, file
        // IO, LLM streaming), this lands in milliseconds in
        // practice.
        if let Some(h) = removed.as_ref() {
            // Belt + suspenders: also cancel the token in case the
            // worker reaches an `await` *between* drop(handle) and
            // the next `select!` poll — `cancel.cancel()` is idempotent.
            h.cancel.cancel();
            h.handle.abort();
        }
        if grace_secs == 0 {
            tracing::warn!(
                spec = %id,
                timeout_secs,
                "scheduler task exceeded timeout — aborting runaway worker (cooperative grace disabled)"
            );
        } else {
            tracing::error!(
                spec = %id,
                timeout_secs,
                grace_secs,
                "scheduler task ignored cooperative cancel after grace window — escalating to hard abort"
            );
        }
        // Ensure the on-disk ledger reflects the timeout
        // even though the aborted worker won't get to
        // overwrite it.
        if let Ok(Some(mut infl)) = store.load_inflight(&id).await
            && !infl.state.is_terminal()
        {
            infl.mark_failed(format!(
                "task timeout (> {timeout_secs}s) — worker hard-aborted after {grace_secs}s grace"
            ));
            let _ = store.save_inflight(&id, &infl).await;
        }
        if let Some(spec) = specs.iter().find(|s| s.id == id) {
            let outcome =
                RunOutcome::Failure(format!("task timeout (> {timeout_secs}s, hard-abort)"));
            apply_outcome(state, notifier, core, spec, &outcome, config).await;
        }
        drop(removed);
    }
    // Stage 3: naturally-finished slots. Successful / errored / cooperatively
    // cancelled handles already updated their own counters via
    // `apply_outcome` from inside the spawned future. We just drop
    // the handle here.
    for id in to_drop {
        let removed = state.lock().await.running.remove(&id);
        drop(removed);
    }

    // Independent pass: detect ledgers whose `last_heartbeat` is
    // way past budget but the in-memory `running` map doesn't know
    // about them (e.g. previous process died and the boot
    // resurrection somehow missed it). Finalise as Failed.
    let now = Utc::now();
    let budget = chrono::Duration::from_std(config.heartbeat_budget)
        .unwrap_or_else(|_| chrono::Duration::seconds(120));
    let running_ids: HashSet<String> = state.lock().await.running.keys().cloned().collect();
    for spec in specs {
        if running_ids.contains(&spec.id) {
            continue;
        }
        let Ok(Some(mut infl)) = store.load_inflight(&spec.id).await else {
            continue;
        };
        if infl.state.is_terminal() {
            continue;
        }
        if infl.is_stale(now, budget) {
            tracing::warn!(
                spec = %spec.id,
                state = ?infl.state,
                attempt = infl.attempt,
                "kicking heartbeat-stale inflight (no in-memory worker) — finalising as Failed"
            );
            infl.mark_failed("heartbeat budget exceeded with no live worker");
            let _ = store.save_inflight(&spec.id, &infl).await;
        }
    }
}

/// Decision returned by [`evaluate_outcome`]. Pure value; `apply_outcome`
/// is the thin async shell that mutates state, calls the notifier, and
/// (for `AutoPause`) patches the spec on disk.
#[derive(Debug, PartialEq, Eq)]
enum FailurePolicy {
    /// Either a success, or a failure under both thresholds — log only.
    Quiet,
    /// First failure reaching `max_retries_before_alert` for this
    /// streak — emit a single `notify_failure`.
    AlertOnce { count: u32 },
    /// `auto_pause_after_failures` reached — patch `paused = true` on
    /// disk, emit a clearly-labelled alert, reset counters.
    AutoPause { count: u32 },
}

/// Pure decision over the failure counter. Easy to unit-test without
/// touching tokio, the store, or the notifier.
fn evaluate_outcome(prev_count: u32, alerted: bool, cfg: &SchedulerConfig) -> FailurePolicy {
    let next = prev_count;
    if cfg.auto_pause_after_failures > 0 && next >= cfg.auto_pause_after_failures {
        return FailurePolicy::AutoPause { count: next };
    }
    if cfg.max_retries_before_alert > 0
        && next >= cfg.max_retries_before_alert
        && !alerted
    {
        return FailurePolicy::AlertOnce { count: next };
    }
    FailurePolicy::Quiet
}

/// Update the in-memory failure counter for `spec` based on an outcome,
/// fire the notifier alert when the streak crosses the alert threshold
/// (once per streak), and auto-pause the spec on disk when the
/// `auto_pause_after_failures` threshold is hit. Auto-pause is the
/// loop breaker — without it a deterministic-failure spec would burn
/// LLM budget on every cron tick forever.
async fn apply_outcome(
    state: &Arc<Mutex<SchedulerState>>,
    notifier: &Arc<dyn TaskNotifier>,
    core: Option<&Arc<AgentCore>>,
    spec: &ResearchSpec,
    outcome: &RunOutcome,
    cfg: &SchedulerConfig,
) {
    let (decision, last_err): (FailurePolicy, String) = match outcome {
        RunOutcome::Success => {
            let mut s = state.lock().await;
            s.failures.remove(&spec.id);
            s.alerted.remove(&spec.id);
            s.last_runs.insert(spec.id.clone(), Utc::now());
            (FailurePolicy::Quiet, String::new())
        }
        RunOutcome::Failure(msg) => {
            let mut s = state.lock().await;
            let counter = s.failures.entry(spec.id.clone()).or_insert(0);
            *counter += 1;
            let count = *counter;
            let already_alerted = s.alerted.contains(&spec.id);
            let decision = evaluate_outcome(count, already_alerted, cfg);
            // Record alert / clear counter as appropriate before
            // releasing the lock so concurrent ticks don't double-fire.
            match &decision {
                FailurePolicy::AlertOnce { .. } => {
                    s.alerted.insert(spec.id.clone());
                }
                FailurePolicy::AutoPause { .. } => {
                    s.failures.remove(&spec.id);
                    s.alerted.remove(&spec.id);
                }
                FailurePolicy::Quiet => {}
            }
            (decision, msg.clone())
        }
    };

    match decision {
        FailurePolicy::Quiet => {}
        FailurePolicy::AlertOnce { count } => {
            tracing::warn!(
                spec = %spec.id,
                consecutive = count,
                "scheduler: alert threshold reached, notifying operator"
            );
            notifier.notify_failure(spec, count, &last_err).await;
        }
        FailurePolicy::AutoPause { count } => {
            tracing::warn!(
                spec = %spec.id,
                consecutive = count,
                "scheduler: auto-pausing spec after consecutive failures"
            );
            // Flip spec.paused = true on disk so cron stops
            // firing. `set_research_paused` also notifies the
            // SchedulerHook so cached planning state catches up
            // immediately.
            let mut paused_ok = true;
            // Reason carrier — surfaced by `/research ls` and
            // `/research state` so operators can tell auto-pauses
            // apart from manual `/research pause` without grepping
            // journalctl. Truncated to keep the on-disk JSON tidy.
            let reason = {
                let short_err: String = last_err.chars().take(200).collect();
                Some(format!(
                    "auto: {count} consecutive failures — last error: {short_err}"
                ))
            };
            if let Some(c) = core {
                if let Err(e) = c
                    .set_research_paused_with_reason(&spec.id, true, reason)
                    .await
                {
                    tracing::error!(
                        spec = %spec.id,
                        "auto-pause: failed to patch spec.paused: {e:#} \
                         (alert will still fire — consider manual /research pause)"
                    );
                    paused_ok = false;
                }
            } else {
                tracing::warn!(
                    spec = %spec.id,
                    "auto-pause: no AgentCore handle; sending alert without disk patch"
                );
                paused_ok = false;
            }
            let label = if paused_ok {
                format!(
                    "auto-paused after {count} consecutive failures · last error: {last_err}"
                )
            } else {
                format!(
                    "{count} consecutive failures (auto-pause patch FAILED — pause manually) · last error: {last_err}"
                )
            };
            notifier.notify_failure(spec, count, &label).await;
        }
    }
}

async fn lookup_last_run_on_disk(
    store: &Arc<dyn ResearchStore>,
    spec_id: &str,
) -> Option<DateTime<Utc>> {
    let runs = store.list_runs(spec_id, Some(1)).await.ok()?;
    runs.first().map(|r| r.finished_at)
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::Duration as ChronoDuration;
    use naked_core::research::SchedulerEvent;
    use std::sync::atomic::{AtomicUsize, Ordering};

    fn spec_with(id: &str, mutate: impl FnOnce(&mut ResearchSpec)) -> ResearchSpec {
        let mut s = ResearchSpec {
            id: id.into(),
            topic: format!("topic for {id}"),
            sources: vec![],
            interval_seconds: None,
            run_at: None,
            cron: None,
            task_timeout_seconds: None,
            session_id: None,
            chat_id: None,
            thread_id: None,
            provider: None,
            model: None,
            max_iterations: None,
            max_wall_seconds: None,
            created_at: Utc::now(),
            paused: false,
            pause_reason: None,
        };
        mutate(&mut s);
        s
    }

    // ── is_due: paused short-circuit ───────────────────────────────────────

    #[test]
    fn paused_specs_are_never_due() {
        let now = Utc::now();
        let spec = spec_with("a", |s| {
            s.paused = true;
            s.interval_seconds = Some(60);
            s.run_at = Some(now - ChronoDuration::hours(1));
            s.cron = Some("* * * * *".into());
        });
        assert!(!is_due(&spec, None, now));
        assert!(!is_due(&spec, Some(now - ChronoDuration::hours(2)), now));
    }

    // ── is_due: legacy interval (back-compat) ──────────────────────────────

    #[test]
    fn no_schedule_means_never_due() {
        let now = Utc::now();
        let spec = spec_with("a", |_| {});
        assert!(!is_due(&spec, None, now));
        let spec_zero = spec_with("a", |s| s.interval_seconds = Some(0));
        assert!(!is_due(&spec_zero, None, now));
    }

    #[test]
    fn never_run_specs_with_interval_fire_immediately() {
        let now = Utc::now();
        let spec = spec_with("a", |s| s.interval_seconds = Some(3600));
        assert!(is_due(&spec, None, now));
    }

    #[test]
    fn interval_fires_only_after_elapses() {
        let now = Utc::now();
        let spec = spec_with("a", |s| s.interval_seconds = Some(60));
        assert!(!is_due(&spec, Some(now - ChronoDuration::seconds(59)), now));
        assert!(is_due(&spec, Some(now - ChronoDuration::seconds(60)), now));
        assert!(is_due(&spec, Some(now - ChronoDuration::seconds(120)), now));
    }

    // ── is_due: run_at one-shot ────────────────────────────────────────────

    #[test]
    fn run_at_in_future_is_not_due() {
        let now = Utc::now();
        let spec = spec_with("a", |s| s.run_at = Some(now + ChronoDuration::minutes(5)));
        assert!(!is_due(&spec, None, now));
    }

    #[test]
    fn run_at_in_past_fires_when_no_prior_run() {
        let now = Utc::now();
        let spec = spec_with("a", |s| s.run_at = Some(now - ChronoDuration::seconds(10)));
        assert!(is_due(&spec, None, now));
    }

    #[test]
    fn run_at_does_not_re_fire_after_a_run_post_target() {
        // last_run is *after* run_at → we already fired the one-shot.
        let now = Utc::now();
        let target = now - ChronoDuration::minutes(10);
        let last = now - ChronoDuration::minutes(5);
        let spec = spec_with("a", |s| s.run_at = Some(target));
        assert!(!is_due(&spec, Some(last), now));
    }

    // ── is_due: cron ───────────────────────────────────────────────────────

    #[test]
    fn cron_every_minute_is_due_after_a_minute() {
        let now = Utc::now();
        let spec = spec_with("a", |s| s.cron = Some("* * * * *".into()));
        // last run a minute ago, "* * * * *" fires every minute → due
        assert!(is_due(&spec, Some(now - ChronoDuration::minutes(2)), now));
    }

    #[test]
    fn cron_invalid_expression_is_silently_skipped() {
        let now = Utc::now();
        let spec = spec_with("a", |s| s.cron = Some("definitely not cron".into()));
        assert!(!is_due(&spec, None, now));
    }

    // ── priority: at > cron > interval ─────────────────────────────────────

    #[test]
    fn run_at_takes_precedence_over_interval() {
        // Even if interval would say "wait", a past run_at fires.
        let now = Utc::now();
        let spec = spec_with("a", |s| {
            s.run_at = Some(now - ChronoDuration::seconds(10));
            s.interval_seconds = Some(99999); // would normally block
        });
        assert!(is_due(&spec, None, now));
    }

    // ── plan_dispatches ────────────────────────────────────────────────────

    #[test]
    fn plan_caps_at_available_slots() {
        let now = Utc::now();
        let specs = vec![
            spec_with("a", |s| s.interval_seconds = Some(1)),
            spec_with("b", |s| s.interval_seconds = Some(1)),
            spec_with("c", |s| s.interval_seconds = Some(1)),
        ];
        let last_runs = HashMap::new();
        let running = HashSet::new();
        let plan = plan_dispatches(&specs, &last_runs, &running, 2, now);
        assert_eq!(plan, vec!["a", "b"]);
    }

    #[test]
    fn plan_skips_already_running_specs() {
        let now = Utc::now();
        let specs = vec![
            spec_with("a", |s| s.interval_seconds = Some(1)),
            spec_with("b", |s| s.interval_seconds = Some(1)),
        ];
        let last_runs = HashMap::new();
        let mut running = HashSet::new();
        running.insert("a".into());
        let plan = plan_dispatches(&specs, &last_runs, &running, 5, now);
        assert_eq!(plan, vec!["b"]);
    }

    #[test]
    fn plan_returns_empty_when_no_slots() {
        let now = Utc::now();
        let specs = vec![spec_with("a", |s| s.interval_seconds = Some(1))];
        let last_runs = HashMap::new();
        let running = HashSet::new();
        let plan = plan_dispatches(&specs, &last_runs, &running, 0, now);
        assert!(plan.is_empty());
    }

    // ── apply_outcome / failure tracking ───────────────────────────────────

    struct CountingNotifier {
        calls: AtomicUsize,
    }
    #[async_trait]
    impl TaskNotifier for CountingNotifier {
        async fn notify_failure(
            &self,
            _spec: &ResearchSpec,
            _consecutive_failures: u32,
            _last_error: &str,
        ) {
            self.calls.fetch_add(1, Ordering::SeqCst);
        }
    }

    fn cfg_with(retries: u32, auto_pause: u32) -> SchedulerConfig {
        SchedulerConfig {
            max_retries_before_alert: retries,
            auto_pause_after_failures: auto_pause,
            ..SchedulerConfig::default()
        }
    }

    #[tokio::test]
    async fn alert_fires_once_per_streak_then_silent_until_pause() {
        let state = Arc::new(Mutex::new(SchedulerState::default()));
        let counting = Arc::new(CountingNotifier {
            calls: AtomicUsize::new(0),
        });
        let notifier: Arc<dyn TaskNotifier> = counting.clone();
        let spec = spec_with("x", |_| {});
        let cfg = cfg_with(3, 0); // alert at 3, auto-pause disabled

        for _ in 0..2 {
            apply_outcome(
                &state,
                &notifier,
                None,
                &spec,
                &RunOutcome::Failure("boom".into()),
                &cfg,
            )
            .await;
        }
        assert_eq!(counting.calls.load(Ordering::SeqCst), 0);

        apply_outcome(
            &state,
            &notifier,
            None,
            &spec,
            &RunOutcome::Failure("boom".into()),
            &cfg,
        )
        .await;
        assert_eq!(counting.calls.load(Ordering::SeqCst), 1);
        // Counter is NOT reset on alert — but the `alerted` set
        // means subsequent failures stay quiet (until success or
        // auto-pause).
        for _ in 0..5 {
            apply_outcome(
                &state,
                &notifier,
                None,
                &spec,
                &RunOutcome::Failure("boom".into()),
                &cfg,
            )
            .await;
        }
        assert_eq!(
            counting.calls.load(Ordering::SeqCst),
            1,
            "alerts should not spam — exactly one per streak"
        );
        assert_eq!(
            *state.lock().await.failures.get("x").unwrap(),
            8,
            "counter keeps climbing for the auto-pause threshold"
        );
    }

    #[tokio::test]
    async fn success_resets_failure_counter_and_alert_flag() {
        let state = Arc::new(Mutex::new(SchedulerState::default()));
        let notifier: Arc<dyn TaskNotifier> = Arc::new(NoopNotifier);
        let spec = spec_with("x", |_| {});
        let cfg = cfg_with(3, 0);
        for _ in 0..3 {
            apply_outcome(
                &state,
                &notifier,
                None,
                &spec,
                &RunOutcome::Failure("boom".into()),
                &cfg,
            )
            .await;
        }
        assert_eq!(*state.lock().await.failures.get("x").unwrap(), 3);
        assert!(state.lock().await.alerted.contains("x"));
        apply_outcome(&state, &notifier, None, &spec, &RunOutcome::Success, &cfg).await;
        assert!(!state.lock().await.failures.contains_key("x"));
        assert!(
            !state.lock().await.alerted.contains("x"),
            "alerted flag should reset on success"
        );
    }

    // ── auto-pause loop breaker ────────────────────────────────────────────

    #[test]
    fn evaluate_outcome_quiet_below_thresholds() {
        let cfg = cfg_with(3, 5);
        assert_eq!(evaluate_outcome(1, false, &cfg), FailurePolicy::Quiet);
        assert_eq!(evaluate_outcome(2, false, &cfg), FailurePolicy::Quiet);
    }

    #[test]
    fn evaluate_outcome_alerts_once_at_threshold() {
        let cfg = cfg_with(3, 5);
        assert_eq!(
            evaluate_outcome(3, false, &cfg),
            FailurePolicy::AlertOnce { count: 3 }
        );
        // Already alerted → quiet until pause.
        assert_eq!(evaluate_outcome(4, true, &cfg), FailurePolicy::Quiet);
    }

    #[test]
    fn evaluate_outcome_pauses_at_or_above_threshold() {
        let cfg = cfg_with(3, 5);
        assert_eq!(
            evaluate_outcome(5, true, &cfg),
            FailurePolicy::AutoPause { count: 5 }
        );
        assert_eq!(
            evaluate_outcome(7, true, &cfg),
            FailurePolicy::AutoPause { count: 7 }
        );
        // Pause takes precedence over alert when both would trigger.
        let cfg_eq = cfg_with(3, 3);
        assert_eq!(
            evaluate_outcome(3, false, &cfg_eq),
            FailurePolicy::AutoPause { count: 3 }
        );
    }

    /// Independent reference oracle for [`evaluate_outcome`]. Encodes
    /// the spec in a different shape than the implementation (early-out
    /// ladder) so a bug in either side surfaces as a mismatch rather
    /// than agreeing on a wrong answer. Precedence is intentional:
    /// AutoPause ranks above AlertOnce because once we pause we no
    /// longer care about the alert latch — the operator sees the auto-
    /// pause notification anyway.
    fn oracle(prev: u32, alerted: bool, cfg: &SchedulerConfig) -> FailurePolicy {
        let alert_armed = cfg.max_retries_before_alert > 0;
        let pause_armed = cfg.auto_pause_after_failures > 0;
        let crossed_alert = alert_armed && prev >= cfg.max_retries_before_alert;
        let crossed_pause = pause_armed && prev >= cfg.auto_pause_after_failures;

        if crossed_pause {
            FailurePolicy::AutoPause { count: prev }
        } else if crossed_alert && !alerted {
            FailurePolicy::AlertOnce { count: prev }
        } else {
            FailurePolicy::Quiet
        }
    }

    /// Exhaustive sweep over the bounded knob space. 21 × 2 × 11 × 11
    /// = 5082 combinations — well over the "≥ 2000" budget called out
    /// in the perfection plan, and small enough to run in <50ms. We
    /// keep this exhaustive rather than property-randomised so a
    /// regression has a deterministic minimal counterexample baked
    /// into the test name.
    #[test]
    fn evaluate_outcome_matches_oracle_for_all_bounded_inputs() {
        let mut checked: u32 = 0;
        for prev in 0u32..=20 {
            for alerted in [false, true] {
                for alert_thr in 0u32..=10 {
                    for pause_thr in 0u32..=10 {
                        let cfg = SchedulerConfig {
                            max_retries_before_alert: alert_thr,
                            auto_pause_after_failures: pause_thr,
                            ..SchedulerConfig::default()
                        };
                        let got = evaluate_outcome(prev, alerted, &cfg);
                        let want = oracle(prev, alerted, &cfg);
                        assert_eq!(
                            got, want,
                            "mismatch for prev={prev} alerted={alerted} \
                             alert_thr={alert_thr} pause_thr={pause_thr}: \
                             got {got:?}, want {want:?}"
                        );
                        checked += 1;
                    }
                }
            }
        }
        assert!(
            checked >= 2000,
            "exhaustive sweep must cover at least 2000 combinations; \
             got {checked} (lower the prev/threshold bounds at your peril)"
        );
    }

    /// Invariants that must hold for *every* configuration. Cheaper to
    /// state than the full oracle and guards against future refactors
    /// that accidentally relax an invariant the design depends on.
    #[test]
    fn evaluate_outcome_invariants_hold_across_bounded_inputs() {
        for prev in 0u32..=20 {
            for alerted in [false, true] {
                for alert_thr in 0u32..=10 {
                    for pause_thr in 0u32..=10 {
                        let cfg = SchedulerConfig {
                            max_retries_before_alert: alert_thr,
                            auto_pause_after_failures: pause_thr,
                            ..SchedulerConfig::default()
                        };
                        let policy = evaluate_outcome(prev, alerted, &cfg);

                        // Invariant 1: with prev = 0 we never alert
                        // or pause — the streak hasn't started.
                        if prev == 0 {
                            assert_eq!(
                                policy,
                                FailurePolicy::Quiet,
                                "prev=0 must always be Quiet (alert_thr={alert_thr}, \
                                 pause_thr={pause_thr}, alerted={alerted})"
                            );
                        }

                        // Invariant 2: both knobs disabled (=0) ⇒
                        // always Quiet, regardless of streak/alert.
                        if alert_thr == 0 && pause_thr == 0 {
                            assert_eq!(
                                policy,
                                FailurePolicy::Quiet,
                                "both thresholds disabled must yield Quiet (prev={prev})"
                            );
                        }

                        // Invariant 3: AlertOnce is never emitted
                        // when the alert latch is already set.
                        if alerted {
                            assert_ne!(
                                policy,
                                FailurePolicy::AlertOnce { count: prev },
                                "AlertOnce must not refire while alerted=true \
                                 (prev={prev}, alert_thr={alert_thr})"
                            );
                        }

                        // Invariant 4: AutoPause only fires when its
                        // threshold is armed AND crossed.
                        if let FailurePolicy::AutoPause { count } = policy {
                            assert_eq!(count, prev);
                            assert!(pause_thr > 0);
                            assert!(prev >= pause_thr);
                        }

                        // Invariant 5: AlertOnce only fires when its
                        // threshold is armed, crossed, *and* the
                        // pause threshold has NOT yet been crossed
                        // (precedence).
                        if let FailurePolicy::AlertOnce { count } = policy {
                            assert_eq!(count, prev);
                            assert!(alert_thr > 0);
                            assert!(prev >= alert_thr);
                            assert!(!alerted);
                            let pause_crossed = pause_thr > 0 && prev >= pause_thr;
                            assert!(
                                !pause_crossed,
                                "AutoPause must outrank AlertOnce when both are crossed"
                            );
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn evaluate_outcome_disabled_thresholds() {
        let cfg = cfg_with(0, 0); // both disabled
        assert_eq!(evaluate_outcome(100, false, &cfg), FailurePolicy::Quiet);
    }

    // ── hook ───────────────────────────────────────────────────────────────

    #[tokio::test]
    async fn hook_wakes_a_pending_waiter() {
        let notify = Arc::new(Notify::new());
        let hook = ResearchSchedulerHook {
            notify: notify.clone(),
            state: std::sync::Weak::new(),
        };
        let woke = Arc::new(AtomicUsize::new(0));

        let woke_clone = woke.clone();
        let notify_clone = notify.clone();
        let waiter = tokio::spawn(async move {
            let fut = notify_clone.notified();
            tokio::pin!(fut);
            tokio::task::yield_now().await;
            fut.await;
            woke_clone.store(1, Ordering::SeqCst);
        });

        tokio::task::yield_now().await;
        tokio::time::sleep(Duration::from_millis(20)).await;
        hook.notify(SchedulerEvent::SpecUpdated {
            spec_id: "x".into(),
        })
        .await;

        tokio::time::timeout(Duration::from_millis(500), waiter)
            .await
            .expect("waiter timed out — hook did not notify")
            .unwrap();
        assert_eq!(woke.load(Ordering::SeqCst), 1);
    }

    // ── Phase 2.3: failure-snapshot + reset_failures via hook ─────────────

    #[tokio::test]
    async fn hook_failure_snapshot_reads_live_state() {
        let state = Arc::new(Mutex::new(SchedulerState::default()));
        {
            let mut s = state.lock().await;
            s.failures.insert("alpha".into(), 4);
            s.alerted.insert("alpha".into());
            s.failures.insert("bravo".into(), 1);
        }
        let hook = ResearchSchedulerHook {
            notify: Arc::new(Notify::new()),
            state: Arc::downgrade(&state),
        };
        assert_eq!(hook.failure_snapshot("alpha").await, Some((4, true)));
        assert_eq!(hook.failure_snapshot("bravo").await, Some((1, false)));
        assert_eq!(hook.failure_snapshot("missing").await, Some((0, false)));
    }

    #[tokio::test]
    async fn hook_failure_snapshot_returns_none_when_state_dropped() {
        let hook = {
            let state = Arc::new(Mutex::new(SchedulerState::default()));
            ResearchSchedulerHook {
                notify: Arc::new(Notify::new()),
                state: Arc::downgrade(&state),
            }
            // `state` dropped here → Weak fails to upgrade.
        };
        assert_eq!(hook.failure_snapshot("anything").await, None);
    }

    #[tokio::test]
    async fn hook_reset_failures_clears_in_memory_counters() {
        let state = Arc::new(Mutex::new(SchedulerState::default()));
        {
            let mut s = state.lock().await;
            s.failures.insert("alpha".into(), 7);
            s.alerted.insert("alpha".into());
            s.failures.insert("bravo".into(), 2);
        }
        let hook = ResearchSchedulerHook {
            notify: Arc::new(Notify::new()),
            state: Arc::downgrade(&state),
        };
        hook.reset_failures("alpha").await;
        let s = state.lock().await;
        assert!(!s.failures.contains_key("alpha"));
        assert!(!s.alerted.contains("alpha"));
        assert_eq!(
            s.failures.get("bravo").copied(),
            Some(2),
            "reset MUST be scoped to the requested spec"
        );
    }

    // ── inflight ledger + resurrection ─────────────────────────────────────

    use naked_core::research::{FsResearchStore, Inflight, RunState};
    use tempfile::tempdir;

    #[tokio::test]
    async fn inflight_round_trip_via_fs_store() {
        let dir = tempdir().unwrap();
        let store = FsResearchStore::new(dir.path().to_path_buf());
        // Pre-create the spec dir so save_inflight has somewhere to land.
        let spec = spec_with("rt-inflight", |_| {});
        store.create_spec(&spec).await.unwrap();

        // No record yet.
        assert!(store.load_inflight(&spec.id).await.unwrap().is_none());
        assert!(
            store
                .list_nonterminal_inflight()
                .await
                .unwrap()
                .is_empty()
        );

        let mut infl = Inflight::scheduled(spec.id.clone(), 1);
        store.save_inflight(&spec.id, &infl).await.unwrap();

        let loaded = store.load_inflight(&spec.id).await.unwrap().unwrap();
        assert_eq!(loaded.state, RunState::Scheduled);
        assert_eq!(loaded.attempt, 1);
        assert_eq!(loaded.spec_id, spec.id);

        // Listing returns scheduled (non-terminal).
        let list = store.list_nonterminal_inflight().await.unwrap();
        assert_eq!(list.len(), 1);
        assert_eq!(list[0].state, RunState::Scheduled);

        // Transition to terminal → list_nonterminal returns empty.
        infl.mark_running();
        infl.mark_completed(Some("run-99".into()));
        store.save_inflight(&spec.id, &infl).await.unwrap();
        assert!(
            store
                .list_nonterminal_inflight()
                .await
                .unwrap()
                .is_empty()
        );

        // load_inflight still returns the terminal record for inspection.
        let loaded = store.load_inflight(&spec.id).await.unwrap().unwrap();
        assert_eq!(loaded.state, RunState::Completed);
        assert_eq!(loaded.run_id.as_deref(), Some("run-99"));
    }

    #[tokio::test]
    async fn purge_terminal_inflight_now_respects_retention_knob() {
        let dir = tempdir().unwrap();
        let store: Arc<dyn ResearchStore> =
            Arc::new(FsResearchStore::new(dir.path().to_path_buf()));

        let s_old = spec_with("old-completed", |_| {});
        store.create_spec(&s_old).await.unwrap();
        let mut old = Inflight::scheduled(&s_old.id, 1);
        old.state = RunState::Completed;
        old.finished_at = Some(Utc::now() - chrono::Duration::days(30));
        store.save_inflight(&s_old.id, &old).await.unwrap();

        let s_new = spec_with("recent-completed", |_| {});
        store.create_spec(&s_new).await.unwrap();
        let mut new_rec = Inflight::scheduled(&s_new.id, 1);
        new_rec.state = RunState::Completed;
        new_rec.finished_at = Some(Utc::now() - chrono::Duration::minutes(1));
        store.save_inflight(&s_new.id, &new_rec).await.unwrap();

        // 1) `inflight_terminal_retention = ZERO` ⇒ no-op.
        let cfg_off = SchedulerConfig {
            inflight_terminal_retention: Duration::ZERO,
            ..SchedulerConfig::default()
        };
        purge_terminal_inflight_now(&store, &cfg_off).await;
        assert!(store.load_inflight(&s_old.id).await.unwrap().is_some());
        assert!(store.load_inflight(&s_new.id).await.unwrap().is_some());

        // 2) Retention = 7 days ⇒ purges the 30-day-old, keeps the 1-min-old.
        let cfg_on = SchedulerConfig {
            inflight_terminal_retention: Duration::from_secs(7 * 24 * 60 * 60),
            ..SchedulerConfig::default()
        };
        purge_terminal_inflight_now(&store, &cfg_on).await;
        assert!(
            store.load_inflight(&s_old.id).await.unwrap().is_none(),
            "30-day-old terminal must be purged"
        );
        assert!(
            store.load_inflight(&s_new.id).await.unwrap().is_some(),
            "recent terminal must be retained"
        );
    }

    // ── Phase 3.5: purge_stale_tmp_files walks both root *and* spec dirs ──

    #[tokio::test]
    async fn purge_stale_tmp_files_removes_root_level_tmp_orphans() {
        let dir = tempdir().unwrap();
        let store: Arc<dyn ResearchStore> =
            Arc::new(FsResearchStore::new(dir.path().to_path_buf()));

        // Root-level orphan from a partially-completed atomic_write
        // (e.g. a future `root.json` summary) — must be swept.
        let root_tmp = dir.path().join("summary.json.tmp");
        tokio::fs::write(&root_tmp, b"half-written").await.unwrap();

        // Root-level non-tmp file (e.g. `scheduler.lock`) — must be
        // left alone, no matter what the extension is.
        let lock_file = dir.path().join("scheduler.lock");
        tokio::fs::write(&lock_file, b"\0").await.unwrap();

        let summary_file = dir.path().join("summary.json");
        tokio::fs::write(&summary_file, b"{}").await.unwrap();

        // Spec-level orphan — also swept (parity with previous behaviour).
        let s = spec_with("with-tmp", |_| {});
        store.create_spec(&s).await.unwrap();
        let spec_dir = dir.path().join(&s.id);
        let spec_tmp = spec_dir.join("inflight.json.tmp");
        tokio::fs::write(&spec_tmp, b"half-written").await.unwrap();

        purge_stale_tmp_files(&store).await;

        assert!(
            !root_tmp.exists(),
            "root-level *.tmp orphan must be purged"
        );
        assert!(
            !spec_tmp.exists(),
            "spec-level *.tmp orphan must be purged"
        );
        assert!(
            lock_file.exists(),
            "scheduler.lock (no .tmp extension) must be left alone"
        );
        assert!(
            summary_file.exists(),
            "non-tmp root files must be left alone"
        );
    }

    #[tokio::test]
    async fn purge_stale_tmp_files_no_op_on_clean_root() {
        let dir = tempdir().unwrap();
        let store: Arc<dyn ResearchStore> =
            Arc::new(FsResearchStore::new(dir.path().to_path_buf()));

        let s = spec_with("clean", |_| {});
        store.create_spec(&s).await.unwrap();

        purge_stale_tmp_files(&store).await;

        let loaded = store.load_spec(&s.id).await.unwrap();
        assert_eq!(loaded.id, s.id, "spec must remain intact after no-op purge");
    }

    #[test]
    fn default_scheduler_config_has_sensible_inflight_purge_window() {
        let cfg = SchedulerConfig::default();
        assert!(
            cfg.inflight_terminal_retention >= Duration::from_secs(7 * 24 * 60 * 60),
            "default retention must keep terminal records around for at least a week"
        );
        assert!(
            cfg.inflight_purge_interval >= Duration::from_secs(60),
            "purge interval must not be smaller than a minute (cheap, but not pointlessly hot)"
        );
        assert!(
            cfg.inflight_purge_interval <= cfg.inflight_terminal_retention,
            "purge interval must be smaller than retention or records pile up unbounded between sweeps"
        );
    }

    #[tokio::test]
    async fn list_nonterminal_skips_terminal_records() {
        let dir = tempdir().unwrap();
        let store = FsResearchStore::new(dir.path().to_path_buf());

        let s_done = spec_with("done", |_| {});
        let s_running = spec_with("running", |_| {});
        store.create_spec(&s_done).await.unwrap();
        store.create_spec(&s_running).await.unwrap();

        let mut done = Inflight::scheduled(s_done.id.clone(), 1);
        done.mark_running();
        done.mark_completed(None);
        store.save_inflight(&s_done.id, &done).await.unwrap();

        let mut running = Inflight::scheduled(s_running.id.clone(), 1);
        running.mark_running();
        store.save_inflight(&s_running.id, &running).await.unwrap();

        let list = store.list_nonterminal_inflight().await.unwrap();
        assert_eq!(list.len(), 1);
        assert_eq!(list[0].spec_id, s_running.id);
    }

    #[tokio::test]
    async fn inflight_serde_legacy_record_without_optional_fields() {
        // Hand-rolled minimal record (older binary): no optional
        // timestamps, no error, no run_id, no attempt → all defaults.
        let raw = r#"{
          "spec_id": "legacy",
          "attempt_id": "deadbeef",
          "state": "running",
          "scheduled_at": "2026-04-21T10:00:00Z"
        }"#;
        let infl: Inflight = serde_json::from_str(raw).unwrap();
        assert_eq!(infl.state, RunState::Running);
        assert_eq!(infl.attempt, 1, "default_attempt should kick in");
        assert!(infl.started_at.is_none());
        assert!(infl.last_heartbeat.is_none());
        assert!(infl.error.is_none());
        assert!(infl.run_id.is_none());
    }

    #[test]
    fn scheduler_config_defaults_have_resurrection_knobs() {
        let cfg = SchedulerConfig::default();
        assert!(cfg.max_resurrection_attempts > 0);
        assert!(cfg.heartbeat_budget > cfg.tick_interval);
        assert!(cfg.auto_pause_after_failures >= cfg.max_retries_before_alert);
    }

    // ── Phase 1.3: pull-model resurrection — capacity-aware ───────────────

    fn make_inflight_for(id: &str, state: RunState, attempt: u32) -> Inflight {
        let mut i = Inflight::scheduled(id, attempt);
        match state {
            RunState::Scheduled => {}
            RunState::Running => i.mark_running(),
            RunState::Completed => {
                i.mark_running();
                i.mark_completed(None);
            }
            RunState::Failed => {
                i.mark_running();
                i.mark_failed("test");
            }
        }
        i
    }

    #[test]
    fn plan_resurrection_drains_caps_at_available_slots() {
        // Ten resurrected entries, only two slots free → drain just two,
        // and pick deterministic order (sorted by spec_id for predictable
        // tests / log review).
        let nonterminal: Vec<Inflight> = (0..10)
            .map(|n| {
                let mut i = make_inflight_for(&format!("spec-{n:02}"), RunState::Scheduled, 2);
                i.scheduled_after_resurrection = true;
                i
            })
            .collect();
        let running: HashSet<String> = HashSet::new();
        let plan = plan_resurrection_drains(&nonterminal, &running, 2);
        assert_eq!(plan.len(), 2);
        assert_eq!(plan[0].0, "spec-00");
        assert_eq!(plan[1].0, "spec-01");
        assert_eq!(plan[0].1, 2, "attempt count carried through");
    }

    #[test]
    fn plan_resurrection_drains_skips_running() {
        let nonterminal = vec![{
            let mut i = make_inflight_for("a", RunState::Scheduled, 3);
            i.scheduled_after_resurrection = true;
            i
        }];
        let mut running = HashSet::new();
        running.insert("a".into());
        let plan = plan_resurrection_drains(&nonterminal, &running, 5);
        assert!(plan.is_empty());
    }

    #[test]
    fn plan_resurrection_drains_only_picks_resurrection_tagged() {
        // Untagged Scheduled entries (fresh dispatches) are NOT pulled by
        // the resurrection drainer — those go through `is_due` planning.
        let nonterminal = vec![
            make_inflight_for("fresh", RunState::Scheduled, 1), // untagged
            {
                let mut i = make_inflight_for("resurrected", RunState::Scheduled, 2);
                i.scheduled_after_resurrection = true;
                i
            },
        ];
        let plan = plan_resurrection_drains(&nonterminal, &HashSet::new(), 5);
        assert_eq!(plan.len(), 1);
        assert_eq!(plan[0].0, "resurrected");
    }

    #[test]
    fn plan_resurrection_drains_zero_slots_returns_empty() {
        let nonterminal = vec![{
            let mut i = make_inflight_for("a", RunState::Scheduled, 2);
            i.scheduled_after_resurrection = true;
            i
        }];
        let plan = plan_resurrection_drains(&nonterminal, &HashSet::new(), 0);
        assert!(plan.is_empty());
    }

    #[test]
    fn plan_resurrection_drains_skips_running_state() {
        // A "running" inflight on disk (worker raced through Scheduled →
        // Running before our boot rewrite landed) — drainer must ignore.
        let mut i = make_inflight_for("racing", RunState::Running, 1);
        i.scheduled_after_resurrection = true; // shouldn't matter
        let plan = plan_resurrection_drains(&[i], &HashSet::new(), 5);
        assert!(plan.is_empty());
    }

    #[tokio::test]
    async fn resurrect_at_boot_marks_healthy_inflights_for_drain() {
        // Fresh boot scenario: previous process died with a Running
        // inflight on disk. New `resurrect_at_boot` re-tags it as
        // Scheduled+resurrection (does NOT spawn) so the dispatch loop
        // can pick it up under the concurrency cap.
        let dir = tempdir().unwrap();
        let store: Arc<dyn ResearchStore> =
            Arc::new(FsResearchStore::new(dir.path().to_path_buf()));

        let spec = spec_with("victim", |_| {});
        store.create_spec(&spec).await.unwrap();
        let mut infl = Inflight::scheduled(spec.id.clone(), 1);
        infl.mark_running();
        infl.error = Some("died mid-run".into()); // shouldn't carry over
        store.save_inflight(&spec.id, &infl).await.unwrap();

        let notifier: Arc<dyn TaskNotifier> = Arc::new(NoopNotifier);
        let cfg = SchedulerConfig::default();
        rebuild_resurrection_queue(&store, &notifier, &cfg).await.unwrap();

        let after = store.load_inflight(&spec.id).await.unwrap().unwrap();
        assert_eq!(
            after.state,
            RunState::Scheduled,
            "should be re-queued, NOT spawned"
        );
        assert!(after.scheduled_after_resurrection);
        assert_eq!(after.attempt, 2, "attempt counter incremented");
        assert!(after.started_at.is_none());
        assert!(after.last_heartbeat.is_none());
        assert!(after.error.is_none());
    }

    #[tokio::test]
    async fn resurrect_at_boot_finalises_paused_specs_as_failed() {
        let dir = tempdir().unwrap();
        let store: Arc<dyn ResearchStore> =
            Arc::new(FsResearchStore::new(dir.path().to_path_buf()));
        let spec = spec_with("paused-victim", |s| s.paused = true);
        store.create_spec(&spec).await.unwrap();
        let mut infl = Inflight::scheduled(spec.id.clone(), 1);
        infl.mark_running();
        store.save_inflight(&spec.id, &infl).await.unwrap();

        let notifier: Arc<dyn TaskNotifier> = Arc::new(NoopNotifier);
        let cfg = SchedulerConfig::default();
        rebuild_resurrection_queue(&store, &notifier, &cfg).await.unwrap();

        let after = store.load_inflight(&spec.id).await.unwrap().unwrap();
        assert_eq!(after.state, RunState::Failed);
        assert!(after.error.unwrap().to_lowercase().contains("paused"));
    }

    #[tokio::test]
    async fn resurrect_at_boot_finalises_cap_hit_as_failed() {
        let dir = tempdir().unwrap();
        let store: Arc<dyn ResearchStore> =
            Arc::new(FsResearchStore::new(dir.path().to_path_buf()));
        let spec = spec_with("cap-victim", |_| {});
        store.create_spec(&spec).await.unwrap();
        let mut infl = Inflight::scheduled(spec.id.clone(), 5); // already at cap
        infl.mark_running();
        store.save_inflight(&spec.id, &infl).await.unwrap();

        let notifier: Arc<dyn TaskNotifier> = Arc::new(NoopNotifier);
        let cfg = SchedulerConfig {
            max_resurrection_attempts: 5,
            ..SchedulerConfig::default()
        };
        rebuild_resurrection_queue(&store, &notifier, &cfg).await.unwrap();

        let after = store.load_inflight(&spec.id).await.unwrap().unwrap();
        assert_eq!(after.state, RunState::Failed);
        assert!(after.error.unwrap().to_lowercase().contains("cap"));
    }

    // ── Phase 1.2: supervisor restarts panicking run-loop ─────────────────

    struct PanicCountingNotifier {
        panics: AtomicUsize,
        last: tokio::sync::Mutex<String>,
    }

    #[async_trait]
    impl TaskNotifier for PanicCountingNotifier {
        async fn notify_failure(
            &self,
            _spec: &ResearchSpec,
            _consecutive_failures: u32,
            _last_error: &str,
        ) {
        }
        async fn notify_supervisor_panic(&self, details: &str) {
            self.panics.fetch_add(1, Ordering::SeqCst);
            *self.last.lock().await = details.to_string();
        }
    }

    #[tokio::test]
    async fn supervisor_restarts_loop_after_panic_and_returns_when_clean() {
        // Loop body that panics on the first call and returns Ok on the
        // second. Supervisor MUST observe one panic, fire the notifier,
        // restart, then exit cleanly.
        let entries = Arc::new(AtomicUsize::new(0));
        let notifier = Arc::new(PanicCountingNotifier {
            panics: AtomicUsize::new(0),
            last: tokio::sync::Mutex::new(String::new()),
        });
        let trait_notifier: Arc<dyn TaskNotifier> = notifier.clone();

        let entries_clone = entries.clone();
        let make_loop = move || {
            let entries_inner = entries_clone.clone();
            async move {
                let n = entries_inner.fetch_add(1, Ordering::SeqCst);
                if n == 0 {
                    panic!("synthetic boom");
                }
            }
        };

        supervised_run_loop(
            "test",
            trait_notifier,
            Duration::from_millis(10),
            make_loop,
        )
        .await;

        assert_eq!(
            entries.load(Ordering::SeqCst),
            2,
            "loop should be entered twice (1× panic + 1× clean exit)"
        );
        assert_eq!(
            notifier.panics.load(Ordering::SeqCst),
            1,
            "exactly one panic notification expected"
        );
        let last = notifier.last.lock().await.clone();
        assert!(last.contains("test"));
        assert!(last.to_lowercase().contains("panic"));
    }

    #[tokio::test]
    async fn supervisor_caps_restart_attempts() {
        // Pathological loop that always panics. Supervisor must give up
        // after the configured cap so a poisoned scheduler can't busy-loop
        // forever burning CPU + alert quota.
        let entries = Arc::new(AtomicUsize::new(0));
        let notifier = Arc::new(PanicCountingNotifier {
            panics: AtomicUsize::new(0),
            last: tokio::sync::Mutex::new(String::new()),
        });
        let trait_notifier: Arc<dyn TaskNotifier> = notifier.clone();

        let entries_clone = entries.clone();
        let make_loop = move || {
            let entries_inner = entries_clone.clone();
            async move {
                entries_inner.fetch_add(1, Ordering::SeqCst);
                panic!("forever boom");
            }
        };

        supervised_run_loop_capped(
            "test-cap",
            trait_notifier,
            Duration::from_millis(0),
            3,
            make_loop,
        )
        .await;

        assert_eq!(entries.load(Ordering::SeqCst), 3, "exactly 3 attempts");
        assert_eq!(
            notifier.panics.load(Ordering::SeqCst),
            3,
            "every attempt should produce a notification"
        );
    }

    #[tokio::test]
    async fn supervisor_returns_immediately_when_loop_exits_cleanly() {
        let entries = Arc::new(AtomicUsize::new(0));
        let notifier: Arc<dyn TaskNotifier> = Arc::new(NoopNotifier);

        let entries_clone = entries.clone();
        let make_loop = move || {
            let entries_inner = entries_clone.clone();
            async move {
                entries_inner.fetch_add(1, Ordering::SeqCst);
            }
        };

        supervised_run_loop("clean", notifier, Duration::from_secs(99), make_loop).await;
        assert_eq!(entries.load(Ordering::SeqCst), 1);
    }

    // ── Phase 1.1: graceful shutdown aborts in-flight tasks ───────────────

    #[tokio::test]
    async fn shutdown_aborts_running_handles_and_marks_inflight_failed() {
        let dir = tempdir().unwrap();
        let store: Arc<dyn ResearchStore> =
            Arc::new(FsResearchStore::new(dir.path().to_path_buf()));

        let spec_a = spec_with("alpha", |_| {});
        let spec_b = spec_with("bravo", |_| {});
        store.create_spec(&spec_a).await.unwrap();
        store.create_spec(&spec_b).await.unwrap();

        // Persist Running inflight ledgers so shutdown has something to flip.
        for spec in [&spec_a, &spec_b] {
            let mut infl = Inflight::scheduled(spec.id.clone(), 1);
            infl.mark_running();
            store.save_inflight(&spec.id, &infl).await.unwrap();
        }

        // Build state with two long-running JoinHandles (`Duration::MAX` → never
        // returns naturally). After shutdown they MUST be aborted.
        let state = Arc::new(Mutex::new(SchedulerState::default()));
        for spec in [&spec_a, &spec_b] {
            let handle: JoinHandle<RunOutcome> = tokio::spawn(async {
                tokio::time::sleep(Duration::from_secs(86_400)).await;
                RunOutcome::Success
            });
            state.lock().await.running.insert(
                spec.id.clone(),
                RunningHandle {
                    started_at: Instant::now(),
                    timeout: Duration::from_secs(600),
                    handle,
                    cancel: CancellationToken::new(),
                    cancel_requested_at: None,
                    attempt_id: "test".into(),
                },
            );
        }

        let aborted = shutdown_running_tasks(&state, &store).await;
        assert_eq!(aborted.len(), 2, "should report both ids as aborted");
        assert!(aborted.iter().any(|id| id == "alpha"));
        assert!(aborted.iter().any(|id| id == "bravo"));

        // In-memory map cleared.
        assert!(state.lock().await.running.is_empty());

        // Both inflight ledgers flipped to Failed with a shutdown reason.
        for spec in [&spec_a, &spec_b] {
            let infl = store.load_inflight(&spec.id).await.unwrap().unwrap();
            assert_eq!(
                infl.state,
                RunState::Failed,
                "inflight for {} should be Failed",
                spec.id
            );
            let err = infl.error.clone().unwrap_or_default();
            assert!(
                err.contains("shutdown"),
                "error message should mention shutdown, got {err:?}"
            );
            assert!(infl.finished_at.is_some());
        }
    }

    #[tokio::test]
    async fn shutdown_with_empty_running_is_noop() {
        let dir = tempdir().unwrap();
        let store: Arc<dyn ResearchStore> =
            Arc::new(FsResearchStore::new(dir.path().to_path_buf()));
        let state = Arc::new(Mutex::new(SchedulerState::default()));
        let aborted = shutdown_running_tasks(&state, &store).await;
        assert!(aborted.is_empty());
    }

    #[tokio::test]
    async fn shutdown_preserves_terminal_inflight_records() {
        // If a worker raced and already wrote Completed/Failed to disk before
        // shutdown landed, the shutdown sweep MUST NOT clobber it with
        // Failed("shutdown"). Operators rely on the terminal record's truth.
        let dir = tempdir().unwrap();
        let store: Arc<dyn ResearchStore> =
            Arc::new(FsResearchStore::new(dir.path().to_path_buf()));
        let spec = spec_with("done-already", |_| {});
        store.create_spec(&spec).await.unwrap();

        let mut infl = Inflight::scheduled(spec.id.clone(), 1);
        infl.mark_running();
        infl.mark_completed(Some("run-42".into()));
        store.save_inflight(&spec.id, &infl).await.unwrap();

        let state = Arc::new(Mutex::new(SchedulerState::default()));
        let handle: JoinHandle<RunOutcome> = tokio::spawn(async {
            tokio::time::sleep(Duration::from_secs(86_400)).await;
            RunOutcome::Success
        });
        state.lock().await.running.insert(
            spec.id.clone(),
            RunningHandle {
                started_at: Instant::now(),
                timeout: Duration::from_secs(600),
                handle,
                cancel: CancellationToken::new(),
                cancel_requested_at: None,
                attempt_id: "test".into(),
            },
        );

        shutdown_running_tasks(&state, &store).await;

        let after = store.load_inflight(&spec.id).await.unwrap().unwrap();
        assert_eq!(
            after.state,
            RunState::Completed,
            "terminal record must survive a shutdown sweep"
        );
        assert_eq!(after.run_id.as_deref(), Some("run-42"));
    }

    #[tokio::test]
    async fn purge_stale_tmp_removes_only_tmp_files() {
        let dir = tempdir().unwrap();
        let store: Arc<dyn ResearchStore> =
            Arc::new(FsResearchStore::new(dir.path().to_path_buf()));

        let spec = spec_with("with-tmp", |_| {});
        store.create_spec(&spec).await.unwrap();
        let spec_dir = dir.path().join(&spec.id);

        // Drop a few stale tmp files alongside legitimate ones.
        tokio::fs::write(spec_dir.join("findings.jsonl.tmp"), b"partial")
            .await
            .unwrap();
        tokio::fs::write(spec_dir.join("report.md.tmp"), b"more partial")
            .await
            .unwrap();
        tokio::fs::write(spec_dir.join("findings.jsonl"), b"complete\n")
            .await
            .unwrap();

        purge_stale_tmp_files(&store).await;

        assert!(
            !spec_dir.join("findings.jsonl.tmp").exists(),
            "tmp file should have been removed"
        );
        assert!(
            !spec_dir.join("report.md.tmp").exists(),
            "tmp file should have been removed"
        );
        assert!(
            spec_dir.join("findings.jsonl").exists(),
            "legitimate file MUST survive cleanup"
        );
        assert!(
            spec_dir.join("spec.json").exists(),
            "spec.json MUST survive cleanup"
        );
    }

    // ── Phase 1.4: cooperative cancel → hard-abort escalation ─────────────

    #[tokio::test]
    async fn sweep_issues_cooperative_cancel_when_task_first_exceeds_timeout() {
        // Worker is over budget but has not been signalled yet. The sweep
        // MUST cancel its CancellationToken, stamp `cancel_requested_at`,
        // and KEEP the slot reserved (we don't free until grace expires
        // — otherwise the next dispatch tick spawns a duplicate while the
        // current worker is still draining).
        let dir = tempdir().unwrap();
        let store: Arc<dyn ResearchStore> =
            Arc::new(FsResearchStore::new(dir.path().to_path_buf()));
        let spec = spec_with("over-budget", |_| {});
        store.create_spec(&spec).await.unwrap();
        let mut infl = Inflight::scheduled(spec.id.clone(), 1);
        infl.mark_running();
        store.save_inflight(&spec.id, &infl).await.unwrap();

        let cancel = CancellationToken::new();
        let cancel_seen = cancel.clone();
        // Worker drops dead only when cancelled — exercises the
        // cooperative path. If the test passes, the worker stops within
        // the grace window and the sweep never escalates to abort.
        let handle: JoinHandle<RunOutcome> = tokio::spawn(async move {
            cancel_seen.cancelled().await;
            RunOutcome::Failure("cancelled by sweep timeout".into())
        });

        let state = Arc::new(Mutex::new(SchedulerState::default()));
        state.lock().await.running.insert(
            spec.id.clone(),
            RunningHandle {
                started_at: Instant::now() - Duration::from_secs(3600),
                timeout: Duration::from_secs(60),
                handle,
                cancel: cancel.clone(),
                cancel_requested_at: None,
                attempt_id: "attempt-1".into(),
            },
        );

        let notifier: Arc<dyn TaskNotifier> = Arc::new(NoopNotifier);
        let cfg = SchedulerConfig {
            cancel_grace_period: Duration::from_secs(5),
            ..SchedulerConfig::default()
        };

        sweep_running(&state, &notifier, std::slice::from_ref(&spec), &cfg, &store, None).await;

        // The token MUST have been cancelled.
        assert!(
            cancel.is_cancelled(),
            "sweep should issue cooperative cancel on first over-budget pass"
        );

        // The slot MUST still be reserved (we are inside the grace window).
        let s = state.lock().await;
        let h = s
            .running
            .get(&spec.id)
            .expect("slot must remain reserved during grace window");
        assert!(
            h.cancel_requested_at.is_some(),
            "cancel_requested_at must be stamped"
        );
        drop(s);

        // Inflight should NOT yet be terminal — only the worker's own
        // future (or stage-2 abort) flips that.
        let infl_after = store.load_inflight(&spec.id).await.unwrap().unwrap();
        assert_eq!(
            infl_after.state,
            RunState::Running,
            "ledger must stay Running while worker is draining"
        );
    }

    #[tokio::test]
    async fn sweep_escalates_to_hard_abort_after_grace_window_expires() {
        // Worker ignored the cooperative cancel for longer than
        // `cancel_grace_period`. Sweep MUST hard-abort, free the slot,
        // and finalise the ledger as Failed.
        let dir = tempdir().unwrap();
        let store: Arc<dyn ResearchStore> =
            Arc::new(FsResearchStore::new(dir.path().to_path_buf()));
        let spec = spec_with("ignores-cancel", |_| {});
        store.create_spec(&spec).await.unwrap();
        let mut infl = Inflight::scheduled(spec.id.clone(), 1);
        infl.mark_running();
        store.save_inflight(&spec.id, &infl).await.unwrap();

        // Worker NEVER honours cancellation — simulates a runaway HTTP
        // / blocking-CPU loop that doesn't reach an await point.
        let handle: JoinHandle<RunOutcome> = tokio::spawn(async {
            tokio::time::sleep(Duration::from_secs(86_400)).await;
            RunOutcome::Success
        });

        let cancel = CancellationToken::new();
        // Pre-stamp `cancel_requested_at` to a value *outside* the grace
        // window — this is the state the sweep finds on its second pass.
        let state = Arc::new(Mutex::new(SchedulerState::default()));
        state.lock().await.running.insert(
            spec.id.clone(),
            RunningHandle {
                started_at: Instant::now() - Duration::from_secs(3600),
                timeout: Duration::from_secs(60),
                handle,
                cancel: cancel.clone(),
                cancel_requested_at: Some(Instant::now() - Duration::from_secs(120)),
                attempt_id: "attempt-1".into(),
            },
        );

        let notifier: Arc<dyn TaskNotifier> = Arc::new(NoopNotifier);
        let cfg = SchedulerConfig {
            cancel_grace_period: Duration::from_secs(5),
            ..SchedulerConfig::default()
        };

        sweep_running(&state, &notifier, std::slice::from_ref(&spec), &cfg, &store, None).await;

        // Slot MUST be freed.
        assert!(
            state.lock().await.running.is_empty(),
            "hard abort must release the slot"
        );
        // Belt-and-suspenders cancel was issued too.
        assert!(cancel.is_cancelled());

        // Inflight finalised as Failed with a hard-abort reason.
        let infl_after = store.load_inflight(&spec.id).await.unwrap().unwrap();
        assert_eq!(infl_after.state, RunState::Failed);
        let err = infl_after.error.unwrap_or_default();
        assert!(
            err.contains("hard-abort") || err.contains("hard"),
            "error must mention hard-abort, got {err:?}"
        );
    }

    #[tokio::test]
    async fn sweep_with_zero_grace_period_aborts_immediately() {
        // `cancel_grace_period == 0` is the legacy behaviour: skip the
        // cooperative stage and go straight to abort. Useful for tests
        // that want deterministic timing.
        let dir = tempdir().unwrap();
        let store: Arc<dyn ResearchStore> =
            Arc::new(FsResearchStore::new(dir.path().to_path_buf()));
        let spec = spec_with("zero-grace", |_| {});
        store.create_spec(&spec).await.unwrap();
        let mut infl = Inflight::scheduled(spec.id.clone(), 1);
        infl.mark_running();
        store.save_inflight(&spec.id, &infl).await.unwrap();

        let handle: JoinHandle<RunOutcome> = tokio::spawn(async {
            tokio::time::sleep(Duration::from_secs(86_400)).await;
            RunOutcome::Success
        });

        let state = Arc::new(Mutex::new(SchedulerState::default()));
        state.lock().await.running.insert(
            spec.id.clone(),
            RunningHandle {
                started_at: Instant::now() - Duration::from_secs(3600),
                timeout: Duration::from_secs(60),
                handle,
                cancel: CancellationToken::new(),
                cancel_requested_at: None,
                attempt_id: "attempt-1".into(),
            },
        );

        let notifier: Arc<dyn TaskNotifier> = Arc::new(NoopNotifier);
        let cfg = SchedulerConfig {
            cancel_grace_period: Duration::ZERO,
            ..SchedulerConfig::default()
        };

        sweep_running(&state, &notifier, std::slice::from_ref(&spec), &cfg, &store, None).await;

        assert!(
            state.lock().await.running.is_empty(),
            "with zero grace, slot must be freed in one pass"
        );
        let infl_after = store.load_inflight(&spec.id).await.unwrap().unwrap();
        assert_eq!(infl_after.state, RunState::Failed);
    }

    #[tokio::test]
    async fn sweep_skips_cancel_when_inside_grace_and_not_yet_expired() {
        // Worker has been signalled but is still inside the grace window.
        // Sweep must NOT escalate to abort — give the worker more time.
        let dir = tempdir().unwrap();
        let store: Arc<dyn ResearchStore> =
            Arc::new(FsResearchStore::new(dir.path().to_path_buf()));
        let spec = spec_with("draining", |_| {});
        store.create_spec(&spec).await.unwrap();
        let mut infl = Inflight::scheduled(spec.id.clone(), 1);
        infl.mark_running();
        store.save_inflight(&spec.id, &infl).await.unwrap();

        let cancel = CancellationToken::new();
        cancel.cancel(); // first pass already cancelled it
        let handle: JoinHandle<RunOutcome> = tokio::spawn(async {
            tokio::time::sleep(Duration::from_secs(86_400)).await;
            RunOutcome::Success
        });

        let state = Arc::new(Mutex::new(SchedulerState::default()));
        state.lock().await.running.insert(
            spec.id.clone(),
            RunningHandle {
                started_at: Instant::now() - Duration::from_secs(3600),
                timeout: Duration::from_secs(60),
                handle,
                cancel: cancel.clone(),
                cancel_requested_at: Some(Instant::now()), // just signalled
                attempt_id: "attempt-1".into(),
            },
        );

        let notifier: Arc<dyn TaskNotifier> = Arc::new(NoopNotifier);
        let cfg = SchedulerConfig {
            cancel_grace_period: Duration::from_secs(60),
            ..SchedulerConfig::default()
        };

        sweep_running(&state, &notifier, std::slice::from_ref(&spec), &cfg, &store, None).await;

        // Slot MUST still be held — worker still inside grace window.
        let s = state.lock().await;
        assert!(
            s.running.contains_key(&spec.id),
            "slot must remain reserved while inside the grace window"
        );
    }

    #[tokio::test]
    async fn sweep_drops_naturally_finished_handles_without_alerting() {
        // A worker that finished cleanly on its own MUST be dropped
        // without firing apply_outcome again (the worker already did).
        let dir = tempdir().unwrap();
        let store: Arc<dyn ResearchStore> =
            Arc::new(FsResearchStore::new(dir.path().to_path_buf()));
        let spec = spec_with("clean-exit", |_| {});
        store.create_spec(&spec).await.unwrap();

        let handle: JoinHandle<RunOutcome> = tokio::spawn(async { RunOutcome::Success });
        // Wait for the future to actually finish before sweep runs so
        // `is_finished()` returns true deterministically.
        tokio::time::sleep(Duration::from_millis(20)).await;

        let state = Arc::new(Mutex::new(SchedulerState::default()));
        state.lock().await.running.insert(
            spec.id.clone(),
            RunningHandle {
                started_at: Instant::now(),
                timeout: Duration::from_secs(600),
                handle,
                cancel: CancellationToken::new(),
                cancel_requested_at: None,
                attempt_id: "attempt-1".into(),
            },
        );

        let notifier: Arc<dyn TaskNotifier> = Arc::new(NoopNotifier);
        let cfg = SchedulerConfig::default();

        sweep_running(&state, &notifier, std::slice::from_ref(&spec), &cfg, &store, None).await;

        assert!(state.lock().await.running.is_empty());
    }

    #[tokio::test]
    async fn scheduler_config_default_has_nonzero_cancel_grace() {
        let cfg = SchedulerConfig::default();
        assert!(
            cfg.cancel_grace_period > Duration::ZERO,
            "default must give workers a chance to cancel cooperatively"
        );
        assert!(
            cfg.cancel_grace_period < cfg.task_timeout,
            "grace window must be shorter than the task timeout itself"
        );
    }
}
