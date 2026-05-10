use super::dispatch::scan_and_dispatch;
use super::tasks::spawn_task;
use super::*;

pub(crate) async fn run_loop(
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
        // F2 of PLAN_NEXT_SESSION: heartbeat the liveness registry so
        // the watchdog arbiter can detect 'scheduler frozen, polling
        // alive' — the symmetric failure mode of incident
        // 2026-05-10 12:47.
        if let Some(l) = &config.liveness {
            l.beat("scheduler.tick");
        }

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
pub(crate) async fn purge_terminal_inflight_now(
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
    match store
        .purge_terminal_inflight(config.clock.now(), retention)
        .await
    {
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
pub(crate) async fn resurrect_at_boot(
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
pub(crate) async fn rebuild_resurrection_queue(
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
        if config.max_resurrection_attempts == 0 || infl.attempt >= config.max_resurrection_attempts
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
pub(crate) fn plan_resurrection_drains(
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
pub(crate) async fn drain_resurrection_queue(
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

pub(crate) fn format_panic_payload(je: &tokio::task::JoinError) -> String {
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
pub(crate) async fn shutdown_running_tasks(
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
pub(crate) async fn purge_stale_tmp_files(store: &Arc<dyn ResearchStore>) {
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
pub(crate) async fn try_remove_tmp(path: &std::path::Path) -> Option<u64> {
    let bytes = tokio::fs::metadata(path)
        .await
        .ok()
        .map(|m| m.len())
        .unwrap_or(0);
    match tokio::fs::remove_file(path).await {
        Ok(()) => Some(bytes),
        Err(e) => {
            tracing::debug!(?path, "purge_stale_tmp_files: {e:#}");
            None
        }
    }
}

pub(crate) async fn sleep_until_next(d: Duration) {
    let deadline = Instant::now() + d;
    sleep(deadline.saturating_duration_since(Instant::now())).await;
}
