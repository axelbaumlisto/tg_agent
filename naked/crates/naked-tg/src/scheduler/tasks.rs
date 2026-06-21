use super::*;

/// Operator context for `/research run`-triggered dispatches.
/// When present, these fields are stamped onto the Inflight record
/// so the scheduler can stream results back to the right chat.
pub(crate) struct OperatorContext {
    pub chat_id: i64,
    pub thread_id: Option<i32>,
    pub session_id: String,
    pub prompt: String,
}

/// Spawn the worker for one scheduling attempt. Owns the on-disk
/// state-machine: writes `Scheduled` before spawn, has the worker
/// flip to `Running`, and finalises to `Completed` / `Failed` from
/// inside the spawned future.
///
/// `attempt` is `1` for a fresh dispatch and `prev_attempt + 1`
/// when called from `resurrect_at_boot`.
pub(crate) async fn spawn_task(
    core: &Arc<AgentCore>,
    state: &Arc<Mutex<SchedulerState>>,
    notifier: &Arc<dyn TaskNotifier>,
    config: &SchedulerConfig,
    spec: &ResearchSpec,
    attempt: u32,
    operator: Option<OperatorContext>,
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
    if let Some(op) = &operator {
        infl.chat_id = Some(op.chat_id);
        infl.thread_id = op.thread_id;
        infl.session_id = Some(op.session_id.clone());
        infl.prompt = Some(op.prompt.clone());
    }
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

        // T6.3 (PLAN_RESEARCH_AGENT_FLOW_v1): synthetic-dispatch path.
        // When wiring.rs has installed `dispatch_fn` AND the effective
        // run origin has a chat_id, route through the operator's chat
        // session instead of spawning an orphan research-channel session.
        // B79: prefer attempt-scoped Inflight origin over the persisted spec
        // so `/research run <id>` from chat streams a bubble even for old/CLI
        // specs whose long-lived `ResearchSpec.chat_id` is null.
        let (effective_chat_id, effective_thread_id) = effective_origin(
            infl.chat_id,
            infl.thread_id,
            spec_for_task.chat_id,
            spec_for_task.thread_id,
        );
        let synthetic_mode = config_clone.dispatch_fn.is_some() && effective_chat_id.is_some();

        tracing::info!(
            spec = %id_for_task,
            attempt = attempt,
            attempt_id = %attempt_id_for_task,
            verify = config_clone.verify_by_default,
            synthetic = synthetic_mode,
            "scheduler launching research run"
        );
        let provider_name = research_provider_name(&core_clone, &spec_for_task);
        let provider = core_clone.provider_for(&provider_name).await;
        let blacklisted = provider.blacklisted_key_count();
        let total = provider.total_key_count();

        // Keep the full RunReport so we can tell a `Cancelled`
        // stop reason apart from a real success — the scheduler
        // treats cancellation as a *failure* (it always means the
        // sweep timeout fired), not as a normal completion.
        let result: Result<(String, StopReason), _> = if should_skip_dispatch(blacklisted, total) {
            record_scheduler_dispatch_skipped();
            tracing::warn!(
                spec = %id_for_task,
                provider = %provider_name,
                blacklisted,
                total,
                "all provider keys/providers blacklisted — skipping dispatch"
            );
            Err(naked_core::error::AgentError::Session(
                "provider exhausted, dispatch skipped".into(),
            ))
        } else if let Some(dispatch) = config_clone.dispatch_fn.as_ref().cloned()
            && let Some(chat_id) = effective_chat_id
        {
            // B64 fix: Synthetic dispatch now returns a structured result.
            // The closure drains AgentHandle to completion and reports
            // success/failure. We map that to scheduler ledger entries.
            // REGISTRY-WAIVE: synthetic_mode flag above mirrors this
            // effective-origin condition; if-let is the lint-friendly form.
            let _ = synthetic_mode; // already logged above
            let thread_id = effective_thread_id;
            let synth = crate::synthetic::SyntheticMessage::from_scheduler_spec_with_run_id(
                &id_for_task,
                chat_id,
                thread_id,
                Some(attempt_id_for_task.clone()),
            );
            // B64: session_id latch — the dispatch closure resolves
            // the session_id inside its future; we need it in the
            // cancel branch to abort the real agent turn.
            // B64 + SESSION_FIX_v1: The dispatch closure now calls
            // stream_response() which blocks for the entire turn (minutes).
            // We pass `sid_latch` INTO the closure so it can publish the
            // session-id immediately after dispatch_for_chat() — BEFORE
            // the blocking stream. The cancel branch reads the latch to
            // call agent.abort(sid) while the turn is still in progress.
            let sid_latch: crate::synthetic::SessionIdLatch =
                std::sync::Arc::new(tokio::sync::Mutex::new(None));
            let dispatch_fut = (dispatch)(synth, sid_latch.clone());
            tokio::select! {
                result = dispatch_fut => {
                    match result {
                        Ok(outcome)
                            if outcome.stop_reason == StopReason::AgentIdle
                                && outcome.errors.is_empty() =>
                        {
                            Ok((attempt_id_for_task.clone(), StopReason::AgentIdle))
                        }
                        Ok(outcome) => {
                            // Turn completed but with errors or abnormal stop.
                            let detail = format!(
                                "synthetic turn finished with {:?}, {} errors: {}",
                                outcome.stop_reason,
                                outcome.errors.len(),
                                outcome.errors.join("; ")
                            );
                            Err(naked_core::error::AgentError::Session(detail))
                        }
                        Err(e) => {
                            // Dispatch setup failed (SessionBusy, send_prompt error).
                            Err(naked_core::error::AgentError::Session(
                                format!("synthetic dispatch error: {e}"),
                            ))
                        }
                    }
                }
                _ = cancel_for_task.cancelled() => {
                    // B64 step 6: abort the real agent session if we
                    // know its id (dispatch started but cancel fired
                    // before the drain finished).
                    if let Some(sid) = sid_latch.lock().await.as_deref() {
                        tracing::warn!(
                            spec = %id_for_task,
                            session_id = %sid,
                            "scheduler cancel: aborting synthetic session"
                        );
                        core_clone.abort(sid).await;
                    }
                    Ok((attempt_id_for_task.clone(), StopReason::Cancelled))
                }
            }
        } else if !synthetic_mode && config_clone.verify_by_default {
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
pub(crate) async fn sweep_running(
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
        to_hard_abort.extend(to_request_cancel);
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
    let now = config.clock.now();
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
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum FailurePolicy {
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
pub(crate) fn should_skip_dispatch(blacklisted: usize, total: usize) -> bool {
    total > 0 && blacklisted >= total
}

fn effective_origin(
    inflight_chat_id: Option<i64>,
    inflight_thread_id: Option<i32>,
    spec_chat_id: Option<i64>,
    spec_thread_id: Option<i32>,
) -> (Option<i64>, Option<i32>) {
    (
        inflight_chat_id.or(spec_chat_id),
        inflight_thread_id.or(spec_thread_id),
    )
}

fn record_scheduler_dispatch_skipped() {
    naked_core::types::SCHEDULER_DISPATCH_SKIPPED_COUNT
        .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
}

fn research_provider_name(core: &AgentCore, spec: &ResearchSpec) -> String {
    let cfg = core.config();
    spec.provider
        .clone()
        .or_else(|| cfg.research.provider.clone())
        .unwrap_or_else(|| cfg.default_provider.clone())
}

pub(crate) fn evaluate_outcome(
    prev_count: u32,
    alerted: bool,
    cfg: &SchedulerConfig,
) -> FailurePolicy {
    let next = prev_count;
    if cfg.auto_pause_after_failures > 0 && next >= cfg.auto_pause_after_failures {
        return FailurePolicy::AutoPause { count: next };
    }
    if cfg.max_retries_before_alert > 0 && next >= cfg.max_retries_before_alert && !alerted {
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
pub(crate) async fn apply_outcome(
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
            s.last_runs.insert(spec.id.clone(), cfg.clock.now());
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
                format!("auto-paused after {count} consecutive failures · last error: {last_err}")
            } else {
                format!(
                    "{count} consecutive failures (auto-pause patch FAILED — pause manually) · last error: {last_err}"
                )
            };
            notifier.notify_failure(spec, count, &label).await;
        }
    }
}

pub(crate) async fn lookup_last_run_on_disk(
    store: &Arc<dyn ResearchStore>,
    spec_id: &str,
) -> Option<DateTime<Utc>> {
    let runs = store.list_runs(spec_id, Some(1)).await.ok()?;
    runs.first().map(|r| r.finished_at)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cfg_with_thresholds(alert: u32, auto_pause: u32) -> SchedulerConfig {
        SchedulerConfig {
            max_retries_before_alert: alert,
            auto_pause_after_failures: auto_pause,
            ..SchedulerConfig::default()
        }
    }

    // ── effective_origin ────────────────────────────────────────────

    #[test]
    fn effective_origin_prefers_inflight_chat_when_spec_missing() {
        assert_eq!(
            effective_origin(Some(42), Some(7), None, None),
            (Some(42), Some(7))
        );
    }

    #[test]
    fn effective_origin_uses_spec_chat_for_scheduled_runs() {
        assert_eq!(
            effective_origin(None, None, Some(99), Some(11)),
            (Some(99), Some(11))
        );
    }

    #[test]
    fn effective_origin_none_when_inflight_and_spec_missing() {
        assert_eq!(effective_origin(None, None, None, None), (None, None));
    }

    #[test]
    fn effective_origin_prefers_inflight_when_both_present() {
        assert_eq!(
            effective_origin(Some(42), Some(7), Some(99), Some(11)),
            (Some(42), Some(7))
        );
    }

    // ── should_skip_dispatch ─────────────────────────────────────────

    #[test]
    fn should_skip_dispatch_requires_nonzero_total_and_full_exhaustion() {
        assert!(!should_skip_dispatch(0, 0));
        assert!(!should_skip_dispatch(1, 3));
        assert!(should_skip_dispatch(3, 3));
        assert!(!should_skip_dispatch(0, 3));
        assert!(should_skip_dispatch(4, 3));
    }

    #[test]
    fn record_scheduler_dispatch_skipped_increments_counter() {
        use std::sync::atomic::Ordering;

        let before = naked_core::types::SCHEDULER_DISPATCH_SKIPPED_COUNT.load(Ordering::Relaxed);
        assert!(should_skip_dispatch(3, 3));
        record_scheduler_dispatch_skipped();
        let after = naked_core::types::SCHEDULER_DISPATCH_SKIPPED_COUNT.load(Ordering::Relaxed);
        assert_eq!(after - before, 1);
    }

    // ── evaluate_outcome ─────────────────────────────────────────────

    #[test]
    fn evaluate_outcome_below_all_thresholds() {
        let cfg = cfg_with_thresholds(3, 10);
        assert!(matches!(
            evaluate_outcome(1, false, &cfg),
            FailurePolicy::Quiet
        ));
        assert!(matches!(
            evaluate_outcome(2, false, &cfg),
            FailurePolicy::Quiet
        ));
    }

    #[test]
    fn evaluate_outcome_at_alert_threshold() {
        let cfg = cfg_with_thresholds(3, 10);
        assert!(matches!(
            evaluate_outcome(3, false, &cfg),
            FailurePolicy::AlertOnce { count: 3 }
        ));
    }

    #[test]
    fn evaluate_outcome_above_alert_already_alerted() {
        let cfg = cfg_with_thresholds(3, 10);
        // Already alerted → Quiet, not AlertOnce again
        assert!(matches!(
            evaluate_outcome(5, true, &cfg),
            FailurePolicy::Quiet
        ));
    }

    #[test]
    fn evaluate_outcome_at_auto_pause() {
        let cfg = cfg_with_thresholds(3, 5);
        assert!(matches!(
            evaluate_outcome(5, true, &cfg),
            FailurePolicy::AutoPause { count: 5 }
        ));
        assert!(matches!(
            evaluate_outcome(5, false, &cfg),
            FailurePolicy::AutoPause { count: 5 }
        ));
    }

    #[test]
    fn evaluate_outcome_zero_thresholds_means_disabled() {
        let cfg = cfg_with_thresholds(0, 0);
        // With both thresholds disabled, always Quiet
        assert!(matches!(
            evaluate_outcome(100, false, &cfg),
            FailurePolicy::Quiet
        ));
    }

    #[test]
    fn evaluate_outcome_auto_pause_takes_priority_over_alert() {
        // When both thresholds fire on same count, auto_pause wins
        let cfg = cfg_with_thresholds(3, 3);
        assert!(matches!(
            evaluate_outcome(3, false, &cfg),
            FailurePolicy::AutoPause { count: 3 }
        ));
    }
}
