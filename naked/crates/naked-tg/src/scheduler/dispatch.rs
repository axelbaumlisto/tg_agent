use super::lifecycle::drain_resurrection_queue;
use super::tasks::{lookup_last_run_on_disk, spawn_task, sweep_running};
use super::*;

/// Is this spec due for a run right now? Honours the four-tier priority
/// documented at the top of this module.
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

pub(crate) async fn scan_and_dispatch(
    core: &Arc<AgentCore>,
    _semaphore: &Arc<Semaphore>,
    state: &Arc<Mutex<SchedulerState>>,
    notifier: &Arc<dyn TaskNotifier>,
    config: &SchedulerConfig,
) -> anyhow::Result<()> {
    let store = core.research_store();
    let specs = store.list_specs().await?;
    let now = config.clock.now();

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
        let available_slots = config.max_concurrent_runs.saturating_sub(s.running.len());
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
                run_at: PatchField::Clear,
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
