use super::*;
use crate::scheduler::lifecycle::*;
use crate::scheduler::tasks::*;
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
    assert_eq!(
        tasks::evaluate_outcome(1, false, &cfg),
        FailurePolicy::Quiet
    );
    assert_eq!(
        tasks::evaluate_outcome(2, false, &cfg),
        FailurePolicy::Quiet
    );
}

#[test]
fn evaluate_outcome_alerts_once_at_threshold() {
    let cfg = cfg_with(3, 5);
    assert_eq!(
        tasks::evaluate_outcome(3, false, &cfg),
        FailurePolicy::AlertOnce { count: 3 }
    );
    // Already alerted → quiet until pause.
    assert_eq!(tasks::evaluate_outcome(4, true, &cfg), FailurePolicy::Quiet);
}

#[test]
fn evaluate_outcome_pauses_at_or_above_threshold() {
    let cfg = cfg_with(3, 5);
    assert_eq!(
        tasks::evaluate_outcome(5, true, &cfg),
        FailurePolicy::AutoPause { count: 5 }
    );
    assert_eq!(
        tasks::evaluate_outcome(7, true, &cfg),
        FailurePolicy::AutoPause { count: 7 }
    );
    // Pause takes precedence over alert when both would trigger.
    let cfg_eq = cfg_with(3, 3);
    assert_eq!(
        tasks::evaluate_outcome(3, false, &cfg_eq),
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
                    let got = tasks::evaluate_outcome(prev, alerted, &cfg);
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
                    let policy = tasks::evaluate_outcome(prev, alerted, &cfg);

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
    assert_eq!(
        tasks::evaluate_outcome(100, false, &cfg),
        FailurePolicy::Quiet
    );
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

use naked_core::research::{FsResearchStore, Inflight, InflightStore, RunState, SpecStore};
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
    assert!(store.list_nonterminal_inflight().await.unwrap().is_empty());

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
    assert!(store.list_nonterminal_inflight().await.unwrap().is_empty());

    // load_inflight still returns the terminal record for inspection.
    let loaded = store.load_inflight(&spec.id).await.unwrap().unwrap();
    assert_eq!(loaded.state, RunState::Completed);
    assert_eq!(loaded.run_id.as_deref(), Some("run-99"));
}

#[tokio::test]
async fn purge_terminal_inflight_now_respects_retention_knob() {
    let dir = tempdir().unwrap();
    let store: Arc<dyn ResearchStore> = Arc::new(FsResearchStore::new(dir.path().to_path_buf()));

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
    lifecycle::purge_terminal_inflight_now(&store, &cfg_off).await;
    assert!(store.load_inflight(&s_old.id).await.unwrap().is_some());
    assert!(store.load_inflight(&s_new.id).await.unwrap().is_some());

    // 2) Retention = 7 days ⇒ purges the 30-day-old, keeps the 1-min-old.
    let cfg_on = SchedulerConfig {
        inflight_terminal_retention: Duration::from_secs(7 * 24 * 60 * 60),
        ..SchedulerConfig::default()
    };
    lifecycle::purge_terminal_inflight_now(&store, &cfg_on).await;
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
    let store: Arc<dyn ResearchStore> = Arc::new(FsResearchStore::new(dir.path().to_path_buf()));

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

    assert!(!root_tmp.exists(), "root-level *.tmp orphan must be purged");
    assert!(!spec_tmp.exists(), "spec-level *.tmp orphan must be purged");
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
    let store: Arc<dyn ResearchStore> = Arc::new(FsResearchStore::new(dir.path().to_path_buf()));

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
    let store: Arc<dyn ResearchStore> = Arc::new(FsResearchStore::new(dir.path().to_path_buf()));

    let spec = spec_with("victim", |_| {});
    store.create_spec(&spec).await.unwrap();
    let mut infl = Inflight::scheduled(spec.id.clone(), 1);
    infl.mark_running();
    infl.error = Some("died mid-run".into()); // shouldn't carry over
    store.save_inflight(&spec.id, &infl).await.unwrap();

    let notifier: Arc<dyn TaskNotifier> = Arc::new(NoopNotifier);
    let cfg = SchedulerConfig::default();
    rebuild_resurrection_queue(&store, &notifier, &cfg)
        .await
        .unwrap();

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
    let store: Arc<dyn ResearchStore> = Arc::new(FsResearchStore::new(dir.path().to_path_buf()));
    let spec = spec_with("paused-victim", |s| s.paused = true);
    store.create_spec(&spec).await.unwrap();
    let mut infl = Inflight::scheduled(spec.id.clone(), 1);
    infl.mark_running();
    store.save_inflight(&spec.id, &infl).await.unwrap();

    let notifier: Arc<dyn TaskNotifier> = Arc::new(NoopNotifier);
    let cfg = SchedulerConfig::default();
    rebuild_resurrection_queue(&store, &notifier, &cfg)
        .await
        .unwrap();

    let after = store.load_inflight(&spec.id).await.unwrap().unwrap();
    assert_eq!(after.state, RunState::Failed);
    assert!(after.error.unwrap().to_lowercase().contains("paused"));
}

#[tokio::test]
async fn resurrect_at_boot_finalises_cap_hit_as_failed() {
    let dir = tempdir().unwrap();
    let store: Arc<dyn ResearchStore> = Arc::new(FsResearchStore::new(dir.path().to_path_buf()));
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
    rebuild_resurrection_queue(&store, &notifier, &cfg)
        .await
        .unwrap();

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

// F1: post-consolidation, the panic-restart property is now tested
// against `spawn_supervised_with_opts` directly. We keep these three
// scheduler-level tests (renamed for clarity) so the integration
// between TaskNotifier and the unified supervisor primitive remains
// pinned — a regression on the adapter
// (`crate::scheduler::NotifierPanicHook`) would otherwise only be
// caught by manual prod observation.

fn make_supervised_handle<F, Fut>(
    name: &'static str,
    notifier: Arc<dyn TaskNotifier>,
    backoff_ms: u64,
    max_attempts: Option<u32>,
    factory: F,
) -> tokio::task::JoinHandle<()>
where
    F: FnMut(tokio_util::sync::CancellationToken) -> Fut + Send + 'static,
    Fut: std::future::Future<Output = anyhow::Result<()>> + Send + 'static,
{
    let panic_hook: Arc<dyn crate::supervised::PanicHook> =
        Arc::new(crate::scheduler::NotifierPanicHook(notifier));
    crate::supervised::spawn_supervised_with_opts(
        crate::supervised::SupervisorOptions {
            name,
            backoff: crate::supervised::Backoff {
                initial: Duration::from_millis(backoff_ms),
                max: Duration::from_millis(backoff_ms.max(1)),
                multiplier: 1,
            },
            shutdown: tokio_util::sync::CancellationToken::new(),
            max_attempts,
            panic_hook: Some(panic_hook),
        },
        factory,
    )
}

#[tokio::test]
async fn supervisor_restarts_loop_after_panic_and_returns_when_clean() {
    let entries = Arc::new(AtomicUsize::new(0));
    let notifier = Arc::new(PanicCountingNotifier {
        panics: AtomicUsize::new(0),
        last: tokio::sync::Mutex::new(String::new()),
    });
    let trait_notifier: Arc<dyn TaskNotifier> = notifier.clone();

    let entries_clone = entries.clone();
    let h = make_supervised_handle("test", trait_notifier, 10, None, move |_tok| {
        let entries_inner = entries_clone.clone();
        async move {
            let n = entries_inner.fetch_add(1, Ordering::SeqCst);
            if n == 0 {
                panic!("synthetic boom");
            }
            Ok(())
        }
    });
    h.await.unwrap();

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
    let entries = Arc::new(AtomicUsize::new(0));
    let notifier = Arc::new(PanicCountingNotifier {
        panics: AtomicUsize::new(0),
        last: tokio::sync::Mutex::new(String::new()),
    });
    let trait_notifier: Arc<dyn TaskNotifier> = notifier.clone();

    let entries_clone = entries.clone();
    let h = make_supervised_handle("test-cap", trait_notifier, 0, Some(3), move |_tok| {
        let entries_inner = entries_clone.clone();
        async move {
            entries_inner.fetch_add(1, Ordering::SeqCst);
            panic!("forever boom");
        }
    });
    h.await.unwrap();

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
    let h = make_supervised_handle("clean", notifier, 99_000, None, move |_tok| {
        let entries_inner = entries_clone.clone();
        async move {
            entries_inner.fetch_add(1, Ordering::SeqCst);
            Ok(())
        }
    });
    h.await.unwrap();
    assert_eq!(entries.load(Ordering::SeqCst), 1);
}

// ── Phase 1.1: graceful shutdown aborts in-flight tasks ───────────────

#[tokio::test]
async fn shutdown_aborts_running_handles_and_marks_inflight_failed() {
    let dir = tempdir().unwrap();
    let store: Arc<dyn ResearchStore> = Arc::new(FsResearchStore::new(dir.path().to_path_buf()));

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
    let store: Arc<dyn ResearchStore> = Arc::new(FsResearchStore::new(dir.path().to_path_buf()));
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
    let store: Arc<dyn ResearchStore> = Arc::new(FsResearchStore::new(dir.path().to_path_buf()));
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
    let store: Arc<dyn ResearchStore> = Arc::new(FsResearchStore::new(dir.path().to_path_buf()));

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
    let store: Arc<dyn ResearchStore> = Arc::new(FsResearchStore::new(dir.path().to_path_buf()));
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

    tasks::sweep_running(
        &state,
        &notifier,
        std::slice::from_ref(&spec),
        &cfg,
        &store,
        None,
    )
    .await;

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
    let store: Arc<dyn ResearchStore> = Arc::new(FsResearchStore::new(dir.path().to_path_buf()));
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

    tasks::sweep_running(
        &state,
        &notifier,
        std::slice::from_ref(&spec),
        &cfg,
        &store,
        None,
    )
    .await;

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
    let store: Arc<dyn ResearchStore> = Arc::new(FsResearchStore::new(dir.path().to_path_buf()));
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

    tasks::sweep_running(
        &state,
        &notifier,
        std::slice::from_ref(&spec),
        &cfg,
        &store,
        None,
    )
    .await;

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
    let store: Arc<dyn ResearchStore> = Arc::new(FsResearchStore::new(dir.path().to_path_buf()));
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

    tasks::sweep_running(
        &state,
        &notifier,
        std::slice::from_ref(&spec),
        &cfg,
        &store,
        None,
    )
    .await;

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
    let store: Arc<dyn ResearchStore> = Arc::new(FsResearchStore::new(dir.path().to_path_buf()));
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

    tasks::sweep_running(
        &state,
        &notifier,
        std::slice::from_ref(&spec),
        &cfg,
        &store,
        None,
    )
    .await;

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

// ── plan_resurrection_drains ─────────────────────────────────────

#[test]
fn resurrection_drains_nothing_with_zero_slots() {
    let inflight = vec![Inflight {
        spec_id: "a".into(),
        state: RunState::Scheduled,
        scheduled_after_resurrection: true,
        attempt: 2,
        ..Default::default()
    }];
    let running = HashSet::new();
    assert!(lifecycle::plan_resurrection_drains(&inflight, &running, 0).is_empty());
}

#[test]
fn resurrection_drains_only_scheduled_resurrected() {
    let inflight = vec![
        Inflight {
            spec_id: "a".into(),
            state: RunState::Scheduled,
            scheduled_after_resurrection: true,
            attempt: 2,
            ..Default::default()
        },
        Inflight {
            spec_id: "b".into(),
            state: RunState::Running, // not eligible
            scheduled_after_resurrection: true,
            attempt: 1,
            ..Default::default()
        },
        Inflight {
            spec_id: "c".into(),
            state: RunState::Scheduled,
            scheduled_after_resurrection: false, // not resurrection
            attempt: 1,
            ..Default::default()
        },
    ];
    let running = HashSet::new();
    let result = lifecycle::plan_resurrection_drains(&inflight, &running, 10);
    assert_eq!(result.len(), 1);
    assert_eq!(result[0].0, "a");
}

#[test]
fn resurrection_skips_already_running() {
    let inflight = vec![Inflight {
        spec_id: "a".into(),
        state: RunState::Scheduled,
        scheduled_after_resurrection: true,
        attempt: 2,
        ..Default::default()
    }];
    let mut running = HashSet::new();
    running.insert("a".to_string());
    assert!(lifecycle::plan_resurrection_drains(&inflight, &running, 10).is_empty());
}

#[test]
fn resurrection_respects_slot_limit() {
    let inflight: Vec<_> = (0..5)
        .map(|i| Inflight {
            spec_id: format!("s{i}"),
            state: RunState::Scheduled,
            scheduled_after_resurrection: true,
            attempt: 1,
            ..Default::default()
        })
        .collect();
    let running = HashSet::new();
    assert_eq!(
        lifecycle::plan_resurrection_drains(&inflight, &running, 2).len(),
        2
    );
}

// ── OperatorContext stamps on Inflight ──────────────────────────────────

#[test]
fn operator_context_struct_fields() {
    let op = tasks::OperatorContext {
        chat_id: 12345,
        thread_id: Some(678),
        session_id: "sess-op".into(),
        prompt: "run spec=foo".into(),
    };
    assert_eq!(op.chat_id, 12345);
    assert_eq!(op.thread_id, Some(678));
    assert_eq!(op.session_id, "sess-op");
    assert_eq!(op.prompt, "run spec=foo");
}

// ── cancel_task ────────────────────────────────────────────────────────

#[tokio::test]
async fn cancel_task_returns_false_for_unknown() {
    let sched = ResearchScheduler::start(std::sync::Weak::new(), SchedulerConfig::default()).0;
    assert!(!sched.cancel_task("nonexistent").await);
}
