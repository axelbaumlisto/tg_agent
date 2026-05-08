use super::*;
use tempfile::tempdir;

fn cfg_with(path: PathBuf) -> ModelHealthConfig {
    ModelHealthConfig {
        enabled: true,
        log_path: Some(path),
        empty_threshold: 3,
        error_threshold: 5,
        quarantine_duration_secs: 3600,
        recovery_window_secs: 600,
        window_hours: 24,
    }
}

#[test]
fn record_success_counts_and_clears_nothing() {
    let dir = tempdir().unwrap();
    let h = ModelHealth::new(cfg_with(dir.path().join("mh.jsonl")));
    h.record_event("p", "m", HealthEventKind::Success);
    let w = h.window_for("p", "m").unwrap();
    assert_eq!(w.success, 1);
    assert!(w.quarantine_until.is_none());
    assert!(h.quarantined_until("p", "m").is_none());
}

#[test]
fn three_empties_trigger_quarantine() {
    let dir = tempdir().unwrap();
    let h = ModelHealth::new(cfg_with(dir.path().join("mh.jsonl")));
    for _ in 0..3 {
        h.record_event("zai", "glm-5-turbo", HealthEventKind::Empty);
    }
    let until = h.quarantined_until("zai", "glm-5-turbo");
    assert!(until.is_some(), "three empties must trigger quarantine");
}

#[test]
fn five_errors_trigger_quarantine() {
    let dir = tempdir().unwrap();
    let h = ModelHealth::new(cfg_with(dir.path().join("mh.jsonl")));
    for _ in 0..5 {
        h.record_event("p", "m", HealthEventKind::Error);
    }
    assert!(h.quarantined_until("p", "m").is_some());
}

#[test]
fn below_threshold_does_not_quarantine() {
    let dir = tempdir().unwrap();
    let h = ModelHealth::new(cfg_with(dir.path().join("mh.jsonl")));
    h.record_event("p", "m", HealthEventKind::Empty);
    h.record_event("p", "m", HealthEventKind::Empty);
    assert!(h.quarantined_until("p", "m").is_none());
    h.record_event("p", "m", HealthEventKind::Error);
    assert!(h.quarantined_until("p", "m").is_none());
}

#[test]
fn log_roundtrip_persists_across_load() {
    let dir = tempdir().unwrap();
    let log = dir.path().join("mh.jsonl");
    let cfg = cfg_with(log.clone());
    {
        let h = ModelHealth::new(cfg.clone());
        h.record_event("p", "m", HealthEventKind::Success);
        h.record_event("p", "m", HealthEventKind::Empty);
        h.record_event("p", "m", HealthEventKind::Error);
    }
    let h2 = ModelHealth::load(cfg);
    let w = h2.window_for("p", "m").unwrap();
    assert_eq!(w.success, 1);
    assert_eq!(w.empty, 1);
    assert_eq!(w.error, 1);
}

#[test]
fn load_skips_events_outside_window() {
    let dir = tempdir().unwrap();
    let log = dir.path().join("mh.jsonl");
    // Hand-craft one ancient line + one fresh line.
    let ancient = HealthEvent {
        ts: Utc::now() - ChronoDuration::hours(48),
        provider: "p".into(),
        model: "m".into(),
        kind: HealthEventKind::Error,
        latency_ms: None,
        detail: None,
    };
    let fresh = HealthEvent {
        ts: Utc::now(),
        provider: "p".into(),
        model: "m".into(),
        kind: HealthEventKind::Success,
        latency_ms: None,
        detail: None,
    };
    append_jsonl(&log, &ancient).unwrap();
    append_jsonl(&log, &fresh).unwrap();
    let cfg = cfg_with(log);
    let h = ModelHealth::load(cfg);
    let w = h.window_for("p", "m").unwrap();
    assert_eq!(w.success, 1, "fresh event counted");
    assert_eq!(w.error, 0, "48h-old error dropped by rolling window");
}

#[test]
fn disabled_tracker_is_noop() {
    let dir = tempdir().unwrap();
    let mut cfg = cfg_with(dir.path().join("mh.jsonl"));
    cfg.enabled = false;
    let h = ModelHealth::new(cfg);
    for _ in 0..10 {
        h.record_event("p", "m", HealthEventKind::Error);
    }
    assert!(h.quarantined_until("p", "m").is_none());
    assert!(h.window_for("p", "m").is_none());
}

#[test]
fn snapshot_returns_all_known_pairs() {
    let dir = tempdir().unwrap();
    let h = ModelHealth::new(cfg_with(dir.path().join("mh.jsonl")));
    h.record_event("p1", "m1", HealthEventKind::Success);
    h.record_event("p2", "m2", HealthEventKind::Error);
    let snap = h.snapshot();
    assert_eq!(snap.len(), 2);
}

#[test]
fn state_label_tracks_transitions() {
    let now = Utc::now();
    let mut w = HealthWindow::default();
    assert_eq!(w.state_label(now), "idle");
    w.success = 3;
    assert_eq!(w.state_label(now), "healthy");
    w.error = 1;
    assert_eq!(w.state_label(now), "degraded");
    w.quarantine_until = Some(now + ChronoDuration::hours(1));
    assert_eq!(w.state_label(now), "quarantined");
}

#[test]
fn observer_receives_first_success_and_quarantine_events() {
    use super::super::enrichment::{ObservationKind, ObservationRecorder};
    let dir = tempdir().unwrap();
    let log = dir.path().join("mh.jsonl");
    let obs_log = dir.path().join("obs.jsonl");
    let h = ModelHealth::new(cfg_with(log));
    h.attach_observer(Arc::new(ObservationRecorder::new(obs_log.clone())));

    // First success emits `FirstSuccess`.
    h.record_event("p", "m", HealthEventKind::Success);
    // Second success does not emit another `FirstSuccess`.
    h.record_event("p", "m", HealthEventKind::Success);
    // Three empties quarantine → `KeyDied` + three `NewFailureMode`.
    for _ in 0..3 {
        h.record_event("p", "m", HealthEventKind::Empty);
    }
    let raw = std::fs::read_to_string(&obs_log).unwrap();
    let kinds: Vec<ObservationKind> = raw
        .lines()
        .filter_map(|l| serde_json::from_str::<super::super::enrichment::ModelObservation>(l).ok())
        .map(|o| o.kind)
        .collect();
    let first_success_count = kinds
        .iter()
        .filter(|k| **k == ObservationKind::FirstSuccess)
        .count();
    let key_died_count = kinds
        .iter()
        .filter(|k| **k == ObservationKind::KeyDied)
        .count();
    let new_failure_count = kinds
        .iter()
        .filter(|k| **k == ObservationKind::NewFailureMode)
        .count();
    assert_eq!(first_success_count, 1, "exactly one FirstSuccess");
    assert_eq!(key_died_count, 1, "quarantine fires once per transition");
    assert_eq!(
        new_failure_count, 3,
        "each empty produces one NewFailureMode signal",
    );
}

#[test]
fn rolling_window_trims_old_events() {
    let dir = tempdir().unwrap();
    let mut cfg = cfg_with(dir.path().join("mh.jsonl"));
    cfg.window_hours = 1;
    let h = ModelHealth::new(cfg);
    // Directly poke the ring to insert a 2h-old empty, then a
    // fresh empty via the public API — the fresh call should trim
    // the old one.
    {
        let mut rings = h.rings.write().unwrap();
        let ring = rings.entry(("p".into(), "m".into())).or_default();
        ring.events.push((
            Utc::now() - ChronoDuration::hours(2),
            HealthEventKind::Empty,
        ));
    }
    h.record_event("p", "m", HealthEventKind::Empty);
    let w = h.window_for("p", "m").unwrap();
    assert_eq!(w.empty, 1, "2h-old empty trimmed out of 1h window");
}
