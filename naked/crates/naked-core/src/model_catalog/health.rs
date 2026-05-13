//! Runtime health tracker for every `(provider, model)` pair. Phase 3
//! of the Model Capabilities Catalog plan.
//!
//! # What it does
//!
//! Every successful / empty / errored turn in [`crate::loop_::AgentLoop`]
//! is reported here. The tracker:
//!
//! 1. Appends one JSONL line to `~/.naked/model_health.jsonl` so the
//!    signal survives restarts (the housekeep script compacts + exports
//!    a Prometheus gauge file once per day — Phase 3 metrics).
//! 2. Maintains an in-memory rolling 24-hour window of per-pair
//!    counters (`success`, `empty`, `error`, `timeout`, `last_success_at`).
//! 3. Applies a simple quarantine policy — 3 empties **or** 5 errors in
//!    24 h flips the pair into a 1-hour quarantine. A subsequent success
//!    within 10 minutes of quarantine clears it (a model that quietly
//!    recovers shouldn't stay blocked).
//! 4. Exposes `quarantined_until(provider, model)` so the
//!    [`crate::model_catalog::ModelSelector`] can skip quarantined
//!    pairs before burning a 90-second HTTP round-trip.
//!
//! # Why in-memory + append-only JSONL
//!
//! The counters are hot-path — read per selection, updated per turn —
//! so they live in a [`parking_lot::RwLock`]-free `std::sync::RwLock`.
//! The JSONL log is the durable source: on restart, [`ModelHealth::load`]
//! replays the last 24 hours of entries into the in-memory window. No
//! database, no schema migrations, grep-able by operators.
//!
//! # Back-compat
//!
//! Every hook is feature-gated on an `Option<Arc<ModelHealth>>` so
//! existing callers that didn't plumb the tracker (tests, CLI) keep
//! compiling and running unchanged. The [`ModelSelector::with_health`]
//! constructor attaches it when present.

use std::collections::HashMap;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::{Arc, RwLock};

use chrono::{DateTime, Duration as ChronoDuration, Utc};
use serde::{Deserialize, Serialize};

use super::enrichment::{ObservationKind, ObservationRecorder};

/// Which kind of signal a turn produced. Mapped onto the three existing
/// termination branches in [`crate::loop_::AgentLoop::run`] (see
/// `p3-health` todo in the plan file for the exact call sites).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum HealthEventKind {
    /// Turn ended with `Idle` after producing at least one content
    /// block or tool call. The operator got a real reply.
    Success,
    /// Provider closed the stream with zero text + zero tool calls + zero
    /// reasoning chunks, and the [`crate::loop_::MAX_EMPTY_CONTENT_RETRIES`]
    /// budget was exhausted. Canonical `glm-5-turbo` failure mode.
    Empty,
    /// Mid-stream provider error or `stream retries exhausted`. Counts
    /// for both 4xx and 5xx.
    Error,
    /// Wall-clock timeout fired (coordinator-level, research runs).
    Timeout,
}

impl HealthEventKind {
    pub fn as_str(self) -> &'static str {
        match self {
            HealthEventKind::Success => "success",
            HealthEventKind::Empty => "empty",
            HealthEventKind::Error => "error",
            HealthEventKind::Timeout => "timeout",
        }
    }
}

/// One JSONL line. Kept flat and boring so `jq` and `rg` can slice it
/// without schema surprises.
#[derive(Debug, Clone, Serialize, Deserialize)]
struct HealthEvent {
    ts: DateTime<Utc>,
    provider: String,
    model: String,
    kind: HealthEventKind,
    /// Turn latency in milliseconds (if the caller measured it).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    latency_ms: Option<u64>,
    /// Short human-readable detail (e.g. the provider error message).
    /// Truncated to 200 chars to keep the log tidy.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    detail: Option<String>,
}

/// Rolling 24-hour aggregate for one `(provider, model)` pair. Read by
/// [`crate::model_catalog::ModelSelector::is_available`] when it
/// considers a pair for a task — a quarantined pair is filtered out of
/// fallback chains before dispatch.
#[derive(Debug, Clone, Default, Serialize)]
pub struct HealthWindow {
    pub success: u32,
    pub empty: u32,
    pub error: u32,
    pub timeout: u32,
    pub last_success_at: Option<DateTime<Utc>>,
    pub last_event_at: Option<DateTime<Utc>>,
    pub quarantine_until: Option<DateTime<Utc>>,
}

impl HealthWindow {
    pub fn total_events(&self) -> u32 {
        self.success + self.empty + self.error + self.timeout
    }

    /// Human-readable state for Prometheus gauge label
    /// (`naked_model_health{state="..."}`). Lives here so the exporter
    /// doesn't have to re-derive the policy rules.
    pub fn state_label(&self, now: DateTime<Utc>) -> &'static str {
        if self.quarantine_until.map(|q| q > now).unwrap_or(false) {
            "quarantined"
        } else if self.error > 0 || self.empty > 0 {
            "degraded"
        } else if self.success > 0 {
            "healthy"
        } else {
            "idle"
        }
    }
}

/// Quarantine + retention configuration. Serialized under
/// `Config.model_health` in `naked.json`; all fields defaulted so
/// existing configs need no changes.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelHealthConfig {
    /// Master switch. Default `true` — the tracker itself is cheap. Set
    /// to `false` to short-circuit all record/quarantine logic for
    /// ultra-constrained environments.
    #[serde(default = "default_true")]
    pub enabled: bool,

    /// Path to the append-only log. Defaults to
    /// `~/.naked/model_health.jsonl`. The exporter (Phase 3 metrics)
    /// reads it; housekeep compacts it.
    #[serde(default)]
    pub log_path: Option<PathBuf>,

    /// Quarantine triggers: >= this many empties in the 24h window.
    #[serde(default = "default_empty_threshold")]
    pub empty_threshold: u32,

    /// Quarantine triggers: >= this many errors in the 24h window.
    #[serde(default = "default_error_threshold")]
    pub error_threshold: u32,

    /// How long a quarantine lasts once triggered, in seconds.
    #[serde(default = "default_quarantine_secs")]
    pub quarantine_duration_secs: u64,

    /// A Success observed this recently (seconds before `now`) clears
    /// an active quarantine — a model that spontaneously recovers
    /// shouldn't stay locked out.
    #[serde(default = "default_recovery_secs")]
    pub recovery_window_secs: u64,

    /// Rolling window horizon, in hours. Events older than this are
    /// trimmed on every `record()` call so counters reflect recent
    /// reality rather than historical trauma.
    #[serde(default = "default_window_hours")]
    pub window_hours: u32,
}

fn default_true() -> bool {
    true
}
fn default_empty_threshold() -> u32 {
    3
}
fn default_error_threshold() -> u32 {
    5
}
fn default_quarantine_secs() -> u64 {
    3600
}
fn default_recovery_secs() -> u64 {
    600
}
fn default_window_hours() -> u32 {
    24
}

impl Default for ModelHealthConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            log_path: None,
            empty_threshold: default_empty_threshold(),
            error_threshold: default_error_threshold(),
            quarantine_duration_secs: default_quarantine_secs(),
            recovery_window_secs: default_recovery_secs(),
            window_hours: default_window_hours(),
        }
    }
}

/// Per-pair event ring, used to trim old entries when recomputing the
/// rolling window. Kept `VecDeque`-free — `Vec<DateTime>` + binary
/// search by timestamp is fine for the low event rate we expect
/// (order-of-magnitude: ~100/day/pair in the hottest config).
#[derive(Debug, Default, Clone)]
struct PairRing {
    events: Vec<(DateTime<Utc>, HealthEventKind)>,
    last_success_at: Option<DateTime<Utc>>,
    quarantine_until: Option<DateTime<Utc>>,
    /// Have we ever observed a success on this pair (across all time,
    /// not just the rolling window)? Powers `FirstSuccess`
    /// observations.
    ever_succeeded: bool,
}

/// Main runtime handle. Clone-cheap via `Arc<ModelHealth>`.
pub struct ModelHealth {
    config: ModelHealthConfig,
    rings: RwLock<HashMap<(String, String), PairRing>>,
    /// Optional sink for the Phase 4 auto-enrichment pipeline. When
    /// attached, state transitions emit [`ObservationKind`] events so
    /// the daily digest can promote them into catalog suggestions.
    observer: RwLock<Option<Arc<ObservationRecorder>>>,
}

impl std::fmt::Debug for ModelHealth {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ModelHealth")
            .field("config", &self.config)
            .field("pairs", &self.rings.read().map(|r| r.len()).unwrap_or(0))
            .finish()
    }
}

impl ModelHealth {
    /// Create a fresh, empty tracker. Used by tests and by the CLI when
    /// health is disabled or no log path is configured. Does not read
    /// any existing log.
    pub fn new(config: ModelHealthConfig) -> Self {
        Self {
            config,
            rings: RwLock::new(HashMap::new()),
            observer: RwLock::new(None),
        }
    }

    /// Attach an [`ObservationRecorder`] so state transitions feed the
    /// auto-enrichment pipeline. Safe to call after construction; the
    /// first call wins quietly if two threads race.
    pub fn attach_observer(&self, observer: Arc<ObservationRecorder>) {
        if let Ok(mut slot) = self.observer.write() {
            *slot = Some(observer);
        }
    }

    fn emit_observation(
        &self,
        provider: &str,
        model: &str,
        kind: ObservationKind,
        detail: Option<String>,
    ) {
        // REGISTRY-WAIVE: intentional fallback: poisoned RwLock → empty result
        let Ok(slot) = self.observer.read() else {
            return;
        };
        if let Some(rec) = slot.as_ref() {
            rec.record(provider, model, kind, detail);
        }
    }

    /// Construct a tracker and replay the last `window_hours` of
    /// events from `log_path`. Missing / unreadable logs are treated as
    /// "no history", with a `warn!` to help operators notice
    /// permissions / disk issues. Malformed lines are skipped with a
    /// `debug!` log — we never crash on a corrupt log.
    pub fn load(config: ModelHealthConfig) -> Self {
        let tracker = Self::new(config.clone());
        if !config.enabled {
            return tracker;
        }
        let Some(path) = Self::resolve_log_path(&config) else {
            return tracker;
        };
        let Ok(contents) = std::fs::read_to_string(&path) else {
            tracing::debug!(path = %path.display(), "model_health: log not present yet");
            return tracker;
        };
        let horizon = Utc::now() - ChronoDuration::hours(i64::from(config.window_hours.max(1)));
        let mut replayed = 0usize;
        let mut rings = tracker.rings.write().expect("rings poisoned");
        for (lineno, line) in contents.lines().enumerate() {
            let line = line.trim();
            if line.is_empty() {
                continue;
            }
            match serde_json::from_str::<HealthEvent>(line) {
                Ok(ev) if ev.ts >= horizon => {
                    let ring = rings
                        .entry((ev.provider.clone(), ev.model.clone()))
                        .or_default();
                    ring.events.push((ev.ts, ev.kind));
                    if ev.kind == HealthEventKind::Success {
                        ring.last_success_at =
                            Some(ring.last_success_at.map_or(ev.ts, |e| e.max(ev.ts)));
                        ring.ever_succeeded = true;
                    }
                    replayed += 1;
                }
                Ok(_) => {}
                Err(e) => {
                    tracing::debug!(
                        line = lineno + 1,
                        error = %e,
                        "model_health: skipping malformed log line"
                    );
                }
            }
        }
        drop(rings);
        tracing::info!(
            path = %path.display(),
            replayed,
            "model_health: loaded rolling window from disk",
        );
        tracker
    }

    /// Record one event. Synchronously appends to the JSONL log (best
    /// effort — any I/O error is logged but never fatal). Updates the
    /// in-memory window and re-evaluates quarantine.
    ///
    /// `latency_ms` and `detail` are optional diagnostic fields. Use
    /// [`ModelHealth::record_event`] for the convenience form without
    /// these.
    pub fn record(
        &self,
        provider: &str,
        model: &str,
        kind: HealthEventKind,
        latency_ms: Option<u64>,
        detail: Option<String>,
    ) {
        if !self.config.enabled {
            return;
        }
        let now = Utc::now();
        let detail_for_obs = detail.clone();
        let event = HealthEvent {
            ts: now,
            provider: provider.to_string(),
            model: model.to_string(),
            kind,
            latency_ms,
            detail: detail.map(|d| truncate_detail(&d)),
        };
        self.append_log(&event);

        let (first_success, resurrected, newly_quarantined, new_failure_detail) = {
            let mut rings = crate::write_or_recover(&self.rings);
            let ring = rings
                .entry((provider.to_string(), model.to_string()))
                .or_default();
            ring.events.push((now, kind));
            let was_quarantined = ring.quarantine_until.map(|u| u > now).unwrap_or(false);
            let first_success = kind == HealthEventKind::Success && !ring.ever_succeeded;
            let mut resurrected = false;
            if kind == HealthEventKind::Success {
                ring.last_success_at = Some(now);
                ring.ever_succeeded = true;
                if let Some(until) = ring.quarantine_until
                    && until > now
                {
                    let recovery = ChronoDuration::seconds(self.config.recovery_window_secs as i64);
                    // If the success came within the configured
                    // recovery window from the quarantine start,
                    // clear it. The start time is approximated as
                    // `until - duration`.
                    let started = until
                        - ChronoDuration::seconds(self.config.quarantine_duration_secs as i64);
                    if now - started <= recovery {
                        tracing::info!(
                            provider = provider,
                            model = model,
                            "model_health: success within recovery window cleared quarantine"
                        );
                        ring.quarantine_until = None;
                        resurrected = true;
                    }
                }
            }
            self.trim_ring(ring, now);
            self.maybe_quarantine(ring, provider, model, now);
            let newly_quarantined =
                !was_quarantined && ring.quarantine_until.map(|u| u > now).unwrap_or(false);
            // Observe a "new failure mode" when a non-success event
            // carries a detail string. Deduplication happens in the
            // promoter — this only dispatches the raw signal.
            let new_failure_detail = match kind {
                HealthEventKind::Empty | HealthEventKind::Error | HealthEventKind::Timeout => {
                    detail_for_obs.or_else(|| Some(kind.as_str().to_string()))
                }
                HealthEventKind::Success => None,
            };
            (
                first_success,
                resurrected,
                newly_quarantined,
                new_failure_detail,
            )
        };

        if first_success {
            self.emit_observation(provider, model, ObservationKind::FirstSuccess, None);
        }
        if resurrected {
            self.emit_observation(provider, model, ObservationKind::Resurrected, None);
        }
        if newly_quarantined {
            self.emit_observation(
                provider,
                model,
                ObservationKind::KeyDied,
                Some(format!(
                    "quarantine triggered by {kind}",
                    kind = kind.as_str()
                )),
            );
        }
        if let Some(d) = new_failure_detail {
            self.emit_observation(provider, model, ObservationKind::NewFailureMode, Some(d));
        }
    }

    /// Convenience form: record with no latency/detail info.
    pub fn record_event(&self, provider: &str, model: &str, kind: HealthEventKind) {
        self.record(provider, model, kind, None, None);
    }

    /// Return the pair's quarantine deadline if it's currently
    /// quarantined. `None` means "fine to use" from the health tracker's
    /// perspective (the selector may still reject it on status / fit
    /// grounds).
    pub fn quarantined_until(&self, provider: &str, model: &str) -> Option<DateTime<Utc>> {
        let now = Utc::now();
        let rings = self.rings.read().ok()?;
        let ring = rings.get(&(provider.to_string(), model.to_string()))?;
        match ring.quarantine_until {
            Some(u) if u > now => Some(u),
            _ => None,
        }
    }

    /// Snapshot every known pair's window. Used by the Prometheus
    /// exporter (Phase 3 metrics) and by tests. The snapshot is a
    /// point-in-time clone; concurrent `record()` calls after this
    /// returns are not reflected.
    pub fn snapshot(&self) -> Vec<((String, String), HealthWindow)> {
        // REGISTRY-WAIVE: intentional fallback: poisoned RwLock → empty result
        let Ok(rings) = self.rings.read() else {
            return Vec::new();
        };
        let now = Utc::now();
        rings
            .iter()
            .map(|((p, m), ring)| ((p.clone(), m.clone()), self.ring_to_window(ring, now)))
            .collect()
    }

    /// Current health window for a single pair. `None` if the pair has
    /// no recorded events yet.
    pub fn window_for(&self, provider: &str, model: &str) -> Option<HealthWindow> {
        let rings = self.rings.read().ok()?;
        let ring = rings.get(&(provider.to_string(), model.to_string()))?;
        Some(self.ring_to_window(ring, Utc::now()))
    }

    fn ring_to_window(&self, ring: &PairRing, now: DateTime<Utc>) -> HealthWindow {
        let horizon = now - ChronoDuration::hours(i64::from(self.config.window_hours.max(1)));
        let mut w = HealthWindow {
            last_success_at: ring.last_success_at,
            quarantine_until: ring.quarantine_until,
            last_event_at: ring.events.last().map(|(t, _)| *t),
            ..Default::default()
        };
        for (ts, kind) in &ring.events {
            if *ts < horizon {
                continue;
            }
            match kind {
                HealthEventKind::Success => w.success += 1,
                HealthEventKind::Empty => w.empty += 1,
                HealthEventKind::Error => w.error += 1,
                HealthEventKind::Timeout => w.timeout += 1,
            }
        }
        w
    }

    fn trim_ring(&self, ring: &mut PairRing, now: DateTime<Utc>) {
        let horizon = now - ChronoDuration::hours(i64::from(self.config.window_hours.max(1)));
        ring.events.retain(|(ts, _)| *ts >= horizon);
    }

    fn maybe_quarantine(
        &self,
        ring: &mut PairRing,
        provider: &str,
        model: &str,
        now: DateTime<Utc>,
    ) {
        // Count within rolling window (ring is already trimmed by
        // `trim_ring`, but be defensive in case the horizon ticks mid-
        // call).
        let mut empties = 0u32;
        let mut errors = 0u32;
        for (_, kind) in &ring.events {
            match kind {
                HealthEventKind::Empty => empties += 1,
                HealthEventKind::Error | HealthEventKind::Timeout => errors += 1,
                _ => {}
            }
        }
        let trigger =
            empties >= self.config.empty_threshold || errors >= self.config.error_threshold;
        if trigger {
            let already_quarantined = ring.quarantine_until.map(|u| u > now).unwrap_or(false);
            if !already_quarantined {
                let until =
                    now + ChronoDuration::seconds(self.config.quarantine_duration_secs as i64);
                ring.quarantine_until = Some(until);
                tracing::warn!(
                    provider = provider,
                    model = model,
                    empties,
                    errors,
                    until = %until,
                    "model_health: pair quarantined"
                );
            }
        }
    }

    fn resolve_log_path(config: &ModelHealthConfig) -> Option<PathBuf> {
        if let Some(p) = &config.log_path {
            return Some(p.clone());
        }
        // Default: ~/.naked/model_health.jsonl. Use `HOME` env for
        // cross-platform; falls back to current dir if unset.
        let home = std::env::var_os("HOME")?;
        Some(
            PathBuf::from(home)
                .join(".naked")
                .join("model_health.jsonl"),
        )
    }

    fn append_log(&self, event: &HealthEvent) {
        let Some(path) = Self::resolve_log_path(&self.config) else {
            return;
        };
        if let Err(e) = append_jsonl(&path, event) {
            tracing::warn!(
                path = %path.display(),
                error = %e,
                "model_health: failed to append to log (tracking continues in memory)",
            );
        }
    }
}

fn append_jsonl(path: &Path, event: &HealthEvent) -> std::io::Result<()> {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let line = serde_json::to_string(event)
        .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;
    let mut f = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(path)?;
    writeln!(f, "{line}")?;
    Ok(())
}

fn truncate_detail(s: &str) -> String {
    const MAX: usize = 200;
    if s.len() <= MAX {
        s.to_string()
    } else {
        // byte-safe truncation on char boundary
        let mut end = MAX;
        while end > 0 && !s.is_char_boundary(end) {
            end -= 1;
        }
        format!("{}…", &s[..end])
    }
}

#[cfg(test)]
#[path = "health_tests.rs"]
mod tests;
