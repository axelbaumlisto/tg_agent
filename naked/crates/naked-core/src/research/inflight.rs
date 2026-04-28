//! Persistent state-machine for individual scheduler launches.
//!
//! Every time the research scheduler decides to run a spec it writes
//! an [`Inflight`] record to `<spec_dir>/inflight.json`. The record
//! transitions through four states:
//!
//! ```text
//!  Scheduled  ──spawn──▶  Running  ──finish──▶  Completed
//!                 │                    │
//!                 │                    └─error──▶  Failed
//!                 │                    └─heartbeat-stale──▶  Timeout
//!                 │
//!                 └─crash─before-spawn──▶  resurrected on next boot
//! ```
//!
//! The on-disk record survives process crashes; on startup the
//! scheduler enumerates every research and re-fires anything still in
//! a non-terminal state. Each resurrection bumps `attempt`; once the
//! `max_resurrection_attempts` cap is hit the entry is finalised as
//! `Failed` and the operator is alerted.
//!
//! Heartbeat semantics: while a task is `Running`, the scheduler's
//! sweep phase refreshes `last_heartbeat` once per tick. A peer
//! observing the file (operator, monitoring) can therefore tell a
//! healthy long-running job from a hung one without poking the
//! process.

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

/// Lifecycle states of a single scheduler attempt. Stored verbatim
/// in `inflight.json`; the four lowercase string values are part of
/// the on-disk contract — never rename them.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RunState {
    /// Scheduler decided to fire this spec; the slot is reserved
    /// in the in-memory `running` map and the inflight ledger is
    /// flushed to disk *before* the worker is spawned. If the
    /// process crashes between this write and the worker entering
    /// `Running`, the resurrection pass picks it up.
    Scheduled,
    /// Worker future is actively running. `started_at` is set; the
    /// sweep phase refreshes `last_heartbeat` each tick.
    Running,
    /// Terminal: worker returned `Ok` and `runs.jsonl` was appended.
    Completed,
    /// Terminal: worker returned an error, hit a timeout, or was
    /// declared failed by the resurrection cap. The `error` field
    /// holds the operator-visible reason.
    Failed,
}

impl RunState {
    /// `true` once we should NOT touch the entry anymore (apart from
    /// archival or operator inspection).
    pub fn is_terminal(self) -> bool {
        matches!(self, Self::Completed | Self::Failed)
    }

    /// Human-friendly Russian label (mirrors the wording the user
    /// asked for: «Запланировано / В работе / Завершена / Ошибка»).
    pub fn ru_label(self) -> &'static str {
        match self {
            Self::Scheduled => "Запланировано",
            Self::Running => "В работе",
            Self::Completed => "Завершена",
            Self::Failed => "Ошибка",
        }
    }
}

/// One scheduler attempt, persisted as `inflight.json` next to
/// `spec.json`. Older fields are `#[serde(default)]` so a binary
/// written before this module landed reads cleanly as `None` /
/// `0` / etc.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Inflight {
    pub spec_id: String,
    /// Stable id for *this* attempt — distinct from the eventual
    /// `RunRecord.run_id` (which is minted by the coordinator on
    /// success). Used to correlate scheduler logs across resurrects.
    pub attempt_id: String,
    pub state: RunState,
    pub scheduled_at: DateTime<Utc>,
    #[serde(default)]
    pub started_at: Option<DateTime<Utc>>,
    #[serde(default)]
    pub last_heartbeat: Option<DateTime<Utc>>,
    #[serde(default)]
    pub finished_at: Option<DateTime<Utc>>,
    /// Last error message — populated on `Failed`, cleared on success.
    #[serde(default)]
    pub error: Option<String>,
    /// 1 on first scheduling, +1 each resurrection. Capped by
    /// `SchedulerConfig.max_resurrection_attempts`.
    #[serde(default = "default_attempt")]
    pub attempt: u32,
    /// Optional id of the underlying coordinator run, set when the
    /// worker emits one. Used by tooling to link the scheduler
    /// attempt to the corresponding `runs.jsonl` row.
    #[serde(default)]
    pub run_id: Option<String>,
    /// `true` when this `Scheduled` record was re-queued by
    /// `resurrect_at_boot` after a process crash. The scheduler's
    /// dispatch loop drains these entries first (still under the
    /// `max_concurrent_runs` cap) so resurrection can never exceed
    /// the configured concurrency. Cleared back to `false` when the
    /// dispatch loop hands the entry to a fresh worker (overwritten
    /// by `Inflight::scheduled` inside `spawn_task`).
    #[serde(default)]
    pub scheduled_after_resurrection: bool,
}

fn default_attempt() -> u32 {
    1
}

impl Inflight {
    /// Fresh `Scheduled` record for a brand-new attempt.
    pub fn scheduled(spec_id: impl Into<String>, attempt: u32) -> Self {
        let now = Utc::now();
        Self {
            spec_id: spec_id.into(),
            attempt_id: short_attempt_id(),
            state: RunState::Scheduled,
            scheduled_at: now,
            started_at: None,
            last_heartbeat: None,
            finished_at: None,
            error: None,
            attempt,
            run_id: None,
            scheduled_after_resurrection: false,
        }
    }

    /// In-place re-queue for the resurrection pull-model: keep the
    /// `attempt_id` (so logs across boots stay correlated) but reset
    /// timestamps + bump the attempt counter so the dispatch loop sees
    /// it as a fresh `Scheduled` candidate. The `scheduled_after_resurrection`
    /// flag tags the entry so the loop drains it under the same
    /// concurrency cap as fresh dispatches — *not* by spawning directly.
    pub fn mark_scheduled_for_resurrection(&mut self, next_attempt: u32) {
        let now = Utc::now();
        self.state = RunState::Scheduled;
        self.scheduled_at = now;
        self.started_at = None;
        self.last_heartbeat = None;
        self.finished_at = None;
        self.error = None;
        self.run_id = None;
        self.attempt = next_attempt;
        self.scheduled_after_resurrection = true;
    }

    /// Mutate to `Running` and stamp `started_at` + first heartbeat.
    pub fn mark_running(&mut self) {
        let now = Utc::now();
        self.state = RunState::Running;
        self.started_at = Some(now);
        self.last_heartbeat = Some(now);
        self.finished_at = None;
        self.error = None;
    }

    /// Stamp a fresh heartbeat. Cheap; called every scheduler tick.
    pub fn heartbeat(&mut self) {
        self.last_heartbeat = Some(Utc::now());
    }

    /// Terminal: success.
    pub fn mark_completed(&mut self, run_id: Option<String>) {
        self.state = RunState::Completed;
        self.finished_at = Some(Utc::now());
        self.run_id = run_id;
        self.error = None;
    }

    /// Terminal: failure. `reason` is operator-visible, free-form.
    pub fn mark_failed(&mut self, reason: impl Into<String>) {
        self.state = RunState::Failed;
        self.finished_at = Some(Utc::now());
        self.error = Some(reason.into());
    }

    /// Whether the on-disk record is "stale" given a heartbeat
    /// budget. Used by the boot resurrector and by the live sweep
    /// (a `Running` entry whose heartbeat hasn't been refreshed in
    /// `≥ budget` is hung).
    pub fn is_stale(&self, now: DateTime<Utc>, budget: chrono::Duration) -> bool {
        match self.state {
            RunState::Running => self
                .last_heartbeat
                .map(|hb| now - hb > budget)
                .unwrap_or(true),
            RunState::Scheduled => now - self.scheduled_at > budget,
            _ => false,
        }
    }
}

/// 8-char hex from a fresh UUID. Wrapped so tests can shim if needed.
fn short_attempt_id() -> String {
    uuid::Uuid::new_v4().to_string()[..8].to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn state_terminal_predicate() {
        assert!(!RunState::Scheduled.is_terminal());
        assert!(!RunState::Running.is_terminal());
        assert!(RunState::Completed.is_terminal());
        assert!(RunState::Failed.is_terminal());
    }

    #[test]
    fn ru_labels_cover_every_state() {
        // Just touch every variant so future contributors notice
        // when the enum grows.
        assert_eq!(RunState::Scheduled.ru_label(), "Запланировано");
        assert_eq!(RunState::Running.ru_label(), "В работе");
        assert_eq!(RunState::Completed.ru_label(), "Завершена");
        assert_eq!(RunState::Failed.ru_label(), "Ошибка");
    }

    #[test]
    fn lifecycle_transitions_set_timestamps() {
        let mut i = Inflight::scheduled("s1", 1);
        assert_eq!(i.state, RunState::Scheduled);
        assert_eq!(i.attempt, 1);
        assert!(i.started_at.is_none());
        assert!(i.last_heartbeat.is_none());

        i.mark_running();
        assert_eq!(i.state, RunState::Running);
        assert!(i.started_at.is_some());
        assert!(i.last_heartbeat.is_some());

        let hb1 = i.last_heartbeat.unwrap();
        std::thread::sleep(std::time::Duration::from_millis(2));
        i.heartbeat();
        let hb2 = i.last_heartbeat.unwrap();
        assert!(hb2 >= hb1);

        i.mark_completed(Some("run-xyz".into()));
        assert_eq!(i.state, RunState::Completed);
        assert!(i.finished_at.is_some());
        assert_eq!(i.run_id.as_deref(), Some("run-xyz"));
    }

    #[test]
    fn mark_failed_records_reason() {
        let mut i = Inflight::scheduled("s1", 1);
        i.mark_running();
        i.mark_failed("provider stream returned 503");
        assert_eq!(i.state, RunState::Failed);
        assert!(i.error.as_deref().unwrap().contains("503"));
    }

    #[test]
    fn is_stale_running_uses_heartbeat() {
        let mut i = Inflight::scheduled("s1", 1);
        i.mark_running();
        i.last_heartbeat = Some(Utc::now() - chrono::Duration::minutes(5));
        assert!(i.is_stale(Utc::now(), chrono::Duration::minutes(2)));
        assert!(!i.is_stale(Utc::now(), chrono::Duration::minutes(10)));
    }

    #[test]
    fn is_stale_terminal_states_are_never_stale() {
        let mut i = Inflight::scheduled("s1", 1);
        i.mark_running();
        i.mark_completed(None);
        i.last_heartbeat = Some(Utc::now() - chrono::Duration::days(7));
        assert!(!i.is_stale(Utc::now(), chrono::Duration::minutes(1)));
    }

    #[test]
    fn resurrection_remark_resets_lifecycle_and_bumps_attempt() {
        let mut i = Inflight::scheduled("s1", 1);
        i.mark_running();
        i.last_heartbeat = Some(Utc::now() - chrono::Duration::hours(1));
        i.error = Some("hung".into());

        i.mark_scheduled_for_resurrection(2);
        assert_eq!(i.state, RunState::Scheduled);
        assert_eq!(i.attempt, 2);
        assert!(i.scheduled_after_resurrection);
        assert!(i.started_at.is_none());
        assert!(i.last_heartbeat.is_none());
        assert!(i.finished_at.is_none());
        assert!(i.error.is_none());
        assert!(i.run_id.is_none());
    }

    #[test]
    fn legacy_record_without_resurrection_flag_defaults_false() {
        // Older binary may have written records before the flag existed.
        let raw = r#"{
          "spec_id": "old",
          "attempt_id": "deadbeef",
          "state": "scheduled",
          "scheduled_at": "2026-04-01T00:00:00Z"
        }"#;
        let infl: Inflight = serde_json::from_str(raw).unwrap();
        assert!(!infl.scheduled_after_resurrection);
    }

    #[test]
    fn serde_round_trip_preserves_state_string() {
        let mut i = Inflight::scheduled("s1", 2);
        i.mark_running();
        let s = serde_json::to_string(&i).unwrap();
        // On-disk contract: the snake_case state value MUST appear.
        assert!(s.contains("\"state\":\"running\""), "got: {s}");
        let back: Inflight = serde_json::from_str(&s).unwrap();
        assert_eq!(back.state, RunState::Running);
        assert_eq!(back.attempt, 2);
    }
}
