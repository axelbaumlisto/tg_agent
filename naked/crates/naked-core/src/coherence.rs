//! Plain-language session-coherence ladder (T7 of `PLAN_QUALITY_v1.md`).
//!
//! Pure transition function — no I/O, no async, no state outside the
//! enum. Driven by signals derived from existing Prometheus counters
//! (`empty_content_retry_total`, `steer_drained_on_abort_total`, …) so
//! we don't need a parallel state machine inside the loop.
//!
//! Used by `/health` to render a human-readable session-health row
//! beside the raw counters.

use serde::{Deserialize, Serialize};

#[derive(Debug, Default, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CoherenceState {
    #[default]
    Healthy,
    GettingCrowded,
    RefreshingContext,
    VerifyingRecentWork,
    ResettingPlan,
}

impl CoherenceState {
    #[must_use]
    pub fn label(self) -> &'static str {
        match self {
            Self::Healthy => "healthy",
            Self::GettingCrowded => "getting crowded",
            Self::RefreshingContext => "refreshing context",
            Self::VerifyingRecentWork => "verifying recent work",
            Self::ResettingPlan => "resetting plan",
        }
    }

    #[must_use]
    pub fn description(self) -> &'static str {
        match self {
            Self::Healthy => "Session stable and focused.",
            Self::GettingCrowded => "Approaching context pressure.",
            Self::RefreshingContext => "Refreshing context before continuing.",
            Self::VerifyingRecentWork => "Checking recent tool results.",
            Self::ResettingPlan => "Rebuilding from canonical context.",
        }
    }

    #[must_use]
    pub fn icon(self) -> &'static str {
        match self {
            Self::Healthy => "✅",
            Self::GettingCrowded => "🟡",
            Self::RefreshingContext => "🔄",
            Self::VerifyingRecentWork => "🔍",
            Self::ResettingPlan => "🔁",
        }
    }

    /// Legacy alias for [`icon`] kept for plan-v9 e2e tests + the
    /// `/usage`-style commands that already render `emoji()`.
    #[must_use]
    pub fn emoji(self) -> &'static str {
        self.icon()
    }
}

/// Pre-T7 derivation: pick a state purely from current token usage
/// vs the model's context window. Kept for compatibility with the
/// `plan_v9_e2e::coherence_tracks_capacity` test and the
/// `commands/session.rs::cmd_status` / `cmd_compact` callers that
/// render "context X% full" badges.
#[must_use]
pub fn from_tokens(tokens: u64, window: u64) -> CoherenceState {
    if window == 0 {
        return CoherenceState::Healthy;
    }
    let ratio = tokens as f64 / window as f64;
    from_capacity(ratio)
}

/// Pre-T7 derivation from a 0..1 capacity ratio. Same thresholds as
/// `from_tokens`. Public so external callers (TG `/status`) can
/// supply their own ratio computed from per-session metadata.
#[must_use]
pub fn from_capacity(ratio: f64) -> CoherenceState {
    if ratio >= 0.90 {
        CoherenceState::ResettingPlan
    } else if ratio >= 0.75 {
        CoherenceState::RefreshingContext
    } else if ratio >= 0.60 {
        CoherenceState::GettingCrowded
    } else {
        CoherenceState::Healthy
    }
}

/// Signal feeding the reducer. Constructed from process-wide counter
/// deltas (we don't store rate state here — the caller passes a
/// pre-classified signal).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CoherenceSignal {
    /// Nothing notable since last tick.
    NoChange,
    /// Empty-content retries climbing → context likely full.
    EmptyContentRetriesRising,
    /// Active compaction in flight.
    CompactionStarted,
    CompactionCompleted,
    CompactionFailed,
    /// Loop guard halted a tool — likely confused state.
    LoopGuardHalt,
    /// Steer drained on abort — user input rescue happened recently.
    SteerRescued,
}

/// Pure transition function. Takes the current state + a signal,
/// returns the next state. No persistence; the caller stores the
/// last value (e.g. in `shared::COHERENCE_STATE: OnceLock<RwLock<...>>`).
#[must_use]
pub fn next_coherence_state(current: CoherenceState, signal: CoherenceSignal) -> CoherenceState {
    use CoherenceSignal::*;
    use CoherenceState::*;
    match signal {
        NoChange => current,
        CompactionStarted => RefreshingContext,
        CompactionCompleted => Healthy,
        CompactionFailed => GettingCrowded,
        EmptyContentRetriesRising => match current {
            Healthy => GettingCrowded,
            other => other,
        },
        LoopGuardHalt => VerifyingRecentWork,
        SteerRescued => match current {
            Healthy => Healthy,
            VerifyingRecentWork | ResettingPlan => current,
            _ => Healthy,
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn no_change_preserves_current() {
        for state in [
            CoherenceState::Healthy,
            CoherenceState::GettingCrowded,
            CoherenceState::RefreshingContext,
        ] {
            assert_eq!(
                next_coherence_state(state, CoherenceSignal::NoChange),
                state
            );
        }
    }

    #[test]
    fn compaction_lifecycle() {
        let s = CoherenceState::Healthy;
        let s = next_coherence_state(s, CoherenceSignal::CompactionStarted);
        assert_eq!(s, CoherenceState::RefreshingContext);
        let s = next_coherence_state(s, CoherenceSignal::CompactionCompleted);
        assert_eq!(s, CoherenceState::Healthy);
    }

    #[test]
    fn empty_retries_promote_only_from_healthy() {
        assert_eq!(
            next_coherence_state(
                CoherenceState::Healthy,
                CoherenceSignal::EmptyContentRetriesRising
            ),
            CoherenceState::GettingCrowded,
        );
        // Already-degraded state doesn't get worse from this signal.
        assert_eq!(
            next_coherence_state(
                CoherenceState::ResettingPlan,
                CoherenceSignal::EmptyContentRetriesRising,
            ),
            CoherenceState::ResettingPlan,
        );
    }

    #[test]
    fn loop_guard_halt_to_verifying() {
        assert_eq!(
            next_coherence_state(CoherenceState::Healthy, CoherenceSignal::LoopGuardHalt),
            CoherenceState::VerifyingRecentWork,
        );
    }

    #[test]
    fn labels_and_icons_distinct() {
        let states = [
            CoherenceState::Healthy,
            CoherenceState::GettingCrowded,
            CoherenceState::RefreshingContext,
            CoherenceState::VerifyingRecentWork,
            CoherenceState::ResettingPlan,
        ];
        let labels: Vec<&str> = states.iter().map(|s| s.label()).collect();
        let icons: Vec<&str> = states.iter().map(|s| s.icon()).collect();
        assert_eq!(
            labels
                .iter()
                .collect::<std::collections::HashSet<_>>()
                .len(),
            states.len(),
            "labels must be unique"
        );
        assert_eq!(
            icons.iter().collect::<std::collections::HashSet<_>>().len(),
            states.len(),
            "icons must be unique"
        );
    }
}
