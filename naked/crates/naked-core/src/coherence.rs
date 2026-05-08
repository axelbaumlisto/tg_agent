//! Session coherence state machine — maps capacity ratio to user-friendly state.

use serde::{Deserialize, Serialize};

#[derive(Debug, Default, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CoherenceState {
    #[default]
    Healthy,
    GettingCrowded,
    /// The loop guard detected repeated failures.
    VerifyingRecentWork,
    RefreshingContext,
    ResettingPlan,
}

impl CoherenceState {
    pub fn label(self) -> &'static str {
        match self {
            Self::Healthy => "healthy",
            Self::GettingCrowded => "getting crowded",
            Self::VerifyingRecentWork => "verifying recent work",
            Self::RefreshingContext => "refreshing context",
            Self::ResettingPlan => "resetting plan",
        }
    }
    pub fn emoji(self) -> &'static str {
        match self {
            Self::Healthy => "🟢",
            Self::GettingCrowded => "🟡",
            Self::VerifyingRecentWork => "🟠",
            Self::RefreshingContext => "🔄",
            Self::ResettingPlan => "🔴",
        }
    }
}

pub fn from_capacity(ratio: f32) -> CoherenceState {
    match ratio {
        r if r >= 0.90 => CoherenceState::ResettingPlan,
        r if r >= 0.75 => CoherenceState::RefreshingContext,
        r if r >= 0.60 => CoherenceState::GettingCrowded,
        _ => CoherenceState::Healthy,
    }
}

pub fn from_tokens(tokens: u64, window: u64) -> CoherenceState {
    if window == 0 {
        return CoherenceState::Healthy;
    }
    from_capacity(tokens as f32 / window as f32)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn healthy() {
        assert_eq!(from_capacity(0.0), CoherenceState::Healthy);
        assert_eq!(from_capacity(0.59), CoherenceState::Healthy);
    }
    #[test]
    fn crowded() {
        assert_eq!(from_capacity(0.60), CoherenceState::GettingCrowded);
        assert_eq!(from_capacity(0.74), CoherenceState::GettingCrowded);
    }
    #[test]
    fn refreshing() {
        assert_eq!(from_capacity(0.75), CoherenceState::RefreshingContext);
        assert_eq!(from_capacity(0.89), CoherenceState::RefreshingContext);
    }
    #[test]
    fn resetting() {
        assert_eq!(from_capacity(0.90), CoherenceState::ResettingPlan);
        assert_eq!(from_capacity(1.0), CoherenceState::ResettingPlan);
    }
    #[test]
    fn from_tok() {
        assert_eq!(from_tokens(50_000, 100_000), CoherenceState::Healthy);
        assert_eq!(from_tokens(95_000, 100_000), CoherenceState::ResettingPlan);
    }
    #[test]
    fn zero_window() {
        assert_eq!(from_tokens(999, 0), CoherenceState::Healthy);
    }
}
