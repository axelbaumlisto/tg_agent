//! Unified retry / back-off primitive used by `loop_.rs`.
//!
//! `Backoff` is intentionally a pure data structure with no async code so
//! it is trivial to unit-test and reason about independent of the loop.

use std::time::Duration;

/// Decision returned by [`Backoff::decide`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RetryDecision {
    /// Caller should sleep for the given duration then retry.
    Retry(Duration),
    /// Retries exhausted — bubble the error to the caller.
    Stop,
}

/// Exponential back-off configuration.
///
/// ```
/// use naked_core::retry::Backoff;
/// use std::time::Duration;
///
/// let b = Backoff { base_ms: 1000, max_attempts: 4 };
/// assert_eq!(b.delay(0), Duration::from_millis(1000));
/// assert_eq!(b.delay(1), Duration::from_millis(2000));
/// ```
#[derive(Debug, Clone)]
pub struct Backoff {
    /// Base delay in milliseconds (delay at attempt 0).
    pub base_ms: u64,
    /// Total number of attempts allowed (including attempt 0).
    /// `decide(attempt)` returns `Stop` when `attempt >= max_attempts`.
    pub max_attempts: usize,
}

impl Backoff {
    /// Returns `base_ms * 2^attempt` as a [`Duration`].
    ///
    /// `attempt` is **0-indexed**: attempt 0 → `base_ms`, attempt 1 →
    /// `2 * base_ms`, etc.  The exponent is clamped at 20 to avoid a
    /// shift-overflow on 64-bit platforms; multiplication is saturating.
    pub fn delay(&self, attempt: usize) -> Duration {
        let shift = attempt.min(20) as u32;
        Duration::from_millis(self.base_ms.saturating_mul(1u64 << shift))
    }

    /// Returns [`RetryDecision::Stop`] when `attempt >= max_attempts`,
    /// otherwise [`RetryDecision::Retry`] with the computed back-off delay.
    pub fn decide(&self, attempt: usize) -> RetryDecision {
        if attempt >= self.max_attempts {
            RetryDecision::Stop
        } else {
            RetryDecision::Retry(self.delay(attempt))
        }
    }
}
