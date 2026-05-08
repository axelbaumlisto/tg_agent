//! Empty-content retry budget extracted from `AgentLoop::run`.
//!
//! Tracks how many consecutive empty-content responses have been seen and
//! computes the back-off delay for each retry.

use super::{EMPTY_CONTENT_BASE_DELAY_MS, MAX_EMPTY_CONTENT_RETRIES};
use crate::retry::Backoff;

pub(super) struct EmptyContentBudget {
    attempts: usize,
    backoff: Backoff,
}

impl EmptyContentBudget {
    pub(super) fn new() -> Self {
        Self {
            attempts: 0,
            backoff: Backoff {
                base_ms: EMPTY_CONTENT_BASE_DELAY_MS,
                max_attempts: MAX_EMPTY_CONTENT_RETRIES,
            },
        }
    }

    /// Returns `Some(delay)` to retry after, or `None` to give up.
    /// Increments the internal attempt counter on each retry.
    pub(super) fn next_delay(&mut self) -> Option<std::time::Duration> {
        if self.attempts >= self.backoff.max_attempts {
            return None;
        }
        let d = self.backoff.delay(self.attempts);
        self.attempts += 1;
        Some(d)
    }

    pub(super) fn reset(&mut self) {
        self.attempts = 0;
    }

    pub(super) fn count(&self) -> usize {
        self.attempts
    }
}
