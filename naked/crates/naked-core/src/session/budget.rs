//! Session context-budget watermark.
//!
//! DM session `093a7d74` on 2026-04-13 climbed to 104 553 input tokens
//! on its last live turn before being silently retired. No warning was
//! ever emitted to operator or the bot itself. `SessionBudget` fires
//! exactly one [`ContextBudgetEvent::ApproachingLimit`] once the input
//! token count crosses 80% of the provider window, giving the
//! coordinator a chance to compact / roll / notify.

/// Event emitted by [`SessionBudget::observe_turn`] when the watermark
/// is crossed for the first time in the session's lifetime.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ContextBudgetEvent {
    ApproachingLimit { percent: u32, tokens: u64 },
}

/// Watermark guard for a single session's input-token usage.
#[derive(Debug)]
pub struct SessionBudget {
    window: u64,
    warned: bool,
    threshold_percent: u32,
}

impl SessionBudget {
    /// Build a budget for the given model context window (tokens). The
    /// default warn threshold is 80% of that window, aligned with how
    /// we recommend people size their `max_tokens` headroom.
    pub fn new(window: u64) -> Self {
        Self::with_threshold(window, 80)
    }

    pub fn with_threshold(window: u64, threshold_percent: u32) -> Self {
        Self {
            window,
            warned: false,
            threshold_percent: threshold_percent.clamp(1, 100),
        }
    }

    /// Record the input-token count of a completed turn. Returns
    /// `Some(ApproachingLimit)` exactly once when the cumulative
    /// observation first crosses the threshold; `None` before and
    /// after.
    pub fn observe_turn(&mut self, input_tokens: u64) -> Option<ContextBudgetEvent> {
        if self.window == 0 || self.warned {
            return None;
        }
        let pct = (100u128.saturating_mul(input_tokens as u128) / self.window as u128) as u32;
        if pct >= self.threshold_percent {
            self.warned = true;
            return Some(ContextBudgetEvent::ApproachingLimit {
                percent: pct,
                tokens: input_tokens,
            });
        }
        None
    }

    /// Reset the "already warned" flag (used after a successful
    /// compaction so the next climb fires again).
    pub fn reset(&mut self) {
        self.warned = false;
    }

    pub fn has_warned(&self) -> bool {
        self.warned
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn below_threshold_quiet() {
        let mut b = SessionBudget::new(128_000);
        assert_eq!(b.observe_turn(10_000), None);
        assert_eq!(b.observe_turn(50_000), None);
    }

    #[test]
    fn crosses_threshold_once() {
        let mut b = SessionBudget::new(128_000);
        let ev = b.observe_turn(105_000);
        match ev {
            Some(ContextBudgetEvent::ApproachingLimit { percent, tokens }) => {
                assert!(percent >= 80);
                assert_eq!(tokens, 105_000);
            }
            None => panic!("expected ApproachingLimit"),
        }
        assert_eq!(
            b.observe_turn(106_000),
            None,
            "second crossing must stay silent"
        );
    }

    #[test]
    fn reset_re_arms_warning() {
        let mut b = SessionBudget::new(128_000);
        assert!(b.observe_turn(105_000).is_some());
        b.reset();
        assert!(b.observe_turn(106_000).is_some());
    }
}
