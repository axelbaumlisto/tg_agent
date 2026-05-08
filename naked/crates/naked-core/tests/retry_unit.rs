//! Unit tests for the `naked_core::retry` module.
//!
//! These are fast, synchronous, and have no external dependencies.

use naked_core::retry::{Backoff, RetryDecision};
use std::time::Duration;

// ── delay() ─────────────────────────────────────────────────────────────────

#[test]
fn delay_doubles() {
    let b = Backoff {
        base_ms: 1000,
        max_attempts: 4,
    };
    assert_eq!(b.delay(0), Duration::from_millis(1000));
    assert_eq!(b.delay(1), Duration::from_millis(2000));
    assert_eq!(b.delay(2), Duration::from_millis(4000));
    assert_eq!(b.delay(3), Duration::from_millis(8000));
}

#[test]
fn delay_base_250() {
    // mirrors EMPTY_CONTENT_BASE_DELAY_MS usage
    let b = Backoff {
        base_ms: 250,
        max_attempts: 2,
    };
    assert_eq!(b.delay(0), Duration::from_millis(250));
    assert_eq!(b.delay(1), Duration::from_millis(500));
}

#[test]
fn delay_does_not_overflow_at_attempt_64() {
    // attempt=64 is clamped to 20 internally; saturating_mul prevents panic
    let b = Backoff {
        base_ms: 1000,
        max_attempts: 100,
    };
    let d = b.delay(64);
    // Must not panic and must be positive
    assert!(d.as_millis() > 0);
    // Must not exceed base_ms * 2^20 (the clamp ceiling)
    assert!(d <= Duration::from_millis(1000u64 << 20));
}

#[test]
fn delay_saturates_rather_than_overflows() {
    // u64::MAX base_ms with high attempt → saturating_mul returns u64::MAX
    let b = Backoff {
        base_ms: u64::MAX,
        max_attempts: 1,
    };
    // Should not panic
    let _ = b.delay(0);
}

// ── decide() ────────────────────────────────────────────────────────────────

#[test]
fn decide_stops_at_max() {
    let b = Backoff {
        base_ms: 1000,
        max_attempts: 3,
    };
    // Attempts 0, 1, 2 are within budget
    assert!(matches!(b.decide(0), RetryDecision::Retry(_)));
    assert!(matches!(b.decide(1), RetryDecision::Retry(_)));
    assert!(matches!(b.decide(2), RetryDecision::Retry(_)));
    // Attempt 3 == max_attempts → Stop
    assert_eq!(b.decide(3), RetryDecision::Stop);
}

#[test]
fn decide_zero_attempts_stops() {
    // max_attempts=0 means "never retry"
    let b = Backoff {
        base_ms: 1000,
        max_attempts: 0,
    };
    assert_eq!(b.decide(0), RetryDecision::Stop);
}

#[test]
fn decide_retry_carries_correct_delay() {
    let b = Backoff {
        base_ms: 500,
        max_attempts: 4,
    };
    if let RetryDecision::Retry(d) = b.decide(2) {
        assert_eq!(d, Duration::from_millis(2000)); // 500 * 2^2
    } else {
        panic!("expected Retry, got Stop");
    }
}
