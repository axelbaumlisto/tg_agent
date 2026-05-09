//! Shared cron-expression helpers used by the in-process schedulers
//! (`research_scheduler` and `memory_scheduler`).
//!
//! Centralizing the parser keeps every scheduler on the same dialect:
//! standard 5-field crontab syntax (`min hour dom mon dow`) — the
//! `cron` crate's own parser internally pads it with seconds=0.

use chrono::{DateTime, Utc};
use cron::Schedule as CronSchedule;
use std::str::FromStr;

/// Parse a cron expression and compute the next firing time strictly after
/// `after` (UTC). Returns `None` for an unparseable expression or when the
/// cron has no next fire (e.g. impossible date).
///
/// Accepts both 5-field (`min hour dom mon dow`) and 6-field (with
/// seconds) forms — 5-field is normalized by prepending `0` so it parses
/// as "fire at second 0 of each matching minute".
pub fn next_cron_after(expr: &str, after: DateTime<Utc>) -> Option<DateTime<Utc>> {
    let normalized = if expr.split_whitespace().count() == 5 {
        format!("0 {expr}")
    } else {
        expr.to_string()
    };
    let schedule = CronSchedule::from_str(&normalized).ok()?;
    schedule.after(&after).next()
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::Duration as ChronoDuration;

    #[test]
    fn cron_5_field_parses_and_advances() {
        let after = Utc::now();
        let next = next_cron_after("* * * * *", after).expect("parses");
        assert!(next > after);
        assert!(next - after < ChronoDuration::minutes(2));
    }

    #[test]
    fn cron_invalid_returns_none() {
        let after = Utc::now();
        assert!(next_cron_after("nonsense", after).is_none());
    }

    #[test]
    fn cron_6_field_with_seconds_parses() {
        let after = Utc::now();
        // every 30 seconds
        let next = next_cron_after("*/30 * * * * *", after).expect("parses");
        assert!(next > after);
    }

    #[test]
    fn cron_daily_4am_utc_parses() {
        let after = Utc::now();
        let next = next_cron_after("0 4 * * *", after).expect("parses");
        assert!(next > after);
        assert_eq!(next.format("%H:%M").to_string(), "04:00");
    }
}

#[cfg(test)]
mod proptests {
    //! Property-based tests for `next_cron_after` (T13 of
    //! PLAN_CORE_HARDENING_v2).
    //!
    //! Properties:
    //!   1. **Strict monotonicity**: returned time is always strictly
    //!      greater than the `after` argument (no fire at the boundary).
    //!   2. **Stability**: feeding the same `(expr, after)` twice yields
    //!      identical results.
    //!   3. **Hourly-cron upper bound**: a `0 * * * *` schedule never
    //!      returns a time more than 1h+1min into the future, regardless
    //!      of what `after` falls on.
    //!   4. **Garbage-in → None**: random non-cron strings return None,
    //!      not panic.

    use super::*;
    use chrono::{Duration as ChronoDuration, TimeZone};
    use proptest::prelude::*;

    /// Generate a UTC time in [2026-01-01, 2030-12-31].
    fn arb_after() -> impl Strategy<Value = DateTime<Utc>> {
        // 5 years × ~31.5M seconds ≈ 1.6 × 10^8
        (0u64..157_680_000u64).prop_map(|secs| {
            let base = Utc.with_ymd_and_hms(2026, 1, 1, 0, 0, 0).unwrap();
            base + ChronoDuration::seconds(secs as i64)
        })
    }

    proptest! {
        #![proptest_config(ProptestConfig {
            cases: 200,
            ..ProptestConfig::default()
        })]

        #[test]
        fn strict_monotone_for_any_minute(after in arb_after()) {
            // "every minute" cron must always fire strictly after `after`.
            let next = next_cron_after("* * * * *", after).expect("parses");
            prop_assert!(
                next > after,
                "non-monotone: next={next} not > after={after}"
            );
        }

        #[test]
        fn stable_across_calls(after in arb_after()) {
            let a = next_cron_after("0 4 * * *", after);
            let b = next_cron_after("0 4 * * *", after);
            prop_assert_eq!(a, b, "next_cron_after is not pure");
        }

        #[test]
        fn hourly_within_1h_1m(after in arb_after()) {
            let next = next_cron_after("0 * * * *", after).expect("parses");
            let delta = next - after;
            prop_assert!(
                delta <= ChronoDuration::minutes(61),
                "hourly cron returned {delta:?} into the future (after={after})"
            );
            prop_assert!(delta > ChronoDuration::zero());
        }

        #[test]
        fn garbage_returns_none_not_panic(garbage in "[a-z!@#$%^&*]{1,30}") {
            // Anything that is NOT a valid cron must return None
            // without panicking. The whitespace test below catches
            // the case where random bytes happen to look like a cron.
            // We pre-filter obvious cron-like inputs.
            if garbage.split_whitespace().count() == 5
                || garbage.split_whitespace().count() == 6
            {
                return Ok(());
            }
            let after = Utc.with_ymd_and_hms(2026, 1, 1, 0, 0, 0).unwrap();
            prop_assert!(next_cron_after(&garbage, after).is_none());
        }
    }
}
