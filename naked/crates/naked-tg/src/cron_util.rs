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
