//! Clock abstraction for the research scheduler.
//!
//! Injecting a `Clock` into [`super::SchedulerConfig`] allows unit and
//! integration tests to drive time deterministically — no `sleep`, no
//! real-wall-clock races.

use chrono::{DateTime, Utc};

/// Abstracts wall-clock access so tests can substitute a deterministic
/// time source without touching `chrono::Utc` globally.
pub trait Clock: Send + Sync + std::fmt::Debug {
    fn now(&self) -> DateTime<Utc>;
}

/// Production clock — delegates to [`chrono::Utc::now`].
#[derive(Debug, Clone, Default)]
pub struct RealClock;

impl Clock for RealClock {
    fn now(&self) -> DateTime<Utc> {
        Utc::now()
    }
}
