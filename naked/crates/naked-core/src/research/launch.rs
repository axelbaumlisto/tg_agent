//! Launch acknowledgement + verification outcomes.
//!
//! `research_launch` today returns `{"status":"launched","verified":true}`
//! synchronously after the scheduler receives the spec id, regardless of
//! whether the coordinator actually took the semaphore and opened a session.
//! That means when the scheduler is saturated, or when the spec failed a
//! pre-flight (e.g. unknown provider), the LLM still believed the run was
//! live and wasted a turn polling `agent_status` for a process that never
//! started.
//!
//! `LaunchOutcome` makes the handshake explicit: the tool returns a
//! `Pending` when the job is queued and flips to `Verified` only once the
//! coordinator reports `agent_status.running == true` before `verify_within`.

use std::time::Duration;

/// Options passed to the launch contract.
#[derive(Debug, Clone)]
pub struct LaunchOptions {
    /// How long to wait for the agent to actually start before giving up.
    pub verify_within: Duration,
}

impl Default for LaunchOptions {
    fn default() -> Self {
        Self {
            verify_within: Duration::from_secs(5),
        }
    }
}

/// Final verdict of a launch attempt.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum LaunchOutcome {
    /// The coordinator confirmed the agent is running (`status.running == true`).
    Verified { pid: u64, run_id: String },
    /// Queued but did not start within `verify_within`.
    FailedToStart { reason: String },
}

impl LaunchOutcome {
    pub fn is_verified(&self) -> bool {
        matches!(self, LaunchOutcome::Verified { .. })
    }
}

/// Poll-driven verifier used by the tool handler. `probe` is an async
/// callback the caller provides so this module stays unit-testable: return
/// `Ok(Some(pid))` once the agent is running, `Ok(None)` if it hasn't
/// started yet, `Err(reason)` on a terminal failure.
pub async fn verify_launch<F, Fut>(
    run_id: String,
    options: LaunchOptions,
    mut probe: F,
) -> LaunchOutcome
where
    F: FnMut() -> Fut,
    Fut: std::future::Future<Output = Result<Option<u64>, String>>,
{
    let deadline = tokio::time::Instant::now() + options.verify_within;
    loop {
        match probe().await {
            Ok(Some(pid)) => return LaunchOutcome::Verified { pid, run_id },
            Ok(None) => {}
            Err(reason) => return LaunchOutcome::FailedToStart { reason },
        }
        if tokio::time::Instant::now() >= deadline {
            return LaunchOutcome::FailedToStart {
                reason: format!(
                    "agent did not start within {}s",
                    options.verify_within.as_secs()
                ),
            };
        }
        tokio::time::sleep(Duration::from_millis(100)).await;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn verify_launch_reports_verified_when_agent_starts() {
        let mut calls = 0;
        let outcome = verify_launch(
            "r1".into(),
            LaunchOptions {
                verify_within: Duration::from_millis(500),
            },
            || {
                calls += 1;
                let n = calls;
                async move {
                    // first probe: not yet; second probe: running
                    if n < 2 {
                        Ok::<_, String>(None)
                    } else {
                        Ok(Some(42))
                    }
                }
            },
        )
        .await;
        assert!(matches!(outcome, LaunchOutcome::Verified { pid: 42, .. }));
    }

    #[tokio::test]
    async fn verify_launch_fails_when_deadline_exceeded() {
        let outcome = verify_launch(
            "r2".into(),
            LaunchOptions {
                verify_within: Duration::from_millis(50),
            },
            || async { Ok::<_, String>(None) },
        )
        .await;
        assert!(matches!(outcome, LaunchOutcome::FailedToStart { .. }));
    }
}
