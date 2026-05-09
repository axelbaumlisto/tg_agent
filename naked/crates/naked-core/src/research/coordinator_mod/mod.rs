//! Research runner. Kicks off one research pass: loads the spec, builds a
//! research-specific prompt (topic, known-findings dedup list, cursor), runs
//! the agent with the `research_save`/`research_list`/`web_fetch` toolkit,
//! collects new findings, regenerates the rolling report, appends a
//! `RunRecord`, and returns a summary to the caller.
//!
//! Intentionally independent of AgentCore's public surface — takes `&dyn
//! AgentRunner` so tests can stub the turn without standing up an LLM. The
//! production wiring (`AgentCore::run_research`) lives in `lib.rs`.

mod briefing_ops;
mod gatekeeper;
mod runner_dispatch;
mod types;

#[allow(unused_imports)] // re-exported via glob for tests.rs (`use super::*`)
use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;

use async_trait::async_trait;
#[allow(unused_imports)]
use chrono::Utc;

use tokio_util::sync::CancellationToken;

use crate::error::Result;
use crate::types::{AgentEvent, AgentHandle};

#[allow(unused_imports)]
use super::spec::{ResearchSpec, RunRecord};
use super::store::ResearchStore;

// Re-export all public types from the types submodule so the public API paths
// (e.g. `research::coordinator_mod::CoordinatorConfig`) remain unchanged.
pub use types::{
    CoordinatorConfig, RunReport, StopReason, VerifiedRunReport, parse_provider_model_pair,
};
pub(crate) use types::{DeadFinding, FuzzyFingerprint, GatekeeperVerdict, VerificationSummary};

/// Abstraction over "run one agent turn in a fresh research session". In
/// production this is `AgentCore` (see `lib.rs::AgentCoreResearchRunner`); in
/// tests it's a stub that records the prompt and pretends to terminate.
#[async_trait]
pub trait AgentRunner: Send + Sync {
    /// Provision an ephemeral session with the research tools wired up, then
    /// send `prompt`. The session is expected to be deleted on drop or by the
    /// caller — the coordinator never queries it again. Returns the live event
    /// handle plus the resolved (provider, model) the session will actually
    /// talk to (so `RunRecord` can be accurate even when overrides apply).
    async fn start_research_turn(
        &self,
        spec: &ResearchSpec,
        prompt: &str,
        config: &CoordinatorConfig,
        run_id: &str,
    ) -> Result<(AgentHandle, String, String)>;

    /// Best-effort cleanup of the ephemeral session. Errors are logged by the
    /// impl; coordinator treats this as fire-and-forget.
    async fn cleanup_research_session(&self, session_id: &str);
}

pub struct ResearchCoordinator {
    store: Arc<dyn ResearchStore>,
    runner: Arc<dyn AgentRunner>,
    config: CoordinatorConfig,
}

impl ResearchCoordinator {
    pub fn new(
        store: Arc<dyn ResearchStore>,
        runner: Arc<dyn AgentRunner>,
        config: CoordinatorConfig,
    ) -> Self {
        Self {
            store,
            runner,
            config,
        }
    }

    pub fn store(&self) -> &Arc<dyn ResearchStore> {
        &self.store
    }

    /// Execute a single research pass. Caller chooses whether to await or to
    /// fire-and-forget via `tokio::spawn` — the coordinator itself awaits.
    ///
    /// Backward-compat shim: dispatches to [`Self::run_once_with_cancel`]
    /// with a fresh, never-cancelled token so existing callers (CLI, ad-hoc
    /// `naked research run`) keep working without plumbing a token.
    pub async fn run_once(&self, spec_id: &str) -> Result<RunReport> {
        self.run_once_with_cancel(spec_id, CancellationToken::new())
            .await
    }

    /// Cancellation-aware single research pass. The supplied
    /// [`CancellationToken`] is observed:
    /// * by the top-level `select!` (returns `StopReason::Cancelled` and
    ///   writes a partial RunRecord), and
    /// * inside [`drain_events`] (drops the inner await as soon as
    ///   cancellation is signalled instead of waiting on the next
    ///   provider event).
    ///
    /// The scheduler's two-step `cancel → abort` shutdown path uses this:
    /// `cancel()` lets in-flight HTTP / file IO finish at the next
    /// natural `await`; if the worker is wedged inside a sync section,
    /// `JoinHandle::abort()` lands the kill at the next yield as a
    /// hard fallback.
    pub async fn run_once_with_cancel(
        &self,
        spec_id: &str,
        cancel: CancellationToken,
    ) -> Result<RunReport> {
        let started = std::time::Instant::now();
        let run_id = uuid::Uuid::new_v4().simple().to_string();
        let spec = self.store.load_spec(spec_id).await?;

        if spec.paused {
            tracing::info!(spec = %spec_id, "skipping paused research");
            return self
                .write_record(&spec, &run_id, 0, StopReason::Paused, started, "-", "-")
                .await;
        }

        let before = self.store.count_findings(spec_id).await.unwrap_or(0);
        let prompt = self.build_prompt(&spec).await?;

        if let Some(ev) = self.config.run_events.as_ref() {
            let topic_preview: String = spec.topic.chars().take(80).collect();
            ev.push(
                spec_id,
                super::run_events::RunEvent::new(
                    super::run_events::EventKind::IterationStart,
                    format!("run starting — {topic_preview}"),
                ),
            )
            .await;
        }

        let wall_secs = spec
            .max_wall_seconds
            .unwrap_or(self.config.default_max_wall_seconds);

        let (mut handle, provider, model) =
            match self.try_start_with_fallback(&spec, &prompt, &run_id).await {
                Ok(v) => v,
                Err(e) => {
                    tracing::error!(spec = %spec_id, "research runner failed to start: {e}");
                    return self
                        .write_record(&spec, &run_id, 0, StopReason::Error, started, "-", "-")
                        .await;
                }
            };

        let mut stats = DrainStats::default();
        let reg_opt = self.config.run_events.as_ref();
        let stop_reason = tokio::select! {
            r = drain_events(&mut handle, &mut stats, &cancel, reg_opt, spec_id) => r,
            _ = tokio::time::sleep(Duration::from_secs(wall_secs)) => {
                eprintln!("  {}", stats.summary_line());
                StopReason::Timeout
            }
            _ = cancel.cancelled() => {
                eprintln!("  {}", stats.summary_line());
                StopReason::Cancelled
            }
        };

        let after = self.store.count_findings(spec_id).await.unwrap_or(before);
        let new_findings = after.saturating_sub(before);

        // Regenerate rolling report from the full findings set. Best-effort —
        // report write failures don't invalidate the run record.
        if let Err(e) = self.regenerate_report(&spec).await {
            tracing::warn!(spec = %spec_id, "report regeneration failed: {e}");
        }
        if let Err(e) = self.regenerate_agent_brief(&spec).await {
            tracing::warn!(spec = %spec_id, "agent brief regeneration failed: {e}");
        }

        let report = self
            .write_record(
                &spec,
                &run_id,
                new_findings,
                stop_reason,
                started,
                &provider,
                &model,
            )
            .await;

        // Agent runner owns the ephemeral session — ask it to tidy up after
        // we've recorded the run. Swallow failures, they're advisory.
        // (The runner decides how to interpret the session id; the coordinator
        // never learns it, so we pass the run_id as the logical handle.)
        self.runner.cleanup_research_session(&run_id).await;
        report
    }
}

mod drain;
pub use drain::{DrainStats, drain_events};

#[cfg(test)]
#[path = "tests.rs"]
mod tests;
