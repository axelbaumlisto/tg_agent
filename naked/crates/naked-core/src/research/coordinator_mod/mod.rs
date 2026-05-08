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

use std::collections::{HashMap, HashSet};
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Duration;

use async_trait::async_trait;
#[allow(unused_imports)]
use chrono::Utc;

use tokio_util::sync::CancellationToken;

use crate::error::Result;
use crate::types::{AgentEvent, AgentHandle};

use super::spec::{ResearchSpec, RunRecord};
use super::store::ResearchStore;

/// Fuzzy fingerprint of a finding's title + price used by the gatekeeper's
/// second-pass dedup. Two listings are considered duplicates when:
///   * their normalized price digits are identical (after stripping currency
///     symbols, spaces, and thousands separators), and
///   * their title token sets overlap by Jaccard ≥ 0.7 (ignoring common
///     filler words and tokens shorter than 3 chars).
///
/// The exact-match `title||price` signature is still applied first; this only
/// catches near-duplicates the strict pass misses (e.g. one title carrying an
/// extra parenthetical like "có bếp+PN").
#[derive(Debug, Clone)]
pub(crate) struct FuzzyFingerprint {
    tokens: std::collections::BTreeSet<String>,
    price_digits: String,
}

impl FuzzyFingerprint {
    /// Stop-list of low-signal words common in real-estate listings (mostly
    /// Vietnamese and English). Tokens here never count toward the Jaccard
    /// similarity. Kept short on purpose — we'd rather miss a dup than wrongly
    /// merge two distinct listings.
    const FILLER: &'static [&'static str] = &[
        "cho",
        "thuê",
        "thue",
        "rent",
        "for",
        "the",
        "and",
        "căn",
        "can",
        "hộ",
        "ho",
        "apartment",
        "with",
        "near",
        "phòng",
        "phong",
        "trong",
        "tại",
        "tai",
        "có",
        "co",
        "đầy",
        "đủ",
        "day",
        "du",
        "fully",
        "furnished",
        "new",
        "mới",
        "moi",
    ];

    pub(crate) fn new(title: Option<&str>, price: Option<&str>) -> Option<Self> {
        let title = title?.trim();
        if title.len() < 10 {
            return None;
        }
        let lower = title.to_lowercase();
        let tokens: std::collections::BTreeSet<String> = lower
            .split(|c: char| !c.is_alphanumeric())
            .filter(|s| s.chars().count() >= 3 && !Self::FILLER.contains(s))
            .map(str::to_string)
            .collect();
        if tokens.len() < 3 {
            return None;
        }
        let price_digits: String = price
            .unwrap_or("")
            .chars()
            .filter(char::is_ascii_digit)
            .collect();
        Some(Self {
            tokens,
            price_digits,
        })
    }

    fn jaccard(&self, other: &Self) -> f32 {
        let intersection = self.tokens.intersection(&other.tokens).count();
        let union = self.tokens.union(&other.tokens).count();
        if union == 0 {
            0.0
        } else {
            intersection as f32 / union as f32
        }
    }

    pub(crate) fn is_duplicate_of(&self, other: &Self) -> bool {
        // Require both: identical normalized price AND high token overlap.
        // Either alone is too noisy.
        if self.price_digits.is_empty() || other.price_digits.is_empty() {
            return false;
        }
        if self.price_digits != other.price_digits {
            return false;
        }
        self.jaccard(other) >= 0.7
    }
}

/// Split a `"provider/model"` string into `(provider, model)`. Returns
/// `None` when the input doesn't carry a provider prefix — callers fall
/// back to whatever provider is already in scope.
///
/// Used by the research fallback chain so a JSON config can express
/// cross-provider fallbacks like:
///
/// ```json
/// "fallback_models": ["kimi-k2-thinking", "moonshot/kimi-k2-thinking", "qwen/qwen3.6-plus"]
/// ```
///
/// The split point is the FIRST `/`. Everything before it is the provider
/// name, everything after is the model id (model ids may legally contain
/// further slashes, e.g. `openrouter/anthropic/claude-3.5-sonnet`).
///
/// Trims whitespace on both halves and rejects empty halves.
pub fn parse_provider_model_pair(input: &str) -> Option<(String, String)> {
    let (p, m) = input.split_once('/')?;
    let p = p.trim();
    let m = m.trim();
    if p.is_empty() || m.is_empty() {
        return None;
    }
    Some((p.to_string(), m.to_string()))
}

/// Coordinator-wide defaults, folded in from `Config.research`.
#[derive(Debug, Clone)]
pub struct CoordinatorConfig {
    /// Provider override for all research runs. `None` means "use global default".
    pub default_provider: Option<String>,
    /// Model override for all research runs.
    pub default_model: Option<String>,
    /// Fallback models tried in order when the primary model is rejected.
    pub fallback_models: Vec<String>,
    /// Safety ceiling on agent iterations if the spec doesn't specify.
    pub default_max_iterations: u32,
    /// Wall-clock ceiling if the spec doesn't specify.
    pub default_max_wall_seconds: u64,
    /// Workspace root used to materialize the ephemeral research session.
    pub workspace: PathBuf,
    /// Gatekeeper quality gate settings, passed from the JSON config.
    pub gatekeeper: crate::config::GatekeeperConfig,
    /// Reasoning level applied to every research session via
    /// `set_session_reasoning`. `None` ⇒ inherit from session/global default.
    /// Accepts `"off"`, `"low"`, `"medium"`, `"high"`.
    pub reasoning: Option<String>,
    /// Provider map for capability-aware fallback filtering. Populated
    /// from `Config.providers` when the coordinator is built. Keyed by
    /// provider id; value holds the `capabilities` block (along with
    /// the `models` list so the selector can confirm the model is
    /// declared). Populated by [`AgentCore::build_coordinator`]; left
    /// empty in unit tests that don't care about capabilities (the
    /// selector then no-ops under the soft-mode contract).
    pub provider_capabilities: HashMap<String, crate::config::ProviderConfig>,
    /// Mirrors [`crate::config::Config::enforce_model_capabilities`]. When
    /// `false`, the fallback chain is passed through unchanged (Phase 2
    /// ships in soft mode by default). When `true`, entries rejected by
    /// the selector are dropped **before** an HTTP request is spent on
    /// them.
    pub enforce_model_capabilities: bool,
    /// Runtime health tracker (Phase 3). When present, the selector
    /// inside `try_start_with_fallback` consults per-pair quarantine
    /// state so a model that started hallucinating 10 min ago gets
    /// skipped instead of costing another 90 s dispatch.
    pub model_health: Option<std::sync::Arc<crate::model_catalog::ModelHealth>>,
    /// Per-run event registry (waterfall progress for TG). When
    /// present, the coordinator emits iteration/tool events into it
    /// so the TG heartbeat task can render a live summary. `None`
    /// leaves the push sites as no-ops — the CLI / tests neither
    /// need nor want the extra plumbing.
    pub run_events: Option<super::run_events::RunEventRegistry>,
}

impl Default for CoordinatorConfig {
    fn default() -> Self {
        Self {
            default_provider: None,
            default_model: None,
            fallback_models: Vec::new(),
            default_max_iterations: 30,
            default_max_wall_seconds: 1200,
            workspace: std::env::current_dir().unwrap_or_else(|_| PathBuf::from(".")),
            gatekeeper: crate::config::GatekeeperConfig::default(),
            reasoning: None,
            provider_capabilities: HashMap::new(),
            enforce_model_capabilities: false,
            model_health: None,
            run_events: None,
        }
    }
}

/// Why a run ended. Opaque to the store (serialized as a string), but the
/// coordinator produces a fixed set we can reason about in tests.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StopReason {
    /// Agent reached `AgentEvent::Idle` naturally.
    AgentIdle,
    /// Hit wall-clock timeout (max_wall_seconds).
    Timeout,
    /// The agent's event stream closed before `Idle`. Usually an internal
    /// error — we still persist whatever was saved.
    StreamClosed,
    /// Spec was paused before we started.
    Paused,
    /// An error bubbled up from the runner.
    Error,
    /// Cooperative cancellation observed via the scheduler-supplied
    /// [`tokio_util::sync::CancellationToken`]. Distinct from `Timeout`
    /// (which is the coordinator's own wall-clock budget) — `Cancelled`
    /// means an external orchestrator (typically the scheduler's
    /// two-step `cancel → abort` on `task_timeout`) asked us to stop.
    Cancelled,
}

impl StopReason {
    pub fn as_str(self) -> &'static str {
        match self {
            StopReason::AgentIdle => "agent_idle",
            StopReason::Timeout => "timeout",
            StopReason::StreamClosed => "stream_closed",
            StopReason::Paused => "paused",
            StopReason::Error => "error",
            StopReason::Cancelled => "cancelled",
        }
    }
}

/// Summary of one `run_once`. Handed back to the caller for logging / TG
/// delivery; also mirrored into `runs.jsonl` via `RunRecord`.
#[derive(Debug, Clone)]
pub struct RunReport {
    pub spec_id: String,
    pub run_id: String,
    pub new_findings: u32,
    pub total_findings_after: u32,
    pub stop_reason: StopReason,
    pub elapsed: Duration,
    pub provider: String,
    pub model: String,
}

/// Result of a gatekeeper-verified research run.
#[derive(Debug, Clone)]
pub struct VerifiedRunReport {
    pub last_run: RunReport,
    pub verification_rounds: u32,
    pub dead_removed: u32,
    pub replacements_found: u32,
    pub final_findings: u32,
    pub remaining_issues: Vec<String>,
}

/// Gatekeeper verdict after verifying all findings.
#[derive(Debug, Default)]
pub(crate) struct GatekeeperVerdict {
    live_urls: u32,
    dead_urls: u32,
    missing_fields: u32,
    stale_dates: u32,
    /// Findings missing listing_date (None or "unknown").
    missing_dates: u32,
    /// Findings missing source_content entirely.
    missing_source_content: u32,
    /// Findings with excerpt < 200 chars (too short to be actionable).
    short_excerpts: u32,
    /// Semantic duplicates (same title+price at different URLs).
    semantic_dupes: u32,
    dead_hashes: HashSet<String>,
    dead_details: Vec<DeadFinding>,
    /// URLs of findings that need quality remediation (re-fetch for date/source).
    remediation_urls: Vec<String>,
    issues: Vec<String>,
}

#[derive(Debug)]
pub(crate) struct DeadFinding {
    url: String,
    title: String,
}

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

    /// Run with a gatekeeper verification loop:
    ///
    /// 1. `run_once` — agent collects findings
    /// 2. `verify_findings` — check every new finding: URL live? fields present?
    /// 3. Remove dead/broken findings from store
    /// 4. If issues found → build a feedback prompt → re-run agent with corrections
    /// 5. Repeat up to `max_verify_rounds` times
    /// 6. Final report is only generated after the last verification passes
    ///
    /// Returns the final `RunReport` from the last pass.
    pub async fn run_verified(
        &self,
        spec_id: &str,
        max_verify_rounds: u32,
    ) -> Result<VerifiedRunReport> {
        self.run_verified_with_cancel(spec_id, max_verify_rounds, CancellationToken::new())
            .await
    }

    /// Cancellation-aware variant of [`Self::run_verified`]. Checks
    /// `cancel.is_cancelled()` between rounds and propagates the token
    /// into both the inner `run_once` and the per-round `drain_events`,
    /// so a cancellation request stops the loop at the next safe point
    /// (between rounds or inside the current `drain_events`).
    pub async fn run_verified_with_cancel(
        &self,
        spec_id: &str,
        max_verify_rounds: u32,
        cancel: CancellationToken,
    ) -> Result<VerifiedRunReport> {
        let gk = &self.config.gatekeeper;
        let max_rounds = if max_verify_rounds > 0 {
            max_verify_rounds
        } else {
            gk.max_rounds
        };

        let verified_started = std::time::Instant::now();
        let mut round = 0u32;
        let mut last_report = self.run_once_with_cancel(spec_id, cancel.clone()).await?;
        let mut all_issues: Vec<String> = Vec::new();
        let mut total_removed = 0u32;
        let mut total_replaced = 0u32;
        let mut prev_remediation_count: Option<usize> = None;

        loop {
            // Cooperative cancellation between rounds: cheaper than letting
            // the next `drain_events` notice and gives the gatekeeper a clean
            // exit point that doesn't truncate verification mid-call.
            if cancel.is_cancelled() {
                tracing::info!(spec = %spec_id, round, "verified loop cancelled between rounds");
                break;
            }
            round += 1;
            tracing::info!(spec = %spec_id, round, "gatekeeper verification round");
            eprintln!("  [gatekeeper] round {round}/{max_rounds}");

            let verdict = self.verify_findings(spec_id).await;

            let current_remediation = verdict.remediation_urls.len();
            eprintln!(
                "  [gatekeeper] verdict: {} live, {} dead, {} missing_fields, {} stale, {} no_date, {} no_source, {} short_excerpt, {} dupes, {} to_remediate",
                verdict.live_urls,
                verdict.dead_urls,
                verdict.missing_fields,
                verdict.stale_dates,
                verdict.missing_dates,
                verdict.missing_source_content,
                verdict.short_excerpts,
                verdict.semantic_dupes,
                current_remediation,
            );

            if !verdict.dead_hashes.is_empty() {
                let removed = self
                    .store
                    .remove_findings_by_hash(spec_id, &verdict.dead_hashes)
                    .await
                    .unwrap_or(0);
                total_removed += removed;
                eprintln!("  [gatekeeper] removed {removed} dead/stale/dupe findings");
            }

            all_issues.extend(verdict.issues.iter().cloned());

            let has_actionable_issues = verdict.dead_urls > 0
                || verdict.missing_fields > 0
                || (gk.require_listing_date && verdict.missing_dates > 0)
                || (gk.require_source_content && verdict.missing_source_content > 0)
                || verdict.short_excerpts > 0;

            let stagnated = if gk.stop_on_stagnation {
                match prev_remediation_count {
                    Some(prev) if current_remediation >= prev && round > 1 => {
                        eprintln!(
                            "  [gatekeeper] stagnation detected: {prev} → {current_remediation} issues (no improvement)"
                        );
                        true
                    }
                    _ => false,
                }
            } else {
                false
            };
            prev_remediation_count = Some(current_remediation);

            if !has_actionable_issues || round >= max_rounds || stagnated {
                if !has_actionable_issues {
                    eprintln!("  [gatekeeper] all findings verified ✓");
                } else if stagnated {
                    eprintln!(
                        "  [gatekeeper] accepting results — agent did its best, remaining issues are likely unfixable"
                    );
                } else {
                    eprintln!(
                        "  [gatekeeper] max rounds ({max_rounds}) reached, accepting remaining issues"
                    );
                }
                break;
            }

            // Build feedback prompt and re-run
            let feedback = self.build_feedback_prompt(spec_id, &verdict).await?;
            eprintln!("  [gatekeeper] sending feedback to agent for re-run...");

            let re_run_id = uuid::Uuid::new_v4().simple().to_string();
            let spec = self.store.load_spec(spec_id).await?;
            let wall_secs = spec
                .max_wall_seconds
                .unwrap_or(self.config.default_max_wall_seconds);

            let (mut handle, provider, model) = match self
                .try_start_with_fallback(&spec, &feedback, &re_run_id)
                .await
            {
                Ok(v) => v,
                Err(e) => {
                    tracing::warn!(spec = %spec_id, "feedback re-run failed to start: {e}");
                    break;
                }
            };

            let started = std::time::Instant::now();
            let mut stats = DrainStats::default();
            if let Some(reg) = self.config.run_events.as_ref() {
                reg.push(
                    spec_id,
                    super::run_events::RunEvent::new(
                        super::run_events::EventKind::IterationStart,
                        format!("gatekeeper re-run round {round}/{max_rounds}"),
                    ),
                )
                .await;
            }
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

            let before = last_report.total_findings_after;
            let after = self.store.count_findings(spec_id).await.unwrap_or(before);
            let new_in_rerun = after.saturating_sub(before.saturating_sub(total_removed));
            total_replaced += new_in_rerun;

            eprintln!(
                "  [gatekeeper] re-run done: stop={:?}, +{new_in_rerun} replacements",
                stop_reason
            );

            last_report = self
                .write_record(
                    &spec,
                    &re_run_id,
                    new_in_rerun,
                    stop_reason,
                    started,
                    &provider,
                    &model,
                )
                .await?;

            self.runner.cleanup_research_session(&re_run_id).await;
        }

        // Final report regeneration after all verification passes
        let spec = self.store.load_spec(spec_id).await?;
        let _ = self.regenerate_report(&spec).await;
        let _ = self.regenerate_agent_brief(&spec).await;

        let final_count = self.store.count_findings(spec_id).await.unwrap_or(0);

        // Append a single summary RunRecord that pins the verification stats
        // to a stable, easy-to-find row. Consumers (LLM tools, TG `/research
        // metrics`) can look up "latest verification" by scanning for the
        // most-recent record with `verification_rounds.is_some()`.
        let summary_run_id = format!("{}-verified", last_report.run_id);
        let verification = VerificationSummary {
            rounds: round,
            dead_removed: total_removed,
            replacements_found: total_replaced,
            remaining_issues: all_issues.len() as u32,
        };
        let summary_report = self
            .write_record_with_verification(
                &spec,
                &summary_run_id,
                0,
                StopReason::AgentIdle,
                verified_started,
                &last_report.provider,
                &last_report.model,
                Some(verification),
            )
            .await
            .unwrap_or_else(|_| last_report.clone());

        Ok(VerifiedRunReport {
            last_run: summary_report,
            verification_rounds: round,
            dead_removed: total_removed,
            replacements_found: total_replaced,
            final_findings: final_count,
            remaining_issues: all_issues,
        })
    }

    /// Try the primary model first; on failure, iterate through
    /// `fallback_models` until one succeeds or all are exhausted.
    ///
    /// The fallback chain is pre-filtered through
    /// [`crate::model_catalog::ModelSelector::filter_chain`] for
    /// `TaskKind::Research` so entries that are deprecated, off-task, or
    /// attached to an unknown provider are skipped **before** we spend a
    /// 90 s HTTP round-trip on them. In soft mode
    /// (`enforce_model_capabilities=false`, Phase 2's default), the
    /// filter is a no-op and the chain is walked exactly as today.
    async fn try_start_with_fallback(
        &self,
        spec: &ResearchSpec,
        prompt: &str,
        run_id: &str,
    ) -> Result<(AgentHandle, String, String)> {
        match self
            .runner
            .start_research_turn(spec, prompt, &self.config, run_id)
            .await
        {
            Ok(v) => return Ok(v),
            Err(e) if self.config.fallback_models.is_empty() => return Err(e),
            Err(e) => {
                let primary = self.config.default_model.as_deref().unwrap_or("(default)");
                tracing::warn!("primary model {primary} failed ({e}), trying fallbacks");
            }
        }

        // Build the chain we'll actually try. In soft mode, pass the
        // raw list straight through so existing operators see no change
        // in ordering or resolution. In hard-enforce mode, ask the
        // selector to drop entries that can't succeed (deprecated,
        // off-task, unknown provider) before we spend HTTP on them.
        let chain: Vec<String> = if self.config.enforce_model_capabilities {
            let mut selector = crate::model_catalog::ModelSelector::from_providers(
                &self.config.provider_capabilities,
                true,
            );
            if let Some(h) = &self.config.model_health {
                selector = selector.with_health(h.clone());
            }
            let default_provider = self.config.default_provider.as_deref().unwrap_or("");
            let filtered = selector.filter_chain(
                &self.config.fallback_models,
                default_provider,
                crate::model_catalog::TaskKind::Research,
                &crate::model_catalog::Budget::default(),
            );
            if filtered.len() != self.config.fallback_models.len() {
                tracing::warn!(
                    kept = filtered.len(),
                    original = self.config.fallback_models.len(),
                    "research fallback chain trimmed by capability selector",
                );
            }
            // Re-render the filtered pairs back to the same shape the
            // runner expects (`default_model` string). When the provider
            // matches the coordinator default, pass a bare model id
            // (matches the original operator spelling); otherwise emit
            // a `provider/model` pair so the runner overrides provider.
            filtered
                .into_iter()
                .map(|(p, m)| {
                    if self.config.default_provider.as_deref() == Some(p.as_str()) {
                        m
                    } else {
                        format!("{p}/{m}")
                    }
                })
                .collect()
        } else {
            self.config.fallback_models.clone()
        };

        for fb_model in &chain {
            let mut fallback_cfg = self.config.clone();
            fallback_cfg.default_model = Some(fb_model.clone());
            match self
                .runner
                .start_research_turn(spec, prompt, &fallback_cfg, run_id)
                .await
            {
                Ok(v) => {
                    tracing::info!("fallback model {fb_model} accepted");
                    return Ok(v);
                }
                Err(e) => {
                    tracing::warn!("fallback model {fb_model} also failed: {e}");
                }
            }
        }

        Err(crate::error::AgentError::ProviderTyped(
            crate::provider::error::ProviderError::Other {
                status: 0,
                body: "all models (primary + fallbacks) rejected by provider".into(),
            },
        ))
    }
}

/// Verification block written into a [`RunRecord`] when emitted by `run_verified`.
#[derive(Debug, Clone, Copy)]
pub(crate) struct VerificationSummary {
    pub rounds: u32,
    pub dead_removed: u32,
    pub replacements_found: u32,
    pub remaining_issues: u32,
}

mod drain;
pub use drain::{DrainStats, drain_events};

#[cfg(test)]
#[path = "tests.rs"]
mod tests;
