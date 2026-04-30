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
    #[allow(dead_code)]
    hash: String,
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

        Err(crate::error::AgentError::Provider(
            "all models (primary + fallbacks) rejected by provider".into(),
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

/// Per-run telemetry accumulated by [`drain_events`].
///
/// Lightweight on purpose — feeds the `[research-summary]` line that
/// `naked research probe` parses to score iterations, and surfaces JS-gated
/// hosts the agent kept ramming without escalating to the browser playbook.
#[derive(Debug, Default, Clone)]
pub struct DrainStats {
    /// Tool invocations grouped by tool name. `BTreeMap` so the printed
    /// summary is deterministic across runs.
    pub tool_counts: std::collections::BTreeMap<String, u32>,
    /// Tool-result bodies that contained one of the canonical
    /// captcha/anti-bot markers. Each occurrence is a "the agent saw a
    /// gated page" signal — high counts with low `Skill` invocations
    /// mean the prompt is not steering the agent to the playbook.
    pub captcha_hits: u32,
    /// Convenience counter pulled out of `tool_counts["Skill"]` so the
    /// summary line is grep-friendly without re-scanning the map.
    pub skill_loads: u32,
    pub text_deltas: u32,
    pub errors: u32,
}

impl DrainStats {
    /// Substrings we treat as "this page is anti-bot gated". Kept short and
    /// case-sensitive — false positives here would inflate the metric and
    /// dilute the signal we use to validate prompt changes.
    const CAPTCHA_MARKERS: &'static [&'static str] = &[
        "xac-thuc-nguoi-dung",           // alonhadat anti-bot interstitial path
        "Vui lòng xác minh",             // VN "please verify" string family
        "Just a moment",                 // Cloudflare interstitial title
        "Attention Required",            // Cloudflare 1020/blocked title
        "Enable JavaScript and cookies", // Cloudflare body line
        "cf_chl_rt_tk",                  // Cloudflare challenge token in URL
    ];

    fn note_tool(&mut self, name: &str) {
        *self.tool_counts.entry(name.to_string()).or_default() += 1;
        if name == "Skill" {
            self.skill_loads += 1;
        }
    }

    fn note_tool_output(&mut self, body: &str) {
        if Self::CAPTCHA_MARKERS.iter().any(|m| body.contains(m)) {
            self.captcha_hits += 1;
        }
    }

    /// Render the `[research-summary]` line. Stable, machine-parseable
    /// shape so `naked research probe` (and any future regression script)
    /// can `grep` for it without depending on tool ordering.
    pub fn summary_line(&self) -> String {
        let tools = self
            .tool_counts
            .iter()
            .map(|(k, v)| format!("{k}={v}"))
            .collect::<Vec<_>>()
            .join(",");
        format!(
            "[research-summary] tools={{{tools}}} captcha_hits={} skill_loads={} text_deltas={} errors={}",
            self.captcha_hits, self.skill_loads, self.text_deltas, self.errors
        )
    }
}

/// Build a short human-readable label for a `ToolStart` event — the
/// waterfall prefers "what is this tool doing" over "what are its raw
/// arguments". Keeps the total under ~60 chars so the heartbeat
/// message stays comfortably below Telegram's 4096-char budget even
/// when every one of the last 5 slots is a long URL.
fn tool_start_label(name: &str, input: &serde_json::Value) -> String {
    let hint = match name {
        "web_fetch" | "browser_navigate" => input
            .get("url")
            .and_then(|v| v.as_str())
            .map(short_url_host),
        "research_save" => input
            .get("url")
            .and_then(|v| v.as_str())
            .map(short_url_host),
        "web_search" | "web_search_exa" => input
            .get("query")
            .and_then(|v| v.as_str())
            .map(|q| q.chars().take(40).collect::<String>()),
        "Skill" => input
            .get("skill")
            .or_else(|| input.get("name"))
            .and_then(|v| v.as_str())
            .map(|s| s.to_string()),
        "browser_snapshot" | "browser_take_screenshot" => None,
        _ => None,
    };
    match hint {
        Some(h) if !h.is_empty() => format!("{name} {h}"),
        _ => name.to_string(),
    }
}

/// `https://www.alonhadat.com.vn/abc/xyz` → `alonhadat.com.vn/xyz`. Drops
/// scheme + `www.` and keeps only the host + last path segment so the
/// label stays readable in a 4096-char heartbeat message.
fn short_url_host(url: &str) -> String {
    let stripped = url
        .trim()
        .trim_start_matches("https://")
        .trim_start_matches("http://")
        .trim_start_matches("www.");
    let mut parts = stripped.splitn(2, '/');
    let host = parts.next().unwrap_or("").to_string();
    let rest = parts.next().unwrap_or("");
    let last_seg = rest.rsplit('/').find(|s| !s.is_empty()).unwrap_or("");
    if last_seg.is_empty() {
        host
    } else {
        let tail: String = last_seg.chars().take(28).collect();
        format!("{host}/{tail}")
    }
}

/// Drain agent events until the stream closes or the agent signals `Idle`.
/// Auto-approves every `PermissionRequest` — research runs are headless,
/// nobody is watching to click "allow".
///
/// `stats` is accumulated in place so the caller still sees partial telemetry
/// when the run is cancelled by the wall-clock timeout (the future is
/// dropped mid-loop in that case, but the borrow already wrote whatever it
/// observed).
async fn drain_events(
    handle: &mut AgentHandle,
    stats: &mut DrainStats,
    cancel: &CancellationToken,
    run_events: Option<&super::run_events::RunEventRegistry>,
    run_id: &str,
) -> StopReason {
    let mut tool_calls = 0u32;
    loop {
        // Race the next agent event against the cancellation token. The
        // outer `select!` in `run_once`/`run_verified` already handles
        // cancellation at the run boundary, but we also check here so a
        // long-running provider stream doesn't make us hang on
        // `events.recv()` for the full wall-clock budget after the
        // scheduler asked us to stop.
        let event = tokio::select! {
            biased;
            _ = cancel.cancelled() => {
                tracing::info!(
                    tool_calls,
                    text_deltas = stats.text_deltas,
                    errors = stats.errors,
                    "drain_events: cancellation observed mid-stream"
                );
                eprintln!("  [research] cancelled mid-stream");
                return StopReason::Cancelled;
            }
            ev = handle.events.recv() => ev,
        };
        match event {
            Some(AgentEvent::Idle) => {
                tracing::info!(
                    tool_calls,
                    text_deltas = stats.text_deltas,
                    errors = stats.errors,
                    "research agent idle"
                );
                eprintln!("  {}", stats.summary_line());
                return StopReason::AgentIdle;
            }
            Some(AgentEvent::Error(e)) => {
                stats.errors += 1;
                tracing::warn!("research agent error #{}: {e}", stats.errors);
                eprintln!("  [research] error #{}: {e}", stats.errors);
            }
            Some(AgentEvent::ToolStart { name, input, .. }) => {
                tool_calls += 1;
                stats.note_tool(&name);
                eprintln!("  [research] tool #{tool_calls}: {name}");
                if let Some(reg) = run_events {
                    let label = tool_start_label(&name, &input);
                    let kind = if name == "Skill" {
                        super::run_events::EventKind::SkillLoaded
                    } else {
                        super::run_events::EventKind::ToolCallStart
                    };
                    reg.push(run_id, super::run_events::RunEvent::new(kind, label))
                        .await;
                }
            }
            Some(AgentEvent::ToolEnd {
                name,
                state,
                output,
                ..
            }) => {
                stats.note_tool_output(&output);
                let preview: String = output.chars().take(200).collect();
                eprintln!("  [research] tool done: {name} state={state:?} → {preview}");
                if let Some(reg) = run_events {
                    let body_marker = DrainStats::CAPTCHA_MARKERS
                        .iter()
                        .any(|m| output.contains(m));
                    if body_marker {
                        reg.push(
                            run_id,
                            super::run_events::RunEvent::new(
                                super::run_events::EventKind::BlockDetected,
                                format!("{name}: captcha/anti-bot wall"),
                            ),
                        )
                        .await;
                    }
                    let out_preview: String = output
                        .chars()
                        .take(60)
                        .collect::<String>()
                        .replace('\n', " ")
                        .trim()
                        .to_string();
                    let label = if out_preview.is_empty() {
                        format!("{name} done")
                    } else {
                        format!("{name} → {out_preview}")
                    };
                    reg.push(
                        run_id,
                        super::run_events::RunEvent::new(
                            super::run_events::EventKind::ToolCallEnd,
                            label,
                        ),
                    )
                    .await;
                }
            }
            Some(AgentEvent::PermissionRequest {
                call_id, tool_name, ..
            }) => {
                tracing::debug!(tool = %tool_name, "auto-approving research tool");
                eprintln!("  [research] auto-approve: {tool_name}");
                let _ = handle
                    .permissions
                    .send(crate::types::PermissionResponse {
                        call_id,
                        allowed: true,
                    })
                    .await;
            }
            Some(AgentEvent::TextDelta(t)) => {
                stats.text_deltas += 1;
                if stats.text_deltas <= 3 || stats.text_deltas.is_multiple_of(50) {
                    let preview: String = t.chars().take(80).collect();
                    eprintln!("  [research] text delta #{}: {preview}", stats.text_deltas);
                }
            }
            Some(_) => continue,
            None => {
                tracing::info!(
                    tool_calls,
                    text_deltas = stats.text_deltas,
                    errors = stats.errors,
                    "research agent stream closed"
                );
                eprintln!(
                    "  [research] stream closed: {tool_calls} tools, {} text deltas, {} errors",
                    stats.text_deltas, stats.errors
                );
                eprintln!("  {}", stats.summary_line());
                return StopReason::StreamClosed;
            }
        }
    }
}

// Re-export so callers outside this module can't forget to avoid the private
// coordinator-side Arc<dyn AgentRunner>.
pub use self::AgentRunner as _AgentRunnerReexport;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::research::spec::{Finding, ResearchSpec, dedup_hash};
    use crate::research::store::FsResearchStore;
    use crate::types::{AgentEvent, PermissionResponse};
    use async_trait::async_trait;
    use tempfile::tempdir;
    use tokio::sync::{Mutex, mpsc};

    #[test]
    fn parse_provider_model_pair_splits_on_first_slash() {
        assert_eq!(
            parse_provider_model_pair("qwen/qwen3.6-plus"),
            Some(("qwen".into(), "qwen3.6-plus".into()))
        );
        assert_eq!(
            parse_provider_model_pair("kimi-code/kimi-for-coding"),
            Some(("kimi-code".into(), "kimi-for-coding".into()))
        );
    }

    #[test]
    fn parse_provider_model_pair_preserves_inner_slashes() {
        // OpenRouter-style ids — the model half can contain `/`.
        assert_eq!(
            parse_provider_model_pair("openrouter/anthropic/claude-3.5-sonnet"),
            Some(("openrouter".into(), "anthropic/claude-3.5-sonnet".into(),))
        );
    }

    #[test]
    fn parse_provider_model_pair_no_slash_returns_none() {
        // Backward-compat: bare model names route through the current
        // provider, the same as before this helper existed.
        assert_eq!(parse_provider_model_pair("qwen3.6-plus"), None);
        assert_eq!(parse_provider_model_pair("MiniMax-M2.5"), None);
    }

    #[test]
    fn parse_provider_model_pair_rejects_empty_halves() {
        assert_eq!(parse_provider_model_pair("/qwen3.6-plus"), None);
        assert_eq!(parse_provider_model_pair("qwen/"), None);
        assert_eq!(parse_provider_model_pair("/"), None);
        // Whitespace-only halves are also rejected.
        assert_eq!(parse_provider_model_pair("  /qwen3.6-plus"), None);
        assert_eq!(parse_provider_model_pair("qwen/   "), None);
    }

    #[test]
    fn parse_provider_model_pair_trims_whitespace() {
        assert_eq!(
            parse_provider_model_pair("  kimi-code  /  kimi-for-coding  "),
            Some(("kimi-code".into(), "kimi-for-coding".into()))
        );
    }

    fn make_spec(id: &str, topic: &str) -> ResearchSpec {
        ResearchSpec {
            id: id.to_string(),
            topic: topic.to_string(),
            sources: vec!["https://example.com".into()],
            interval_seconds: None,
            run_at: None,
            cron: None,
            task_timeout_seconds: None,
            session_id: None,
            chat_id: None,
            thread_id: None,
            provider: None,
            model: None,
            max_iterations: None,
            max_wall_seconds: Some(5),
            created_at: Utc::now(),
            paused: false,
            pause_reason: None,
        }
    }

    /// Test double: records every `config.default_model` seen, always
    /// returns a provider error so the coordinator walks the entire
    /// (selector-trimmed) fallback chain. Used by the Phase 2 tests to
    /// assert that `try_start_with_fallback` consults the capability
    /// selector before dispatching.
    struct RecordingFailingRunner {
        seen_models: Arc<Mutex<Vec<String>>>,
    }

    #[async_trait]
    impl AgentRunner for RecordingFailingRunner {
        async fn start_research_turn(
            &self,
            _spec: &ResearchSpec,
            _prompt: &str,
            config: &CoordinatorConfig,
            _run_id: &str,
        ) -> Result<(AgentHandle, String, String)> {
            let m = config
                .default_model
                .clone()
                .unwrap_or_else(|| "(none)".into());
            self.seen_models.lock().await.push(m);
            Err(crate::error::AgentError::Provider(
                "recording runner forces fallback walk".into(),
            ))
        }

        async fn cleanup_research_session(&self, _session_id: &str) {}
    }

    fn cap_provider(
        models: &[&str],
        caps: &[(&str, crate::model_catalog::ModelCapabilities)],
    ) -> crate::config::ProviderConfig {
        crate::config::ProviderConfig {
            provider_type: "openai_compat".into(),
            api_key: "k".into(),
            api_keys: Vec::new(),
            base_url: None,
            models: models.iter().map(|s| s.to_string()).collect(),
            max_tokens: None,
            temperature: None,
            context_window: None,
            headers: Default::default(),
            supports_vision: None,
            model_aliases: Default::default(),
            capabilities: caps
                .iter()
                .map(|(k, v)| (k.to_string(), v.clone()))
                .collect(),
        }
    }

    #[tokio::test]
    async fn try_start_with_fallback_soft_mode_walks_entire_chain() {
        // Soft mode (enforce=false) must preserve legacy behaviour: every
        // fallback is attempted, in order, even if the selector would
        // have dropped it in hard mode.
        let tmp = tempdir().unwrap();
        let store: Arc<dyn ResearchStore> =
            Arc::new(FsResearchStore::new(tmp.path().to_path_buf()));
        let spec = make_spec("sel-soft", "x");
        store.create_spec(&spec).await.unwrap();

        let seen = Arc::new(Mutex::new(Vec::new()));
        let runner = Arc::new(RecordingFailingRunner {
            seen_models: seen.clone(),
        });

        let mut dead_caps = crate::model_catalog::ModelCapabilities::unknown();
        dead_caps.status = crate::model_catalog::ModelStatus::Deprecated;
        let mut providers = HashMap::new();
        providers.insert(
            "zai".into(),
            cap_provider(&["glm-5", "dead-model"], &[("dead-model", dead_caps)]),
        );

        let coord_cfg = CoordinatorConfig {
            default_provider: Some("zai".into()),
            default_model: Some("glm-5".into()),
            fallback_models: vec!["zai/dead-model".into(), "zai/glm-5".into()],
            provider_capabilities: providers,
            enforce_model_capabilities: false,
            ..CoordinatorConfig::default()
        };
        let coord = ResearchCoordinator::new(store.clone(), runner, coord_cfg);
        let _ = coord.run_once("sel-soft").await;

        let seen = seen.lock().await.clone();
        // Primary + 2 fallbacks = 3 attempts. Deprecated stays in soft mode.
        assert_eq!(
            seen.len(),
            3,
            "soft mode must try every entry, got {seen:?}"
        );
        assert!(seen.iter().any(|m| m.ends_with("dead-model")));
    }

    #[tokio::test]
    async fn try_start_with_fallback_hard_mode_drops_deprecated() {
        let tmp = tempdir().unwrap();
        let store: Arc<dyn ResearchStore> =
            Arc::new(FsResearchStore::new(tmp.path().to_path_buf()));
        let spec = make_spec("sel-hard", "x");
        store.create_spec(&spec).await.unwrap();

        let seen = Arc::new(Mutex::new(Vec::new()));
        let runner = Arc::new(RecordingFailingRunner {
            seen_models: seen.clone(),
        });

        let mut dead = crate::model_catalog::ModelCapabilities::unknown();
        dead.status = crate::model_catalog::ModelStatus::Deprecated;
        let chat_only = crate::model_catalog::ModelCapabilities {
            task_fit: vec![crate::model_catalog::TaskKind::Chat],
            ..crate::model_catalog::ModelCapabilities::unknown()
        };
        let live = crate::model_catalog::ModelCapabilities {
            task_fit: vec![crate::model_catalog::TaskKind::Research],
            ..crate::model_catalog::ModelCapabilities::unknown()
        };
        let mut providers = HashMap::new();
        providers.insert(
            "qwen".into(),
            cap_provider(
                &["qwen3.6-plus", "chat-only-model", "qwen3.5-plus"],
                &[
                    ("qwen3.6-plus", dead),
                    ("chat-only-model", chat_only),
                    ("qwen3.5-plus", live),
                ],
            ),
        );

        let coord_cfg = CoordinatorConfig {
            default_provider: Some("qwen".into()),
            default_model: Some("qwen3.6-plus".into()),
            // Mix: one deprecated, one off-task, one valid.
            fallback_models: vec![
                "qwen/qwen3.6-plus".into(),
                "qwen/chat-only-model".into(),
                "qwen/qwen3.5-plus".into(),
            ],
            provider_capabilities: providers,
            enforce_model_capabilities: true,
            ..CoordinatorConfig::default()
        };
        let coord = ResearchCoordinator::new(store.clone(), runner, coord_cfg);
        let _ = coord.run_once("sel-hard").await;

        let seen = seen.lock().await.clone();
        // Primary always tried (operator-pinned). Fallbacks: only the
        // live qwen3.5-plus survives the selector trim.
        assert_eq!(seen.len(), 2, "hard mode trims chain; got {seen:?}");
        // Primary first — carries whatever default_model we passed.
        assert!(seen[0].contains("qwen3.6-plus"), "primary first: {seen:?}");
        // Then only the live one survives.
        assert!(
            seen[1] == "qwen3.5-plus" || seen[1] == "qwen/qwen3.5-plus",
            "expected only live fallback, got {seen:?}"
        );
    }

    #[tokio::test]
    async fn try_start_with_fallback_hard_mode_rewrites_bare_model_back_to_bare() {
        // Regression guard: in hard mode, if the operator wrote a bare
        // model id (no provider prefix) and the selector is happy with
        // it, we must hand the *same* bare id back to the runner so
        // downstream resolution matches the operator's spelling.
        let tmp = tempdir().unwrap();
        let store: Arc<dyn ResearchStore> =
            Arc::new(FsResearchStore::new(tmp.path().to_path_buf()));
        let spec = make_spec("sel-bare", "x");
        store.create_spec(&spec).await.unwrap();

        let seen = Arc::new(Mutex::new(Vec::new()));
        let runner = Arc::new(RecordingFailingRunner {
            seen_models: seen.clone(),
        });

        let live = crate::model_catalog::ModelCapabilities {
            task_fit: vec![crate::model_catalog::TaskKind::Research],
            ..crate::model_catalog::ModelCapabilities::unknown()
        };
        let mut providers = HashMap::new();
        providers.insert(
            "kimi-code".into(),
            cap_provider(&["kimi-for-coding"], &[("kimi-for-coding", live)]),
        );

        let coord_cfg = CoordinatorConfig {
            default_provider: Some("kimi-code".into()),
            default_model: Some("kimi-for-coding".into()),
            fallback_models: vec!["kimi-for-coding".into()],
            provider_capabilities: providers,
            enforce_model_capabilities: true,
            ..CoordinatorConfig::default()
        };
        let coord = ResearchCoordinator::new(store.clone(), runner, coord_cfg);
        let _ = coord.run_once("sel-bare").await;

        let seen = seen.lock().await.clone();
        // Primary + 1 fallback = 2 attempts. Fallback must stay bare.
        assert_eq!(seen.len(), 2, "primary + fallback expected, got {seen:?}");
        assert_eq!(
            seen[1], "kimi-for-coding",
            "bare entry must remain bare after round-trip",
        );
    }

    /// Test double: sends a scripted sequence of events, optionally writing
    /// findings to the store mid-stream to simulate the real tool path.
    struct ScriptedRunner {
        events: Mutex<Vec<AgentEvent>>,
        findings: Mutex<Vec<Finding>>,
        store: Arc<dyn ResearchStore>,
        close_without_idle: bool,
        delay_per_event: Duration,
    }

    #[async_trait]
    impl AgentRunner for ScriptedRunner {
        async fn start_research_turn(
            &self,
            _spec: &ResearchSpec,
            _prompt: &str,
            _config: &CoordinatorConfig,
            _run_id: &str,
        ) -> Result<(AgentHandle, String, String)> {
            let (tx, rx) = mpsc::channel(16);
            let (perm_tx, _perm_rx) = mpsc::channel::<PermissionResponse>(4);
            let events = std::mem::take(&mut *self.events.lock().await);
            let findings = std::mem::take(&mut *self.findings.lock().await);
            let store = self.store.clone();
            let close_without_idle = self.close_without_idle;
            let delay = self.delay_per_event;
            tokio::spawn(async move {
                for ev in events {
                    if delay > Duration::ZERO {
                        tokio::time::sleep(delay).await;
                    }
                    let _ = tx.send(ev).await;
                }
                for f in findings {
                    let _ = store.try_append_finding(&f).await;
                }
                if !close_without_idle {
                    let _ = tx.send(AgentEvent::Idle).await;
                }
            });
            Ok((
                AgentHandle {
                    events: rx,
                    permissions: perm_tx,
                },
                "test-provider".into(),
                "test-model".into(),
            ))
        }

        async fn cleanup_research_session(&self, _session_id: &str) {}
    }

    fn finding(spec_id: &str, url: &str, run_id: &str) -> Finding {
        use crate::research::spec::{content_hash, host_path_hash};
        Finding {
            id: uuid::Uuid::new_v4().simple().to_string(),
            research_id: spec_id.to_string(),
            run_id: run_id.to_string(),
            url: url.to_string(),
            title: Some("t".into()),
            excerpt: None,
            price: None,
            listing_date: None,
            source_content: None,
            dedup_hash: dedup_hash(url),
            host_path_hash: host_path_hash(url),
            content_hash: content_hash(""),
            seen_at: Utc::now(),
        }
    }

    #[tokio::test]
    async fn run_once_agent_idle_writes_run_record() {
        let tmp = tempdir().unwrap();
        let store: Arc<dyn ResearchStore> =
            Arc::new(FsResearchStore::new(tmp.path().to_path_buf()));
        let spec = make_spec("jag-1", "jag");
        store.create_spec(&spec).await.unwrap();

        let runner = Arc::new(ScriptedRunner {
            events: Mutex::new(vec![]),
            findings: Mutex::new(vec![
                finding("jag-1", "https://ex.com/a", "r"),
                finding("jag-1", "https://ex.com/b", "r"),
            ]),
            store: store.clone(),
            close_without_idle: false,
            delay_per_event: Duration::ZERO,
        });
        let coord = ResearchCoordinator::new(store.clone(), runner, CoordinatorConfig::default());
        let report = coord.run_once("jag-1").await.unwrap();

        assert_eq!(report.stop_reason, StopReason::AgentIdle);
        assert_eq!(report.new_findings, 2);
        assert_eq!(report.total_findings_after, 2);
        assert_eq!(report.provider, "test-provider");
        assert_eq!(report.model, "test-model");

        let runs = store.list_runs("jag-1", None).await.unwrap();
        assert_eq!(runs.len(), 1);
        assert_eq!(runs[0].new_findings, 2);
        assert_eq!(runs[0].stop_reason, "agent_idle");
        assert!(store.read_report("jag-1").await.unwrap().is_some());
    }

    #[tokio::test]
    async fn run_once_timeout_short_circuits_stream() {
        let tmp = tempdir().unwrap();
        let store: Arc<dyn ResearchStore> =
            Arc::new(FsResearchStore::new(tmp.path().to_path_buf()));
        let mut spec = make_spec("jag-2", "jag");
        spec.max_wall_seconds = Some(1);
        store.create_spec(&spec).await.unwrap();

        // Scripted runner sleeps 5s before emitting Idle — coordinator must
        // time out first and still write a run record.
        let runner = Arc::new(ScriptedRunner {
            events: Mutex::new(vec![AgentEvent::TextDelta("hi".into())]),
            findings: Mutex::new(vec![]),
            store: store.clone(),
            close_without_idle: false,
            delay_per_event: Duration::from_secs(5),
        });
        let coord = ResearchCoordinator::new(store.clone(), runner, CoordinatorConfig::default());
        let report = coord.run_once("jag-2").await.unwrap();

        assert_eq!(report.stop_reason, StopReason::Timeout);
        let runs = store.list_runs("jag-2", None).await.unwrap();
        assert_eq!(runs.len(), 1);
        assert_eq!(runs[0].stop_reason, "timeout");
    }

    #[tokio::test]
    async fn run_once_with_cancel_short_circuits_when_token_fired() {
        // Scripted runner sleeps 60s before emitting any event. We cancel
        // the token after 100ms and expect the coordinator to return
        // `StopReason::Cancelled` in well under a second.
        let tmp = tempdir().unwrap();
        let store: Arc<dyn ResearchStore> =
            Arc::new(FsResearchStore::new(tmp.path().to_path_buf()));
        let mut spec = make_spec("cancel-1", "jag");
        spec.max_wall_seconds = Some(60); // generous, cancellation must win
        store.create_spec(&spec).await.unwrap();

        let runner = Arc::new(ScriptedRunner {
            events: Mutex::new(vec![AgentEvent::TextDelta("slow".into())]),
            findings: Mutex::new(vec![]),
            store: store.clone(),
            close_without_idle: false,
            delay_per_event: Duration::from_secs(60),
        });
        let coord = ResearchCoordinator::new(store.clone(), runner, CoordinatorConfig::default());

        let cancel = CancellationToken::new();
        let cancel_fire = cancel.clone();
        tokio::spawn(async move {
            tokio::time::sleep(Duration::from_millis(100)).await;
            cancel_fire.cancel();
        });

        let started = std::time::Instant::now();
        let report = coord
            .run_once_with_cancel("cancel-1", cancel)
            .await
            .unwrap();
        let elapsed = started.elapsed();
        assert_eq!(
            report.stop_reason,
            StopReason::Cancelled,
            "expected Cancelled, got {:?}",
            report.stop_reason
        );
        assert!(
            elapsed < Duration::from_secs(2),
            "cancellation should land sub-second, took {elapsed:?}"
        );
        let runs = store.list_runs("cancel-1", None).await.unwrap();
        assert_eq!(runs.len(), 1);
        assert_eq!(runs[0].stop_reason, "cancelled");
    }

    #[tokio::test]
    async fn drain_events_returns_cancelled_immediately_on_pre_cancelled_token() {
        // Pre-cancelled token: drain_events MUST observe it on the very
        // first iteration and return without consuming any event.
        let (_perm_tx, _perm_rx) = tokio::sync::mpsc::channel(1);
        let (ev_tx, ev_rx) = tokio::sync::mpsc::channel(8);
        let mut handle = AgentHandle {
            events: ev_rx,
            permissions: _perm_tx,
        };
        // Push some events; they must be ignored.
        ev_tx.send(AgentEvent::TextDelta("a".into())).await.unwrap();
        ev_tx.send(AgentEvent::Idle).await.unwrap();

        let mut stats = DrainStats::default();
        let cancel = CancellationToken::new();
        cancel.cancel();
        let reason = drain_events(&mut handle, &mut stats, &cancel, None, "test-run").await;
        assert_eq!(reason, StopReason::Cancelled);
        assert_eq!(
            stats.text_deltas, 0,
            "no events should have been consumed before cancellation"
        );
    }

    #[tokio::test]
    async fn run_verified_breaks_between_rounds_on_cancel() {
        // Scripted runner emits Idle quickly each round. Cancel the token
        // and assert that no further round starts after cancellation.
        let tmp = tempdir().unwrap();
        let store: Arc<dyn ResearchStore> =
            Arc::new(FsResearchStore::new(tmp.path().to_path_buf()));
        let spec = make_spec("verif-cancel", "jag");
        store.create_spec(&spec).await.unwrap();

        let runner = Arc::new(ScriptedRunner {
            events: Mutex::new(vec![]),
            findings: Mutex::new(vec![]),
            store: store.clone(),
            close_without_idle: false,
            delay_per_event: Duration::ZERO,
        });
        let coord = ResearchCoordinator::new(store.clone(), runner, CoordinatorConfig::default());
        let cancel = CancellationToken::new();
        cancel.cancel(); // already cancelled

        let started = std::time::Instant::now();
        let result = coord
            .run_verified_with_cancel("verif-cancel", 5, cancel)
            .await;
        let elapsed = started.elapsed();
        assert!(
            result.is_ok(),
            "should still produce a report, got {result:?}"
        );
        assert!(
            elapsed < Duration::from_secs(3),
            "cancelled verified loop should finish quickly, took {elapsed:?}"
        );
    }

    #[tokio::test]
    async fn run_once_paused_spec_is_skipped() {
        let tmp = tempdir().unwrap();
        let store: Arc<dyn ResearchStore> =
            Arc::new(FsResearchStore::new(tmp.path().to_path_buf()));
        let mut spec = make_spec("jag-p", "jag");
        spec.paused = true;
        store.create_spec(&spec).await.unwrap();

        let runner = Arc::new(ScriptedRunner {
            events: Mutex::new(vec![]),
            findings: Mutex::new(vec![finding("jag-p", "https://ex.com/x", "r")]),
            store: store.clone(),
            close_without_idle: false,
            delay_per_event: Duration::ZERO,
        });
        let coord = ResearchCoordinator::new(store.clone(), runner, CoordinatorConfig::default());
        let report = coord.run_once("jag-p").await.unwrap();
        assert_eq!(report.stop_reason, StopReason::Paused);
        assert_eq!(report.new_findings, 0);
        // Runner never fired — findings still zero.
        assert_eq!(store.count_findings("jag-p").await.unwrap(), 0);
    }

    #[tokio::test]
    async fn run_once_stream_closed_is_reported() {
        let tmp = tempdir().unwrap();
        let store: Arc<dyn ResearchStore> =
            Arc::new(FsResearchStore::new(tmp.path().to_path_buf()));
        let spec = make_spec("jag-c", "jag");
        store.create_spec(&spec).await.unwrap();

        let runner = Arc::new(ScriptedRunner {
            events: Mutex::new(vec![AgentEvent::TextDelta("bye".into())]),
            findings: Mutex::new(vec![]),
            store: store.clone(),
            close_without_idle: true,
            delay_per_event: Duration::ZERO,
        });
        let coord = ResearchCoordinator::new(store.clone(), runner, CoordinatorConfig::default());
        let report = coord.run_once("jag-c").await.unwrap();
        assert_eq!(report.stop_reason, StopReason::StreamClosed);
    }

    #[tokio::test]
    async fn build_prompt_carries_topic_sources_and_dedup_list() {
        let tmp = tempdir().unwrap();
        let store: Arc<dyn ResearchStore> =
            Arc::new(FsResearchStore::new(tmp.path().to_path_buf()));
        let spec = make_spec("jag-3", "Jaguar XF cheap Hanoi");
        store.create_spec(&spec).await.unwrap();
        store
            .try_append_finding(&finding("jag-3", "https://chotot.com/ad/42", "prev"))
            .await
            .unwrap();

        let runner = Arc::new(ScriptedRunner {
            events: Mutex::new(vec![]),
            findings: Mutex::new(vec![]),
            store: store.clone(),
            close_without_idle: false,
            delay_per_event: Duration::ZERO,
        });
        let coord = ResearchCoordinator::new(store.clone(), runner, CoordinatorConfig::default());
        let prompt = coord.build_prompt(&spec).await.unwrap();
        assert!(prompt.contains("Jaguar XF cheap Hanoi"));
        assert!(prompt.contains("https://example.com"));
        assert!(prompt.contains("https://chotot.com/ad/42"));
        assert!(prompt.contains("research_save"));
    }

    /// After splitting the long JS-gated recipe into the
    /// `web-browser-playbook` skill, the prompt itself only needs to:
    ///
    /// 1. Tell the agent the playbook *exists* and to load it via the
    ///    `Skill` tool when it sees one of the gated hosts. Pin both the
    ///    skill name and the canonical hostnames so a rename of either
    ///    side breaks this test loudly.
    /// 2. Keep the **hard rules around contacts** inline — these are
    ///    safety contracts the gatekeeper enforces and they MUST be on
    ///    the agent's screen on every run, not behind an on-demand load.
    ///    Verifies the literal `Contacts hidden behind site captcha —
    ///    visit URL` phrase, the no-fabrication rule, and the
    ///    no-`Liên hệ qua` paraphrase rule.
    ///
    /// The full per-host button text and CSS-selector tables live in
    /// `naked/skills/web-browser-playbook/SKILL.md`; that file has its
    /// own contract test below.
    #[tokio::test]
    async fn build_prompt_points_to_browser_skill_and_keeps_contact_safety_hatch() {
        let tmp = tempdir().unwrap();
        let store: Arc<dyn ResearchStore> =
            Arc::new(FsResearchStore::new(tmp.path().to_path_buf()));
        let spec = make_spec("vn-1", "Da Nang restaurant rentals");
        store.create_spec(&spec).await.unwrap();

        let runner = Arc::new(ScriptedRunner {
            events: Mutex::new(vec![]),
            findings: Mutex::new(vec![]),
            store: store.clone(),
            close_without_idle: false,
            delay_per_event: Duration::ZERO,
        });
        let coord = ResearchCoordinator::new(store.clone(), runner, CoordinatorConfig::default());
        let prompt = coord.build_prompt(&spec).await.unwrap();

        // (1) Pointer to the skill + hostnames the agent should map to it.
        assert!(
            prompt.contains("web-browser-playbook"),
            "prompt must name the skill so the agent can call \
             Skill(skill=\"web-browser-playbook\") — without the literal \
             name the on-demand path is unreachable"
        );
        for host in [
            "alonhadat.com.vn",
            "nhadat24h.net",
            "batdongsan.com.vn",
            "dotproperty.com.vn",
            "mogi.vn",
            "homedy.com",
        ] {
            assert!(
                prompt.contains(host),
                "host `{host}` missing — agent will not know to load the \
                 playbook before fetching this domain"
            );
        }

        // (2) Hard contact rules stay inline (gatekeeper-enforced).
        for clause in [
            "Contacts hidden behind site captcha — visit URL",
            "Never invent a phone number",
            "Liên hệ qua",
        ] {
            assert!(
                prompt.contains(clause),
                "safety clause `{clause}` missing — without it the \
                 gatekeeper has no shared phrase with the agent and \
                 fabricated contacts can slip through"
            );
        }
    }

    /// The skill body itself is the source of truth for the per-host
    /// recipes; the prompt only points to it. If anyone deletes the file
    /// or removes one of the host rows, the agent's on-demand load
    /// produces an empty/incomplete playbook and the click-to-reveal
    /// flow silently degrades to "no contacts found".
    ///
    /// We pin the same surface the prompt promises:
    /// - File exists at the canonical project path.
    /// - YAML-ish front-matter has the `name:` and `description:` lines
    ///   the `SkillResolver` and `SkillTool` rely on.
    /// - Every host the prompt mentions has at least one row in the
    ///   playbook.
    /// - The four MCP browser tool names we tell the agent to use
    ///   (`browser_navigate`, `browser_snapshot`, `browser_click`,
    ///   `browser_wait`) appear at least once each.
    /// - The captcha safety prefix is mentioned (so the playbook agrees
    ///   with the gatekeeper-enforced inline rule).
    #[test]
    fn web_browser_playbook_skill_file_matches_prompt_contract() {
        // CARGO_MANIFEST_DIR for naked-core is `naked/crates/naked-core`,
        // so the project skills live two levels up.
        let manifest = std::path::Path::new(env!("CARGO_MANIFEST_DIR"));
        let skill_path = manifest
            .join("../../skills/web-browser-playbook/SKILL.md")
            .canonicalize()
            .expect(
                "skills/web-browser-playbook/SKILL.md must exist — the prompt \
                 forwards every JS-gated request to it; missing file means \
                 broken on-demand load",
            );
        let body = std::fs::read_to_string(&skill_path).expect("skill file readable");

        // Front-matter the SkillResolver / SkillTool depend on.
        assert!(
            body.contains("name: web-browser-playbook"),
            "skill must declare its canonical name in front-matter; \
             SkillResolver matches case-insensitively but the description \
             hook in SkillTool::build_description prefers the declared name"
        );
        assert!(
            body.contains("description:"),
            "skill must have a `description:` line so SkillTool can show \
             it in the tool catalog the LLM reads"
        );

        // Every host the prompt forwards must have a row here.
        for host in [
            "alonhadat.com.vn",
            "nhadat24h.net",
            "batdongsan.com.vn",
            "dotproperty.com.vn",
            "chotot.com",
            "mogi.vn",
        ] {
            assert!(
                body.contains(host),
                "skill body missing host `{host}` — prompt forwards but \
                 playbook has no recipe → degraded silently"
            );
        }

        // MCP browser tool names the playbook tells the agent to call.
        for tool in [
            "browser_navigate",
            "browser_snapshot",
            "browser_click",
            "browser_wait",
        ] {
            assert!(
                body.contains(tool),
                "skill body missing MCP tool reference `{tool}` — recipe \
                 is incomplete"
            );
        }

        // Safety hatch agreement with the inline prompt rule.
        assert!(
            body.contains("Contacts hidden behind site captcha — visit URL"),
            "skill must mention the literal captcha-fallback phrase so the \
             agent applies it consistently with the gatekeeper-checked \
             phrase from the inline prompt rule"
        );

        // The "hard captcha walls — abandon fast" section is what stopped
        // the live agent from burning the whole wall-clock budget on
        // unsolvable Cloudflare/recaptcha pages (see probe runs #3 and #5
        // in 2026-04-19 CHANGELOG entry). Without it the agent flailed
        // through 6+ browser_evaluate / browser_run_code calls trying to
        // "investigate" the captcha. We pin the unique markers from that
        // section so an accidental delete during a future skill rewrite
        // surfaces as a test failure rather than a regression on live
        // alonhadat / Cloudflare URLs.
        for marker in [
            "Hard captcha walls",
            "xac-thuc",          // alonhadat interstitial path used as detection signal
            "at most ONE retry", // the rule that caps the flailing
            "do not flail",      // explicit anti-pattern phrase agent quotes back
        ] {
            assert!(
                body.to_lowercase().contains(&marker.to_lowercase()),
                "skill body missing hard-captcha guidance marker `{marker}` — \
                 without it the agent will burn its whole wall-clock budget \
                 on unsolvable bot challenges"
            );
        }

        // Operator-side levers that ship with the agent_role
        // BrowserRuntime field (proxy + Chromium extensions). The
        // playbook is the single source of truth pointing operators
        // at these knobs; if the section disappears, captcha-prone
        // deployments will sit at sub-50% extraction rates without
        // anyone realising there's a config knob to flip.
        for marker in [
            "Residential proxy", // names the technique unambiguously
            "BrowserRuntime",    // field operators set in naked.json
            "CapSolver",         // canonical example extension
            "CAPSOLVER_API_KEY", // env var the extension reads
        ] {
            assert!(
                body.contains(marker),
                "skill body missing operator-lever marker `{marker}` — \
                 captcha-prone deployments need the config-side fix \
                 documented next to the model-side guidance"
            );
        }
    }

    #[test]
    fn drain_stats_summary_line_is_grep_friendly() {
        // Pin the on-wire shape of the [research-summary] line because
        // `naked research probe` and any future regression script greps
        // for it. Reordering keys, dropping fields, or changing
        // separators silently breaks every downstream consumer.
        let mut s = DrainStats::default();
        s.note_tool("Skill");
        s.note_tool("browser_navigate");
        s.note_tool("browser_navigate");
        s.note_tool_output("Just a moment... Enable JavaScript and cookies");
        s.note_tool_output("plain page body, no markers");
        s.text_deltas = 7;
        s.errors = 0;

        let line = s.summary_line();
        assert!(line.starts_with("[research-summary] "));
        assert!(line.contains("Skill=1"));
        assert!(line.contains("browser_navigate=2"));
        assert!(line.contains("captcha_hits=1"));
        assert!(line.contains("skill_loads=1"));
        assert!(line.contains("text_deltas=7"));
        assert!(line.contains("errors=0"));
    }

    #[tokio::test]
    async fn verify_findings_catches_quality_issues() {
        let tmp = tempdir().unwrap();
        let store: Arc<dyn ResearchStore> =
            Arc::new(FsResearchStore::new(tmp.path().to_path_buf()));
        let spec = make_spec("gk-1", "test gatekeeper");
        store.create_spec(&spec).await.unwrap();

        // Finding 1: complete — should pass (using a reliably live URL)
        let mut f1 = finding("gk-1", "https://www.google.com", "r1");
        f1.title = Some("Google".into());
        f1.price = Some("free".into());
        f1.listing_date = Some("18/04/2026".into());
        f1.excerpt = Some("A".repeat(250));
        f1.source_content = Some("B".repeat(200));
        store.try_append_finding(&f1).await.unwrap();

        // Finding 2: missing date, short excerpt, no source_content
        let mut f2 = finding("gk-1", "https://httpbin.org/get", "r1");
        f2.title = Some("Httpbin".into());
        f2.price = Some("10 USD".into());
        f2.listing_date = None;
        f2.excerpt = Some("short".into());
        f2.source_content = None;
        store.try_append_finding(&f2).await.unwrap();

        // Finding 3: stale date (2024)
        let mut f3 = finding("gk-1", "https://httpbin.org/status/200", "r1");
        f3.title = Some("Old listing".into());
        f3.price = Some("5 USD".into());
        f3.listing_date = Some("01/01/2024".into());
        f3.excerpt = Some("C".repeat(300));
        f3.source_content = Some("D".repeat(200));
        store.try_append_finding(&f3).await.unwrap();

        let runner = Arc::new(ScriptedRunner {
            events: Mutex::new(vec![]),
            findings: Mutex::new(vec![]),
            store: store.clone(),
            close_without_idle: false,
            delay_per_event: Duration::ZERO,
        });
        let coord = ResearchCoordinator::new(store.clone(), runner, CoordinatorConfig::default());

        let verdict = coord.verify_findings("gk-1").await;

        // f2: missing date, short excerpt, missing source_content
        assert!(
            verdict.missing_dates >= 1,
            "should detect missing date, got {}",
            verdict.missing_dates
        );
        assert!(
            verdict.short_excerpts >= 1,
            "should detect short excerpt, got {}",
            verdict.short_excerpts
        );
        assert!(
            verdict.missing_source_content >= 1,
            "should detect missing source_content, got {}",
            verdict.missing_source_content
        );

        // f3: stale date should be flagged for removal
        assert!(
            verdict.stale_dates >= 1,
            "should detect stale date, got {}",
            verdict.stale_dates
        );
        assert!(
            !verdict.dead_hashes.is_empty(),
            "stale finding should be in dead_hashes"
        );

        // f2 (live URL with quality issues) should be in remediation_urls
        assert!(
            !verdict.remediation_urls.is_empty(),
            "should have remediation URLs"
        );

        // Feedback prompt should include specific sections
        let feedback = coord.build_feedback_prompt("gk-1", &verdict).await.unwrap();
        assert!(
            feedback.contains("Missing listing_date"),
            "feedback should mention missing dates"
        );
        assert!(
            feedback.contains("source_content"),
            "feedback should mention missing source_content"
        );
        assert!(
            feedback.contains("Too-short excerpts"),
            "feedback should mention short excerpts"
        );
    }

    #[tokio::test]
    async fn upsert_finding_updates_existing() {
        let tmp = tempdir().unwrap();
        let store: Arc<dyn ResearchStore> =
            Arc::new(FsResearchStore::new(tmp.path().to_path_buf()));
        let spec = make_spec("up-1", "upsert test");
        store.create_spec(&spec).await.unwrap();

        let mut f = finding("up-1", "https://example.com/listing/1", "r1");
        f.title = Some("Original title".into());
        f.excerpt = Some("short".into());
        f.listing_date = None;
        f.source_content = None;
        assert!(
            !store.upsert_finding(&f).await.unwrap(),
            "first insert should not be update"
        );
        assert_eq!(store.count_findings("up-1").await.unwrap(), 1);

        // Now upsert with better data
        f.title = Some("Updated title".into());
        f.excerpt = Some("A".repeat(500));
        f.listing_date = Some("18/04/2026".into());
        f.source_content = Some("B".repeat(1000));
        assert!(
            store.upsert_finding(&f).await.unwrap(),
            "second save should be update"
        );
        assert_eq!(
            store.count_findings("up-1").await.unwrap(),
            1,
            "count should not change"
        );

        let findings = store.list_findings("up-1", None).await.unwrap();
        assert_eq!(findings.len(), 1);
        assert_eq!(findings[0].title.as_deref(), Some("Updated title"));
        assert_eq!(findings[0].listing_date.as_deref(), Some("18/04/2026"));
        assert!(findings[0].source_content.is_some());
    }

    #[test]
    fn fuzzy_fingerprint_matches_near_duplicates() {
        let a = FuzzyFingerprint::new(
            Some("Cho thuê căn hộ 70m² 2PN District 7"),
            Some("$500/month"),
        )
        .expect("fp a");
        let b = FuzzyFingerprint::new(
            Some("Cho thuê 70m² 2PN District 7 có bếp đầy đủ"),
            Some("USD 500 / mo"),
        )
        .expect("fp b");
        assert!(a.is_duplicate_of(&b), "near-dup pair must collapse");
        assert!(b.is_duplicate_of(&a));
    }

    #[test]
    fn fuzzy_fingerprint_distinguishes_different_listings() {
        let a = FuzzyFingerprint::new(Some("Cho thuê căn hộ 70m² 2PN District 7"), Some("$500/mo"))
            .expect("fp a");
        // Different area + different district + different price → unrelated.
        let b = FuzzyFingerprint::new(
            Some("Cho thuê căn hộ 120m² 3PN District 2"),
            Some("$900/mo"),
        )
        .expect("fp b");
        assert!(!a.is_duplicate_of(&b));
    }

    #[test]
    fn fuzzy_fingerprint_requires_both_price_and_token_overlap() {
        let a = FuzzyFingerprint::new(Some("Cho thuê căn hộ 70m² 2PN District 7"), Some("$500/mo"))
            .expect("fp a");
        // Same title but a different price should NOT collapse — prevents
        // false merges of two units in the same building at different rates.
        let b = FuzzyFingerprint::new(Some("Cho thuê căn hộ 70m² 2PN District 7"), Some("$650/mo"))
            .expect("fp b");
        assert!(!a.is_duplicate_of(&b));
    }

    #[test]
    fn fuzzy_fingerprint_skips_too_generic_titles() {
        let fp = FuzzyFingerprint::new(Some("Cho thuê 70m²"), Some("$500"));
        // After dropping filler / short tokens this falls below the 3-token
        // floor, so we refuse to fingerprint it (would over-collapse).
        assert!(fp.is_none());
    }

    #[test]
    fn fuzzy_fingerprint_none_for_short_title() {
        let fp = FuzzyFingerprint::new(Some("hi"), Some("100"));
        assert!(fp.is_none(), "title <10 chars should return None");
    }

    #[test]
    fn fuzzy_fingerprint_none_without_title() {
        let fp = FuzzyFingerprint::new(None, Some("1000"));
        assert!(fp.is_none());
    }

    #[test]
    fn fuzzy_fingerprint_none_without_price() {
        // Title alone without price — fingerprint still works (price_digits empty)
        let fp = FuzzyFingerprint::new(Some("beautiful apartment ocean view"), None);
        assert!(fp.is_some(), "should fingerprint even without price");
    }

    #[test]
    fn fuzzy_fingerprint_jaccard_identical() {
        let a =
            FuzzyFingerprint::new(Some("luxury condo ocean view phuket"), Some("5000000")).unwrap();
        let b =
            FuzzyFingerprint::new(Some("luxury condo ocean view phuket"), Some("5000000")).unwrap();
        assert!(a.is_duplicate_of(&b));
    }

    #[test]
    fn fuzzy_fingerprint_different_price_not_dup() {
        let a =
            FuzzyFingerprint::new(Some("luxury condo ocean view phuket"), Some("5000000")).unwrap();
        let b =
            FuzzyFingerprint::new(Some("luxury condo ocean view phuket"), Some("9999999")).unwrap();
        assert!(
            !a.is_duplicate_of(&b),
            "different prices should not be duplicates"
        );
    }
}
