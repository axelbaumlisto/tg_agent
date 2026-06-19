//! run_verified*, try_start_with_fallback — larger ResearchCoordinator methods.

use std::time::Duration;

use tokio_util::sync::CancellationToken;

use crate::error::Result;
use crate::types::AgentHandle;

use super::{
    DrainStats, ResearchCoordinator, ResearchSpec, StopReason, VerificationSummary,
    VerifiedRunReport, drain_events,
};

impl ResearchCoordinator {
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
                    crate::research::run_events::RunEvent::new(
                        crate::research::run_events::EventKind::IterationStart,
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
                .write_record(super::briefing_ops::WriteRecordArgs {
                    spec: &spec,
                    run_id: &re_run_id,
                    new_findings: new_in_rerun,
                    reason: stop_reason,
                    started,
                    provider: &provider,
                    model: &model,
                    verification: None,
                })
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
            .write_record(super::briefing_ops::WriteRecordArgs {
                spec: &spec,
                run_id: &summary_run_id,
                new_findings: 0,
                reason: StopReason::AgentIdle,
                started: verified_started,
                provider: &last_report.provider,
                model: &last_report.model,
                verification: Some(verification),
            })
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
    pub(crate) async fn try_start_with_fallback(
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
