//! Research operations on AgentCore.

#[allow(unused_imports)]
use crate::error::{AgentError, Result};
#[allow(unused_imports)]
use crate::provider;
#[allow(unused_imports)]
use crate::research::coordinator::CoordinatorConfig;
#[allow(unused_imports)]
use crate::research::{
    self, ResearchCoordinator, ResearchPatch, ResearchSpec, ResearchStore, RunReport,
    VerifiedRunReport,
};
#[allow(unused_imports)]
use crate::types;
#[allow(unused_imports)]
use crate::{
    AgentCore, AgentCoreResearchRunner, acquire_research_permit, apply_research_patch,
    new_research_id, read_or_recover, write_or_recover, write_research_memory_link_for,
};
#[allow(unused_imports)]
use std::collections::HashMap;
#[allow(unused_imports)]
use std::sync::Arc;
#[allow(unused_imports)]
use tokio::sync::RwLock;
#[allow(unused_imports)]
use tokio_util::sync::CancellationToken;

/// RAII guard that owns the lifecycle of a research run's
/// `(cancel_token, events)` registration. Inserts the entry on
/// `install`, removes it on drop — so every exit path of
/// `run_research_*` (Ok, Err, panic) cleans up without boilerplate.
struct ResearchCancelGuard {
    cancels: Arc<RwLock<HashMap<String, CancellationToken>>>,
    events: research::RunEventRegistry,
    spec_id: String,
}

impl ResearchCancelGuard {
    async fn install(
        cancels: Arc<RwLock<HashMap<String, CancellationToken>>>,
        events: research::RunEventRegistry,
        spec_id: &str,
        token: CancellationToken,
    ) -> Self {
        cancels.write().await.insert(spec_id.to_string(), token);
        Self {
            cancels,
            events,
            spec_id: spec_id.to_string(),
        }
    }
}

impl Drop for ResearchCancelGuard {
    fn drop(&mut self) {
        // Take ownership of the necessary fields so the spawned future
        // doesn't outlive the guard. All state is cheaply clone-able.
        let cancels = self.cancels.clone();
        let events = self.events.clone();
        let spec_id = std::mem::take(&mut self.spec_id);
        tokio::spawn(async move {
            cancels.write().await.remove(&spec_id);
            events.drop_run(&spec_id).await;
        });
    }
}

impl AgentCore {
    /// Create a new research and persist its spec.
    ///
    /// `topic` is free text; the returned id is derived from it (slug + short
    /// random suffix) so it fits in a URL / systemd template.
    pub async fn create_research(
        &self,
        topic: &str,
        sources: Vec<String>,
        session_id: Option<String>,
        chat_id: Option<i64>,
        thread_id: Option<i32>,
    ) -> Result<ResearchSpec> {
        if !self.config().research.enabled {
            return Err(AgentError::Config("research subsystem is disabled".into()));
        }
        let mut seeds = sources;
        if seeds.is_empty() {
            seeds = self.config().research.default_sources.clone();
        }
        // Schedule defaults from config:
        // - default_cron takes priority over default_interval_seconds
        // - auto_first_run = true → run_at = now (scheduler picks up next tick)
        // - default_interval_seconds = 0 → no schedule (manual only)
        let rcfg = &self.config().research;
        let cron = rcfg.default_cron.clone();
        let interval = if cron.is_some() {
            None // cron takes priority
        } else if rcfg.default_interval_seconds > 0 {
            Some(rcfg.default_interval_seconds)
        } else {
            None
        };
        let run_at = if rcfg.auto_first_run {
            Some(chrono::Utc::now())
        } else {
            None
        };

        let spec = ResearchSpec {
            id: new_research_id(topic),
            topic: topic.trim().to_string(),
            sources: seeds,
            interval_seconds: interval,
            run_at,
            cron,
            task_timeout_seconds: None,
            session_id,
            chat_id,
            thread_id,
            provider: rcfg.provider.clone(),
            model: rcfg.model.clone(),
            max_iterations: Some(rcfg.max_iterations),
            max_wall_seconds: Some(rcfg.max_wall_seconds),
            created_at: chrono::Utc::now(),
            paused: false,
            pause_reason: None,
        };
        self.research.store.create_spec(&spec).await?;
        self.scheduler_hook()
            .notify(research::SchedulerEvent::SpecCreated {
                spec_id: spec.id.clone(),
            })
            .await;
        Ok(spec)
    }

    pub async fn list_research(&self) -> Result<Vec<ResearchSpec>> {
        self.research.list_specs().await
    }

    pub async fn load_research(&self, id: &str) -> Result<ResearchSpec> {
        self.research.load_spec(id).await
    }

    pub async fn delete_research(&self, id: &str) -> Result<()> {
        self.research.store.delete_spec(id).await?;
        self.scheduler_hook()
            .notify(research::SchedulerEvent::SpecRemoved {
                spec_id: id.to_string(),
            })
            .await;
        Ok(())
    }

    pub async fn set_research_paused(&self, id: &str, paused: bool) -> Result<()> {
        self.set_research_paused_with_reason(id, paused, None).await
    }

    /// Pause/resume a research spec and stamp a human-readable reason.
    /// `reason` is honoured ONLY when `paused == true`; on resume it is
    /// always cleared back to `None` so a subsequent `pause` doesn't
    /// inherit the previous reason silently.
    ///
    /// Used by the scheduler when auto-pausing after a failure streak
    /// (`reason = Some("auto: 5 consecutive failures — last error: …")`)
    /// so `/research ls` can show "auto" vs. user-initiated pauses
    /// without operators having to dig through `journalctl`.
    pub async fn set_research_paused_with_reason(
        &self,
        id: &str,
        paused: bool,
        reason: Option<String>,
    ) -> Result<()> {
        let mut spec = self.research.store.load_spec(id).await?;
        spec.paused = paused;
        spec.pause_reason = if paused { reason } else { None };
        self.research.store.save_spec(&spec).await?;
        self.scheduler_hook()
            .notify(research::SchedulerEvent::SpecUpdated {
                spec_id: id.to_string(),
            })
            .await;
        Ok(())
    }

    /// Answer a free-form question against a research's accumulated findings.
    ///
    /// Builds a one-shot prompt containing the topic + the latest N findings
    /// (title, price, date, url, excerpt) and calls the configured research
    /// provider/model with no tools. The LLM is asked to answer ONLY from the
    /// supplied corpus and to cite URLs by `[1]`-style numeric indices.
    ///
    /// Used by `/research ask` in the Telegram bot. Returns the raw text the
    /// model produced; callers should surface it as-is.
    pub async fn ask_research(&self, id: &str, question: &str) -> Result<String> {
        use tokio_stream::StreamExt;

        if !self.config().research.enabled {
            return Err(AgentError::Config("research subsystem is disabled".into()));
        }
        let q = question.trim();
        if q.is_empty() {
            return Err(AgentError::Config("question must not be empty".into()));
        }

        let spec = self.research.store.load_spec(id).await?;
        // Cap context: 30 most recent findings keeps us safely under typical
        // 16k token windows even for very long excerpts.
        const MAX_FINDINGS: usize = 30;
        const EXCERPT_BUDGET: usize = 600;
        let findings = self
            .research
            .store
            .list_findings(id, Some(MAX_FINDINGS))
            .await?;

        if findings.is_empty() {
            return Ok(format!(
                "No findings yet for `{id}` — run `/research run {id}` first."
            ));
        }

        let corpus = build_ask_corpus(&findings, EXCERPT_BUDGET);

        let provider_name = spec
            .provider
            .clone()
            .or_else(|| self.config().research.provider.clone())
            .unwrap_or_else(|| self.config().default_provider.clone());
        let model = spec
            .model
            .clone()
            .or_else(|| self.config().research.model.clone())
            .unwrap_or_else(|| self.config().default_model.clone());
        let provider = self.provider_for(&provider_name).await;

        let system = "You are a research assistant. Answer the user's question \
                      using ONLY the numbered findings provided. If the corpus \
                      does not contain the answer, say so plainly — do NOT \
                      invent details. When you cite specific findings, refer to \
                      them by their bracketed number, e.g. `[3]`. Keep the \
                      reply concise (under 1500 chars) and in the same language \
                      as the question.";
        let user = format!(
            "Research topic: {topic}\n\n# Findings\n\n{corpus}# Question\n\n{q}",
            topic = spec.topic,
        );

        let request = provider::ChatRequest {
            model,
            system: system.to_string(),
            messages: vec![serde_json::json!({"role": "user", "content": user})],
            tools: vec![],
            max_tokens: 1500,
            temperature: Some(0.2),
            reasoning: None,
        };

        let mut stream = provider.stream_chat(request).await.map_err(|e| {
            AgentError::ProviderTyped(crate::provider::error::ProviderError::Other {
                status: 0,
                body: format!("research_ask LLM: {e}"),
            })
        })?;

        let mut text = String::new();
        while let Some(chunk) = stream.next().await {
            match chunk {
                types::StreamChunk::Text(t) => text.push_str(&t),
                types::StreamChunk::Done => break,
                types::StreamChunk::Error(e) => {
                    return Err(AgentError::ProviderTyped(
                        crate::provider::error::ProviderError::Other {
                            status: 0,
                            body: format!("research_ask stream: {e}"),
                        },
                    ));
                }
                _ => {}
            }
        }
        if text.trim().is_empty() {
            return Err(AgentError::ProviderTyped(
                crate::provider::error::ProviderError::Other {
                    status: 0,
                    body: "research_ask LLM returned empty".into(),
                },
            ));
        }
        Ok(text)
    }

    /// Apply a partial mutation to a research spec. Only the fields set on
    /// `patch` (non-`None`) are touched; everything else is preserved.
    ///
    /// `interval_seconds` uses a double-`Option` so callers can distinguish
    /// "leave the schedule alone" (`None`) from "clear the schedule"
    /// (`Some(None)`).
    ///
    /// Returns the post-update spec for echoing back to LLM/UI callers.
    pub async fn update_research(&self, id: &str, patch: ResearchPatch) -> Result<ResearchSpec> {
        let mut spec = self.research.store.load_spec(id).await?;
        apply_research_patch(&mut spec, patch);
        self.research.store.save_spec(&spec).await?;
        tracing::info!(
            spec_id = %spec.id,
            topic = %spec.topic,
            interval_seconds = ?spec.interval_seconds,
            sources = spec.sources.len(),
            "research spec updated"
        );
        self.scheduler_hook()
            .notify(research::SchedulerEvent::SpecUpdated {
                spec_id: spec.id.clone(),
            })
            .await;
        Ok(spec)
    }

    /// Execute a single research pass synchronously. Returns the run summary.
    /// Telegram/CLI callers typically `tokio::spawn` this — a live run can
    /// take up to `max_wall_seconds` (20 min default).
    pub async fn run_research(self: Arc<Self>, id: &str) -> Result<RunReport> {
        self.run_research_with_cancel(id, tokio_util::sync::CancellationToken::new())
            .await
    }

    /// Cancellation-aware variant of [`Self::run_research`]. The scheduler
    /// uses this so its two-step `cancel → abort` shutdown can stop a
    /// runaway research at the next coordinator `await` point — without
    /// it, `JoinHandle::abort` lands only at the worker's next yield
    /// point, which can be many seconds away inside an HTTP/LLM stream.
    pub async fn run_research_with_cancel(
        self: Arc<Self>,
        id: &str,
        cancel: tokio_util::sync::CancellationToken,
    ) -> Result<RunReport> {
        if !self.config().research.enabled {
            return Err(AgentError::Config("research subsystem is disabled".into()));
        }
        let _permit = acquire_research_permit(&self.research.run_semaphore, id).await?;
        // Register the cancel token so the TG "Stop & clarify" callback
        // can signal it by spec_id. Cleared on exit — every path below
        // goes through the guard's `drop`.
        let _guard = ResearchCancelGuard::install(
            self.research.cancels.clone(),
            self.research.run_events.clone(),
            id,
            cancel.clone(),
        )
        .await;
        let coord = self.build_coordinator();
        let report = coord.run_once_with_cancel(id, cancel).await?;
        if let Err(e) = self
            .write_research_memory_link(id, &report.run_id, None)
            .await
        {
            tracing::warn!(spec = %id, "failed to record research memory link: {e}");
        }
        Ok(report)
    }

    /// Execute research with gatekeeper verification loop.
    /// After collecting findings, verifies URLs are live and data is complete.
    /// Dead findings are removed and the agent is re-run with feedback to find
    /// replacements. Up to `max_rounds` verification passes.
    pub async fn run_research_verified(
        self: Arc<Self>,
        id: &str,
        max_rounds: u32,
    ) -> Result<research::VerifiedRunReport> {
        self.run_research_verified_with_cancel(
            id,
            max_rounds,
            tokio_util::sync::CancellationToken::new(),
        )
        .await
    }

    /// Cancellation-aware variant of [`Self::run_research_verified`]. See
    /// [`Self::run_research_with_cancel`] for the rationale.
    pub async fn run_research_verified_with_cancel(
        self: Arc<Self>,
        id: &str,
        max_rounds: u32,
        cancel: tokio_util::sync::CancellationToken,
    ) -> Result<research::VerifiedRunReport> {
        if !self.config().research.enabled {
            return Err(AgentError::Config("research subsystem is disabled".into()));
        }
        let _permit = acquire_research_permit(&self.research.run_semaphore, id).await?;
        let _guard = ResearchCancelGuard::install(
            self.research.cancels.clone(),
            self.research.run_events.clone(),
            id,
            cancel.clone(),
        )
        .await;
        let coord = self.build_coordinator();
        let report = coord
            .run_verified_with_cancel(id, max_rounds, cancel)
            .await?;
        if let Err(e) = self
            .write_research_memory_link(id, &report.last_run.run_id, Some(&report))
            .await
        {
            tracing::warn!(spec = %id, "failed to record research memory link: {e}");
        }
        Ok(report)
    }

    /// Append a single global memory entry summarising a finished research run.
    /// Used as a breadcrumb so the LLM can answer "как там наше исследование"
    /// without trawling `runs.jsonl`. The entry's `source` field is set to
    /// `research/<spec_id>` so memory listings group naturally.
    ///
    /// `verified` is `Some(..)` when the run came from `run_verified`, with
    /// gatekeeper round/dead/replacement counters folded into the line.
    /// `report_path` falls back to `report=<unavailable>` for non-fs stores.
    /// Failures are returned to the caller so they can be logged at the
    /// invocation site without the helper logging twice.
    pub async fn write_research_memory_link(
        &self,
        spec_id: &str,
        run_id: &str,
        verified: Option<&VerifiedRunReport>,
    ) -> Result<()> {
        write_research_memory_link_for(
            self.research.store.as_ref(),
            &self.config().workspace,
            spec_id,
            run_id,
            verified,
            None,
        )
        .await
    }

    fn build_coordinator(self: &Arc<Self>) -> ResearchCoordinator {
        let coord_cfg = CoordinatorConfig {
            default_provider: self.config().research.provider.clone(),
            default_model: self.config().research.model.clone(),
            fallback_models: self.config().research.fallback_models.clone(),
            default_max_iterations: self.config().research.max_iterations,
            default_max_wall_seconds: self.config().research.max_wall_seconds,
            workspace: self.config().workspace.clone(),
            gatekeeper: self.config().research.gatekeeper.clone(),
            reasoning: self.config().research.reasoning.clone(),
            provider_capabilities: self.config().providers.clone(),
            enforce_model_capabilities: self.config().enforce_model_capabilities,
            model_health: Some(self.provider_svc.health()),
            run_events: Some(self.research.run_events.clone()),
        };
        let runner: Arc<dyn research::AgentRunner> =
            Arc::new(AgentCoreResearchRunner::new(self.clone()));
        ResearchCoordinator::new(self.research.store.clone(), runner, coord_cfg)
    }

    /// Shared waterfall registry. The TG heartbeat task polls this
    /// every ~20 s to render the live-progress message; no other
    /// consumer is expected today but the method is exposed so
    /// future telemetry (metrics, CLI `/research tail`) can tap the
    /// same source of truth.
    pub fn research_run_events(&self) -> research::RunEventRegistry {
        self.research.run_events().clone()
    }

    /// Snapshot the latest `limit` events for `run_id`. Empty when
    /// the run has completed and the registry has been cleaned up,
    /// or when the run_id is unknown.
    pub async fn research_run_events_snapshot(
        &self,
        run_id: &str,
        limit: usize,
    ) -> Vec<research::RunEvent> {
        self.research.run_events.snapshot(run_id, limit).await
    }

    /// Stop a live research run cooperatively. Looks up the
    /// cancellation token installed by `run_research*` and signals
    /// it; the coordinator's `select!` arm picks the signal up at
    /// the next `await` boundary and returns
    /// [`research::StopReason::Cancelled`]. Returns `false` when the
    /// `run_id` is unknown (already finished, or TG callback fired
    /// after cleanup), which the caller typically surfaces as a
    /// benign "run already done".
    pub async fn cancel_research_run(&self, run_id: &str) -> bool {
        if let Some(token) = self.research.cancels.read().await.get(run_id).cloned() {
            token.cancel();
            true
        } else {
            false
        }
    }

    // ── Research accessors (moved from lib.rs) ─────────────────────────

    pub fn research_run_permits(&self) -> Arc<tokio::sync::Semaphore> {
        self.research.run_semaphore.clone()
    }

    pub fn set_scheduler_hook(&self, hook: Arc<dyn research::SchedulerHook>) {
        *write_or_recover(&self.research.scheduler_hook) = hook;
    }

    fn scheduler_hook(&self) -> Arc<dyn research::SchedulerHook> {
        read_or_recover(&self.research.scheduler_hook).clone()
    }

    pub async fn scheduler_failure_snapshot(&self, spec_id: &str) -> Option<(u32, bool)> {
        self.scheduler_hook().failure_snapshot(spec_id).await
    }

    pub async fn reset_research_failures(&self, id: &str) -> Result<()> {
        self.scheduler_hook().reset_failures(id).await;
        let mut spec = self.research.store.load_spec(id).await?;
        let needs_save = spec.paused || spec.pause_reason.is_some();
        spec.paused = false;
        spec.pause_reason = None;
        if needs_save {
            self.research.store.save_spec(&spec).await?;
        }
        self.scheduler_hook()
            .notify(research::SchedulerEvent::SpecUpdated {
                spec_id: id.to_string(),
            })
            .await;
        Ok(())
    }

    pub fn research_store(&self) -> Arc<dyn ResearchStore> {
        self.research.store.clone()
    }
}

/// Build a numbered corpus string from findings for the ask-research prompt.
fn build_ask_corpus(findings: &[crate::research::Finding], excerpt_budget: usize) -> String {
    let mut corpus = String::new();
    for (i, f) in findings.iter().rev().enumerate() {
        let title = f.title.as_deref().unwrap_or("(untitled)");
        let price = f.price.as_deref().unwrap_or("?");
        let date = f.listing_date.as_deref().unwrap_or("?");
        let excerpt = f
            .excerpt
            .as_deref()
            .map(|e| {
                if e.chars().count() > excerpt_budget {
                    format!("{}…", e.chars().take(excerpt_budget).collect::<String>())
                } else {
                    e.to_string()
                }
            })
            .unwrap_or_default();
        corpus.push_str(&format!(
            "[{idx}] {title} — {price} ({date})\n  url: {url}\n  excerpt: {excerpt}\n\n",
            idx = i + 1,
            url = f.url,
        ));
    }
    corpus
}

// ── ResearchRunner impl for AgentCore ─────────────────────────────────

#[async_trait::async_trait]
impl research::ResearchRunner for AgentCore {
    async fn load_research(&self, id: &str) -> Result<research::ResearchSpec> {
        self.load_research(id).await
    }

    async fn set_research_paused(&self, id: &str, paused: bool) -> Result<()> {
        self.set_research_paused(id, paused).await
    }

    async fn update_research(
        &self,
        id: &str,
        patch: research::patch::ResearchPatch,
    ) -> Result<research::ResearchSpec> {
        self.update_research(id, patch).await
    }

    async fn run_research(&self, id: &str) -> Result<research::RunReport> {
        let arc = self
            .self_ref
            .read()
            .expect("self_ref lock")
            .as_ref()
            .and_then(|w| w.upgrade())
            .expect("AgentCore self_ref not initialized");
        AgentCore::run_research(arc, id).await
    }

    async fn run_research_verified(
        &self,
        id: &str,
        max_rounds: u32,
    ) -> Result<research::VerifiedRunReport> {
        // Clone the Arc<Self> from self_ref to satisfy the coordinator's Arc receiver
        let arc = self
            .self_ref
            .read()
            .expect("self_ref lock")
            .as_ref()
            .and_then(|w| w.upgrade())
            .expect("AgentCore self_ref not initialized");
        arc.run_research_verified(id, max_rounds).await
    }

    fn research_verify_config(&self) -> (bool, u32) {
        let cfg = &self.config().research;
        (cfg.verify_by_default, cfg.gatekeeper.max_rounds)
    }
}

#[cfg(test)]
mod boundary_tests {
    #[test]
    fn research_ops_no_session_access() {
        let src = include_str!("research_ops.rs");
        for pattern in [
            "self.ss.",
            ".sessions.write",
            "session_sender",
            "session_mcp",
        ] {
            let hits: Vec<_> = src
                .lines()
                .enumerate()
                .filter(|(_, l)| !l.trim_start().starts_with("//"))
                .filter(|(_, l)| !l.contains("pattern"))
                .filter(|(_, l)| !l.contains("cfg(test)"))
                .filter(|(_, l)| !l.contains('"')) // skip string literals in test code
                .filter(|(_, l)| l.contains(pattern))
                .collect();
            assert!(
                hits.is_empty(),
                "research_ops.rs violates boundary: found '{pattern}' at lines {:?}",
                hits.iter().map(|(n, _)| n + 1).collect::<Vec<_>>()
            );
        }
    }
}
