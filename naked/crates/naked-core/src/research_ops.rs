//! Research operations on AgentCore.
//!
//! Two buckets:
//! 1. **Pure store/registry methods** (15 of them) — delegate to
//!    [`crate::services::research::ResearchService`]. Kept on `AgentCore`
//!    only for back-compat: scout-survey shows ~120 external call sites
//!    in `naked-tg`, `naked-cli`, and integration tests.
//! 2. **Coordinator-bound methods** (`run_research*`, `ask_research`,
//!    `build_coordinator`) — stay here. They need `Arc<Self>` for the
//!    runner adapter, plus a `Provider` from the provider service.
//!
//! Plan: naked/docs/PLAN_CORE_HARDENING_v2.md §4 T1.

use std::sync::Arc;

use tokio_util::sync::CancellationToken;

use crate::error::{AgentError, Result};
use crate::provider;
use crate::research::coordinator::CoordinatorConfig;
use crate::research::{
    self, ResearchCoordinator, ResearchPatch, ResearchSpec, ResearchStore, RunReport,
    VerifiedRunReport,
};
use crate::services::research::{CreateSchedule, ScheduleUpdate};
use crate::types;
use crate::{AgentCore, AgentCoreResearchRunner, acquire_research_permit};

// RAII guard that owns the lifecycle of a research run's
// `(cancel_token, events)` registration. Inserts the entry on
// `install`, removes it on drop — so every exit path of
// `run_research_*` (Ok, Err, panic) cleans up without boilerplate.
struct ResearchCancelGuard {
    cancels: Arc<tokio::sync::RwLock<std::collections::HashMap<String, CancellationToken>>>,
    events: research::RunEventRegistry,
    spec_id: String,
}

impl ResearchCancelGuard {
    async fn install(
        cancels: Arc<tokio::sync::RwLock<std::collections::HashMap<String, CancellationToken>>>,
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
    // ── Bucket A: 1-line delegates to ResearchService ─────────────────

    pub async fn create_research(
        &self,
        topic: &str,
        sources: Vec<String>,
        session_id: Option<String>,
        chat_id: Option<i64>,
        thread_id: Option<i32>,
        schedule: CreateSchedule,
    ) -> Result<ResearchSpec> {
        self.research_svc
            .create_research(topic, sources, session_id, chat_id, thread_id, schedule)
            .await
    }

    pub async fn list_research(&self) -> Result<Vec<ResearchSpec>> {
        self.research_svc.list_research().await
    }

    pub async fn load_research(&self, id: &str) -> Result<ResearchSpec> {
        self.research_svc.load_research(id).await
    }

    pub async fn delete_research(&self, id: &str) -> Result<()> {
        self.research_svc.delete_research(id).await
    }

    pub async fn set_research_paused(&self, id: &str, paused: bool) -> Result<()> {
        self.research_svc.set_research_paused(id, paused).await
    }

    pub async fn set_research_paused_with_reason(
        &self,
        id: &str,
        paused: bool,
        reason: Option<String>,
    ) -> Result<()> {
        self.research_svc
            .set_research_paused_with_reason(id, paused, reason)
            .await
    }

    pub async fn update_research(&self, id: &str, patch: ResearchPatch) -> Result<ResearchSpec> {
        self.research_svc.update_research(id, patch).await
    }

    pub async fn set_research_schedule(&self, id: &str, update: ScheduleUpdate) -> Result<()> {
        self.research_svc.set_research_schedule(id, update).await
    }

    pub fn research_run_events(&self) -> research::RunEventRegistry {
        self.research_svc.research_run_events()
    }

    pub async fn research_run_events_snapshot(
        &self,
        run_id: &str,
        limit: usize,
    ) -> Vec<research::RunEvent> {
        self.research_svc
            .research_run_events_snapshot(run_id, limit)
            .await
    }

    // B4 (PLAN_RESEARCH_FLOW_CLOSURE_v1): cancel_research_run removed.
    // All callers now use AgentCore::abort(session_id) (INV-CANCEL-1).

    pub fn research_run_permits(&self) -> Arc<tokio::sync::Semaphore> {
        self.research_svc.research_run_permits()
    }

    pub fn set_scheduler_hook(&self, hook: Arc<dyn research::SchedulerHook>) {
        self.research_svc.set_scheduler_hook(hook);
    }

    pub async fn scheduler_failure_snapshot(&self, spec_id: &str) -> Option<(u32, bool)> {
        self.research_svc.scheduler_failure_snapshot(spec_id).await
    }

    pub async fn reset_research_failures(&self, id: &str) -> Result<()> {
        self.research_svc.reset_research_failures(id).await
    }

    pub fn research_store(&self) -> Arc<dyn ResearchStore> {
        self.research_svc.research_store()
    }

    pub async fn write_research_memory_link(
        &self,
        spec_id: &str,
        run_id: &str,
        verified: Option<&VerifiedRunReport>,
    ) -> Result<()> {
        self.research_svc
            .write_research_memory_link(spec_id, run_id, verified)
            .await
    }

    // ── Bucket B: ask_research (uses provider service for one-shot) ───

    /// Answer a free-form question against a research's accumulated findings.
    ///
    /// Builds a one-shot prompt containing the topic + the latest N findings
    /// (title, price, date, url, excerpt) and calls the configured research
    /// provider/model with no tools. The LLM is asked to answer ONLY from the
    /// supplied corpus and to cite URLs by `[1]`-style numeric indices.
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

    // ── Bucket C: coordinator-bound runs (need Arc<Self>) ─────────────

    /// Execute a single research pass synchronously. Returns the run summary.
    /// Telegram/CLI callers typically `tokio::spawn` this — a live run can
    /// take up to `max_wall_seconds` (20 min default).
    pub async fn run_research(self: Arc<Self>, id: &str) -> Result<RunReport> {
        self.run_research_with_cancel(id, CancellationToken::new())
            .await
    }

    pub async fn run_research_with_cancel(
        self: Arc<Self>,
        id: &str,
        cancel: CancellationToken,
    ) -> Result<RunReport> {
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
        let report = coord.run_once_with_cancel(id, cancel).await?;
        if let Err(e) = self
            .write_research_memory_link(id, &report.run_id, None)
            .await
        {
            tracing::warn!(spec = %id, "failed to record research memory link: {e}");
        }
        Ok(report)
    }

    pub async fn run_research_verified(
        self: Arc<Self>,
        id: &str,
        max_rounds: u32,
    ) -> Result<research::VerifiedRunReport> {
        self.run_research_verified_with_cancel(id, max_rounds, CancellationToken::new())
            .await
    }

    pub async fn run_research_verified_with_cancel(
        self: Arc<Self>,
        id: &str,
        max_rounds: u32,
        cancel: CancellationToken,
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

    async fn set_research_schedule(&self, id: &str, update: ScheduleUpdate) -> Result<()> {
        self.set_research_schedule(id, update).await
    }

    async fn update_research(
        &self,
        id: &str,
        patch: research::patch::ResearchPatch,
    ) -> Result<research::ResearchSpec> {
        self.update_research(id, patch).await
    }

    async fn run_research(&self, id: &str) -> Result<research::RunReport> {
        let arc = crate::read_or_recover(&self.self_ref)
            .as_ref()
            .and_then(|w| w.upgrade())
            .expect("AgentCore self_ref not initialised before run_research");
        AgentCore::run_research(arc, id).await
    }

    async fn run_research_verified(
        &self,
        id: &str,
        max_rounds: u32,
    ) -> Result<research::VerifiedRunReport> {
        let arc = crate::read_or_recover(&self.self_ref)
            .as_ref()
            .and_then(|w| w.upgrade())
            .expect("AgentCore self_ref not initialised before run_research_verified");
        arc.run_research_verified(id, max_rounds).await
    }

    fn research_verify_config(&self) -> (bool, u32) {
        let cfg = &self.config().research;
        (cfg.verify_by_default, cfg.gatekeeper.max_rounds)
    }

    async fn delete_research(&self, id: &str) -> Result<()> {
        self.delete_research(id).await
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
