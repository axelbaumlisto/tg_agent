use std::collections::HashMap;
use std::sync::Arc;

use async_trait::async_trait;
use tokio::sync::RwLock;

use crate::AgentCore;
use crate::error::Result;
use crate::research::parse_provider_model_pair;
use crate::research::{self, CoordinatorConfig, ResearchSpec};
use crate::types;

/// Adapter so the research coordinator can drive `AgentCore` without AgentCore
/// having a direct dep on the coordinator's `AgentRunner` trait bounds (keeps
/// the coordinator unit-testable with stubs).
///
/// `run_session_map` keeps a `run_id → session_id` table so
/// [`Self::cleanup_research_session`] can abort the underlying
/// session's [`AgentLoop`] task. Without this, cancelling a research
/// run would only stop the coordinator's `drain_events` loop while the
/// background `tokio::spawn` keeps running tools (the cancel-safety
/// bug observed in production: worker continued executing for minutes
/// after a `Stop` button press).
pub(crate) struct AgentCoreResearchRunner {
    core: Arc<AgentCore>,
    run_session_map: Arc<RwLock<HashMap<String, String>>>,
}

impl AgentCoreResearchRunner {
    pub(crate) fn new(core: Arc<AgentCore>) -> Self {
        Self {
            core,
            run_session_map: Arc::new(RwLock::new(HashMap::new())),
        }
    }
}

#[async_trait]
impl research::AgentRunner for AgentCoreResearchRunner {
    async fn start_research_turn(
        &self,
        spec: &ResearchSpec,
        prompt: &str,
        config: &CoordinatorConfig,
        run_id: &str,
    ) -> Result<(types::AgentHandle, String, String)> {
        // 1. Provision an ephemeral session on the "research" channel. This
        //    lives on disk under `session_dir/<id>` — intentional, because it
        //    makes `/sessions` show a breadcrumb of every research run for
        //    later inspection.
        let workspace = config.workspace.clone();
        let session_id = self
            .core
            .create_session_with_channel(&workspace, "research")
            .await;

        // 2. Apply per-session provider/model override so this turn runs on
        //    the research model (default: kimi-for-coding via kimi-code).
        //    The precedence is spec > config.research > global default.
        //
        //    `config.default_model` may carry a `provider/model` pair (the
        //    fallback chain uses this to switch to qwen when kimi fails — see
        //    `try_start_with_fallback`). When it does, the embedded provider
        //    overrides everything else for this turn.
        let (parsed_provider, parsed_model) = config
            .default_model
            .as_deref()
            .and_then(parse_provider_model_pair)
            .map(|(p, m)| (Some(p), Some(m)))
            .unwrap_or_else(|| {
                (
                    config.default_provider.clone(),
                    config.default_model.clone(),
                )
            });
        let provider = spec.provider.clone().or(parsed_provider);
        let model = spec.model.clone().or(parsed_model);
        if (provider.is_some() || model.is_some())
            && let Err(e) = self
                .core
                .set_session_provider(&session_id, provider.as_deref(), model.as_deref())
                .await
        {
            tracing::warn!("research override failed: {e}");
        }

        // 2b. Apply reasoning level (e.g. "medium" for kimi-for-coding) so
        //     it surfaces as `reasoning_effort` in the OAI-compat request.
        if let Some(level) = config.reasoning.as_deref()
            && let Err(e) = self.core.set_session_reasoning(&session_id, level).await
        {
            tracing::warn!("research reasoning override failed: {e}");
        }

        // 3. Auto-approve tool permissions for the research session.
        //    Research runs are headless — nobody is watching to click "allow".
        let now_ts = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_secs() as i64)
            .unwrap_or(0);
        if let Err(e) = self.core.set_session_yolo(&session_id, Some(now_ts)).await {
            tracing::warn!("failed to enable yolo for research session: {e}");
        }

        // 4. Install the ambient research context so the `research_save`
        //    tools know where to write and which run_id to tag findings with.
        //    Reset the per-run save counter so the new clear-target guard
        //    (`research_set_target`) starts at 0 for this run rather than
        //    inheriting a count from whatever happened previously.
        self.core.research.context.set_id(Some(spec.id.clone()));
        self.core
            .research
            .context
            .set_run_id(Some(run_id.to_string()));
        self.core.research.context.reset_saves();

        let handle = self.core.send_prompt(&session_id, prompt).await?;

        // Remember which session backs this run so cleanup (incl. the cancel
        // path) can abort the spawned AgentLoop task. Without this, a
        // cancelled research run keeps burning model tokens / proxy
        // bandwidth in the background.
        self.run_session_map
            .write()
            .await
            .insert(run_id.to_string(), session_id.clone());

        // Resolve what we actually settled on after overrides were applied, so
        // the run record is truthful.
        let effective_provider = provider
            .clone()
            .unwrap_or_else(|| self.core.config().default_provider.clone());
        let effective_model = model
            .clone()
            .unwrap_or_else(|| self.core.config().default_model.clone());

        Ok((handle, effective_provider, effective_model))
    }

    async fn cleanup_research_session(&self, run_id: &str) {
        // Abort the AgentLoop task that backs this run. Idempotent — if the
        // task already finished naturally, the cancel call is a no-op and
        // the session state is already `Idle`. The mapping is removed
        // unconditionally so we don't accumulate stale entries.
        let session_id = self.run_session_map.write().await.remove(run_id);
        if let Some(sid) = session_id {
            tracing::info!(
                run_id, session = %sid,
                "research cleanup: aborting underlying agent session",
            );
            self.core.abort(&sid).await;
        }

        self.core.research.context.set_id(None);
        self.core.research.context.set_run_id(None);
        self.core.research.context.reset_saves();
    }
}
