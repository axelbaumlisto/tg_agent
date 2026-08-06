//! Tool factories — each function builds one category of tools.
//!
//! Replaces the monolithic `build_tool_registry_for` with composable builders.
//! Each function takes ONLY what it needs — no `&self` on AgentCore.

use std::path::Path;
use std::sync::Arc;

use tokio::sync::RwLock;

use crate::AgentCore;
use crate::ResearchState;
use crate::agent_registry::AgentRegistry;
use crate::config::Config;
use crate::mcp::client::{McpRegistry, McpServer};
use crate::mcp::wrapper::McpToolWrapper;
use crate::provider::Provider;
use crate::skill::resolver::SkillResolver;
use crate::skill::tool::SkillTool;
use crate::tool::Tool;
use crate::tool::agent_control::{AgentStatusTool, AgentStopTool};
use crate::tool::bash::BashTool;
use crate::tool::fff_registry::FffRegistryConfig;
use crate::tool::fff_tools::FffState;
use crate::tool::file_ops::{EditFileTool, FileSnapshotTool, ReadFileTool, WriteFileTool};
use crate::tool::memory::MemoryTool;
use crate::tool::remote::RemoteContext;
// Old search tools replaced by fff (SIMD + frecency):
// use crate::tool::search::{GlobSearchTool, GrepSearchTool};
use crate::tool::sub_agent::SubAgentTool;
use crate::tool::web_fetch::WebFetchTool;
use crate::tool::web_fetch_tls::WebFetchTlsTool;
use crate::tool::web_fetch_wayback::WebFetchWaybackTool;
use crate::tool::web_search::WebSearchTool;

/// Context for building core tools (reduces argument count).
pub(crate) struct CoreToolCtx<'a> {
    pub config: &'a Config,
    pub remote_ctx: &'a RemoteContext,
    pub agent_registry: &'a AgentRegistry,
    pub search: &'a crate::SearchState,
    pub provider: &'a Arc<dyn Provider>,
    pub model: &'a str,
    pub workspace: &'a Path,
    pub sender_id: Option<String>,
    pub todo_list: &'a crate::tool::todo_tool::TodoList,
    pub plan_state: &'a crate::tool::plan_tool::PlanState,
    pub fff_registry: &'a Arc<crate::tool::fff_registry::FffPickerRegistry>,
    pub fs_cache: &'a Arc<crate::tool::fs_cache::FsCache>,
    pub persistent_bash: &'a Arc<crate::tool::persistent_bash::PersistentBashManager>,
    pub session_id: &'a str,
}

/// Build core tools: bash, file ops, search, web, memory, sub-agent.
pub(crate) async fn core_tools(ctx: &CoreToolCtx<'_>) -> Vec<Box<dyn Tool>> {
    let sub_agent = SubAgentTool::new(
        ctx.provider.clone(),
        ctx.model.to_string(),
        ctx.config.tool_timeout_secs,
        ctx.config.exa_api_keys.clone(),
    )
    .with_stale_edit_guard(ctx.config.stale_edit_guard_enabled)
    .with_hashline_edit(ctx.config.hashline_edit_enabled)
    .with_registry(ctx.agent_registry.clone());

    let bash_tool: Box<dyn Tool> = if ctx.remote_ctx.is_remote().await {
        let ops = ctx.remote_ctx.ops().await;
        Box::new(BashTool::with_ops(ctx.config.tool_timeout_secs, ops))
    } else if ctx.config.persistent_bash_enabled {
        Box::new(
            BashTool::new(ctx.config.tool_timeout_secs)
                .with_persistent(ctx.persistent_bash.clone(), ctx.session_id.to_string()),
        )
    } else {
        Box::new(BashTool::new(ctx.config.tool_timeout_secs))
    };

    let memory_ctx = crate::tool::memory::MemoryContext::new();
    memory_ctx.set_user_id(ctx.sender_id.clone());

    let fs_cache = if ctx.config.fs_cache_enabled {
        ctx.fs_cache.set_max_bytes(ctx.config.fs_cache_max_bytes);
        Some(ctx.fs_cache.clone())
    } else {
        None
    };

    // fff-powered search engine. Fast-index mode reuses one process-wide
    // picker per canonical workspace; flag-off falls back to legacy per-turn state.
    let fff_cfg = FffRegistryConfig::from_config(ctx.config);
    let fff_state = if fff_cfg.enabled {
        FffState::with_registry(ctx.fff_registry.clone(), ctx.workspace, fff_cfg)
    } else {
        FffState::new(ctx.workspace)
    };

    let mut tools: Vec<Box<dyn Tool>> = vec![
        bash_tool,
        Box::new(ReadFileTool::new(fs_cache.clone())),
        Box::new(FileSnapshotTool::new(fs_cache.clone())),
        Box::new(WriteFileTool::new(fs_cache.clone())),
        Box::new(
            EditFileTool::new(ctx.config.stale_edit_guard_enabled)
                .with_hashline_edit(ctx.config.hashline_edit_enabled)
                .with_fs_cache(fs_cache.clone()),
        ),
        // fff-powered search (SIMD + frecency + git-aware):
        Box::new(crate::tool::fff_tools::FffFindTool::new(&fff_state)),
        Box::new(crate::tool::fff_tools::FffGrepTool::new(&fff_state)),
        Box::new(sub_agent),
        Box::new(AgentStatusTool::new(ctx.agent_registry.clone())),
        Box::new(AgentStopTool::new(ctx.agent_registry.clone())),
        Box::new(WebSearchTool::new(
            ctx.search.exa_key_pool.clone(),
            ctx.search.tavily_key_pool.clone(),
            ctx.search.serpapi_key_pool.clone(),
        )),
        Box::new(WebFetchTool::with_components(
            ctx.search.cloud_scraper.clone(),
            ctx.search.host_policy.clone(),
        )),
        Box::new(WebFetchTlsTool::new()),
        Box::new(WebFetchWaybackTool::new()),
        Box::new(MemoryTool::with_context(
            ctx.workspace.to_path_buf(),
            memory_ctx,
        )),
        Box::new(super::test_runner::RunTestsTool),
        Box::new(super::git_tools::GitLogTool),
        Box::new(super::git_tools::GitDiffTool),
        Box::new(super::apply_patch::ApplyPatchTool::new(fs_cache)),
        Box::new(super::diagnostics::DiagnosticsTool),
        Box::new(super::validate_data::ValidateDataTool),
        Box::new(super::todo_tool::TodoTool::new(ctx.todo_list.clone())),
        Box::new(super::plan_tool::PlanTool::new(ctx.plan_state.clone())),
        Box::new(super::review_tool::ReviewTool),
        Box::new(super::recall_archive::RecallArchiveTool::new(
            ctx.config.session_dir.clone(),
        )),
    ];

    // PLAN_SNAPSHOTS_DISABLE_FLAG_v1: only register revert_turn when snapshots
    // are enabled, so disabled sessions never advertise a Dangerous tool
    // (no approval prompt, no open_or_init / side-repo creation).
    // Mirrors research_tools gating.
    if ctx.config.snapshots_enabled {
        tools.push(Box::new(super::revert_turn::RevertTurnTool));
    }

    tools
}

/// Build research tools (if enabled).
pub(crate) fn research_tools(
    config: &Config,
    research: &ResearchState,
    self_ref: &std::sync::RwLock<Option<std::sync::Weak<AgentCore>>>,
) -> Vec<Box<dyn Tool>> {
    use crate::research::ops_tool::*;
    use crate::research::ops_tool_extra::*;
    use crate::research::tool::*;

    if !config.research.enabled {
        return Vec::new();
    }

    let mut tools: Vec<Box<dyn Tool>> = vec![
        Box::new(ResearchSaveTool::new(
            research.store.clone(),
            research.context.clone(),
            config.research.gatekeeper.clone(),
        )),
        Box::new(ResearchListTool::new(
            research.store.clone(),
            research.context.clone(),
        )),
        Box::new(ResearchSaveCursorTool::new(
            research.store.clone(),
            research.context.clone(),
        )),
        Box::new(ResearchStatusTool::new(research.store.clone())),
        Box::new(ResearchCreateTool::new(
            research.store.clone(),
            config.research.clone(),
        )),
        Box::new(ResearchListSpecsTool::new(
            research.store.clone(),
            config.research.clone(),
        )),
        Box::new(ResearchMetricsTool::new(research.store.clone())),
        Box::new(ResearchHelpTool::new(config.research.clone())),
        Box::new(ResearchFindingsTool::new(research.store.clone())),
        Box::new(ResearchSetTargetTool::new(
            research.store.clone(),
            research.context.clone(),
        )),
    ];

    if let Some(weak) = self_ref.read().expect("self_ref lock").clone() {
        // T2.5 (PLAN_RESEARCH_AGENT_FLOW_v1): research_run tool lets the LLM
        // trigger a research spec run inline in the current agent turn, sharing
        // the parent cancel token. Registered here so it's visible in the agent's
        // tool list alongside research_save, research_launch, etc.
        tools.push(Box::new(crate::tool::research_run::ResearchRunTool::new(
            weak.clone(),
        )));

        // Upcast Weak<AgentCore> → Weak<dyn ResearchRunner> for ISP
        let runner_weak: std::sync::Weak<dyn crate::research::ResearchRunner> = weak;
        // research_launch removed (PLAN_UNIFIED_TURN_v1) — research_run
        // covers the same functionality with streaming + abort support.
        tools.push(Box::new(ResearchUpdateSpecTool::new(runner_weak.clone())));
        tools.push(Box::new(ResearchSetScheduleTool::new(runner_weak.clone())));
        tools.push(Box::new(ResearchPauseTool::new(runner_weak.clone())));
        tools.push(Box::new(ResearchResumeTool::new(runner_weak.clone())));
        tools.push(Box::new(ResearchDeleteTool::new(runner_weak)));
    }

    tools
}

/// Build skill tools from configured skill roots.
pub(crate) fn skill_tools(skill_roots: &[std::path::PathBuf]) -> Vec<Box<dyn Tool>> {
    tracing::debug!("skill_roots: {:?}", skill_roots);
    let resolver = SkillResolver::new(skill_roots.to_vec());
    let available = resolver.list();
    tracing::info!(
        "Skills: {} found in {} roots",
        available.len(),
        skill_roots.len()
    );
    for (name, hit) in &available {
        tracing::debug!("  skill: {name} -> {}", hit.path.display());
    }
    let orphans = resolver.find_orphans();
    if !orphans.is_empty() {
        tracing::warn!(
            "Skills: {} directory(ies) in skill_roots have NO SKILL.{{json,md,toml}} manifest — \
             invisible to the `Skill` tool. Add a manifest or remove the directory:",
            orphans.len()
        );
        for (root, path) in &orphans {
            tracing::warn!(
                "  orphan skill dir: {} (root: {})",
                path.display(),
                root.display()
            );
        }
    }
    vec![Box::new(SkillTool::new(resolver, &available))]
}

/// Build MCP tools from global + per-session servers.
pub(crate) async fn mcp_tools(
    mcp_registry: &RwLock<McpRegistry>,
    session_mcp: &[Arc<McpServer>],
) -> Vec<Box<dyn Tool>> {
    let mut tools: Vec<Box<dyn Tool>> = Vec::new();
    let mcp_reg = mcp_registry.read().await;
    for server in mcp_reg.servers() {
        tools.extend(McpToolWrapper::wrap_all(Arc::clone(server)));
    }
    for server in session_mcp {
        tools.extend(McpToolWrapper::wrap_all(Arc::clone(server)));
    }
    tools
}

/// Build extra tools injected by the embedding binary.
pub(crate) async fn extra_tools(
    factories: &RwLock<crate::ExtraToolFactories>,
) -> Vec<Box<dyn Tool>> {
    factories.read().await.iter().map(|f| f()).collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_support::TestCore;

    #[test]
    fn tool_factory_advertises_research_run() {
        let tc = TestCore::build();
        let core = &tc.core;
        let config = core.config();
        let tools = research_tools(&config, &core.research, &core.self_ref);
        let names: Vec<String> = tools.iter().map(|t| t.spec().name.clone()).collect();
        assert!(
            names.iter().any(|n| n == "research_run"),
            "research_run tool must appear in research_tools(); got: {names:?}"
        );
    }

    #[test]
    fn research_launch_removed() {
        // PLAN_UNIFIED_TURN_v1: research_launch was removed — research_run
        // is the only way to execute a research spec from the LLM.
        let tc = TestCore::build();
        let config = tc.core.config();
        let tools = research_tools(&config, &tc.core.research, &tc.core.self_ref);
        let names: Vec<String> = tools.iter().map(|t| t.spec().name.clone()).collect();
        assert!(
            names.iter().any(|n| n == "research_run"),
            "research_run must be in the list"
        );
        assert!(
            !names.iter().any(|n| n == "research_launch"),
            "research_launch must NOT be in the list (removed)"
        );
    }

    #[tokio::test]
    async fn s7_remember_tool_is_not_registered() {
        let tc = TestCore::build();
        let core = &tc.core;
        let workspace = tc.workspace();
        std::fs::create_dir_all(&workspace).expect("test workspace dir");
        let session_id = core.create_session(&workspace).await;
        let config = core.config();
        let effective = config.default_effective();
        let provider = core.provider_for(&config.default_provider).await;
        let names = core
            .build_tool_registry_for(
                &session_id,
                &effective,
                &provider,
                &effective.model,
                &workspace,
            )
            .await
            .tool_names();

        assert!(
            names.iter().any(|name| name == "memory"),
            "memory remains the single persistent-memory writer; got: {names:?}"
        );
        assert!(
            !names.iter().any(|name| name == "remember"),
            "remember must not be registered because it bypassed MarkdownMemoryStore; got: {names:?}"
        );
    }

    #[tokio::test]
    async fn revert_turn_registered_only_when_snapshots_enabled() {
        async fn tool_names_for(snapshots_enabled: bool) -> Vec<String> {
            let tc = TestCore::build();
            let core = &tc.core;
            let mut config = core.config().as_ref().clone();
            config.snapshots_enabled = snapshots_enabled;
            core.reload_config(config);

            let workspace = tc.workspace();
            std::fs::create_dir_all(&workspace).expect("test workspace dir");
            let session_id = core.create_session(&workspace).await;
            let config = core.config();
            let effective = config.default_effective();
            let provider = core.provider_for(&config.default_provider).await;
            core.build_tool_registry_for(
                &session_id,
                &effective,
                &provider,
                &effective.model,
                &workspace,
            )
            .await
            .tool_names()
        }

        let disabled = tool_names_for(false).await;
        assert!(
            !disabled.iter().any(|name| name == "revert_turn"),
            "revert_turn must not be registered when snapshots are disabled; got: {disabled:?}"
        );

        let enabled = tool_names_for(true).await;
        assert!(
            enabled.iter().any(|name| name == "revert_turn"),
            "revert_turn must be registered when snapshots are enabled; got: {enabled:?}"
        );
    }
}
