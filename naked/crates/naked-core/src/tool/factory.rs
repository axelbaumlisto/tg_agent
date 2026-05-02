//! Tool factories — each function builds one category of tools.
//!
//! Replaces the monolithic `build_tool_registry_for` with composable builders.
//! Each function takes ONLY what it needs — no `&self` on AgentCore.

use std::path::Path;
use std::sync::Arc;

use tokio::sync::RwLock;

use crate::agent_registry::AgentRegistry;
use crate::config::Config;
use crate::mcp::client::{McpRegistry, McpServer};
use crate::mcp::wrapper::McpToolWrapper;
use crate::provider::Provider;
use crate::ResearchState;
use crate::skill::resolver::SkillResolver;
use crate::skill::tool::SkillTool;
use crate::tool::bash::BashTool;
use crate::tool::file_ops::{EditFileTool, ReadFileTool, WriteFileTool};
use crate::tool::memory::MemoryTool;
use crate::tool::remote::RemoteContext;
use crate::tool::search::{GlobSearchTool, GrepSearchTool};
use crate::tool::sub_agent::SubAgentTool;
use crate::tool::web_fetch::WebFetchTool;
use crate::tool::web_fetch_tls::WebFetchTlsTool;
use crate::tool::web_fetch_wayback::WebFetchWaybackTool;
use crate::tool::web_search::WebSearchTool;
use crate::tool::agent_control::{AgentStatusTool, AgentStopTool};
use crate::tool::Tool;
use crate::AgentCore;


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
}

/// Build core tools: bash, file ops, search, web, memory, sub-agent.
pub(crate) async fn core_tools(ctx: &CoreToolCtx<'_>) -> Vec<Box<dyn Tool>> {
    let sub_agent = SubAgentTool::new(
        ctx.provider.clone(),
        ctx.model.to_string(),
        ctx.config.tool_timeout_secs,
        ctx.config.exa_api_keys.clone(),
    )
    .with_registry(ctx.agent_registry.clone());

    let bash_tool: Box<dyn Tool> = if ctx.remote_ctx.is_remote().await {
        let ops = ctx.remote_ctx.ops().await;
        Box::new(BashTool::with_ops(ctx.config.tool_timeout_secs, ops))
    } else {
        Box::new(BashTool::new(ctx.config.tool_timeout_secs))
    };

    let memory_ctx = crate::tool::memory::MemoryContext::new();
    memory_ctx.set_user_id(ctx.sender_id.clone());

    vec![
        bash_tool,
        Box::new(ReadFileTool),
        Box::new(WriteFileTool),
        Box::new(EditFileTool),
        Box::new(GlobSearchTool),
        Box::new(GrepSearchTool),
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
        Box::new(MemoryTool::with_context(ctx.workspace.to_path_buf(), memory_ctx)),
    ]
}

/// Build research tools (if enabled).
pub(crate) fn research_tools(
    config: &Config,
    research: &ResearchState,
    self_ref: &std::sync::RwLock<Option<std::sync::Weak<AgentCore>>>,
) -> Vec<Box<dyn Tool>> {
    use crate::research::tool::*;
    use crate::research::ops_tool::*;

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

    if let Some(weak) = self_ref.read().unwrap().clone() {
        tools.push(Box::new(ResearchLaunchTool::new(weak.clone())));
        tools.push(Box::new(ResearchUpdateSpecTool::new(weak.clone())));
        tools.push(Box::new(ResearchSetScheduleTool::new(weak.clone())));
        tools.push(Box::new(ResearchPauseTool::new(weak.clone())));
        tools.push(Box::new(ResearchResumeTool::new(weak)));
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
    factories
        .read()
        .await
        .iter()
        .map(|f| f())
        .collect()
}
