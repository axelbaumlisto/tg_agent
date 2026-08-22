use std::path::Path;
use std::sync::Arc;

use async_trait::async_trait;
use tokio::sync::mpsc;
use tokio_util::sync::CancellationToken;

use crate::agent_registry::{AgentRegistry, AgentStatus};
use crate::history::ConversationHistory;
use crate::loop_::{AgentLoop, LoopConfig};
use crate::provider::Provider;
use crate::tool::bash::BashTool;
use crate::tool::file_ops::{EditFileTool, FileSnapshotTool, ReadFileTool, WriteFileTool};
use crate::tool::registry::ToolRegistry;
use crate::tool::search::{GlobSearchTool, GrepSearchTool};
use crate::tool::web_search::WebSearchTool;
use crate::types::{
    AgentEvent, ContentBlock, Permission, Role, SubAgentEvent, ToolResult, ToolSpec,
};

use super::Tool;

const SUB_AGENT_MAX_ITERATIONS: usize = 30;

pub struct SubAgentTool {
    provider: Arc<dyn Provider>,
    model: String,
    tool_timeout_secs: u64,
    exa_keys: Vec<String>,
    registry: AgentRegistry,
    stale_edit_guard_enabled: bool,
    hashline_edit_enabled: bool,
}

impl SubAgentTool {
    pub fn new(
        provider: Arc<dyn Provider>,
        model: String,
        tool_timeout_secs: u64,
        exa_keys: Vec<String>,
    ) -> Self {
        Self {
            provider,
            model,
            tool_timeout_secs,
            exa_keys,
            registry: AgentRegistry::new(),
            stale_edit_guard_enabled: false,
            hashline_edit_enabled: false,
        }
    }

    pub fn with_stale_edit_guard(mut self, enabled: bool) -> Self {
        self.stale_edit_guard_enabled = enabled;
        self
    }

    pub fn with_hashline_edit(mut self, enabled: bool) -> Self {
        self.hashline_edit_enabled = enabled;
        self
    }

    pub fn with_registry(mut self, registry: AgentRegistry) -> Self {
        self.registry = registry;
        self
    }

    pub fn registry(&self) -> &AgentRegistry {
        &self.registry
    }

    fn build_tools(&self, mode: &str) -> ToolRegistry {
        let tools: Vec<Box<dyn Tool>> =
            match crate::agent_role::canonicalize_role(mode) {
                Some(crate::agent_role::CanonicalRole::General)
                | Some(crate::agent_role::CanonicalRole::Implementer)
                | Some(crate::agent_role::CanonicalRole::Custom) => vec![
                    Box::new(BashTool::new(self.tool_timeout_secs)),
                    Box::new(ReadFileTool::default()),
                    Box::new(FileSnapshotTool::default()),
                    Box::new(WriteFileTool::default()),
                    Box::new(
                        EditFileTool::new(self.stale_edit_guard_enabled)
                            .with_hashline_edit(self.hashline_edit_enabled),
                    ),
                    Box::new(GlobSearchTool),
                    Box::new(GrepSearchTool),
                    Box::new(WebSearchTool::from_legacy_exa(self.exa_keys.clone())),
                ],
                // Read-only postures (Explore, Plan, Review, Verifier) and
                // unknown/None roles all fail closed to the read-only set.
                // BashTool is intentionally excluded: a shell can write files,
                // run editors, and mutate the workspace regardless of which
                // file-op tools are present, so it is incompatible with the
                // ReadOnly permission contract.
                _ => vec![
                    Box::new(ReadFileTool::default()),
                    Box::new(FileSnapshotTool::default()),
                    Box::new(GlobSearchTool),
                    Box::new(GrepSearchTool),
                    Box::new(WebSearchTool::from_legacy_exa(self.exa_keys.clone())),
                ],
            };
        ToolRegistry::new(tools)
    }
}

#[async_trait]
impl Tool for SubAgentTool {
    fn spec(&self) -> ToolSpec {
        ToolSpec {
            name: "sub_agent".into(),
            description: "Delegate a task to a child agent that can use tools independently. \
                          Use for exploration, analysis, or isolated subtasks. \
                          The child agent runs with its own context and returns a text result. \
                          Mode picks the role posture: explore (read-only mapping), \
                          plan (strategy, no edits), review (read+grade), implementer \
                          (focused edit), verifier (test runner), general (default)."
                .into(),
            parameters: serde_json::json!({
                "type": "object",
                "properties": {
                    "prompt": {
                        "type": "string",
                        "description": "Task description for the sub-agent"
                    },
                    "mode": {
                        "type": "string",
                        "description": "Sub-agent role posture. Accepted: general/explore/plan/review/implementer/verifier/custom (case-insensitive aliases supported: worker, explorer, planning, code-review, builder, tester). Default: explore."
                    }
                },
                "required": ["prompt"]
            }),
            permission: Permission::ReadOnly,
        }
    }

    fn effective_permission(&self, input: &serde_json::Value, _cwd: &Path) -> Permission {
        let mode = input
            .get("mode")
            .and_then(|v| v.as_str())
            .unwrap_or("explore");
        // T10 of PLAN_QUALITY_v1: route through canonical role
        // taxonomy. Roles that write files get WorkspaceWrite;
        // read-only postures stay ReadOnly. Unknown role names
        // fall back to ReadOnly (safe default).
        match crate::agent_role::canonicalize_role(mode) {
            Some(crate::agent_role::CanonicalRole::General)
            | Some(crate::agent_role::CanonicalRole::Implementer)
            | Some(crate::agent_role::CanonicalRole::Custom) => Permission::WorkspaceWrite,
            _ => Permission::ReadOnly,
        }
    }

    async fn execute(&self, input: serde_json::Value, cwd: &Path) -> ToolResult {
        let (drop_tx, _) = mpsc::channel(1);
        self.execute_with_progress(input, cwd, drop_tx).await
    }

    async fn execute_with_progress(
        &self,
        input: serde_json::Value,
        cwd: &Path,
        progress: mpsc::Sender<AgentEvent>,
    ) -> ToolResult {
        let prompt = match input.get("prompt").and_then(|v| v.as_str()) {
            Some(p) if !p.trim().is_empty() => p.trim().to_string(),
            _ => {
                return ToolResult::err("Error: 'prompt' field is required and must be non-empty");
            }
        };

        let mode = input
            .get("mode")
            .and_then(|v| v.as_str())
            .unwrap_or("explore");

        let agent_id = format!(
            "sa-{}",
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_millis()
                % 100_000
        );

        let preview = if prompt.len() > 80 {
            format!("{}…", &prompt[..prompt.floor_char_boundary(80)])
        } else {
            prompt.clone()
        };

        let cancel = CancellationToken::new();
        self.registry
            .register(&agent_id, &preview, mode, cancel.clone())
            .await;

        let _ = progress
            .send(AgentEvent::SubAgentProgress {
                agent_id: agent_id.clone(),
                event: SubAgentEvent::Started {
                    prompt_preview: preview,
                },
            })
            .await;

        let tools = self.build_tools(mode);
        let config = LoopConfig {
            max_iterations: SUB_AGENT_MAX_ITERATIONS,
            max_wall: None,
            cwd: cwd.to_path_buf(),
            model: self.model.clone(),
            max_tokens: 8192,
            temperature: Some(0.0),
            ..Default::default()
        };

        let system = format!(
            "You are a focused sub-agent. Complete the task and report your findings concisely.\n\
             Working directory: {}\n\
             Mode: {mode}",
            cwd.display()
        );

        let mut history = ConversationHistory::new(system);
        history.push_user(&prompt);

        let provider_box: Box<dyn Provider> = Box::new(ArcProvider(self.provider.clone()));
        let agent = AgentLoop::new(provider_box, tools, config);

        let (tx, mut rx) = mpsc::channel(256);

        // Forward child events to parent in real-time (not after completion).
        let fwd_progress = progress.clone();
        let fwd_agent_id = agent_id.clone();
        let fwd_registry = self.registry.clone();
        let (text_tx, mut text_rx) = mpsc::channel::<String>(256);
        let fwd_handle = tokio::spawn(async move {
            while let Some(event) = rx.recv().await {
                match &event {
                    AgentEvent::ToolStart { name, input, .. } => {
                        fwd_registry.update_tool(&fwd_agent_id, name).await;
                        let inp_preview = serde_json::to_string(input).unwrap_or_default();
                        let inp_short = if inp_preview.len() > 120 {
                            format!("{}…", &inp_preview[..inp_preview.floor_char_boundary(120)])
                        } else {
                            inp_preview
                        };
                        let _ = fwd_progress
                            .send(AgentEvent::SubAgentProgress {
                                agent_id: fwd_agent_id.clone(),
                                event: SubAgentEvent::ToolUse {
                                    name: name.clone(),
                                    input_preview: inp_short,
                                },
                            })
                            .await;
                    }
                    AgentEvent::ToolEnd { name, state, .. } => {
                        let _ = fwd_progress
                            .send(AgentEvent::SubAgentProgress {
                                agent_id: fwd_agent_id.clone(),
                                event: SubAgentEvent::ToolDone {
                                    name: name.clone(),
                                    state: *state,
                                },
                            })
                            .await;
                    }
                    AgentEvent::TextDelta(t) => {
                        let _ = text_tx.send(t.clone()).await;
                        let _ = fwd_progress
                            .send(AgentEvent::SubAgentProgress {
                                agent_id: fwd_agent_id.clone(),
                                event: SubAgentEvent::TextDelta(t.clone()),
                            })
                            .await;
                    }
                    AgentEvent::ThinkingDelta(_) | AgentEvent::Heartbeat => {
                        let _ = fwd_progress.send(AgentEvent::Heartbeat).await;
                    }
                    _ => {}
                }
            }
        });

        let result = agent.run(&mut history, tx, cancel, None, None).await;

        // Wait for the forwarder to drain remaining events.
        drop(fwd_handle);
        let mut text_parts = Vec::new();
        while let Ok(t) = text_rx.try_recv() {
            text_parts.push(t);
        }

        match result {
            Ok(usage) => {
                self.registry
                    .finish(&agent_id, AgentStatus::Completed, usage.total_tokens())
                    .await;
                let _ = progress
                    .send(AgentEvent::SubAgentProgress {
                        agent_id: agent_id.clone(),
                        event: SubAgentEvent::Finished {
                            tokens: usage.total_tokens(),
                        },
                    })
                    .await;

                let mut output = if text_parts.is_empty() {
                    history
                        .messages()
                        .iter()
                        .rev()
                        .find(|m| m.role == Role::Assistant)
                        .map(|m| {
                            m.blocks
                                .iter()
                                .filter_map(|b| {
                                    if let ContentBlock::Text { text } = b {
                                        Some(text.as_str())
                                    } else {
                                        None
                                    }
                                })
                                .collect::<Vec<_>>()
                                .join("\n")
                        })
                        .unwrap_or_else(|| "(sub-agent produced no text output)".into())
                } else {
                    text_parts.join("")
                };

                output.push_str(&format!(
                    "\n\n[sub-agent: {} tokens, mode={mode}]",
                    usage.total_tokens(),
                ));

                ToolResult::ok(output)
            }
            Err(e) => {
                self.registry
                    .finish(&agent_id, AgentStatus::Failed(e.to_string()), 0)
                    .await;
                let _ = progress
                    .send(AgentEvent::SubAgentProgress {
                        agent_id: agent_id.clone(),
                        event: SubAgentEvent::Error(e.to_string()),
                    })
                    .await;
                ToolResult::err(format!("Sub-agent error: {e}"))
            }
        }
    }
}

struct ArcProvider(Arc<dyn Provider>);

#[async_trait]
impl Provider for ArcProvider {
    fn name(&self) -> &str {
        self.0.name()
    }
    fn models(&self) -> Vec<crate::types::ModelInfo> {
        self.0.models()
    }
    async fn stream_chat(
        &self,
        request: crate::provider::ChatRequest,
    ) -> crate::error::Result<
        std::pin::Pin<Box<dyn tokio_stream::Stream<Item = crate::types::StreamChunk> + Send>>,
    > {
        self.0.stream_chat(request).await
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_support::NoopProvider;

    fn make_tool() -> SubAgentTool {
        SubAgentTool::new(Arc::new(NoopProvider), "mock".into(), 30, vec![])
    }

    #[test]
    fn spec_has_required_fields() {
        let tool = make_tool();
        let spec = tool.spec();
        assert_eq!(spec.name, "sub_agent");
        assert!(!spec.description.is_empty());
        assert!(spec.parameters.get("properties").is_some());
    }

    #[test]
    fn build_tools_explore_includes_read_tools() {
        let tool = make_tool();
        let registry = tool.build_tools("explore");
        assert!(registry.get("read_file").is_some());
        assert!(registry.get("glob_search").is_some());
        assert!(registry.get("grep_search").is_some());
    }

    /// Every accepted alias for read-only roles must produce a registry
    /// that contains no write_file or edit_file tool.
    #[test]
    fn build_tools_readonly_aliases_have_no_write_tools() {
        let tool = make_tool();
        // Explore aliases
        for alias in &["explore", "explorer", "exploration"] {
            let registry = tool.build_tools(alias);
            assert!(
                registry.get("write_file").is_none(),
                "write_file present for alias '{alias}'"
            );
            assert!(
                registry.get("edit_file").is_none(),
                "edit_file present for alias '{alias}'"
            );
        }
        // Plan aliases
        for alias in &["plan", "planning", "awaiter"] {
            let registry = tool.build_tools(alias);
            assert!(
                registry.get("write_file").is_none(),
                "write_file present for alias '{alias}'"
            );
            assert!(
                registry.get("edit_file").is_none(),
                "edit_file present for alias '{alias}'"
            );
        }
        // Review aliases
        for alias in &["review", "reviewer", "code-review"] {
            let registry = tool.build_tools(alias);
            assert!(
                registry.get("write_file").is_none(),
                "write_file present for alias '{alias}'"
            );
            assert!(
                registry.get("edit_file").is_none(),
                "edit_file present for alias '{alias}'"
            );
        }
        // Verifier aliases
        for alias in &["verifier", "verify", "verification", "validator", "tester"] {
            let registry = tool.build_tools(alias);
            assert!(
                registry.get("write_file").is_none(),
                "write_file present for alias '{alias}'"
            );
            assert!(
                registry.get("edit_file").is_none(),
                "edit_file present for alias '{alias}'"
            );
        }
        // Unknown / None role must also fail closed
        let registry = tool.build_tools("unknown-role-xyz");
        assert!(
            registry.get("write_file").is_none(),
            "write_file present for unknown role"
        );
        assert!(
            registry.get("edit_file").is_none(),
            "edit_file present for unknown role"
        );
    }

    #[test]
    fn build_tools_write_roles_include_write_tools() {
        let tool = make_tool();
        for alias in &[
            "general", "worker", "default", "general-purpose",
            "implementer", "implement", "implementation", "builder",
            "custom",
        ] {
            let registry = tool.build_tools(alias);
            assert!(
                registry.get("write_file").is_some(),
                "write_file missing for alias '{alias}'"
            );
            assert!(
                registry.get("edit_file").is_some(),
                "edit_file missing for alias '{alias}'"
            );
            assert!(
                registry.get("bash").is_some(),
                "bash missing for alias '{alias}'"
            );
        }
    }

    #[tokio::test]
    async fn execute_missing_prompt_errors() {
        let tool = make_tool();
        let res = tool
            .execute(serde_json::json!({}), std::path::Path::new("/tmp"))
            .await;
        assert!(res.is_error);
        assert!(res.output.contains("prompt"));
    }

    #[tokio::test]
    async fn execute_empty_prompt_errors() {
        let tool = make_tool();
        let res = tool
            .execute(
                serde_json::json!({"prompt": "   "}),
                std::path::Path::new("/tmp"),
            )
            .await;
        assert!(res.is_error);
    }
}
