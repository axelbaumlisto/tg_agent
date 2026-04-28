use std::path::Path;

use async_trait::async_trait;

use crate::agent_registry::AgentRegistry;
use crate::types::{Permission, ToolResult, ToolSpec};

use super::Tool;

/// Query status of running/completed sub-agents.
pub struct AgentStatusTool {
    registry: AgentRegistry,
}

impl AgentStatusTool {
    pub fn new(registry: AgentRegistry) -> Self {
        Self { registry }
    }
}

#[async_trait]
impl Tool for AgentStatusTool {
    fn spec(&self) -> ToolSpec {
        ToolSpec {
            name: "agent_status".into(),
            description: "List running and recently completed sub-agents with their status, \
                          tools used, and token count. Use to monitor sub-agent progress."
                .into(),
            parameters: serde_json::json!({
                "type": "object",
                "properties": {
                    "agent_id": {
                        "type": "string",
                        "description": "Optional: query a specific agent by ID. Omit to list all."
                    }
                }
            }),
            permission: Permission::ReadOnly,
        }
    }

    async fn execute(&self, input: serde_json::Value, _cwd: &Path) -> ToolResult {
        let specific_id = input.get("agent_id").and_then(|v| v.as_str());

        if let Some(id) = specific_id {
            match self.registry.get(id).await {
                Some(entry) => {
                    let elapsed = entry.started_at.elapsed().as_secs();
                    let output = format!(
                        "Agent: {}\nStatus: {}\nMode: {}\nPrompt: {}\n\
                         Tokens: {}\nTools: {}\nLast tool: {}\nElapsed: {}s",
                        entry.agent_id,
                        entry.status,
                        entry.mode,
                        entry.prompt_preview,
                        entry.tokens,
                        entry.tools_used.join(", "),
                        entry.last_tool.as_deref().unwrap_or("-"),
                        elapsed,
                    );
                    ToolResult {
                        output,
                        is_error: false,
                    }
                }
                None => ToolResult {
                    output: format!("Agent '{id}' not found"),
                    is_error: true,
                },
            }
        } else {
            let agents = self.registry.list_all().await;
            if agents.is_empty() {
                return ToolResult {
                    output: "No sub-agents have been started in this session.".into(),
                    is_error: false,
                };
            }

            let mut lines = Vec::with_capacity(agents.len() + 1);
            lines.push(format!("{} agent(s):", agents.len()));
            for e in &agents {
                let elapsed = e.started_at.elapsed().as_secs();
                let tool_info = e.last_tool.as_deref().unwrap_or("-");
                lines.push(format!(
                    "  {} [{}] {}s | {} tok | last: {} | {}",
                    e.agent_id, e.status, elapsed, e.tokens, tool_info, e.prompt_preview
                ));
            }
            ToolResult {
                output: lines.join("\n"),
                is_error: false,
            }
        }
    }
}

/// Stop a running sub-agent by ID.
pub struct AgentStopTool {
    registry: AgentRegistry,
}

impl AgentStopTool {
    pub fn new(registry: AgentRegistry) -> Self {
        Self { registry }
    }
}

#[async_trait]
impl Tool for AgentStopTool {
    fn spec(&self) -> ToolSpec {
        ToolSpec {
            name: "agent_stop".into(),
            description: "Cancel a running sub-agent by its ID. \
                          Use agent_status to find agent IDs first."
                .into(),
            parameters: serde_json::json!({
                "type": "object",
                "properties": {
                    "agent_id": {
                        "type": "string",
                        "description": "ID of the sub-agent to stop"
                    }
                },
                "required": ["agent_id"]
            }),
            permission: Permission::WorkspaceWrite,
        }
    }

    async fn execute(&self, input: serde_json::Value, _cwd: &Path) -> ToolResult {
        let agent_id = match input.get("agent_id").and_then(|v| v.as_str()) {
            Some(id) => id,
            None => {
                return ToolResult {
                    output: "Error: 'agent_id' is required".into(),
                    is_error: true,
                };
            }
        };

        if self.registry.cancel(agent_id).await {
            ToolResult {
                output: format!("Agent '{agent_id}' cancellation requested"),
                is_error: false,
            }
        } else {
            ToolResult {
                output: format!("Agent '{agent_id}' not found or already finished"),
                is_error: true,
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tokio_util::sync::CancellationToken;

    #[tokio::test]
    async fn status_empty() {
        let reg = AgentRegistry::new();
        let tool = AgentStatusTool::new(reg);
        let r = tool.execute(serde_json::json!({}), Path::new("/tmp")).await;
        assert!(!r.is_error);
        assert!(r.output.contains("No sub-agents"));
    }

    #[tokio::test]
    async fn status_with_agents() {
        let reg = AgentRegistry::new();
        reg.register("sa-1", "task A", "explore", CancellationToken::new())
            .await;
        reg.register("sa-2", "task B", "general", CancellationToken::new())
            .await;
        reg.update_tool("sa-1", "web_search").await;

        let tool = AgentStatusTool::new(reg);
        let r = tool.execute(serde_json::json!({}), Path::new("/tmp")).await;
        assert!(!r.is_error);
        assert!(r.output.contains("2 agent(s)"));
        assert!(r.output.contains("sa-1"));
        assert!(r.output.contains("sa-2"));
    }

    #[tokio::test]
    async fn status_specific() {
        let reg = AgentRegistry::new();
        reg.register("sa-5", "deep search", "explore", CancellationToken::new())
            .await;

        let tool = AgentStatusTool::new(reg);
        let r = tool
            .execute(serde_json::json!({"agent_id": "sa-5"}), Path::new("/tmp"))
            .await;
        assert!(!r.is_error);
        assert!(r.output.contains("deep search"));
    }

    #[tokio::test]
    async fn stop_running_agent() {
        let reg = AgentRegistry::new();
        let cancel = CancellationToken::new();
        let c2 = cancel.clone();
        reg.register("sa-10", "long task", "general", cancel).await;

        let tool = AgentStopTool::new(reg);
        let r = tool
            .execute(serde_json::json!({"agent_id": "sa-10"}), Path::new("/tmp"))
            .await;
        assert!(!r.is_error);
        assert!(c2.is_cancelled());
    }

    #[tokio::test]
    async fn stop_nonexistent() {
        let reg = AgentRegistry::new();
        let tool = AgentStopTool::new(reg);
        let r = tool
            .execute(serde_json::json!({"agent_id": "sa-999"}), Path::new("/tmp"))
            .await;
        assert!(r.is_error);
    }
}
