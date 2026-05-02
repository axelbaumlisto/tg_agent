pub mod agent_control;
pub mod bash;
pub mod file_lock;
pub mod file_ops;
pub mod image_result;
pub mod factory;
pub mod ops;
pub mod policy;
pub mod remote;
pub mod memory;
pub mod registry;
pub mod search;
pub mod sub_agent;
pub mod web_fetch;
pub mod web_fetch_tls;
pub mod web_fetch_wayback;
pub mod web_search;

use std::path::Path;

use async_trait::async_trait;
use tokio::sync::mpsc;

use crate::types::{AgentEvent, Permission, ToolResult, ToolSpec};

#[async_trait]
pub trait Tool: Send + Sync {
    fn spec(&self) -> ToolSpec;
    async fn execute(&self, input: serde_json::Value, cwd: &Path) -> ToolResult;

    /// Execute with a channel to report progress (heartbeats, sub-agent events).
    /// Default: delegates to `execute()`, ignoring the channel.
    /// Override in tools that run long or spawn child agents.
    async fn execute_with_progress(
        &self,
        input: serde_json::Value,
        cwd: &Path,
        _progress: mpsc::Sender<AgentEvent>,
    ) -> ToolResult {
        self.execute(input, cwd).await
    }

    /// Permission level for a specific invocation. Override to escalate based on input
    /// (e.g. writing files outside workspace).
    fn effective_permission(&self, _input: &serde_json::Value, _cwd: &Path) -> Permission {
        self.spec().permission
    }
}
