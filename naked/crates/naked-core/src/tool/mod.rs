pub mod agent_control;
pub mod apply_patch;
pub mod approval_cache;
pub mod arg_repair;
pub mod bash;
pub mod diagnostics;
pub mod diff_format;
pub mod factory;
pub mod fetch_common;
pub mod fff_tools;
pub mod file_lock;
pub mod file_ops;
pub mod file_search;
pub mod git_tools;
pub mod image_result;
pub mod large_output;
pub mod memory;
pub mod ops;
pub mod plan_tool;
pub mod policy;
pub mod recall_archive;
pub mod registry;
pub mod remember;
pub mod remote;
pub mod research_run;
pub mod revert_turn;
pub mod review_tool;
pub mod schema_sanitize;
pub mod search;
pub mod sub_agent;
pub mod test_runner;
pub mod todo_tool;
pub mod validate_data;
pub mod validator;
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

/// Parse a tool's JSON input into a typed struct, returning a `ToolResult`
/// error on failure. Eliminates the repeated `match serde_json::from_value`
/// boilerplate across tool implementations.
pub fn parse_tool_input<T: serde::de::DeserializeOwned>(
    input: serde_json::Value,
) -> Result<T, ToolResult> {
    serde_json::from_value(input).map_err(|e| ToolResult::err(format!("Invalid input: {e}")))
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde::Deserialize;

    #[derive(Debug, Deserialize)]
    struct TestInput {
        name: String,
        count: u32,
    }

    #[test]
    fn parse_valid_input() {
        let input = serde_json::json!({"name": "test", "count": 5});
        let result: Result<TestInput, ToolResult> = parse_tool_input(input);
        assert!(result.is_ok());
        let parsed = result.unwrap();
        assert_eq!(parsed.name, "test");
        assert_eq!(parsed.count, 5);
    }

    #[test]
    fn parse_invalid_input_returns_tool_error() {
        let input = serde_json::json!({"wrong_field": true});
        let result: Result<TestInput, ToolResult> = parse_tool_input(input);
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(err.is_error);
        assert!(err.output.contains("Invalid input"));
    }

    #[test]
    fn parse_empty_object() {
        let input = serde_json::json!({});
        let result: Result<TestInput, ToolResult> = parse_tool_input(input);
        assert!(result.is_err());
    }
}
