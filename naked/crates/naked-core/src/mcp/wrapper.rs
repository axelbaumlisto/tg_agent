use std::path::Path;
use std::sync::Arc;

use async_trait::async_trait;

use crate::tool::Tool;
use crate::types::{Permission, ToolResult, ToolSpec};

use super::client::McpServer;
use super::protocol::McpToolInfo;

/// Wraps a single MCP tool as a first-class `impl Tool`.
/// Each MCP tool gets its own wrapper instance so the agent
/// treats them identically to built-in tools.
pub struct McpToolWrapper {
    server: Arc<McpServer>,
    tool_info: McpToolInfo,
}

impl McpToolWrapper {
    pub fn new(server: Arc<McpServer>, tool_info: McpToolInfo) -> Self {
        Self { server, tool_info }
    }

    /// Create wrappers for all tools on a given server.
    pub fn wrap_all(server: Arc<McpServer>) -> Vec<Box<dyn Tool>> {
        server
            .tools()
            .iter()
            .map(|info| {
                Box::new(McpToolWrapper::new(Arc::clone(&server), info.clone())) as Box<dyn Tool>
            })
            .collect()
    }
}

#[async_trait]
impl Tool for McpToolWrapper {
    fn spec(&self) -> ToolSpec {
        ToolSpec {
            name: self.tool_info.name.clone(),
            description: self.tool_info.description.clone().unwrap_or_default(),
            parameters: {
                let mut schema = self
                    .tool_info
                    .input_schema
                    .clone()
                    .unwrap_or(serde_json::json!({"type": "object"}));
                crate::tool::schema_sanitize::sanitize(&mut schema);
                schema
            },
            permission: Permission::Dangerous,
        }
    }

    async fn execute(&self, input: serde_json::Value, _cwd: &Path) -> ToolResult {
        const MAX_MCP_OUTPUT: usize = 16_384;
        match self.server.call_tool(&self.tool_info.name, input).await {
            Ok(result) => {
                let output = result
                    .content
                    .iter()
                    .filter_map(|c| c.as_text())
                    .collect::<Vec<_>>()
                    .join("\n");
                let output = if output.is_empty() {
                    "(no output)".into()
                } else if output.len() <= MAX_MCP_OUTPUT {
                    output
                } else {
                    let mut end = MAX_MCP_OUTPUT;
                    while end > 0 && !output.is_char_boundary(end) {
                        end -= 1;
                    }
                    let cut = output.len() - end;
                    format!("{}\n\n[output truncated — {cut} bytes cut]", &output[..end])
                };
                if result.is_error {
                    ToolResult::err(output)
                } else {
                    ToolResult::ok(output)
                }
            }
            Err(e) => ToolResult::err(format!("MCP call failed: {e}")),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mcp::protocol::JsonRpcResponse;
    use crate::mcp::transport::MockTransport;

    fn init_response() -> JsonRpcResponse {
        JsonRpcResponse {
            jsonrpc: "2.0".into(),
            id: Some(0),
            result: Some(serde_json::json!({
                "protocolVersion": "2024-11-05",
                "capabilities": {},
                "serverInfo": {"name": "test", "version": "1.0"}
            })),
            error: None,
        }
    }

    fn tools_list_response() -> JsonRpcResponse {
        JsonRpcResponse {
            jsonrpc: "2.0".into(),
            id: Some(1),
            result: Some(serde_json::json!({
                "tools": [{
                    "name": "mcp_read",
                    "description": "MCP read file",
                    "inputSchema": {"type": "object", "properties": {"path": {"type": "string"}}}
                }]
            })),
            error: None,
        }
    }

    fn call_response(text: &str, is_error: bool) -> JsonRpcResponse {
        JsonRpcResponse {
            jsonrpc: "2.0".into(),
            id: Some(2),
            result: Some(serde_json::json!({
                "content": [{"type": "text", "text": text}],
                "isError": is_error,
            })),
            error: None,
        }
    }

    #[tokio::test]
    async fn wrapper_spec() {
        let transport = Arc::new(MockTransport::new(vec![
            init_response(),
            tools_list_response(),
        ]));
        let server = Arc::new(
            McpServer::connect_with_transport("srv".into(), transport)
                .await
                .unwrap(),
        );

        let wrappers = McpToolWrapper::wrap_all(server);
        assert_eq!(wrappers.len(), 1);

        let spec = wrappers[0].spec();
        assert_eq!(spec.name, "mcp_read");
        assert_eq!(spec.description, "MCP read file");
        assert_eq!(spec.permission, Permission::Dangerous);
    }

    #[tokio::test]
    async fn wrapper_execute_success() {
        let transport = Arc::new(MockTransport::new(vec![
            init_response(),
            tools_list_response(),
            call_response("file data here", false),
        ]));
        let server = Arc::new(
            McpServer::connect_with_transport("srv".into(), transport)
                .await
                .unwrap(),
        );

        let wrappers = McpToolWrapper::wrap_all(server);
        let result = wrappers[0]
            .execute(serde_json::json!({"path": "/tmp/x.txt"}), Path::new("/tmp"))
            .await;

        assert!(!result.is_error);
        assert_eq!(result.output, "file data here");
    }

    #[tokio::test]
    async fn wrapper_execute_error() {
        let transport = Arc::new(MockTransport::new(vec![
            init_response(),
            tools_list_response(),
            call_response("not found", true),
        ]));
        let server = Arc::new(
            McpServer::connect_with_transport("srv".into(), transport)
                .await
                .unwrap(),
        );

        let wrappers = McpToolWrapper::wrap_all(server);
        let result = wrappers[0]
            .execute(serde_json::json!({}), Path::new("/tmp"))
            .await;

        assert!(result.is_error);
        assert_eq!(result.output, "not found");
    }

    #[tokio::test]
    async fn wrapper_execute_empty_output() {
        let transport = Arc::new(MockTransport::new(vec![
            init_response(),
            tools_list_response(),
            JsonRpcResponse {
                jsonrpc: "2.0".into(),
                id: Some(2),
                result: Some(serde_json::json!({"content": [], "isError": false})),
                error: None,
            },
        ]));
        let server = Arc::new(
            McpServer::connect_with_transport("srv".into(), transport)
                .await
                .unwrap(),
        );

        let wrappers = McpToolWrapper::wrap_all(server);
        let result = wrappers[0]
            .execute(serde_json::json!({}), Path::new("/tmp"))
            .await;

        assert!(!result.is_error);
        assert_eq!(result.output, "(no output)");
    }
}
