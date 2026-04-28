use std::sync::Arc;

use crate::config::McpServerConfig;
use crate::error::{AgentError, Result};

use super::protocol::{JsonRpcRequest, McpToolCallResult, McpToolInfo, McpToolsListResult};
use super::transport::{McpTransport, StdioTransport};

/// A connected MCP server with its discovered tools.
pub struct McpServer {
    pub name: String,
    transport: Arc<dyn McpTransport>,
    tools: Vec<McpToolInfo>,
    next_id: std::sync::atomic::AtomicU64,
}

impl McpServer {
    /// Connect to an MCP server, perform handshake, and discover tools.
    pub async fn connect(config: &McpServerConfig) -> Result<Self> {
        let transport: Arc<dyn McpTransport> = match config.transport {
            crate::config::McpTransportType::Stdio => {
                Arc::new(StdioTransport::spawn(config).await?)
            }
            _ => {
                return Err(AgentError::Provider(format!(
                    "MCP transport {:?} not yet supported",
                    config.transport
                )));
            }
        };

        let server = Self {
            name: config.name.clone(),
            transport,
            tools: Vec::new(),
            next_id: std::sync::atomic::AtomicU64::new(10),
        };

        server.initialize().await?;
        let tools = server.list_tools().await?;

        Ok(Self { tools, ..server })
    }

    /// Connect using a pre-built transport (for testing).
    pub async fn connect_with_transport(
        name: String,
        transport: Arc<dyn McpTransport>,
    ) -> Result<Self> {
        let server = Self {
            name,
            transport,
            tools: Vec::new(),
            next_id: std::sync::atomic::AtomicU64::new(10),
        };

        server.initialize().await?;
        let tools = server.list_tools().await?;

        Ok(Self { tools, ..server })
    }

    async fn initialize(&self) -> Result<()> {
        let init_req = JsonRpcRequest::new(
            0,
            "initialize",
            Some(serde_json::json!({
                "protocolVersion": "2024-11-05",
                "capabilities": {},
                "clientInfo": {
                    "name": "naked",
                    "version": "0.1.0"
                }
            })),
        );

        let resp = self.transport.send_and_recv(&init_req).await?;
        resp.into_result()?;

        let notif = JsonRpcRequest::notification("notifications/initialized", None);
        self.transport.send(&notif).await?;

        Ok(())
    }

    async fn list_tools(&self) -> Result<Vec<McpToolInfo>> {
        let req = JsonRpcRequest::new(1, "tools/list", None);
        let resp = self.transport.send_and_recv(&req).await?;
        let value = resp.into_result()?;

        let result: McpToolsListResult = serde_json::from_value(value)
            .map_err(|e| AgentError::Provider(format!("MCP tools/list parse: {e}")))?;

        Ok(result.tools)
    }

    pub fn tools(&self) -> &[McpToolInfo] {
        &self.tools
    }

    /// Call a tool on this MCP server.
    pub async fn call_tool(
        &self,
        tool_name: &str,
        arguments: serde_json::Value,
    ) -> Result<McpToolCallResult> {
        let id = self
            .next_id
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let req = JsonRpcRequest::new(
            id,
            "tools/call",
            Some(serde_json::json!({
                "name": tool_name,
                "arguments": arguments,
            })),
        );

        let resp = self.transport.send_and_recv(&req).await?;
        let value = resp.into_result()?;

        let result: McpToolCallResult = serde_json::from_value(value)
            .map_err(|e| AgentError::Provider(format!("MCP tools/call parse: {e}")))?;

        Ok(result)
    }

    pub async fn close(&self) -> Result<()> {
        self.transport.close().await
    }
}

/// Registry of all connected MCP servers and their tools.
pub struct McpRegistry {
    servers: Vec<Arc<McpServer>>,
}

impl McpRegistry {
    pub fn new() -> Self {
        Self {
            servers: Vec::new(),
        }
    }

    /// Connect to all configured MCP servers. Logs errors but continues.
    pub async fn connect_all(configs: &[McpServerConfig]) -> Self {
        let mut servers = Vec::new();
        for config in configs {
            match McpServer::connect(config).await {
                Ok(server) => {
                    tracing::info!(
                        "MCP server '{}' connected: {} tools",
                        server.name,
                        server.tools.len()
                    );
                    servers.push(Arc::new(server));
                }
                Err(e) => {
                    tracing::warn!("MCP server '{}' failed to connect: {e}", config.name);
                }
            }
        }
        Self { servers }
    }

    /// Connect using pre-built servers (for testing).
    pub fn from_servers(servers: Vec<Arc<McpServer>>) -> Self {
        Self { servers }
    }

    pub fn servers(&self) -> &[Arc<McpServer>] {
        &self.servers
    }

    /// All tools across all servers, as (server_name, tool_info) pairs.
    pub fn all_tools(&self) -> Vec<(String, McpToolInfo)> {
        let mut tools = Vec::new();
        for server in &self.servers {
            for tool in &server.tools {
                tools.push((server.name.clone(), tool.clone()));
            }
        }
        tools
    }

    /// Find which server owns a given tool name.
    pub fn find_server_for_tool(&self, tool_name: &str) -> Option<Arc<McpServer>> {
        for server in &self.servers {
            if server.tools.iter().any(|t| t.name == tool_name) {
                return Some(Arc::clone(server));
            }
        }
        None
    }

    pub async fn close_all(&self) {
        for server in &self.servers {
            if let Err(e) = server.close().await {
                tracing::warn!("MCP server '{}' close error: {e}", server.name);
            }
        }
    }
}

impl Default for McpRegistry {
    fn default() -> Self {
        Self::new()
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
                "serverInfo": {"name": "test-server", "version": "1.0"}
            })),
            error: None,
        }
    }

    fn tools_list_response(tools: Vec<serde_json::Value>) -> JsonRpcResponse {
        JsonRpcResponse {
            jsonrpc: "2.0".into(),
            id: Some(1),
            result: Some(serde_json::json!({"tools": tools})),
            error: None,
        }
    }

    #[tokio::test]
    async fn connect_with_transport_performs_handshake() {
        let transport = Arc::new(MockTransport::new(vec![
            init_response(),
            tools_list_response(vec![
                serde_json::json!({"name": "read_file", "description": "Read a file"}),
            ]),
        ]));

        let server = McpServer::connect_with_transport("test".into(), transport.clone())
            .await
            .unwrap();

        assert_eq!(server.name, "test");
        assert_eq!(server.tools().len(), 1);
        assert_eq!(server.tools()[0].name, "read_file");

        let sent = transport.sent_requests().await;
        assert_eq!(sent.len(), 3); // initialize, notifications/initialized, tools/list
        assert_eq!(sent[0].method, "initialize");
        assert_eq!(sent[1].method, "notifications/initialized");
        assert_eq!(sent[2].method, "tools/list");
    }

    #[tokio::test]
    async fn call_tool_sends_correct_request() {
        let call_response = JsonRpcResponse {
            jsonrpc: "2.0".into(),
            id: Some(2),
            result: Some(serde_json::json!({
                "content": [{"type": "text", "text": "file contents"}],
                "isError": false,
            })),
            error: None,
        };
        let transport = Arc::new(MockTransport::new(vec![
            init_response(),
            tools_list_response(vec![]),
            call_response,
        ]));

        let server = McpServer::connect_with_transport("test".into(), transport.clone())
            .await
            .unwrap();

        let result = server
            .call_tool("read_file", serde_json::json!({"path": "/tmp/a.txt"}))
            .await
            .unwrap();

        assert!(!result.is_error);
        assert_eq!(result.content[0].as_text(), Some("file contents"));
    }

    #[tokio::test]
    async fn registry_find_server() {
        let transport = Arc::new(MockTransport::new(vec![
            init_response(),
            tools_list_response(vec![
                serde_json::json!({"name": "tool_a"}),
                serde_json::json!({"name": "tool_b"}),
            ]),
        ]));

        let server = McpServer::connect_with_transport("srv1".into(), transport)
            .await
            .unwrap();

        let registry = McpRegistry::from_servers(vec![Arc::new(server)]);

        assert!(registry.find_server_for_tool("tool_a").is_some());
        assert!(registry.find_server_for_tool("tool_b").is_some());
        assert!(registry.find_server_for_tool("tool_c").is_none());
    }

    #[tokio::test]
    async fn registry_all_tools() {
        let transport = Arc::new(MockTransport::new(vec![
            init_response(),
            tools_list_response(vec![
                serde_json::json!({"name": "x"}),
                serde_json::json!({"name": "y"}),
            ]),
        ]));

        let server = McpServer::connect_with_transport("srv".into(), transport)
            .await
            .unwrap();

        let registry = McpRegistry::from_servers(vec![Arc::new(server)]);
        let all = registry.all_tools();
        assert_eq!(all.len(), 2);
        assert_eq!(all[0].0, "srv");
    }

    #[test]
    fn registry_default_is_empty() {
        let r = McpRegistry::default();
        assert!(r.servers().is_empty());
        assert!(r.all_tools().is_empty());
    }
}
