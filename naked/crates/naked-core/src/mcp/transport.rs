use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};

use async_trait::async_trait;
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};
use tokio::process::{Child, Command};
use tokio::sync::Mutex;

use crate::config::McpServerConfig;
use crate::error::{AgentError, Result};
use crate::provider::error::ProviderError;

use super::protocol::{JsonRpcRequest, JsonRpcResponse};

#[async_trait]
pub trait McpTransport: Send + Sync {
    async fn send(&self, req: &JsonRpcRequest) -> Result<()>;
    async fn send_and_recv(&self, req: &JsonRpcRequest) -> Result<JsonRpcResponse>;
    async fn close(&self) -> Result<()>;
}

pub struct StdioTransport {
    stdin: Mutex<tokio::process::ChildStdin>,
    stdout: Mutex<BufReader<tokio::process::ChildStdout>>,
    #[allow(dead_code)]
    child: Mutex<Child>,
    next_id: AtomicU64,
}

impl StdioTransport {
    pub async fn spawn(config: &McpServerConfig) -> Result<Self> {
        let mut cmd = Command::new(&config.command);
        cmd.args(&config.args)
            .stdin(std::process::Stdio::piped())
            .stdout(std::process::Stdio::piped())
            .stderr(std::process::Stdio::piped());

        for (k, v) in &config.env {
            let resolved = crate::config::expand_env(v).unwrap_or_else(|_| v.clone());
            cmd.env(k, &resolved);
        }

        let mut child = cmd.spawn().map_err(|e| {
            AgentError::ProviderTyped(ProviderError::Mcp {
                context: format!("MCP spawn '{}': {e}", config.command),
                source: String::new(),
            })
        })?;

        let stdin = child.stdin.take().ok_or_else(|| {
            AgentError::ProviderTyped(ProviderError::Mcp {
                context: "MCP process stdin not captured".into(),
                source: String::new(),
            })
        })?;
        let stdout = child.stdout.take().ok_or_else(|| {
            AgentError::ProviderTyped(ProviderError::Mcp {
                context: "MCP process stdout not captured".into(),
                source: String::new(),
            })
        })?;

        if let Some(stderr) = child.stderr.take() {
            let cmd_name = config.command.clone();
            tokio::spawn(async move {
                let mut reader = BufReader::new(stderr);
                let mut buf = String::new();
                loop {
                    buf.clear();
                    match tokio::io::AsyncBufReadExt::read_line(&mut reader, &mut buf).await {
                        Ok(0) | Err(_) => break,
                        Ok(_) => {
                            tracing::debug!(mcp = %cmd_name, "stderr: {}", buf.trim_end());
                        }
                    }
                }
            });
        }

        Ok(Self {
            stdin: Mutex::new(stdin),
            stdout: Mutex::new(BufReader::new(stdout)),
            child: Mutex::new(child),
            next_id: AtomicU64::new(1),
        })
    }

    pub fn next_id(&self) -> u64 {
        self.next_id.fetch_add(1, Ordering::Relaxed)
    }
}

#[async_trait]
impl McpTransport for StdioTransport {
    async fn send(&self, req: &JsonRpcRequest) -> Result<()> {
        let mut line = serde_json::to_string(req).map_err(|e| {
            AgentError::ProviderTyped(crate::provider::error::ProviderError::Mcp {
                context: "serialize".into(),
                source: e.to_string(),
            })
        })?;
        line.push('\n');

        let mut stdin = self.stdin.lock().await;
        stdin.write_all(line.as_bytes()).await?;
        stdin.flush().await?;
        Ok(())
    }

    async fn send_and_recv(&self, req: &JsonRpcRequest) -> Result<JsonRpcResponse> {
        self.send(req).await?;

        let mut stdout = self.stdout.lock().await;
        let mut line = String::new();
        match tokio::time::timeout(
            std::time::Duration::from_secs(60),
            stdout.read_line(&mut line),
        )
        .await
        {
            Ok(Ok(_)) => {}
            Ok(Err(e)) => return Err(e.into()),
            Err(_) => {
                return Err(AgentError::ProviderTyped(
                    crate::provider::error::ProviderError::Mcp {
                        context: "response timed out after 60s".into(),
                        source: String::new(),
                    },
                ));
            }
        }

        if line.trim().is_empty() {
            return Err(AgentError::ProviderTyped(ProviderError::Mcp {
                context: "MCP: empty response".into(),
                source: String::new(),
            }));
        }

        let resp: JsonRpcResponse = serde_json::from_str(line.trim()).map_err(|e| {
            AgentError::ProviderTyped(crate::provider::error::ProviderError::Mcp {
                context: "parse response".into(),
                source: e.to_string(),
            })
        })?;

        Ok(resp)
    }

    async fn close(&self) -> Result<()> {
        let mut child = self.child.lock().await;
        let _ = child.kill().await;
        Ok(())
    }
}

/// In-memory transport for testing. Pairs requests with canned responses.
#[cfg(test)]
pub struct MockTransport {
    responses: Mutex<Vec<JsonRpcResponse>>,
    sent: Mutex<Vec<JsonRpcRequest>>,
}

#[cfg(test)]
impl MockTransport {
    pub fn new(responses: Vec<JsonRpcResponse>) -> Self {
        Self {
            responses: Mutex::new(responses),
            sent: Mutex::new(Vec::new()),
        }
    }

    pub async fn sent_requests(&self) -> Vec<JsonRpcRequest> {
        self.sent.lock().await.clone()
    }
}

#[cfg(test)]
#[async_trait]
impl McpTransport for MockTransport {
    async fn send(&self, req: &JsonRpcRequest) -> Result<()> {
        self.sent.lock().await.push(req.clone());
        Ok(())
    }

    async fn send_and_recv(&self, req: &JsonRpcRequest) -> Result<JsonRpcResponse> {
        self.sent.lock().await.push(req.clone());
        let mut responses = self.responses.lock().await;
        if responses.is_empty() {
            Err(AgentError::ProviderTyped(ProviderError::Other {
                status: 0,
                body: "MockTransport: no more responses".into(),
            }))
        } else {
            Ok(responses.remove(0))
        }
    }

    async fn close(&self) -> Result<()> {
        Ok(())
    }
}

/// Build environment variables map from McpServerConfig, inheriting current env.
pub fn build_env(config: &McpServerConfig) -> HashMap<String, String> {
    let mut env: HashMap<String, String> = std::env::vars().collect();
    for (k, v) in &config.env {
        let resolved = crate::config::expand_env(v).unwrap_or_else(|_| v.clone());
        env.insert(k.clone(), resolved);
    }
    env
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn mock_transport_send_and_recv() {
        let resp = JsonRpcResponse {
            jsonrpc: "2.0".into(),
            id: Some(1),
            result: Some(serde_json::json!({"ok": true})),
            error: None,
        };
        let transport = MockTransport::new(vec![resp]);

        let req = JsonRpcRequest::new(1, "test/method", None);
        let result = transport.send_and_recv(&req).await.unwrap();

        assert_eq!(result.result.unwrap()["ok"], true);
        let sent = transport.sent_requests().await;
        assert_eq!(sent.len(), 1);
        assert_eq!(sent[0].method, "test/method");
    }

    #[tokio::test]
    async fn mock_transport_exhausted() {
        let transport = MockTransport::new(vec![]);
        let req = JsonRpcRequest::new(1, "test", None);
        assert!(transport.send_and_recv(&req).await.is_err());
    }

    #[tokio::test]
    async fn mock_transport_send_only() {
        let transport = MockTransport::new(vec![]);
        let req = JsonRpcRequest::notification("notify", None);
        assert!(transport.send(&req).await.is_ok());
        assert_eq!(transport.sent_requests().await.len(), 1);
    }

    #[test]
    fn build_env_merges() {
        let config = McpServerConfig {
            name: "test".into(),
            env: HashMap::from([("MY_KEY".into(), "my_val".into())]),
            ..default_server_config()
        };
        let env = build_env(&config);
        assert_eq!(env.get("MY_KEY").unwrap(), "my_val");
    }

    fn default_server_config() -> McpServerConfig {
        McpServerConfig {
            name: String::new(),
            transport: crate::config::McpTransportType::Stdio,
            command: String::new(),
            args: Vec::new(),
            env: HashMap::new(),
            url: None,
            headers: HashMap::new(),
            tool_timeout_secs: None,
        }
    }
}
