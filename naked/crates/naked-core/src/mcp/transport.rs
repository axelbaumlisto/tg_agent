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

/// Fallback when a server config does not set `tool_timeout_secs` — the value
/// that used to be hardcoded for every server.
const DEFAULT_MCP_TIMEOUT_SECS: u64 = 60;

pub struct StdioTransport {
    stdin: Mutex<tokio::process::ChildStdin>,
    stdout: Mutex<BufReader<tokio::process::ChildStdout>>,
    #[allow(dead_code)]
    child: Mutex<Child>,
    next_id: AtomicU64,
    /// B120a: one in-flight request per transport.
    ///
    /// `stdin` and `stdout` were separate locks, so two callers could
    /// interleave a write with the other's read. Read-only tools genuinely run
    /// in parallel (`loop_/tools.rs` uses `join_all`), so this was reachable.
    /// Holding one lock across the whole exchange makes "the next line is my
    /// answer" true instead of merely usual.
    exchange: Mutex<()>,
    /// B120c: the configured deadline, previously ignored in favour of a
    /// hardcoded 60s.
    request_timeout: std::time::Duration,
    /// B120a: set when a timeout leaves an unread reply in the pipe. The
    /// stream is then desynchronised and cannot be trusted again.
    desynced: std::sync::atomic::AtomicBool,
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
            exchange: Mutex::new(()),
            // B120c: honour the configured deadline. It was parsed and then
            // ignored in favour of a hardcoded 60s, so a short timeout set by
            // an operator did nothing at all.
            request_timeout: std::time::Duration::from_secs(
                config.tool_timeout_secs.unwrap_or(DEFAULT_MCP_TIMEOUT_SECS),
            ),
            desynced: std::sync::atomic::AtomicBool::new(false),
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
        // B120a: hold one lock for the whole exchange.
        let _exchange = self.exchange.lock().await;

        if self.desynced.load(std::sync::atomic::Ordering::Relaxed) {
            return Err(AgentError::ProviderTyped(ProviderError::Mcp {
                context: "MCP: transport desynchronised by an earlier timeout — refusing to guess which reply is mine".into(),
                source: String::new(),
            }));
        }

        let timeout = self.request_timeout;

        // B120b: the write half needs a deadline too. A server that stops
        // reading its stdin used to park the turn forever, because the read
        // timeout below could never start.
        match tokio::time::timeout(timeout, self.send(req)).await {
            Ok(Ok(())) => {}
            Ok(Err(e)) => return Err(e),
            Err(_) => {
                // A partial line may have been written; the server's framing is
                // no longer trustworthy.
                self.desynced
                    .store(true, std::sync::atomic::Ordering::Relaxed);
                return Err(AgentError::ProviderTyped(ProviderError::Mcp {
                    context: format!("MCP: request write timed out after {timeout:?}"),
                    source: String::new(),
                }));
            }
        }

        let mut stdout = self.stdout.lock().await;
        let mut line = String::new();
        match tokio::time::timeout(timeout, stdout.read_line(&mut line)).await {
            Ok(Ok(_)) => {}
            Ok(Err(e)) => return Err(e.into()),
            Err(_) => {
                // The reply may still arrive later and would then be read as
                // the NEXT call's answer. Refuse to reuse this stream.
                self.desynced
                    .store(true, std::sync::atomic::Ordering::Relaxed);
                return Err(AgentError::ProviderTyped(
                    crate::provider::error::ProviderError::Mcp {
                        context: format!("response timed out after {timeout:?}"),
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

        // B120a: THE check. Every request carries a unique `id` and every
        // response echoes it, but nothing compared them, so a stale or
        // out-of-order reply parsed cleanly into a valid-looking result and the
        // agent answered confidently from another tool's output. A wrong answer
        // that looks right is worse than an error.
        if resp.id != req.id {
            self.desynced
                .store(true, std::sync::atomic::Ordering::Relaxed);
            return Err(AgentError::ProviderTyped(ProviderError::Mcp {
                context: format!(
                    "MCP: response id {:?} does not match request id {:?} — refusing to use another call's result",
                    resp.id, req.id
                ),
                source: String::new(),
            }));
        }

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

    /// Build a fake MCP server from a shell script over real pipes, so the
    /// framing logic under test is the real one (MockTransport replaces
    /// `send_and_recv` wholesale and therefore cannot see any of this).
    fn scripted_server(script: &str) -> McpServerConfig {
        McpServerConfig {
            name: "fake".into(),
            command: "sh".into(),
            args: vec!["-c".into(), script.into()],
            tool_timeout_secs: Some(3),
            ..Default::default()
        }
    }

    /// B120a: a reply carrying someone else's id must be REFUSED, not used.
    ///
    /// Requests carry a unique id and responses echo it, but nothing compared
    /// them. A stale reply (left in the pipe after a timeout) or an
    /// out-of-order one parsed cleanly into a valid-looking result, so the
    /// agent answered confidently from another tool's output. Read-only tools
    /// really do run concurrently (`loop_/tools.rs` uses `join_all`), and
    /// production runs two MCP servers, so this was reachable.
    #[tokio::test]
    async fn response_with_foreign_id_is_rejected_not_used() {
        // Always answers id=999, whatever was asked.
        let t = StdioTransport::spawn(&scripted_server(
            r#"while read line; do printf '{"jsonrpc":"2.0","id":999,"result":{"stolen":true}}\n'; done"#,
        ))
        .await
        .expect("spawn fake server");

        let req = JsonRpcRequest::new(7, "tools/call", None);
        let err = t
            .send_and_recv(&req)
            .await
            .expect_err("a foreign id must not be accepted as this call's answer");
        let msg = format!("{err}");
        assert!(
            msg.contains("does not match request id"),
            "error must name the id mismatch, got: {msg}"
        );
    }

    /// A well-behaved server must still work — the guard must not reject
    /// legitimate traffic.
    #[tokio::test]
    async fn matching_id_is_accepted() {
        let t = StdioTransport::spawn(&scripted_server(
            r#"while read line; do id=$(printf '%s' "$line" | sed -E 's/.*"id":([0-9]+).*/\1/'); printf '{"jsonrpc":"2.0","id":%s,"result":{"ok":true}}\n' "$id"; done"#,
        ))
        .await
        .expect("spawn fake server");

        let req = JsonRpcRequest::new(42, "tools/call", None);
        let resp = t.send_and_recv(&req).await.expect("echoed id must be fine");
        assert_eq!(resp.id, Some(42));
    }

    /// B120a: once a timeout leaves an unread reply in the pipe, the stream is
    /// desynchronised — the next call must not inherit that pending answer.
    #[tokio::test]
    async fn transport_refuses_to_continue_after_a_timeout_desync() {
        // Never answers: forces the read timeout.
        let t = StdioTransport::spawn(&scripted_server("sleep 30"))
            .await
            .expect("spawn fake server");

        let first = JsonRpcRequest::new(1, "tools/call", None);
        assert!(
            t.send_and_recv(&first).await.is_err(),
            "a silent server must time out"
        );

        let second = JsonRpcRequest::new(2, "tools/call", None);
        let err = format!("{}", t.send_and_recv(&second).await.unwrap_err());
        assert!(
            err.contains("desynchronised"),
            "after a timeout the transport must refuse to guess, got: {err}"
        );
    }

    /// B120c: the configured timeout must be the one that fires.
    #[tokio::test]
    async fn configured_timeout_is_used_instead_of_the_old_hardcoded_60s() {
        let t = StdioTransport::spawn(&scripted_server("sleep 30"))
            .await
            .expect("spawn fake server");
        let started = std::time::Instant::now();
        let _ = t.send_and_recv(&JsonRpcRequest::new(1, "x", None)).await;
        let waited = started.elapsed();
        assert!(
            waited < std::time::Duration::from_secs(15),
            "a 3s configured timeout must not wait the old hardcoded 60s; waited {waited:?}"
        );
    }

    /// B120b: the WRITE half needs its own deadline.
    ///
    /// The 60s timeout only ever covered `read_line`. A server that never
    /// reads its stdin fills the pipe, `write_all`/`flush` block, and the read
    /// timeout can never start — the turn parks forever. Writing more than one
    /// pipe buffer (64 KiB on Linux) to a non-reading child reproduces it.
    #[tokio::test]
    async fn write_half_times_out_when_the_server_never_reads_stdin() {
        // Never reads stdin; just stays alive so the pipe stays open.
        let t = StdioTransport::spawn(&scripted_server("sleep 30"))
            .await
            .expect("spawn fake server");

        let big = "x".repeat(512 * 1024);
        let req = JsonRpcRequest::new(1, "tools/call", Some(serde_json::json!({ "payload": big })));

        let started = std::time::Instant::now();
        let err = t
            .send_and_recv(&req)
            .await
            .expect_err("a server that never drains stdin must not park us forever");
        assert!(
            started.elapsed() < std::time::Duration::from_secs(20),
            "the write must be bounded by the configured timeout"
        );
        let msg = format!("{err}");
        assert!(
            msg.contains("write timed out") || msg.contains("timed out"),
            "error must say it timed out, got: {msg}"
        );
    }
}
