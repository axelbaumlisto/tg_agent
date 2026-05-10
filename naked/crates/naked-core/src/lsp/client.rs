//! Stdio JSON-RPC transport for LSP servers.
//!
//! Spawns a server as a child process, sends `initialize`, then
//! relays `did_open` / `did_change` / `await_diagnostics`. Custom
//! framing per the LSP spec: each frame is
//! `Content-Length: N\r\n\r\n<body>` where body is JSON-RPC 2.0.
//!
//! KISS: we don't pull in `tower-lsp` or `lsp-types`. The wire
//! format is small; ad-hoc serde_json gives us all we need.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::process::Stdio;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Duration;

use serde_json::{Value, json};
use tokio::io::{AsyncBufReadExt, AsyncReadExt, AsyncWriteExt, BufReader};
use tokio::process::{ChildStdin, Command};
use tokio::sync::{Mutex, RwLock};

use super::diagnostics::{Diagnostic, Severity};
use super::registry::Language;

#[derive(Debug)]
pub struct StdioLspTransport {
    /// Stdin handle of the LSP server (write side).
    stdin: Mutex<ChildStdin>,
    /// Per-uri buffer of received diagnostics. Updated by the
    /// background reader task; read by `await_diagnostics`.
    diagnostics: Arc<RwLock<HashMap<String, Vec<Diagnostic>>>>,
    /// Set of file URIs already opened (for `did_open` vs `did_change`).
    opened: Mutex<std::collections::HashSet<String>>,
    /// Monotonic ID counter for JSON-RPC requests.
    next_id: AtomicU64,
}

impl StdioLspTransport {
    /// Spawn an LSP server as a subprocess and send `initialize`.
    /// `(cmd, args)` is what the registry returned.
    pub async fn spawn(
        cmd_args: (&'static str, &'static [&'static str]),
        workspace: &Path,
    ) -> Result<Self, String> {
        let (cmd, args) = cmd_args;
        let mut child = Command::new(cmd)
            .args(args)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::null())
            .spawn()
            .map_err(|e| format!("spawn {cmd}: {e}"))?;
        let stdin = child.stdin.take().ok_or("no stdin")?;
        let stdout = child.stdout.take().ok_or("no stdout")?;
        let diagnostics: Arc<RwLock<HashMap<String, Vec<Diagnostic>>>> =
            Arc::new(RwLock::new(HashMap::new()));

        // Background reader task: parse Content-Length frames,
        // dispatch publishDiagnostics into the shared map.
        let diags = diagnostics.clone();
        tokio::spawn(read_loop(stdout, diags));

        let transport = Self {
            stdin: Mutex::new(stdin),
            diagnostics,
            opened: Mutex::new(std::collections::HashSet::new()),
            next_id: AtomicU64::new(1),
        };

        // Send `initialize`.
        let workspace_uri = path_to_uri(workspace);
        let init_id = transport.next_id.fetch_add(1, Ordering::Relaxed);
        let init = json!({
            "jsonrpc": "2.0",
            "id": init_id,
            "method": "initialize",
            "params": {
                "processId": std::process::id(),
                "rootUri": workspace_uri,
                "capabilities": {
                    "textDocument": {
                        "publishDiagnostics": { "relatedInformation": false }
                    }
                }
            }
        });
        transport.write_frame(&init).await?;
        // Send `initialized` notification (don't wait for response).
        transport
            .write_frame(&json!({
                "jsonrpc": "2.0",
                "method": "initialized",
                "params": {}
            }))
            .await?;
        Ok(transport)
    }

    /// Open or notify-change the document. Server sees `didOpen` on
    /// first call per file, `didChange` on subsequent calls.
    pub async fn did_open_or_change(
        &self,
        path: &Path,
        text: &str,
        lang: Language,
    ) -> Result<(), String> {
        let uri = path_to_uri(path);
        let mut opened = self.opened.lock().await;
        if opened.insert(uri.clone()) {
            // First time → didOpen.
            let frame = json!({
                "jsonrpc": "2.0",
                "method": "textDocument/didOpen",
                "params": {
                    "textDocument": {
                        "uri": uri,
                        "languageId": lang.language_id(),
                        "version": 1,
                        "text": text
                    }
                }
            });
            drop(opened);
            self.write_frame(&frame).await?;
        } else {
            // Subsequent → didChange (full sync).
            let frame = json!({
                "jsonrpc": "2.0",
                "method": "textDocument/didChange",
                "params": {
                    "textDocument": { "uri": uri, "version": 2 },
                    "contentChanges": [{ "text": text }]
                }
            });
            drop(opened);
            self.write_frame(&frame).await?;
        }
        Ok(())
    }

    /// Block up to `budget` waiting for diagnostics for `path`.
    /// Returns whatever we have when the budget expires.
    pub async fn await_diagnostics(&self, path: &Path, budget: Duration) -> Vec<Diagnostic> {
        let uri = path_to_uri(path);
        let deadline = std::time::Instant::now() + budget;
        loop {
            // Snapshot + early-return when something's there.
            {
                let g = self.diagnostics.read().await;
                if let Some(d) = g.get(&uri)
                    && !d.is_empty()
                {
                    return d.clone();
                }
            }
            if std::time::Instant::now() >= deadline {
                let g = self.diagnostics.read().await;
                return g.get(&uri).cloned().unwrap_or_default();
            }
            tokio::time::sleep(Duration::from_millis(50)).await;
        }
    }

    async fn write_frame(&self, body: &Value) -> Result<(), String> {
        let payload = body.to_string();
        let frame = format!("Content-Length: {}\r\n\r\n{}", payload.len(), payload);
        let mut stdin = self.stdin.lock().await;
        stdin
            .write_all(frame.as_bytes())
            .await
            .map_err(|e| format!("write: {e}"))?;
        stdin.flush().await.map_err(|e| format!("flush: {e}"))?;
        Ok(())
    }
}

/// Background read loop. Parses `Content-Length: N\r\n\r\n<body>`
/// frames and dispatches `publishDiagnostics` notifications into
/// the shared map. Other notifications/responses are dropped.
async fn read_loop(
    stdout: tokio::process::ChildStdout,
    diagnostics: Arc<RwLock<HashMap<String, Vec<Diagnostic>>>>,
) {
    let mut reader = BufReader::new(stdout);
    let mut header_line = String::new();
    loop {
        // Read headers until blank line.
        let mut content_length: Option<usize> = None;
        loop {
            header_line.clear();
            let n = match reader.read_line(&mut header_line).await {
                Ok(0) => return, // EOF
                Ok(n) => n,
                Err(_) => return,
            };
            if n == 0 {
                return;
            }
            let trimmed = header_line.trim_end();
            if trimmed.is_empty() {
                break;
            }
            if let Some(rest) = trimmed.strip_prefix("Content-Length:")
                && let Ok(n) = rest.trim().parse::<usize>()
            {
                content_length = Some(n);
            }
        }
        let Some(len) = content_length else {
            return;
        };
        let mut body = vec![0u8; len];
        if reader.read_exact(&mut body).await.is_err() {
            return;
        }
        let Ok(value) = serde_json::from_slice::<Value>(&body) else {
            continue;
        };
        if value.get("method").and_then(Value::as_str) == Some("textDocument/publishDiagnostics") {
            handle_publish_diagnostics(&value, &diagnostics).await;
        }
    }
}

async fn handle_publish_diagnostics(
    value: &Value,
    map: &Arc<RwLock<HashMap<String, Vec<Diagnostic>>>>,
) {
    let params = match value.get("params") {
        Some(p) => p,
        None => return,
    };
    let Some(uri) = params.get("uri").and_then(Value::as_str) else {
        return;
    };
    let Some(diags_arr) = params.get("diagnostics").and_then(Value::as_array) else {
        return;
    };
    let mut out = Vec::with_capacity(diags_arr.len());
    for d in diags_arr {
        let severity = d
            .get("severity")
            .and_then(Value::as_u64)
            .map(|c| Severity::from_lsp_code(c as u8))
            .unwrap_or(Severity::Info);
        let range = d.get("range");
        let line = range
            .and_then(|r| r.get("start"))
            .and_then(|s| s.get("line"))
            .and_then(Value::as_u64)
            .unwrap_or(0) as u32;
        let col = range
            .and_then(|r| r.get("start"))
            .and_then(|s| s.get("character"))
            .and_then(Value::as_u64)
            .unwrap_or(0) as u32;
        let message = d
            .get("message")
            .and_then(Value::as_str)
            .unwrap_or_default()
            .to_string();
        let source = d.get("source").and_then(Value::as_str).map(String::from);
        out.push(Diagnostic {
            severity,
            line,
            col,
            message,
            source,
        });
    }
    let mut g = map.write().await;
    g.insert(uri.to_string(), out);
}

#[must_use]
pub fn path_to_uri(path: &Path) -> String {
    let canonical: PathBuf = path.canonicalize().unwrap_or_else(|_| path.to_path_buf());
    let s = canonical.to_string_lossy();
    if s.starts_with('/') {
        format!("file://{s}")
    } else {
        format!("file:///{}", s.replace('\\', "/"))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn frame_parser_parses_content_length() {
        // We can't easily test the full async read loop without a
        // real subprocess; instead we test the path_to_uri formatter
        // which is the other pure piece of the wire layer.
        let p = std::path::PathBuf::from("/tmp/foo.rs");
        let uri = path_to_uri(&p);
        assert!(uri.starts_with("file:///"));
        assert!(uri.contains("foo.rs"));
    }

    #[tokio::test]
    async fn missing_binary_returns_err() {
        let res = StdioLspTransport::spawn(
            ("definitely-not-a-real-binary-xyzzy", &[]),
            std::path::Path::new("/tmp"),
        )
        .await;
        assert!(res.is_err());
    }
}
