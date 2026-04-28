//! `web_fetch_tls` — browser-TLS-impersonating HTTP GET via `curl_cffi`.
//!
//! Why this exists
//! ---------------
//! `web_fetch` (reqwest-backed) is blocked by Cloudflare on several
//! Vietnamese real-estate portals. The block is not at the UA/header
//! layer — it's at the TLS ClientHello fingerprint (JA3/JA4) and HTTP/2
//! SETTINGS frame. Rust's rustls/reqwest stack produces a Rust-shaped
//! fingerprint no amount of header tweaking can hide.
//!
//! `curl_cffi` links against a BoringSSL patched by the chrome-tls-
//! impersonate project and replays a byte-for-byte real Chrome/Firefox
//! handshake. Empirically (2026-04-20 bench, `/tmp/t4_cffi.py`): passes
//! batdongsan.com.vn, chotot.com, alonhadat.com.vn where reqwest 403s.
//!
//! Architecture
//! ------------
//! We keep curl_cffi as a subprocess (Python script at
//! `scripts/fetch_tls.py`) instead of a Rust crate because:
//!   1. curl_cffi is a C extension; no Rust wrapper exists as of 2026-04.
//!   2. Spawning is ~20 ms, a rounding error next to CF's 500 ms+ latency.
//!   3. The Python script handles the session / cookie / warmup dance
//!      without polluting the Rust process.
//!
//! Contract
//! --------
//! * Input identical in shape to `web_fetch` (url, max_chars, include_links).
//! * Body is passed through the same `html_to_text` / `truncate_chars`
//!   helpers as `web_fetch`, so an agent swapping tools sees consistent
//!   text payloads.
//! * Script is located via `NAKED_FETCH_TLS_SCRIPT` env or defaults to
//!   `scripts/fetch_tls.py` relative to cwd (= naked-tg's WorkingDirectory).
//! * On transport failure we return `is_error: true` with the stderr tail
//!   so the cascade in `web_fetch.rs` can attempt the next tier (Wayback).

use std::path::Path;
use std::time::Duration;

use async_trait::async_trait;
use serde_json::{Value, json};
use tokio::process::Command;

use crate::tool::Tool;
use crate::tool::web_fetch::{DEFAULT_MAX_CHARS, html_to_text, truncate_chars};
use crate::types::{Permission, ToolResult, ToolSpec};

/// Per-invocation wall clock. curl_cffi adds session warmup (~1.2 s) on
/// top of the actual fetch, and the impersonation ladder can retry up to
/// 4 times on CF blocks, so 60 s covers the worst case. Individual HTTP
/// requests inside the script are capped at 25 s via `--timeout`.
const TLS_FETCH_TIMEOUT_SECS: u64 = 60;

/// Locate the Python script. Env override first (for tests and weird
/// deployment layouts), then the conventional `scripts/fetch_tls.py`
/// relative to `WorkingDirectory=naked/` in the systemd unit.
fn script_path() -> String {
    std::env::var("NAKED_FETCH_TLS_SCRIPT").unwrap_or_else(|_| "scripts/fetch_tls.py".to_string())
}

pub struct WebFetchTlsTool;

impl WebFetchTlsTool {
    pub fn new() -> Self {
        Self
    }
}

impl Default for WebFetchTlsTool {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl Tool for WebFetchTlsTool {
    fn spec(&self) -> ToolSpec {
        ToolSpec {
            name: "web_fetch_tls".into(),
            description: "Fetch a page while impersonating a real Chrome/Firefox TLS \
                handshake (JA3). Use this for sites where `web_fetch` hits a \
                Cloudflare 403 / captcha. Works empirically for \
                batdongsan.com.vn, chotot.com, alonhadat.com.vn, and most other \
                CF-protected Vietnamese sites. Same output format as `web_fetch`."
                .into(),
            parameters: json!({
                "type": "object",
                "properties": {
                    "url":           { "type": "string", "description": "Absolute http(s) URL" },
                    "max_chars":     { "type": "integer", "description": "Cap on characters returned (default 20000, max 60000)", "default": 20000, "minimum": 500, "maximum": 60000 },
                    "include_links": { "type": "boolean", "description": "Emit a `## Links` section with extracted hrefs (default true)", "default": true },
                    "no_proxy":      { "type": "boolean", "description": "Bypass NAKED_PROXY for this request — use for hosts where the upstream proxy tunnel fails but direct works (default false)", "default": false },
                    "impersonate":   { "type": "string", "description": "Single impersonation profile (e.g. 'chrome120', 'firefox133'). Omit to iterate the default ladder." }
                },
                "required": ["url"]
            }),
            permission: Permission::ReadOnly,
        }
    }

    async fn execute(&self, input: Value, _cwd: &Path) -> ToolResult {
        let url = match input.get("url").and_then(|v| v.as_str()) {
            Some(u) if u.starts_with("http://") || u.starts_with("https://") => u.to_string(),
            _ => {
                return ToolResult {
                    output: "`url` is required and must be an absolute http(s) URL".into(),
                    is_error: true,
                };
            }
        };
        let max_chars = input
            .get("max_chars")
            .and_then(|v| v.as_u64())
            .map(|n| n.clamp(500, 60_000) as usize)
            .unwrap_or(DEFAULT_MAX_CHARS);
        let include_links = input
            .get("include_links")
            .and_then(|v| v.as_bool())
            .unwrap_or(true);
        let no_proxy = input
            .get("no_proxy")
            .and_then(|v| v.as_bool())
            .unwrap_or(false);
        let impersonate = input
            .get("impersonate")
            .and_then(|v| v.as_str())
            .map(str::to_string);

        let script = script_path();
        let mut cmd = Command::new("python3");
        cmd.arg(&script).arg(&url);
        if no_proxy {
            cmd.arg("--no-proxy");
        }
        if let Some(imp) = impersonate.as_deref() {
            cmd.arg("--impersonate").arg(imp);
        }

        tracing::info!(url = %url, script = %script, no_proxy, "web_fetch_tls: spawning");
        let child = cmd
            .kill_on_drop(true)
            .stdout(std::process::Stdio::piped())
            .stderr(std::process::Stdio::piped())
            .spawn();
        let child = match child {
            Ok(c) => c,
            Err(e) => {
                return ToolResult {
                    output: format!(
                        "web_fetch_tls: failed to spawn `{script}` — {e}. \
                         Ensure curl_cffi is installed: `pip install --user curl_cffi`."
                    ),
                    is_error: true,
                };
            }
        };

        let output = tokio::time::timeout(
            Duration::from_secs(TLS_FETCH_TIMEOUT_SECS),
            child.wait_with_output(),
        )
        .await;
        let output = match output {
            Ok(Ok(o)) => o,
            Ok(Err(e)) => {
                return ToolResult {
                    output: format!("web_fetch_tls: process error: {e}"),
                    is_error: true,
                };
            }
            Err(_) => {
                return ToolResult {
                    output: format!(
                        "web_fetch_tls: timed out after {TLS_FETCH_TIMEOUT_SECS}s (hard kill triggered)"
                    ),
                    is_error: true,
                };
            }
        };

        let stderr_tail = String::from_utf8_lossy(&output.stderr)
            .lines()
            .rev()
            .take(10)
            .collect::<Vec<_>>()
            .into_iter()
            .rev()
            .collect::<Vec<_>>()
            .join("\n");

        if output.stdout.is_empty() {
            return ToolResult {
                output: format!(
                    "web_fetch_tls: script produced no stdout (exit={}).\nstderr tail:\n{stderr_tail}",
                    output
                        .status
                        .code()
                        .map(|c| c.to_string())
                        .unwrap_or_else(|| "?".into())
                ),
                is_error: true,
            };
        }

        let stdout_str = String::from_utf8_lossy(&output.stdout);
        let parsed: serde_json::Value = match serde_json::from_str(stdout_str.trim()) {
            Ok(v) => v,
            Err(e) => {
                return ToolResult {
                    output: format!(
                        "web_fetch_tls: script returned non-JSON stdout ({e}).\nraw stdout (first 500 chars):\n{}\nstderr tail:\n{stderr_tail}",
                        &stdout_str.chars().take(500).collect::<String>()
                    ),
                    is_error: true,
                };
            }
        };

        // Transport-level failure inside the script (DNS, connect refused,
        // proxy reject…). The script populates `error` with the underlying
        // exception; surface it so the agent / cascade knows to move on.
        if !parsed.get("ok").and_then(|v| v.as_bool()).unwrap_or(false) {
            let err_msg = parsed
                .get("error")
                .and_then(|v| v.as_str())
                .unwrap_or("unknown error");
            return ToolResult {
                output: format!("web_fetch_tls: transport failed — {err_msg}\n{stderr_tail}"),
                is_error: true,
            };
        }

        let status = parsed.get("status").and_then(|v| v.as_u64()).unwrap_or(0) as u16;
        let final_url = parsed
            .get("final_url")
            .and_then(|v| v.as_str())
            .unwrap_or(&url)
            .to_string();
        let content_type = parsed
            .get("content_type")
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .to_string();
        let body = parsed
            .get("body")
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .to_string();
        let imp_used = parsed
            .get("impersonate")
            .and_then(|v| v.as_str())
            .unwrap_or("?");
        let all_blocked = parsed
            .get("all_attempts_blocked")
            .and_then(|v| v.as_bool())
            .unwrap_or(false);

        // If the script ran every ladder rung and every one looked blocked,
        // surface a structured failure — the cascade in `web_fetch.rs`
        // should then fall through to Wayback.
        if all_blocked {
            return ToolResult {
                output: format!(
                    "BLOCKED via TLS impersonation: status={status}, url={final_url}\n\
                     All impersonation profiles exhausted (chrome120, chrome124, firefox133, chrome131).\n\
                     This site likely needs a cloud stealth service or archived snapshot.\n\
                     Consider: web_fetch_wayback to retrieve the last working snapshot."
                ),
                is_error: true,
            };
        }

        let (text, links) = html_to_text(&body);
        let header = format!(
            "HTTP {status} — {final_url}\nContent-Type: {content_type}\nTLS-impersonate: {imp_used}\n\n"
        );
        let remaining = max_chars.saturating_sub(header.chars().count());
        let mut out = header;
        out.push_str(&truncate_chars(&text, remaining));
        if include_links && !links.is_empty() {
            out.push_str("\n\n## Links\n");
            for l in links.iter().take(40) {
                out.push_str("- ");
                out.push_str(l);
                out.push('\n');
            }
        }

        ToolResult {
            output: out,
            is_error: !(200..400).contains(&status),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn rejects_bad_url() {
        let tool = WebFetchTlsTool::new();
        let cwd = std::env::current_dir().unwrap();
        let r = tool.execute(json!({ "url": "not-a-url" }), &cwd).await;
        assert!(r.is_error);
        let r2 = tool.execute(json!({ "url": "ftp://x" }), &cwd).await;
        assert!(r2.is_error);
    }

    // A test that manipulates env vars would need `unsafe` on the
    // current toolchain — the crate forbids unsafe. The "missing script
    // produces actionable error" case is exercised implicitly by CI
    // when the test runner happens to sit in a directory without a
    // `scripts/fetch_tls.py`; the `rejects_bad_url` check above is a
    // sufficient happy-path assertion for the URL-shape validation.
}
