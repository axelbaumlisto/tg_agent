//! `web_fetch_wayback` — retrieve the most recent archived snapshot of a URL
//! from the Internet Archive's Wayback Machine.
//!
//! When to use
//! -----------
//! Use this when `web_fetch` AND `web_fetch_tls` both return BLOCKED on a
//! site — it's the "give me *something*" tier. Empirically (2026-04-20
//! bench) the Wayback has usable snapshots for:
//!   * nhatot.com         (refreshed roughly monthly, ok for price/count trends)
//!   * batdongsan.com.vn  (monthly, stale but listings survive)
//!   * dotproperty.com.vn (sparse)
//!
//! Snapshots ARE stale (typically 2-8 weeks) — the caller/agent must
//! clearly label any extracted data with the `timestamp` returned here.
//!
//! API
//! ---
//! Two-step round trip:
//!   1. GET `https://archive.org/wayback/available?url=<target>&timestamp=<YYYYMMDD?>`
//!      → JSON `{archived_snapshots.closest.{url,timestamp,status,available}}`
//!   2. GET that snapshot URL with `User-Agent: Mozilla/…`. Wayback serves
//!      vanilla HTTPS, no CF, so reqwest works fine (no proxy needed).
//!
//! We explicitly DO NOT route through `NAKED_PROXY`: residential/SOCKS5
//! proxies often add 500 ms+ of latency to archive.org for no gain, and
//! archive.org rate-limits proxy IPs more aggressively.

use std::path::Path;
use std::time::Duration;

use async_trait::async_trait;
use serde_json::{Value, json};

use crate::tool::Tool;
use crate::tool::fetch_common::{self, DEFAULT_MAX_CHARS};
use crate::types::{Permission, ToolResult, ToolSpec};

const WAYBACK_AVAILABLE_ENDPOINT: &str = "https://archive.org/wayback/available";
const WAYBACK_TIMEOUT_SECS: u64 = 30;
/// archive.org's Wayback HTML wrapper prepends a toolbar frame and a lot
/// of injected JS (~50 KB). The raw archived content follows — 3 MB
/// upper bound catches the largest real listing pages we'd want.
const WAYBACK_MAX_BYTES: u64 = 3 * 1024 * 1024;

pub struct WebFetchWaybackTool {
    client: reqwest::Client,
}

impl WebFetchWaybackTool {
    pub fn new() -> Self {
        // Fresh client with NO proxy — see module docs for rationale.
        let client = reqwest::Client::builder()
            .user_agent(concat!(
                "Mozilla/5.0 (X11; Linux x86_64) ",
                "AppleWebKit/537.36 (KHTML, like Gecko) ",
                "Chrome/120.0.0.0 Safari/537.36 ",
                "naked-tg/web_fetch_wayback"
            ))
            .timeout(Duration::from_secs(WAYBACK_TIMEOUT_SECS))
            .redirect(reqwest::redirect::Policy::limited(5))
            .build()
            .expect("web_fetch_wayback: reqwest client init");
        Self { client }
    }
}

impl Default for WebFetchWaybackTool {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl Tool for WebFetchWaybackTool {
    fn spec(&self) -> ToolSpec {
        ToolSpec {
            name: "web_fetch_wayback".into(),
            description: "Retrieve the most recent archived snapshot of a URL from the \
                Internet Archive Wayback Machine. Use as a last-resort fallback when \
                `web_fetch` AND `web_fetch_tls` both report BLOCKED. Snapshots ARE \
                STALE (weeks/months old) — always cite the returned timestamp when \
                using the data."
                .into(),
            parameters: json!({
                "type": "object",
                "properties": {
                    "url":       { "type": "string", "description": "Absolute http(s) URL of the original page" },
                    "timestamp": { "type": "string", "description": "Optional 'closest-to' timestamp in YYYY[MM[DD[hhmmss]]] form. Omit for most recent." },
                    "max_chars": { "type": "integer", "description": "Cap on characters returned (default 20000, max 60000)", "default": 20000, "minimum": 500, "maximum": 60000 },
                    "include_links": { "type": "boolean", "description": "Emit a `## Links` section (default true)", "default": true }
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
                return ToolResult::err("`url` is required and must be an absolute http(s) URL");
            }
        };
        let ts_hint = input
            .get("timestamp")
            .and_then(|v| v.as_str())
            .map(str::to_string);
        let max_chars = input
            .get("max_chars")
            .and_then(|v| v.as_u64())
            .map(|n| n.clamp(500, 60_000) as usize)
            .unwrap_or(DEFAULT_MAX_CHARS);
        let include_links = input
            .get("include_links")
            .and_then(|v| v.as_bool())
            .unwrap_or(true);

        let snap = match self.resolve_and_fetch(&url, ts_hint.as_deref()).await {
            Ok(s) => s,
            Err(e) => return e,
        };
        let WaybackSnapshot {
            body: body_text,
            timestamp,
            snapshot_url,
            snapshot_status,
            content_type,
            status,
        } = snap;
        let header = format!(
            "Wayback snapshot: HTTP {status} (origin HTTP {snapshot_status})\nTimestamp: {timestamp} ({})\nOriginal URL: {url}\nSnapshot URL: {snapshot_url}\nContent-Type: {content_type}\n\n",
            format_timestamp(&timestamp),
        );
        fetch_common::format_fetch_output(
            &body_text,
            &header,
            include_links,
            max_chars,
            (200..400).contains(&status.as_u16()),
        )
    }
}

/// Rewrite `http://web.archive.org/web/<TS>/<url>` to the `<TS>if_` form
/// which returns the raw archived HTML without Wayback's toolbar iframe.
/// Returns `None` if the input doesn't match the expected pattern — the
/// caller then uses the original URL (worst case: 30 KB of Wayback chrome
/// at the top of the body).
fn rewrite_to_raw_snapshot(snapshot_url: &str) -> Option<String> {
    const MARKER: &str = "/web/";
    let (prefix, rest) = snapshot_url.split_once(MARKER)?;
    // `rest` starts with `<TIMESTAMP>/<original_url>`.
    let (ts, orig) = rest.split_once('/')?;
    // Don't double-rewrite if already `if_`-suffixed.
    if ts.ends_with("if_") {
        return None;
    }
    Some(format!("{prefix}{MARKER}{ts}if_/{orig}"))
}

/// Turn `20260325202740` into `2026-03-25 20:27:40 UTC` — the agent reads
/// this header when deciding whether a snapshot is fresh enough to cite.
fn format_timestamp(ts: &str) -> String {
    if ts.len() < 8 {
        return ts.to_string();
    }
    let y = &ts[0..4];
    let m = ts.get(4..6).unwrap_or("01");
    let d = ts.get(6..8).unwrap_or("01");
    let hh = ts.get(8..10).unwrap_or("00");
    let mm = ts.get(10..12).unwrap_or("00");
    let ss = ts.get(12..14).unwrap_or("00");
    format!("{y}-{m}-{d} {hh}:{mm}:{ss} UTC")
}

/// Resolved Wayback snapshot data.
struct WaybackSnapshot {
    body: String,
    timestamp: String,
    snapshot_url: String,
    snapshot_status: String,
    content_type: String,
    status: reqwest::StatusCode,
}

impl WebFetchWaybackTool {
    /// Resolve Wayback availability and fetch the snapshot body.
    async fn resolve_and_fetch(
        &self,
        url: &str,
        ts_hint: Option<&str>,
    ) -> Result<WaybackSnapshot, ToolResult> {
        let avail_url = match ts_hint {
            Some(ts) if !ts.is_empty() => {
                format!("{WAYBACK_AVAILABLE_ENDPOINT}?url={url}&timestamp={ts}")
            }
            _ => format!("{WAYBACK_AVAILABLE_ENDPOINT}?url={url}"),
        };
        tracing::info!(url = %url, "web_fetch_wayback: querying availability");
        let avail_resp = self.client.get(&avail_url).send().await.map_err(|e| {
            ToolResult::err(format!(
                "web_fetch_wayback: availability request failed: {e}"
            ))
        })?;
        if !avail_resp.status().is_success() {
            return Err(ToolResult::err(format!(
                "web_fetch_wayback: availability API returned HTTP {}",
                avail_resp.status()
            )));
        }
        let avail_json: serde_json::Value = avail_resp.json().await.map_err(|e| {
            ToolResult::err(format!(
                "web_fetch_wayback: availability JSON parse failed: {e}"
            ))
        })?;

        let closest = avail_json
            .get("archived_snapshots")
            .and_then(|v| v.get("closest"));
        let (snapshot_url, timestamp, snapshot_status) = match closest {
            Some(c)
                if c.get("available")
                    .and_then(|v| v.as_bool())
                    .unwrap_or(false) =>
            {
                let u = c
                    .get("url")
                    .and_then(|v| v.as_str())
                    .unwrap_or("")
                    .to_string();
                let ts = c
                    .get("timestamp")
                    .and_then(|v| v.as_str())
                    .unwrap_or("")
                    .to_string();
                let st = c
                    .get("status")
                    .and_then(|v| v.as_str())
                    .unwrap_or("")
                    .to_string();
                (u, ts, st)
            }
            _ => {
                return Err(ToolResult::err(format!(
                    "web_fetch_wayback: no archived snapshot available for {url}"
                )));
            }
        };
        if snapshot_url.is_empty() {
            return Err(ToolResult::err(format!(
                "web_fetch_wayback: empty snapshot URL for {url}"
            )));
        }

        let raw_url =
            rewrite_to_raw_snapshot(&snapshot_url).unwrap_or_else(|| snapshot_url.clone());
        tracing::info!(snapshot = %raw_url, timestamp = %timestamp, "fetching snapshot");
        let resp = self.client.get(&raw_url).send().await.map_err(|e| {
            ToolResult::err(format!("web_fetch_wayback: snapshot fetch failed: {e}"))
        })?;

        let status = resp.status();
        let content_type = resp
            .headers()
            .get(reqwest::header::CONTENT_TYPE)
            .and_then(|v| v.to_str().ok())
            .unwrap_or("")
            .to_string();
        if let Some(len) = resp.content_length()
            && len > WAYBACK_MAX_BYTES
        {
            return Err(ToolResult::err(format!(
                "web_fetch_wayback: too large ({len} bytes, limit {WAYBACK_MAX_BYTES})"
            )));
        }
        let bytes = resp
            .bytes()
            .await
            .map_err(|e| ToolResult::err(format!("web_fetch_wayback: body read failed: {e}")))?;
        if bytes.len() as u64 > WAYBACK_MAX_BYTES {
            return Err(ToolResult::err(format!(
                "web_fetch_wayback: body too large ({} bytes)",
                bytes.len()
            )));
        }

        Ok(WaybackSnapshot {
            body: String::from_utf8_lossy(&bytes).into_owned(),
            timestamp,
            snapshot_url,
            snapshot_status,
            content_type,
            status,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rewrite_adds_if_suffix() {
        let raw = rewrite_to_raw_snapshot(
            "http://web.archive.org/web/20260325202740/https://www.nhatot.com/mua-ban",
        );
        assert_eq!(
            raw.as_deref(),
            Some("http://web.archive.org/web/20260325202740if_/https://www.nhatot.com/mua-ban")
        );
    }

    #[test]
    fn rewrite_skips_already_raw() {
        assert!(
            rewrite_to_raw_snapshot(
                "http://web.archive.org/web/20260325202740if_/https://www.nhatot.com/",
            )
            .is_none()
        );
    }

    #[test]
    fn format_timestamp_renders_human_readable() {
        assert_eq!(
            format_timestamp("20260325202740"),
            "2026-03-25 20:27:40 UTC"
        );
        assert_eq!(format_timestamp("20260101"), "2026-01-01 00:00:00 UTC");
        assert_eq!(format_timestamp("bad"), "bad");
    }

    #[tokio::test]
    async fn rejects_bad_url() {
        let tool = WebFetchWaybackTool::new();
        let cwd = std::env::current_dir().unwrap();
        let r = tool.execute(json!({ "url": "not-a-url" }), &cwd).await;
        assert!(r.is_error);
    }
}
