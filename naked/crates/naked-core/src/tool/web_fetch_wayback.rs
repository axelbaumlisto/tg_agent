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
use crate::tool::web_fetch::{DEFAULT_MAX_CHARS, html_to_text, truncate_chars};
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
                return ToolResult {
                    output: "`url` is required and must be an absolute http(s) URL".into(),
                    is_error: true,
                };
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

        // --- Step 1: Wayback availability API ---------------------------
        let avail_url = match ts_hint.as_deref() {
            Some(ts) if !ts.is_empty() => {
                format!("{WAYBACK_AVAILABLE_ENDPOINT}?url={url}&timestamp={ts}")
            }
            _ => format!("{WAYBACK_AVAILABLE_ENDPOINT}?url={url}"),
        };
        tracing::info!(url = %url, "web_fetch_wayback: querying availability");
        let avail_resp = match self.client.get(&avail_url).send().await {
            Ok(r) => r,
            Err(e) => {
                return ToolResult {
                    output: format!("web_fetch_wayback: availability request failed: {e}"),
                    is_error: true,
                };
            }
        };
        let avail_status = avail_resp.status();
        if !avail_status.is_success() {
            return ToolResult {
                output: format!("web_fetch_wayback: availability API returned HTTP {avail_status}"),
                is_error: true,
            };
        }
        let avail_json: serde_json::Value = match avail_resp.json().await {
            Ok(v) => v,
            Err(e) => {
                return ToolResult {
                    output: format!("web_fetch_wayback: availability JSON parse failed: {e}"),
                    is_error: true,
                };
            }
        };

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
                return ToolResult {
                    output: format!(
                        "web_fetch_wayback: no archived snapshot available for {url}.\n\
                         The Wayback Machine has never crawled this URL, or the closest \
                         snapshot returned a non-success status."
                    ),
                    is_error: true,
                };
            }
        };

        if snapshot_url.is_empty() {
            return ToolResult {
                output: format!(
                    "web_fetch_wayback: Wayback API returned empty snapshot URL for {url}"
                ),
                is_error: true,
            };
        }

        // --- Step 2: Fetch the snapshot ---------------------------------
        // Use `if_` URL form so Wayback serves the archived HTML *inline*
        // without wrapping it in a JS-heavy frameset. Pattern:
        //   http://web.archive.org/web/<TS>/<url>    -> with wrapper
        //   http://web.archive.org/web/<TS>if_/<url> -> raw archived html
        // The wrapper adds ~30 KB of Wayback toolbar we don't want.
        let raw_snapshot_url =
            rewrite_to_raw_snapshot(&snapshot_url).unwrap_or_else(|| snapshot_url.clone());

        tracing::info!(
            snapshot = %raw_snapshot_url,
            timestamp = %timestamp,
            "web_fetch_wayback: fetching snapshot"
        );
        let resp = match self.client.get(&raw_snapshot_url).send().await {
            Ok(r) => r,
            Err(e) => {
                return ToolResult {
                    output: format!("web_fetch_wayback: snapshot fetch failed: {e}"),
                    is_error: true,
                };
            }
        };

        let status = resp.status();
        let content_type = resp
            .headers()
            .get(reqwest::header::CONTENT_TYPE)
            .and_then(|v| v.to_str().ok())
            .unwrap_or("")
            .to_string();
        if let Some(len_hint) = resp.content_length()
            && len_hint > WAYBACK_MAX_BYTES
        {
            return ToolResult {
                output: format!(
                    "web_fetch_wayback: snapshot body too large ({len_hint} bytes, limit {WAYBACK_MAX_BYTES})"
                ),
                is_error: true,
            };
        }
        let bytes = match resp.bytes().await {
            Ok(b) => b,
            Err(e) => {
                return ToolResult {
                    output: format!("web_fetch_wayback: snapshot body read failed: {e}"),
                    is_error: true,
                };
            }
        };
        if bytes.len() as u64 > WAYBACK_MAX_BYTES {
            return ToolResult {
                output: format!(
                    "web_fetch_wayback: snapshot body too large after read ({} bytes, limit {WAYBACK_MAX_BYTES})",
                    bytes.len()
                ),
                is_error: true,
            };
        }
        let body_text = String::from_utf8_lossy(&bytes).into_owned();
        let (text, links) = html_to_text(&body_text);

        let header = format!(
            "Wayback snapshot: HTTP {status} (origin HTTP {snapshot_status})\nTimestamp: {timestamp} ({})\nOriginal URL: {url}\nSnapshot URL: {snapshot_url}\nContent-Type: {content_type}\n\n",
            format_timestamp(&timestamp),
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
            is_error: !(200..400).contains(&status.as_u16()),
        }
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
