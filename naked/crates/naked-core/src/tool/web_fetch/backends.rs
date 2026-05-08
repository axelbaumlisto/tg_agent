//! Third-party scrape backends used by the fallback cascade.
//!
//! - `fetch_via_tls_subprocess` — Tier-3: spawn `scripts/fetch_tls.py`
//!   (curl_cffi JA3 impersonation) and parse its JSON result.
//! - `fetch_wayback_snapshot` — Tier-4: fetch the most-recent Wayback
//!   Machine snapshot for a URL.
//! - `format_wayback_ts` — format a Wayback timestamp string.

use std::time::Duration;

use tokio::process::Command;

use super::http::FetchOutcome;

/// Spawn `scripts/fetch_tls.py` and unpack the JSON result into a
/// `FetchOutcome`-shaped record. Kept here (not in `web_fetch_tls.rs`)
/// so the cascade inside `WebFetchTool::execute` can call it without
/// cross-crate tool dispatch. The standalone `WebFetchTlsTool` remains
/// the explicit path — identical subprocess contract.
pub(super) async fn fetch_via_tls_subprocess(url: &str) -> Result<FetchOutcome, String> {
    let script =
        std::env::var("NAKED_FETCH_TLS_SCRIPT").unwrap_or_else(|_| "scripts/fetch_tls.py".into());
    // 55 s < WebFetchTlsTool's 60 s cap so we don't race its kill.
    let out = tokio::time::timeout(
        Duration::from_secs(55),
        Command::new("python3")
            .arg(&script)
            .arg(url)
            .kill_on_drop(true)
            .output(),
    )
    .await
    .map_err(|_| "tls subprocess timeout after 55s".to_string())?
    .map_err(|e| format!("tls spawn: {e}"))?;

    if out.stdout.is_empty() {
        let tail = String::from_utf8_lossy(&out.stderr)
            .lines()
            .rev()
            .take(3)
            .collect::<Vec<_>>()
            .into_iter()
            .rev()
            .collect::<Vec<_>>()
            .join(" | ");
        return Err(format!("tls no stdout (stderr: {tail})"));
    }

    let parsed: serde_json::Value =
        serde_json::from_slice(&out.stdout).map_err(|e| format!("tls json parse: {e}"))?;

    if !parsed.get("ok").and_then(|v| v.as_bool()).unwrap_or(false) {
        let err_msg = parsed
            .get("error")
            .and_then(|v| v.as_str())
            .unwrap_or("unknown");
        return Err(format!("tls transport: {err_msg}"));
    }

    Ok(FetchOutcome {
        status: parsed.get("status").and_then(|v| v.as_u64()).unwrap_or(0) as u16,
        final_url: parsed
            .get("final_url")
            .and_then(|v| v.as_str())
            .unwrap_or(url)
            .to_string(),
        content_type: parsed
            .get("content_type")
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .to_string(),
        text: parsed
            .get("body")
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .to_string(),
        transport_error: None,
    })
}

/// Internal Wayback fetcher for the `web_fetch` cascade (Tier-4). Returns
/// `(body_html, timestamp, snapshot_url)` on success. The standalone
/// `WebFetchWaybackTool` remains the explicit path for when the agent
/// *wants* archive data directly.
pub(super) async fn fetch_wayback_snapshot(
    client: &reqwest::Client,
    url: &str,
) -> Result<(String, String, String), String> {
    let avail = format!("https://archive.org/wayback/available?url={url}");
    let resp = client
        .get(&avail)
        .send()
        .await
        .map_err(|e| format!("availability: {e}"))?;
    if !resp.status().is_success() {
        return Err(format!("availability HTTP {}", resp.status()));
    }
    let json: serde_json::Value = resp
        .json()
        .await
        .map_err(|e| format!("availability json: {e}"))?;
    let closest = json
        .get("archived_snapshots")
        .and_then(|v| v.get("closest"));
    let (snapshot_url, timestamp) = match closest {
        Some(c)
            if c.get("available")
                .and_then(|v| v.as_bool())
                .unwrap_or(false) =>
        {
            (
                c.get("url")
                    .and_then(|v| v.as_str())
                    .unwrap_or("")
                    .to_string(),
                c.get("timestamp")
                    .and_then(|v| v.as_str())
                    .unwrap_or("")
                    .to_string(),
            )
        }
        _ => return Err("no snapshot available".into()),
    };
    if snapshot_url.is_empty() {
        return Err("empty snapshot url".into());
    }
    // Rewrite to `if_` form to skip the Wayback toolbar wrapper.
    let raw_url = snapshot_url
        .split_once("/web/")
        .and_then(|(prefix, rest)| {
            let (ts, orig) = rest.split_once('/')?;
            if ts.ends_with("if_") {
                return None;
            }
            Some(format!("{prefix}/web/{ts}if_/{orig}"))
        })
        .unwrap_or_else(|| snapshot_url.clone());

    let snap = client
        .get(&raw_url)
        .send()
        .await
        .map_err(|e| format!("snapshot: {e}"))?;
    if !snap.status().is_success() {
        return Err(format!("snapshot HTTP {}", snap.status()));
    }
    let body = snap
        .text()
        .await
        .map_err(|e| format!("snapshot body: {e}"))?;
    Ok((body, timestamp, snapshot_url))
}

/// Human-readable Wayback timestamp for header lines.
/// Local copy (not `pub use`'d from web_fetch_wayback) because the cascade
/// is hot-path code that shouldn't depend on tool ordering at module load.
pub(super) fn format_wayback_ts(ts: &str) -> String {
    if ts.len() < 8 {
        return ts.to_string();
    }
    let y = &ts[0..4];
    let m = ts.get(4..6).unwrap_or("01");
    let d = ts.get(6..8).unwrap_or("01");
    format!("{y}-{m}-{d}")
}
