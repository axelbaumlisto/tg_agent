//! `web_fetch` — reqwest-backed HTTP GET with HTML-to-text extraction.
//!
//! Tradeoffs vs. the browser MCP:
//! - Pro: zero external dependencies at runtime, starts in microseconds, no
//!   Chromium footprint. Perfect for SSR marketplaces (chotot, batdongsan,
//!   auto.ru) where listings are in the initial HTML response.
//! - Con: can't execute JS. For SPA sites the agent should fall back to the
//!   Playwright MCP (`browser_navigate` + `browser_snapshot`).
//!
//! HTML → text conversion is deliberately minimal: strip `<script>`/`<style>`
//! blocks, then drop the rest of the tags. Not a real DOM parser, but good
//! enough to feed a language model 2-4 KB of usable text per page. Inline
//! links inside `<a>` tags are preserved on a separate line so the agent can
//! `research_save` them without re-fetching.

use std::path::Path;
use std::sync::Arc;

use async_trait::async_trait;
use serde_json::{Value, json};

use crate::scrape::host_policy::{HostPolicy, Outcome as TierOutcome, Tier};
use crate::scrape::multi::MultiCloudScraper;
use crate::tool::Tool;
use crate::types::{Permission, ToolResult, ToolSpec};

mod backends;
mod cascade;
pub mod error;
pub mod extract;
mod http;

pub use error::{BlockKind, detect_block};
/// Re-export `html_to_text` at the module level so `fetch_common` and sibling
/// tools (`web_fetch_tls`, `web_fetch_wayback`) can keep their existing paths.
pub(crate) use extract::html_to_text;

/// Hard ceiling on response bytes pulled through `web_fetch`. Prevents a
/// misbehaving site from forcing us to load a 500 MB asset before the timeout
/// trips. 2 MB covers every reasonable HTML page.
pub(super) const DEFAULT_MAX_BYTES: u64 = 2 * 1024 * 1024;
/// Maximum chars the tool returns — keeps a single fetch from spending the
/// full context window. Exposed `pub(crate)` so sibling fetch tools
/// (web_fetch_tls, web_fetch_wayback) return the same-sized payloads
/// and the agent's context budgeting is consistent across backends.
pub(crate) const DEFAULT_MAX_CHARS: usize = 20_000;
/// Hard per-request wall clock. Aligned with typical MCP timeouts.
pub(super) const DEFAULT_TIMEOUT_SECS: u64 = 30;

pub struct WebFetchTool {
    pub(super) client: reqwest::Client,
    /// Optional cloud-scrape cascade (ScrapingBee + Firecrawl). Inserted as
    /// Tier 3.5 between TLS impersonation and Wayback. `None` when no keys
    /// are configured — the cascade simply skips that tier.
    pub(super) cloud: Option<Arc<MultiCloudScraper>>,
    /// Adaptive per-host tier selector. Tracks which tiers have been
    /// blocked for which hosts and skips them on the next fetch — saves
    /// ~10 s per request on sites where reqwest/url-prefix are guaranteed
    /// to fail (CF-protected portals).
    pub(super) host_policy: Arc<HostPolicy>,
}

impl WebFetchTool {
    pub fn new() -> Self {
        Self::with_cloud_scraper(None)
    }

    /// Construct with a cloud-scrape cascade for Tier 3.5. Pass `None` to
    /// preserve the legacy behaviour (skip the tier).
    pub fn with_cloud_scraper(cloud: Option<Arc<MultiCloudScraper>>) -> Self {
        Self::with_components(cloud, Arc::new(HostPolicy::new()))
    }
}

impl Default for WebFetchTool {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl Tool for WebFetchTool {
    fn spec(&self) -> ToolSpec {
        ToolSpec {
            name: "web_fetch".into(),
            description: "Fetch a web page via HTTP GET and return cleaned text. \
                Good for server-rendered pages (news, marketplaces, docs). For \
                JavaScript-heavy sites, use the browser MCP instead. Up to 20 000 \
                characters returned; larger pages are truncated."
                .into(),
            parameters: json!({
                "type": "object",
                "properties": {
                    "url":           { "type": "string", "description": "Absolute http(s) URL" },
                    "max_chars":     { "type": "integer", "description": "Cap on characters returned (default 20000, max 60000)", "default": 20000, "minimum": 500, "maximum": 60000 },
                    "include_links": { "type": "boolean", "description": "Emit a `## Links` section with extracted href values (default true)", "default": true },
                    "skip_fallback": { "type": "boolean", "description": "Do NOT auto-escalate to web_fetch_tls/web_fetch_wayback on failure. Use when you specifically want the raw reqwest behaviour (e.g. for testing).", "default": false }
                },
                "required": ["url"]
            }),
            permission: Permission::ReadOnly,
        }
    }

    async fn execute(&self, input: Value, _cwd: &Path) -> ToolResult {
        let url = match input.get("url").and_then(|v| v.as_str()) {
            Some(u) if u.starts_with("http://") || u.starts_with("https://") => u.to_string(),
            _ => return ToolResult::err("`url` is required and must be an absolute http(s) URL"),
        };

        // Network policy check:
        if let Some(host) = crate::network_policy::host_from_url(&url) {
            let policy = crate::network_policy::NetworkPolicy::default();
            if policy.check(&host) == crate::network_policy::NetDecision::Deny {
                return ToolResult::err(format!("Blocked by network policy: {host}"));
            }
        }
        let max_chars = input
            .get("max_chars")
            .and_then(|v| v.as_u64())
            .map(|n| n.clamp(500, 60_000) as usize)
            .unwrap_or(DEFAULT_MAX_CHARS);
        let include_links = input
            .get("include_links")
            .and_then(|v| v.as_bool())
            .unwrap_or(true);
        let skip_fallback = input
            .get("skip_fallback")
            .and_then(|v| v.as_bool())
            .unwrap_or(false);

        // Adaptive start tier. When `skip_fallback` is set, the caller
        // explicitly wants the legacy reqwest-only behaviour, so we
        // ignore the host policy entirely.
        let start_tier = if skip_fallback {
            Tier::Reqwest
        } else {
            self.host_policy.recommended_start_tier(&url)
        };
        let mut cascade_notes: Vec<String> = Vec::new();
        if start_tier > Tier::Reqwest {
            cascade_notes.push(format!(
                "host-policy: skip → start at {}",
                start_tier.name()
            ));
        }

        // --- Tier-1: direct reqwest ---------------------------------------
        let mut primary = if start_tier <= Tier::Reqwest {
            let res = match self.fetch_once(&url).await {
                Ok(f) => f,
                Err(msg) => http::FetchOutcome {
                    status: 0,
                    final_url: url.clone(),
                    content_type: String::new(),
                    text: String::new(),
                    transport_error: Some(msg),
                },
            };
            let blocked = res.transport_error.is_some()
                || detect_block(res.status, &res.text).is_some()
                || !(200..400).contains(&res.status);
            self.host_policy.record(
                &url,
                Tier::Reqwest,
                if blocked {
                    TierOutcome::Blocked
                } else {
                    TierOutcome::Ok
                },
            );
            res
        } else {
            // Synthesise a "degraded" outcome so the cascade naturally
            // proceeds into the next tier.
            http::FetchOutcome {
                status: 0,
                final_url: url.clone(),
                content_type: String::new(),
                text: String::new(),
                transport_error: Some("skipped by host policy".into()),
            }
        };

        if let Some(early) = self
            .try_fallback_cascade(
                &mut primary,
                &url,
                start_tier,
                skip_fallback,
                max_chars,
                include_links,
                &mut cascade_notes,
            )
            .await
        {
            return early;
        }
        // --- Report failure after the full cascade ---------------------
        if let Some(msg) = primary.transport_error {
            return ToolResult::err(format!(
                "HTTP request failed: {msg}\nFallback cascade: {}",
                cascade_notes.join(" → ")
            ));
        }

        if let Some(kind) = detect_block(primary.status, &primary.text) {
            let kind_label = match kind {
                BlockKind::Cloudflare => "Cloudflare challenge",
                BlockKind::AntiBotWall => "anti-bot wall",
                BlockKind::JsShell => "JS-rendered shell (no visible text)",
            };
            let cascade_summary = if cascade_notes.is_empty() {
                "no fallback attempted".to_string()
            } else {
                cascade_notes.join(" → ")
            };
            let hint = format!(
                "BLOCKED: {kind_label} on {final_url}\n\
                 HTTP {status} — body suppressed.\n\
                 Fallback cascade: {cascade_summary}\n\n\
                 All automatic tiers (reqwest + URL-prefix + TLS impersonation + Wayback) exhausted.\n\
                 This site requires a cloud stealth service (Scrapfly / Browserbase / Kernel) \
                 OR a live browser session with human interaction. Consider switching to an \
                 alternative source: mogi.vn, homedy.com, nhadat24h.net, batdongsan.com.vn.",
                kind_label = kind_label,
                status = primary.status,
                final_url = primary.final_url,
                cascade_summary = cascade_summary,
            );
            return ToolResult::err(hint);
        }

        let header = format!(
            "HTTP {} — {}\nContent-Type: {}\n\n",
            primary.status, primary.final_url, primary.content_type
        );
        super::fetch_common::format_fetch_output(
            &primary.text,
            &header,
            include_links,
            max_chars,
            (200..400).contains(&primary.status),
        )
    }
}

#[cfg(test)]
#[path = "web_fetch_tests.rs"]
mod tests;
