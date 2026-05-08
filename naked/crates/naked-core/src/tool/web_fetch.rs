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
use std::time::Duration;

use async_trait::async_trait;
use serde_json::{Value, json};
use tokio::process::Command;

use crate::scrape::host_policy::{HostPolicy, Outcome as TierOutcome, Tier};
use crate::scrape::multi::MultiCloudScraper;
use crate::tool::Tool;
use crate::types::{Permission, ToolResult, ToolSpec};

/// Classification of an upstream-blocked HTTP response.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BlockKind {
    /// Cloudflare challenge / `Just a moment…` interstitial.
    Cloudflare,
    /// Generic anti-bot wall (Akamai, PerimeterX, etc.) — kept as a
    /// catch-all so future patterns can be added without another enum
    /// rename.
    AntiBotWall,
}

/// Heuristic detector for "the HTTP response looks fine but the body is
/// useless because an anti-bot wall is blocking us". Returns `None` when
/// the response looks clean.
///
/// The test `red_d2_cloudflare_challenge_detected` pins this contract:
/// operators need a typed signal to switch to Playwright instead of
/// staring at an empty body.
pub fn detect_block(status: u16, body: &str) -> Option<BlockKind> {
    let lower_small = body
        .chars()
        .take(4096)
        .collect::<String>()
        .to_ascii_lowercase();
    let cf_signals = [
        "cf-browser-verification",
        "cf-chl-bypass",
        "__cf_chl",
        "just a moment…",
        "just a moment...",
        "attention required! | cloudflare",
        "checking your browser before accessing",
    ];
    if cf_signals.iter().any(|s| lower_small.contains(s)) {
        return Some(BlockKind::Cloudflare);
    }
    if status == 403 && lower_small.contains("cloudflare") {
        return Some(BlockKind::Cloudflare);
    }
    let generic_walls = ["access denied", "request blocked", "enable javascript"];
    if (status == 403 || status == 429 || status == 503)
        && generic_walls.iter().any(|s| lower_small.contains(s))
    {
        return Some(BlockKind::AntiBotWall);
    }
    None
}

/// Hard ceiling on response bytes pulled through `web_fetch`. Prevents a
/// misbehaving site from forcing us to load a 500 MB asset before the timeout
/// trips. 2 MB covers every reasonable HTML page.
const DEFAULT_MAX_BYTES: u64 = 2 * 1024 * 1024;
/// Maximum chars the tool returns — keeps a single fetch from spending the
/// full context window. Exposed `pub(crate)` so sibling fetch tools
/// (web_fetch_tls, web_fetch_wayback) return the same-sized payloads
/// and the agent's context budgeting is consistent across backends.
pub(crate) const DEFAULT_MAX_CHARS: usize = 20_000;
/// Hard per-request wall clock. Aligned with typical MCP timeouts.
const DEFAULT_TIMEOUT_SECS: u64 = 30;

pub struct WebFetchTool {
    client: reqwest::Client,
    /// Optional cloud-scrape cascade (ScrapingBee + Firecrawl). Inserted as
    /// Tier 3.5 between TLS impersonation and Wayback. `None` when no keys
    /// are configured — the cascade simply skips that tier.
    cloud: Option<Arc<MultiCloudScraper>>,
    /// Adaptive per-host tier selector. Tracks which tiers have been
    /// blocked for which hosts and skips them on the next fetch — saves
    /// ~10 s per request on sites where reqwest/url-prefix are guaranteed
    /// to fail (CF-protected portals).
    host_policy: Arc<HostPolicy>,
}

/// Browser-like User-Agent. Anti-bot vendors (Cloudflare, Akamai) flag
/// tools advertising themselves as `curl/*`, `python-requests`, or
/// `naked/*` within milliseconds. A realistic UA alone doesn't defeat JA3
/// fingerprinting (that needs a real browser), but it stops the trivial
/// "block by UA substring" rules that run in front of the full challenge.
const BROWSER_USER_AGENT: &str = concat!(
    "Mozilla/5.0 (X11; Linux x86_64) ",
    "AppleWebKit/537.36 (KHTML, like Gecko) ",
    "Chrome/120.0.0.0 Safari/537.36",
);

/// Tier-1 fallback for the `NAKED_PROXY` env var: pick an HTTP proxy from
/// `scripts/proxy_pool.py`. Only `NAKED_PROXY` (and its legacy `YT_PROXY`
/// alias) is read from Rust — keeping pool selection in Python means the
/// systemd unit stays the single source of truth.
fn resolve_proxy_url() -> Option<String> {
    if std::env::var("NAKED_PROXY_DISABLE").ok().as_deref() == Some("1")
        || std::env::var("YT_PROXY_DISABLE").ok().as_deref() == Some("1")
    {
        return None;
    }
    std::env::var("NAKED_PROXY")
        .ok()
        .or_else(|| std::env::var("YT_PROXY").ok())
        .filter(|s| !s.is_empty())
}

/// Tier-2 fallback: a CORS-anywhere-style URL-prefix proxy. When the direct
/// request hits an anti-bot wall OR returns a non-success status, we retry
/// once by rewriting the target URL to `${prefix}/${target}`. See
/// `scripts/proxy_pool.py::URL_PREFIX_POOL` for the current endpoints and
/// why they exist. Returns `None` when disabled or unset.
fn resolve_url_prefix() -> Option<String> {
    if std::env::var("NAKED_FETCH_URL_PREFIX_DISABLE")
        .ok()
        .as_deref()
        == Some("1")
    {
        return None;
    }
    std::env::var("NAKED_FETCH_URL_PREFIX")
        .ok()
        .map(|s| s.trim().trim_end_matches('/').to_string())
        .filter(|s| !s.is_empty() && (s.starts_with("http://") || s.starts_with("https://")))
}

/// Guard against proxy loops: if the caller already handed us a URL that
/// lives under the prefix domain, we must not wrap it again.
fn is_already_prefixed(url: &str, prefix: &str) -> bool {
    url.starts_with(prefix)
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

    /// Full constructor — used by `AgentCore` to share a single
    /// [`HostPolicy`] across every fetch tool instance, so per-host
    /// learning persists for the lifetime of the agent process.
    pub fn with_components(
        cloud: Option<Arc<MultiCloudScraper>>,
        host_policy: Arc<HostPolicy>,
    ) -> Self {
        let mut builder = reqwest::Client::builder()
            .user_agent(BROWSER_USER_AGENT)
            .timeout(Duration::from_secs(DEFAULT_TIMEOUT_SECS))
            .redirect(reqwest::redirect::Policy::limited(5));

        if let Some(proxy_url) = resolve_proxy_url() {
            match reqwest::Proxy::all(&proxy_url) {
                Ok(proxy) => {
                    let safe = sanitize_proxy_url(&proxy_url);
                    tracing::info!(proxy = %safe, "web_fetch: routing through proxy");
                    builder = builder.proxy(proxy);
                }
                Err(e) => {
                    tracing::warn!(error = %e, "web_fetch: invalid NAKED_PROXY; running direct");
                }
            }
        }

        let client = builder.build().expect("web_fetch: reqwest client init");
        Self {
            client,
            cloud,
            host_policy,
        }
    }
    /// Fallback cascade: URL-prefix → TLS → cloud-scrape → Wayback.
    #[allow(clippy::too_many_arguments)]
    async fn try_fallback_cascade(
        &self,
        primary: &mut FetchOutcome,
        url: &str,
        start_tier: Tier,
        skip_fallback: bool,
        max_chars: usize,
        include_links: bool,
        cascade_notes: &mut Vec<String>,
    ) -> Option<super::ToolResult> {
        // Tier-2 fallback: retry once via the CORS-prefix pool only if the
        // primary path clearly failed (transport error, anti-bot wall, or
        // non-success HTTP status). Successful 200-with-body is returned
        // immediately — no point burning a second round-trip.
        if primary.is_degraded()
            && start_tier <= Tier::UrlPrefix
            && let Some(prefix) = resolve_url_prefix()
            && !is_already_prefixed(url, &prefix)
        {
            let wrapped = format!("{prefix}/{url}");
            tracing::info!(
                target = %url,
                via = %prefix,
                primary_status = primary.status,
                primary_error = %primary.transport_error.as_deref().unwrap_or(""),
                "web_fetch: primary degraded, retrying via URL-prefix proxy",
            );
            match self.fetch_once(&wrapped).await {
                Ok(mut fallback) => {
                    // The CORS proxy may itself return 5xx when the
                    // upstream target is bad; in that case we still
                    // prefer the primary result (it's at least
                    // authoritative about what the origin said).
                    if !fallback.is_degraded() {
                        // Rewrite final_url back to the original
                        // target so downstream dedup/memory doesn't
                        // treat `ws-xxx.onrender.com/...` as a new
                        // canonical URL for this content.
                        fallback.final_url = url.to_string();
                        *primary = fallback;
                        self.host_policy
                            .record(url, Tier::UrlPrefix, TierOutcome::Ok);
                    } else {
                        tracing::info!(
                            fallback_status = fallback.status,
                            fallback_error = %fallback.transport_error.as_deref().unwrap_or(""),
                            "web_fetch: URL-prefix fallback also degraded; keeping primary result",
                        );
                        self.host_policy
                            .record(url, Tier::UrlPrefix, TierOutcome::Blocked);
                    }
                }
                Err(msg) => {
                    tracing::warn!(error = %msg, "web_fetch: URL-prefix fallback failed");
                    self.host_policy
                        .record(url, Tier::UrlPrefix, TierOutcome::Blocked);
                }
            }
        }

        // --- Tier-3: TLS impersonation via curl_cffi subprocess ---------
        // Reqwest+URL-prefix both reached dead ends. Before giving up,
        // try the JA3-impersonation path — it's the only thing that
        // empirically cracks batdongsan/chotot/alonhadat CF walls.
        let is_blocked_primary = primary.transport_error.is_some()
            || detect_block(primary.status, &primary.text).is_some()
            || !(200..400).contains(&primary.status);
        if is_blocked_primary && start_tier <= Tier::Reqwest {
            cascade_notes.push(match primary.transport_error.as_deref() {
                Some(e) => format!("reqwest: transport error ({e})"),
                None => format!(
                    "reqwest: HTTP {} {}",
                    primary.status,
                    detect_block(primary.status, &primary.text)
                        .map(|k| format!("({:?})", k))
                        .unwrap_or_default()
                ),
            });
        }

        if is_blocked_primary && !skip_fallback && start_tier <= Tier::Tls {
            tracing::debug!(
                url = %url,
                "web_fetch: primary blocked/failed, escalating to TLS impersonation",
            );
            match fetch_via_tls_subprocess(url).await {
                Ok(tls_body) => {
                    if let Some(kind) = detect_block(tls_body.status, &tls_body.text) {
                        cascade_notes.push(format!("tls: HTTP {} ({:?})", tls_body.status, kind));
                        self.host_policy
                            .record(url, Tier::Tls, TierOutcome::Blocked);
                    } else if !(200..400).contains(&tls_body.status) {
                        cascade_notes.push(format!("tls: HTTP {}", tls_body.status));
                        self.host_policy
                            .record(url, Tier::Tls, TierOutcome::Blocked);
                    } else {
                        *primary = tls_body;
                        cascade_notes.push("tls: OK (used)".into());
                        self.host_policy.record(url, Tier::Tls, TierOutcome::Ok);
                    }
                }
                Err(msg) => {
                    cascade_notes.push(format!("tls: {msg}"));
                    self.host_policy
                        .record(url, Tier::Tls, TierOutcome::Blocked);
                }
            }
        }

        // --- Tier-3.5: Cloud-scrape cascade (ScrapingBee / Firecrawl) ---
        // Reqwest, URL-prefix, AND TLS impersonation all reached blocks.
        // Burn a paid request through ScrapingBee/Firecrawl — they bring
        // residential IPs + headless browsers and crack the remaining
        // ~30% of CF-protected VN portals (chotot listing detail pages,
        // batdongsan ads with phone-reveal walls, dotproperty SPA).
        let mid_blocked = primary.transport_error.is_some()
            || detect_block(primary.status, &primary.text).is_some()
            || !(200..400).contains(&primary.status);
        if mid_blocked
            && !skip_fallback
            && start_tier <= Tier::Cloud
            && let Some(cloud) = self.cloud.as_ref()
            && cloud.is_active()
        {
            tracing::debug!(
                url = %url,
                engines = %cloud.engine_summary(),
                "web_fetch: escalating to cloud-scrape cascade",
            );
            match cloud.scrape(url).await {
                Ok(cloud_body) => {
                    if let Some(kind) = detect_block(cloud_body.status, &cloud_body.body) {
                        cascade_notes.push(format!(
                            "{}: HTTP {} ({:?})",
                            cloud_body.provider, cloud_body.status, kind
                        ));
                        self.host_policy
                            .record(url, Tier::Cloud, TierOutcome::Blocked);
                    } else if !(200..400).contains(&cloud_body.status) {
                        cascade_notes.push(format!(
                            "{}: HTTP {}",
                            cloud_body.provider, cloud_body.status
                        ));
                        self.host_policy
                            .record(url, Tier::Cloud, TierOutcome::Blocked);
                    } else {
                        *primary = FetchOutcome {
                            status: cloud_body.status,
                            final_url: cloud_body.final_url,
                            content_type: cloud_body.content_type,
                            text: cloud_body.body,
                            transport_error: None,
                        };
                        cascade_notes.push(format!("{}: OK (used)", cloud_body.provider));
                        self.host_policy.record(url, Tier::Cloud, TierOutcome::Ok);
                    }
                }
                Err(msg) => {
                    cascade_notes.push(format!("cloud-scrape: {msg}"));
                    self.host_policy
                        .record(url, Tier::Cloud, TierOutcome::Blocked);
                }
            }
        }

        // --- Tier-4: Wayback snapshot -----------------------------------
        // Last resort. Serves stale content but at least gives the agent
        // *something* to reason about. Labelled clearly in the header so
        // the agent knows to cite the archive timestamp.
        let still_blocked = primary.transport_error.is_some()
            || detect_block(primary.status, &primary.text).is_some()
            || !(200..400).contains(&primary.status);
        if still_blocked && !skip_fallback && start_tier <= Tier::Wayback {
            tracing::info!(url = %url, "web_fetch: escalating to Wayback snapshot");
            match fetch_wayback_snapshot(&self.client, url).await {
                Ok((wb_body, wb_timestamp, wb_snapshot_url)) => {
                    cascade_notes.push(format!("wayback: OK (ts={wb_timestamp}, used)"));
                    self.host_policy.record(url, Tier::Wayback, TierOutcome::Ok);
                    let cascade_summary = cascade_notes.join(" → ");
                    let header = format!(
                        "Wayback snapshot (timestamp {}): {}\nOriginal URL: {url}\nFallback cascade: {cascade_summary}\n\n",
                        format_wayback_ts(&wb_timestamp),
                        wb_snapshot_url,
                    );
                    return Some(super::fetch_common::format_fetch_output(
                        &wb_body,
                        &header,
                        include_links,
                        max_chars,
                        true,
                    ));
                }
                Err(msg) => {
                    cascade_notes.push(format!("wayback: {msg}"));
                    self.host_policy
                        .record(url, Tier::Wayback, TierOutcome::Blocked);
                }
            }
        }

        None
    }
}

/// Strip `user:pass@` from a proxy URL before logging. String-based so we
/// don't pull in the `url` crate just for a log line.
fn sanitize_proxy_url(raw: &str) -> String {
    // Split once on "://" to preserve the scheme, then on the last "@" in
    // the authority to drop credentials.
    if let Some((scheme, rest)) = raw.split_once("://")
        && let Some((_creds, hostport)) = rest.rsplit_once('@')
    {
        return format!("{scheme}://{hostport}");
    }
    raw.to_string()
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
                Err(msg) => FetchOutcome {
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
            FetchOutcome {
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

/// Spawn `scripts/fetch_tls.py` and unpack the JSON result into a
/// `FetchOutcome`-shaped record. Kept here (not in `web_fetch_tls.rs`)
/// so the cascade inside `WebFetchTool::execute` can call it without
/// cross-crate tool dispatch. The standalone `WebFetchTlsTool` remains
/// the explicit path — identical subprocess contract.
async fn fetch_via_tls_subprocess(url: &str) -> Result<FetchOutcome, String> {
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
async fn fetch_wayback_snapshot(
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
fn format_wayback_ts(ts: &str) -> String {
    if ts.len() < 8 {
        return ts.to_string();
    }
    let y = &ts[0..4];
    let m = ts.get(4..6).unwrap_or("01");
    let d = ts.get(6..8).unwrap_or("01");
    format!("{y}-{m}-{d}")
}

/// Result of a single HTTP round-trip. Lives outside the `Tool::execute`
/// method so we can attempt the request twice (direct + URL-prefix
/// fallback) without duplicating 40 lines of send/read/bound-check code.
struct FetchOutcome {
    status: u16,
    final_url: String,
    content_type: String,
    text: String,
    /// Populated when reqwest couldn't complete the round-trip at all
    /// (DNS failure, connection reset, TLS error, …). Status/body are
    /// meaningless in that case; caller must special-case this.
    transport_error: Option<String>,
}

impl FetchOutcome {
    /// "Should we even try the fallback path?" Transport errors and
    /// 4xx/5xx are both considered degraded. 200-with-body is not —
    /// anti-bot detection runs separately on the text and can promote
    /// an HTTP 200 to a `BLOCKED` return.
    fn is_degraded(&self) -> bool {
        if self.transport_error.is_some() {
            return true;
        }
        !(200..400).contains(&self.status)
    }
}

impl WebFetchTool {
    async fn fetch_once(&self, url: &str) -> Result<FetchOutcome, String> {
        // Vietnamese real-estate portals vary their bot-wall aggressiveness
        // by `Accept-Language` — sending a realistic `vi,en` list (as a
        // Chrome install in VN would) reduces false-positive captcha rate
        // on chotot.com/alonhadat.com.vn. Doesn't help on CF-fingerprinted
        // sites (batdongsan), which need Playwright anyway.
        let resp = self
            .client
            .get(url)
            .header(reqwest::header::ACCEPT_LANGUAGE, "vi,en-US;q=0.9,en;q=0.8")
            .header(
                reqwest::header::ACCEPT,
                "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            )
            .send()
            .await
            .map_err(|e| e.to_string())?;

        let status = resp.status();
        let final_url = resp.url().to_string();
        let content_type = resp
            .headers()
            .get(reqwest::header::CONTENT_TYPE)
            .and_then(|v| v.to_str().ok())
            .unwrap_or("")
            .to_string();

        if let Some(len_hint) = resp.content_length()
            && len_hint > DEFAULT_MAX_BYTES
        {
            return Err(format!(
                "response too large ({len_hint} bytes, limit {DEFAULT_MAX_BYTES})"
            ));
        }

        let bytes = resp.bytes().await.map_err(|e| e.to_string())?;
        if bytes.len() as u64 > DEFAULT_MAX_BYTES {
            return Err(format!(
                "response too large ({} bytes, limit {})",
                bytes.len(),
                DEFAULT_MAX_BYTES
            ));
        }
        let text = String::from_utf8_lossy(&bytes).into_owned();

        Ok(FetchOutcome {
            status: status.as_u16(),
            final_url,
            content_type,
            text,
            transport_error: None,
        })
    }
}

/// Minimal HTML → plain-text stripper. Not a DOM parser — we explicitly
/// don't rely on one because the output only needs to be "readable by an
/// LLM", and a 300 KB page with nested `<script>` tags would otherwise
/// require `scraper` / `html5ever` (+1.5 MB binary bloat).
///
/// Strategy:
/// 1. Drop `<script>...</script>` and `<style>...</style>` blocks wholesale.
/// 2. Extract `href` values from `<a>` tags into a separate list (preserved
///    so the research agent can save them without re-fetching).
/// 3. Replace any remaining `<tag>` with a single space.
/// 4. Collapse runs of whitespace to one space / one newline.
///
/// Exposed `pub(crate)` so `web_fetch_tls` / `web_fetch_wayback` can reuse
/// it — consistent text shape across backends keeps the agent's prompt
/// handling trivial.
pub(crate) fn html_to_text(html: &str) -> (String, Vec<String>) {
    let cleaned = drop_block(html, "script");
    let cleaned = drop_block(&cleaned, "style");
    let links = extract_hrefs(&cleaned);
    let mut out = String::with_capacity(cleaned.len());
    let mut in_tag = false;
    for ch in cleaned.chars() {
        match ch {
            '<' => in_tag = true,
            '>' => {
                in_tag = false;
                out.push(' ');
            }
            _ if in_tag => {}
            _ => out.push(ch),
        }
    }
    let decoded = decode_basic_entities(&out);
    let collapsed = collapse_whitespace(&decoded);
    (collapsed, links)
}

/// Drop all `<tag ...>...</tag>` blocks, case-insensitive.
fn drop_block(input: &str, tag: &str) -> String {
    let lower = input.to_ascii_lowercase();
    let open = format!("<{tag}");
    let close = format!("</{tag}>");
    let mut out = String::with_capacity(input.len());
    let mut i = 0;
    while i < input.len() {
        match lower[i..].find(&open) {
            Some(rel) => {
                let start = i + rel;
                out.push_str(&input[i..start]);
                // Find end of the opening `<tag ...>` then `</tag>`.
                let after_open = match input[start..].find('>') {
                    Some(p) => start + p + 1,
                    None => break,
                };
                match lower[after_open..].find(&close) {
                    Some(end_rel) => {
                        i = after_open + end_rel + close.len();
                    }
                    None => break,
                }
            }
            None => {
                out.push_str(&input[i..]);
                break;
            }
        }
    }
    out
}

/// Extract href values from `<a ... href="..."...>`. Naïve but good enough for
/// a stripped HTML body. Preserves order, deduplicates, skips fragments /
/// javascript: URLs.
fn extract_hrefs(input: &str) -> Vec<String> {
    let lower = input.to_ascii_lowercase();
    let mut out = Vec::new();
    let mut seen = std::collections::HashSet::new();
    let mut i = 0;
    while i < input.len() {
        let Some(rel) = lower[i..].find("<a") else {
            break;
        };
        let start = i + rel;
        let Some(close) = input[start..].find('>') else {
            break;
        };
        let tag_slice = &input[start..start + close];
        if let Some(href) = find_attr(tag_slice, "href")
            && !href.is_empty()
            && !href.starts_with('#')
            && !href.to_ascii_lowercase().starts_with("javascript:")
            && seen.insert(href.clone())
        {
            out.push(href);
        }
        i = start + close + 1;
    }
    out
}

fn find_attr(tag: &str, name: &str) -> Option<String> {
    let lower = tag.to_ascii_lowercase();
    let needle = format!("{name}=");
    let pos = lower.find(&needle)?;
    let after = pos + needle.len();
    let tag_bytes = tag.as_bytes();
    let quote = *tag_bytes.get(after)?;
    let (open, end_at) = if quote == b'"' || quote == b'\'' {
        (
            after + 1,
            tag[after + 1..]
                .find(quote as char)
                .map(|p| after + 1 + p)?,
        )
    } else {
        let rest = &tag[after..];
        let stop: &[char] = &[' ', '\t', '>', '\n', '\r'];
        let stop_rel = rest.find(stop).unwrap_or(rest.len());
        (after, after + stop_rel)
    };
    Some(tag[open..end_at].to_string())
}

fn decode_basic_entities(s: &str) -> String {
    // Only the handful that break readability. A full entity decode would
    // require a dep table; leave anything exotic as-is.
    s.replace("&amp;", "&")
        .replace("&lt;", "<")
        .replace("&gt;", ">")
        .replace("&quot;", "\"")
        .replace("&apos;", "'")
        .replace("&nbsp;", " ")
}

fn collapse_whitespace(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    let mut last_newline = false;
    let mut last_space = false;
    for ch in s.chars() {
        if ch == '\n' || ch == '\r' {
            if !last_newline {
                out.push('\n');
                last_newline = true;
                last_space = true;
            }
        } else if ch.is_whitespace() {
            if !last_space {
                out.push(' ');
                last_space = true;
            }
        } else {
            out.push(ch);
            last_newline = false;
            last_space = false;
        }
    }
    out.trim().to_string()
}

#[cfg(test)]
#[path = "web_fetch_tests.rs"]
mod tests;
