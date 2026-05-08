//! HTTP client construction, direct fetch, and backend helpers for
//! `web_fetch`.
//!
//! Contains:
//! - `WebFetchTool::with_components` constructor (reqwest client setup)
//! - `WebFetchTool::fetch_once` — single-attempt direct fetch
//! - `FetchOutcome` — typed result of one HTTP round-trip
//! - TLS-impersonation subprocess helper (`fetch_via_tls_subprocess`)
//! - Wayback snapshot helper (`fetch_wayback_snapshot`)
//! - Proxy URL helpers

use std::time::Duration;

use crate::scrape::host_policy::HostPolicy;
use crate::scrape::multi::MultiCloudScraper;

use super::{DEFAULT_MAX_BYTES, DEFAULT_TIMEOUT_SECS, WebFetchTool};

/// Browser-like User-Agent. Anti-bot vendors (Cloudflare, Akamai) flag
/// tools advertising themselves as `curl/*`, `python-requests`, or
/// `naked/*` within milliseconds. A realistic UA alone doesn't defeat JA3
/// fingerprinting (that needs a real browser), but it stops the trivial
/// "block by UA substring" rules that run in front of the full challenge.
pub(super) const BROWSER_USER_AGENT: &str = concat!(
    "Mozilla/5.0 (X11; Linux x86_64) ",
    "AppleWebKit/537.36 (KHTML, like Gecko) ",
    "Chrome/120.0.0.0 Safari/537.36",
);

/// Tier-1 fallback for the `NAKED_PROXY` env var: pick an HTTP proxy from
/// `scripts/proxy_pool.py`. Only `NAKED_PROXY` (and its legacy `YT_PROXY`
/// alias) is read from Rust — keeping pool selection in Python means the
/// systemd unit stays the single source of truth.
pub(super) fn resolve_proxy_url() -> Option<String> {
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
pub(super) fn resolve_url_prefix() -> Option<String> {
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
pub(super) fn is_already_prefixed(url: &str, prefix: &str) -> bool {
    url.starts_with(prefix)
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

impl WebFetchTool {
    /// Full constructor — used by `AgentCore` to share a single
    /// [`HostPolicy`] across every fetch tool instance, so per-host
    /// learning persists for the lifetime of the agent process.
    pub fn with_components(
        cloud: Option<std::sync::Arc<MultiCloudScraper>>,
        host_policy: std::sync::Arc<HostPolicy>,
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

    /// Single HTTP round-trip. Vietnamese real-estate portals vary their
    /// bot-wall aggressiveness by `Accept-Language` — sending a realistic
    /// `vi,en` list reduces false-positive captcha rate on
    /// chotot.com/alonhadat.com.vn. Doesn't help on CF-fingerprinted
    /// sites (batdongsan), which need Playwright anyway.
    pub(super) async fn fetch_once(&self, url: &str) -> Result<FetchOutcome, String> {
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

/// Result of a single HTTP round-trip. Lives here so both `fetch_once` and
/// `try_fallback_cascade` can share the type without cross-module loops.
pub(super) struct FetchOutcome {
    pub(super) status: u16,
    pub(super) final_url: String,
    pub(super) content_type: String,
    pub(super) text: String,
    /// Populated when reqwest couldn't complete the round-trip at all
    /// (DNS failure, connection reset, TLS error, …). Status/body are
    /// meaningless in that case; caller must special-case this.
    pub(super) transport_error: Option<String>,
}

impl FetchOutcome {
    /// "Should we even try the fallback path?" Transport errors and
    /// 4xx/5xx are both considered degraded. 200-with-body is not —
    /// anti-bot detection runs separately on the text and can promote
    /// an HTTP 200 to a `BLOCKED` return.
    pub(super) fn is_degraded(&self) -> bool {
        if self.transport_error.is_some() {
            return true;
        }
        !(200..400).contains(&self.status)
    }
}
