//! Cloud-scraping backends for the `web_fetch` cascade.
//!
//! These are paid third-party services (ScrapingBee, Firecrawl) that bring
//! their own residential / datacenter proxy pools, headless browsers, and
//! anti-bot bypass infrastructure. They sit in the cascade **between** TLS
//! impersonation (free, fast, ~70% effective on VN portals) and the Wayback
//! Machine (free, instant, but stale). Insertion point in
//! `tool::web_fetch.rs::execute`:
//!
//! ```text
//!   reqwest ─► URL-prefix proxy ─► TLS impersonation
//!                                         │
//!                                         ▼  (block detected)
//!                                  cloud scraper (this module)
//!                                         │
//!                                         ▼  (still blocked)
//!                                    Wayback snapshot
//! ```
//!
//! ## Design notes
//!
//! - Each engine implements [`CloudScraper`] (`async fn scrape -> ScrapeResult`).
//! - Engines pull keys from a shared [`crate::keys::pool::KeyPool`] so dead
//!   keys can be marked at the call site and skipped on the next round-robin
//!   (same plumbing as the search backends in [`crate::search`]).
//! - [`multi::MultiCloudScraper`] tries engines in order and returns the
//!   first successful, non-blocked response — keeping the cascade itself
//!   simple (one `if let Some(scraper) = ..` arm).
//! - All engines accept a JS-rendering hint; we always set it because the
//!   sites in scope (batdongsan, dotproperty, alonhadat) hide their
//!   contact buttons behind JS — a static fetch through the cloud is no
//!   better than `web_fetch_tls`.

pub mod firecrawl;
pub mod host_policy;
pub mod multi;
pub mod scrapingbee;

use async_trait::async_trait;

/// Output of a single cloud-scrape attempt.
///
/// Mirrors [`crate::tool::web_fetch::FetchOutcome`] in shape so the cascade
/// can swap one for the other without re-plumbing the post-processing
/// (HTML→text, link extraction, header rendering).
#[derive(Debug, Clone)]
pub struct ScrapeResult {
    /// HTTP status the upstream origin returned (NOT the cloud API's status —
    /// the engines normalise that already so the cascade only sees real
    /// origin codes).
    pub status: u16,
    /// Final URL after all redirects the cloud service followed.
    pub final_url: String,
    /// MIME type from the origin's `Content-Type` header. Empty when the
    /// scraper didn't surface one.
    pub content_type: String,
    /// Raw HTML body. The caller runs it through `html_to_text`.
    pub body: String,
    /// Which engine served the response — surfaced in the cascade summary so
    /// the operator can see "scrapingbee" vs "firecrawl" in the agent log.
    pub provider: &'static str,
}

/// Anti-bot cloud-scrape backend.
#[async_trait]
pub trait CloudScraper: Send + Sync {
    /// Stable engine identifier used in logs and cascade summaries.
    fn name(&self) -> &'static str;

    /// Fetch `url` through the cloud service. Returns `Err` only on transport
    /// failures (network down, key invalid, quota exceeded). A successful
    /// fetch that returns a Cloudflare interstitial body is reported via
    /// [`ScrapeResult`] and detected by the cascade's normal block-check.
    async fn scrape(&self, url: &str) -> Result<ScrapeResult, String>;
}
