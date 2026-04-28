//! Pre-save validators for `research_save`.
//!
//! Two categories, both cheap enough to run inside the tool handler before
//! we ever persist the finding:
//!
//! 1. **URL specificity** — reject category / listing-index URLs that have no
//!    concrete `id`-segment. Examples of what we want to catch:
//!    `https://mogi.vn/da-nang/quan-hai-chau/thue-mat-bang-cua-hang-shop`
//!    (DM 2423e530 on 2026-04-22 saved a dozen of these, Gatekeeper ripped
//!    them out afterwards).
//!
//! 2. **Freshness** — reject listings with `posted_at > 90d`. Same incident
//!    burned an entire turn before the Gatekeeper noticed.
//!
//! Both are intentionally heuristic; a caller may still override by passing
//! validated metadata (e.g. a server-confirmed age via HTTP HEAD).

use chrono::{DateTime, Duration, Utc};

/// Verdict on whether a URL looks like a concrete listing (has a per-item
/// identifier) or a category/index page.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum UrlSpecificity {
    /// URL contains a per-listing identifier — safe to save.
    ConcreteListing,
    /// URL looks like a category / search / index page — reject.
    CategoryPage { reason: String },
}

impl UrlSpecificity {
    /// Classify a URL purely from its path structure. No network access.
    ///
    /// Heuristics (tuned against mogi.vn / batdongsan.com.vn / common
    /// real-estate portals seen in research runs):
    ///
    /// - path ends in `-id<digits>` or `/id<digits>` → concrete
    /// - path matches `/<digits>/<digits>?.html?$` with a short numeric id → concrete
    /// - last path segment contains at least one numeric run ≥6 digits → concrete
    /// - otherwise → CategoryPage
    pub fn classify(url: &str) -> Self {
        let Some(path) = extract_path(url) else {
            return UrlSpecificity::CategoryPage {
                reason: "url has no path".to_string(),
            };
        };
        let path_norm = path.trim_end_matches('/').to_lowercase();
        if path_norm.is_empty() || path_norm == "/" {
            return UrlSpecificity::CategoryPage {
                reason: "root / listing index".to_string(),
            };
        }
        // -id<digits> or /id<digits> suffix
        if let Some(last) = path_norm.rsplit('/').next() {
            if let Some(idx) = last.rfind("-id") {
                let tail = &last[idx + 3..];
                if tail.chars().all(|c| c.is_ascii_digit()) && !tail.is_empty() {
                    return UrlSpecificity::ConcreteListing;
                }
            }
            if let Some(tail) = last.strip_prefix("id")
                && tail.chars().all(|c| c.is_ascii_digit())
                && tail.len() >= 4
            {
                return UrlSpecificity::ConcreteListing;
            }
            // long numeric run in the slug
            let mut run = 0;
            for ch in last.chars() {
                if ch.is_ascii_digit() {
                    run += 1;
                    if run >= 6 {
                        return UrlSpecificity::ConcreteListing;
                    }
                } else {
                    run = 0;
                }
            }
            // .html with numeric filename
            if let Some(stem) = last.strip_suffix(".html")
                && stem.chars().all(|c| c.is_ascii_digit())
                && stem.len() >= 5
            {
                return UrlSpecificity::ConcreteListing;
            }
        }
        UrlSpecificity::CategoryPage {
            reason: "path has no per-item id (no `-id<digits>` / numeric run)".to_string(),
        }
    }

    pub fn is_concrete(&self) -> bool {
        matches!(self, UrlSpecificity::ConcreteListing)
    }
}

fn extract_path(url: &str) -> Option<String> {
    let rest = url.split_once("://").map(|(_, r)| r).unwrap_or(url);
    let slash = rest.find('/')?;
    let mut path = rest[slash..].to_string();
    if let Some(q) = path.find('?') {
        path.truncate(q);
    }
    if let Some(h) = path.find('#') {
        path.truncate(h);
    }
    Some(path)
}

/// Case-insensitive substrings we treat as "this `excerpt` or
/// `source_content` is not real listing data, it's a captcha /
/// anti-bot placeholder" — e.g. an agent that ignored the web_fetch
/// "BLOCKED" hint and tried to save the raw HTML anyway. Having this
/// at the tool boundary means a malfunctioning prompt can't silently
/// pollute findings.jsonl with stubs that the gatekeeper later has
/// to rip out.
///
/// Keep the list small and high-precision. False positives reject
/// legitimate findings; false negatives only mean a stub slips
/// through (the gatekeeper still catches those on the next pass).
const CAPTCHA_STUB_MARKERS: &[&str] = &[
    "just a moment",
    "just a moment...",
    "just a moment…",
    "attention required! cloudflare",
    "attention required | cloudflare",
    "enable javascript and cookies",
    "enable cookies to continue",
    "verify you are human",
    "verifying you are human",
    "please enable js",
    "vui lòng xác minh",
    "xac-thuc-nguoi-dung",
    "checking your browser before accessing",
    "cf_chl_rt_tk",
];

/// Returns the first matching captcha-stub marker (lowercased) when
/// `body` looks like the agent tried to save a placeholder page
/// instead of real listing content. `None` otherwise.
pub fn looks_like_captcha_stub(body: &str) -> Option<&'static str> {
    let lowered = body.to_lowercase();
    for marker in CAPTCHA_STUB_MARKERS {
        if lowered.contains(marker) {
            return Some(*marker);
        }
    }
    None
}

/// Pre-save freshness guard. `posted_at` is the listing's publication date
/// (best-effort extracted from the crawl). `is_stale(limit)` returns true
/// when the listing is older than `limit`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Freshness {
    posted_at: DateTime<Utc>,
}

impl Freshness {
    pub fn from_posted_at(posted_at: DateTime<Utc>) -> Self {
        Self { posted_at }
    }

    pub fn is_stale(&self, limit: Duration) -> bool {
        let age = Utc::now().signed_duration_since(self.posted_at);
        age > limit
    }

    pub fn age(&self) -> Duration {
        Utc::now().signed_duration_since(self.posted_at)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn category_url_rejected() {
        let v = UrlSpecificity::classify(
            "https://mogi.vn/da-nang/quan-hai-chau/thue-mat-bang-cua-hang-shop",
        );
        match v {
            UrlSpecificity::CategoryPage { .. } => {}
            _ => panic!("expected CategoryPage, got {v:?}"),
        }
    }

    #[test]
    fn concrete_listing_with_id_suffix_accepted() {
        let v = UrlSpecificity::classify(
            "https://mogi.vn/quan-ngu-hanh-son/thue-can-ho-chung-cu-...-id22092735",
        );
        assert!(matches!(v, UrlSpecificity::ConcreteListing));
    }

    #[test]
    fn concrete_listing_with_long_numeric_run_accepted() {
        let v = UrlSpecificity::classify("https://example.com/listing/123456");
        assert!(matches!(v, UrlSpecificity::ConcreteListing));
    }

    #[test]
    fn root_url_rejected() {
        let v = UrlSpecificity::classify("https://mogi.vn/");
        assert!(matches!(v, UrlSpecificity::CategoryPage { .. }));
    }

    #[test]
    fn freshness_stale_when_older_than_limit() {
        let posted = Utc::now() - Duration::days(120);
        let f = Freshness::from_posted_at(posted);
        assert!(f.is_stale(Duration::days(90)));
    }

    #[test]
    fn freshness_fresh_when_recent() {
        let posted = Utc::now() - Duration::days(15);
        let f = Freshness::from_posted_at(posted);
        assert!(!f.is_stale(Duration::days(90)));
    }

    #[test]
    fn captcha_stub_detected_in_excerpt() {
        let excerpt = "Just a moment... We are checking your browser before accessing the site.";
        assert!(looks_like_captcha_stub(excerpt).is_some());
    }

    #[test]
    fn captcha_stub_detects_vietnamese_verify_prompt() {
        let body = "Vui lòng xác minh bạn là người thật để tiếp tục.";
        assert!(looks_like_captcha_stub(body).is_some());
    }

    #[test]
    fn captcha_stub_passes_through_real_listing() {
        let body = "Cho thuê mặt bằng kinh doanh 120m2, mặt tiền đường Lê Quang Đạo, \
                    giá 28 triệu/tháng. Liên hệ: 0905-123-456 (Zalo).";
        assert!(looks_like_captcha_stub(body).is_none());
    }
}
