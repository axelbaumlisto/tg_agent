//! ScrapingBee cloud-scraping backend.
//!
//! API docs: <https://www.scrapingbee.com/documentation/>
//!
//! Endpoint: `GET https://app.scrapingbee.com/api/v1/`
//! Required params: `api_key`, `url`
//! Useful options:
//! - `render_js=true` — runs the page in a Chromium instance (paid credits ×5).
//! - `premium_proxy=true` — residential IP pool, recommended for VN portals.
//! - `country_code=vn` — geolocate the request to Vietnam (cuts a lot of CF
//!   challenges that fire on foreign-IP signals).
//! - `timeout=30000` — milliseconds; ScrapingBee caps at 140 000.
//!
//! Response is the raw HTML body (or JSON when `?json_response=true`, which
//! we don't use). Status codes:
//! - 200: success
//! - 401: bad/expired api_key — mark dead, retry next key
//! - 422: invalid URL or unsupported domain
//! - 429: out of monthly credits — mark dead
//! - 500: ScrapingBee internal — surface as transport error

use std::sync::Arc;
use std::time::Duration;

use async_trait::async_trait;
use reqwest::Client;

use super::{CloudScraper, ScrapeResult};
use crate::keys::pool::KeyPool;

/// How long we wait for ScrapingBee to return. They suggest 30 s for
/// JS-rendered pages on premium proxy; 60 s gives us a buffer for cold
/// browser instances on the first request to a fresh domain.
const SCRAPINGBEE_TIMEOUT_SECS: u64 = 60;

pub struct ScrapingBeeEngine {
    client: Client,
    keys: Arc<KeyPool>,
    base_url: String,
}

impl ScrapingBeeEngine {
    pub fn new(keys: Arc<KeyPool>) -> Self {
        Self::with_base_url(keys, "https://app.scrapingbee.com".into())
    }

    pub fn with_base_url(keys: Arc<KeyPool>, base_url: String) -> Self {
        let client = Client::builder()
            .timeout(Duration::from_secs(SCRAPINGBEE_TIMEOUT_SECS))
            .build()
            .expect("scrapingbee: reqwest client init");
        Self {
            client,
            keys,
            base_url,
        }
    }

    pub fn key_count(&self) -> usize {
        self.keys.size()
    }
}

#[async_trait]
impl CloudScraper for ScrapingBeeEngine {
    fn name(&self) -> &'static str {
        "scrapingbee"
    }

    async fn scrape(&self, url: &str) -> Result<ScrapeResult, String> {
        // Try up to 3 keys before giving up — covers the case where the first
        // two are out of credits but a third is still good. We don't loop
        // through every key, that would burn budget on a permanently-dead
        // pool; the round-robin cursor takes care of fairness across calls.
        let mut last_err = String::from("scrapingbee: pool empty");
        for attempt in 0..3 {
            let api_key = match self.keys.next() {
                Some(k) => k,
                None => return Err(last_err),
            };
            match self.scrape_once(&api_key, url).await {
                Ok(res) => return Ok(res),
                Err(e) => {
                    if e.contains("401") || e.contains("403") || e.contains("429") {
                        tracing::warn!(
                            attempt, error = %e,
                            "scrapingbee: key rejected, marking dead and retrying"
                        );
                        self.keys.mark_dead(&api_key);
                    }
                    last_err = e;
                }
            }
        }
        Err(last_err)
    }
}

impl ScrapingBeeEngine {
    async fn scrape_once(&self, api_key: &str, url: &str) -> Result<ScrapeResult, String> {
        let req = self
            .client
            .get(format!("{}/api/v1/", self.base_url))
            .query(&[
                ("api_key", api_key),
                ("url", url),
                ("render_js", "true"),
                ("premium_proxy", "true"),
                ("country_code", "vn"),
                ("timeout", "55000"),
            ]);
        let resp = req.send().await.map_err(|e| format!("scrapingbee: {e}"))?;
        let status_api = resp.status().as_u16();
        // ScrapingBee surfaces the *origin* status via the `Spb-Original-Status`
        // header. When absent we fall back to the API status, which is a
        // close-enough approximation (ScrapingBee returns 200 for any
        // upstream 2xx-3xx, and mirrors most error codes).
        let origin_status = resp
            .headers()
            .get("Spb-Original-Status")
            .and_then(|v| v.to_str().ok())
            .and_then(|s| s.parse::<u16>().ok())
            .unwrap_or(status_api);
        let final_url = resp
            .headers()
            .get("Spb-Resolved-Url")
            .and_then(|v| v.to_str().ok())
            .map(|s| s.to_string())
            .unwrap_or_else(|| url.to_string());
        let content_type = resp
            .headers()
            .get("Spb-Original-Content-Type")
            .and_then(|v| v.to_str().ok())
            .map(|s| s.to_string())
            .unwrap_or_default();
        if !(200..400).contains(&status_api) {
            // Pull a tail of the body so the cascade summary contains the
            // ScrapingBee error message, not just the bare status code.
            let body = resp.text().await.unwrap_or_default();
            let preview: String = body.chars().take(200).collect();
            return Err(format!("scrapingbee: HTTP {status_api} — {preview}"));
        }
        let body = resp.text().await.map_err(|e| format!("scrapingbee: {e}"))?;
        Ok(ScrapeResult {
            status: origin_status,
            final_url,
            content_type,
            body,
            provider: "scrapingbee",
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use wiremock::matchers::{method, path};
    use wiremock::{Mock, MockServer, ResponseTemplate};

    fn test_pool() -> Arc<KeyPool> {
        Arc::new(KeyPool::from_keys(vec!["test-key".into()]))
    }

    #[tokio::test]
    async fn successful_scrape() {
        let mock = MockServer::start().await;
        Mock::given(method("GET"))
            .and(path("/api/v1/"))
            .respond_with(
                ResponseTemplate::new(200)
                    .set_body_string("<html><body>Hello</body></html>")
                    .append_header("content-type", "text/html")
                    .append_header("Spb-resolved-url", "https://example.com/page"),
            )
            .mount(&mock)
            .await;

        let engine = ScrapingBeeEngine::with_base_url(test_pool(), mock.uri());
        let result = engine.scrape("https://example.com/page").await.unwrap();
        assert_eq!(result.status, 200);
        assert!(result.body.contains("Hello"));
        assert_eq!(result.provider, "scrapingbee");
    }

    #[tokio::test]
    async fn unauthorized_marks_key_dead() {
        let mock = MockServer::start().await;
        Mock::given(method("GET"))
            .and(path("/api/v1/"))
            .respond_with(ResponseTemplate::new(401))
            .mount(&mock)
            .await;

        let engine = ScrapingBeeEngine::with_base_url(test_pool(), mock.uri());
        let err = engine.scrape("https://example.com").await.unwrap_err();
        assert!(err.contains("dead") || err.contains("401"));
    }
}
