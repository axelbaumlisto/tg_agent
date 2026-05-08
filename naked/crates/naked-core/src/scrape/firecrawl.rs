//! Firecrawl cloud-scraping backend.
//!
//! API docs: <https://docs.firecrawl.dev/api-reference/endpoint/scrape>
//!
//! Endpoint: `POST https://api.firecrawl.dev/v1/scrape`
//! Body: `{ "url": "...", "formats": ["html"], "onlyMainContent": false }`
//! Auth: `Authorization: Bearer <api_key>`
//!
//! Why secondary to ScrapingBee:
//! - Smaller free tier (Firecrawl: 500 credits/mo, ScrapingBee: 1000).
//! - Slower (always JS-renders, even for static pages).
//! - But: better at SPA Vietnamese real-estate sites (e.g. dotproperty)
//!   where the listing details are stamped in only after a `fetch()` call.
//!
//! Status codes:
//! - 200: `{ success: true, data: { html, metadata } }`
//! - 401: invalid api_key
//! - 402: out of credits
//! - 429: rate limited
//! - 500: Firecrawl internal

use std::sync::Arc;
use std::time::Duration;

use async_trait::async_trait;
use reqwest::Client;
use serde_json::json;

use super::{CloudScraper, ScrapeResult};
use crate::keys::pool::KeyPool;

const FIRECRAWL_TIMEOUT_SECS: u64 = 90;

pub struct FirecrawlEngine {
    client: Client,
    keys: Arc<KeyPool>,
    base_url: String,
}

impl FirecrawlEngine {
    pub fn new(keys: Arc<KeyPool>) -> Self {
        Self::with_base_url(keys, "https://api.firecrawl.dev".into())
    }

    pub fn with_base_url(keys: Arc<KeyPool>, base_url: String) -> Self {
        let client = Client::builder()
            .timeout(Duration::from_secs(FIRECRAWL_TIMEOUT_SECS))
            .build()
            .expect("firecrawl: reqwest client init");
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
impl CloudScraper for FirecrawlEngine {
    fn name(&self) -> &'static str {
        "firecrawl"
    }

    async fn scrape(&self, url: &str) -> Result<ScrapeResult, String> {
        let mut last_err = String::from("firecrawl: pool empty");
        for attempt in 0..2 {
            let api_key = match self.keys.next() {
                Some(k) => k,
                None => return Err(last_err),
            };
            match self.scrape_once(&api_key, url).await {
                Ok(res) => return Ok(res),
                Err(e) => {
                    if e.contains("401") || e.contains("402") || e.contains("429") {
                        tracing::warn!(
                            attempt, error = %e,
                            "firecrawl: key rejected, marking dead and retrying"
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

impl FirecrawlEngine {
    async fn scrape_once(&self, api_key: &str, url: &str) -> Result<ScrapeResult, String> {
        let resp = self
            .client
            .post(format!("{}/v1/scrape", self.base_url))
            .bearer_auth(api_key)
            .json(&json!({
                "url": url,
                "formats": ["html"],
                "onlyMainContent": false,
                "waitFor": 1500,
                "timeout": 60000,
            }))
            .send()
            .await
            .map_err(|e| format!("firecrawl: {e}"))?;
        let status = resp.status().as_u16();
        if !(200..400).contains(&status) {
            let body = resp.text().await.unwrap_or_default();
            let preview: String = body.chars().take(200).collect();
            return Err(format!("firecrawl: HTTP {status} — {preview}"));
        }
        let envelope: serde_json::Value =
            resp.json().await.map_err(|e| format!("firecrawl: {e}"))?;
        if !envelope
            .get("success")
            .and_then(|v| v.as_bool())
            .unwrap_or(false)
        {
            let err_msg = envelope
                .get("error")
                .and_then(|v| v.as_str())
                .unwrap_or("unknown");
            return Err(format!("firecrawl: payload not successful — {err_msg}"));
        }
        let data = envelope
            .get("data")
            .ok_or("firecrawl: missing data field")?;
        let body = data
            .get("html")
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .to_string();
        let final_url = data
            .get("metadata")
            .and_then(|m| m.get("sourceURL"))
            .and_then(|v| v.as_str())
            .map(|s| s.to_string())
            .unwrap_or_else(|| url.to_string());
        let content_type = data
            .get("metadata")
            .and_then(|m| m.get("contentType"))
            .and_then(|v| v.as_str())
            .map(|s| s.to_string())
            .unwrap_or_else(|| "text/html".to_string());
        let origin_status = data
            .get("metadata")
            .and_then(|m| m.get("statusCode"))
            .and_then(|v| v.as_u64())
            .unwrap_or(200) as u16;
        Ok(ScrapeResult {
            status: origin_status,
            final_url,
            content_type,
            body,
            provider: "firecrawl",
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
        Mock::given(method("POST"))
            .and(path("/v1/scrape"))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "success": true,
                "data": {
                    "html": "<html><body>Page content</body></html>",
                    "metadata": {
                        "statusCode": 200,
                        "sourceURL": "https://example.com"
                    }
                }
            })))
            .mount(&mock)
            .await;

        let engine = FirecrawlEngine::with_base_url(test_pool(), mock.uri());
        let result = engine.scrape("https://example.com").await.unwrap();
        assert!(result.body.contains("Page content"));
        assert_eq!(result.provider, "firecrawl");
    }

    #[tokio::test]
    async fn api_failure_returns_err() {
        let mock = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/v1/scrape"))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "success": false,
                "error": "Rate limit exceeded"
            })))
            .mount(&mock)
            .await;

        let engine = FirecrawlEngine::with_base_url(test_pool(), mock.uri());
        let err = engine.scrape("https://example.com").await.unwrap_err();
        assert!(err.contains("Rate limit") || err.contains("fail"));
    }
}
