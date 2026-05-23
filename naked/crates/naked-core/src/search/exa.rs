//! Exa.ai (`api.exa.ai`) `SearchEngine` adapter. Posts to `/search` with
//! `numResults` + `livecrawl=fallback` so out-of-index URLs still come back
//! with a fresh crawl.

use std::sync::Arc;
use std::time::Duration;

use async_trait::async_trait;

use crate::keys::pool::KeyPool;

use super::{SearchEngine, SearchHit};

pub struct ExaEngine {
    pool: Arc<KeyPool>,
    client: reqwest::Client,
    base_url: String,
}

impl ExaEngine {
    pub fn new(pool: Arc<KeyPool>) -> Self {
        Self::with_base_url(pool, "https://api.exa.ai".into())
    }

    pub fn with_base_url(pool: Arc<KeyPool>, base_url: String) -> Self {
        Self {
            pool,
            client: reqwest::Client::builder()
                .timeout(Duration::from_secs(15))
                .build()
                .unwrap_or_default(),
            base_url,
        }
    }
}

#[async_trait]
impl SearchEngine for ExaEngine {
    fn name(&self) -> &'static str {
        "exa"
    }

    async fn search(&self, query: &str, num: usize) -> Result<Vec<SearchHit>, String> {
        let key = self.pool.next().ok_or("exa: key pool empty")?;

        let body = serde_json::json!({
            "query": query,
            "numResults": num,
            "type": "auto",
            "contents": {
                "text": { "maxCharacters": 800, "includeHtmlTags": false }
            },
            "livecrawl": "fallback"
        });

        let resp = self
            .client
            .post(format!("{}/search", self.base_url))
            .header("x-api-key", &key)
            .json(&body)
            .send()
            .await
            .map_err(|e| format!("exa: request error: {e}"))?;

        let status = resp.status();
        if status == reqwest::StatusCode::UNAUTHORIZED || status == reqwest::StatusCode::FORBIDDEN {
            self.pool.mark_dead(&key);
            return Err(format!("exa: dead key (HTTP {status}), marked dead"));
        }
        if !status.is_success() {
            return Err(format!("exa: HTTP {status}"));
        }

        let data: serde_json::Value = resp
            .json()
            .await
            .map_err(|e| format!("exa: parse error: {e}"))?;

        let mut out = Vec::new();
        if let Some(items) = data["results"].as_array() {
            for item in items.iter().take(num) {
                let url = item["url"].as_str().unwrap_or("").to_string();
                if url.is_empty() {
                    continue;
                }
                let title = item["title"].as_str().unwrap_or("").to_string();
                let raw = item["text"].as_str().unwrap_or("");
                let snippet = truncate_at_char_boundary(raw, 300);
                out.push(SearchHit {
                    url,
                    title,
                    snippet,
                    source_engine: "exa",
                });
            }
        }
        Ok(out)
    }
}

fn truncate_at_char_boundary(s: &str, max: usize) -> String {
    crate::util::head_truncate(s, max).to_string()
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
    async fn parses_successful_response() {
        let mock = MockServer::start().await;
        Mock::given(method("POST")).and(path("/search"))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "results": [
                    {"url": "https://example.com/a", "title": "Exa Result", "text": "Some text content here"},
                ]
            })))
            .mount(&mock).await;

        let engine = ExaEngine::with_base_url(test_pool(), mock.uri());
        let hits = engine.search("test", 5).await.unwrap();
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].source_engine, "exa");
        assert!(hits[0].snippet.contains("Some text"));
    }

    #[tokio::test]
    async fn forbidden_marks_key_dead() {
        let mock = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/search"))
            .respond_with(ResponseTemplate::new(403))
            .mount(&mock)
            .await;

        let engine = ExaEngine::with_base_url(test_pool(), mock.uri());
        let err = engine.search("test", 5).await.unwrap_err();
        assert!(err.contains("dead key"));
    }
}
