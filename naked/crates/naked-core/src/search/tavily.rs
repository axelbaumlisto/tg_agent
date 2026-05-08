//! Tavily (`api.tavily.com`) `SearchEngine` adapter. AI search with rich
//! per-result `content` snippets — usually 200-500 chars of cleaned text,
//! perfect for the snippet-fast-path in `research/coordinator.rs`.

use std::sync::Arc;
use std::time::Duration;

use async_trait::async_trait;

use crate::keys::pool::KeyPool;

use super::{SearchEngine, SearchHit};

pub struct TavilyEngine {
    pool: Arc<KeyPool>,
    client: reqwest::Client,
    base_url: String,
}

impl TavilyEngine {
    pub fn new(pool: Arc<KeyPool>) -> Self {
        Self::with_base_url(pool, "https://api.tavily.com".into())
    }

    /// Constructor with custom base URL (for testing with wiremock).
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
impl SearchEngine for TavilyEngine {
    fn name(&self) -> &'static str {
        "tavily"
    }

    async fn search(&self, query: &str, num: usize) -> Result<Vec<SearchHit>, String> {
        let key = self.pool.next().ok_or("tavily: key pool empty")?;
        let body = serde_json::json!({
            "api_key": key,
            "query": query,
            "max_results": num,
            "search_depth": "basic",
            "include_answer": false,
            "include_raw_content": false,
            "include_images": false,
        });

        let resp = self
            .client
            .post(format!("{}/search", self.base_url))
            .json(&body)
            .send()
            .await
            .map_err(|e| format!("tavily: request error: {e}"))?;

        let status = resp.status();
        if status == reqwest::StatusCode::UNAUTHORIZED || status == reqwest::StatusCode::FORBIDDEN {
            self.pool.mark_dead(&key);
            return Err(format!("tavily: dead key (HTTP {status}), marked dead"));
        }
        if !status.is_success() {
            return Err(format!("tavily: HTTP {status}"));
        }

        let data: serde_json::Value = resp
            .json()
            .await
            .map_err(|e| format!("tavily: parse error: {e}"))?;

        let mut out = Vec::new();
        if let Some(items) = data["results"].as_array() {
            for item in items.iter().take(num) {
                let url = item["url"].as_str().unwrap_or("").to_string();
                if url.is_empty() {
                    continue;
                }
                let title = item["title"].as_str().unwrap_or("").to_string();
                let snippet = item["content"].as_str().unwrap_or("").to_string();
                out.push(SearchHit {
                    url,
                    title,
                    snippet,
                    source_engine: "tavily",
                });
            }
        }
        Ok(out)
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
    async fn parses_successful_response() {
        let mock = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/search"))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "results": [
                    {"url": "https://example.com/1", "title": "Result 1", "content": "Snippet 1"},
                    {"url": "https://example.com/2", "title": "Result 2", "content": "Snippet 2"},
                ]
            })))
            .mount(&mock)
            .await;

        let engine = TavilyEngine::with_base_url(test_pool(), mock.uri());
        let hits = engine.search("test query", 5).await.unwrap();
        assert_eq!(hits.len(), 2);
        assert_eq!(hits[0].url, "https://example.com/1");
        assert_eq!(hits[0].source_engine, "tavily");
    }

    #[tokio::test]
    async fn empty_results() {
        let mock = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/search"))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "results": []
            })))
            .mount(&mock)
            .await;

        let engine = TavilyEngine::with_base_url(test_pool(), mock.uri());
        let hits = engine.search("nothing", 5).await.unwrap();
        assert!(hits.is_empty());
    }

    #[tokio::test]
    async fn unauthorized_marks_key_dead() {
        let mock = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/search"))
            .respond_with(ResponseTemplate::new(401))
            .mount(&mock)
            .await;

        let pool = test_pool();
        let engine = TavilyEngine::with_base_url(pool.clone(), mock.uri());
        let err = engine.search("test", 5).await.unwrap_err();
        assert!(err.contains("dead key"));
    }

    #[tokio::test]
    async fn server_error_returns_err() {
        let mock = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/search"))
            .respond_with(ResponseTemplate::new(500))
            .mount(&mock)
            .await;

        let engine = TavilyEngine::with_base_url(test_pool(), mock.uri());
        let err = engine.search("test", 5).await.unwrap_err();
        assert!(err.contains("500"));
    }
}
