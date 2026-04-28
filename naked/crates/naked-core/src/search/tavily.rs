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
}

impl TavilyEngine {
    pub fn new(pool: Arc<KeyPool>) -> Self {
        Self {
            pool,
            client: reqwest::Client::builder()
                .timeout(Duration::from_secs(15))
                .build()
                .unwrap_or_default(),
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
            .post("https://api.tavily.com/search")
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
