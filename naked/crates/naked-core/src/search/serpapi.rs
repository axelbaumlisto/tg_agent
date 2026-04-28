//! SerpAPI (`serpapi.com`) `SearchEngine` adapter — real Google SERPs with
//! `organic_results[].link / title / snippet`. Use this when you need
//! semantically-ranked results from the source-of-truth search engine, not
//! Tavily's AI-curated take.

use std::sync::Arc;
use std::time::Duration;

use async_trait::async_trait;

use crate::keys::pool::KeyPool;

use super::{SearchEngine, SearchHit};

pub struct SerpApiEngine {
    pool: Arc<KeyPool>,
    client: reqwest::Client,
}

impl SerpApiEngine {
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
impl SearchEngine for SerpApiEngine {
    fn name(&self) -> &'static str {
        "serpapi"
    }

    async fn search(&self, query: &str, num: usize) -> Result<Vec<SearchHit>, String> {
        let key = self.pool.next().ok_or("serpapi: key pool empty")?;

        let resp = self
            .client
            .get("https://serpapi.com/search")
            .query(&[
                ("api_key", key.as_str()),
                ("engine", "google"),
                ("q", query),
                ("num", &num.to_string()),
                ("hl", "vi"),
                ("gl", "vn"),
            ])
            .send()
            .await
            .map_err(|e| format!("serpapi: request error: {e}"))?;

        let status = resp.status();
        if status == reqwest::StatusCode::UNAUTHORIZED || status == reqwest::StatusCode::FORBIDDEN {
            self.pool.mark_dead(&key);
            return Err(format!("serpapi: dead key (HTTP {status}), marked dead"));
        }
        if !status.is_success() {
            return Err(format!("serpapi: HTTP {status}"));
        }

        let data: serde_json::Value = resp
            .json()
            .await
            .map_err(|e| format!("serpapi: parse error: {e}"))?;

        // SerpAPI returns 200 + {"error":"…"} for invalid keys
        if let Some(err) = data.get("error").and_then(|v| v.as_str()) {
            if err.contains("Invalid") || err.contains("invalid") {
                self.pool.mark_dead(&key);
            }
            return Err(format!("serpapi: api error: {err}"));
        }

        let mut out = Vec::new();
        if let Some(items) = data["organic_results"].as_array() {
            for item in items.iter().take(num) {
                let url = item["link"].as_str().unwrap_or("").to_string();
                if url.is_empty() {
                    continue;
                }
                let title = item["title"].as_str().unwrap_or("").to_string();
                let snippet = item["snippet"].as_str().unwrap_or("").to_string();
                out.push(SearchHit {
                    url,
                    title,
                    snippet,
                    source_engine: "serpapi",
                });
            }
        }
        Ok(out)
    }
}
