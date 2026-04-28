//! Multi-engine search abstraction.
//!
//! [`SearchEngine`] is the trait every backend implements; [`MultiEngineSearch`]
//! fans a single query out to all enabled engines in parallel and dedups the
//! results by URL.
//!
//! Engines:
//! - [`exa::ExaEngine`]      — semantic + full-text via api.exa.ai
//! - [`tavily::TavilyEngine`] — AI-curated snippets via api.tavily.com
//! - [`serpapi::SerpApiEngine`] — real Google SERPs via serpapi.com
//! - [`ddg::DdgEngine`]      — DuckDuckGo HTML scrape, no key needed (legacy fallback)

pub mod ddg;
pub mod exa;
pub mod multi;
pub mod serpapi;
pub mod snippet;
pub mod tavily;

use async_trait::async_trait;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SearchHit {
    pub url: String,
    pub title: String,
    pub snippet: String,
    pub source_engine: &'static str,
}

#[async_trait]
pub trait SearchEngine: Send + Sync {
    fn name(&self) -> &'static str;
    async fn search(&self, query: &str, num: usize) -> Result<Vec<SearchHit>, String>;
}
