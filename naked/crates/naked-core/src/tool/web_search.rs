//! `web_search` tool — thin wrapper around [`crate::search::multi::MultiEngineSearch`].
//!
//! Wires together every keyed engine that has at least one usable key in its
//! pool, plus DDG as a no-key fallback. Output is unified — each result line
//! carries `(via <engine>)` so the agent can see which backend won.

use std::path::Path;
use std::sync::Arc;

use async_trait::async_trait;

use crate::keys::pool::KeyPool;
use crate::search::SearchEngine;
use crate::search::ddg::DdgEngine;
use crate::search::exa::ExaEngine;
use crate::search::multi::MultiEngineSearch;
use crate::search::serpapi::SerpApiEngine;
use crate::search::snippet::SnippetExtractor;
use crate::search::tavily::TavilyEngine;
use crate::types::{Permission, ToolResult, ToolSpec};

use super::Tool;

pub struct WebSearchTool {
    multi: MultiEngineSearch,
    engine_summary: String,
    extractor: SnippetExtractor,
}

impl WebSearchTool {
    /// Build the tool from per-provider [`KeyPool`]s. Engines whose pool is
    /// empty are simply not wired up. DDG is always added as a last-resort
    /// fallback — it costs nothing and works without keys.
    pub fn new(
        exa_pool: Arc<KeyPool>,
        tavily_pool: Arc<KeyPool>,
        serpapi_pool: Arc<KeyPool>,
    ) -> Self {
        let mut engines: Vec<Box<dyn SearchEngine>> = Vec::new();
        let mut summary_parts = Vec::new();

        let exa_size = exa_pool.size();
        if exa_size > 0 {
            engines.push(Box::new(ExaEngine::new(exa_pool)));
            summary_parts.push(format!("exa({exa_size})"));
        }
        let tavily_size = tavily_pool.size();
        if tavily_size > 0 {
            engines.push(Box::new(TavilyEngine::new(tavily_pool)));
            summary_parts.push(format!("tavily({tavily_size})"));
        }
        let serpapi_size = serpapi_pool.size();
        if serpapi_size > 0 {
            engines.push(Box::new(SerpApiEngine::new(serpapi_pool)));
            summary_parts.push(format!("serpapi({serpapi_size})"));
        }
        engines.push(Box::new(DdgEngine::new()));
        summary_parts.push("ddg".to_string());

        let engine_summary = summary_parts.join("+");
        tracing::info!(engines = %engine_summary, "WebSearchTool initialized");

        Self {
            multi: MultiEngineSearch::new(engines),
            engine_summary,
            extractor: SnippetExtractor::new(),
        }
    }

    /// Backwards-compat constructor used by the legacy code path that only
    /// had Exa keys from `Config::exa_api_keys`. Wraps them in an
    /// in-memory provider so existing callers keep working until every
    /// site is migrated to the new `Config::keys` block.
    pub fn from_legacy_exa(exa_keys: Vec<String>) -> Self {
        use crate::keys::KeyProvider;
        use std::time::Duration;

        struct StaticProvider {
            keys: Vec<String>,
        }
        impl KeyProvider for StaticProvider {
            fn fetch(&self, _: &str) -> Result<Vec<String>, String> {
                Ok(self.keys.clone())
            }
        }

        let pool = if exa_keys.is_empty() {
            Arc::new(KeyPool::empty("exa"))
        } else {
            Arc::new(KeyPool::new(
                vec![Arc::new(StaticProvider { keys: exa_keys })],
                "exa",
                Duration::from_secs(3600),
            ))
        };
        let empty_tavily = Arc::new(KeyPool::empty("tavily"));
        let empty_serp = Arc::new(KeyPool::empty("serpapi"));
        Self::new(pool, empty_tavily, empty_serp)
    }
}

#[async_trait]
impl Tool for WebSearchTool {
    fn spec(&self) -> ToolSpec {
        ToolSpec {
            name: "web_search".into(),
            description: format!(
                "Search the web for information. Engines (in parallel, dedup by URL): {}. \
                 Each result is tagged with the engine that returned it.",
                self.engine_summary
            ),
            parameters: serde_json::json!({
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search query"
                    },
                    "num_results": {
                        "type": "integer",
                        "description": "Number of results per engine (default: 5, max: 10)"
                    }
                },
                "required": ["query"]
            }),
            permission: Permission::ReadOnly,
        }
    }

    async fn execute(&self, input: serde_json::Value, _cwd: &Path) -> ToolResult {
        let query = match input.get("query").and_then(|v| v.as_str()) {
            Some(q) if !q.trim().is_empty() => q.trim(),
            _ => {
                return ToolResult {
                    output: "Error: 'query' field is required and must be non-empty".into(),
                    is_error: true,
                };
            }
        };
        let num = input
            .get("num_results")
            .and_then(|v| v.as_u64())
            .unwrap_or(5)
            .min(10) as usize;

        let hits = self.multi.search(query, num).await;
        if hits.is_empty() {
            return ToolResult {
                output: format!(
                    "[engines: {}] no results for: {query}",
                    self.multi.engine_count()
                ),
                is_error: true,
            };
        }

        let mut blocks = Vec::with_capacity(hits.len());
        let mut fast_path_count = 0usize;
        for h in &hits {
            let snippet = if h.snippet.is_empty() {
                String::new()
            } else {
                format!("\n  {}", truncate(&h.snippet, 320))
            };
            let combined = format!("{} {}", h.title, h.snippet);
            let extracted = self.extractor.extract(&combined);
            let extracted_line = match &extracted {
                Some(f) => {
                    fast_path_count += 1;
                    let phone = f.phone.as_deref().unwrap_or("—");
                    format!(
                        "\n  ⚡ extracted: price={} VND, area={} m², district={}, phone={}",
                        f.price_vnd_per_month, f.area_m2, f.district, phone
                    )
                }
                None => String::new(),
            };
            blocks.push(format!(
                "• {title}  (via {engine})\n  {url}{snippet}{extracted}",
                title = h.title,
                engine = h.source_engine,
                url = h.url,
                extracted = extracted_line,
            ));
        }
        if fast_path_count > 0 {
            tracing::info!(
                fast_path = fast_path_count,
                total = hits.len(),
                "snippet-fast-path: structured data extracted from search snippets"
            );
        }

        ToolResult {
            output: format!(
                "[engines: {}] {} results:\n\n{}",
                self.multi.engine_count(),
                hits.len(),
                blocks.join("\n\n")
            ),
            is_error: false,
        }
    }
}

fn truncate(s: &str, max: usize) -> String {
    if s.chars().count() <= max {
        return s.to_string();
    }
    let truncated: String = s.chars().take(max).collect();
    format!("{truncated}…")
}
