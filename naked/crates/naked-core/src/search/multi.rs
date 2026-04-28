//! Parallel fan-out over [`SearchEngine`] backends with URL-dedup.

use std::collections::HashSet;

use futures_util::future::join_all;

use super::{SearchEngine, SearchHit};

pub struct MultiEngineSearch {
    engines: Vec<Box<dyn SearchEngine>>,
}

impl MultiEngineSearch {
    pub fn new(engines: Vec<Box<dyn SearchEngine>>) -> Self {
        Self { engines }
    }

    pub fn engine_count(&self) -> usize {
        self.engines.len()
    }

    pub fn engine_names(&self) -> Vec<&'static str> {
        self.engines.iter().map(|e| e.name()).collect()
    }

    /// Run `query` against every engine in parallel. Per-engine errors are
    /// logged but never bubble — a single bad backend does not abort the
    /// search. Results are dedup'd by URL, preserving first-seen order
    /// (and, crucially, first-seen `source_engine` for telemetry).
    pub async fn search(&self, query: &str, num: usize) -> Vec<SearchHit> {
        let futs = self.engines.iter().map(|e| {
            let q = query.to_string();
            async move { (e.name(), e.search(&q, num).await) }
        });
        let results = join_all(futs).await;

        let mut seen: HashSet<String> = HashSet::new();
        let mut out: Vec<SearchHit> = Vec::new();
        for (name, res) in results {
            match res {
                Ok(hits) => {
                    for h in hits {
                        if seen.insert(h.url.clone()) {
                            out.push(h);
                        }
                    }
                }
                Err(e) => tracing::warn!(engine = %name, "search engine failed: {e}"),
            }
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use async_trait::async_trait;

    struct DummyEngine {
        name: &'static str,
        hits: Vec<SearchHit>,
    }

    #[async_trait]
    impl SearchEngine for DummyEngine {
        fn name(&self) -> &'static str {
            self.name
        }
        async fn search(&self, _q: &str, _n: usize) -> Result<Vec<SearchHit>, String> {
            Ok(self.hits.clone())
        }
    }

    struct FailingEngine;
    #[async_trait]
    impl SearchEngine for FailingEngine {
        fn name(&self) -> &'static str {
            "failing"
        }
        async fn search(&self, _q: &str, _n: usize) -> Result<Vec<SearchHit>, String> {
            Err("boom".into())
        }
    }

    fn hit(url: &str, engine: &'static str) -> SearchHit {
        SearchHit {
            url: url.into(),
            title: "T".into(),
            snippet: "S".into(),
            source_engine: engine,
        }
    }

    #[tokio::test]
    async fn dedup_by_url_keeps_first_seen_engine() {
        let a = DummyEngine {
            name: "a",
            hits: vec![hit("https://x.com/1", "a")],
        };
        let b = DummyEngine {
            name: "b",
            hits: vec![hit("https://x.com/1", "b"), hit("https://x.com/2", "b")],
        };
        let multi = MultiEngineSearch::new(vec![Box::new(a), Box::new(b)]);
        let hits = multi.search("q", 10).await;
        assert_eq!(hits.len(), 2);
        assert_eq!(hits[0].url, "https://x.com/1");
        assert_eq!(hits[0].source_engine, "a");
        assert_eq!(hits[1].url, "https://x.com/2");
    }

    #[tokio::test]
    async fn failing_engine_does_not_break_others() {
        let multi = MultiEngineSearch::new(vec![
            Box::new(FailingEngine),
            Box::new(DummyEngine {
                name: "good",
                hits: vec![hit("https://x.com/1", "good")],
            }),
        ]);
        let hits = multi.search("q", 5).await;
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].source_engine, "good");
    }

    #[tokio::test]
    async fn no_engines_returns_empty() {
        let multi = MultiEngineSearch::new(Vec::new());
        let hits = multi.search("q", 5).await;
        assert!(hits.is_empty());
    }
}
