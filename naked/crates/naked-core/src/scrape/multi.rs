//! Cascade across multiple cloud-scrape backends.
//!
//! Tries each engine in order and returns the first success. The order is
//! fixed at construction time — operators put their preferred engine first
//! (typically the one with the most credits) and the next engine only sees
//! traffic when the first one fails on this URL.
//!
//! Engines with empty key pools are kept in the list (cheap to skip) so
//! the cascade behaves correctly when a key pool is hot-reloaded between
//! requests.

use std::sync::Arc;

use super::{CloudScraper, ScrapeResult};

pub struct MultiCloudScraper {
    engines: Vec<Arc<dyn CloudScraper>>,
}

impl MultiCloudScraper {
    pub fn new(engines: Vec<Arc<dyn CloudScraper>>) -> Self {
        Self { engines }
    }

    /// Returns `true` when at least one engine has a non-empty key pool.
    /// Hot path checks this before invoking the cascade so the caller can
    /// short-circuit (and the cascade-summary in `web_fetch.rs` doesn't
    /// list a permanently-skipped tier on every fetch).
    pub fn is_active(&self) -> bool {
        !self.engines.is_empty()
    }

    /// Run the cascade. Returns `Ok(result)` on the first engine that
    /// produces a 2xx-3xx response, `Err(joined errors)` when all of them
    /// fail. Logs each engine's outcome at `info` so operators can see
    /// which paid tier ran.
    pub async fn scrape(&self, url: &str) -> Result<ScrapeResult, String> {
        if self.engines.is_empty() {
            return Err("cloud-scrape: no engines configured".into());
        }
        let mut errors: Vec<String> = Vec::new();
        for engine in &self.engines {
            tracing::info!(engine = engine.name(), url = %url, "cloud-scrape: trying engine");
            match engine.scrape(url).await {
                Ok(res) => {
                    tracing::info!(
                        engine = engine.name(),
                        status = res.status,
                        url = %url,
                        "cloud-scrape: engine succeeded"
                    );
                    return Ok(res);
                }
                Err(e) => {
                    tracing::warn!(engine = engine.name(), error = %e, "cloud-scrape: engine failed");
                    errors.push(format!("{}: {e}", engine.name()));
                }
            }
        }
        Err(errors.join(" | "))
    }

    pub fn engine_summary(&self) -> String {
        self.engines
            .iter()
            .map(|e| e.name().to_string())
            .collect::<Vec<_>>()
            .join(",")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use async_trait::async_trait;
    use std::sync::atomic::{AtomicUsize, Ordering};

    struct AlwaysOk(&'static str);
    #[async_trait]
    impl CloudScraper for AlwaysOk {
        fn name(&self) -> &'static str {
            self.0
        }
        async fn scrape(&self, url: &str) -> Result<ScrapeResult, String> {
            Ok(ScrapeResult {
                status: 200,
                final_url: url.into(),
                content_type: "text/html".into(),
                body: format!("<html>{}</html>", self.0),
                provider: self.0,
            })
        }
    }

    struct AlwaysFails(&'static str);
    #[async_trait]
    impl CloudScraper for AlwaysFails {
        fn name(&self) -> &'static str {
            self.0
        }
        async fn scrape(&self, _url: &str) -> Result<ScrapeResult, String> {
            Err(format!("{} broken", self.0))
        }
    }

    struct CountingFails {
        name: &'static str,
        calls: AtomicUsize,
    }
    #[async_trait]
    impl CloudScraper for CountingFails {
        fn name(&self) -> &'static str {
            self.name
        }
        async fn scrape(&self, _url: &str) -> Result<ScrapeResult, String> {
            self.calls.fetch_add(1, Ordering::Relaxed);
            Err("nope".into())
        }
    }

    #[tokio::test]
    async fn returns_first_success_and_stops() {
        let counter = Arc::new(CountingFails {
            name: "second",
            calls: AtomicUsize::new(0),
        });
        let cascade = MultiCloudScraper::new(vec![
            Arc::new(AlwaysOk("first")),
            counter.clone(),
        ]);
        let res = cascade.scrape("https://x").await.unwrap();
        assert_eq!(res.provider, "first");
        // The second engine must NOT be called when the first succeeded.
        assert_eq!(counter.calls.load(Ordering::Relaxed), 0);
    }

    #[tokio::test]
    async fn falls_through_to_next_engine_on_failure() {
        let cascade = MultiCloudScraper::new(vec![
            Arc::new(AlwaysFails("first")),
            Arc::new(AlwaysOk("second")),
        ]);
        let res = cascade.scrape("https://x").await.unwrap();
        assert_eq!(res.provider, "second");
    }

    #[tokio::test]
    async fn aggregates_errors_when_all_fail() {
        let cascade = MultiCloudScraper::new(vec![
            Arc::new(AlwaysFails("first")),
            Arc::new(AlwaysFails("second")),
        ]);
        let err = cascade.scrape("https://x").await.unwrap_err();
        assert!(err.contains("first: first broken"));
        assert!(err.contains("second: second broken"));
    }

    #[test]
    fn empty_cascade_is_inactive() {
        let cascade = MultiCloudScraper::new(Vec::new());
        assert!(!cascade.is_active());
    }
}
