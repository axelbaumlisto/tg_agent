//! Search key pool and cloud-scraper infrastructure.
//!
//! Extracted from lib.rs — owns SearchState construction and lifecycle.

use std::sync::Arc;

use crate::keys::pool::KeyPool;

pub(crate) struct SearchState {
    pub exa_key_pool: Arc<KeyPool>,
    pub tavily_key_pool: Arc<KeyPool>,
    pub serpapi_key_pool: Arc<KeyPool>,
    pub cloud_scraper: Option<Arc<crate::scrape::multi::MultiCloudScraper>>,
    pub host_policy: Arc<crate::scrape::host_policy::HostPolicy>,
}

impl SearchState {
    /// Build from config. Reads key pools from disk + env, initialises scraper cascade.
    pub fn from_config(legacy_exa_keys: &[String]) -> Self {
        let (exa, tavily, serpapi) = build_search_key_pools(legacy_exa_keys);
        let cloud_scraper = build_cloud_scraper();
        let host_policy = Arc::new(crate::scrape::host_policy::HostPolicy::new());
        Self {
            exa_key_pool: exa,
            tavily_key_pool: tavily,
            serpapi_key_pool: serpapi,
            cloud_scraper,
            host_policy,
        }
    }
}

fn build_search_key_pools(
    legacy_exa_keys: &[String],
) -> (Arc<KeyPool>, Arc<KeyPool>, Arc<KeyPool>) {
    use crate::keys::KeyProvider;
    use crate::keys::env::EnvKeyProvider;
    use crate::keys::fs::FilesystemKeyProvider;
    use std::time::Duration;

    let fs_path = FilesystemKeyProvider::default_path();
    let fs_provider: Arc<dyn KeyProvider> = Arc::new(FilesystemKeyProvider::new(fs_path.clone()));
    let env_provider: Arc<dyn KeyProvider> = Arc::new(EnvKeyProvider::defaults());

    // Bridge legacy `Config::exa_api_keys` (set from `.env` at startup) into
    // the Exa pool so a host without the JSON file still gets its keys.
    struct StaticProvider {
        keys: Vec<String>,
    }
    impl KeyProvider for StaticProvider {
        fn fetch(&self, _: &str) -> std::result::Result<Vec<String>, String> {
            Ok(self.keys.clone())
        }
    }
    let legacy_exa: Arc<dyn KeyProvider> = Arc::new(StaticProvider {
        keys: legacy_exa_keys.to_vec(),
    });

    let ttl = Duration::from_secs(6 * 3600);

    let exa = Arc::new(KeyPool::new(
        vec![fs_provider.clone(), env_provider.clone(), legacy_exa],
        "exa",
        ttl,
    ));
    let tavily = Arc::new(KeyPool::new(
        vec![fs_provider.clone(), env_provider.clone()],
        "tavily",
        ttl,
    ));
    let serpapi = Arc::new(KeyPool::new(
        vec![fs_provider.clone(), env_provider.clone()],
        "serpapi",
        ttl,
    ));

    tracing::info!(
        pool_path = %fs_path.display(),
        exa = exa.size(),
        tavily = tavily.size(),
        serpapi = serpapi.size(),
        "search key pools initialized"
    );
    (exa, tavily, serpapi)
}

/// Build the optional cloud-scrape cascade for [`WebFetchTool`] Tier 3.5.
///
/// Reads ScrapingBee + Firecrawl keys from the same `~/.naked/secrets/
/// search_pool.alive.json` source as the search engines, with the
/// `SCRAPINGBEE_API_KEYS` / `FIRECRAWL_API_KEYS` CSV env-vars as
/// fallback. Returns `None` when no keys are available so the cascade
/// silently skips the tier — the existing 4-tier path still works on
/// hosts that haven't provisioned cloud-scrape credentials.
fn build_cloud_scraper() -> Option<Arc<crate::scrape::multi::MultiCloudScraper>> {
    use crate::keys::KeyProvider;
    use crate::keys::env::EnvKeyProvider;
    use crate::keys::fs::FilesystemKeyProvider;
    use crate::scrape::CloudScraper;
    use crate::scrape::firecrawl::FirecrawlEngine;
    use crate::scrape::multi::MultiCloudScraper;
    use crate::scrape::scrapingbee::ScrapingBeeEngine;
    use std::time::Duration;

    let fs_provider: Arc<dyn KeyProvider> = Arc::new(FilesystemKeyProvider::new(
        FilesystemKeyProvider::default_path(),
    ));
    let env_provider: Arc<dyn KeyProvider> = Arc::new(EnvKeyProvider::defaults());
    let ttl = Duration::from_secs(6 * 3600);

    let scrapingbee_pool = Arc::new(KeyPool::new(
        vec![fs_provider.clone(), env_provider.clone()],
        "scrapingbee",
        ttl,
    ));
    let firecrawl_pool = Arc::new(KeyPool::new(
        vec![fs_provider, env_provider],
        "firecrawl",
        ttl,
    ));

    let mut engines: Vec<Arc<dyn CloudScraper>> = Vec::new();
    if scrapingbee_pool.size() > 0 {
        engines.push(Arc::new(ScrapingBeeEngine::new(scrapingbee_pool.clone())));
    }
    if firecrawl_pool.size() > 0 {
        engines.push(Arc::new(FirecrawlEngine::new(firecrawl_pool.clone())));
    }

    if engines.is_empty() {
        tracing::info!("cloud-scrape: no keys available, Tier 3.5 disabled");
        return None;
    }

    let summary = engines
        .iter()
        .map(|e| e.name().to_string())
        .collect::<Vec<_>>()
        .join(",");
    tracing::info!(
        engines = %summary,
        scrapingbee_keys = scrapingbee_pool.size(),
        firecrawl_keys = firecrawl_pool.size(),
        "cloud-scrape: cascade initialized",
    );
    Some(Arc::new(MultiCloudScraper::new(engines)))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn search_state_builds_with_empty_keys() {
        // Verify construction doesn't panic. Pool size may be >0
        // because EnvKeyProvider / FilesystemKeyProvider discover
        // real keys from the host environment (DDG instances, .env,
        // secrets JSON). The legacy_exa_keys slice is just ONE of
        // several providers.
        let state = SearchState::from_config(&[]);
        // Smoke: the state is usable (host_policy constructed, pools alive).
        let _ = state.exa_key_pool.size();
        let _ = state.tavily_key_pool.size();
        let _ = state.serpapi_key_pool.size();
    }

    #[test]
    fn search_state_builds_with_some_keys() {
        let keys = vec!["test-key-1".to_string(), "test-key-2".to_string()];
        let state = SearchState::from_config(&keys);
        // Legacy keys are merged INTO the exa pool alongside env/fs keys.
        // At minimum, the 2 we passed must be present.
        assert!(
            state.exa_key_pool.size() >= 2,
            "exa pool must contain at least the legacy keys: got {}",
            state.exa_key_pool.size()
        );
    }
}
