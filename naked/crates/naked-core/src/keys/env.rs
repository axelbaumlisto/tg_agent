//! [`EnvKeyProvider`] — legacy fallback that parses comma-separated env vars
//! such as `EXA_API_KEYS`, `TAVILY_API_KEYS`, `SERPAPI_KEYS`,
//! `FIRECRAWL_API_KEYS`, `SCRAPINGBEE_API_KEYS`, `PERPLEXITY_API_KEYS`,
//! `BRIGHTDATA_API_KEYS`. Designed for hosts that don't ship the standalone
//! JSON pool from [`super::fs::FilesystemKeyProvider`].

use super::KeyProvider;

#[derive(Debug, Clone)]
pub struct EnvKeyProvider {
    var_map: Vec<(&'static str, &'static str)>,
}

impl EnvKeyProvider {
    /// Map every supported provider to its conventional env-var name.
    pub fn defaults() -> Self {
        Self {
            var_map: vec![
                ("exa", "EXA_API_KEYS"),
                ("tavily", "TAVILY_API_KEYS"),
                ("serpapi", "SERPAPI_KEYS"),
                ("perplexity", "PERPLEXITY_API_KEYS"),
                ("firecrawl", "FIRECRAWL_API_KEYS"),
                ("scrapingbee", "SCRAPINGBEE_API_KEYS"),
                ("brightdata", "BRIGHTDATA_API_KEYS"),
            ],
        }
    }

    /// Construct with a custom `(key_type, env_var)` mapping. Useful for
    /// tests that need to avoid touching the global env.
    pub fn with_map(var_map: Vec<(&'static str, &'static str)>) -> Self {
        Self { var_map }
    }
}

impl KeyProvider for EnvKeyProvider {
    fn fetch(&self, key_type: &str) -> Result<Vec<String>, String> {
        for (t, var) in &self.var_map {
            if *t == key_type {
                return Ok(std::env::var(var)
                    .unwrap_or_default()
                    .split(',')
                    .map(|s| s.trim().to_string())
                    .filter(|s| !s.is_empty())
                    .collect());
            }
        }
        Ok(Vec::new())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn unknown_type_returns_empty() {
        let p = EnvKeyProvider::with_map(vec![("foo", "_NAKED_TEST_NEVER_SET_VAR_X1")]);
        assert!(p.fetch("foo").unwrap().is_empty());
        assert!(p.fetch("bar").unwrap().is_empty());
    }
}
