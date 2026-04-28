//! API-key sourcing for search/scrape providers.
//!
//! Two implementations of [`KeyProvider`] live in this module:
//!
//! * [`fs::FilesystemKeyProvider`] — reads `~/.naked/secrets/search_pool.alive.json`
//!   produced by `naked/scripts/import_search_keys.py`. This is the **primary**
//!   source — it is portable, validated offline, and requires zero runtime
//!   dependencies on the source database.
//! * [`env::EnvKeyProvider`] — legacy fallback that parses comma-separated
//!   `EXA_API_KEYS` / `TAVILY_API_KEYS` / `SERPAPI_KEYS` / etc. Kept so an
//!   operator can hand-configure a single host without provisioning the JSON.
//!
//! [`pool::KeyPool`] composes any number of providers, dedups them, and serves
//! keys via a round-robin cursor with a configurable TTL refresh.

pub mod env;
pub mod fs;
pub mod pool;

/// Source of API keys keyed by provider type (`"exa"`, `"tavily"`, …).
///
/// Implementations MUST be cheap-to-clone and safe to call concurrently —
/// the pool may invoke `fetch` from multiple coordinator turns.
pub trait KeyProvider: Send + Sync {
    /// Return all currently-known keys for `key_type`. An empty vector is a
    /// valid response (no keys configured for this provider). Errors should
    /// only be returned for *unexpected* failures (corrupt file, bug); a
    /// missing source file is **not** an error.
    fn fetch(&self, key_type: &str) -> Result<Vec<String>, String>;
}
