//! [`FilesystemKeyProvider`] — reads the standalone JSON pool produced by
//! `naked/scripts/import_search_keys.py`.
//!
//! Schema (subset, see script for full layout):
//!
//! ```json
//! {
//!   "version": 1,
//!   "validated_at": "2026-04-23T11:12:35Z",
//!   "keys": {
//!     "tavily": ["tvly-…", …],
//!     "serpapi": ["…"],
//!     "exa": [],
//!     …
//!   }
//! }
//! ```
//!
//! Missing file → empty vector (NOT an error). Malformed JSON or wrong type
//! at `keys.<type>` → empty vector with a `tracing::warn!` so operators see it
//! once at refresh time without crashing the bot.

use std::path::PathBuf;

use serde::Deserialize;

use super::KeyProvider;

#[derive(Debug, Clone)]
pub struct FilesystemKeyProvider {
    path: PathBuf,
}

impl FilesystemKeyProvider {
    pub fn new(path: PathBuf) -> Self {
        Self { path }
    }

    /// Convenience: default location used by the importer script.
    /// Returns `~/.naked/secrets/search_pool.alive.json`, or — when no `HOME`
    /// is set — the relative path with the same suffix.
    pub fn default_path() -> PathBuf {
        std::env::var_os("HOME")
            .map(|h| PathBuf::from(h).join(".naked/secrets/search_pool.alive.json"))
            .unwrap_or_else(|| PathBuf::from(".naked/secrets/search_pool.alive.json"))
    }
}

#[derive(Deserialize)]
struct PoolFile {
    keys: serde_json::Map<String, serde_json::Value>,
}

impl KeyProvider for FilesystemKeyProvider {
    fn fetch(&self, key_type: &str) -> Result<Vec<String>, String> {
        if !self.path.exists() {
            return Ok(Vec::new());
        }
        let raw = std::fs::read_to_string(&self.path)
            .map_err(|e| format!("read {}: {e}", self.path.display()))?;
        let parsed: PoolFile = match serde_json::from_str(&raw) {
            Ok(p) => p,
            Err(e) => {
                tracing::warn!(path = %self.path.display(), "key pool: malformed json: {e}");
                return Ok(Vec::new());
            }
        };
        let Some(arr) = parsed.keys.get(key_type).and_then(|v| v.as_array()) else {
            return Ok(Vec::new());
        };
        Ok(arr
            .iter()
            .filter_map(|v| v.as_str().map(|s| s.trim().to_string()))
            .filter(|s| !s.is_empty())
            .collect())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use tempfile::NamedTempFile;

    fn make_pool(json: &str) -> NamedTempFile {
        let mut f = NamedTempFile::new().unwrap();
        f.write_all(json.as_bytes()).unwrap();
        f
    }

    #[test]
    fn loads_keys_for_requested_type() {
        let f = make_pool(
            r#"{ "version": 1, "keys": {
                "tavily":  ["tvly-A", "tvly-B"],
                "exa":     ["3f5a-1"],
                "serpapi": []
            }}"#,
        );
        let p = FilesystemKeyProvider::new(f.path().to_owned());
        assert_eq!(p.fetch("tavily").unwrap(), vec!["tvly-A", "tvly-B"]);
        assert_eq!(p.fetch("exa").unwrap(), vec!["3f5a-1"]);
        assert!(p.fetch("serpapi").unwrap().is_empty());
        assert!(p.fetch("nonexistent_provider").unwrap().is_empty());
    }

    #[test]
    fn missing_file_returns_empty_not_error() {
        let p = FilesystemKeyProvider::new("/nonexistent/file.json".into());
        assert!(p.fetch("tavily").unwrap().is_empty());
    }

    #[test]
    fn malformed_json_returns_empty_not_error() {
        let f = make_pool(r#"{ this is not json"#);
        let p = FilesystemKeyProvider::new(f.path().to_owned());
        assert!(p.fetch("tavily").unwrap().is_empty());
    }

    #[test]
    fn empty_strings_are_filtered() {
        let f = make_pool(
            r#"{ "version": 1, "keys": {
                "tavily": ["", " tvly-X ", "  "]
            }}"#,
        );
        let p = FilesystemKeyProvider::new(f.path().to_owned());
        assert_eq!(p.fetch("tavily").unwrap(), vec!["tvly-X"]);
    }
}
