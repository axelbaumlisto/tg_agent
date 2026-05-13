//! Persistent storage for the permissions ruleset.
//!
//! On-disk format: pretty-printed JSON at
//! `~/.naked/permissions.json`. Atomic writes (`tmp + rename`) so a
//! crash mid-write doesn't corrupt the file.

use std::path::PathBuf;

use super::ruleset::Ruleset;

#[must_use]
pub fn default_path() -> Option<PathBuf> {
    std::env::var_os("HOME")
        .map(PathBuf::from)
        .map(|h| h.join(".naked").join("permissions.json"))
}

#[derive(Debug, Default)]
pub struct Store;

impl Store {
    /// Load the ruleset from `~/.naked/permissions.json`. Missing
    /// file returns an empty ruleset; corrupt JSON returns empty
    /// with a `tracing::warn!`.
    pub fn load() -> Ruleset {
        let Some(path) = default_path() else {
            return Ruleset::default();
        };
        Self::load_from(&path)
    }

    pub fn load_from(path: &std::path::Path) -> Ruleset {
        // REGISTRY-WAIVE: intentional fallback: missing path → empty result
        let Ok(text) = std::fs::read_to_string(path) else {
            return Ruleset::default();
        };
        match serde_json::from_str::<Ruleset>(&text) {
            Ok(r) => r,
            Err(e) => {
                tracing::warn!("permissions: corrupt {path:?}: {e}; using empty ruleset");
                Ruleset::default()
            }
        }
    }

    pub fn save(ruleset: &Ruleset) -> Result<(), String> {
        let Some(path) = default_path() else {
            return Err("no $HOME".into());
        };
        Self::save_to(&path, ruleset)
    }

    pub fn save_to(path: &std::path::Path, ruleset: &Ruleset) -> Result<(), String> {
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent).map_err(|e| format!("create dir: {e}"))?;
        }
        let text = serde_json::to_string_pretty(ruleset).map_err(|e| format!("serialize: {e}"))?;
        let tmp = path.with_extension("json.tmp");
        std::fs::write(&tmp, text).map_err(|e| format!("write tmp: {e}"))?;
        std::fs::rename(&tmp, path).map_err(|e| format!("rename: {e}"))?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::super::ruleset::{Action, Rule};
    use super::*;

    #[test]
    fn roundtrip_via_temp_file() {
        let dir = tempfile::TempDir::new().unwrap();
        let p = dir.path().join("permissions.json");
        let mut rs = Ruleset::default();
        rs.push(Rule::new("read", "src/**", Action::Allow));
        rs.push(Rule::new("write", ".env*", Action::Deny));
        Store::save_to(&p, &rs).unwrap();
        let loaded = Store::load_from(&p);
        assert_eq!(loaded.rules.len(), 2);
        assert_eq!(loaded.rules[0].permission, "read");
        assert_eq!(loaded.rules[0].action, Action::Allow);
    }

    #[test]
    fn missing_file_returns_empty() {
        let dir = tempfile::TempDir::new().unwrap();
        let p = dir.path().join("nonexistent.json");
        let r = Store::load_from(&p);
        assert!(r.rules.is_empty());
    }

    #[test]
    fn corrupt_file_returns_empty() {
        let dir = tempfile::TempDir::new().unwrap();
        let p = dir.path().join("corrupt.json");
        std::fs::write(&p, "{not valid json").unwrap();
        let r = Store::load_from(&p);
        assert!(r.rules.is_empty());
    }

    #[test]
    fn save_uses_atomic_temp_rename() {
        let dir = tempfile::TempDir::new().unwrap();
        let p = dir.path().join("permissions.json");
        let rs = Ruleset::default();
        Store::save_to(&p, &rs).unwrap();
        // tmp file should not exist after successful save.
        assert!(!p.with_extension("json.tmp").exists());
        assert!(p.exists());
    }
}
