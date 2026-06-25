//! Loaded-config hash helpers for B42 observability.
//!
//! The hash is intentionally a fast `DefaultHasher` digest over the exact
//! JSON bytes that `Config::load_with_source_bytes` parsed. That makes it
//! deterministic for a given loaded file and avoids hashing `Config`'s
//! HashMap-backed structure.

use naked_core::config::Config;
use std::hash::{DefaultHasher, Hash, Hasher};
use std::path::PathBuf;

pub type ConfigSourceBytes = Option<(PathBuf, Vec<u8>)>;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ConfigLoadedHash {
    pub hash_hex: String,
    pub source: String,
}

pub fn hash_config_bytes(bytes: &[u8]) -> String {
    let mut h = DefaultHasher::new();
    bytes.hash(&mut h);
    format!("{:016x}", h.finish())
}

pub fn loaded_hash_from_source(source: ConfigSourceBytes, _config: &Config) -> ConfigLoadedHash {
    match source {
        Some((path, bytes)) => ConfigLoadedHash {
            hash_hex: hash_config_bytes(&bytes),
            source: path.display().to_string(),
        },
        None => {
            // No file was loaded (all-default config). Hash a fixed sentinel
            // so even ad-hoc binary invocations remain observable without
            // serializing `Config` maps or adding a dependency. Production uses
            // the raw loaded file bytes above.
            ConfigLoadedHash {
                hash_hex: hash_config_bytes(b"<naked-config-default>\n"),
                source: "<default>".to_string(),
            }
        }
    }
}

pub fn render_health_config_hash(hash: Option<&ConfigLoadedHash>) -> String {
    match hash {
        Some(hash) => format!(
            "🧾 <b>Config</b>: loaded hash <code>{}</code>",
            hash.hash_hex
        ),
        None => "🧾 <b>Config</b>: loaded hash <code>unknown</code>".to_string(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn config_loaded_hash_differs_and_renders() {
        let json_a = br#"{"default_provider":"qwen","default_model":"qwen-plus"}"#;
        let json_b = br#"{"default_provider":"deepseek","default_model":"deepseek-chat"}"#;
        let _cfg_a = Config::from_json_str(std::str::from_utf8(json_a).unwrap()).unwrap();
        let _cfg_b = Config::from_json_str(std::str::from_utf8(json_b).unwrap()).unwrap();

        let hash_a = hash_config_bytes(json_a);
        let hash_b = hash_config_bytes(json_b);
        let hash_a_again = hash_config_bytes(json_a);

        assert_ne!(
            hash_a, hash_b,
            "distinct loaded configs must have distinct hashes"
        );
        assert_eq!(
            hash_a, hash_a_again,
            "same config bytes must hash deterministically"
        );

        let artifact = ConfigLoadedHash {
            hash_hex: hash_a.clone(),
            source: "test-a.json".to_string(),
        };
        let health = render_health_config_hash(Some(&artifact));
        assert!(
            health.contains(&hash_a),
            "health render must expose loaded config hash; health={health}"
        );
    }
}
