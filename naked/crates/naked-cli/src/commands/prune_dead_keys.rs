//! Offline dead-provider-key pruning helper.

use std::{fs, path::PathBuf};

use anyhow::{Context, Result, bail};
use naked_core::{config::filter_dead_keys_from_json, tool::diff_format::unified_diff};

pub(crate) fn prune_dead_keys_cmd(args: &[String]) -> Result<()> {
    let parsed = PruneArgs::parse(args)?;
    let before = fs::read_to_string(&parsed.config)
        .with_context(|| format!("read config {}", parsed.config.display()))?;
    let after = filter_dead_keys_from_json(&before).map_err(anyhow::Error::msg)?;

    if before == after {
        println!(
            "No dead rotation keys to prune in {}",
            parsed.config.display()
        );
        return Ok(());
    }

    let diff = unified_diff(
        parsed.config.to_string_lossy().as_ref(),
        &mask_json_for_output(&before),
        &mask_json_for_output(&after),
    );

    if parsed.apply {
        fs::write(&parsed.config, after)
            .with_context(|| format!("write config {}", parsed.config.display()))?;
        println!("Applied dead-key prune to {}", parsed.config.display());
        println!("Reminder: commit state/naked.json to HEAD immediately (AB-11 protocol).");
    } else {
        println!(
            "Dry-run: dead-key prune would change {}",
            parsed.config.display()
        );
        println!("Re-run with --apply while naked-tg.service is stopped to write changes.");
    }

    if diff.is_empty() {
        println!("(masked diff empty after redaction)");
    } else {
        println!("{diff}");
    }

    Ok(())
}

#[derive(Debug, PartialEq, Eq)]
struct PruneArgs {
    config: PathBuf,
    apply: bool,
}

impl PruneArgs {
    fn parse(args: &[String]) -> Result<Self> {
        let mut config: Option<PathBuf> = None;
        let mut apply = false;
        let mut i = 0;
        while i < args.len() {
            match args[i].as_str() {
                "--help" | "-h" | "help" => {
                    print_usage();
                    std::process::exit(0);
                }
                "--apply" => {
                    apply = true;
                    i += 1;
                }
                "--config" => {
                    let Some(path) = args.get(i + 1) else {
                        bail!("--config requires a path");
                    };
                    config = Some(PathBuf::from(path));
                    i += 2;
                }
                unknown => bail!("unknown prune-dead-keys arg: {unknown}"),
            }
        }

        let config = match config {
            Some(path) => path,
            None => std::env::var_os("NAKED_CONFIG")
                .map(PathBuf::from)
                .unwrap_or_else(|| PathBuf::from("state/naked.json")),
        };

        Ok(Self { config, apply })
    }
}

fn print_usage() {
    println!("Usage: naked prune-dead-keys [--config <path>] [--apply]");
    println!();
    println!("Runs naked_core::config::loader::filter_dead_keys_from_json offline.");
    println!("Default is dry-run: print a masked unified diff without writing.");
}

fn mask_json_for_output(json: &str) -> String {
    match serde_json::from_str::<serde_json::Value>(json) {
        Ok(mut value) => {
            mask_value(&mut value, None);
            serde_json::to_string_pretty(&value).unwrap_or_else(|_| mask_plain_text(json))
        }
        Err(_) => mask_plain_text(json),
    }
}

fn mask_value(value: &mut serde_json::Value, key_hint: Option<&str>) {
    match value {
        serde_json::Value::String(s) => {
            if key_says_secret(key_hint) || looks_secret_like(s) {
                *s = mask_string(s);
            }
        }
        serde_json::Value::Array(values) => {
            for value in values {
                mask_value(value, key_hint);
            }
        }
        serde_json::Value::Object(map) => {
            for (key, value) in map.iter_mut() {
                mask_value(value, Some(key));
            }
        }
        serde_json::Value::Null | serde_json::Value::Bool(_) | serde_json::Value::Number(_) => {}
    }
}

fn mask_plain_text(text: &str) -> String {
    text.split_inclusive(|c: char| c.is_whitespace())
        .map(|token| {
            let trimmed_len = token.trim_end_matches(|c: char| c.is_whitespace()).len();
            let (word, suffix) = token.split_at(trimmed_len);
            format!("{}{}", mask_string(word), suffix)
        })
        .collect()
}

fn mask_string(s: &str) -> String {
    if s.starts_with('$') {
        return s.to_string();
    }
    if s.is_empty() {
        return String::new();
    }
    if s.chars().count() <= 4 {
        return "****".to_string();
    }

    let last4_start = s
        .char_indices()
        .rev()
        .nth(3)
        .map(|(idx, _)| idx)
        .unwrap_or(0);
    format!("****{}", &s[last4_start..])
}

fn key_says_secret(key_hint: Option<&str>) -> bool {
    let Some(key) = key_hint else {
        return false;
    };
    let lower = key.to_ascii_lowercase();
    lower.contains("api_key")
        || lower.contains("api_keys")
        || lower.contains("dead_api_keys")
        || lower.contains("low_balance_keys")
        || lower.contains("token")
        || lower.contains("secret")
        || lower.contains("password")
        || lower.contains("authorization")
}

fn looks_secret_like(s: &str) -> bool {
    let lower = s.to_ascii_lowercase();
    lower.starts_with("sk-")
        || lower.starts_with("sk_")
        || lower.starts_with("sk/")
        || (s.len() >= 24
            && s.chars()
                .all(|c| c.is_ascii_alphanumeric() || matches!(c, '-' | '_' | '/')))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_defaults_to_env_or_state() {
        let args = PruneArgs::parse(&[]).unwrap();
        assert!(
            args.config.ends_with("state/naked.json") || std::env::var_os("NAKED_CONFIG").is_some()
        );
        assert!(!args.apply);
    }

    #[test]
    fn parse_config_and_apply() {
        let args = PruneArgs::parse(&[
            "--config".to_string(),
            "/tmp/cfg.json".to_string(),
            "--apply".to_string(),
        ])
        .unwrap();
        assert_eq!(args.config, PathBuf::from("/tmp/cfg.json"));
        assert!(args.apply);
    }

    #[test]
    fn masks_secret_strings_last4_only() {
        // Deliberate `/` break so this fake literal does NOT match the
        // `sk-[A-Za-z0-9_-]{20,}` sanitiser regex in
        // scripts/export_github_sanitized.sh; length > 14 so mask_string
        // produces the ****last4 form. (Same pattern as dead_key_persist.rs.)
        let masked = mask_json_for_output(
            r#"{"api_key":"sk/FAKE-abcdefghijklmnopqrstuvwxyz","model":"not-secret"}"#,
        );
        assert!(masked.contains("****wxyz"));
        assert!(masked.contains("not-secret"));
        assert!(!masked.contains("sk/FAKE-abcdefghijklmnopqrstuv"));
    }

    #[test]
    fn masks_short_secret_strings_without_leaking_original_chars() {
        let masked = mask_json_for_output(
            r#"{"api_key":"abc","api_keys":["ok1","bad"],"_dead_api_keys_auto_x":["bad"],"_low_balance_keys_auto_x":["zip"],"env":"$ENV_KEY"}"#,
        );
        assert!(masked.contains("$ENV_KEY"));
        assert!(!masked.contains("abc"));
        assert!(!masked.contains("ok1"));
        assert!(!masked.contains("bad"));
        assert!(!masked.contains("zip"));
        assert!(masked.matches("****").count() >= 5);
    }
}
