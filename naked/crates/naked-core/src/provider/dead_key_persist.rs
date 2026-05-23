//! B46 / PLAN_PROVIDER_HEALTH_v1 — auto-persist permanently-dead provider
//! keys to `state/naked.json` so the next boot doesn't try them again.
//!
//! ## Why
//!
//! Without this, every restart of the bot probes ALL configured keys via
//! `provider audit`, even those that returned 401/402 last time and got
//! marked `permanent_blacklist` in-memory by [`super::resilient`]. After
//! a restart the in-memory blacklist is empty → ~7 wasted HTTP calls
//! per `deepseek` turn (one per dead key) until the rotation skips them.
//!
//! ## What
//!
//! When [`super::resilient`] marks a key PERMANENTLY blacklisted, call
//! [`persist_dead_key`] with:
//!   - `provider_name` (e.g. `"deepseek"`)
//!   - `key_value` — the literal `api_key` string (may be `"$ENV_VAR"`
//!     unexpanded or the raw secret if hard-coded)
//!
//! Side effects:
//!   1. Read `state/naked.json` (path from `NAKED_CONFIG` env, default `state/naked.json`)
//!   2. Find `providers.<provider_name>`
//!   3. Remove `key_value` from `api_keys[]` if present
//!   4. If `api_key` (primary) matches `key_value`, **don't touch primary**
//!      (operator must decide who replaces primary; we just stop using stale
//!      rotation pool). Log WARN if primary itself is dead.
//!   5. Append `key_value` to `_dead_api_keys_auto_YYYY-MM-DD` array
//!      (creating the array if missing). Adjacent to operator's
//!      `_dead_api_keys_<date>` lists.
//!   6. Atomic write via tmpfile + rename.
//!
//! ## Gated by env var
//!
//! `NAKED_AUTO_PERSIST_DEAD_KEYS=1` — opt-in. Off by default so the
//! function is a pure no-op until operator chooses to enable.
//!
//! ## Concurrency
//!
//! Multiple keys may go dead simultaneously across providers. Serialised
//! via [`PERSIST_MUTEX`] — short critical section (read JSON, mutate,
//! write). Worst case ~3-5ms per persist. Acceptable for an event that
//! fires once per dead key per process lifetime.
//!
//! ## B42 interaction
//!
//! Bot's own write to `state/naked.json` will trip the B42 mtime watcher.
//! We emit a structured `tracing::info!` `kind="dead_key_persist"` so the
//! audit script can correlate the mtime jump with the intentional cause.

use serde_json::{Map, Value};
use std::path::{Path, PathBuf};
use std::sync::Mutex;

/// Process-wide lock so concurrent persists from different provider
/// futures don't race on the same file.
static PERSIST_MUTEX: Mutex<()> = Mutex::new(());

/// Env-var gate. Returns `true` only when operator explicitly opted in.
fn enabled() -> bool {
    matches!(
        std::env::var("NAKED_AUTO_PERSIST_DEAD_KEYS")
            .ok()
            .as_deref(),
        Some("1" | "true" | "TRUE" | "yes" | "on")
    )
}

/// Resolve the live `naked.json` path used by the running bot.
/// Defaults to `state/naked.json` relative to CWD if env not set.
fn config_path() -> PathBuf {
    std::env::var("NAKED_CONFIG")
        .map(PathBuf::from)
        .unwrap_or_else(|_| PathBuf::from("state/naked.json"))
}

/// Mask a key for safe logging. Keeps first 10 chars, last 4.
/// Env-var placeholders (`$NAME`) are returned unchanged because they're
/// already not-a-secret.
fn mask_key(key: &str) -> String {
    if key.starts_with('$') {
        return key.to_string();
    }
    if key.len() <= 14 {
        return "<masked-short>".into();
    }
    format!("{}...{}", &key[..10], &key[key.len() - 4..]) // REGISTRY-WAIVE: B48 — API key is ASCII
}

/// Today's date in YYYY-MM-DD UTC. Used as suffix for the auto-dead array.
fn today_utc() -> String {
    chrono::Utc::now().format("%Y-%m-%d").to_string()
}

/// Public entry point — called from [`super::resilient`] when a key is
/// marked PERMANENTLY blacklisted (auth/payment).
///
/// No-op when `NAKED_AUTO_PERSIST_DEAD_KEYS` is unset / falsy. Errors are
/// logged but never propagated — failing to persist must NOT crash the
/// bot mid-turn.
pub fn persist_dead_key(provider_name: &str, key_value: &str) {
    if !enabled() {
        return;
    }
    let _guard = match PERSIST_MUTEX.lock() {
        Ok(g) => g,
        Err(poison) => poison.into_inner(),
    };
    let path = config_path();
    if let Err(e) = persist_inner(&path, provider_name, key_value) {
        tracing::warn!(
            target: "naked_core::provider::dead_key_persist",
            provider = provider_name,
            key_masked = %mask_key(key_value),
            path = %path.display(),
            "B46: dead-key persist failed: {e}"
        );
    }
}

/// Pure logic — testable. Returns the path of the auto-dead-list bucket
/// that was modified, or `Ok(None)` if nothing changed (already in list,
/// or primary key match).
pub fn persist_inner(
    path: &Path,
    provider_name: &str,
    key_value: &str,
) -> Result<Option<String>, String> {
    let content = std::fs::read_to_string(path).map_err(|e| format!("read: {e}"))?;
    let mut cfg: Value = serde_json::from_str(&content).map_err(|e| format!("parse: {e}"))?;

    let providers = cfg
        .get_mut("providers")
        .and_then(Value::as_object_mut)
        .ok_or("config has no `providers` object")?;
    let provider = providers
        .get_mut(provider_name)
        .and_then(Value::as_object_mut)
        .ok_or_else(|| format!("provider `{provider_name}` not found"))?;

    // 1. If this IS the primary key, log and bail — operator's call.
    if let Some(primary) = provider.get("api_key").and_then(Value::as_str)
        && primary == key_value
    {
        tracing::warn!(
            target: "naked_core::provider::dead_key_persist",
            provider = provider_name,
            key_masked = %mask_key(key_value),
            "B46: PRIMARY key is dead — operator action required (not auto-rotated)"
        );
        return Ok(None);
    }

    // 2. Remove from api_keys[] rotation pool, if present.
    let mut removed_from_pool = false;
    if let Some(rotation) = provider.get_mut("api_keys").and_then(Value::as_array_mut) {
        let before = rotation.len();
        rotation.retain(|v| v.as_str() != Some(key_value));
        removed_from_pool = rotation.len() != before;
    }

    // 3. Append to _dead_api_keys_auto_<date>.
    let bucket_name = format!("_dead_api_keys_auto_{}", today_utc());
    let already_listed = provider
        .get(&bucket_name)
        .and_then(Value::as_array)
        .map(|arr| arr.iter().any(|v| v.as_str() == Some(key_value)))
        .unwrap_or(false);
    if already_listed && !removed_from_pool {
        return Ok(None); // idempotent no-op
    }
    let bucket = provider
        .entry(bucket_name.clone())
        .or_insert_with(|| Value::Array(Vec::new()));
    if let Some(arr) = bucket.as_array_mut()
        && !arr.iter().any(|v| v.as_str() == Some(key_value))
    {
        arr.push(Value::String(key_value.to_string()));
    }

    // 4. Atomic write via tmp + rename.
    let pretty = serde_json::to_string_pretty(&cfg).map_err(|e| format!("serialize: {e}"))?;
    let tmp = path.with_extension("json.tmp.b46");
    std::fs::write(&tmp, pretty).map_err(|e| format!("write tmp: {e}"))?;
    std::fs::rename(&tmp, path).map_err(|e| format!("rename: {e}"))?;

    tracing::info!(
        target: "naked_core::provider::dead_key_persist",
        kind = "dead_key_persist",
        provider = provider_name,
        key_masked = %mask_key(key_value),
        bucket = %bucket_name,
        removed_from_rotation = removed_from_pool,
        "B46: persisted dead key to naked.json"
    );

    Ok(Some(bucket_name))
}

#[allow(dead_code)] // helper for tests
fn provider_object<'a>(cfg: &'a Value, name: &str) -> Option<&'a Map<String, Value>> {
    cfg.get("providers")?.get(name)?.as_object()
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn write_fixture(json_val: Value) -> tempfile::NamedTempFile {
        let f = tempfile::NamedTempFile::new().unwrap();
        std::fs::write(f.path(), json_val.to_string()).unwrap();
        f
    }

    fn read_back(path: &Path) -> Value {
        serde_json::from_str(&std::fs::read_to_string(path).unwrap()).unwrap()
    }

    #[test]
    fn removes_from_rotation_and_appends_to_auto_bucket() {
        let f = write_fixture(json!({
            "providers": {
                "deepseek": {
                    "api_key": "sk-alive",
                    "api_keys": ["sk-alive", "sk-dead-1", "sk-other"]
                }
            }
        }));
        let res = persist_inner(f.path(), "deepseek", "sk-dead-1").unwrap();
        assert!(res.is_some());
        let v = read_back(f.path());
        let dp = provider_object(&v, "deepseek").unwrap();
        let rotation: Vec<&str> = dp["api_keys"]
            .as_array()
            .unwrap()
            .iter()
            .map(|x| x.as_str().unwrap())
            .collect();
        assert_eq!(rotation, vec!["sk-alive", "sk-other"]);
        let today = today_utc();
        let bucket_key = format!("_dead_api_keys_auto_{today}");
        let bucket: Vec<&str> = dp[&bucket_key]
            .as_array()
            .unwrap()
            .iter()
            .map(|x| x.as_str().unwrap())
            .collect();
        assert_eq!(bucket, vec!["sk-dead-1"]);
    }

    #[test]
    fn idempotent_double_persist() {
        let f = write_fixture(json!({
            "providers": {
                "deepseek": {
                    "api_key": "sk-a",
                    "api_keys": ["sk-dead"]
                }
            }
        }));
        let r1 = persist_inner(f.path(), "deepseek", "sk-dead").unwrap();
        assert!(r1.is_some());
        // Second call: rotation already empty, bucket already lists it.
        let r2 = persist_inner(f.path(), "deepseek", "sk-dead").unwrap();
        assert!(r2.is_none(), "second call should be no-op");
    }

    #[test]
    fn primary_key_is_not_auto_rotated() {
        let f = write_fixture(json!({
            "providers": {
                "deepseek": {
                    "api_key": "sk-primary",
                    "api_keys": []
                }
            }
        }));
        let r = persist_inner(f.path(), "deepseek", "sk-primary").unwrap();
        assert!(r.is_none(), "primary should not be touched");
        let v = read_back(f.path());
        let dp = provider_object(&v, "deepseek").unwrap();
        // Primary unchanged, no auto-bucket created.
        assert_eq!(dp["api_key"], "sk-primary");
        let today = today_utc();
        assert!(dp.get(&format!("_dead_api_keys_auto_{today}")).is_none());
    }

    #[test]
    fn missing_provider_returns_err() {
        let f = write_fixture(json!({"providers": {"qwen": {"api_key": "x"}}}));
        let r = persist_inner(f.path(), "deepseek", "sk-anything");
        assert!(r.is_err());
        assert!(r.unwrap_err().contains("not found"));
    }

    #[test]
    fn appends_to_existing_today_bucket() {
        let today = today_utc();
        let bucket_key = format!("_dead_api_keys_auto_{today}");
        let f = write_fixture(json!({
            "providers": {
                "deepseek": {
                    "api_key": "sk-a",
                    "api_keys": ["sk-dead-2"],
                    bucket_key.clone(): ["sk-already-listed"]
                }
            }
        }));
        let r = persist_inner(f.path(), "deepseek", "sk-dead-2").unwrap();
        assert!(r.is_some());
        let v = read_back(f.path());
        let dp = provider_object(&v, "deepseek").unwrap();
        let bucket: Vec<&str> = dp[&bucket_key]
            .as_array()
            .unwrap()
            .iter()
            .map(|x| x.as_str().unwrap())
            .collect();
        assert_eq!(bucket.len(), 2);
        assert!(bucket.contains(&"sk-already-listed"));
        assert!(bucket.contains(&"sk-dead-2"));
    }

    #[test]
    fn mask_key_handles_env_vars_and_short_keys() {
        assert_eq!(mask_key("$DEEPSEEK_API_KEY"), "$DEEPSEEK_API_KEY");
        assert_eq!(mask_key("short"), "<masked-short>");
        // Fake string with deliberate `/` break in the prefix so it does NOT
        // match the `sk-[A-Za-z0-9_-]{20,}` sanitiser regex used by
        // `naked/scripts/export_github_sanitized.sh`. Length still > 14 so
        // mask_key produces the prefix..suffix form rather than
        // `<masked-short>`.
        assert_eq!(
            mask_key("sk/FAKE-test-key-padding-here-1234"),
            "sk/FAKE-te...1234"
        );
    }

    // `enabled()` is gated by env var. Direct mutation requires `unsafe`
    // (Rust 2024 edition). To avoid `unsafe` in tests + B43 race risk, we
    // skip the env-toggle test and verify the gate by reading current state
    // only. The contract is one-liner: `matches!(env::var(...), Some("1"|...))`.
    #[test]
    fn enabled_returns_bool_without_panic() {
        // Either branch is fine; we just assert this doesn't panic.
        let _ = enabled();
    }
}
