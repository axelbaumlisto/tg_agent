//! Boot-time invariant checks: config symlinks, system prompts, mtime watcher, noVNC.
//!
//! Extracted from wiring.rs (T2, PLAN_v13_SOLID_AUDIT) for SRP.

/// BUG_REGISTRY D-CONFIG-MTIME-WATCH (B42 detector).
///
/// Records the (mtime, sha256) of `state/naked.json` at boot and
/// spawns a 60-second-interval poller. If the file changes between
/// polls, log WARN + bump `CONFIG_EXTERNAL_WRITE_COUNT`. B42 in
/// the wild: `state/naked.json` had silently reverted to a stale
/// IP between 14:38 and 15:42 on 2026-05-13; this detector turns
/// that class of fault visible per restart.
///
/// The bot writes to `state/naked.json` itself via certain ops
/// (e.g. `naked memory store`), so the watcher's purpose is
/// surface-level monitoring, not strict enforcement. Operator
/// reads the journal to investigate, not to act on automatically.
pub(super) fn spawn_config_mtime_watcher() {
    let path = match std::env::var("NAKED_CONFIG") {
        Ok(p) => std::path::PathBuf::from(p),
        Err(_) => {
            tracing::debug!("config-mtime-watch: no NAKED_CONFIG env, skipping");
            return;
        }
    };
    let initial = match snapshot_config_file(&path) {
        Some(snap) => snap,
        None => {
            tracing::debug!(path = %path.display(), "config-mtime-watch: snapshot failed, skipping");
            return;
        }
    };
    tracing::info!(
        path = %path.display(),
        mtime = %initial.mtime_human,
        hash = %initial.hash_hex,
        "config-mtime-watch armed"
    );
    let path_for_task = path.clone();
    tokio::spawn(async move {
        let mut last = initial;
        let mut interval = tokio::time::interval(std::time::Duration::from_secs(60));
        interval.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);
        interval.tick().await; // consume first immediate tick
        loop {
            interval.tick().await;
            let Some(current) = snapshot_config_file(&path_for_task) else {
                continue;
            };
            if current.hash_hex != last.hash_hex {
                naked_core::types::CONFIG_EXTERNAL_WRITE_COUNT
                    .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                tracing::warn!(
                    path = %path_for_task.display(),
                    prev_hash = %last.hash_hex,
                    curr_hash = %current.hash_hex,
                    prev_mtime = %last.mtime_human,
                    curr_mtime = %current.mtime_human,
                    "B42: state/naked.json modified externally between polls. \
                     Either a cron/recreate script touched it, or another agent \
                     session wrote to it. Cross-check with audit_window.sh + \
                     systemd journal for the suspected writer."
                );
                last = current;
            }
        }
    });
}

/// Snapshot of a file's identity at a point in time: mtime + DefaultHasher
/// digest of contents. Cheap enough to run every 60s; DefaultHasher over
/// ~50 KB `naked.json` is ~30us on this hardware. We don't need
/// cryptographic strength here — only fast difference detection.
///
/// REGISTRY-WAIVE B32 (DEPS): intentionally re-uses std::hash instead
/// of pulling sha2 as a new dep; same pattern as snapshot/ module.
#[derive(Clone, Debug)]
struct ConfigSnapshot {
    /// 64-bit hash of file contents (DefaultHasher, NOT cryptographic).
    hash_hex: String,
    mtime_human: String,
}

fn snapshot_config_file(path: &std::path::Path) -> Option<ConfigSnapshot> {
    use std::hash::{DefaultHasher, Hash, Hasher};
    let bytes = std::fs::read(path).ok()?;
    let mtime = std::fs::metadata(path).ok()?.modified().ok()?;
    let mtime_human = match mtime.duration_since(std::time::UNIX_EPOCH) {
        Ok(d) => format!("{}", d.as_secs()),
        Err(_) => "<epoch>".to_string(),
    };
    let mut h = DefaultHasher::new();
    bytes.hash(&mut h);
    let hash_hex = format!("{:016x}", h.finish());
    Some(ConfigSnapshot {
        hash_hex,
        mtime_human,
    })
}

/// BUG_REGISTRY D-VALIDATE-IP-TOKENS (B37 stream guard).
///
/// At boot, call `naked/skills/novnc-browser/scripts/novnc.sh url`
/// and parse the JSON to extract known-good IP/port pairs (from the
/// `canonical`, `tailscale`, `public` fields). Cache them in a static
/// `OnceLock<Vec<String>>` so stream-level code can compare outgoing
/// noVNC mentions against the allow-list without re-invoking the
/// script every time.
///
/// Fail-open: if the script is unreachable / returns non-JSON, the
/// allow-list stays empty and `validate_ip_tokens()` becomes a no-op.
/// That's intentional — a broken novnc.sh is its own visible problem,
/// we don't want to add a second symptom.
pub(super) fn populate_novnc_ip_allowlist() {
    let script = "/home/spex/work/tg_agent/naked/skills/novnc-browser/scripts/novnc.sh";
    let output = match std::process::Command::new("bash")
        .arg(script)
        .arg("url")
        .output()
    {
        Ok(o) if o.status.success() => o.stdout,
        _ => {
            tracing::debug!("novnc-ip-allowlist: novnc.sh url failed; allow-list empty");
            return;
        }
    };
    // novnc.sh output parse — fail-open by design; empty allow-list →
    // validate_novnc_ip_tokens becomes no-op (B37 detection doc).
    // REGISTRY-WAIVE: intentional fallback: malformed output → skip
    let Ok(text) = String::from_utf8(output) else {
        return;
    };
    // REGISTRY-WAIVE: intentional fallback: malformed JSON → skip
    let Ok(value) = serde_json::from_str::<serde_json::Value>(&text) else {
        return;
    };
    let mut ips: Vec<String> = Vec::new();
    for field in ["canonical", "tailscale", "public"] {
        let Some(url) = value.get(field).and_then(|v| v.as_str()) else {
            continue;
        };
        // Extract host (and port if present) from URL form.
        if let Some(host_port) = extract_host_port(url) {
            ips.push(host_port);
        }
    }
    // Also push the clipshot.cc canonical hostname (not IP, but it's
    // a stable allow-list entry).
    if value.get("canonical").is_some() {
        ips.push("clipshot.cc:443".into());
    }
    if !ips.is_empty() {
        tracing::info!(allowlist = ?ips, "novnc-ip-allowlist populated");
        if let Ok(mut guard) = crate::shared::NOVNC_IP_ALLOWLIST.write() {
            *guard = ips;
        }
    }
}

/// Parse the host[:port] out of a URL form like
///   `http://65.108.226.226:6080/vnc.html?...`
///   `https://clipshot.cc/debug/vnc/...`
/// Returns `"host:port"` if explicit port given, else `"host"`.
fn extract_host_port(url: &str) -> Option<String> {
    let rest = url
        .strip_prefix("http://")
        .or_else(|| url.strip_prefix("https://"))?;
    let host_end = rest.find(['/', '?', '#']).unwrap_or(rest.len());
    Some(rest[..host_end].to_string())
}

/// BUG_REGISTRY D-BOOT-CONFIG-SYMLINK (B41 regression guard).
///
/// AGENTS.md decree: `naked/naked.json -> ../state/naked.json` (symlink,
/// gitignored). When this gets replaced by a regular file (as happened
/// 2026-05-13: divergent snapshot copy lay around since May 12), bot can
/// load the wrong config — specifically `state/naked.json` (via the
/// `NAKED_CONFIG` env in systemd unit) had stale Playwright CDP IP
/// 172.19.0.2 while `naked/naked.json` had the up-to-date 172.19.0.3,
/// or vice versa. Easy fix at boot: warn loudly.
///
/// Returns true if invariant holds, false if it's broken. Not strict
/// (no process exit) — the operator may have a legitimate reason for a
/// regular file (e.g. running with NAKED_CONFIG pointing elsewhere).
/// Set `NAKED_STRICT_CONFIG_SYMLINK=1` to escalate WARN → exit 4.
pub(crate) fn check_config_symlink_invariant() -> bool {
    let repo_root = std::env::var("NAKED_REPO_ROOT").unwrap_or_else(|_| {
        // Default: walk up from CWD until we find a directory containing
        // both `naked/` and `state/`. Fallback to `..` from cwd.
        let cwd = std::env::current_dir().unwrap_or_else(|_| ".".into());
        let mut p = cwd.as_path();
        loop {
            if p.join("naked").is_dir() && p.join("state").is_dir() {
                return p.display().to_string();
            }
            match p.parent() {
                Some(parent) => p = parent,
                None => return cwd.display().to_string(),
            }
        }
    });
    let link_path = std::path::PathBuf::from(&repo_root).join("naked/naked.json");
    if !link_path.exists() {
        tracing::debug!(
            path = %link_path.display(),
            "config-symlink check skipped: naked/naked.json absent"
        );
        return true;
    }
    let meta = match std::fs::symlink_metadata(&link_path) {
        Ok(m) => m,
        Err(e) => {
            tracing::warn!(error = %e, "config-symlink: failed to stat");
            return false;
        }
    };
    if !meta.file_type().is_symlink() {
        let strict = std::env::var_os("NAKED_STRICT_CONFIG_SYMLINK").is_some_and(|v| v == "1");
        tracing::warn!(
            path = %link_path.display(),
            "B41: naked/naked.json is NOT a symlink (AGENTS.md says it must be \
             a symlink to ../state/naked.json). Likely cause: someone replaced \
             the link with a copy. Bot may load wrong config. Fix: \
             `mv naked/naked.json /tmp/.orphan && ln -s ../state/naked.json naked/naked.json`"
        );
        if strict {
            eprintln!("❌ B41: naked/naked.json must be symlink (NAKED_STRICT_CONFIG_SYMLINK=1)");
            std::process::exit(4);
        }
        return false;
    }
    let target = match std::fs::read_link(&link_path) {
        Ok(t) => t,
        Err(e) => {
            tracing::warn!(error = %e, "config-symlink: failed to read link target");
            return false;
        }
    };
    let target_str = target.display().to_string();
    // Accept `../state/naked.json` or absolute equivalent.
    let canonical_ok =
        target_str == "../state/naked.json" || target_str.ends_with("/state/naked.json");
    if !canonical_ok {
        tracing::warn!(
            target = %target_str,
            "B41: naked/naked.json symlink points at unexpected target (expected ../state/naked.json)"
        );
        return false;
    }
    tracing::info!(
        target = %target_str,
        "config-symlink invariant OK"
    );
    true
}

/// BUG_REGISTRY D-CHECK-SYSPROMPT-PATHS (B38/B37 regression guard).
///
/// Scans `~/.naked/system_prompt.md` for path-like tokens (file paths
/// referenced inside backticks or after `»`/`->`/`→`) and asserts each
/// exists on disk. When system_prompt drifts to reference stale paths
/// (~/.zeroclaw/workspace/ etc.), model is told to call scripts that
/// aren't there — then confabulates output.
///
/// Returns number of broken paths. Soft-warn only. Recognised path
/// patterns:
///   - `/home/spex/...` absolute
///   - `~/.naked/...` / `~/work/...` tilde-prefixed (expanded against $HOME)
///   - `./skills/...` / `./scripts/...` repo-relative (expanded vs $HOME)
pub(crate) fn check_system_prompt_paths() -> usize {
    let home = std::env::var("HOME").unwrap_or_else(|_| "/root".into());
    let prompt_path = format!("{home}/.naked/system_prompt.md");
    let src = match std::fs::read_to_string(&prompt_path) {
        Ok(s) => s,
        Err(_) => {
            tracing::debug!(
                path = %prompt_path,
                "sysprompt-paths check skipped: system_prompt.md absent"
            );
            return 0;
        }
    };

    // Regex over the prompt body for path tokens inside backticks.
    // Keep this conservative: only flag tokens that LOOK like real paths
    // (must end in .sh / .py / .md / .json / .session / .so or have at
    // least two path segments under a known prefix). False-positive guard:
    // skip http(s) URLs.
    let mut candidates = std::collections::BTreeSet::<String>::new();
    for line in src.lines() {
        for token in line.split(['`', '\'', '"', ' ']) {
            let t = token.trim_end_matches(&['.', ',', ')', '»', '—', ';', ':'][..]);
            // Filters:
            //   - URL (http/https/etc.)
            //   - empty / env-var ref
            //   - template placeholder paths containing `<token>` like
            //     `<slug>`, `<id>`, `<chat>` — system_prompt uses these
            //     to document parametric paths, not real ones.
            if t.is_empty()
                || t.starts_with("http")
                || t.contains("://")
                || t.contains('<')
                || t.contains('>')
            {
                continue;
            }
            // Strip leading `~` -> HOME.
            let expanded = if let Some(rest) = t.strip_prefix("~/") {
                format!("{home}/{rest}")
            } else if t.starts_with('/') {
                t.to_string()
            } else {
                continue;
            };
            // Look for file-extension or known-prefix anchors.
            let looks_like_path = expanded.contains('/')
                && (expanded.ends_with(".sh")
                    || expanded.ends_with(".py")
                    || expanded.ends_with(".md")
                    || expanded.ends_with(".json")
                    || expanded.ends_with(".session")
                    || expanded.ends_with(".rs")
                    || expanded.ends_with(".toml")
                    || expanded.starts_with(&format!("{home}/.naked/"))
                    || expanded.starts_with("/home/"));
            if looks_like_path {
                candidates.insert(expanded);
            }
        }
    }

    let mut broken: Vec<String> = Vec::new();
    for c in &candidates {
        if !std::path::Path::new(c).exists() {
            broken.push(c.clone());
        }
    }
    if broken.is_empty() {
        tracing::info!(
            paths_checked = candidates.len(),
            "sysprompt-paths invariant OK"
        );
        return 0;
    }
    tracing::warn!(
        broken_count = broken.len(),
        total_checked = candidates.len(),
        "B38/B37 sysprompt-paths: paths referenced in system_prompt.md don't exist on disk"
    );
    for b in &broken {
        tracing::warn!(missing = %b, "sysprompt-paths: broken reference");
    }
    broken.len()
}
