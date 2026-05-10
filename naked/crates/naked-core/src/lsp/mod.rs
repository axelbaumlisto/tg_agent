//! Post-edit LSP diagnostics (T2 of `PLAN_QUALITY_v1.md`).
//!
//! After every successful `edit_file` / `apply_patch` / `write_file`
//! the loop calls [`LspManager::diagnostics_for`]. The manager
//! looks up (or lazily spawns) the right LSP server for the file's
//! language, sends a `textDocument/didChange`, waits up to
//! `poll_after_edit_ms`, and returns whatever
//! `textDocument/publishDiagnostics` notifications arrived. The
//! caller renders them as a synthetic system message which is
//! prepended to the next request.
//!
//! Failure model: every operation is best-effort. Missing LSP
//! binary, server crash, or timeout returns `None` /
//! `Vec::default()` and logs at `tracing::debug!` — the agent loop
//! never blocks on the LSP subsystem.
//!
//! Submodules:
//!   * [`client`] — stdio JSON-RPC transport (Content-Length framing).
//!   * [`diagnostics`] — typed Diagnostic + render_for_model.
//!   * [`registry`] — language detection + default server commands.

pub mod client;
pub mod diagnostics;
pub mod registry;

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Duration;

use tokio::sync::Mutex;

pub use client::StdioLspTransport;
pub use diagnostics::{Diagnostic, Severity, render_for_model};
pub use registry::{Language, server_for_extension};

/// Knobs read from `[lsp]` in `naked.json`. All defaults match the
/// plan acceptance row in PLAN_QUALITY_v1 §Q.2.
///
/// **Default is enabled** so the model gets post-edit compiler
/// feedback out of the box — the biggest quality-multiplier in
/// PLAN_QUALITY_v1. Spawning is lazy (server starts on first edit
/// per language only); failure-mode is silent (`tracing::debug!`,
/// no-op). Operators who explicitly want to disable it can set
/// `NAKED_LSP_DISABLED=1` in the environment, or build an
/// `LspConfig { enabled: false, ..Default::default() }` programmatically.
#[derive(Debug, Clone)]
pub struct LspConfig {
    /// Globally enabled? Default `true`.
    pub enabled: bool,
    /// How long after `didChange` to wait for diagnostics. Default 5s.
    pub poll_after_edit: Duration,
    /// Cap per file. Default 20.
    pub max_diagnostics_per_file: usize,
    /// Whether to surface warnings. Default false (errors only).
    pub include_warnings: bool,
}

impl Default for LspConfig {
    fn default() -> Self {
        // Honour `NAKED_LSP_DISABLED` env opt-out. Any value that
        // parses as truthy turns LSP off; missing or empty leaves
        // it on. We use the same truthy set as `DEEPSEEK_MEMORY`
        // / GitHub Actions so it feels familiar.
        let disabled = matches!(
            std::env::var("NAKED_LSP_DISABLED")
                .unwrap_or_default()
                .trim()
                .to_ascii_lowercase()
                .as_str(),
            "1" | "true" | "yes" | "on"
        );
        Self {
            enabled: !disabled,
            poll_after_edit: Duration::from_millis(5000),
            max_diagnostics_per_file: 20,
            include_warnings: false,
        }
    }
}

/// Lazy pool: one transport per language, spawned on first use.
#[derive(Debug, Default)]
pub struct LspManager {
    config: LspConfig,
    pool: Mutex<HashMap<Language, Arc<StdioLspTransport>>>,
}

impl LspManager {
    pub fn new(config: LspConfig) -> Self {
        Self {
            config,
            pool: Mutex::new(HashMap::new()),
        }
    }

    /// Diagnostics-after-edit hot path. Returns the raw Vec<Diagnostic>
    /// for `path` after `didChange` + a poll window. Empty vec on any
    /// failure (missing binary, crashed server, no diagnostics).
    pub async fn diagnostics_for(&self, workspace: &Path, path: &Path) -> Vec<Diagnostic> {
        if !self.config.enabled {
            return Vec::new();
        }
        let Some(lang) = registry::detect_language(path) else {
            return Vec::new();
        };
        let transport = match self.get_or_spawn(lang, workspace).await {
            Ok(t) => t,
            Err(e) => {
                tracing::debug!("lsp spawn failed for {lang:?}: {e}");
                return Vec::new();
            }
        };
        let absolute = absolutize(workspace, path);
        let body = std::fs::read_to_string(&absolute).unwrap_or_default();
        if let Err(e) = transport.did_open_or_change(&absolute, &body, lang).await {
            tracing::debug!("did_change failed: {e}");
            return Vec::new();
        }
        let diags = transport
            .await_diagnostics(&absolute, self.config.poll_after_edit)
            .await;
        let mut filtered: Vec<Diagnostic> = diags
            .into_iter()
            .filter(|d| self.config.include_warnings || matches!(d.severity, Severity::Error))
            .take(self.config.max_diagnostics_per_file)
            .collect();
        filtered.sort_by_key(|d| (d.severity as u8, d.line, d.col));
        filtered
    }

    /// Pool insertion + lazy spawn.
    async fn get_or_spawn(
        &self,
        lang: Language,
        workspace: &Path,
    ) -> Result<Arc<StdioLspTransport>, String> {
        let mut pool = self.pool.lock().await;
        if let Some(t) = pool.get(&lang) {
            return Ok(t.clone());
        }
        let cmd = server_for_extension(lang).ok_or_else(|| format!("no LSP for {lang:?}"))?;
        let t = StdioLspTransport::spawn(cmd, workspace).await?;
        let arc = Arc::new(t);
        pool.insert(lang, arc.clone());
        Ok(arc)
    }
}

fn absolutize(workspace: &Path, path: &Path) -> PathBuf {
    if path.is_absolute() {
        path.to_path_buf()
    } else {
        workspace.join(path)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn disabled_returns_empty() {
        // Smoke: with enabled=false, diagnostics_for is a no-op.
        // Default is now enabled=true (PLAN_QUALITY_v1 follow-up),
        // so we explicitly disable for this test.
        let mgr = LspManager::new(LspConfig {
            enabled: false,
            ..Default::default()
        });
        let rt = tokio::runtime::Runtime::new().unwrap();
        let diags = rt.block_on(mgr.diagnostics_for(Path::new("/tmp"), Path::new("foo.rs")));
        assert!(diags.is_empty());
    }

    #[test]
    fn default_is_enabled_unless_env_disables() {
        // Without env override, default is on.
        // (We can't safely mutate $NAKED_LSP_DISABLED in this test
        // because of the workspace's deny-unsafe-code lint, but if
        // it happens to be set in the CI environment the env-opt-out
        // code path is exercised.)
        let cfg = LspConfig::default();
        let env_disabled = std::env::var("NAKED_LSP_DISABLED")
            .map(|v| {
                matches!(
                    v.trim().to_ascii_lowercase().as_str(),
                    "1" | "true" | "yes" | "on"
                )
            })
            .unwrap_or(false);
        assert_eq!(
            cfg.enabled, !env_disabled,
            "default enabled state must reflect env opt-out"
        );
    }

    #[test]
    fn unknown_extension_returns_empty() {
        let mgr = LspManager::new(LspConfig {
            enabled: true,
            ..Default::default()
        });
        let rt = tokio::runtime::Runtime::new().unwrap();
        let diags = rt.block_on(mgr.diagnostics_for(Path::new("/tmp"), Path::new("readme.txt")));
        assert!(diags.is_empty());
    }
}
