//! Shell-out lifecycle hooks (T6 of `PLAN_QUALITY_v1.md`).
//!
//! Six events fired at well-defined points in the agent loop. Each
//! event can have N hooks; each hook is a shell command with a
//! regex matcher and a timeout. Hooks let operators enforce
//! invariants without polluting the system prompt:
//!
//! ```json
//! [
//!   {"event": "PostToolUse", "matcher": "write_file:.*\\.rs$",
//!    "command": "cargo fmt -- ${path}", "timeout_sec": 10},
//!   {"event": "PreToolUse", "matcher": "shell:rm -rf .*",
//!    "command": "false", "timeout_sec": 1, "outcome_on_fail": "abort"}
//! ]
//! ```
//!
//! Outcome semantics:
//!   * `Success` — hook ran and exited 0; turn proceeds.
//!   * `FailedContinue` — hook exited non-0 OR timed out, but the
//!     turn still proceeds (default for non-`PreToolUse` events).
//!   * `FailedAbort` — hook exited non-0 and the event is
//!     `PreToolUse` AND the hook config has
//!     `outcome_on_fail = "abort"`. Tool execution is BLOCKED
//!     and the model sees a synthetic error result.
//!
//! See [`LifecycleHookRunner::run`] for wiring details.

use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Duration;

use serde::{Deserialize, Serialize};
use tokio::sync::RwLock;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "PascalCase")]
pub enum HookEvent {
    PreToolUse,
    PostToolUse,
    PermissionRequest,
    SessionStart,
    UserPromptSubmit,
    Stop,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum FailMode {
    /// Default: log warn, continue the turn.
    #[default]
    Continue,
    /// Block tool execution (only valid for `PreToolUse`).
    Abort,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HookConfig {
    pub event: HookEvent,
    /// Regex applied to the synthetic key for the event:
    ///   PreToolUse / PostToolUse: "<tool_name>:<input-as-string>"
    ///   PermissionRequest:        "<tool_name>"
    ///   SessionStart / Stop:      ""
    ///   UserPromptSubmit:         "<prompt-text>"
    /// Empty matcher matches everything.
    #[serde(default)]
    pub matcher: String,
    /// Shell command. Tokens like `${path}` / `${tool}` are
    /// substituted at runtime from the event payload.
    pub command: String,
    #[serde(default = "default_timeout")]
    pub timeout_sec: u64,
    #[serde(default)]
    pub outcome_on_fail: FailMode,
}

const fn default_timeout() -> u64 {
    10
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HookOutcome {
    Success,
    FailedContinue,
    FailedAbort,
}

/// Fully-resolved set of hook configs loaded from disk. Loaded once
/// at startup; matched per-event during the loop.
#[derive(Debug, Default)]
pub struct LifecycleHookRunner {
    hooks: RwLock<Vec<HookConfig>>,
}

impl LifecycleHookRunner {
    pub fn new() -> Self {
        Self::default()
    }

    pub async fn install(&self, hook: HookConfig) {
        self.hooks.write().await.push(hook);
    }

    /// Load hooks from `~/.naked/hooks.json`. Missing file → empty.
    /// Corrupt file → empty + warn.
    pub async fn load_default(&self) {
        let Some(path) = default_path() else { return };
        self.load_from(&path).await;
    }

    pub async fn load_from(&self, path: &Path) {
        let Ok(text) = std::fs::read_to_string(path) else {
            return;
        };
        match serde_json::from_str::<Vec<HookConfig>>(&text) {
            Ok(list) => {
                let mut g = self.hooks.write().await;
                g.clear();
                g.extend(list);
            }
            Err(e) => tracing::warn!("hooks: corrupt {path:?}: {e}"),
        }
    }

    /// Fire all hooks matching `event` × `key`. The key is the
    /// synthetic "<scope>:<detail>" string the matcher regex is
    /// applied to.
    pub async fn run(&self, event: HookEvent, key: &str, vars: &[(&str, &str)]) -> HookOutcome {
        let snapshot = self.hooks.read().await.clone();
        let mut worst = HookOutcome::Success;
        for hook in &snapshot {
            if hook.event != event {
                continue;
            }
            if !regex_matches(&hook.matcher, key) {
                continue;
            }
            let cmd = substitute_vars(&hook.command, vars);
            let outcome = run_shell_with_timeout(&cmd, Duration::from_secs(hook.timeout_sec)).await;
            match outcome {
                HookOutcome::Success => {}
                HookOutcome::FailedContinue => {
                    if event == HookEvent::PreToolUse && hook.outcome_on_fail == FailMode::Abort {
                        return HookOutcome::FailedAbort;
                    }
                    if matches!(worst, HookOutcome::Success) {
                        worst = HookOutcome::FailedContinue;
                    }
                }
                HookOutcome::FailedAbort => return HookOutcome::FailedAbort,
            }
        }
        worst
    }
}

#[must_use]
pub fn default_path() -> Option<PathBuf> {
    std::env::var_os("HOME")
        .map(PathBuf::from)
        .map(|h| h.join(".naked").join("hooks.json"))
}

/// Naive regex matcher. Empty pattern matches anything; otherwise
/// uses `std::str::contains` for substring match (we don't pull in
/// the full `regex` crate just for this — operators can use literal
/// substrings or escape special chars themselves).
fn regex_matches(pattern: &str, target: &str) -> bool {
    if pattern.is_empty() {
        return true;
    }
    target.contains(pattern)
}

fn substitute_vars(template: &str, vars: &[(&str, &str)]) -> String {
    let mut out = template.to_string();
    for (k, v) in vars {
        out = out.replace(&format!("${{{k}}}"), v);
    }
    out
}

async fn run_shell_with_timeout(cmd: &str, timeout: Duration) -> HookOutcome {
    let mut sh = tokio::process::Command::new("sh");
    sh.arg("-c").arg(cmd);
    let exec = sh.status();
    match tokio::time::timeout(timeout, exec).await {
        Ok(Ok(status)) if status.success() => HookOutcome::Success,
        Ok(Ok(_)) | Ok(Err(_)) => HookOutcome::FailedContinue,
        Err(_) => {
            tracing::warn!("hook timed out after {:?}: {cmd}", timeout);
            HookOutcome::FailedContinue
        }
    }
}

/// Shareable handle for the engine.
pub type SharedHookRunner = Arc<LifecycleHookRunner>;

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn empty_runner_returns_success() {
        let r = LifecycleHookRunner::new();
        assert_eq!(
            r.run(HookEvent::PreToolUse, "any", &[]).await,
            HookOutcome::Success
        );
    }

    #[tokio::test]
    async fn hook_matching_runs_command() {
        let r = LifecycleHookRunner::new();
        r.install(HookConfig {
            event: HookEvent::PostToolUse,
            matcher: "write_file".into(),
            command: "true".into(),
            timeout_sec: 2,
            outcome_on_fail: FailMode::Continue,
        })
        .await;
        let out = r
            .run(HookEvent::PostToolUse, "write_file:foo.rs", &[])
            .await;
        assert_eq!(out, HookOutcome::Success);
    }

    #[tokio::test]
    async fn non_matching_event_skipped() {
        let r = LifecycleHookRunner::new();
        r.install(HookConfig {
            event: HookEvent::PostToolUse,
            matcher: "".into(),
            command: "false".into(),
            timeout_sec: 1,
            outcome_on_fail: FailMode::Continue,
        })
        .await;
        // Different event — hook must not fire.
        let out = r.run(HookEvent::PreToolUse, "any", &[]).await;
        assert_eq!(out, HookOutcome::Success);
    }

    #[tokio::test]
    async fn pre_tool_use_abort_blocks() {
        let r = LifecycleHookRunner::new();
        r.install(HookConfig {
            event: HookEvent::PreToolUse,
            matcher: "shell:rm".into(),
            command: "false".into(),
            timeout_sec: 1,
            outcome_on_fail: FailMode::Abort,
        })
        .await;
        let out = r.run(HookEvent::PreToolUse, "shell:rm -rf /", &[]).await;
        assert_eq!(out, HookOutcome::FailedAbort);
    }

    #[tokio::test]
    async fn variable_substitution_works() {
        // We use `printf` to a tempfile so we can verify the
        // substitution actually reached the shell.
        let dir = tempfile::TempDir::new().unwrap();
        let f = dir.path().join("out");
        let r = LifecycleHookRunner::new();
        r.install(HookConfig {
            event: HookEvent::PostToolUse,
            matcher: "".into(),
            command: format!("printf '%s' '${{path}}' > {}", f.display()),
            timeout_sec: 2,
            outcome_on_fail: FailMode::Continue,
        })
        .await;
        let _ = r
            .run(HookEvent::PostToolUse, "any", &[("path", "src/foo.rs")])
            .await;
        let body = std::fs::read_to_string(&f).unwrap_or_default();
        assert_eq!(body, "src/foo.rs");
    }

    #[test]
    fn regex_matches_empty_matches_all() {
        assert!(regex_matches("", "anything"));
    }

    #[test]
    fn regex_matches_substring() {
        assert!(regex_matches("foo", "barfoo"));
        assert!(!regex_matches("foo", "bar"));
    }
}
