//! Tool execution policy — decides whether a tool call should run, ask the user, or be denied.
//!
//! The agent loop delegates ALL permission decisions to `dyn ToolPolicy`.
//! This means adding a new policy rule (e.g. "block network tools after midnight")
//! requires ZERO changes to `loop_.rs`.

use std::path::Path;

use crate::types::Permission;

/// Decision for a single tool call.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ToolDecision {
    /// Execute immediately, no user confirmation needed.
    Execute,
    /// Ask the user for confirmation (includes the permission level for UI).
    AskUser(Permission),
    /// Deny execution with a reason.
    Deny(String),
}

/// Policy that decides how each tool call is handled.
///
/// Implement this trait to customize permission behavior without
/// touching the agent loop.
pub trait ToolPolicy: Send + Sync {
    /// Classify a tool call: run it, ask the user, or deny.
    ///
    /// Arguments:
    /// - `name`: tool name (e.g. "bash", "read_file")
    /// - `input`: tool arguments as JSON
    /// - `cwd`: current working directory
    /// - `permission`: the tool's own permission level for this invocation
    fn classify(
        &self,
        name: &str,
        input: &serde_json::Value,
        cwd: &Path,
        permission: Permission,
    ) -> ToolDecision;
}

/// Default policy: read-only tools execute, everything else asks the user.
///
/// This is the current behavior extracted from `loop_.rs`.
pub struct DefaultPolicy;

impl ToolPolicy for DefaultPolicy {
    fn classify(
        &self,
        _name: &str,
        _input: &serde_json::Value,
        _cwd: &Path,
        permission: Permission,
    ) -> ToolDecision {
        match permission {
            Permission::ReadOnly => ToolDecision::Execute,
            perm => ToolDecision::AskUser(perm),
        }
    }
}

// ---------------------------------------------------------------------------
// Composable policy pipeline (replaces nested wrapper pattern)
// ---------------------------------------------------------------------------

/// A single policy rule in a pipeline.  Returns `Some(decision)` to
/// short-circuit, or `None` to pass to the next rule.
pub trait PolicyRule: Send + Sync {
    fn check(
        &self,
        name: &str,
        input: &serde_json::Value,
        cwd: &Path,
        permission: Permission,
    ) -> Option<ToolDecision>;
}

/// Evaluates rules in order; first `Some` wins.  If no rule matches,
/// falls back to permission-based default (readonly=Execute, else=AskUser).
pub struct CompositePolicy {
    rules: Vec<Box<dyn PolicyRule>>,
}

impl CompositePolicy {
    pub fn new(rules: Vec<Box<dyn PolicyRule>>) -> Self {
        Self { rules }
    }
}

impl ToolPolicy for CompositePolicy {
    fn classify(
        &self,
        name: &str,
        input: &serde_json::Value,
        cwd: &Path,
        permission: Permission,
    ) -> ToolDecision {
        for rule in &self.rules {
            if let Some(decision) = rule.check(name, input, cwd, permission) {
                return decision;
            }
        }
        // Default fallback
        match permission {
            Permission::ReadOnly => ToolDecision::Execute,
            perm => ToolDecision::AskUser(perm),
        }
    }
}

// ---------------------------------------------------------------------------
// Safety policy — blocklist wrapper
// ---------------------------------------------------------------------------

/// Dangerous command patterns that are ALWAYS denied regardless of
/// user permissions or yolo mode. Protects against catastrophic
/// mistakes (rm -rf /, dd to disk, fork bombs).
const BLOCKED_PATTERNS: &[&str] = &[
    "rm -rf /",
    "rm -rf /*",
    "rm -Rf /",
    "rm -fr /",
    "dd if=",
    "mkfs.",
    "> /dev/sd",
    "> /dev/nvme",
    ":(){ :|:& };:", // fork bomb
    "chmod -R 777 /",
    "chown -R",
    "shutdown",
    "reboot",
    "init 0",
    "init 6",
];

/// Check if a bash command matches any blocked pattern.
fn is_blocked_command(command: &str) -> Option<&'static str> {
    let lower = command.to_ascii_lowercase();
    let trimmed = lower.trim();
    for &pattern in BLOCKED_PATTERNS {
        if let Some(pos) = trimmed.find(pattern) {
            let after = pos + pattern.len();
            // For path-based patterns ("rm -rf /"), only match if
            // the pattern is at end, followed by space, or followed by *.
            // This prevents "rm -rf /tmp" from matching "rm -rf /".
            if pattern.ends_with('/') {
                if after >= trimmed.len()
                    || trimmed.as_bytes()[after] == b'*'
                    || trimmed.as_bytes()[after] == b' '
                {
                    return Some(pattern);
                }
            } else {
                return Some(pattern);
            }
        }
    }
    // Also block piping to /dev/sda etc.
    if trimmed.contains("/dev/sd") && (trimmed.contains('>') || trimmed.contains("dd ")) {
        return Some("/dev/sd write");
    }
    None
}

// Old SafetyPolicy/SelfProtectPolicy wrappers removed — use default_pipeline() instead.

/// Extract the target path from a tool call's JSON input.
fn extract_target_path(name: &str, input: &serde_json::Value) -> Option<String> {
    match name {
        "write_file" | "edit_file" | "read_file" => input
            .get("file_path")
            .or_else(|| input.get("path"))
            .and_then(|v| v.as_str())
            .map(String::from),
        "apply_patch" => input.get("path").and_then(|v| v.as_str()).map(String::from),
        _ => None,
    }
}

/// Check if a bash command targets protected paths.
fn bash_targets_own_source(command: &str, protected_dir: &str) -> bool {
    let cmd = command.trim();
    if (cmd.contains("cargo build") || cmd.contains("cargo install"))
        && (cmd.contains(protected_dir) || !cmd.contains("--manifest-path"))
    {
        return true;
    }
    let destructive = ["sed -i", "tee ", "mv ", "cp ", "rm ", "> ", ">> "];
    if destructive.iter().any(|d| cmd.contains(d)) && cmd.contains(protected_dir) {
        return true;
    }
    false
}

/// Safety blocklist as a composable rule.
pub struct SafetyRule;

impl PolicyRule for SafetyRule {
    fn check(
        &self,
        name: &str,
        input: &serde_json::Value,
        _cwd: &Path,
        _permission: Permission,
    ) -> Option<ToolDecision> {
        if name == "bash"
            && let Some(cmd) = input.get("command").and_then(|v| v.as_str())
            && let Some(pattern) = is_blocked_command(cmd)
        {
            return Some(ToolDecision::Deny(format!(
                "\u{1f6d1} Blocked: command matches safety pattern `{pattern}`"
            )));
        }
        None
    }
}

/// Self-protection as a composable rule.
pub struct SelfProtectRule {
    protected_dir: Option<String>,
}

impl SelfProtectRule {
    pub fn new(own_source_dir: Option<std::path::PathBuf>) -> Self {
        Self {
            protected_dir: own_source_dir
                .map(|p| p.canonicalize().unwrap_or(p).display().to_string()),
        }
    }
}

impl PolicyRule for SelfProtectRule {
    fn check(
        &self,
        name: &str,
        input: &serde_json::Value,
        cwd: &Path,
        _permission: Permission,
    ) -> Option<ToolDecision> {
        let protected = self.protected_dir.as_ref()?;

        if matches!(name, "write_file" | "edit_file" | "apply_patch")
            && let Some(target) = extract_target_path(name, input)
        {
            let abs = if std::path::Path::new(&target).is_absolute() {
                target.clone()
            } else {
                cwd.join(&target).display().to_string()
            };
            if abs.contains(protected) {
                return Some(ToolDecision::Deny(format!(
                    "\u{1f6e1}\u{fe0f} Self-protection: cannot modify own source code at `{target}`."
                )));
            }
        }

        if name == "bash"
            && let Some(cmd) = input.get("command").and_then(|v| v.as_str())
            && bash_targets_own_source(cmd, protected)
        {
            return Some(ToolDecision::Deny(format!(
                "\u{1f6e1}\u{fe0f} Self-protection: bash command would modify own source code. Blocked: `{}`",
                if cmd.len() > 80 { &cmd[..80] } else { cmd }
            )));
        }

        None
    }
}

/// Build the default production policy pipeline.
pub fn default_pipeline(own_source_dir: Option<std::path::PathBuf>) -> CompositePolicy {
    CompositePolicy::new(vec![
        Box::new(SelfProtectRule::new(own_source_dir)),
        Box::new(SafetyRule),
        // No more rules → fallback to permission-based default
    ])
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_policy_readonly_executes() {
        let policy = DefaultPolicy;
        let decision = policy.classify(
            "read_file",
            &serde_json::json!({"path": "foo.rs"}),
            Path::new("/tmp"),
            Permission::ReadOnly,
        );
        assert_eq!(decision, ToolDecision::Execute);
    }

    #[test]
    fn default_policy_write_asks() {
        let policy = DefaultPolicy;
        let decision = policy.classify(
            "write_file",
            &serde_json::json!({"path": "foo.rs"}),
            Path::new("/tmp"),
            Permission::WorkspaceWrite,
        );
        assert_eq!(decision, ToolDecision::AskUser(Permission::WorkspaceWrite));
    }

    #[test]
    fn default_policy_dangerous_asks() {
        let policy = DefaultPolicy;
        let decision = policy.classify(
            "bash",
            &serde_json::json!({"command": "rm -rf /"}),
            Path::new("/tmp"),
            Permission::Dangerous,
        );
        assert_eq!(decision, ToolDecision::AskUser(Permission::Dangerous));
    }

    /// Custom policy that denies all bash commands.
    struct NoBashPolicy;
    impl ToolPolicy for NoBashPolicy {
        fn classify(
            &self,
            name: &str,
            _input: &serde_json::Value,
            _cwd: &Path,
            permission: Permission,
        ) -> ToolDecision {
            if name == "bash" {
                return ToolDecision::Deny("bash is disabled".into());
            }
            match permission {
                Permission::ReadOnly => ToolDecision::Execute,
                perm => ToolDecision::AskUser(perm),
            }
        }
    }

    #[test]
    fn custom_policy_blocks_bash() {
        let policy = NoBashPolicy;
        let decision = policy.classify(
            "bash",
            &serde_json::json!({"command": "ls"}),
            Path::new("/tmp"),
            Permission::Dangerous,
        );
        assert_eq!(decision, ToolDecision::Deny("bash is disabled".into()));
    }

    #[test]
    fn custom_policy_allows_read() {
        let policy = NoBashPolicy;
        let decision = policy.classify(
            "read_file",
            &serde_json::json!({}),
            Path::new("/tmp"),
            Permission::ReadOnly,
        );
        assert_eq!(decision, ToolDecision::Execute);
    }

    // ── Safety blocklist tests ───────────────────────────────────────

    #[test]
    fn safety_blocks_rm_rf_root() {
        let policy = default_pipeline(None);
        let d = policy.classify(
            "bash",
            &serde_json::json!({"command": "rm -rf /"}),
            Path::new("/tmp"),
            Permission::Dangerous,
        );
        assert!(matches!(d, ToolDecision::Deny(_)));
    }

    #[test]
    fn safety_blocks_rm_rf_star() {
        let policy = default_pipeline(None);
        let d = policy.classify(
            "bash",
            &serde_json::json!({"command": "sudo rm -rf /*"}),
            Path::new("/tmp"),
            Permission::Dangerous,
        );
        assert!(matches!(d, ToolDecision::Deny(_)));
    }

    #[test]
    fn safety_blocks_dd() {
        let policy = default_pipeline(None);
        let d = policy.classify(
            "bash",
            &serde_json::json!({"command": "dd if=/dev/zero of=/dev/sda"}),
            Path::new("/tmp"),
            Permission::Dangerous,
        );
        assert!(matches!(d, ToolDecision::Deny(_)));
    }

    #[test]
    fn safety_blocks_fork_bomb() {
        let policy = default_pipeline(None);
        let d = policy.classify(
            "bash",
            &serde_json::json!({"command": ":(){ :|:& };:"}),
            Path::new("/tmp"),
            Permission::Dangerous,
        );
        assert!(matches!(d, ToolDecision::Deny(_)));
    }

    #[test]
    fn safety_blocks_mkfs() {
        let policy = default_pipeline(None);
        let d = policy.classify(
            "bash",
            &serde_json::json!({"command": "mkfs.ext4 /dev/sda1"}),
            Path::new("/tmp"),
            Permission::Dangerous,
        );
        assert!(matches!(d, ToolDecision::Deny(_)));
    }

    #[test]
    fn safety_allows_normal_rm() {
        let policy = default_pipeline(None);
        let d = policy.classify(
            "bash",
            &serde_json::json!({"command": "rm -rf /tmp/test_dir"}),
            Path::new("/tmp"),
            Permission::Dangerous,
        );
        // Not blocked — "rm -rf /tmp" doesn't match "rm -rf /" (trailing content)
        assert_eq!(d, ToolDecision::AskUser(Permission::Dangerous));
    }

    #[test]
    fn safety_allows_safe_commands() {
        let policy = default_pipeline(None);
        for cmd in ["ls -la", "cargo test", "python3 main.py", "git status"] {
            let d = policy.classify(
                "bash",
                &serde_json::json!({"command": cmd}),
                Path::new("/tmp"),
                Permission::Dangerous,
            );
            assert_eq!(
                d,
                ToolDecision::AskUser(Permission::Dangerous),
                "should allow: {cmd}"
            );
        }
    }

    #[test]
    fn safety_ignores_non_bash_tools() {
        let policy = default_pipeline(None);
        let d = policy.classify(
            "write_file",
            &serde_json::json!({"command": "rm -rf /", "path": "x"}),
            Path::new("/tmp"),
            Permission::WorkspaceWrite,
        );
        // Not bash — blocklist doesn't apply
        assert_eq!(d, ToolDecision::AskUser(Permission::WorkspaceWrite));
    }

    #[test]
    fn safety_blocks_shutdown() {
        let policy = default_pipeline(None);
        let d = policy.classify(
            "bash",
            &serde_json::json!({"command": "shutdown -h now"}),
            Path::new("/tmp"),
            Permission::Dangerous,
        );
        assert!(matches!(d, ToolDecision::Deny(_)));
    }

    #[test]
    fn is_blocked_returns_pattern() {
        assert_eq!(is_blocked_command("rm -rf /"), Some("rm -rf /"));
        assert_eq!(is_blocked_command("ls -la"), None);
    }

    // ── CompositePolicy / pipeline tests ─────────────────────────────

    #[test]
    fn composite_first_deny_wins() {
        let pipeline = default_pipeline(Some(std::path::PathBuf::from("/bot/crates")));
        let d = pipeline.classify(
            "write_file",
            &serde_json::json!({"file_path": "/bot/crates/core/main.rs"}),
            Path::new("/bot"),
            Permission::WorkspaceWrite,
        );
        assert!(matches!(d, ToolDecision::Deny(msg) if msg.contains("Self-protection")));
    }

    #[test]
    fn composite_fallback_when_no_rule_matches() {
        let pipeline = default_pipeline(None);
        let d = pipeline.classify(
            "read_file",
            &serde_json::json!({"path": "foo.rs"}),
            Path::new("/tmp"),
            Permission::ReadOnly,
        );
        assert_eq!(d, ToolDecision::Execute);
    }

    // ── Self-protection tests ────────────────────────────────────────

    #[test]
    fn self_protect_blocks_write_to_own_source() {
        let policy = default_pipeline(Some(std::path::PathBuf::from("/home/bot/naked/crates")));
        let d = policy.classify(
            "write_file",
            &serde_json::json!({"file_path": "/home/bot/naked/crates/naked-core/src/main.rs"}),
            Path::new("/home/bot/naked"),
            Permission::WorkspaceWrite,
        );
        assert!(matches!(d, ToolDecision::Deny(msg) if msg.contains("Self-protection")));
    }

    #[test]
    fn self_protect_allows_read_own_source() {
        let policy = default_pipeline(Some(std::path::PathBuf::from("/home/bot/naked/crates")));
        let d = policy.classify(
            "read_file",
            &serde_json::json!({"file_path": "/home/bot/naked/crates/naked-core/src/main.rs"}),
            Path::new("/home/bot/naked"),
            Permission::ReadOnly,
        );
        assert_eq!(d, ToolDecision::Execute);
    }

    #[test]
    fn self_protect_allows_write_elsewhere() {
        let policy = default_pipeline(Some(std::path::PathBuf::from("/home/bot/naked/crates")));
        let d = policy.classify(
            "write_file",
            &serde_json::json!({"file_path": "/tmp/script.py"}),
            Path::new("/tmp"),
            Permission::WorkspaceWrite,
        );
        assert_eq!(d, ToolDecision::AskUser(Permission::WorkspaceWrite));
    }

    #[test]
    fn self_protect_blocks_cargo_build_own() {
        let policy = default_pipeline(Some(std::path::PathBuf::from("/home/bot/naked/crates")));
        let d = policy.classify(
            "bash",
            &serde_json::json!({"command": "cd /home/bot/naked && cargo build --release"}),
            Path::new("/home/bot/naked"),
            Permission::Dangerous,
        );
        // cargo build without --manifest-path in default cwd → blocked
        assert!(matches!(d, ToolDecision::Deny(msg) if msg.contains("Self-protection")));
    }

    #[test]
    fn self_protect_allows_cargo_test_other_project() {
        let policy = default_pipeline(Some(std::path::PathBuf::from("/home/bot/naked/crates")));
        let d = policy.classify(
            "bash",
            &serde_json::json!({"command": "cargo build --manifest-path /home/user/project/Cargo.toml"}),
            Path::new("/home/user/project"),
            Permission::Dangerous,
        );
        assert_eq!(d, ToolDecision::AskUser(Permission::Dangerous));
    }

    #[test]
    fn self_protect_blocks_sed_on_own_source() {
        let policy = default_pipeline(Some(std::path::PathBuf::from("/home/bot/naked/crates")));
        let d = policy.classify(
            "bash",
            &serde_json::json!({"command": "sed -i 's/foo/bar/' /home/bot/naked/crates/naked-core/src/lib.rs"}),
            Path::new("/home/bot/naked"),
            Permission::Dangerous,
        );
        assert!(matches!(d, ToolDecision::Deny(_)));
    }

    #[test]
    fn self_protect_disabled_when_none() {
        let policy = default_pipeline(None);
        let d = policy.classify(
            "write_file",
            &serde_json::json!({"file_path": "/anywhere/main.rs"}),
            Path::new("/tmp"),
            Permission::WorkspaceWrite,
        );
        assert_eq!(d, ToolDecision::AskUser(Permission::WorkspaceWrite));
    }
}
