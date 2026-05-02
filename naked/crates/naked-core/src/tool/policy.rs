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
        assert_eq!(
            decision,
            ToolDecision::Deny("bash is disabled".into())
        );
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
}
