//! Approval cache with call fingerprints.
//!
//! After the user approves a tool call, its fingerprint is cached.
//! Subsequent calls with the same fingerprint are auto-approved.
//! Fingerprints are semantic — `cargo test --verbose` matches
//! `cargo test` but not `rm -rf`.

use std::collections::HashSet;
use std::sync::{Arc, Mutex};

/// Compute a fingerprint for a tool call.
/// The fingerprint captures the tool name + semantic portion of args.
pub fn fingerprint(tool_name: &str, input: &serde_json::Value) -> String {
    match tool_name {
        "bash" => {
            // Use command prefix (first 2-3 tokens, skipping flags):
            let cmd = input.get("command").and_then(|v| v.as_str()).unwrap_or("");
            let prefix = command_prefix(cmd);
            format!("bash:{prefix}")
        }
        "write_file" | "edit_file" => {
            let path = input
                .get("file_path")
                .or_else(|| input.get("path"))
                .and_then(|v| v.as_str())
                .unwrap_or("?");
            format!("{tool_name}:{path}")
        }
        _ => format!("tool:{tool_name}"),
    }
}

/// Extract canonical command prefix using arity dictionary.
fn command_prefix(cmd: &str) -> String {
    crate::command_arity::canonical_prefix(cmd)
}

/// Thread-safe approval cache.
#[derive(Debug, Clone, Default)]
pub struct ApprovalCache {
    approved: Arc<Mutex<HashSet<String>>>,
}

impl ApprovalCache {
    pub fn new() -> Self {
        Self::default()
    }

    /// Check if a call fingerprint was previously approved.
    pub fn is_approved(&self, fp: &str) -> bool {
        self.approved
            .lock()
            .map(|g| g.contains(fp))
            .unwrap_or(false)
    }

    /// Record an approval.
    pub fn approve(&self, fp: &str) {
        if let Ok(mut g) = self.approved.lock() {
            g.insert(fp.to_string());
        }
    }

    /// Number of cached approvals.
    pub fn len(&self) -> usize {
        self.approved.lock().map(|g| g.len()).unwrap_or(0)
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fingerprint_bash_uses_prefix() {
        let fp1 = fingerprint("bash", &serde_json::json!({"command": "cargo test"}));
        let fp2 = fingerprint(
            "bash",
            &serde_json::json!({"command": "cargo test --verbose"}),
        );
        let fp3 = fingerprint("bash", &serde_json::json!({"command": "rm -rf /tmp"}));
        assert_eq!(fp1, fp2); // same prefix
        assert_ne!(fp1, fp3); // different command
    }

    #[test]
    fn fingerprint_file_ops_uses_path() {
        let fp1 = fingerprint(
            "write_file",
            &serde_json::json!({"file_path": "src/main.rs"}),
        );
        let fp2 = fingerprint(
            "write_file",
            &serde_json::json!({"file_path": "src/main.rs"}),
        );
        let fp3 = fingerprint(
            "write_file",
            &serde_json::json!({"file_path": "src/lib.rs"}),
        );
        assert_eq!(fp1, fp2);
        assert_ne!(fp1, fp3);
    }

    #[test]
    fn fingerprint_unknown_tool() {
        let fp = fingerprint("custom_tool", &serde_json::json!({}));
        assert_eq!(fp, "tool:custom_tool");
    }

    #[test]
    fn cache_approve_and_check() {
        let cache = ApprovalCache::new();
        assert!(!cache.is_approved("bash:cargo test"));
        cache.approve("bash:cargo test");
        assert!(cache.is_approved("bash:cargo test"));
        assert!(!cache.is_approved("bash:rm -rf"));
    }

    #[test]
    fn cache_is_thread_safe() {
        let cache = ApprovalCache::new();
        let c2 = cache.clone();
        cache.approve("a");
        assert!(c2.is_approved("a"));
    }
}
