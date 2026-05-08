//! Working set — tracks files the agent has interacted with.
//!
//! Observes tool calls (read_file, write_file, edit_file, bash) and
//! maintains a compact set of "active" paths. This set is:
//! 1. Injected into system prompt for context awareness
//! 2. Preserved during compaction so the model retains focus
//!
//! KISS: no repo walking, no fuzzy matching. Just observe tool I/O.

use std::collections::BTreeSet;
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

/// Maximum files in the working set before LRU eviction.
const MAX_FILES: usize = 20;

/// Tracks recently-accessed files for context awareness.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct WorkingSet {
    /// Files ordered by most-recent access (last = most recent).
    files: Vec<PathBuf>,
}

impl WorkingSet {
    pub fn new() -> Self {
        Self { files: Vec::new() }
    }

    /// Record a file access (read, write, edit).
    pub fn touch(&mut self, path: &Path) {
        // Remove if already present (will be re-added at end):
        self.files.retain(|p| p != path);
        self.files.push(path.to_path_buf());
        // Evict oldest if over limit:
        while self.files.len() > MAX_FILES {
            self.files.remove(0);
        }
    }

    /// Extract file paths from a tool call and touch them.
    pub fn observe_tool(&mut self, tool_name: &str, input: &serde_json::Value) {
        match tool_name {
            "read_file" | "write_file" | "edit_file" => {
                if let Some(path) = input
                    .get("file_path")
                    .or_else(|| input.get("path"))
                    .and_then(|v| v.as_str())
                {
                    self.touch(Path::new(path));
                }
            }
            "apply_patch" => {
                if let Some(patch) = input.get("patch").and_then(|v| v.as_str()) {
                    for line in patch.lines() {
                        if let Some(rest) = line.strip_prefix("+++ ") {
                            let path = rest.trim().strip_prefix("b/").unwrap_or(rest.trim());
                            if path != "/dev/null" && !path.is_empty() {
                                self.touch(Path::new(path));
                            }
                        }
                    }
                }
                if let Some(path) = input.get("path").and_then(|v| v.as_str()) {
                    self.touch(Path::new(path));
                }
            }
            "file_search" => {
                if let Some(path) = input.get("path").and_then(|v| v.as_str()) {
                    self.touch(Path::new(path));
                }
            }
            "bash" => {
                if let Some(cmd) = input.get("command").and_then(|v| v.as_str()) {
                    for path in extract_paths_from_command(cmd) {
                        self.touch(Path::new(&path));
                    }
                }
            }
            _ => {}
        }
    }

    /// Format as a compact block for system prompt injection.
    pub fn to_prompt_block(&self) -> String {
        if self.files.is_empty() {
            return String::new();
        }
        let mut block = String::from("\n[Working set — recently active files]\n");
        for (i, path) in self.files.iter().rev().take(10).enumerate() {
            block.push_str(&format!("  {}. {}\n", i + 1, path.display()));
        }
        block
    }

    /// Get the list of active files (most recent first).
    pub fn active_files(&self) -> Vec<&Path> {
        self.files.iter().rev().map(|p| p.as_path()).collect()
    }

    pub fn len(&self) -> usize {
        self.files.len()
    }

    pub fn is_empty(&self) -> bool {
        self.files.is_empty()
    }

    /// Return indices of messages that mention any file in the working set.
    pub fn pinned_message_indices(
        &self,
        messages: &[crate::types::ConversationMessage],
    ) -> Vec<usize> {
        if self.files.is_empty() {
            return Vec::new();
        }
        let needles: Vec<String> = self.files.iter().map(|p| p.display().to_string()).collect();
        messages
            .iter()
            .enumerate()
            .filter_map(|(i, msg)| {
                let text = msg.text_content();
                if needles.iter().any(|n| text.contains(n.as_str())) {
                    Some(i)
                } else {
                    None
                }
            })
            .collect()
    }

    /// Top N paths (most-recent first) as display strings.
    pub fn top_paths(&self, n: usize) -> Vec<String> {
        self.files
            .iter()
            .rev()
            .take(n)
            .map(|p| p.display().to_string())
            .collect()
    }
}

/// Best-effort extraction of file paths from bash commands.
fn extract_paths_from_command(cmd: &str) -> Vec<String> {
    let mut paths = Vec::new();
    let unique: BTreeSet<String> = BTreeSet::new();
    for token in cmd.split_whitespace() {
        // Skip flags and operators:
        if token.starts_with('-') || token.contains('|') || token.contains('>') {
            continue;
        }
        // Looks like a file path:
        if (token.contains('/') || token.contains('.'))
            && !token.starts_with("http")
            && !token.starts_with("//")
            && token.len() < 200
        {
            let cleaned = token.trim_matches(|c: char| c == '\'' || c == '"' || c == ';');
            if !cleaned.is_empty() && !unique.contains(cleaned) {
                paths.push(cleaned.to_string());
            }
        }
    }
    paths
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn touch_and_order() {
        let mut ws = WorkingSet::new();
        ws.touch(Path::new("a.rs"));
        ws.touch(Path::new("b.rs"));
        ws.touch(Path::new("a.rs")); // re-touch → moves to end
        let active = ws.active_files();
        assert_eq!(active[0], Path::new("a.rs")); // most recent
        assert_eq!(active[1], Path::new("b.rs"));
    }

    #[test]
    fn eviction_at_max() {
        let mut ws = WorkingSet::new();
        for i in 0..25 {
            ws.touch(Path::new(&format!("file{i}.rs")));
        }
        assert_eq!(ws.len(), MAX_FILES);
    }

    #[test]
    fn observe_read_file() {
        let mut ws = WorkingSet::new();
        ws.observe_tool(
            "read_file",
            &serde_json::json!({"file_path": "src/main.rs"}),
        );
        assert_eq!(ws.len(), 1);
        assert_eq!(ws.active_files()[0], Path::new("src/main.rs"));
    }

    #[test]
    fn observe_bash_extracts_paths() {
        let mut ws = WorkingSet::new();
        ws.observe_tool(
            "bash",
            &serde_json::json!({"command": "cat src/lib.rs && cargo test"}),
        );
        assert!(
            ws.active_files()
                .iter()
                .any(|p| *p == Path::new("src/lib.rs"))
        );
    }

    #[test]
    fn observe_unknown_tool_noop() {
        let mut ws = WorkingSet::new();
        ws.observe_tool("web_search", &serde_json::json!({"query": "test"}));
        assert!(ws.is_empty());
    }

    #[test]
    fn prompt_block_empty_when_empty() {
        let ws = WorkingSet::new();
        assert!(ws.to_prompt_block().is_empty());
    }

    #[test]
    fn prompt_block_format() {
        let mut ws = WorkingSet::new();
        ws.touch(Path::new("foo.rs"));
        ws.touch(Path::new("bar.rs"));
        let block = ws.to_prompt_block();
        assert!(block.contains("Working set"));
        assert!(block.contains("bar.rs")); // most recent first
    }

    #[test]
    fn extract_paths_from_command_works() {
        let paths = extract_paths_from_command("cat src/main.rs | grep foo > /tmp/out.txt");
        assert!(paths.contains(&"src/main.rs".to_string()));
        // > and | tokens are skipped, but /tmp/out.txt has / so may be included
    }

    #[test]
    fn extract_paths_skips_urls() {
        let paths = extract_paths_from_command("curl https://example.com/api src/test.rs");
        assert!(!paths.iter().any(|p| p.contains("example.com")));
        assert!(paths.contains(&"src/test.rs".to_string()));
    }

    #[test]
    fn observe_apply_patch_extracts_paths() {
        let mut ws = WorkingSet::new();
        ws.observe_tool(
            "apply_patch",
            &serde_json::json!({"patch": "--- a/old.rs\n+++ b/src/new.rs\n@@ -1 +1 @@"}),
        );
        assert!(
            ws.active_files()
                .iter()
                .any(|p| *p == Path::new("src/new.rs"))
        );
    }

    #[test]
    fn observe_file_search_touches_path() {
        let mut ws = WorkingSet::new();
        ws.observe_tool("file_search", &serde_json::json!({"path": "src/lib.rs"}));
        assert_eq!(ws.active_files()[0], Path::new("src/lib.rs"));
    }

    #[test]
    fn pinned_message_indices_finds_relevant() {
        use crate::types::ConversationMessage;
        let mut ws = WorkingSet::new();
        ws.touch(Path::new("src/main.rs"));
        let messages = vec![
            ConversationMessage::user("hello"),
            ConversationMessage::user("edit src/main.rs please"),
            ConversationMessage::user("thanks"),
        ];
        assert_eq!(ws.pinned_message_indices(&messages), vec![1]);
    }

    #[test]
    fn pinned_empty_ws_returns_empty() {
        use crate::types::ConversationMessage;
        let ws = WorkingSet::new();
        assert!(
            ws.pinned_message_indices(&[ConversationMessage::user("src/main.rs")])
                .is_empty()
        );
    }

    #[test]
    fn top_paths_returns_most_recent_first() {
        let mut ws = WorkingSet::new();
        ws.touch(Path::new("a.rs"));
        ws.touch(Path::new("b.rs"));
        ws.touch(Path::new("c.rs"));
        assert_eq!(ws.top_paths(2), vec!["c.rs", "b.rs"]);
    }
}
