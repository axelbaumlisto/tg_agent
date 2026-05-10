//! Post-edit LSP hook integration for the agent loop.
//!
//! Lives in `loop_/` so the heavy `crate::lsp` module isn't loaded
//! into the loop's hot path unless it's actually invoked. The
//! hook is fired after every successful `edit_file` /
//! `apply_patch` / `write_file` tool call.

use std::path::PathBuf;

/// Extract the file paths that a tool call has just edited. Returns
/// the empty vec for non-edit tools so callers can use the result
/// length as a "is this an edit?" check.
#[must_use]
#[allow(dead_code)] // wired in a follow-up commit (T2 wiring step)
pub fn edited_paths_for_tool(tool_name: &str, input: &serde_json::Value) -> Vec<PathBuf> {
    match tool_name {
        "edit_file" | "write_file" => input
            .get("path")
            .and_then(|v| v.as_str())
            .map(|p| vec![PathBuf::from(p)])
            .unwrap_or_default(),
        "apply_patch" => {
            let mut out = Vec::new();
            // Shape 1: `{"path": "...", "patch": "..."}`.
            if let Some(p) = input.get("path").and_then(|v| v.as_str()) {
                out.push(PathBuf::from(p));
            }
            // Shape 2: `{"files": [{"path": "...", "content": "..."}]}`.
            if let Some(files) = input.get("files").and_then(|v| v.as_array()) {
                for entry in files {
                    if let Some(p) = entry.get("path").and_then(|v| v.as_str()) {
                        out.push(PathBuf::from(p));
                    }
                }
            }
            // Shape 3: parse `+++ b/<path>` headers from raw diff.
            if out.is_empty()
                && let Some(diff) = input.get("patch").and_then(|v| v.as_str())
            {
                out.extend(parse_patch_paths(diff));
            }
            out
        }
        _ => Vec::new(),
    }
}

#[allow(dead_code)] // wired in a follow-up commit (T2 wiring step)
fn parse_patch_paths(diff: &str) -> Vec<PathBuf> {
    let mut out = Vec::new();
    for line in diff.lines() {
        if let Some(rest) = line.strip_prefix("+++ ") {
            let trimmed = rest.trim();
            let path = trimmed.strip_prefix("b/").unwrap_or(trimmed);
            if path == "/dev/null" {
                continue;
            }
            out.push(PathBuf::from(path));
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn edit_file_extracts_path() {
        let v = json!({"path": "src/foo.rs", "content": "x"});
        let paths = edited_paths_for_tool("edit_file", &v);
        assert_eq!(paths, vec![PathBuf::from("src/foo.rs")]);
    }

    #[test]
    fn write_file_extracts_path() {
        let v = json!({"path": "a/b.py"});
        let paths = edited_paths_for_tool("write_file", &v);
        assert_eq!(paths, vec![PathBuf::from("a/b.py")]);
    }

    #[test]
    fn apply_patch_files_array() {
        let v = json!({
            "files": [
                {"path": "x.rs", "content": "a"},
                {"path": "y.rs", "content": "b"}
            ]
        });
        let paths = edited_paths_for_tool("apply_patch", &v);
        assert_eq!(paths.len(), 2);
        assert!(paths.contains(&PathBuf::from("x.rs")));
    }

    #[test]
    fn apply_patch_unified_diff_headers() {
        let diff = "diff --git a/foo.rs b/foo.rs\n--- a/foo.rs\n+++ b/foo.rs\n@@\n+x\n";
        let v = json!({"patch": diff});
        let paths = edited_paths_for_tool("apply_patch", &v);
        assert_eq!(paths, vec![PathBuf::from("foo.rs")]);
    }

    #[test]
    fn unknown_tool_returns_empty() {
        let v = json!({"path": "foo.rs"});
        assert!(edited_paths_for_tool("read_file", &v).is_empty());
        assert!(edited_paths_for_tool("bash", &v).is_empty());
    }

    #[test]
    fn dev_null_in_diff_skipped() {
        let diff = "+++ /dev/null\n+++ b/keep.rs\n";
        let v = json!({"patch": diff});
        let paths = edited_paths_for_tool("apply_patch", &v);
        assert_eq!(paths, vec![PathBuf::from("keep.rs")]);
    }
}
