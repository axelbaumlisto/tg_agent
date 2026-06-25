//! `apply_patch` tool — apply unified diffs to files.
//!
//! More reliable than edit_file for multi-line changes.
//! Supports fuzzy matching of context lines.

use crate::types::{Permission, ToolResult, ToolSpec};
use std::path::Path;
use std::sync::Arc;

pub struct ApplyPatchTool {
    fs_cache: Option<Arc<super::fs_cache::FsCache>>,
}

impl ApplyPatchTool {
    pub fn new(fs_cache: Option<Arc<super::fs_cache::FsCache>>) -> Self {
        Self { fs_cache }
    }
}

impl Default for ApplyPatchTool {
    fn default() -> Self {
        Self::new(None)
    }
}

#[async_trait::async_trait]
impl crate::tool::Tool for ApplyPatchTool {
    fn spec(&self) -> ToolSpec {
        ToolSpec {
            name: "apply_patch".into(),
            description: "Apply a unified diff patch to a file. More reliable than edit_file \
                          for multi-line changes. Send the patch in standard unified diff format."
                .into(),
            parameters: serde_json::json!({
                "type": "object",
                "properties": {
                    "path": { "type": "string", "description": "File path to patch" },
                    "patch": { "type": "string", "description": "Unified diff (lines starting with +/-/@@ )" }
                },
                "required": ["path", "patch"]
            }),
            permission: Permission::WorkspaceWrite,
        }
    }

    async fn execute(&self, input: serde_json::Value, cwd: &Path) -> ToolResult {
        let file_path = match input.get("path").and_then(|v| v.as_str()) {
            Some(p) => p,
            None => {
                return ToolResult::err("path required");
            }
        };
        let patch = match input.get("patch").and_then(|v| v.as_str()) {
            Some(p) => p,
            None => {
                return ToolResult::err("patch required");
            }
        };

        let path = if Path::new(file_path).is_absolute() {
            std::path::PathBuf::from(file_path)
        } else {
            cwd.join(file_path)
        };

        let _file_guard = super::file_lock::lock_file(&path).await;
        let old = match tokio::fs::read_to_string(&path).await {
            Ok(s) => s,
            Err(e) => {
                return ToolResult::err(format!("Cannot read {}: {e}", path.display()));
            }
        };

        match apply_unified_diff(&old, patch) {
            Ok(new_content) => {
                let diff = super::diff_format::unified_diff(file_path, &old, &new_content);
                match super::atomic_write::atomic_replace_file(&path, new_content.as_bytes()).await
                {
                    Ok(()) => {
                        if let Some(cache) = &self.fs_cache {
                            cache.invalidate(&path).await;
                        }
                        ToolResult::ok(if diff.is_empty() {
                            "Patch applied (no changes)".to_string()
                        } else {
                            format!("Patch applied:\n\n{diff}")
                        })
                    }
                    Err(e) => ToolResult::err(format!("Write failed: {e}")),
                }
            }
            Err(e) => ToolResult::err(format!("Patch failed: {e}")),
        }
    }
}

/// Apply unified diff to content. Simple line-based approach.
fn apply_unified_diff(original: &str, patch: &str) -> Result<String, String> {
    let orig_lines: Vec<String> = original.lines().map(String::from).collect();
    let mut result: Vec<String> = orig_lines;

    for hunk in parse_hunks(patch) {
        let start = find_context(&result, &hunk.context_before, hunk.orig_start)
            .ok_or_else(|| format!("Cannot find context at line {}", hunk.orig_start))?;

        // Remove old lines:
        let remove_count = hunk.remove_lines.len();
        let end = (start + remove_count).min(result.len());
        result.splice(start..end, hunk.add_lines.clone());
    }

    Ok(result.join("\n") + if original.ends_with('\n') { "\n" } else { "" })
}

struct Hunk {
    orig_start: usize,
    context_before: Vec<String>,
    remove_lines: Vec<String>,
    add_lines: Vec<String>,
}

fn parse_hunks(patch: &str) -> Vec<Hunk> {
    let mut hunks = Vec::new();
    let mut current: Option<Hunk> = None;

    for line in patch.lines() {
        if line.starts_with("@@") {
            if let Some(h) = current.take() {
                hunks.push(h);
            }
            let start = parse_hunk_header(line);
            current = Some(Hunk {
                orig_start: start,
                context_before: Vec::new(),
                remove_lines: Vec::new(),
                add_lines: Vec::new(),
            });
        } else if let Some(ref mut h) = current {
            if let Some(stripped) = line.strip_prefix('-') {
                h.remove_lines.push(stripped.to_string());
            } else if let Some(stripped) = line.strip_prefix('+') {
                h.add_lines.push(stripped.to_string());
            } else if line.starts_with(' ') || line.is_empty() {
                let ctx = line.strip_prefix(' ').unwrap_or(line);
                if h.remove_lines.is_empty() && h.add_lines.is_empty() {
                    h.context_before.push(ctx.to_string());
                }
            }
        }
    }
    if let Some(h) = current {
        hunks.push(h);
    }
    hunks
}

fn parse_hunk_header(line: &str) -> usize {
    // @@ -10,3 +10,5 @@
    line.split('-')
        .nth(1)
        .and_then(|s| s.split(',').next())
        .and_then(|s| s.trim().parse::<usize>().ok())
        .map(|n| n.saturating_sub(1)) // 0-indexed
        .unwrap_or(0)
}

fn find_context(lines: &[String], context: &[String], hint: usize) -> Option<usize> {
    if context.is_empty() {
        return Some(hint.min(lines.len()));
    }
    // Try near hint first:
    let window = 10;
    let start = hint.saturating_sub(window);
    let end = (hint + window).min(lines.len());
    for i in start..end {
        if matches_context(lines, i, context) {
            return Some(i + context.len());
        }
    }
    // Full scan:
    for i in 0..lines.len() {
        if matches_context(lines, i, context) {
            return Some(i + context.len());
        }
    }
    // Fuzzy: just use hint
    Some(hint.min(lines.len()))
}

fn matches_context(lines: &[String], start: usize, context: &[String]) -> bool {
    if start + context.len() > lines.len() {
        return false;
    }
    context
        .iter()
        .enumerate()
        .all(|(i, c)| lines[start + i].trim() == c.trim())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tool::Tool;

    #[tokio::test]
    async fn apply_patch_uses_atomic_replace_success() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("patch.txt");
        std::fs::write(&path, "alpha\nbeta\ngamma\n").unwrap();

        let tool = ApplyPatchTool::default();
        let result = tool
            .execute(
                serde_json::json!({
                    "path": "patch.txt",
                    "patch": "@@ -2,1 +2,1 @@\n-beta\n+BETA\n"
                }),
                dir.path(),
            )
            .await;

        assert!(!result.is_error, "{}", result.output);
        assert_eq!(
            std::fs::read_to_string(&path).unwrap(),
            "alpha\nBETA\ngamma\n"
        );
        assert_no_atomic_temps(dir.path(), "patch.txt");
    }

    #[tokio::test]
    async fn apply_patch_failure_leaves_original() {
        let dir = tempfile::tempdir().unwrap();
        let protected = dir.path().join("protected");
        std::fs::create_dir(&protected).unwrap();
        let path = protected.join("patch.txt");
        let original = "alpha\nbeta\ngamma\n";
        std::fs::write(&path, original).unwrap();

        let original_mode = make_read_only_dir(&protected);
        let tool = ApplyPatchTool::default();
        let result = tool
            .execute(
                serde_json::json!({
                    "path": path.to_string_lossy(),
                    "patch": "@@ -2,1 +2,1 @@\n-beta\n+BETA\n"
                }),
                dir.path(),
            )
            .await;
        restore_dir_mode(&protected, original_mode);

        assert!(result.is_error);
        assert!(result.output.contains("Write failed:"), "{}", result.output);
        assert_eq!(std::fs::read_to_string(&path).unwrap(), original);
        assert_no_atomic_temps(&protected, "patch.txt");
    }

    #[tokio::test]
    async fn apply_patch_serializes_concurrent_same_file() {
        for i in 0..10 {
            let dir = tempfile::tempdir().unwrap();
            let path = dir.path().join("patch.txt");
            std::fs::write(&path, "a = 1\nb = 2\n").unwrap();

            let barrier = Arc::new(tokio::sync::Barrier::new(2));
            let cwd_a = dir.path().to_path_buf();
            let barrier_a = barrier.clone();
            let patch_a = tokio::spawn(async move {
                barrier_a.wait().await;
                ApplyPatchTool::default()
                    .execute(
                        serde_json::json!({
                            "path": "patch.txt",
                            "patch": "@@ -1,1 +1,1 @@\n-a = 1\n+a = 10\n"
                        }),
                        &cwd_a,
                    )
                    .await
            });

            let cwd_b = dir.path().to_path_buf();
            let barrier_b = barrier.clone();
            let patch_b = tokio::spawn(async move {
                barrier_b.wait().await;
                ApplyPatchTool::default()
                    .execute(
                        serde_json::json!({
                            "path": "patch.txt",
                            "patch": "@@ -2,1 +2,1 @@\n-b = 2\n+b = 20\n"
                        }),
                        &cwd_b,
                    )
                    .await
            });

            let (result_a, result_b) =
                tokio::time::timeout(std::time::Duration::from_secs(2), async {
                    tokio::join!(patch_a, patch_b)
                })
                .await
                .unwrap_or_else(|_| {
                    panic!("concurrent apply_patch ops timed out on iteration {i}")
                });
            let result_a = result_a.unwrap();
            let result_b = result_b.unwrap();
            assert!(!result_a.is_error, "iteration {i}: {}", result_a.output);
            assert!(!result_b.is_error, "iteration {i}: {}", result_b.output);
            assert_eq!(
                std::fs::read_to_string(&path).unwrap(),
                "a = 10\nb = 20\n",
                "iteration {i} lost one concurrent patch"
            );
        }
    }

    #[test]
    fn simple_add_line() {
        let orig = "line1\nline2\nline3\n";
        let patch = "@@ -2,1 +2,2 @@\n line2\n+inserted\n line3\n";
        let result = apply_unified_diff(orig, patch).unwrap();
        assert!(result.contains("inserted"));
        assert!(result.contains("line2"));
    }

    #[test]
    fn simple_remove_line() {
        let orig = "a\nb\nc\n";
        let patch = "@@ -1,3 +1,2 @@\n a\n-b\n c\n";
        let result = apply_unified_diff(orig, patch).unwrap();
        assert!(!result.contains("\nb\n"));
        assert!(result.contains("a"));
        assert!(result.contains("c"));
    }

    #[test]
    fn replace_line() {
        let orig = "old line\n";
        let patch = "@@ -1,1 +1,1 @@\n-old line\n+new line\n";
        let result = apply_unified_diff(orig, patch).unwrap();
        assert_eq!(result.trim(), "new line");
    }

    #[test]
    fn parse_hunk_header_works() {
        assert_eq!(parse_hunk_header("@@ -10,3 +10,5 @@"), 9);
        assert_eq!(parse_hunk_header("@@ -1,1 +1,1 @@"), 0);
    }

    fn assert_no_atomic_temps(dir: &Path, file_name: &str) {
        let prefix = format!(".{file_name}.tmp.");
        let leftovers: Vec<_> = std::fs::read_dir(dir)
            .unwrap()
            .filter_map(Result::ok)
            .filter(|entry| entry.file_name().to_string_lossy().starts_with(&prefix))
            .map(|entry| entry.path())
            .collect();
        assert!(leftovers.is_empty(), "leftover temp files: {leftovers:?}");
    }

    #[cfg(unix)]
    fn make_read_only_dir(path: &Path) -> Option<std::fs::Permissions> {
        use std::os::unix::fs::PermissionsExt;

        let original = std::fs::metadata(path).unwrap().permissions();
        std::fs::set_permissions(path, std::fs::Permissions::from_mode(0o555)).unwrap();
        Some(original)
    }

    #[cfg(not(unix))]
    fn make_read_only_dir(_path: &Path) -> Option<std::fs::Permissions> {
        None
    }

    #[cfg(unix)]
    fn restore_dir_mode(path: &Path, original: Option<std::fs::Permissions>) {
        if let Some(original) = original {
            std::fs::set_permissions(path, original).unwrap();
        }
    }

    #[cfg(not(unix))]
    fn restore_dir_mode(_path: &Path, _original: Option<std::fs::Permissions>) {}
}
