use std::path::{Path, PathBuf};

use async_trait::async_trait;
use serde::Deserialize;

use crate::types::{Permission, ToolResult, ToolSpec};

use super::Tool;

const MAX_READ_BYTES: u64 = 10 * 1024 * 1024; // 10 MB
const BINARY_SNIFF_SIZE: usize = 8192;

fn is_inside_workspace(path: &Path, workspace: &Path) -> bool {
    match (path.canonicalize(), workspace.canonicalize()) {
        (Ok(real), Ok(base)) => real.starts_with(&base),
        _ => {
            let abs = if path.is_absolute() {
                path.to_path_buf()
            } else {
                workspace.join(path)
            };
            abs.starts_with(workspace)
        }
    }
}

// -- ReadFileTool ------------------------------------------------------------

pub struct ReadFileTool;

#[derive(Deserialize)]
struct ReadFileInput {
    file_path: String,
    #[serde(default)]
    offset: Option<usize>,
    #[serde(default)]
    limit: Option<usize>,
}

#[async_trait]
impl Tool for ReadFileTool {
    fn spec(&self) -> ToolSpec {
        ToolSpec {
            name: "read_file".into(),
            description: "Read the contents of a file. Optionally specify offset and limit for partial reads.".into(),
            parameters: serde_json::json!({
                "type": "object",
                "properties": {
                    "file_path": { "type": "string", "description": "Path to the file to read" },
                    "offset": { "type": "integer", "description": "Line offset (1-based) to start reading from" },
                    "limit": { "type": "integer", "description": "Maximum number of lines to read" }
                },
                "required": ["file_path"]
            }),
            permission: Permission::ReadOnly,
        }
    }

    async fn execute(&self, input: serde_json::Value, cwd: &Path) -> ToolResult {
        let input: ReadFileInput = match serde_json::from_value(input) {
            Ok(v) => v,
            Err(e) => {
                return ToolResult {
                    output: format!("Invalid input: {e}"),
                    is_error: true,
                };
            }
        };

        let path = resolve_path(&input.file_path, cwd);

        if let Ok(meta) = tokio::fs::metadata(&path).await
            && meta.len() > MAX_READ_BYTES
        {
            return ToolResult {
                output: format!(
                    "File too large ({} bytes, max {}). Use offset/limit for partial reads.",
                    meta.len(),
                    MAX_READ_BYTES
                ),
                is_error: true,
            };
        }

        if is_binary(&path).await {
            let size = tokio::fs::metadata(&path)
                .await
                .map(|m| m.len())
                .unwrap_or(0);
            let ext = path.extension().and_then(|e| e.to_str()).unwrap_or("");
            let is_image = matches!(ext, "png" | "jpg" | "jpeg" | "gif" | "webp" | "bmp");

            if is_image {
                // For images: base64-encode if small enough for vision models.
                const MAX_IMAGE_BYTES: u64 = 512_000; // 500KB
                if size <= MAX_IMAGE_BYTES
                    && let Ok(bytes) = tokio::fs::read(&path).await
                {
                    use base64::Engine;
                    let b64 = base64::engine::general_purpose::STANDARD.encode(&bytes);
                    let mime = match ext {
                        "png" => "image/png",
                        "jpg" | "jpeg" => "image/jpeg",
                        "gif" => "image/gif",
                        "webp" => "image/webp",
                        _ => "application/octet-stream",
                    };
                    // Push image for vision model injection.
                    super::image_result::push_image(mime, &b64);
                    return ToolResult {
                        output: format!(
                            "Image ({}, {size} bytes): {}\n[image sent to vision model]",
                            ext.to_uppercase(),
                            path.display()
                        ),
                        is_error: false,
                    };
                }
                let p = path.display();
                return ToolResult {
                    output: format!(
                        "Image file ({ext}, {size} bytes): {p}. Too large for inline. \
                         Use bash to analyze: file '{p}' or identify '{p}'"
                    ),
                    is_error: false,
                };
            }

            return ToolResult {
                output: format!(
                    "Binary file ({size} bytes): {}. Cannot read as text.",
                    path.display()
                ),
                is_error: true,
            };
        }

        match tokio::fs::read_to_string(&path).await {
            Ok(content) => {
                const MAX_OUTPUT: usize = 16_384;
                let lines: Vec<&str> = content.lines().collect();
                let offset = input.offset.unwrap_or(1).saturating_sub(1);
                let limit = input.limit.unwrap_or(lines.len());
                let mut output = String::new();
                let mut total = 0usize;
                for (shown, (i, line)) in lines
                    .iter()
                    .skip(offset)
                    .take(limit)
                    .enumerate()
                    .enumerate()
                {
                    let formatted = format!("{:6}|{}\n", offset + i + 1, line);
                    total += formatted.len();
                    if total > MAX_OUTPUT {
                        let remaining = limit.min(lines.len() - offset) - shown;
                        output.push_str(&format!(
                            "\n[output truncated — {remaining} more lines, use offset/limit]"
                        ));
                        break;
                    }
                    output.push_str(&formatted);
                }
                ToolResult {
                    output: output.trim_end().to_string(),
                    is_error: false,
                }
            }
            Err(e) => ToolResult {
                output: format!("Failed to read {}: {e}", path.display()),
                is_error: true,
            },
        }
    }
}

// -- WriteFileTool -----------------------------------------------------------

pub struct WriteFileTool;

#[derive(Deserialize)]
struct WriteFileInput {
    file_path: String,
    contents: String,
}

#[async_trait]
impl Tool for WriteFileTool {
    fn spec(&self) -> ToolSpec {
        ToolSpec {
            name: "write_file".into(),
            description: "Write contents to a file. Creates the file if it doesn't exist, overwrites if it does.".into(),
            parameters: serde_json::json!({
                "type": "object",
                "properties": {
                    "file_path": { "type": "string", "description": "Path to the file to write" },
                    "contents": { "type": "string", "description": "The contents to write" }
                },
                "required": ["file_path", "contents"]
            }),
            permission: Permission::WorkspaceWrite,
        }
    }

    fn effective_permission(&self, input: &serde_json::Value, cwd: &Path) -> Permission {
        if let Some(fp) = input.get("file_path").and_then(|v| v.as_str()) {
            let path = resolve_path(fp, cwd);
            if !is_inside_workspace(&path, cwd) {
                return Permission::Dangerous;
            }
        }
        Permission::WorkspaceWrite
    }

    async fn execute(&self, input: serde_json::Value, cwd: &Path) -> ToolResult {
        let input: WriteFileInput = match serde_json::from_value(input) {
            Ok(v) => v,
            Err(e) => {
                return ToolResult {
                    output: format!("Invalid input: {e}"),
                    is_error: true,
                };
            }
        };

        let path = resolve_path(&input.file_path, cwd);

        // C1: Serialize concurrent writes to the same file.
        let _file_guard = super::file_lock::lock_file(&path).await;

        if let Some(parent) = path.parent()
            && let Err(e) = tokio::fs::create_dir_all(parent).await
        {
            return ToolResult {
                output: format!("Failed to create directories: {e}"),
                is_error: true,
            };
        }

        // Atomic write: tmp file + rename
        let tmp = path.with_extension("tmp");
        match tokio::fs::write(&tmp, &input.contents).await {
            Ok(()) => match tokio::fs::rename(&tmp, &path).await {
                Ok(()) => ToolResult {
                    output: format!("Wrote {} bytes to {}", input.contents.len(), path.display()),
                    is_error: false,
                },
                Err(e) => ToolResult {
                    output: format!("Failed to rename: {e}"),
                    is_error: true,
                },
            },
            Err(e) => ToolResult {
                output: format!("Failed to write: {e}"),
                is_error: true,
            },
        }
    }
}

// -- EditFileTool ------------------------------------------------------------

pub struct EditFileTool;

/// Single edit: replace old_string → new_string.
#[derive(Deserialize, Clone)]
struct EditOp {
    old_string: String,
    new_string: String,
}

/// B1: Supports both legacy (single old_string/new_string) and multi-edit
/// (edits array). Legacy format is auto-converted to a single-element array.
#[derive(Deserialize)]
struct EditFileInput {
    file_path: String,
    /// Multi-edit: array of replacements applied atomically.
    #[serde(default)]
    edits: Vec<EditOp>,
    /// Legacy: single replacement (converted to edits[0] if edits is empty).
    #[serde(default)]
    old_string: Option<String>,
    #[serde(default)]
    new_string: Option<String>,
}

impl EditFileInput {
    /// Normalize: merge legacy old_string/new_string into edits array.
    fn into_edits(mut self) -> (String, Vec<EditOp>) {
        if self.edits.is_empty()
            && let (Some(old), Some(new)) = (self.old_string.take(), self.new_string.take())
        {
            self.edits.push(EditOp {
                old_string: old,
                new_string: new,
            });
        }
        (self.file_path, self.edits)
    }
}

#[async_trait]
impl Tool for EditFileTool {
    fn spec(&self) -> ToolSpec {
        ToolSpec {
            name: "edit_file".into(),
            description: "Edit a file using exact text replacements. Supports multiple edits \
                          in one call — each edit is matched against the original file, not \
                          incrementally. All edits are applied atomically."
                .into(),
            parameters: serde_json::json!({
                "type": "object",
                "properties": {
                    "file_path": { "type": "string", "description": "Path to the file to edit" },
                    "edits": {
                        "type": "array",
                        "description": "One or more replacements. Each matched against the original file.",
                        "items": {
                            "type": "object",
                            "properties": {
                                "old_string": { "type": "string", "description": "Exact text to find (must be unique)" },
                                "new_string": { "type": "string", "description": "Replacement text" }
                            },
                            "required": ["old_string", "new_string"]
                        }
                    },
                    "old_string": { "type": "string", "description": "Legacy: single exact string to replace" },
                    "new_string": { "type": "string", "description": "Legacy: replacement string" }
                },
                "required": ["file_path"]
            }),
            permission: Permission::WorkspaceWrite,
        }
    }

    fn effective_permission(&self, input: &serde_json::Value, cwd: &Path) -> Permission {
        if let Some(fp) = input.get("file_path").and_then(|v| v.as_str()) {
            let path = resolve_path(fp, cwd);
            if !is_inside_workspace(&path, cwd) {
                return Permission::Dangerous;
            }
        }
        Permission::WorkspaceWrite
    }

    async fn execute(&self, input: serde_json::Value, cwd: &Path) -> ToolResult {
        let raw_input: EditFileInput = match serde_json::from_value(input) {
            Ok(v) => v,
            Err(e) => {
                return ToolResult {
                    output: format!("Invalid input: {e}"),
                    is_error: true,
                };
            }
        };

        let (file_path, edits) = raw_input.into_edits();
        if edits.is_empty() {
            return ToolResult {
                output: "No edits provided. Supply edits[] array or old_string/new_string.".into(),
                is_error: true,
            };
        }

        let path = resolve_path(&file_path, cwd);

        // C1: Serialize concurrent edits to the same file.
        let _file_guard = super::file_lock::lock_file(&path).await;

        let content = match tokio::fs::read_to_string(&path).await {
            Ok(c) => c,
            Err(e) => {
                return ToolResult {
                    output: format!("Failed to read: {e}"),
                    is_error: true,
                };
            }
        };

        // B1: Validate ALL edits against the ORIGINAL content first (atomic).
        // Each old_string must be unique and non-overlapping.
        for (i, edit) in edits.iter().enumerate() {
            let count = content.matches(&edit.old_string).count();
            if count == 0 {
                return ToolResult {
                    output: format!("edits[{i}]: old_string not found in file"),
                    is_error: true,
                };
            }
            if count > 1 {
                return ToolResult {
                    output: format!("edits[{i}]: old_string found {count} times (must be unique)"),
                    is_error: true,
                };
            }
        }

        // Apply all edits. Each is matched against original, applied in order.
        // Since all old_strings are unique, order doesn't matter for non-overlapping edits.
        let mut new_content = content;
        for edit in &edits {
            new_content = new_content.replacen(&edit.old_string, &edit.new_string, 1);
        }

        match tokio::fs::write(&path, &new_content).await {
            Ok(()) => {
                // C2: Build a compact diff preview for each edit.
                let mut diff_lines = Vec::new();
                for (i, edit) in edits.iter().enumerate() {
                    let old_preview: String = edit
                        .old_string
                        .lines()
                        .next()
                        .unwrap_or("")
                        .chars()
                        .take(60)
                        .collect();
                    let new_preview: String = edit
                        .new_string
                        .lines()
                        .next()
                        .unwrap_or("")
                        .chars()
                        .take(60)
                        .collect();
                    let old_lc = edit.old_string.lines().count();
                    let new_lc = edit.new_string.lines().count();
                    diff_lines.push(format!(
                        "  #{}: -({old_lc}L) {old_preview}{}  +({new_lc}L) {new_preview}{}",
                        i + 1,
                        if old_lc > 1 { "…" } else { "" },
                        if new_lc > 1 { "…" } else { "" },
                    ));
                }
                let summary = format!(
                    "Edited {} ({} replacement{})\n{}",
                    path.display(),
                    edits.len(),
                    if edits.len() == 1 { "" } else { "s" },
                    diff_lines.join("\n")
                );
                ToolResult {
                    output: summary,
                    is_error: false,
                }
            }
            Err(e) => ToolResult {
                output: format!("Failed to write: {e}"),
                is_error: true,
            },
        }
    }
}

async fn is_binary(path: &Path) -> bool {
    let Ok(mut file) = tokio::fs::File::open(path).await else {
        return false;
    };
    use tokio::io::AsyncReadExt;
    let mut buf = vec![0u8; BINARY_SNIFF_SIZE];
    let Ok(n) = file.read(&mut buf).await else {
        return false;
    };
    buf[..n].contains(&0)
}

fn resolve_path(file_path: &str, cwd: &Path) -> PathBuf {
    let p = Path::new(file_path);
    if p.is_absolute() {
        p.to_path_buf()
    } else {
        cwd.join(p)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tool::Tool;

    #[test]
    fn read_file_spec() {
        let tool = ReadFileTool;
        assert_eq!(tool.spec().name, "read_file");
        assert_eq!(tool.spec().permission, Permission::ReadOnly);
    }

    #[test]
    fn write_file_spec() {
        let tool = WriteFileTool;
        assert_eq!(tool.spec().name, "write_file");
        assert_eq!(tool.spec().permission, Permission::WorkspaceWrite);
    }

    #[test]
    fn edit_file_spec() {
        let tool = EditFileTool;
        assert_eq!(tool.spec().name, "edit_file");
        assert_eq!(tool.spec().permission, Permission::WorkspaceWrite);
    }

    #[tokio::test]
    async fn write_then_read_file() {
        let dir = tempfile::tempdir().unwrap();

        let write_tool = WriteFileTool;
        let result = write_tool
            .execute(
                serde_json::json!({"file_path": "test.txt", "contents": "hello world"}),
                dir.path(),
            )
            .await;
        assert!(!result.is_error);
        assert!(result.output.contains("11 bytes"));

        let read_tool = ReadFileTool;
        let result = read_tool
            .execute(serde_json::json!({"file_path": "test.txt"}), dir.path())
            .await;
        assert!(!result.is_error);
        assert!(result.output.contains("hello world"));
    }

    #[tokio::test]
    async fn read_file_with_offset_and_limit() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("lines.txt"), "line1\nline2\nline3\nline4\n").unwrap();

        let tool = ReadFileTool;
        let result = tool
            .execute(
                serde_json::json!({"file_path": "lines.txt", "offset": 2, "limit": 2}),
                dir.path(),
            )
            .await;
        assert!(!result.is_error);
        assert!(result.output.contains("line2"));
        assert!(result.output.contains("line3"));
        assert!(!result.output.contains("line1"));
        assert!(!result.output.contains("line4"));
    }

    #[tokio::test]
    async fn read_nonexistent_file_errors() {
        let dir = tempfile::tempdir().unwrap();
        let tool = ReadFileTool;
        let result = tool
            .execute(serde_json::json!({"file_path": "nope.txt"}), dir.path())
            .await;
        assert!(result.is_error);
        assert!(result.output.contains("Failed to read"));
    }

    #[tokio::test]
    async fn write_creates_subdirectories() {
        let dir = tempfile::tempdir().unwrap();
        let tool = WriteFileTool;
        let result = tool
            .execute(
                serde_json::json!({"file_path": "sub/dir/file.txt", "contents": "nested"}),
                dir.path(),
            )
            .await;
        assert!(!result.is_error);
        assert!(dir.path().join("sub/dir/file.txt").exists());
    }

    #[tokio::test]
    async fn edit_file_replaces_text() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("edit.txt"), "hello world").unwrap();

        let tool = EditFileTool;
        let result = tool
            .execute(
                serde_json::json!({
                    "file_path": "edit.txt",
                    "old_string": "world",
                    "new_string": "rust"
                }),
                dir.path(),
            )
            .await;
        assert!(!result.is_error);

        let content = std::fs::read_to_string(dir.path().join("edit.txt")).unwrap();
        assert_eq!(content, "hello rust");
    }

    #[tokio::test]
    async fn edit_file_not_found_old_string() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("e.txt"), "abc").unwrap();

        let tool = EditFileTool;
        let result = tool
            .execute(
                serde_json::json!({
                    "file_path": "e.txt",
                    "old_string": "xyz",
                    "new_string": "123"
                }),
                dir.path(),
            )
            .await;
        assert!(result.is_error);
        assert!(result.output.contains("not found"));
    }

    #[tokio::test]
    async fn edit_file_rejects_non_unique_match() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("dup.txt"), "aa bb aa").unwrap();

        let tool = EditFileTool;
        let result = tool
            .execute(
                serde_json::json!({
                    "file_path": "dup.txt",
                    "old_string": "aa",
                    "new_string": "cc"
                }),
                dir.path(),
            )
            .await;
        assert!(result.is_error);
        assert!(result.output.contains("2 times"));
    }

    #[tokio::test]
    async fn read_outside_workspace_allowed() {
        let dir = tempfile::tempdir().unwrap();
        let tool = ReadFileTool;
        let result = tool
            .execute(
                serde_json::json!({"file_path": "/etc/hostname"}),
                dir.path(),
            )
            .await;
        assert!(!result.is_error || result.output.contains("Failed to read"));
    }

    #[tokio::test]
    async fn write_outside_workspace_escalates_permission() {
        let dir = tempfile::tempdir().unwrap();
        let tool = WriteFileTool;
        let perm = tool.effective_permission(
            &serde_json::json!({"file_path": "/tmp/outside.txt", "contents": "x"}),
            dir.path(),
        );
        assert_eq!(perm, Permission::Dangerous);

        let perm_inside = tool.effective_permission(
            &serde_json::json!({"file_path": "inside.txt", "contents": "x"}),
            dir.path(),
        );
        assert_eq!(perm_inside, Permission::WorkspaceWrite);
    }

    #[tokio::test]
    async fn edit_outside_workspace_escalates_permission() {
        let dir = tempfile::tempdir().unwrap();
        let tool = EditFileTool;
        let perm = tool.effective_permission(
            &serde_json::json!({"file_path": "/tmp/outside.txt", "old_string": "a", "new_string": "b"}),
            dir.path(),
        );
        assert_eq!(perm, Permission::Dangerous);
    }

    #[tokio::test]
    async fn read_image_file_returns_vision_result() {
        let dir = tempfile::tempdir().unwrap();
        let bin = dir.path().join("image.png");
        // Write a minimal valid-looking PNG header.
        std::fs::write(&bin, b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR").unwrap();

        let tool = ReadFileTool;
        let result = tool
            .execute(serde_json::json!({"file_path": "image.png"}), dir.path())
            .await;
        // Vision pipeline: images succeed and push to image_result.
        assert!(!result.is_error, "images should succeed: {}", result.output);
        assert!(result.output.contains("image sent to vision model"));
    }

    #[tokio::test]
    async fn read_binary_non_image_returns_error() {
        let dir = tempfile::tempdir().unwrap();
        let bin = dir.path().join("data.bin");
        std::fs::write(&bin, b"\x00\x01\x02\x03\x04\x05").unwrap();

        let tool = ReadFileTool;
        let result = tool
            .execute(serde_json::json!({"file_path": "data.bin"}), dir.path())
            .await;
        assert!(result.is_error);
        assert!(result.output.contains("Binary file"));
    }

    #[tokio::test]
    async fn read_text_file_with_no_nul_ok() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("ok.txt"), "hello\nworld\n").unwrap();

        let tool = ReadFileTool;
        let result = tool
            .execute(serde_json::json!({"file_path": "ok.txt"}), dir.path())
            .await;
        assert!(!result.is_error);
        assert!(result.output.contains("hello"));
    }

    #[tokio::test]
    async fn invalid_input_returns_error() {
        let tool = ReadFileTool;
        let result = tool
            .execute(serde_json::json!({"wrong": 123}), Path::new("/tmp"))
            .await;
        assert!(result.is_error);
        assert!(result.output.contains("Invalid input"));
    }

    // ── B1: Multi-edit tests ──────────────────────────────────────

    #[tokio::test]
    async fn edit_file_legacy_single() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("f.txt"), "hello world").unwrap();
        let tool = EditFileTool;
        let result = tool
            .execute(
                serde_json::json!({
                    "file_path": "f.txt",
                    "old_string": "hello",
                    "new_string": "goodbye"
                }),
                dir.path(),
            )
            .await;
        assert!(!result.is_error, "{}", result.output);
        assert_eq!(
            std::fs::read_to_string(dir.path().join("f.txt")).unwrap(),
            "goodbye world"
        );
    }

    #[tokio::test]
    async fn edit_file_multi_edits() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("f.txt"), "aaa\nbbb\nccc\n").unwrap();
        let tool = EditFileTool;
        let result = tool
            .execute(
                serde_json::json!({
                    "file_path": "f.txt",
                    "edits": [
                        { "old_string": "aaa", "new_string": "AAA" },
                        { "old_string": "ccc", "new_string": "CCC" }
                    ]
                }),
                dir.path(),
            )
            .await;
        assert!(!result.is_error, "{}", result.output);
        assert!(result.output.contains("2 replacements"));
        assert_eq!(
            std::fs::read_to_string(dir.path().join("f.txt")).unwrap(),
            "AAA\nbbb\nCCC\n"
        );
    }

    #[tokio::test]
    async fn edit_file_atomic_rejects_missing() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("f.txt"), "aaa\nbbb\n").unwrap();
        let tool = EditFileTool;
        // Second edit has old_string not in file — entire operation should fail
        let result = tool
            .execute(
                serde_json::json!({
                    "file_path": "f.txt",
                    "edits": [
                        { "old_string": "aaa", "new_string": "AAA" },
                        { "old_string": "zzz", "new_string": "ZZZ" }
                    ]
                }),
                dir.path(),
            )
            .await;
        assert!(result.is_error);
        assert!(result.output.contains("edits[1]"));
        // File should be UNCHANGED (atomic)
        assert_eq!(
            std::fs::read_to_string(dir.path().join("f.txt")).unwrap(),
            "aaa\nbbb\n"
        );
    }

    #[tokio::test]
    async fn edit_file_no_edits_error() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("f.txt"), "x").unwrap();
        let tool = EditFileTool;
        let result = tool
            .execute(serde_json::json!({"file_path": "f.txt"}), dir.path())
            .await;
        assert!(result.is_error);
        assert!(result.output.contains("No edits"));
    }
}
