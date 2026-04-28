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

#[derive(Deserialize)]
struct EditFileInput {
    file_path: String,
    old_string: String,
    new_string: String,
}

#[async_trait]
impl Tool for EditFileTool {
    fn spec(&self) -> ToolSpec {
        ToolSpec {
            name: "edit_file".into(),
            description: "Replace an exact string occurrence in a file with a new string.".into(),
            parameters: serde_json::json!({
                "type": "object",
                "properties": {
                    "file_path": { "type": "string", "description": "Path to the file to edit" },
                    "old_string": { "type": "string", "description": "The exact string to find and replace" },
                    "new_string": { "type": "string", "description": "The replacement string" }
                },
                "required": ["file_path", "old_string", "new_string"]
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
        let input: EditFileInput = match serde_json::from_value(input) {
            Ok(v) => v,
            Err(e) => {
                return ToolResult {
                    output: format!("Invalid input: {e}"),
                    is_error: true,
                };
            }
        };

        let path = resolve_path(&input.file_path, cwd);

        let content = match tokio::fs::read_to_string(&path).await {
            Ok(c) => c,
            Err(e) => {
                return ToolResult {
                    output: format!("Failed to read: {e}"),
                    is_error: true,
                };
            }
        };

        let count = content.matches(&input.old_string).count();
        if count == 0 {
            return ToolResult {
                output: "old_string not found in file".into(),
                is_error: true,
            };
        }
        if count > 1 {
            return ToolResult {
                output: format!("old_string found {count} times (must be unique)"),
                is_error: true,
            };
        }

        let new_content = content.replacen(&input.old_string, &input.new_string, 1);
        match tokio::fs::write(&path, &new_content).await {
            Ok(()) => ToolResult {
                output: format!("Edited {}", path.display()),
                is_error: false,
            },
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
    async fn read_binary_file_returns_error() {
        let dir = tempfile::tempdir().unwrap();
        let bin = dir.path().join("image.png");
        std::fs::write(&bin, b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR").unwrap();

        let tool = ReadFileTool;
        let result = tool
            .execute(serde_json::json!({"file_path": "image.png"}), dir.path())
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
}
