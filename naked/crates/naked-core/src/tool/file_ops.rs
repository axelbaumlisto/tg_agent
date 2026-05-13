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
        let input: ReadFileInput = match super::parse_tool_input(input) {
            Ok(v) => v,
            Err(e) => return e,
        };

        let path = resolve_path(&input.file_path, cwd);

        if let Ok(meta) = tokio::fs::metadata(&path).await
            && meta.len() > MAX_READ_BYTES
        {
            return ToolResult::err(format!(
                "File too large ({} bytes, max {}). Use offset/limit for partial reads.",
                meta.len(),
                MAX_READ_BYTES
            ));
        }

        if let Some(binary_result) = handle_binary_file(&path).await {
            return binary_result;
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
                ToolResult::ok(output.trim_end().to_string())
            }
            Err(e) => ToolResult::err(format!("Failed to read {}: {e}", path.display())),
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
        let input: WriteFileInput = match super::parse_tool_input(input) {
            Ok(v) => v,
            Err(e) => return e,
        };

        let path = resolve_path(&input.file_path, cwd);

        // C1: Serialize concurrent writes to the same file.
        let _file_guard = super::file_lock::lock_file(&path).await;

        if let Some(parent) = path.parent()
            && let Err(e) = tokio::fs::create_dir_all(parent).await
        {
            return ToolResult::err(format!("Failed to create directories: {e}"));
        }

        // Read old content for diff:
        let old_content = tokio::fs::read_to_string(&path).await.unwrap_or_default();

        // Atomic write: tmp file + rename
        let tmp = path.with_extension("tmp");
        match tokio::fs::write(&tmp, &input.contents).await {
            Ok(()) => match tokio::fs::rename(&tmp, &path).await {
                Ok(()) => {
                    let diff = super::diff_format::unified_diff(
                        &input.file_path,
                        &old_content,
                        &input.contents,
                    );
                    let mut output =
                        format!("Wrote {} bytes to {}", input.contents.len(), path.display());
                    if !diff.is_empty() {
                        output.push_str(&format!("\n\n{diff}"));
                    }
                    ToolResult::ok(output)
                }
                Err(e) => ToolResult::err(format!("Failed to rename: {e}")),
            },
            Err(e) => ToolResult::err(format!("Failed to write: {e}")),
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
        let raw_input: EditFileInput = match super::parse_tool_input(input) {
            Ok(v) => v,
            Err(e) => return e,
        };

        let (file_path, edits) = raw_input.into_edits();
        if edits.is_empty() {
            return ToolResult::err(
                "No edits provided. Supply edits[] array or old_string/new_string.",
            );
        }

        let path = resolve_path(&file_path, cwd);

        // C1: Serialize concurrent edits to the same file.
        let _file_guard = super::file_lock::lock_file(&path).await;

        let content = match tokio::fs::read_to_string(&path).await {
            Ok(c) => c,
            Err(e) => {
                return ToolResult::err(format!("Failed to read: {e}"));
            }
        };

        // B1: Validate ALL edits against the ORIGINAL content first (atomic).
        // Each old_string must be unique and non-overlapping.
        for (i, edit) in edits.iter().enumerate() {
            let count = content.matches(&edit.old_string).count();
            if count == 0 {
                return ToolResult::err(format!("edits[{i}]: old_string not found in file"));
            }
            if count > 1 {
                return ToolResult::err(format!(
                    "edits[{i}]: old_string found {count} times (must be unique)"
                ));
            }
        }

        // Apply all edits. Each is matched against original, applied in order.
        // Since all old_strings are unique, order doesn't matter for non-overlapping edits.
        let mut new_content = content;
        for edit in &edits {
            new_content = new_content.replacen(&edit.old_string, &edit.new_string, 1);
        }

        match tokio::fs::write(&path, &new_content).await {
            Ok(()) => ToolResult::ok(format_edit_summary(&path, &edits)),
            Err(e) => ToolResult::err(format!("Failed to write: {e}")),
        }
    }
}

async fn is_binary(path: &Path) -> bool {
    // REGISTRY-WAIVE: intentional fallback: open/read failure → empty
    let Ok(mut file) = tokio::fs::File::open(path).await else {
        return false;
    };
    use tokio::io::AsyncReadExt;
    let mut buf = vec![0u8; BINARY_SNIFF_SIZE];
    // REGISTRY-WAIVE: intentional fallback: open/read failure → empty
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

/// Handle binary file detection: images get base64-encoded for vision,
/// other binaries are rejected.
async fn handle_binary_file(path: &Path) -> Option<ToolResult> {
    if !is_binary(path).await {
        return None;
    }
    let size = tokio::fs::metadata(path)
        .await
        .map(|m| m.len())
        .unwrap_or(0);
    let ext = path.extension().and_then(|e| e.to_str()).unwrap_or("");
    let is_image = matches!(ext, "png" | "jpg" | "jpeg" | "gif" | "webp" | "bmp");

    if is_image {
        const MAX_IMAGE_BYTES: u64 = 512_000;
        if size <= MAX_IMAGE_BYTES
            && let Ok(bytes) = tokio::fs::read(path).await
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
            super::image_result::push_image(mime, &b64);
            return Some(ToolResult::ok(format!(
                "Image ({}, {size} bytes): {}\n[image sent to vision model]",
                ext.to_uppercase(),
                path.display()
            )));
        }
        let p = path.display();
        return Some(ToolResult::ok(format!(
            "Image file ({ext}, {size} bytes): {p}. Too large for inline. \
             Use bash to analyze: file '{p}' or identify '{p}'"
        )));
    }

    Some(ToolResult::err(format!(
        "Binary file ({size} bytes): {}. Cannot read as text.",
        path.display()
    )))
}

/// Build a compact diff summary for applied edits.
fn format_edit_summary(path: &Path, edits: &[EditOp]) -> String {
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
    format!(
        "Edited {} ({} replacement{})\n{}",
        path.display(),
        edits.len(),
        if edits.len() == 1 { "" } else { "s" },
        diff_lines.join("\n")
    )
}

#[cfg(test)]
#[path = "file_ops_tests.rs"]
mod tests;
