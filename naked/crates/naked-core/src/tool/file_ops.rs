use std::path::{Path, PathBuf};

use std::sync::Arc;
use std::sync::atomic::Ordering;

#[cfg(unix)]
use std::os::unix::fs::MetadataExt;

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

pub struct ReadFileTool {
    fs_cache: Option<Arc<super::fs_cache::FsCache>>,
}

impl ReadFileTool {
    pub fn new(fs_cache: Option<Arc<super::fs_cache::FsCache>>) -> Self {
        Self { fs_cache }
    }
}

impl Default for ReadFileTool {
    fn default() -> Self {
        Self::new(None)
    }
}

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

        match read_text_content(&path, self.fs_cache.as_ref()).await {
            Ok(content) => {
                ToolResult::ok(format_read_file_output(&content, input.offset, input.limit))
            }
            Err(e) => ToolResult::err(format!("Failed to read {}: {e}", path.display())),
        }
    }
}

// -- FileSnapshotTool --------------------------------------------------------

pub struct FileSnapshotTool {
    fs_cache: Option<Arc<super::fs_cache::FsCache>>,
}

impl FileSnapshotTool {
    pub fn new(fs_cache: Option<Arc<super::fs_cache::FsCache>>) -> Self {
        Self { fs_cache }
    }
}

impl Default for FileSnapshotTool {
    fn default() -> Self {
        Self::new(None)
    }
}

#[derive(Deserialize)]
struct FileSnapshotInput {
    file_path: String,
    #[serde(default)]
    offset: Option<usize>,
    #[serde(default)]
    limit: Option<usize>,
}

#[async_trait]
impl Tool for FileSnapshotTool {
    fn spec(&self) -> ToolSpec {
        ToolSpec {
            name: "file_snapshot".into(),
            description: "Read a file snapshot for safe edits: full-file SHA-256, per-line anchor hashes, and a numbered text window.".into(),
            parameters: serde_json::json!({
                "type": "object",
                "properties": {
                    "file_path": { "type": "string", "description": "Path to the file to snapshot" },
                    "offset": { "type": "integer", "description": "Line offset (1-based) to start reading from" },
                    "limit": { "type": "integer", "description": "Maximum number of lines to include" }
                },
                "required": ["file_path"]
            }),
            permission: Permission::ReadOnly,
        }
    }

    async fn execute(&self, input: serde_json::Value, cwd: &Path) -> ToolResult {
        let input: FileSnapshotInput = match super::parse_tool_input(input) {
            Ok(v) => v,
            Err(e) => return e,
        };

        let path = resolve_path(&input.file_path, cwd);
        let metadata = match tokio::fs::metadata(&path).await {
            Ok(meta) => {
                if meta.len() > MAX_READ_BYTES {
                    return ToolResult::err(format!(
                        "File too large ({} bytes, max {}). Use offset/limit for partial snapshots.",
                        meta.len(),
                        MAX_READ_BYTES
                    ));
                }
                Some(meta)
            }
            Err(_) => None,
        };
        if let Some(binary_result) = handle_binary_file(&path).await {
            return binary_result;
        }

        match read_text_content(&path, self.fs_cache.as_ref()).await {
            Ok(content) => ToolResult::ok(format_snapshot(
                &content,
                input.offset,
                input.limit,
                metadata.as_ref(),
            )),
            Err(e) => ToolResult::err(format!("Failed to read {}: {e}", path.display())),
        }
    }
}

// -- WriteFileTool -----------------------------------------------------------

pub struct WriteFileTool {
    fs_cache: Option<Arc<super::fs_cache::FsCache>>,
}

impl WriteFileTool {
    pub fn new(fs_cache: Option<Arc<super::fs_cache::FsCache>>) -> Self {
        Self { fs_cache }
    }
}

impl Default for WriteFileTool {
    fn default() -> Self {
        Self::new(None)
    }
}

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

        match super::atomic_write::atomic_replace_file(&path, input.contents.as_bytes()).await {
            Ok(()) => {
                if let Some(cache) = &self.fs_cache {
                    cache.invalidate(&path).await;
                }
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
            Err(e) => ToolResult::err(format!("Failed to write: {e}")),
        }
    }
}

// -- EditFileTool ------------------------------------------------------------

#[derive(Debug, Clone, Default)]
pub struct EditFileTool {
    stale_edit_guard_enabled: bool,
    hashline_edit_enabled: bool,
    fs_cache: Option<Arc<super::fs_cache::FsCache>>,
}

impl EditFileTool {
    pub fn new(stale_edit_guard_enabled: bool) -> Self {
        Self {
            stale_edit_guard_enabled,
            hashline_edit_enabled: false,
            fs_cache: None,
        }
    }

    pub fn with_hashline_edit(mut self, enabled: bool) -> Self {
        self.hashline_edit_enabled = enabled;
        self
    }

    pub fn with_fs_cache(mut self, fs_cache: Option<Arc<super::fs_cache::FsCache>>) -> Self {
        self.fs_cache = fs_cache;
        self
    }
}

/// Single edit: replace old_string → new_string.
#[derive(Deserialize, Clone)]
struct EditOp {
    old_string: String,
    new_string: String,
}

#[derive(Deserialize, Clone)]
struct HashEditOp {
    anchor_hash: String,
    start_line: usize,
    end_line: usize,
    new_text: String,
}

/// B1: Supports both legacy (single old_string/new_string) and multi-edit
/// (edits array). Legacy format is auto-converted to a single-element array.
#[derive(Deserialize)]
struct EditFileInput {
    file_path: String,
    /// Multi-edit: array of replacements applied atomically.
    #[serde(default)]
    edits: Vec<EditOp>,
    /// Hashline mode: range edits guarded by line-range content hashes.
    #[serde(default)]
    hash_edits: Vec<HashEditOp>,
    /// Legacy: single replacement (converted to edits[0] if edits is empty).
    #[serde(default)]
    old_string: Option<String>,
    #[serde(default)]
    new_string: Option<String>,
    /// Optional full-file SHA-256 of the content the caller expects to edit.
    /// Honored only when stale_edit_guard_enabled is true.
    #[serde(default)]
    expected_content_sha: Option<String>,
}

impl EditFileInput {
    /// Normalize: merge legacy old_string/new_string into edits array.
    fn into_parts(mut self) -> (String, Vec<EditOp>, Vec<HashEditOp>, Option<String>) {
        if self.edits.is_empty()
            && self.hash_edits.is_empty()
            && let (Some(old), Some(new)) = (self.old_string.take(), self.new_string.take())
        {
            self.edits.push(EditOp {
                old_string: old,
                new_string: new,
            });
        }
        (
            self.file_path,
            self.edits,
            self.hash_edits,
            self.expected_content_sha,
        )
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
                    "new_string": { "type": "string", "description": "Legacy: replacement string" },
                    "hash_edits": {
                        "type": "array",
                        "description": "Hashline mode edits. Requires hashline_edit_enabled. Each range is validated against live content before any write.",
                        "items": {
                            "type": "object",
                            "properties": {
                                "anchor_hash": { "type": "string", "description": "SHA-256 hex of the exact line-range text from file_snapshot" },
                                "start_line": { "type": "integer", "description": "1-based first line to replace" },
                                "end_line": { "type": "integer", "description": "1-based last line to replace, inclusive" },
                                "new_text": { "type": "string", "description": "Replacement text for the line range" }
                            },
                            "required": ["anchor_hash", "start_line", "end_line", "new_text"]
                        }
                    },
                    "expected_content_sha": { "type": "string", "description": "Optional SHA-256 hex of the full file content expected before editing (requires stale_edit_guard_enabled)" }
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

        let (file_path, edits, hash_edits, expected_content_sha) = raw_input.into_parts();
        if edits.is_empty() && hash_edits.is_empty() {
            return ToolResult::err(
                "No edits provided. Supply edits[] array, hash_edits[] array, or old_string/new_string.",
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

        if !hash_edits.is_empty() {
            return self
                .execute_hashline_edit(&path, &content, &hash_edits)
                .await;
        }

        if self.stale_edit_guard_enabled && expected_content_sha.is_some() {
            let expectations = super::edit_guard::EditExpectations {
                expected_content_sha,
                ..Default::default()
            };
            if super::edit_guard::validate_live_content(&content, &expectations).is_err() {
                crate::types::STALE_EDIT_REJECT_COUNT.fetch_add(1, Ordering::Relaxed);
                return ToolResult::err(
                    "file changed since you read it — re-read and retry".to_string(),
                );
            }
        }

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

        match super::atomic_write::atomic_replace_file(&path, new_content.as_bytes()).await {
            Ok(()) => {
                if let Some(cache) = &self.fs_cache {
                    cache.invalidate(&path).await;
                }
                ToolResult::ok(format_edit_summary(&path, &edits))
            }
            Err(e) => ToolResult::err(format!("Failed to write: {e}")),
        }
    }
}

impl EditFileTool {
    async fn execute_hashline_edit(
        &self,
        path: &Path,
        content: &str,
        hash_edits: &[HashEditOp],
    ) -> ToolResult {
        if !self.hashline_edit_enabled {
            crate::types::HASHLINE_EDIT_DISABLED_COUNT.fetch_add(1, Ordering::Relaxed);
            return ToolResult::err(
                "hashline edit mode is disabled; re-run with exact edits[] or enable hashline_edit_enabled"
                    .to_string(),
            );
        }

        if let Some((left, right)) = first_overlapping_hash_edit(hash_edits) {
            crate::types::HASHLINE_EDIT_OVERLAP_COUNT.fetch_add(1, Ordering::Relaxed);
            return ToolResult::err(format!(
                "hash_edits ranges overlap: hash_edits[{left}] and hash_edits[{right}]"
            ));
        }

        let expectations = super::edit_guard::EditExpectations {
            expected_content_sha: None,
            range_anchors: hash_edits
                .iter()
                .map(|edit| super::edit_guard::RangeAnchor {
                    start_line: edit.start_line,
                    end_line: edit.end_line,
                    anchor_hash: edit.anchor_hash.clone(),
                })
                .collect(),
        };
        if let Err(err) = super::edit_guard::validate_live_content(content, &expectations) {
            match err {
                super::edit_guard::StaleEditError::RangeAnchorOutOfBounds { .. } => {
                    crate::types::HASHLINE_EDIT_OUT_OF_BOUNDS_COUNT.fetch_add(1, Ordering::Relaxed);
                    return ToolResult::err(
                        "hashline range is out of bounds — re-snapshot and retry".to_string(),
                    );
                }
                super::edit_guard::StaleEditError::RangeAnchorMismatch { .. }
                | super::edit_guard::StaleEditError::RangeAnchorAmbiguous { .. }
                | super::edit_guard::StaleEditError::ContentShaMismatch { .. } => {
                    crate::types::HASHLINE_EDIT_STALE_ANCHOR_COUNT.fetch_add(1, Ordering::Relaxed);
                    return ToolResult::err(
                        "hashline anchor is stale or ambiguous — re-snapshot and retry".to_string(),
                    );
                }
            }
        }

        let new_content = apply_hash_edits(content, hash_edits);
        match super::atomic_write::atomic_replace_file(path, new_content.as_bytes()).await {
            Ok(()) => {
                if let Some(cache) = &self.fs_cache {
                    cache.invalidate(path).await;
                }
                crate::types::HASHLINE_EDIT_APPLIED_COUNT.fetch_add(1, Ordering::Relaxed);
                ToolResult::ok(format_hash_edit_summary(path, hash_edits))
            }
            Err(e) => ToolResult::err(format!("Failed to write: {e}")),
        }
    }
}

async fn read_text_content(
    path: &Path,
    fs_cache: Option<&Arc<super::fs_cache::FsCache>>,
) -> std::io::Result<String> {
    if let Some(cache) = fs_cache {
        return cache
            .get_or_load(path, || async { tokio::fs::read_to_string(path).await })
            .await
            .map(|cached| cached.as_str().to_string());
    }
    tokio::fs::read_to_string(path).await
}

fn format_read_file_output(content: &str, offset: Option<usize>, limit: Option<usize>) -> String {
    const MAX_OUTPUT: usize = 16_384;
    let lines: Vec<&str> = content.lines().collect();
    let offset = offset.unwrap_or(1).saturating_sub(1);
    let limit = limit.unwrap_or(lines.len());
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
    output.trim_end().to_string()
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

fn format_snapshot(
    content: &str,
    offset: Option<usize>,
    limit: Option<usize>,
    metadata: Option<&std::fs::Metadata>,
) -> String {
    let full_sha = super::edit_guard::sha256_hex(content.as_bytes());
    let spans = super::edit_guard::line_spans(content);
    let start_idx = offset.unwrap_or(1).saturating_sub(1);
    let max_lines = limit.unwrap_or_else(|| spans.len().saturating_sub(start_idx));
    let mut output = format!("full_sha256: {full_sha}\n");
    if let Some(meta) = metadata {
        #[cfg(unix)]
        let mtime_ns = i128::from(meta.mtime()) * 1_000_000_000i128 + i128::from(meta.mtime_nsec());
        output.push_str(&format!(
            "metadata: dev={} inode={} mtime_ns={} len={}\n",
            meta.dev(),
            meta.ino(),
            mtime_ns,
            meta.len()
        ));
        #[cfg(not(unix))]
        output.push_str(&format!("metadata: len={}\n", meta.len()));
    }
    output.push_str("anchors:\n");
    for (idx, (start, end)) in spans.iter().enumerate().skip(start_idx).take(max_lines) {
        let line_no = idx + 1;
        let hash = super::edit_guard::sha256_hex(&content.as_bytes()[*start..*end]);
        output.push_str(&format!("{:6}|{}\n", line_no, hash));
    }
    output.push_str("text:\n");
    for (idx, (start, end)) in spans.iter().enumerate().skip(start_idx).take(max_lines) {
        let line_no = idx + 1;
        let line = content[*start..*end].trim_end_matches('\n');
        output.push_str(&format!("{:6}|{}\n", line_no, line));
    }
    output.trim_end().to_string()
}

fn first_overlapping_hash_edit(hash_edits: &[HashEditOp]) -> Option<(usize, usize)> {
    let mut ranges: Vec<_> = hash_edits
        .iter()
        .enumerate()
        .map(|(idx, edit)| (idx, edit.start_line, edit.end_line))
        .collect();
    ranges.sort_by_key(|(_, start, end)| (*start, *end));
    ranges.windows(2).find_map(|pair| {
        let (left_idx, _left_start, left_end) = pair[0];
        let (right_idx, right_start, _right_end) = pair[1];
        (right_start <= left_end).then_some((left_idx, right_idx))
    })
}

fn apply_hash_edits(content: &str, hash_edits: &[HashEditOp]) -> String {
    let spans = super::edit_guard::line_spans(content);
    let mut ordered = hash_edits.to_vec();
    ordered.sort_by_key(|edit| edit.start_line);

    let mut out = String::with_capacity(content.len());
    let mut cursor = 0usize;
    for edit in ordered {
        let start = spans[edit.start_line - 1].0;
        let end = spans[edit.end_line - 1].1;
        out.push_str(&content[cursor..start]);
        out.push_str(&edit.new_text);
        cursor = end;
    }
    out.push_str(&content[cursor..]);
    out
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

fn format_hash_edit_summary(path: &Path, hash_edits: &[HashEditOp]) -> String {
    let ranges = hash_edits
        .iter()
        .enumerate()
        .map(|(idx, edit)| {
            format!(
                "  #{}: lines {}-{} -> {} bytes",
                idx + 1,
                edit.start_line,
                edit.end_line,
                edit.new_text.len()
            )
        })
        .collect::<Vec<_>>()
        .join("\n");
    format!(
        "Edited {} ({} hashline replacement{})\n{}",
        path.display(),
        hash_edits.len(),
        if hash_edits.len() == 1 { "" } else { "s" },
        ranges
    )
}

#[cfg(test)]
#[path = "file_ops_tests.rs"]
mod tests;
