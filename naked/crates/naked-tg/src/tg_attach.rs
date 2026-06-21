//! `telegram_attach` tool — lets the agent send files to the Telegram chat.
//!
//! The tool queues file paths during a turn. After the turn completes,
//! `stream_response` drains the queue and delivers files via Telegram API.
//!
//! Design: the tool itself does no I/O to Telegram — it only validates
//! paths and stages them. Delivery is the caller's responsibility.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::{Arc, LazyLock};

use serde_json::json;
use tokio::sync::Mutex;

static WORKSPACE_RUN_BINDINGS: LazyLock<Mutex<HashMap<PathBuf, String>>> =
    LazyLock::new(|| Mutex::new(HashMap::new()));

pub async fn bind_workspace_run(workspace: PathBuf, run_id: String) {
    WORKSPACE_RUN_BINDINGS
        .lock()
        .await
        .insert(workspace, run_id);
}

pub async fn unbind_workspace_run(workspace: &Path, run_id: &str) {
    let mut guard = WORKSPACE_RUN_BINDINGS.lock().await;
    if guard.get(workspace).is_some_and(|id| id == run_id) {
        guard.remove(workspace);
    }
}

async fn run_id_for_cwd(cwd: &Path) -> Option<String> {
    WORKSPACE_RUN_BINDINGS.lock().await.get(cwd).cloned()
}

use naked_core::tool::Tool;
use naked_core::types::{Permission, ToolResult, ToolSpec};

/// Shared queue of file paths staged for Telegram delivery.
pub type AttachmentQueue = Arc<Mutex<Vec<StagedAttachment>>>;

/// A file staged for delivery after the turn completes.
#[derive(Debug, Clone)]
pub struct StagedAttachment {
    pub path: PathBuf,
    pub file_name: String,
    pub run_id: Option<String>,
}

/// Create a new shared attachment queue.
pub fn new_queue() -> AttachmentQueue {
    Arc::new(Mutex::new(Vec::new()))
}

/// The `telegram_attach` tool. Accepts file paths from the agent and
/// stages them for Telegram delivery.
pub struct TelegramAttachTool {
    queue: AttachmentQueue,
    run_id: Option<String>,
    max_attachments: usize,
    max_file_bytes: u64,
}

impl TelegramAttachTool {
    pub fn new(queue: AttachmentQueue) -> Self {
        Self {
            queue,
            run_id: None,
            max_attachments: 10,
            max_file_bytes: 50 * 1024 * 1024, // 50 MB (Telegram bot limit)
        }
    }

    pub fn new_for_run(queue: AttachmentQueue, run_id: impl Into<String>) -> Self {
        Self {
            queue,
            run_id: Some(run_id.into()),
            max_attachments: 10,
            max_file_bytes: 50 * 1024 * 1024,
        }
    }
}

#[async_trait::async_trait]
impl Tool for TelegramAttachTool {
    fn spec(&self) -> ToolSpec {
        ToolSpec {
            name: "telegram_attach".into(),
            description: "Send one or more files to the Telegram chat. \
                Files are delivered after the current response completes. \
                Accepts absolute paths to existing files."
                .into(),
            parameters: json!({
                "type": "object",
                "required": ["paths"],
                "properties": {
                    "paths": {
                        "type": "array",
                        "items": { "type": "string" },
                        "description": "Absolute file paths to attach"
                    }
                }
            }),
            permission: Permission::ReadOnly,
        }
    }

    async fn execute(&self, input: serde_json::Value, cwd: &std::path::Path) -> ToolResult {
        let paths: Vec<String> = match input.get("paths").and_then(|v| {
            v.as_array().map(|arr| {
                arr.iter()
                    .filter_map(|p| p.as_str().map(String::from))
                    .collect::<Vec<_>>()
            })
        }) {
            Some(p) if !p.is_empty() => p,
            _ => {
                return ToolResult {
                    output: "Error: 'paths' must be a non-empty array of strings.".into(),
                    is_error: true,
                };
            }
        };

        let run_id = match self.run_id.clone() {
            Some(id) => Some(id),
            None => run_id_for_cwd(cwd).await,
        };
        let mut queue = self.queue.lock().await;
        let mut added = Vec::new();
        let mut errors = Vec::new();

        for path_str in &paths {
            let path = PathBuf::from(path_str);

            // Check attachment count limit
            if queue.len() >= self.max_attachments {
                errors.push(format!(
                    "{path_str}: attachment limit reached (max {})",
                    self.max_attachments
                ));
                continue;
            }

            // Check file exists and is a regular file
            match tokio::fs::metadata(&path).await {
                Ok(meta) => {
                    if !meta.is_file() {
                        errors.push(format!("{path_str}: not a regular file"));
                        continue;
                    }
                    if meta.len() > self.max_file_bytes {
                        errors.push(format!(
                            "{path_str}: too large ({} MB, max {} MB)",
                            meta.len() / (1024 * 1024),
                            self.max_file_bytes / (1024 * 1024)
                        ));
                        continue;
                    }
                }
                Err(e) => {
                    errors.push(format!("{path_str}: {e}"));
                    continue;
                }
            }

            let file_name = path
                .file_name()
                .map(|n| n.to_string_lossy().into_owned())
                .unwrap_or_else(|| "attachment".into());

            queue.push(StagedAttachment {
                path,
                file_name: file_name.clone(),
                run_id: run_id.clone(),
            });
            added.push(file_name);
        }

        let mut output = String::new();
        if !added.is_empty() {
            output.push_str(&format!(
                "Queued {} file(s) for Telegram delivery: {}",
                added.len(),
                added.join(", ")
            ));
        }
        if !errors.is_empty() {
            if !output.is_empty() {
                output.push('\n');
            }
            output.push_str(&format!("Errors: {}", errors.join("; ")));
        }

        ToolResult {
            output,
            is_error: added.is_empty(),
        }
    }
}

pub async fn drain_for_run(queue: &AttachmentQueue, run_id: &str) -> Vec<StagedAttachment> {
    let mut guard = queue.lock().await;
    let mut selected = Vec::new();
    let mut retained = Vec::new();
    for att in guard.drain(..) {
        if att.run_id.as_deref() == Some(run_id) || att.run_id.is_none() {
            // Legacy/unscoped attachments are delivered to the finishing run for
            // backwards compatibility. Run-scoped tools/tests prove B is retained.
            selected.push(att);
        } else {
            retained.push(att);
        }
    }
    *guard = retained;
    selected
}

/// Guess whether a file should be sent as a photo (image) or document.
pub fn is_image_path(path: &std::path::Path) -> bool {
    let ext = path
        .extension()
        .map(|e| e.to_string_lossy().to_lowercase())
        .unwrap_or_default();
    matches!(ext.as_str(), "jpg" | "jpeg" | "png" | "gif" | "webp")
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_tool() -> (TelegramAttachTool, AttachmentQueue) {
        let queue = new_queue();
        let tool = TelegramAttachTool::new(queue.clone());
        (tool, queue)
    }

    #[test]
    fn spec_name_and_params() {
        let (tool, _) = make_tool();
        let spec = tool.spec();
        assert_eq!(spec.name, "telegram_attach");
        assert!(
            spec.parameters["required"]
                .as_array()
                .unwrap()
                .iter()
                .any(|v| v == "paths")
        );
    }

    #[tokio::test]
    async fn empty_paths_returns_error() {
        let (tool, _) = make_tool();
        let result = tool
            .execute(json!({"paths": []}), std::path::Path::new("/tmp"))
            .await;
        assert!(result.is_error);
    }

    #[tokio::test]
    async fn missing_paths_returns_error() {
        let (tool, _) = make_tool();
        let result = tool.execute(json!({}), std::path::Path::new("/tmp")).await;
        assert!(result.is_error);
    }

    #[tokio::test]
    async fn nonexistent_file_returns_error() {
        let (tool, queue) = make_tool();
        let result = tool
            .execute(
                json!({"paths": ["/tmp/nonexistent_tg_attach_test_xyz.bin"]}),
                std::path::Path::new("/tmp"),
            )
            .await;
        assert!(result.is_error);
        assert!(queue.lock().await.is_empty());
    }

    #[tokio::test]
    async fn valid_file_queued() {
        let tmp = tempfile::NamedTempFile::new().unwrap();
        std::fs::write(tmp.path(), "hello").unwrap();

        let (tool, queue) = make_tool();
        let result = tool
            .execute(
                json!({"paths": [tmp.path().to_str().unwrap()]}),
                std::path::Path::new("/tmp"),
            )
            .await;

        assert!(!result.is_error, "result: {}", result.output);
        assert!(result.output.contains("Queued 1 file"));

        let staged = queue.lock().await;
        assert_eq!(staged.len(), 1);
        assert_eq!(staged[0].path, tmp.path());
    }

    #[tokio::test]
    async fn multiple_files_queued() {
        let tmp1 = tempfile::NamedTempFile::new().unwrap();
        let tmp2 = tempfile::NamedTempFile::new().unwrap();
        std::fs::write(tmp1.path(), "a").unwrap();
        std::fs::write(tmp2.path(), "b").unwrap();

        let (tool, queue) = make_tool();
        let result = tool
            .execute(
                json!({
                    "paths": [
                        tmp1.path().to_str().unwrap(),
                        tmp2.path().to_str().unwrap()
                    ]
                }),
                std::path::Path::new("/tmp"),
            )
            .await;

        assert!(!result.is_error);
        assert_eq!(queue.lock().await.len(), 2);
    }

    #[tokio::test]
    async fn attachments_finish_a_does_not_deliver_b() {
        let tmp_a = tempfile::NamedTempFile::new().unwrap();
        let tmp_b = tempfile::NamedTempFile::new().unwrap();
        std::fs::write(tmp_a.path(), "a").unwrap();
        std::fs::write(tmp_b.path(), "b").unwrap();
        let queue = new_queue();
        let tool_a = TelegramAttachTool::new_for_run(queue.clone(), "run-a");
        let tool_b = TelegramAttachTool::new_for_run(queue.clone(), "run-b");
        tool_a
            .execute(
                json!({"paths": [tmp_a.path().to_str().unwrap()]}),
                std::path::Path::new("/tmp"),
            )
            .await;
        tool_b
            .execute(
                json!({"paths": [tmp_b.path().to_str().unwrap()]}),
                std::path::Path::new("/tmp"),
            )
            .await;

        let delivered_a = drain_for_run(&queue, "run-a").await;
        assert_eq!(delivered_a.len(), 1);
        assert_eq!(delivered_a[0].run_id.as_deref(), Some("run-a"));
        let remaining = queue.lock().await;
        assert_eq!(
            remaining.len(),
            1,
            "seeded-fail: global drain would remove B here"
        );
        assert_eq!(remaining[0].run_id.as_deref(), Some("run-b"));
    }

    #[tokio::test]
    async fn directory_rejected() {
        let tmp = tempfile::tempdir().unwrap();
        let (tool, queue) = make_tool();
        let result = tool
            .execute(
                json!({"paths": [tmp.path().to_str().unwrap()]}),
                std::path::Path::new("/tmp"),
            )
            .await;
        assert!(result.is_error);
        assert!(queue.lock().await.is_empty());
    }

    #[test]
    fn is_image_detection() {
        assert!(is_image_path(&PathBuf::from("photo.jpg")));
        assert!(is_image_path(&PathBuf::from("img.PNG")));
        assert!(!is_image_path(&PathBuf::from("doc.pdf")));
        assert!(!is_image_path(&PathBuf::from("script.sh")));
    }
}
