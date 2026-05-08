//! `remember` tool — model writes directly to persistent memory.
//!
//! Auto-approved, side-effect only on MEMORY.md. The model can call
//! this when it notices a durable preference or correction worth keeping.

use std::path::{Path, PathBuf};

use crate::types::{Permission, ToolResult, ToolSpec};

pub struct RememberTool {
    memory_dir: PathBuf,
}

impl RememberTool {
    pub fn new(memory_dir: &Path) -> Self {
        Self {
            memory_dir: memory_dir.to_path_buf(),
        }
    }

    fn memory_path(&self) -> PathBuf {
        self.memory_dir.join("MEMORY.md")
    }
}

#[async_trait::async_trait]
impl crate::tool::Tool for RememberTool {
    fn spec(&self) -> ToolSpec {
        ToolSpec {
            name: "remember".into(),
            description: "Save a note to persistent memory. Use for preferences, corrections, \
                          or facts that should persist across sessions. The note is appended \
                          to MEMORY.md under the specified category."
                .into(),
            parameters: serde_json::json!({
                "type": "object",
                "properties": {
                    "category": {
                        "type": "string",
                        "enum": ["Preferences", "Corrections", "Facts"],
                        "description": "Category: Preferences (user likes/dislikes), Corrections (fix wrong behavior), Facts (persistent knowledge)"
                    },
                    "note": {
                        "type": "string",
                        "description": "The note to remember (one line, concise)"
                    }
                },
                "required": ["category", "note"]
            }),
            permission: Permission::ReadOnly, // auto-approved — only writes MEMORY.md
        }
    }

    async fn execute(&self, input: serde_json::Value, _cwd: &Path) -> ToolResult {
        let category = input
            .get("category")
            .and_then(|v| v.as_str())
            .unwrap_or("Preferences");
        let note = match input.get("note").and_then(|v| v.as_str()) {
            Some(n) if !n.trim().is_empty() => n.trim(),
            _ => {
                return ToolResult::err("Error: 'note' is required and must not be empty");
            }
        };

        let path = self.memory_path();

        // Read existing content:
        let content = tokio::fs::read_to_string(&path).await.unwrap_or_default();

        // Find the right section and append:
        let section_header = format!("## {category}");
        let id = &uuid::Uuid::new_v4().to_string()[..8];
        let date = chrono::Utc::now().format("%Y-%m-%d");
        let entry = format!("- {note} <!-- id:{id} created:{date} source:remember_tool -->");

        let new_content = if content.contains(&section_header) {
            // Append after section header:
            content.replacen(&section_header, &format!("{section_header}\n{entry}"), 1)
        } else {
            // Add new section at end:
            format!("{content}\n{section_header}\n{entry}\n")
        };

        if let Some(parent) = path.parent() {
            let _ = tokio::fs::create_dir_all(parent).await;
        }
        match tokio::fs::write(&path, &new_content).await {
            Ok(()) => ToolResult::ok(format!("Remembered under {category}: {note}")),
            Err(e) => ToolResult::err(format!("Failed to write MEMORY.md: {e}")),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tool::Tool;

    #[tokio::test]
    async fn remember_creates_entry() {
        let tmp = tempfile::tempdir().unwrap();
        let tool = RememberTool::new(tmp.path());
        let result = tool
            .execute(
                serde_json::json!({"category": "Preferences", "note": "use dark theme"}),
                Path::new("/tmp"),
            )
            .await;
        assert!(!result.is_error);
        assert!(result.output.contains("Remembered"));

        let content = std::fs::read_to_string(tmp.path().join("MEMORY.md")).unwrap();
        assert!(content.contains("use dark theme"));
        assert!(content.contains("remember_tool"));
    }

    #[tokio::test]
    async fn remember_appends_to_existing_section() {
        let tmp = tempfile::tempdir().unwrap();
        let path = tmp.path().join("MEMORY.md");
        std::fs::write(&path, "# Memory\n\n## Preferences\n- existing\n").unwrap();

        let tool = RememberTool::new(tmp.path());
        tool.execute(
            serde_json::json!({"category": "Preferences", "note": "new pref"}),
            Path::new("/tmp"),
        )
        .await;

        let content = std::fs::read_to_string(&path).unwrap();
        assert!(content.contains("existing"));
        assert!(content.contains("new pref"));
    }

    #[tokio::test]
    async fn remember_creates_new_section() {
        let tmp = tempfile::tempdir().unwrap();
        let path = tmp.path().join("MEMORY.md");
        std::fs::write(&path, "# Memory\n").unwrap();

        let tool = RememberTool::new(tmp.path());
        tool.execute(
            serde_json::json!({"category": "Facts", "note": "server is in Germany"}),
            Path::new("/tmp"),
        )
        .await;

        let content = std::fs::read_to_string(&path).unwrap();
        assert!(content.contains("## Facts"));
        assert!(content.contains("server is in Germany"));
    }

    #[tokio::test]
    async fn remember_rejects_empty_note() {
        let tmp = tempfile::tempdir().unwrap();
        let tool = RememberTool::new(tmp.path());
        let result = tool
            .execute(
                serde_json::json!({"category": "Preferences", "note": ""}),
                Path::new("/tmp"),
            )
            .await;
        assert!(result.is_error);
    }
}
