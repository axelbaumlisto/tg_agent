//! `revert_turn` — model-callable workspace rollback.
//!
//! Mirrors `/restore N` (slash command) but speaks JSON. The model
//! calls this when the user says "undo the last edit" and the
//! offset gets parsed from natural language. Conversation history
//! is NOT modified — only working-tree files are restored from the
//! side-git snapshot repo.
//!
//! Approval: `Dangerous` because this MUTATES the workspace. The
//! existing approval flow (per-tool gate / YOLO) decides whether
//! to surface it to the user.

use std::path::Path;

use serde_json::Value;

use crate::snapshot::SnapshotRepo;
use crate::types::{Permission, ToolResult, ToolSpec};

/// Default offset: revert the most-recent turn.
const DEFAULT_OFFSET: u64 = 1;
/// Hard cap so the model can't ask for arbitrary history.
const MAX_OFFSET: u64 = 50;

pub struct RevertTurnTool;

#[async_trait::async_trait]
impl crate::tool::Tool for RevertTurnTool {
    fn spec(&self) -> ToolSpec {
        ToolSpec {
            name: "revert_turn".into(),
            description: format!(
                "Roll back workspace files to the snapshot taken before a recent turn. \
                 Use when the user explicitly asks to undo, revert, or roll back. \
                 `turn_offset` is 1-based: 1 reverts the most recent turn, 2 the previous \
                 one, max {MAX_OFFSET}. Conversation history is NOT modified — only \
                 working-tree files are restored from the side-git snapshot repo."
            ),
            parameters: serde_json::json!({
                "type": "object",
                "properties": {
                    "turn_offset": {
                        "type": "integer",
                        "minimum": 1,
                        "maximum": MAX_OFFSET,
                        "description": "How many turns back to revert (default 1)."
                    }
                },
                "additionalProperties": false
            }),
            permission: Permission::Dangerous,
        }
    }

    async fn execute(&self, input: Value, cwd: &Path) -> ToolResult {
        let offset = input
            .get("turn_offset")
            .and_then(Value::as_u64)
            .unwrap_or(DEFAULT_OFFSET);
        if offset == 0 || offset > MAX_OFFSET {
            return ToolResult {
                output: format!("turn_offset must be 1..={MAX_OFFSET}; got {offset}"),
                is_error: true,
            };
        }

        let cwd = cwd.to_path_buf();
        let res = tokio::task::spawn_blocking(move || -> Result<String, String> {
            let repo = SnapshotRepo::open_or_init(&cwd)
                .map_err(|e| format!("snapshot repo init failed: {e}"))?;
            let list = repo
                .list((MAX_OFFSET as usize).saturating_mul(2) + 16)
                .map_err(|e| format!("list snapshots: {e}"))?;
            let pre_turns: Vec<_> = list
                .into_iter()
                .filter(|s| s.label.starts_with("pre-turn:"))
                .collect();
            let target = pre_turns.get((offset - 1) as usize).ok_or_else(|| {
                format!(
                    "Only {} pre-turn snapshot(s) exist; turn_offset={offset} out of range.",
                    pre_turns.len(),
                )
            })?;
            repo.restore(&target.id)
                .map_err(|e| format!("restore failed: {e}"))?;
            Ok(format!(
                "Reverted to snapshot {} (label='{}', ts={})",
                target.id.as_str(),
                target.label,
                target.ts_unix,
            ))
        })
        .await;

        match res {
            Ok(Ok(msg)) => ToolResult {
                output: msg,
                is_error: false,
            },
            Ok(Err(msg)) => ToolResult {
                output: msg,
                is_error: true,
            },
            Err(e) => ToolResult {
                output: format!("revert_turn task panicked: {e}"),
                is_error: true,
            },
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tool::Tool;

    #[tokio::test]
    async fn invalid_offset_zero_errors() {
        let t = RevertTurnTool;
        let r = t
            .execute(serde_json::json!({"turn_offset": 0}), Path::new("/tmp"))
            .await;
        assert!(r.is_error);
        assert!(r.output.contains("turn_offset must be"));
    }

    #[tokio::test]
    async fn invalid_offset_too_high_errors() {
        let t = RevertTurnTool;
        let r = t
            .execute(serde_json::json!({"turn_offset": 999}), Path::new("/tmp"))
            .await;
        assert!(r.is_error);
    }

    #[tokio::test]
    async fn empty_history_errors_gracefully() {
        // Default offset on a workspace that has no snapshots → error,
        // not a panic.
        let dir = tempfile::TempDir::new().unwrap();
        let t = RevertTurnTool;
        // Must not panic; either result is acceptable as long as
        // we get back a ToolResult.
        let _ = t.execute(serde_json::json!({}), dir.path()).await;
    }
}
