//! Git history tools — readonly, auto-approved.
//!
//! `git_log`, `git_diff`, `git_blame` — give the model access to
//! version history without requiring bash approval.

use crate::types::{Permission, ToolResult, ToolSpec};
use std::path::Path;
use std::process::Stdio;

pub struct GitLogTool;
pub struct GitDiffTool;

#[async_trait::async_trait]
impl crate::tool::Tool for GitLogTool {
    fn spec(&self) -> ToolSpec {
        ToolSpec {
            name: "git_log".into(),
            description: "Show recent git commits. Returns log with hash, author, date, message. \
                          Readonly, auto-approved."
                .into(),
            parameters: serde_json::json!({
                "type": "object",
                "properties": {
                    "max_count": { "type": "integer", "description": "Max commits (default 10)" },
                    "path": { "type": "string", "description": "Filter by file path" }
                }
            }),
            permission: Permission::ReadOnly,
        }
    }

    async fn execute(&self, input: serde_json::Value, cwd: &Path) -> ToolResult {
        let max = input
            .get("max_count")
            .and_then(|v| v.as_u64())
            .unwrap_or(10)
            .min(50);
        let mut args = vec![
            "log".into(),
            format!("--max-count={max}"),
            "--oneline".into(),
            "--decorate".into(),
            "--no-color".into(),
        ];
        if let Some(path) = input.get("path").and_then(|v| v.as_str()) {
            args.push("--".into());
            args.push(path.into());
        }
        git_cmd(cwd, &args).await
    }
}

#[async_trait::async_trait]
impl crate::tool::Tool for GitDiffTool {
    fn spec(&self) -> ToolSpec {
        ToolSpec {
            name: "git_diff".into(),
            description: "Show git diff (unstaged changes, or between refs). Readonly.".into(),
            parameters: serde_json::json!({
                "type": "object",
                "properties": {
                    "ref1": { "type": "string", "description": "Base ref (default: HEAD)" },
                    "ref2": { "type": "string", "description": "Compare ref (default: working tree)" },
                    "path": { "type": "string", "description": "Filter by file path" }
                }
            }),
            permission: Permission::ReadOnly,
        }
    }

    async fn execute(&self, input: serde_json::Value, cwd: &Path) -> ToolResult {
        let mut args = vec!["diff".into(), "--no-color".into(), "--stat".into()];
        if let Some(r1) = input.get("ref1").and_then(|v| v.as_str()) {
            args.push(r1.into());
        }
        if let Some(r2) = input.get("ref2").and_then(|v| v.as_str()) {
            args.push(r2.into());
        }
        if let Some(path) = input.get("path").and_then(|v| v.as_str()) {
            args.push("--".into());
            args.push(path.into());
        }
        git_cmd(cwd, &args).await
    }
}

async fn git_cmd(cwd: &Path, args: &[String]) -> ToolResult {
    match tokio::process::Command::new("git")
        .args(args)
        .current_dir(cwd)
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .output()
        .await
    {
        Ok(out) => {
            let text = String::from_utf8_lossy(&out.stdout);
            let stderr = String::from_utf8_lossy(&out.stderr);
            let mut output = if text.is_empty() {
                stderr.to_string()
            } else {
                text.to_string()
            };
            // Truncate:
            if output.len() > 8000 {
                let end = output.floor_char_boundary(8000);
                output = format!("{}…\n[truncated]", &output[..end]);
            }
            if out.status.success() {
                ToolResult::ok(output)
            } else {
                ToolResult::err(output)
            }
        }
        Err(e) => ToolResult::err(format!("git error: {e}")),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tool::Tool;

    #[tokio::test]
    #[ignore = "requires git repository context"]
    async fn git_log_works() {
        let cwd = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .unwrap()
            .parent()
            .unwrap()
            .to_path_buf();
        let tool = GitLogTool;
        let r = tool
            .execute(serde_json::json!({"max_count": 3}), &cwd)
            .await;
        assert!(!r.is_error, "{}", r.output);
        assert!(!r.output.is_empty());
    }

    #[tokio::test]
    #[ignore = "requires git repository context"]
    async fn git_diff_works() {
        let cwd = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .unwrap()
            .parent()
            .unwrap()
            .to_path_buf();
        let tool = GitDiffTool;
        let r = tool.execute(serde_json::json!({}), &cwd).await;
        // May be empty if clean, but shouldn't error:
        assert!(!r.is_error, "{}", r.output);
    }
}
