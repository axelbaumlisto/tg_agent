//! `run_tests` tool — auto-approved test execution.
//!
//! Auto-approved (ReadOnly) so the model can run tests in a TDD loop
//! without waiting for user confirmation each time.

use crate::types::{Permission, ToolResult, ToolSpec};
use std::path::Path;

pub struct RunTestsTool;

#[async_trait::async_trait]
impl crate::tool::Tool for RunTestsTool {
    fn spec(&self) -> ToolSpec {
        ToolSpec {
            name: "run_tests".into(),
            description: "Run tests in the workspace. Auto-approved for fast TDD loops. \
                          Detects project type: cargo test (Rust), pytest (Python), \
                          npm test (Node). Returns pass/fail + output."
                .into(),
            parameters: serde_json::json!({
                "type": "object",
                "properties": {
                    "filter": {
                        "type": "string",
                        "description": "Optional test name filter (e.g. 'test_auth' or 'my_module::tests')"
                    }
                }
            }),
            permission: Permission::ReadOnly, // auto-approved
        }
    }

    async fn execute(&self, input: serde_json::Value, cwd: &Path) -> ToolResult {
        let filter = input.get("filter").and_then(|v| v.as_str()).unwrap_or("");

        let (cmd, args) = detect_test_command(cwd, filter);

        let output = match tokio::time::timeout(
            std::time::Duration::from_secs(120),
            tokio::process::Command::new(&cmd)
                .args(&args)
                .current_dir(cwd)
                .stdout(std::process::Stdio::piped())
                .stderr(std::process::Stdio::piped())
                .output(),
        )
        .await
        {
            Ok(Ok(out)) => out,
            Ok(Err(e)) => {
                return ToolResult::err(format!("Failed to run {cmd}: {e}"));
            }
            Err(_) => {
                return ToolResult::err(format!("{cmd} timed out after 120s"));
            }
        };

        let stdout = String::from_utf8_lossy(&output.stdout);
        let stderr = String::from_utf8_lossy(&output.stderr);
        let combined = if stderr.is_empty() {
            stdout.to_string()
        } else {
            format!("{stdout}\n{stderr}")
        };

        // Truncate if huge:
        let truncated = if combined.len() > 8000 {
            let end = combined.floor_char_boundary(8000);
            format!(
                "{}…\n[truncated, {} bytes total]",
                &combined[..end],
                combined.len()
            )
        } else {
            combined
        };

        if output.status.success() {
            ToolResult::ok(format!("✅ Tests passed\n\n{truncated}"))
        } else {
            ToolResult::err(format!(
                "❌ Tests failed (exit {})\n\n{truncated}",
                output.status.code().unwrap_or(-1)
            ))
        }
    }
}

fn detect_test_command(cwd: &Path, filter: &str) -> (String, Vec<String>) {
    if cwd.join("Cargo.toml").exists() {
        let mut args = vec!["test".into(), "--workspace".into()];
        if !filter.is_empty() {
            args.push("--".into());
            args.push(filter.into());
        }
        ("cargo".into(), args)
    } else if cwd.join("pytest.ini").exists()
        || cwd.join("pyproject.toml").exists()
        || cwd.join("setup.py").exists()
    {
        let mut args = vec!["-m".into(), "pytest".into(), "-x".into()];
        if !filter.is_empty() {
            args.push("-k".into());
            args.push(filter.into());
        }
        ("python3".into(), args)
    } else if cwd.join("package.json").exists() {
        ("npm".into(), vec!["test".into()])
    } else {
        ("echo".into(), vec!["No test framework detected".into()])
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tool::Tool;

    #[test]
    fn detect_rust() {
        let tmp = tempfile::tempdir().unwrap();
        std::fs::write(tmp.path().join("Cargo.toml"), "[package]").unwrap();
        let (cmd, args) = detect_test_command(tmp.path(), "");
        assert_eq!(cmd, "cargo");
        assert!(args.contains(&"test".into()));
    }

    #[test]
    fn detect_python() {
        let tmp = tempfile::tempdir().unwrap();
        std::fs::write(tmp.path().join("pyproject.toml"), "").unwrap();
        let (cmd, _) = detect_test_command(tmp.path(), "");
        assert_eq!(cmd, "python3");
    }

    #[test]
    fn detect_node() {
        let tmp = tempfile::tempdir().unwrap();
        std::fs::write(tmp.path().join("package.json"), "{}").unwrap();
        let (cmd, _) = detect_test_command(tmp.path(), "");
        assert_eq!(cmd, "npm");
    }

    #[test]
    fn detect_with_filter() {
        let tmp = tempfile::tempdir().unwrap();
        std::fs::write(tmp.path().join("Cargo.toml"), "[package]").unwrap();
        let (_, args) = detect_test_command(tmp.path(), "my_test");
        assert!(args.contains(&"my_test".into()));
    }

    #[tokio::test]
    async fn run_tests_in_workspace() {
        // Run actual cargo test in our workspace:
        let tool = RunTestsTool;
        let cwd = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .unwrap()
            .parent()
            .unwrap()
            .to_path_buf();
        let result = tool
            .execute(
                serde_json::json!({"filter": "diff_format::tests::identical"}),
                &cwd,
            )
            .await;
        assert!(
            !result.is_error,
            "cargo test should pass: {}",
            result.output
        );
        assert!(result.output.contains("passed"));
    }
}
