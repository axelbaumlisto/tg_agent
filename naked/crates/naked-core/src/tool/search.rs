use std::path::Path;

use async_trait::async_trait;
use serde::Deserialize;

use crate::types::{Permission, ToolResult, ToolSpec};

use super::Tool;

const MAX_GLOB_RESULTS: usize = 200;

// -- GlobSearchTool ----------------------------------------------------------

pub struct GlobSearchTool;

#[derive(Deserialize)]
struct GlobInput {
    pattern: String,
    #[serde(default)]
    path: Option<String>,
}

#[async_trait]
impl Tool for GlobSearchTool {
    fn spec(&self) -> ToolSpec {
        ToolSpec {
            name: "glob_search".into(),
            description: "Find files matching a glob pattern recursively.".into(),
            parameters: serde_json::json!({
                "type": "object",
                "properties": {
                    "pattern": { "type": "string", "description": "Glob pattern (e.g. **/*.rs)" },
                    "path": { "type": "string", "description": "Base directory (default: workspace root)" }
                },
                "required": ["pattern"]
            }),
            permission: Permission::ReadOnly,
        }
    }

    async fn execute(&self, input: serde_json::Value, cwd: &Path) -> ToolResult {
        let input: GlobInput = match serde_json::from_value(input) {
            Ok(v) => v,
            Err(e) => {
                return ToolResult {
                    output: format!("Invalid input: {e}"),
                    is_error: true,
                };
            }
        };

        let base = input
            .path
            .as_deref()
            .map(|p| {
                let pb = Path::new(p);
                if pb.is_absolute() {
                    pb.to_path_buf()
                } else {
                    cwd.join(pb)
                }
            })
            .unwrap_or_else(|| cwd.to_path_buf());

        let full_pattern = if input.pattern.starts_with("**/") || input.pattern.starts_with('/') {
            format!("{}/{}", base.display(), input.pattern)
        } else {
            format!("{}/**/{}", base.display(), input.pattern)
        };

        match glob::glob(&full_pattern) {
            Ok(paths) => {
                let mut results: Vec<String> = paths
                    .filter_map(|p| p.ok())
                    .take(MAX_GLOB_RESULTS + 1)
                    .map(|p| p.strip_prefix(cwd).unwrap_or(&p).display().to_string())
                    .collect();
                let truncated = results.len() > MAX_GLOB_RESULTS;
                if truncated {
                    results.truncate(MAX_GLOB_RESULTS);
                }
                results.sort();
                if results.is_empty() {
                    ToolResult {
                        output: "No files found".into(),
                        is_error: false,
                    }
                } else {
                    let mut output = results.join("\n");
                    if truncated {
                        output.push_str(&format!(
                            "\n\n(truncated — showing first {MAX_GLOB_RESULTS} results)"
                        ));
                    }
                    ToolResult {
                        output,
                        is_error: false,
                    }
                }
            }
            Err(e) => ToolResult {
                output: format!("Glob error: {e}"),
                is_error: true,
            },
        }
    }
}

// -- GrepSearchTool ----------------------------------------------------------

pub struct GrepSearchTool;

#[derive(Deserialize)]
struct GrepInput {
    pattern: String,
    #[serde(default)]
    path: Option<String>,
    #[serde(default)]
    include: Option<String>,
}

#[async_trait]
impl Tool for GrepSearchTool {
    fn spec(&self) -> ToolSpec {
        ToolSpec {
            name: "grep_search".into(),
            description: "Search file contents using ripgrep (rg). Returns matching lines with file paths and line numbers.".into(),
            parameters: serde_json::json!({
                "type": "object",
                "properties": {
                    "pattern": { "type": "string", "description": "Regex pattern to search for" },
                    "path": { "type": "string", "description": "Directory or file to search in" },
                    "include": { "type": "string", "description": "Glob to filter files (e.g. *.rs)" }
                },
                "required": ["pattern"]
            }),
            permission: Permission::ReadOnly,
        }
    }

    async fn execute(&self, input: serde_json::Value, cwd: &Path) -> ToolResult {
        let input: GrepInput = match serde_json::from_value(input) {
            Ok(v) => v,
            Err(e) => {
                return ToolResult {
                    output: format!("Invalid input: {e}"),
                    is_error: true,
                };
            }
        };

        let mut cmd = tokio::process::Command::new("rg");
        cmd.arg("--line-number")
            .arg("--no-heading")
            .arg("--color=never")
            .arg("--max-count=100")
            .arg(&input.pattern);

        if let Some(ref inc) = input.include {
            cmd.arg("--glob").arg(inc);
        }

        let search_path = input
            .path
            .as_deref()
            .map(|p| {
                let pb = Path::new(p);
                if pb.is_absolute() {
                    pb.to_path_buf()
                } else {
                    cwd.join(pb)
                }
            })
            .unwrap_or_else(|| cwd.to_path_buf());

        cmd.arg(search_path);
        cmd.current_dir(cwd);

        match cmd.output().await {
            Ok(output) => {
                const MAX_GREP_OUTPUT: usize = 16_384;
                let stdout = String::from_utf8_lossy(&output.stdout);
                if stdout.is_empty() {
                    ToolResult {
                        output: "No matches found".into(),
                        is_error: false,
                    }
                } else if stdout.len() <= MAX_GREP_OUTPUT {
                    ToolResult {
                        output: stdout.to_string(),
                        is_error: false,
                    }
                } else {
                    let mut end = MAX_GREP_OUTPUT;
                    while end > 0 && !stdout.is_char_boundary(end) {
                        end -= 1;
                    }
                    if let Some(nl) = stdout[..end].rfind('\n') {
                        end = nl;
                    }
                    let cut = stdout.len() - end;
                    ToolResult {
                        output: format!(
                            "{}\n\n[output truncated — {cut} bytes cut, narrow your search]",
                            &stdout[..end]
                        ),
                        is_error: false,
                    }
                }
            }
            Err(e) => ToolResult {
                output: format!("rg failed: {e}"),
                is_error: true,
            },
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tool::Tool;

    #[test]
    fn glob_search_spec() {
        let tool = GlobSearchTool;
        let spec = tool.spec();
        assert_eq!(spec.name, "glob_search");
        assert_eq!(spec.permission, crate::types::Permission::ReadOnly);
    }

    #[test]
    fn grep_search_spec() {
        let tool = GrepSearchTool;
        let spec = tool.spec();
        assert_eq!(spec.name, "grep_search");
        assert_eq!(spec.permission, crate::types::Permission::ReadOnly);
    }

    #[tokio::test]
    async fn glob_finds_files() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("a.rs"), "").unwrap();
        std::fs::write(dir.path().join("b.txt"), "").unwrap();

        let tool = GlobSearchTool;
        let result = tool
            .execute(serde_json::json!({"pattern": "*.rs"}), dir.path())
            .await;
        assert!(!result.is_error);
        assert!(result.output.contains("a.rs"));
        assert!(!result.output.contains("b.txt"));
    }

    #[tokio::test]
    async fn glob_no_matches_returns_no_files() {
        let dir = tempfile::tempdir().unwrap();
        let tool = GlobSearchTool;
        let result = tool
            .execute(serde_json::json!({"pattern": "*.xyz"}), dir.path())
            .await;
        assert!(!result.is_error);
        assert!(result.output.contains("No files found"));
    }

    #[tokio::test]
    async fn grep_finds_pattern() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("code.rs"), "fn main() {}\nfn helper() {}").unwrap();

        let tool = GrepSearchTool;
        let result = tool
            .execute(
                serde_json::json!({"pattern": "fn main", "path": dir.path().to_str().unwrap()}),
                dir.path(),
            )
            .await;
        assert!(!result.is_error);
        assert!(result.output.contains("fn main"));
    }

    #[tokio::test]
    async fn grep_no_matches() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("empty.txt"), "nothing here").unwrap();

        let tool = GrepSearchTool;
        let result = tool
            .execute(
                serde_json::json!({"pattern": "nonexistent_string_xyz"}),
                dir.path(),
            )
            .await;
        assert!(!result.is_error);
        assert!(result.output.contains("No matches"));
    }

    #[tokio::test]
    async fn glob_invalid_input() {
        let tool = GlobSearchTool;
        let result = tool
            .execute(serde_json::json!({"wrong": 1}), Path::new("/tmp"))
            .await;
        assert!(result.is_error);
    }
}
