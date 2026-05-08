//! `file_search` tool — fuzzy file finder, readonly, auto-approved.
//!
//! Walks the workspace (respecting .gitignore) and fuzzy-matches by name.
//! Faster and safer than bash find.

use crate::types::{Permission, ToolResult, ToolSpec};
use std::path::Path;

pub struct FileSearchTool;

#[async_trait::async_trait]
impl crate::tool::Tool for FileSearchTool {
    fn spec(&self) -> ToolSpec {
        ToolSpec {
            name: "file_search".into(),
            description: "Fuzzy search for files by name in the workspace. \
                          Respects .gitignore. Returns top matches with paths. \
                          Use for finding files when you don't know exact location."
                .into(),
            parameters: serde_json::json!({
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Fuzzy search query (e.g. 'main.rs', 'config', 'test_auth')"
                    },
                    "max_results": {
                        "type": "integer",
                        "description": "Max results (default 20)"
                    }
                },
                "required": ["query"]
            }),
            permission: Permission::ReadOnly,
        }
    }

    async fn execute(&self, input: serde_json::Value, cwd: &Path) -> ToolResult {
        let query = match input.get("query").and_then(|v| v.as_str()) {
            Some(q) if !q.is_empty() => q,
            _ => {
                return ToolResult::err("query is required");
            }
        };
        let max = input
            .get("max_results")
            .and_then(|v| v.as_u64())
            .unwrap_or(20)
            .min(50) as usize;
        let query_lower = query.to_ascii_lowercase();

        // Walk workspace, collect matching files:
        let mut matches: Vec<(String, f64)> = Vec::new();

        let walker = ignore::WalkBuilder::new(cwd)
            .hidden(true)
            .git_ignore(true)
            .max_depth(Some(8))
            .build();

        for entry in walker.flatten() {
            if !entry.file_type().map(|t| t.is_file()).unwrap_or(false) {
                continue;
            }
            let path = entry.path();
            let rel = path.strip_prefix(cwd).unwrap_or(path);
            let rel_str = rel.to_string_lossy();
            let rel_lower = rel_str.to_ascii_lowercase();

            // Score: exact substring > fuzzy match:
            let score = if rel_lower.contains(&query_lower) {
                1.0 - (rel_str.len() as f64 / 200.0) // shorter paths score higher
            } else {
                fuzzy_score(&query_lower, &rel_lower)
            };

            if score > 0.0 {
                matches.push((rel_str.to_string(), score));
            }
        }

        matches.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        matches.truncate(max);

        if matches.is_empty() {
            return ToolResult::ok(format!("No files matching '{query}'"));
        }

        let mut output = format!("{} files matching '{query}':\n", matches.len());
        for (path, _score) in &matches {
            output.push_str(&format!("  {path}\n"));
        }

        ToolResult::ok(output)
    }
}

/// Simple fuzzy scoring: check if all query chars appear in order.
fn fuzzy_score(query: &str, target: &str) -> f64 {
    let mut qi = query.chars().peekable();
    let mut matched = 0;
    for ch in target.chars() {
        if qi.peek() == Some(&ch) {
            qi.next();
            matched += 1;
        }
    }
    if qi.peek().is_some() {
        0.0 // not all query chars matched
    } else {
        matched as f64 / target.len().max(1) as f64 * 0.5 // fuzzy matches score lower
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tool::Tool;

    #[test]
    fn fuzzy_score_exact_substring() {
        assert!(fuzzy_score("main", "src/main.rs") > 0.0);
    }

    #[test]
    fn fuzzy_score_no_match() {
        assert_eq!(fuzzy_score("xyz", "abc"), 0.0);
    }

    #[tokio::test]
    async fn search_finds_cargo_toml() {
        let cwd = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .unwrap()
            .parent()
            .unwrap()
            .to_path_buf();
        let tool = FileSearchTool;
        let r = tool
            .execute(serde_json::json!({"query": "Cargo.toml"}), &cwd)
            .await;
        assert!(!r.is_error);
        assert!(r.output.contains("Cargo.toml"));
    }

    #[tokio::test]
    async fn search_empty_query_errors() {
        let tool = FileSearchTool;
        let r = tool
            .execute(serde_json::json!({"query": ""}), Path::new("/tmp"))
            .await;
        assert!(r.is_error);
    }
}
