//! `review` — static code review: TODO/FIXME, unwrap(), long functions/files.

use std::path::Path;

use crate::types::{Permission, ToolResult, ToolSpec};

#[derive(Debug)]
struct Issue {
    severity: &'static str,
    line: usize,
    message: String,
}

fn analyze(content: &str, filename: &str) -> Vec<Issue> {
    let mut issues = Vec::new();
    let is_test = filename.contains("test") || filename.contains("tests/");
    let lines: Vec<&str> = content.lines().collect();

    // File-level:
    if lines.len() > 500 {
        issues.push(Issue {
            severity: "info",
            line: 0,
            message: format!("File has {} lines — consider splitting", lines.len()),
        });
    }

    let mut fn_start: Option<(usize, &str)> = None;
    let mut brace_depth: i32 = 0;

    for (i, line) in lines.iter().enumerate() {
        let ln = i + 1;
        let trimmed = line.trim();

        // TODO/FIXME/HACK:
        for marker in ["TODO", "FIXME", "HACK", "XXX"] {
            if trimmed.contains(marker) {
                issues.push(Issue {
                    severity: "info",
                    line: ln,
                    message: format!("{marker} comment"),
                });
            }
        }

        // unwrap() outside tests:
        if !is_test
            && (trimmed.contains(".unwrap()") || trimmed.contains(".expect("))
            && !trimmed.starts_with("//")
        {
            issues.push(Issue {
                severity: "warning",
                line: ln,
                message: "unwrap()/expect() — consider ? or handle error".into(),
            });
        }

        // Track function length:
        if trimmed.starts_with("fn ")
            || trimmed.starts_with("pub fn ")
            || trimmed.starts_with("async fn ")
            || trimmed.starts_with("pub async fn ")
        {
            fn_start = Some((ln, trimmed));
            brace_depth = 0;
        }
        if fn_start.is_some() {
            brace_depth += trimmed.chars().filter(|&c| c == '{').count() as i32;
            brace_depth -= trimmed.chars().filter(|&c| c == '}').count() as i32;
            if brace_depth <= 0 {
                if let Some((start, _)) = fn_start {
                    let fn_len = ln - start;
                    if fn_len > 50 {
                        issues.push(Issue {
                            severity: "warning",
                            line: start,
                            message: format!("Function is {fn_len} lines — consider splitting"),
                        });
                    }
                }
                fn_start = None;
            }
        }
    }
    issues
}

pub struct ReviewTool;

#[async_trait::async_trait]
impl crate::tool::Tool for ReviewTool {
    fn spec(&self) -> ToolSpec {
        ToolSpec {
            name: "review".into(),
            description:
                "Static code review: find TODOs, unwrap()s, long functions. Pass a file path."
                    .into(),
            parameters: serde_json::json!({
                "type": "object",
                "properties": { "path": { "type": "string", "description": "File to review" } },
                "required": ["path"]
            }),
            permission: Permission::ReadOnly,
        }
    }

    async fn execute(&self, input: serde_json::Value, cwd: &Path) -> ToolResult {
        let file = match input.get("path").and_then(|v| v.as_str()) {
            Some(p) => p,
            None => {
                return ToolResult::err("path required");
            }
        };
        let path = if Path::new(file).is_absolute() {
            file.into()
        } else {
            cwd.join(file)
        };
        let content = match tokio::fs::read_to_string(&path).await {
            Ok(c) => c,
            Err(e) => {
                return ToolResult::err(format!("Read error: {e}"));
            }
        };
        let issues = analyze(&content, file);
        if issues.is_empty() {
            return ToolResult::ok("✓ No issues found");
        }
        let output = issues
            .iter()
            .map(|i| {
                if i.line > 0 {
                    format!("[{}] L{}: {}", i.severity, i.line, i.message)
                } else {
                    format!("[{}] {}", i.severity, i.message)
                }
            })
            .collect::<Vec<_>>()
            .join("\n");
        ToolResult::ok(format!("{} issues:\n{output}", issues.len()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn finds_todos() {
        let issues = analyze("// TODO: fix this\nlet x = 1;", "src/lib.rs");
        assert!(issues.iter().any(|i| i.message.contains("TODO")));
    }

    #[test]
    fn finds_unwrap() {
        let issues = analyze("let x = foo().unwrap();", "src/lib.rs");
        assert!(issues.iter().any(|i| i.message.contains("unwrap")));
    }

    #[test]
    fn no_unwrap_in_tests() {
        let issues = analyze("let x = foo().unwrap();", "tests/unit.rs");
        assert!(!issues.iter().any(|i| i.message.contains("unwrap")));
    }

    #[test]
    fn clean_file() {
        let issues = analyze("fn main() {\n    println!(\"ok\");\n}\n", "src/main.rs");
        assert!(issues.is_empty());
    }
}
