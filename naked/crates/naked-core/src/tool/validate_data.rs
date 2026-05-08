//! `validate_data` — JSON/TOML validation tool.

use crate::types::{Permission, ToolResult, ToolSpec};
use std::path::Path;

pub struct ValidateDataTool;

fn detect_format<'a>(hint: &'a str, content: &str) -> &'a str {
    if hint != "auto" {
        return hint;
    }
    let t = content.trim_start();
    if t.starts_with('{') {
        return "json";
    }
    // `[` can be JSON array or TOML section — try JSON parse:
    if t.starts_with('[') && serde_json::from_str::<serde_json::Value>(content).is_ok() {
        return "json";
    }
    "toml"
}

fn validate(content: &str, fmt: &str) -> Result<(), String> {
    match fmt {
        "json" => serde_json::from_str::<serde_json::Value>(content)
            .map(|_| ())
            .map_err(|e| e.to_string()),
        "toml" => toml::from_str::<toml::Value>(content)
            .map(|_| ())
            .map_err(|e| e.to_string()),
        other => Err(format!("Unknown format: {other}")),
    }
}

#[async_trait::async_trait]
impl crate::tool::Tool for ValidateDataTool {
    fn spec(&self) -> ToolSpec {
        ToolSpec {
            name: "validate_data".into(),
            description: "Validate JSON or TOML data. Returns parse errors with details.".into(),
            parameters: serde_json::json!({
                "type": "object",
                "properties": {
                    "content": { "type": "string", "description": "Inline data to validate" },
                    "path":    { "type": "string", "description": "File path (alternative to content)" },
                    "format":  { "type": "string", "enum": ["auto","json","toml"], "description": "Default: auto-detect" }
                }
            }),
            permission: Permission::ReadOnly,
        }
    }

    async fn execute(&self, input: serde_json::Value, cwd: &Path) -> ToolResult {
        let hint = input
            .get("format")
            .and_then(|v| v.as_str())
            .unwrap_or("auto");
        let content = if let Some(c) = input.get("content").and_then(|v| v.as_str()) {
            c.to_string()
        } else if let Some(p) = input.get("path").and_then(|v| v.as_str()) {
            let path = if Path::new(p).is_absolute() {
                p.into()
            } else {
                cwd.join(p)
            };
            match tokio::fs::read_to_string(&path).await {
                Ok(s) => s,
                Err(e) => {
                    return ToolResult::err(format!("Read error: {e}"));
                }
            }
        } else {
            return ToolResult::err("Provide 'content' or 'path'");
        };
        let fmt = detect_format(hint, &content);
        match validate(&content, fmt) {
            Ok(()) => ToolResult::ok(format!("✓ Valid {fmt}")),
            Err(e) => ToolResult::ok(format!("✗ Invalid {fmt}: {e}")),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn valid_json() {
        assert!(validate(r#"{"a":1}"#, "json").is_ok());
    }
    #[test]
    fn invalid_json() {
        assert!(validate(r#"{"a":}"#, "json").is_err());
    }
    #[test]
    fn valid_toml() {
        assert!(validate("[s]\nk=\"v\"", "toml").is_ok());
    }
    #[test]
    fn invalid_toml() {
        assert!(validate("[s\nk=", "toml").is_err());
    }
    #[test]
    fn detect_json() {
        assert_eq!(detect_format("auto", r#"{"a":1}"#), "json");
    }
    #[test]
    fn detect_toml() {
        assert_eq!(detect_format("auto", "key=\"val\""), "toml");
    }
}
