use serde::{Deserialize, Serialize};

// -- Tool spec ---------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Permission {
    ReadOnly,
    WorkspaceWrite,
    Dangerous,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolSpec {
    pub name: String,
    pub description: String,
    pub parameters: serde_json::Value,
    pub permission: Permission,
}

#[derive(Debug, Clone)]
pub struct ToolResult {
    pub output: String,
    pub is_error: bool,
}

impl ToolResult {
    /// Convenience: successful result.
    pub fn ok(output: impl Into<String>) -> Self {
        Self {
            output: output.into(),
            is_error: false,
        }
    }

    /// Convenience: error result.
    pub fn err(output: impl Into<String>) -> Self {
        Self {
            output: output.into(),
            is_error: true,
        }
    }
}

// -- Model info --------------------------------------------------------------

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelInfo {
    pub provider: String,
    pub model_id: String,
    pub display_name: String,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tool_result_ok_creates_success() {
        let r = ToolResult::ok("hello");
        assert_eq!(r.output, "hello");
        assert!(!r.is_error);
    }

    #[test]
    fn tool_result_err_creates_error() {
        let r = ToolResult::err("oops");
        assert_eq!(r.output, "oops");
        assert!(r.is_error);
    }

    #[test]
    fn tool_result_ok_from_string() {
        let r = ToolResult::ok(format!("count: {}", 42));
        assert_eq!(r.output, "count: 42");
        assert!(!r.is_error);
    }
}
