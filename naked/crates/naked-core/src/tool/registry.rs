use std::collections::HashMap;
use std::path::Path;

use tokio::sync::mpsc;

use crate::provider::tool_spec_to_anthropic_json;
use crate::types::{AgentEvent, ToolResult, ToolSpec};

use super::Tool;

pub struct ToolRegistry {
    tools: HashMap<String, Box<dyn Tool>>,
}

impl ToolRegistry {
    pub fn new(tools: Vec<Box<dyn Tool>>) -> Self {
        let mut map = HashMap::new();
        for tool in tools {
            map.insert(tool.spec().name.clone(), tool);
        }
        Self { tools: map }
    }

    pub fn specs(&self) -> Vec<ToolSpec> {
        self.tools.values().map(|t| t.spec()).collect()
    }

    pub fn schemas_json(&self) -> Vec<serde_json::Value> {
        self.specs()
            .iter()
            .map(tool_spec_to_anthropic_json)
            .collect()
    }

    pub fn get(&self, name: &str) -> Option<&dyn Tool> {
        self.tools.get(name).map(|t| t.as_ref())
    }

    pub async fn execute(&self, name: &str, input: serde_json::Value, cwd: &Path) -> ToolResult {
        match self.tools.get(name) {
            Some(tool) => tool.execute(input, cwd).await,
            None => ToolResult::err(format!("Unknown tool: {name}")),
        }
    }

    pub async fn execute_with_progress(
        &self,
        name: &str,
        input: serde_json::Value,
        cwd: &Path,
        progress: mpsc::Sender<AgentEvent>,
    ) -> ToolResult {
        let mut result = match self.tools.get(name) {
            Some(tool) => {
                tool.execute_with_progress(input.clone(), cwd, progress)
                    .await
            }
            None => {
                return ToolResult::err(format!("Unknown tool: {name}"));
            }
        };

        // Large output routing: truncate oversized tool results.
        result.output = super::large_output::route_large_output(
            &result.output,
            super::large_output::DEFAULT_THRESHOLD_CHARS,
        );

        // Post-edit validation: if a file-modifying tool succeeded,
        // run a language-specific check and append diagnostics.
        if !result.is_error
            && let Some(path) = Self::edited_file_path(name, &input)
        {
            let full = if path.is_absolute() {
                path
            } else {
                cwd.join(&path)
            };
            if let Some(validation) = super::validator::run_post_edit_check(&full, cwd).await {
                result.output.push_str(&validation.to_tool_suffix());
            }
        }

        result
    }

    /// Extract the file path from a file-modifying tool's input.
    fn edited_file_path(tool_name: &str, input: &serde_json::Value) -> Option<std::path::PathBuf> {
        match tool_name {
            "write_file" | "edit_file" | "apply_patch" => input
                .get("file_path")
                .or_else(|| input.get("path"))
                .and_then(|v| v.as_str())
                .map(std::path::PathBuf::from),
            _ => None,
        }
    }

    pub fn tool_names(&self) -> Vec<String> {
        self.tools.keys().cloned().collect()
    }

    pub fn register(&mut self, tool: Box<dyn Tool>) {
        let name = tool.spec().name.clone();
        self.tools.insert(name, tool);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::Permission;

    struct DummyTool {
        name: String,
        output: String,
    }

    #[async_trait::async_trait]
    impl Tool for DummyTool {
        fn spec(&self) -> ToolSpec {
            ToolSpec {
                name: self.name.clone(),
                description: "dummy".into(),
                parameters: serde_json::json!({"type": "object"}),
                permission: Permission::ReadOnly,
            }
        }

        async fn execute(&self, _input: serde_json::Value, _cwd: &Path) -> ToolResult {
            ToolResult::ok(self.output.clone())
        }
    }

    #[test]
    fn registry_stores_and_retrieves_tools() {
        let tools: Vec<Box<dyn Tool>> = vec![
            Box::new(DummyTool {
                name: "a".into(),
                output: "out_a".into(),
            }),
            Box::new(DummyTool {
                name: "b".into(),
                output: "out_b".into(),
            }),
        ];
        let reg = ToolRegistry::new(tools);

        assert!(reg.get("a").is_some());
        assert!(reg.get("b").is_some());
        assert!(reg.get("c").is_none());
    }

    #[test]
    fn specs_returns_all_tools() {
        let tools: Vec<Box<dyn Tool>> = vec![
            Box::new(DummyTool {
                name: "x".into(),
                output: "".into(),
            }),
            Box::new(DummyTool {
                name: "y".into(),
                output: "".into(),
            }),
        ];
        let reg = ToolRegistry::new(tools);
        let specs = reg.specs();
        assert_eq!(specs.len(), 2);
    }

    #[test]
    fn schemas_json_formats_correctly() {
        let tools: Vec<Box<dyn Tool>> = vec![Box::new(DummyTool {
            name: "test".into(),
            output: "".into(),
        })];
        let reg = ToolRegistry::new(tools);
        let schemas = reg.schemas_json();
        assert_eq!(schemas.len(), 1);
        assert_eq!(schemas[0]["name"], "test");
        assert!(schemas[0]["input_schema"].is_object());
    }

    #[tokio::test]
    async fn execute_unknown_tool_returns_error() {
        let reg = ToolRegistry::new(Vec::new());
        let result = reg
            .execute("nope", serde_json::json!({}), Path::new("/tmp"))
            .await;
        assert!(result.is_error);
        assert!(result.output.contains("Unknown tool"));
    }

    #[tokio::test]
    async fn execute_known_tool_runs() {
        let tools: Vec<Box<dyn Tool>> = vec![Box::new(DummyTool {
            name: "greet".into(),
            output: "hello".into(),
        })];
        let reg = ToolRegistry::new(tools);
        let result = reg
            .execute("greet", serde_json::json!({}), Path::new("/tmp"))
            .await;
        assert!(!result.is_error);
        assert_eq!(result.output, "hello");
    }

    #[test]
    fn register_adds_tool() {
        let mut reg = ToolRegistry::new(Vec::new());
        assert!(reg.get("new_tool").is_none());
        reg.register(Box::new(DummyTool {
            name: "new_tool".into(),
            output: "".into(),
        }));
        assert!(reg.get("new_tool").is_some());
    }

    #[test]
    fn tool_names_returns_all_names() {
        let tools: Vec<Box<dyn Tool>> = vec![
            Box::new(DummyTool {
                name: "a".into(),
                output: "".into(),
            }),
            Box::new(DummyTool {
                name: "b".into(),
                output: "".into(),
            }),
        ];
        let reg = ToolRegistry::new(tools);
        let mut names = reg.tool_names();
        names.sort();
        assert_eq!(names, vec!["a", "b"]);
    }
}
