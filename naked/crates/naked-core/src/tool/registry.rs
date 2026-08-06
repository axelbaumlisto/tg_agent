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

        // B104: NO truncation here. This layer used to clamp every tool
        // result to the fixed `DEFAULT_THRESHOLD_CHARS` (12K, head 4K +
        // tail 2K) BEFORE the model-aware router in `loop_::tools`
        // (`route_large_output_aware`) ever saw the output. Because this
        // ran first and truncation is not idempotent-recoverable, the
        // model-aware limits (24K for >=100K windows, 180K for >=500K)
        // were dead code for large results: a 16.6 KB skill body reaching
        // a 200K-window model still lost its middle 10.6 KB.
        //
        // The single production execution path (`loop_::tools::
        // push_tool_outcome`) applies `route_large_output_aware` with the
        // live context window, and `ConversationHistory::push_tool_result`
        // enforces its own hard 8K backstop, so dropping the clamp here
        // cannot leave output unbounded in history. TG/TUI renderers cap
        // their own display separately (`handle_tool_end`, `tail_trim`).

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

    // ── D-INV-TOOL-OUTPUT-WINDOW-AWARE (B104) ──────────────────────────
    //
    // The dispatch layer must NOT pre-truncate tool output: it has no
    // access to the live context window, so any clamp here silently
    // overrides the model-aware router in `loop_::tools`. Reinstating
    // `route_large_output(&result.output, DEFAULT_THRESHOLD_CHARS)` makes
    // both tests below fail.

    /// A skill-body-sized result (16.6 KB, the real telegram-reader
    /// payload size) must survive dispatch byte-identical.
    #[tokio::test]
    async fn b104_dispatch_does_not_truncate_skill_sized_output() {
        let body = "## Section\nsome skill instructions line\n".repeat(450);
        assert!(
            body.len() > super::super::large_output::DEFAULT_THRESHOLD_CHARS,
            "fixture must exceed the old 12K clamp to be meaningful"
        );
        let tools: Vec<Box<dyn Tool>> = vec![Box::new(DummyTool {
            name: "skill".into(),
            output: body.clone(),
        })];
        let reg = ToolRegistry::new(tools);
        let (tx, _rx) = mpsc::channel(64);
        let result = reg
            .execute_with_progress("skill", serde_json::json!({}), Path::new("/tmp"), tx)
            .await;

        assert_eq!(
            result.output.len(),
            body.len(),
            "dispatch must not truncate; the model-aware router decides"
        );
        assert!(
            !result.output.contains("omitted"),
            "no truncation marker may be injected at dispatch time"
        );
    }

    /// The middle of an oversized result must reach the caller intact —
    /// this is the exact content class B104 destroyed (skill sections
    /// living between the kept 4K head and 2K tail).
    #[tokio::test]
    async fn b104_dispatch_preserves_middle_of_large_output() {
        // Must exceed DEFAULT_THRESHOLD_CHARS (12K) or the old clamp would
        // pass this test by doing nothing — keep head/tail past 4K/2K too so
        // the marker lands in the section the old router discarded.
        let head = "HEAD\n".repeat(1_600); // 8K > HEAD_CHARS(4K)
        let middle = "MIDDLE_MARKER_UNIQUE\n".to_string();
        let tail = "TAIL\n".repeat(1_600); // 8K > TAIL_CHARS(2K)
        let body = format!("{head}{middle}{tail}");
        assert!(
            body.len() > super::super::large_output::DEFAULT_THRESHOLD_CHARS,
            "fixture must exceed the old 12K clamp to be meaningful"
        );
        let tools: Vec<Box<dyn Tool>> = vec![Box::new(DummyTool {
            name: "skill".into(),
            output: body.clone(),
        })];
        let reg = ToolRegistry::new(tools);
        let (tx, _rx) = mpsc::channel(64);
        let result = reg
            .execute_with_progress("skill", serde_json::json!({}), Path::new("/tmp"), tx)
            .await;

        assert!(
            result.output.contains("MIDDLE_MARKER_UNIQUE"),
            "middle content must survive dispatch (B104 dropped exactly this)"
        );
        assert_eq!(result.output, body);
    }
}
