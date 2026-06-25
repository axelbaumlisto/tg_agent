//! fff-powered file search tools — fuzzy find + content grep.
//!
//! Uses `fff-search` crate for SIMD-accelerated, frecency-ranked search.

use std::path::Path;
use std::sync::Arc;
use std::sync::atomic::Ordering;

use fff_query_parser::QueryParser;
use fff_search::SharedFilePicker;
use fff_search::file_picker::FuzzySearchOptions;
use fff_search::grep::GrepSearchOptions;

use crate::tool::fff_registry::{
    FffPickerBackend, FffPickerHandle, FffPickerRegistry, FffRegistryConfig,
};
use crate::types::{
    FFF_GREP_FALLBACK_COUNT, FFF_GREP_FAST_INDEX_COUNT, Permission, ToolResult, ToolSpec,
};

#[derive(Clone)]
enum FffStateInner {
    Legacy(SharedFilePicker),
    Registry {
        registry: Arc<FffPickerRegistry>,
        workspace: std::path::PathBuf,
        cfg: FffRegistryConfig,
    },
}

/// Search state shared by the paired find/grep tools for one registry build.
#[derive(Clone)]
pub struct FffState {
    inner: FffStateInner,
}

impl FffState {
    pub fn new(workspace: &Path) -> Self {
        let cfg = FffRegistryConfig {
            enabled: false,
            max_workspaces: 0,
            cache_max_bytes: 0,
        };
        let registry = FffPickerRegistry::new();
        let handle = registry.get_or_fallback(workspace, cfg).expect("fff init");
        Self {
            inner: FffStateInner::Legacy(handle.picker),
        }
    }

    pub fn with_registry(
        registry: Arc<FffPickerRegistry>,
        workspace: &Path,
        cfg: FffRegistryConfig,
    ) -> Self {
        Self {
            inner: FffStateInner::Registry {
                registry,
                workspace: workspace.to_path_buf(),
                cfg,
            },
        }
    }

    fn picker_handle(&self) -> Result<FffPickerHandle, String> {
        match &self.inner {
            FffStateInner::Legacy(picker) => Ok(FffPickerHandle {
                backend: FffPickerBackend::Fallback,
                picker: picker.clone(),
            }),
            FffStateInner::Registry {
                registry,
                workspace,
                cfg,
            } => registry.get_or_fallback(workspace, *cfg),
        }
    }
}

/// Fuzzy file search by name.
pub struct FffFindTool {
    state: FffState,
}

impl FffFindTool {
    pub fn new(state: &FffState) -> Self {
        Self {
            state: state.clone(),
        }
    }
}

#[async_trait::async_trait]
impl crate::tool::Tool for FffFindTool {
    fn spec(&self) -> ToolSpec {
        ToolSpec {
            name: "glob_search".into(),
            description: "Find files by name. Fuzzy, typo-resistant, frecency-ranked. \
                          Respects .gitignore. Use for finding files when you don't know exact location."
                .into(),
            parameters: serde_json::json!({
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "File name query (1-2 terms, e.g. 'main.rs', 'config')"
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

    async fn execute(&self, input: serde_json::Value, _cwd: &Path) -> ToolResult {
        let query = match input.get("query").and_then(|v| v.as_str()) {
            Some(q) if !q.is_empty() => q.to_string(),
            _ => return ToolResult::err("query required"),
        };
        let max = input
            .get("max_results")
            .and_then(|v| v.as_u64())
            .unwrap_or(20)
            .min(50) as usize;
        let handle = match self.state.picker_handle() {
            Ok(handle) => handle,
            Err(e) => return ToolResult::err(e),
        };

        run_blocking(move || execute_find(handle.picker, query, max)).await
    }
}

/// Content grep with frecency ranking.
pub struct FffGrepTool {
    state: FffState,
}

impl FffGrepTool {
    pub fn new(state: &FffState) -> Self {
        Self {
            state: state.clone(),
        }
    }
}

#[async_trait::async_trait]
impl crate::tool::Tool for FffGrepTool {
    fn spec(&self) -> ToolSpec {
        ToolSpec {
            name: "grep_search".into(),
            description: "Search file contents. Frecency-ranked, smart-case. \
                          Search for identifiers, strings, patterns. Returns matching lines with paths."
                .into(),
            parameters: serde_json::json!({
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search text — bare text, not regex (e.g. 'fn main', 'TODO')"
                    },
                    "max_results": {
                        "type": "integer",
                        "description": "Max matches (default 30)"
                    }
                },
                "required": ["query"]
            }),
            permission: Permission::ReadOnly,
        }
    }

    async fn execute(&self, input: serde_json::Value, _cwd: &Path) -> ToolResult {
        let query = match input.get("query").and_then(|v| v.as_str()) {
            Some(q) if !q.is_empty() => q.to_string(),
            _ => return ToolResult::err("query required"),
        };
        let max = input
            .get("max_results")
            .and_then(|v| v.as_u64())
            .unwrap_or(30)
            .min(100) as usize;
        let handle = match self.state.picker_handle() {
            Ok(handle) => handle,
            Err(e) => return ToolResult::err(e),
        };
        match handle.backend {
            FffPickerBackend::FastIndex => {
                FFF_GREP_FAST_INDEX_COUNT.fetch_add(1, Ordering::Relaxed);
            }
            FffPickerBackend::Fallback => {
                FFF_GREP_FALLBACK_COUNT.fetch_add(1, Ordering::Relaxed);
            }
        }

        run_blocking(move || execute_grep(handle.picker, query, max)).await
    }
}

async fn run_blocking(f: impl FnOnce() -> ToolResult + Send + 'static) -> ToolResult {
    match tokio::task::spawn_blocking(f).await {
        Ok(result) => result,
        Err(e) => ToolResult::err(format!("fff task failed: {e}")),
    }
}

fn execute_find(picker: SharedFilePicker, query: String, max: usize) -> ToolResult {
    let parser = QueryParser::default();
    let parsed = parser.parse(&query);
    let opts = FuzzySearchOptions::default();

    let picker_guard = match picker.read() {
        Ok(guard) => guard,
        Err(e) => return ToolResult::err(format!("fff picker lock failed: {e}")),
    };
    let Some(picker_ref) = picker_guard.as_ref() else {
        return ToolResult::err("fff picker missing");
    };
    let results = picker_ref.fuzzy_search(&parsed, None, opts);

    if results.items.is_empty() {
        return ToolResult::ok(format!("No files matching '{query}'"));
    }

    let mut output = format!(
        "{} files matching '{query}':\n",
        results.items.len().min(max)
    );
    for item in results.items.iter().take(max) {
        let path = item.relative_path(picker_ref);
        output.push_str(&format!("  {path}\n"));
    }

    ToolResult::ok(output)
}

fn execute_grep(picker: SharedFilePicker, query: String, max: usize) -> ToolResult {
    let parsed = fff_search::grep::parse_grep_query(&query);
    let opts = GrepSearchOptions {
        page_limit: max,
        ..Default::default()
    };

    let picker_guard = match picker.read() {
        Ok(guard) => guard,
        Err(e) => return ToolResult::err(format!("fff picker lock failed: {e}")),
    };
    let Some(picker_ref) = picker_guard.as_ref() else {
        return ToolResult::err("fff picker missing");
    };
    let results = picker_ref.grep(&parsed, &opts);

    if results.matches.is_empty() {
        return ToolResult::ok(format!("No matches for '{query}'"));
    }

    let lines = results
        .matches
        .iter()
        .take(max)
        .map(|m| {
            let file = &results.files[m.file_index];
            let path = file.relative_path(picker_ref);
            format!("{path}:{}: ...", m.line_number)
        })
        .collect::<Vec<_>>();

    ToolResult::ok(format_grep_output(&query, results.matches.len(), lines))
}

fn format_grep_output(query: &str, match_count: usize, lines: Vec<String>) -> String {
    let mut output = format!("{match_count} matches for '{query}':\n\n");
    for line in lines {
        output.push_str(&line);
        output.push('\n');
    }

    if output.len() > 8000 {
        let end = output.floor_char_boundary(8000);
        output = format!("{}…\n[truncated]", &output[..end]);
    }

    output
}

#[cfg(test)]
mod tests {
    use std::time::Duration;

    use super::*;
    use crate::tool::Tool;

    fn fast_state(workspace: &Path) -> FffState {
        FffState::with_registry(
            Arc::new(FffPickerRegistry::new()),
            workspace,
            FffRegistryConfig {
                enabled: true,
                max_workspaces: 4,
                cache_max_bytes: 8 * 1024 * 1024,
            },
        )
    }

    #[test]
    fn fff_state_creates() {
        let tmp = tempfile::tempdir().unwrap();
        std::fs::write(tmp.path().join("test.rs"), "fn main() {}").unwrap();
        let _state = FffState::new(tmp.path());
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn fff_grep_contract_stable_current_path_line_shape() {
        let tmp = tempfile::tempdir().unwrap();
        std::fs::write(tmp.path().join("alpha.txt"), "needle one\nother\n").unwrap();

        let off_tool = FffGrepTool::new(&FffState::new(tmp.path()));
        let off = off_tool
            .execute(
                serde_json::json!({"query":"needle","max_results":5}),
                tmp.path(),
            )
            .await;
        assert!(!off.is_error, "{}", off.output);
        assert_eq!(off.output, "No matches for 'needle'");

        let state = fast_state(tmp.path());
        let handle = state.picker_handle().unwrap();
        assert!(
            handle
                .picker
                .wait_for_indexing_complete(Duration::from_secs(5))
        );
        let on_tool = FffGrepTool::new(&state);
        let on = on_tool
            .execute(
                serde_json::json!({"query":"needle","max_results":5}),
                tmp.path(),
            )
            .await;
        assert!(!on.is_error, "{}", on.output);
        assert_contract_shape(&on.output);
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn fff_grep_spawn_blocking_does_not_starve_readonly_batch() {
        let tmp = tempfile::tempdir().unwrap();
        std::fs::write(tmp.path().join("alpha.txt"), "needle one\n").unwrap();
        let state = fast_state(tmp.path());
        let handle = state.picker_handle().unwrap();
        assert!(
            handle
                .picker
                .wait_for_indexing_complete(Duration::from_secs(5))
        );
        let tool = FffGrepTool::new(&state);

        let grep_fut = tool.execute(serde_json::json!({"query":"needle"}), tmp.path());
        let timer = tokio::time::sleep(Duration::from_millis(10));
        let (grep_result, _) = tokio::join!(grep_fut, timer);

        assert!(!grep_result.is_error, "{}", grep_result.output);
        assert_contract_shape(&grep_result.output);
    }

    fn assert_contract_shape(output: &str) {
        let line = output
            .lines()
            .find(|line| line.contains(":1: ..."))
            .unwrap_or_else(|| panic!("missing stable path:line shape in output: {output}"));
        assert!(line.ends_with(":1: ..."), "bad grep line shape: {line}");
        assert!(!line.contains("needle one"), "match text must stay omitted");
    }
}
