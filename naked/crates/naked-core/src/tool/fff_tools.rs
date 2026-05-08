//! fff-powered file search tools — fuzzy find + content grep.
//!
//! Uses `fff-search` crate for SIMD-accelerated, frecency-ranked search.

use std::path::Path;
use std::sync::Arc;

use fff_query_parser::QueryParser;
use fff_search::file_picker::FuzzySearchOptions;
use fff_search::file_picker::{FFFMode, FilePicker, FilePickerOptions};
use fff_search::grep::GrepSearchOptions;
use parking_lot::Mutex;

use crate::types::{Permission, ToolResult, ToolSpec};

/// Shared FilePicker per workspace.
pub struct FffState {
    picker: Arc<Mutex<FilePicker>>,
}

impl FffState {
    pub fn new(workspace: &Path) -> Self {
        let opts = FilePickerOptions {
            base_path: workspace.to_string_lossy().into_owned(),
            mode: FFFMode::Ai,
            enable_mmap_cache: false,
            enable_content_indexing: false,
            watch: false,
            cache_budget: None,
        };
        let picker = FilePicker::new(opts).expect("fff init");
        Self {
            picker: Arc::new(Mutex::new(picker)),
        }
    }

    pub fn picker(&self) -> Arc<Mutex<FilePicker>> {
        self.picker.clone()
    }
}

/// Fuzzy file search by name.
pub struct FffFindTool {
    picker: Arc<Mutex<FilePicker>>,
}

impl FffFindTool {
    pub fn new(state: &FffState) -> Self {
        Self {
            picker: state.picker(),
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
            Some(q) if !q.is_empty() => q,
            _ => {
                return ToolResult::err("query required");
            }
        };
        let max = input
            .get("max_results")
            .and_then(|v| v.as_u64())
            .unwrap_or(20)
            .min(50) as usize;

        let parser = QueryParser::default();
        let parsed = parser.parse(query);
        let opts = FuzzySearchOptions::default();

        let picker = self.picker.lock();
        let results = picker.fuzzy_search(&parsed, None, opts);

        if results.items.is_empty() {
            return ToolResult::ok(format!("No files matching '{query}'"));
        }

        let mut output = format!(
            "{} files matching '{query}':\n",
            results.items.len().min(max)
        );
        for item in results.items.iter().take(max) {
            let path = item.relative_path(&*picker);
            output.push_str(&format!("  {path}\n"));
        }

        ToolResult::ok(output)
    }
}

/// Content grep with frecency ranking.
pub struct FffGrepTool {
    picker: Arc<Mutex<FilePicker>>,
}

impl FffGrepTool {
    pub fn new(state: &FffState) -> Self {
        Self {
            picker: state.picker(),
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
            Some(q) if !q.is_empty() => q,
            _ => {
                return ToolResult::err("query required");
            }
        };
        let max = input
            .get("max_results")
            .and_then(|v| v.as_u64())
            .unwrap_or(30)
            .min(100) as usize;

        let parsed = fff_search::grep::parse_grep_query(query);
        let opts = GrepSearchOptions {
            page_limit: max,
            ..Default::default()
        };

        let picker = self.picker.lock();
        let results = picker.grep(&parsed, &opts);

        if results.matches.is_empty() {
            return ToolResult::ok(format!("No matches for '{query}'"));
        }

        let mut output = format!("{} matches for '{query}':\n\n", results.matches.len());
        for m in results.matches.iter().take(max) {
            let file = &results.files[m.file_index];
            let path = file.relative_path(&*picker);
            output.push_str(&format!("{path}:{}: ...\n", m.line_number));
        }

        if output.len() > 8000 {
            let end = output.floor_char_boundary(8000);
            output = format!("{}…\n[truncated]", &output[..end]);
        }

        ToolResult::ok(output)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fff_state_creates() {
        let tmp = tempfile::tempdir().unwrap();
        std::fs::write(tmp.path().join("test.rs"), "fn main() {}").unwrap();
        let _state = FffState::new(tmp.path());
        // Just verify it doesn't panic:
    }
}
