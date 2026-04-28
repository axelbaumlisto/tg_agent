use std::path::{Path, PathBuf};
use std::sync::{Arc, RwLock};

use async_trait::async_trait;
use serde_json::json;

use crate::memory::daily;
use crate::memory::dreams;
use crate::memory::service::MemoryService;
use crate::memory::store::MarkdownMemoryStore;
use crate::memory::types::{MemoryScope, MemoryType};
use crate::types::{Permission, ToolResult, ToolSpec};

use super::Tool;

/// Ambient "who is the caller" for memory operations. Set by the channel
/// (e.g. Telegram) before a turn runs, so the model can invoke `memory` with
/// `scope=user` without having to guess a user id.
#[derive(Clone, Default)]
pub struct MemoryContext {
    inner: Arc<RwLock<Option<String>>>,
}

impl MemoryContext {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn set_user_id(&self, id: Option<String>) {
        if let Ok(mut guard) = self.inner.write() {
            *guard = id;
        }
    }

    pub fn user_id(&self) -> Option<String> {
        self.inner.read().ok().and_then(|g| g.clone())
    }
}

pub struct MemoryTool {
    workspace: PathBuf,
    context: MemoryContext,
}

impl MemoryTool {
    pub fn new(workspace: PathBuf) -> Self {
        Self {
            workspace,
            context: MemoryContext::new(),
        }
    }

    pub fn with_context(workspace: PathBuf, context: MemoryContext) -> Self {
        Self { workspace, context }
    }
}

#[async_trait]
impl Tool for MemoryTool {
    fn spec(&self) -> ToolSpec {
        ToolSpec {
            name: "memory".into(),
            description: "Persistent memory across sessions with two-tier storage. \
                `store` writes to MEMORY.md (durable, surfaced in every system prompt). \
                `search`/`list`/`delete` operate on MEMORY.md. \
                `dreams` shows the daily-digest audit log (what got promoted/rejected). \
                `daily_drafts` shows today's short-lived draft entries (silent flushes, \
                auto-classified messages) that may be promoted to MEMORY.md tomorrow. \
                `stats` returns counts (durable rules, drafts today, promotions/rejections \
                over the last 7 days). Use `store` for explicit user rules / repeated \
                corrections; use `dreams`/`stats` only when the user asks about the \
                digest's behaviour."
                .into(),
            parameters: json!({
                "type": "object",
                "properties": {
                    "action": {
                        "type": "string",
                        "enum": ["store", "search", "list", "delete", "dreams", "daily_drafts", "stats"],
                        "description": "Action to perform"
                    },
                    "content": {
                        "type": "string",
                        "description": "For store: the rule/fact to remember. For search: the query string."
                    },
                    "memory_type": {
                        "type": "string",
                        "enum": ["preference", "correction", "project_knowledge", "failure"],
                        "description": "Type of memory (required for store)"
                    },
                    "scope": {
                        "type": "string",
                        "enum": ["project", "global", "user"],
                        "description": "Scope: project (current workspace), global (all projects), or user (the current author in a chat). Default: project"
                    },
                    "user_id": {
                        "type": "string",
                        "description": "Explicit user id for scope=user. If omitted, the active chat author is used."
                    },
                    "id": {
                        "type": "string",
                        "description": "Memory entry ID (required for delete)"
                    }
                },
                "required": ["action"]
            }),
            permission: Permission::ReadOnly,
        }
    }

    async fn execute(&self, input: serde_json::Value, _cwd: &Path) -> ToolResult {
        let action = input.get("action").and_then(|v| v.as_str()).unwrap_or("");
        let content = input.get("content").and_then(|v| v.as_str()).unwrap_or("");
        let scope_str = input
            .get("scope")
            .and_then(|v| v.as_str())
            .unwrap_or("project")
            .to_ascii_lowercase();
        let explicit_user = input
            .get("user_id")
            .and_then(|v| v.as_str())
            .map(str::to_string);

        let resolved_scope: Result<MemoryScope, String> = match scope_str.as_str() {
            "project" => Ok(MemoryScope::Project),
            "global" => Ok(MemoryScope::Global),
            "user" => match explicit_user.clone().or_else(|| self.context.user_id()) {
                Some(id) if !id.is_empty() => Ok(MemoryScope::User(id)),
                _ => {
                    Err("scope=user requires a user_id or an active chat author in context".into())
                }
            },
            other => other.parse::<MemoryScope>(),
        };

        match action {
            "store" => {
                if content.is_empty() {
                    return ToolResult {
                        output: "Error: content is required for store action".into(),
                        is_error: true,
                    };
                }
                let type_str = input
                    .get("memory_type")
                    .and_then(|v| v.as_str())
                    .unwrap_or("preference");
                let memory_type: MemoryType = match type_str.parse() {
                    Ok(t) => t,
                    Err(e) => {
                        return ToolResult {
                            output: format!("Error: {e}"),
                            is_error: true,
                        };
                    }
                };
                let scope = match resolved_scope {
                    Ok(s) => s,
                    Err(e) => {
                        return ToolResult {
                            output: format!("Error: {e}"),
                            is_error: true,
                        };
                    }
                };
                let scope_label = scope.to_string();

                match MemoryService::store(&self.workspace, scope, memory_type, content, "model") {
                    Ok(true) => ToolResult {
                        output: format!("Stored {scope_label} {memory_type} memory: {content}"),
                        is_error: false,
                    },
                    Ok(false) => ToolResult {
                        output: "Memory already exists (duplicate skipped)".into(),
                        is_error: false,
                    },
                    Err(e) => ToolResult {
                        output: format!("Error storing memory: {e}"),
                        is_error: true,
                    },
                }
            }

            "search" => {
                if content.is_empty() {
                    return ToolResult {
                        output: "Error: content (query) is required for search".into(),
                        is_error: true,
                    };
                }
                let sender = explicit_user.clone().or_else(|| self.context.user_id());
                let results =
                    MemoryService::search_for(&self.workspace, content, sender.as_deref());
                if results.is_empty() {
                    ToolResult {
                        output: "No memories found matching query".into(),
                        is_error: false,
                    }
                } else {
                    // Bump the in-memory recall counter for every hit so the
                    // promotion gate sees evidence the model actively used
                    // this rule. Best-effort — silently no-ops on poison.
                    for e in &results {
                        daily::record_recall(&e.scope, &e.content);
                    }
                    let lines: Vec<String> = results
                        .iter()
                        .map(|e| {
                            format!(
                                "[{}/{}] (id:{}) {}",
                                e.scope, e.memory_type, e.id, e.content
                            )
                        })
                        .collect();
                    ToolResult {
                        output: format!("Found {} memories:\n{}", results.len(), lines.join("\n")),
                        is_error: false,
                    }
                }
            }

            "list" => {
                let scope_filter = match scope_str.as_str() {
                    "global" => Some(MemoryScope::Global),
                    "project" => Some(MemoryScope::Project),
                    "user" => match resolved_scope {
                        Ok(s) => Some(s),
                        Err(e) => {
                            return ToolResult {
                                output: format!("Error: {e}"),
                                is_error: true,
                            };
                        }
                    },
                    _ => None,
                };
                let entries = MemoryService::list(&self.workspace, scope_filter);
                if entries.is_empty() {
                    ToolResult {
                        output: "No memories stored".into(),
                        is_error: false,
                    }
                } else {
                    let lines: Vec<String> = entries
                        .iter()
                        .map(|e| {
                            format!(
                                "[{}/{}] (id:{}) {}",
                                e.scope, e.memory_type, e.id, e.content
                            )
                        })
                        .collect();
                    ToolResult {
                        output: format!("{} memories:\n{}", entries.len(), lines.join("\n")),
                        is_error: false,
                    }
                }
            }

            "delete" => {
                let id = input.get("id").and_then(|v| v.as_str()).unwrap_or("");
                if id.is_empty() {
                    return ToolResult {
                        output: "Error: id is required for delete action".into(),
                        is_error: true,
                    };
                }
                let delete_result = if scope_str == "user" {
                    match resolved_scope {
                        Ok(MemoryScope::User(uid)) => MemoryService::delete_user(&uid, id),
                        Ok(_) => unreachable!(),
                        Err(e) => {
                            return ToolResult {
                                output: format!("Error: {e}"),
                                is_error: true,
                            };
                        }
                    }
                } else {
                    MemoryService::delete(&self.workspace, id)
                };
                match delete_result {
                    Ok(true) => ToolResult {
                        output: format!("Deleted memory {id}"),
                        is_error: false,
                    },
                    Ok(false) => ToolResult {
                        output: format!("Memory {id} not found"),
                        is_error: false,
                    },
                    Err(e) => ToolResult {
                        output: format!("Error deleting memory: {e}"),
                        is_error: true,
                    },
                }
            }

            "dreams" => {
                let scope = match resolved_scope {
                    Ok(s) => s,
                    Err(e) => {
                        return ToolResult {
                            output: format!("Error: {e}"),
                            is_error: true,
                        };
                    }
                };
                let entries = dreams::read_dreams(&self.workspace, &scope);
                if entries.is_empty() {
                    return ToolResult {
                        output: format!(
                            "No dream entries for scope={scope} yet. The daily digest \
                             writes here once per UTC day after evaluating drafts."
                        ),
                        is_error: false,
                    };
                }
                // Show only the last 7 entries — DREAMS.md can grow up to
                // dreams_retention_days (default 90) and we don't want to
                // dump a huge blob into the model's context.
                let recent: Vec<_> = entries.iter().rev().take(7).collect();
                let mut lines: Vec<String> = Vec::new();
                lines.push(format!(
                    "Last {} dream entries (scope={}):",
                    recent.len(),
                    scope
                ));
                for d in recent {
                    lines.push(format!(
                        "\n## {} ({})",
                        d.date.format("%Y-%m-%d"),
                        d.scope
                    ));
                    lines.push(format!("Summary: {}", d.summary));
                    if !d.promoted.is_empty() {
                        lines.push(format!("Promoted ({}):", d.promoted.len()));
                        for p in &d.promoted {
                            lines.push(format!("  + {p}"));
                        }
                    }
                    if !d.rejected.is_empty() {
                        lines.push(format!("Rejected ({}):", d.rejected.len()));
                        for r in &d.rejected {
                            let reason = r.reason.as_deref().unwrap_or("(no reason)");
                            lines.push(format!("  - {} — {reason}", r.content));
                        }
                    }
                }
                ToolResult {
                    output: lines.join("\n"),
                    is_error: false,
                }
            }

            "daily_drafts" => {
                let scope = match resolved_scope {
                    Ok(s) => s,
                    Err(e) => {
                        return ToolResult {
                            output: format!("Error: {e}"),
                            is_error: true,
                        };
                    }
                };
                let today = chrono::Utc::now().date_naive();
                let entries = MarkdownMemoryStore::read_daily(&self.workspace, &scope, today);
                if entries.is_empty() {
                    return ToolResult {
                        output: format!(
                            "No draft entries for {today} (scope={scope}). \
                             Drafts come from session-close snapshots, pre-compaction \
                             flushes, and auto-classified messages."
                        ),
                        is_error: false,
                    };
                }
                let lines: Vec<String> = entries
                    .iter()
                    .map(|e| {
                        format!(
                            "[{}/{}] (src:{}) {}",
                            e.scope, e.memory_type, e.source, e.content
                        )
                    })
                    .collect();
                ToolResult {
                    output: format!(
                        "Today's drafts ({} entries, scope={scope}):\n{}",
                        entries.len(),
                        lines.join("\n")
                    ),
                    is_error: false,
                }
            }

            "stats" => {
                let scope = match resolved_scope {
                    Ok(s) => s,
                    Err(e) => {
                        return ToolResult {
                            output: format!("Error: {e}"),
                            is_error: true,
                        };
                    }
                };
                let durable = MemoryService::list(&self.workspace, Some(scope.clone())).len();
                let today = chrono::Utc::now().date_naive();
                let drafts_today =
                    MarkdownMemoryStore::read_daily(&self.workspace, &scope, today).len();
                let dream_entries = dreams::read_dreams(&self.workspace, &scope);
                let week_cutoff = today - chrono::Duration::days(7);
                let recent: Vec<_> = dream_entries
                    .iter()
                    .filter(|d| d.date >= week_cutoff)
                    .collect();
                let promoted_7d: usize = recent.iter().map(|d| d.promoted.len()).sum();
                let rejected_7d: usize = recent.iter().map(|d| d.rejected.len()).sum();
                let last_run = dream_entries
                    .iter()
                    .map(|d| d.date)
                    .max()
                    .map(|d| d.to_string())
                    .unwrap_or_else(|| "never".to_string());
                ToolResult {
                    output: format!(
                        "Memory stats (scope={scope}):\n  \
                         durable rules (MEMORY.md): {durable}\n  \
                         drafts today ({today}): {drafts_today}\n  \
                         dream entries (7d): {}\n  \
                         promoted (7d): {promoted_7d}\n  \
                         rejected (7d): {rejected_7d}\n  \
                         last digest run: {last_run}",
                        recent.len(),
                    ),
                    is_error: false,
                }
            }

            _ => ToolResult {
                output: format!(
                    "Unknown action: {action}. Use store, search, list, delete, \
                     dreams, daily_drafts, or stats."
                ),
                is_error: true,
            },
        }
    }
}
