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
                    "id": {
                        "type": "string",
                        "description": "Memory entry ID (required for delete)"
                    }
                },
                "required": ["action"]
            }),
            // `memory` contains write actions; `effective_permission` keeps read actions
            // auto-approved and gates only store/delete like bash.rs gates by command risk.
            permission: Permission::WorkspaceWrite,
        }
    }

    fn effective_permission(&self, input: &serde_json::Value, _cwd: &Path) -> Permission {
        match input.get("action").and_then(|v| v.as_str()) {
            Some("store" | "delete") => Permission::WorkspaceWrite,
            _ => Permission::ReadOnly,
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
        let trusted_user = self.context.user_id();

        let trusted_user_scope = || match trusted_user.clone() {
            Some(id) if !id.is_empty() => Ok(MemoryScope::User(id)),
            _ => Err("scope=user requires an active chat author in trusted context".into()),
        };
        let resolved_scope: Result<MemoryScope, String> = match scope_str.as_str() {
            "project" => Ok(MemoryScope::Project),
            "global" => Ok(MemoryScope::Global),
            "user" => trusted_user_scope(),
            other => match other.parse::<MemoryScope>() {
                Ok(MemoryScope::User(_)) => trusted_user_scope(),
                parsed => parsed,
            },
        };

        self.dispatch_action(
            action,
            content,
            &input,
            &scope_str,
            resolved_scope,
            trusted_user,
        )
        .await
    }
}

fn is_user_scope_request(scope: &str) -> bool {
    scope == "user" || scope.starts_with("user:")
}

impl MemoryTool {
    // REGISTRY-WAIVE: too_many_arguments — refactor-defer, signature complexity acceptable
    #[allow(clippy::too_many_arguments)]
    async fn dispatch_action(
        &self,
        action: &str,
        content: &str,
        input: &serde_json::Value,
        scope_str: &str,
        resolved_scope: Result<MemoryScope, String>,
        trusted_user: Option<String>,
    ) -> ToolResult {
        match action {
            "store" => {
                if content.is_empty() {
                    return ToolResult::err("Error: content is required for store action");
                }
                let type_str = input
                    .get("memory_type")
                    .and_then(|v| v.as_str())
                    .unwrap_or("preference");
                let memory_type: MemoryType = match type_str.parse() {
                    Ok(t) => t,
                    Err(e) => {
                        return ToolResult::err(format!("Error: {e}"));
                    }
                };
                let scope = match resolved_scope {
                    Ok(s) => s,
                    Err(e) => {
                        return ToolResult::err(format!("Error: {e}"));
                    }
                };
                let scope_label = scope.to_string();

                match MemoryService::store(&self.workspace, scope, memory_type, content, "model") {
                    Ok(outcome) if outcome.inserted => {
                        if outcome.truncated {
                            ToolResult::ok(format!(
                                "Stored {scope_label} {memory_type} memory (truncated to {} chars): {}",
                                outcome.content.chars().count(),
                                outcome.content
                            ))
                        } else {
                            ToolResult::ok(format!(
                                "Stored {scope_label} {memory_type} memory: {}",
                                outcome.content
                            ))
                        }
                    }
                    Ok(_) => ToolResult::ok("Memory already exists (duplicate skipped)"),
                    Err(e) => ToolResult::err(format!("Error storing memory: {e}")),
                }
            }

            "search" => {
                if content.is_empty() {
                    return ToolResult::err("Error: content (query) is required for search");
                }
                let sender = if is_user_scope_request(scope_str) {
                    match &resolved_scope {
                        Ok(MemoryScope::User(uid)) => Some(uid.clone()),
                        Ok(_) => None,
                        Err(e) => {
                            return ToolResult::err(format!("Error: {e}"));
                        }
                    }
                } else {
                    trusted_user.clone()
                };
                let results =
                    MemoryService::search_for(&self.workspace, content, sender.as_deref());
                if results.is_empty() {
                    ToolResult::ok("No memories found matching query")
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
                    ToolResult::ok(format!(
                        "Found {} memories:\n{}",
                        results.len(),
                        lines.join("\n")
                    ))
                }
            }

            "list" => {
                let scope_filter = match scope_str {
                    "global" => Some(MemoryScope::Global),
                    "project" => Some(MemoryScope::Project),
                    s if is_user_scope_request(s) => match resolved_scope {
                        Ok(s) => Some(s),
                        Err(e) => {
                            return ToolResult::err(format!("Error: {e}"));
                        }
                    },
                    _ => None,
                };
                let entries = MemoryService::list(&self.workspace, scope_filter);
                if entries.is_empty() {
                    ToolResult::ok("No memories stored")
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
                    ToolResult::ok(format!("{} memories:\n{}", entries.len(), lines.join("\n")))
                }
            }

            "delete" => {
                let id = input.get("id").and_then(|v| v.as_str()).unwrap_or("");
                if id.is_empty() {
                    return ToolResult::err("Error: id is required for delete action");
                }
                let delete_result = if scope_str == "user" {
                    match resolved_scope {
                        Ok(MemoryScope::User(uid)) => MemoryService::delete_user(&uid, id),
                        Ok(_) => unreachable!(),
                        Err(e) => {
                            return ToolResult::err(format!("Error: {e}"));
                        }
                    }
                } else {
                    MemoryService::delete(&self.workspace, id)
                };
                match delete_result {
                    Ok(true) => ToolResult::ok(format!("Deleted memory {id}")),
                    Ok(false) => ToolResult::ok(format!("Memory {id} not found")),
                    Err(e) => ToolResult::err(format!("Error deleting memory: {e}")),
                }
            }

            "dreams" => {
                let scope = match resolved_scope {
                    Ok(s) => s,
                    Err(e) => {
                        return ToolResult::err(format!("Error: {e}"));
                    }
                };
                let entries = dreams::read_dreams(&self.workspace, &scope);
                if entries.is_empty() {
                    return ToolResult::ok(format!(
                        "No dream entries for scope={scope} yet. The daily digest \
                         writes here once per UTC day after evaluating drafts."
                    ));
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
                    lines.push(format!("\n## {} ({})", d.date.format("%Y-%m-%d"), d.scope));
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
                ToolResult::ok(lines.join("\n"))
            }

            "daily_drafts" => {
                let scope = match resolved_scope {
                    Ok(s) => s,
                    Err(e) => {
                        return ToolResult::err(format!("Error: {e}"));
                    }
                };
                let today = chrono::Utc::now().date_naive();
                let entries = MarkdownMemoryStore::read_daily(&self.workspace, &scope, today);
                if entries.is_empty() {
                    return ToolResult::ok(format!(
                        "No draft entries for {today} (scope={scope}). \
                         Drafts come from session-close snapshots, pre-compaction \
                         flushes, and auto-classified messages."
                    ));
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
                ToolResult::ok(format!(
                    "Today's drafts ({} entries, scope={scope}):\n{}",
                    entries.len(),
                    lines.join("\n")
                ))
            }

            "stats" => {
                let scope = match resolved_scope {
                    Ok(s) => s,
                    Err(e) => {
                        return ToolResult::err(format!("Error: {e}"));
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
                ToolResult::ok(format!(
                    "Memory stats (scope={scope}):\n  \
                     durable rules (MEMORY.md): {durable}\n  \
                     drafts today ({today}): {drafts_today}\n  \
                     dream entries (7d): {}\n  \
                     promoted (7d): {promoted_7d}\n  \
                     rejected (7d): {rejected_7d}\n  \
                     last digest run: {last_run}",
                    recent.len(),
                ))
            }

            _ => ToolResult::err(format!(
                "Unknown action: {action}. Use store, search, list, delete, \
                 dreams, daily_drafts, or stats."
            )),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::memory::store::{MAX_ENTRY_CHARS, MemoryPaths};
    use crate::tool::policy::{ToolDecision, ToolPolicy, default_pipeline};
    use std::path::Path;
    use tempfile::tempdir;

    fn make_tool(dir: &Path) -> MemoryTool {
        MemoryTool::new(dir.to_path_buf())
    }

    fn make_tool_with_user(dir: &Path, user_id: &str) -> MemoryTool {
        let context = MemoryContext::new();
        context.set_user_id(Some(user_id.to_string()));
        MemoryTool::with_context(dir.to_path_buf(), context)
    }

    #[tokio::test]
    async fn store_and_search_round_trip() {
        let tmp = tempdir().unwrap();
        let tool = make_tool(tmp.path());
        let res = tool
            .execute(
                serde_json::json!({"action": "store", "content": "prefer dark theme", "memory_type": "preference"}),
                tmp.path(),
            )
            .await;
        assert!(!res.is_error, "store failed: {}", res.output);

        let res = tool
            .execute(
                serde_json::json!({"action": "search", "content": "dark"}),
                tmp.path(),
            )
            .await;
        assert!(!res.is_error);
        assert!(
            res.output.contains("dark theme"),
            "search result: {}",
            res.output
        );
    }

    #[tokio::test]
    async fn user_scope_reads_ignore_model_supplied_foreign_user_id() {
        let tmp = tempdir().unwrap();
        let _root = MemoryPaths::set_test_root(tmp.path().join("home"));
        let tool = make_tool_with_user(tmp.path(), "alice");

        let bob = MemoryService::store(
            tmp.path(),
            MemoryScope::User("bob".into()),
            MemoryType::Preference,
            "bob-private-password",
            "test",
        )
        .unwrap();
        assert!(bob.inserted);
        let alice = MemoryService::store(
            tmp.path(),
            MemoryScope::User("alice".into()),
            MemoryType::Preference,
            "alice-private-note",
            "test",
        )
        .unwrap();
        assert!(alice.inserted);

        let search = tool
            .execute(
                serde_json::json!({
                    "action": "search",
                    "scope": "user",
                    "user_id": "bob",
                    "content": "private"
                }),
                tmp.path(),
            )
            .await;
        assert!(!search.is_error, "search failed: {}", search.output);
        eprintln!("foreign-user search output:\n{}", search.output);
        assert!(
            search.output.contains("alice-private-note"),
            "trusted current user memory should still be readable: {}",
            search.output
        );
        assert!(
            !search.output.contains("bob-private-password"),
            "model-supplied foreign user_id leaked another user's memory: {}",
            search.output
        );

        let list = tool
            .execute(
                serde_json::json!({"action": "list", "scope": "user", "user_id": "bob"}),
                tmp.path(),
            )
            .await;
        assert!(!list.is_error, "list failed: {}", list.output);
        eprintln!("foreign-user list output:\n{}", list.output);
        assert!(
            list.output.contains("alice-private-note"),
            "list: {}",
            list.output
        );
        assert!(
            !list.output.contains("bob-private-password"),
            "list: {}",
            list.output
        );

        let legacy_scope_spelling = tool
            .execute(
                serde_json::json!({"action": "list", "scope": "user:bob"}),
                tmp.path(),
            )
            .await;
        assert!(
            !legacy_scope_spelling.is_error,
            "legacy user:<id> spelling should resolve through trusted context: {}",
            legacy_scope_spelling.output
        );
        assert!(
            legacy_scope_spelling.output.contains("alice-private-note"),
            "list: {}",
            legacy_scope_spelling.output
        );
        assert!(
            !legacy_scope_spelling
                .output
                .contains("bob-private-password"),
            "list: {}",
            legacy_scope_spelling.output
        );
    }

    #[tokio::test]
    async fn user_scope_read_without_trusted_context_is_rejected() {
        let tmp = tempdir().unwrap();
        let _root = MemoryPaths::set_test_root(tmp.path().join("home"));
        let tool = make_tool(tmp.path());
        MemoryService::store(
            tmp.path(),
            MemoryScope::User("bob".into()),
            MemoryType::Preference,
            "bob-private-password",
            "test",
        )
        .unwrap();

        let res = tool
            .execute(
                serde_json::json!({"action": "list", "scope": "user", "user_id": "bob"}),
                tmp.path(),
            )
            .await;
        assert!(
            res.is_error,
            "foreign user read should require trusted context: {}",
            res.output
        );
        assert!(
            res.output.contains("trusted context"),
            "error: {}",
            res.output
        );
        assert!(
            !res.output.contains("bob-private-password"),
            "error leaked memory: {}",
            res.output
        );

        let legacy_scope_spelling = tool
            .execute(
                serde_json::json!({"action": "list", "scope": "user:bob"}),
                tmp.path(),
            )
            .await;
        assert!(
            legacy_scope_spelling.is_error,
            "user:<id> must not bypass trusted context: {}",
            legacy_scope_spelling.output
        );
        assert!(
            !legacy_scope_spelling
                .output
                .contains("bob-private-password"),
            "error leaked memory: {}",
            legacy_scope_spelling.output
        );
    }

    #[tokio::test]
    async fn short_store_output_is_unchanged() {
        let tmp = tempdir().unwrap();
        let _root = MemoryPaths::set_test_root(tmp.path().join("home"));
        let tool = make_tool(tmp.path());
        let res = tool
            .execute(
                serde_json::json!({"action": "store", "content": "prefer dark theme", "memory_type": "preference"}),
                tmp.path(),
            )
            .await;
        assert!(!res.is_error, "store failed: {}", res.output);
        eprintln!("short store output: {}", res.output);
        assert_eq!(
            res.output, "Stored project preference memory: prefer dark theme",
            "short store output should stay byte-for-byte unchanged"
        );
    }

    #[tokio::test]
    async fn overlong_store_reports_truncated_persisted_content() {
        let tmp = tempdir().unwrap();
        let _root = MemoryPaths::set_test_root(tmp.path().join("home"));
        let tool = make_tool(tmp.path());
        let submitted = format!("{}DO_NOT_REPORT", "x".repeat(MAX_ENTRY_CHARS + 20));
        let expected = format!("{}...", "x".repeat(MAX_ENTRY_CHARS.saturating_sub(3)));

        let res = tool
            .execute(
                serde_json::json!({"action": "store", "content": submitted, "memory_type": "preference"}),
                tmp.path(),
            )
            .await;
        assert!(!res.is_error, "store failed: {}", res.output);
        eprintln!("overlong store output: {}", res.output);
        assert!(
            res.output.contains("truncated to 500 chars"),
            "output: {}",
            res.output
        );
        assert!(
            res.output.contains(&expected),
            "output did not echo stored text: {}",
            res.output
        );
        assert!(
            !res.output.contains("DO_NOT_REPORT"),
            "output echoed unpersisted tail: {}",
            res.output
        );

        let stored = MemoryService::list(tmp.path(), Some(MemoryScope::Project));
        assert_eq!(stored.len(), 1);
        assert_eq!(stored[0].content, expected);
    }

    #[tokio::test]
    async fn same_user_reads_still_work_with_trusted_context() {
        let tmp = tempdir().unwrap();
        let _root = MemoryPaths::set_test_root(tmp.path().join("home"));
        let tool = make_tool_with_user(tmp.path(), "alice");
        MemoryService::store(
            tmp.path(),
            MemoryScope::User("alice".into()),
            MemoryType::Preference,
            "alice likes compact summaries",
            "test",
        )
        .unwrap();

        let res = tool
            .execute(
                serde_json::json!({"action": "search", "scope": "user", "content": "compact"}),
                tmp.path(),
            )
            .await;
        assert!(!res.is_error, "same-user search failed: {}", res.output);
        eprintln!("same-user search output:\n{}", res.output);
        assert!(
            res.output.contains("alice likes compact summaries"),
            "search: {}",
            res.output
        );
    }

    #[tokio::test]
    async fn list_returns_stored_entries() {
        let tmp = tempdir().unwrap();
        let tool = make_tool(tmp.path());
        tool.execute(
            serde_json::json!({"action": "store", "content": "rule one", "memory_type": "correction"}),
            tmp.path(),
        )
        .await;
        let res = tool
            .execute(serde_json::json!({"action": "list"}), tmp.path())
            .await;
        assert!(!res.is_error);
        assert!(res.output.contains("rule one"), "list: {}", res.output);
    }

    #[tokio::test]
    async fn unknown_action_returns_error() {
        let tmp = tempdir().unwrap();
        let tool = make_tool(tmp.path());
        let res = tool
            .execute(serde_json::json!({"action": "explode"}), tmp.path())
            .await;
        assert!(res.is_error);
        assert!(res.output.contains("Unknown action"));
    }

    #[tokio::test]
    async fn store_without_content_errors() {
        let tmp = tempdir().unwrap();
        let tool = make_tool(tmp.path());
        let res = tool
            .execute(serde_json::json!({"action": "store"}), tmp.path())
            .await;
        assert!(res.is_error, "should fail without content: {}", res.output);
    }

    #[tokio::test]
    async fn search_empty_memory_returns_clean() {
        let tmp = tempdir().unwrap();
        let tool = make_tool(tmp.path());
        let res = tool
            .execute(
                serde_json::json!({"action": "search", "content": "anything"}),
                tmp.path(),
            )
            .await;
        assert!(!res.is_error);
    }

    #[test]
    fn s7_memory_tool_spec_is_write_class_because_it_can_mutate() {
        let tmp = tempdir().unwrap();
        let tool = make_tool(tmp.path());
        assert_eq!(tool.spec().permission, Permission::WorkspaceWrite);
    }

    #[test]
    fn s7_memory_schema_has_no_legacy_facts_category() {
        let tmp = tempdir().unwrap();
        let tool = make_tool(tmp.path());
        let schema = tool.spec().parameters.to_string();
        assert!(
            !schema.contains("Facts"),
            "legacy remember/Facts category must not be advertised: {schema}"
        );
        assert!(
            schema.contains("project_knowledge"),
            "persistent factual knowledge must use MemoryType::ProjectKnowledge: {schema}"
        );
        assert!(
            !schema.contains("user_id"),
            "model-facing schema must not let the model choose a user id: {schema}"
        );
    }

    #[test]
    fn s7_memory_write_actions_require_approval_but_reads_stay_auto() {
        let tmp = tempdir().unwrap();
        let tool = make_tool(tmp.path());
        let policy = default_pipeline(None);

        for input in [
            serde_json::json!({"action": "store", "content": "prefer rust", "memory_type": "preference", "scope": "global"}),
            serde_json::json!({"action": "delete", "id": "abc123", "scope": "global"}),
        ] {
            let permission = tool.effective_permission(&input, tmp.path());
            assert_ne!(
                permission,
                Permission::ReadOnly,
                "write action stayed read-only: {input}"
            );
            assert_eq!(
                policy.classify("memory", &input, tmp.path(), permission),
                ToolDecision::AskUser(Permission::WorkspaceWrite),
                "memory write action must not be auto-approved: {input}"
            );
        }

        for action in ["search", "list", "dreams", "daily_drafts", "stats"] {
            let input = serde_json::json!({"action": action, "content": "rust"});
            let permission = tool.effective_permission(&input, tmp.path());
            assert_eq!(
                permission,
                Permission::ReadOnly,
                "read action was gated: {action}"
            );
            assert_eq!(
                policy.classify("memory", &input, tmp.path(), permission),
                ToolDecision::Execute,
                "memory read action must remain auto-approved: {action}"
            );
        }
    }
}
