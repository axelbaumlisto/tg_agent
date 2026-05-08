//! `todo` — persistent in-memory todo list per session.

use std::path::Path;
use std::sync::{Arc, Mutex};

use serde::{Deserialize, Serialize};

use crate::types::{Permission, ToolResult, ToolSpec};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TodoStatus {
    Pending,
    InProgress,
    Completed,
}

impl TodoStatus {
    pub fn symbol(self) -> &'static str {
        match self {
            Self::Pending => "○",
            Self::InProgress => "◎",
            Self::Completed => "●",
        }
    }
    pub fn parse(s: &str) -> Option<Self> {
        match s.trim().to_lowercase().as_str() {
            "pending" => Some(Self::Pending),
            "in_progress" | "inprogress" => Some(Self::InProgress),
            "completed" | "done" => Some(Self::Completed),
            _ => None,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TodoItem {
    pub id: u32,
    pub content: String,
    pub status: TodoStatus,
}

#[derive(Debug, Clone, Default)]
pub struct TodoList {
    items: Arc<Mutex<Vec<TodoItem>>>,
    next_id: Arc<Mutex<u32>>,
}

impl TodoList {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn add(&self, content: &str) -> u32 {
        let mut id = crate::lock_or_recover(&self.next_id);
        *id += 1;
        let item = TodoItem {
            id: *id,
            content: content.to_string(),
            status: TodoStatus::Pending,
        };
        crate::lock_or_recover(&self.items).push(item);
        *id
    }

    pub fn update(&self, id: u32, status: TodoStatus) -> bool {
        let mut items = crate::lock_or_recover(&self.items);
        if let Some(item) = items.iter_mut().find(|i| i.id == id) {
            item.status = status;
            true
        } else {
            false
        }
    }

    pub fn remove(&self, id: u32) -> bool {
        let mut items = crate::lock_or_recover(&self.items);
        let len = items.len();
        items.retain(|i| i.id != id);
        items.len() < len
    }

    pub fn list(&self) -> Vec<TodoItem> {
        crate::lock_or_recover(&self.items).clone()
    }

    pub fn completion_pct(&self) -> u8 {
        let items = crate::lock_or_recover(&self.items);
        if items.is_empty() {
            return 0;
        }
        let done = items
            .iter()
            .filter(|i| i.status == TodoStatus::Completed)
            .count();
        ((done * 100) / items.len()) as u8
    }

    pub fn render(&self) -> String {
        let items = self.list();
        if items.is_empty() {
            return "No todos".into();
        }
        let mut out: Vec<String> = items
            .iter()
            .map(|i| format!("{} [{}] {}", i.status.symbol(), i.id, i.content))
            .collect();
        out.push(format!("{}% complete", self.completion_pct()));
        out.join("\n")
    }
}

pub struct TodoTool {
    list: TodoList,
}
impl TodoTool {
    pub fn new(list: TodoList) -> Self {
        Self { list }
    }
}

#[async_trait::async_trait]
impl crate::tool::Tool for TodoTool {
    fn spec(&self) -> ToolSpec {
        ToolSpec {
            name: "todo".into(),
            description: "Manage a todo list: add, update status, remove, list items. Persistent within session.".into(),
            parameters: serde_json::json!({
                "type": "object",
                "properties": {
                    "action":  { "type": "string", "enum": ["add","update","remove","list"] },
                    "content": { "type": "string", "description": "Text for add" },
                    "id":      { "type": "integer", "description": "Item ID for update/remove" },
                    "status":  { "type": "string", "enum": ["pending","in_progress","completed"], "description": "New status for update" }
                },
                "required": ["action"]
            }),
            permission: Permission::ReadOnly,
        }
    }

    async fn execute(&self, input: serde_json::Value, _cwd: &Path) -> ToolResult {
        let action = input
            .get("action")
            .and_then(|v| v.as_str())
            .unwrap_or("list");
        match action {
            "add" => {
                let content = input
                    .get("content")
                    .and_then(|v| v.as_str())
                    .unwrap_or("(empty)");
                let id = self.list.add(content);
                ToolResult::ok(format!("Added todo #{id}: {content}"))
            }
            "update" => {
                let id = input.get("id").and_then(|v| v.as_u64()).unwrap_or(0) as u32;
                let status = input
                    .get("status")
                    .and_then(|v| v.as_str())
                    .and_then(TodoStatus::parse);
                match status {
                    Some(s) if self.list.update(id, s) => {
                        ToolResult::ok(format!("Updated #{id} → {}", s.symbol()))
                    }
                    Some(_) => ToolResult::err(format!("Todo #{id} not found")),
                    None => ToolResult::err("Invalid status"),
                }
            }
            "remove" => {
                let id = input.get("id").and_then(|v| v.as_u64()).unwrap_or(0) as u32;
                if self.list.remove(id) {
                    ToolResult::ok(format!("Removed #{id}"))
                } else {
                    ToolResult::err(format!("Todo #{id} not found"))
                }
            }
            _ => ToolResult::ok(self.list.render()),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn add_and_list() {
        let tl = TodoList::new();
        tl.add("task A");
        tl.add("task B");
        assert_eq!(tl.list().len(), 2);
    }

    #[test]
    fn update_status() {
        let tl = TodoList::new();
        let id = tl.add("x");
        assert!(tl.update(id, TodoStatus::InProgress));
        assert_eq!(tl.list()[0].status, TodoStatus::InProgress);
    }

    #[test]
    fn remove_item() {
        let tl = TodoList::new();
        let id = tl.add("x");
        tl.add("y");
        assert!(tl.remove(id));
        assert_eq!(tl.list().len(), 1);
    }

    #[test]
    fn completion_pct() {
        let tl = TodoList::new();
        let a = tl.add("a");
        let _b = tl.add("b");
        let c = tl.add("c");
        tl.add("d");
        tl.update(a, TodoStatus::Completed);
        tl.update(c, TodoStatus::Completed);
        assert_eq!(tl.completion_pct(), 50);
    }

    #[test]
    fn render_output() {
        let tl = TodoList::new();
        tl.add("hello");
        let r = tl.render();
        assert!(r.contains("hello") && r.contains("○"));
    }
}
