//! `plan` — ordered step tracker with status progression.

use std::path::Path;
use std::sync::{Arc, Mutex};

use serde::{Deserialize, Serialize};

use crate::types::{Permission, ToolResult, ToolSpec};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum StepStatus {
    Pending,
    InProgress,
    Completed,
}

impl StepStatus {
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
pub struct PlanStep {
    pub step: String,
    pub status: StepStatus,
}

#[derive(Debug, Clone, Default)]
pub struct PlanState {
    steps: Arc<Mutex<Vec<PlanStep>>>,
}

impl PlanState {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn set(&self, steps: Vec<String>) {
        *crate::lock_or_recover(&self.steps) = steps
            .into_iter()
            .map(|s| PlanStep {
                step: s,
                status: StepStatus::Pending,
            })
            .collect();
    }

    pub fn update(&self, index: usize, status: StepStatus) -> bool {
        let mut steps = crate::lock_or_recover(&self.steps);
        if let Some(s) = steps.get_mut(index) {
            s.status = status;
            true
        } else {
            false
        }
    }

    pub fn get(&self) -> Vec<PlanStep> {
        crate::lock_or_recover(&self.steps).clone()
    }

    pub fn all_completed(&self) -> bool {
        let steps = crate::lock_or_recover(&self.steps);
        !steps.is_empty() && steps.iter().all(|s| s.status == StepStatus::Completed)
    }

    pub fn render(&self) -> String {
        let steps = self.get();
        if steps.is_empty() {
            return "No plan set".into();
        }
        steps
            .iter()
            .enumerate()
            .map(|(i, s)| format!("{} {}. {}", s.status.symbol(), i + 1, s.step))
            .collect::<Vec<_>>()
            .join("\n")
    }
}

pub struct PlanTool {
    state: PlanState,
}
impl PlanTool {
    pub fn new(state: PlanState) -> Self {
        Self { state }
    }
}

#[async_trait::async_trait]
impl crate::tool::Tool for PlanTool {
    fn spec(&self) -> ToolSpec {
        ToolSpec {
            name: "plan".into(),
            description: "Create or update a step-by-step plan. Actions: set (create plan), update (change step status), get (show plan).".into(),
            parameters: serde_json::json!({
                "type": "object",
                "properties": {
                    "action": { "type": "string", "enum": ["set","update","get"] },
                    "steps":  { "type": "array", "items": {"type":"string"}, "description": "Steps for 'set'" },
                    "index":  { "type": "integer", "description": "0-based step index for 'update'" },
                    "status": { "type": "string", "enum": ["pending","in_progress","completed"] }
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
            .unwrap_or("get");
        match action {
            "set" => {
                let steps: Vec<String> = input
                    .get("steps")
                    .and_then(|v| v.as_array())
                    .map(|a| {
                        a.iter()
                            .filter_map(|v| v.as_str().map(String::from))
                            .collect()
                    })
                    .unwrap_or_default();
                if steps.is_empty() {
                    return ToolResult::err("Provide steps array");
                }
                self.state.set(steps);
                ToolResult::ok(format!("Plan set:\n{}", self.state.render()))
            }
            "update" => {
                let idx = input.get("index").and_then(|v| v.as_u64()).unwrap_or(0) as usize;
                let status = input
                    .get("status")
                    .and_then(|v| v.as_str())
                    .and_then(StepStatus::parse);
                match status {
                    Some(s) if self.state.update(idx, s) => ToolResult::ok(format!(
                        "Step {} → {}\n{}",
                        idx + 1,
                        s.symbol(),
                        self.state.render()
                    )),
                    Some(_) => ToolResult::err(format!("Step index {idx} out of bounds")),
                    None => ToolResult::err("Invalid status"),
                }
            }
            _ => ToolResult::ok(self.state.render()),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn set_and_get() {
        let p = PlanState::new();
        p.set(vec!["A".into(), "B".into(), "C".into()]);
        assert_eq!(p.get().len(), 3);
        assert_eq!(p.get()[0].status, StepStatus::Pending);
    }

    #[test]
    fn update_status() {
        let p = PlanState::new();
        p.set(vec!["X".into()]);
        assert!(p.update(0, StepStatus::InProgress));
        assert_eq!(p.get()[0].status, StepStatus::InProgress);
    }

    #[test]
    fn out_of_bounds() {
        assert!(!PlanState::new().update(5, StepStatus::Completed));
    }

    #[test]
    fn symbols() {
        assert_eq!(StepStatus::Pending.symbol(), "○");
        assert_eq!(StepStatus::InProgress.symbol(), "◎");
        assert_eq!(StepStatus::Completed.symbol(), "●");
    }

    #[test]
    fn all_completed() {
        let p = PlanState::new();
        p.set(vec!["a".into(), "b".into()]);
        assert!(!p.all_completed());
        p.update(0, StepStatus::Completed);
        p.update(1, StepStatus::Completed);
        assert!(p.all_completed());
    }
}
