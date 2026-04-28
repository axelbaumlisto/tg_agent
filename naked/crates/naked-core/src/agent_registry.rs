use std::collections::HashMap;
use std::sync::Arc;
use std::time::Instant;

use tokio::sync::RwLock;
use tokio_util::sync::CancellationToken;

/// Snapshot of a running/completed sub-agent.
#[derive(Debug, Clone)]
pub struct AgentEntry {
    pub agent_id: String,
    pub prompt_preview: String,
    pub mode: String,
    pub status: AgentStatus,
    pub tokens: u64,
    pub tools_used: Vec<String>,
    pub last_tool: Option<String>,
    pub started_at: Instant,
    pub finished_at: Option<Instant>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AgentStatus {
    Running,
    Completed,
    Failed(String),
    Cancelled,
}

impl std::fmt::Display for AgentStatus {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Running => write!(f, "running"),
            Self::Completed => write!(f, "completed"),
            Self::Failed(e) => write!(f, "failed: {e}"),
            Self::Cancelled => write!(f, "cancelled"),
        }
    }
}

/// Shared registry of sub-agent lifecycle. Thread-safe via RwLock.
#[derive(Clone)]
pub struct AgentRegistry {
    inner: Arc<RwLock<RegistryInner>>,
}

struct RegistryInner {
    agents: HashMap<String, AgentEntry>,
    cancels: HashMap<String, CancellationToken>,
}

impl AgentRegistry {
    pub fn new() -> Self {
        Self {
            inner: Arc::new(RwLock::new(RegistryInner {
                agents: HashMap::new(),
                cancels: HashMap::new(),
            })),
        }
    }

    pub async fn register(
        &self,
        agent_id: &str,
        prompt_preview: &str,
        mode: &str,
        cancel: CancellationToken,
    ) {
        let mut inner = self.inner.write().await;
        inner.agents.insert(
            agent_id.to_string(),
            AgentEntry {
                agent_id: agent_id.to_string(),
                prompt_preview: prompt_preview.to_string(),
                mode: mode.to_string(),
                status: AgentStatus::Running,
                tokens: 0,
                tools_used: Vec::new(),
                last_tool: None,
                started_at: Instant::now(),
                finished_at: None,
            },
        );
        inner.cancels.insert(agent_id.to_string(), cancel);
    }

    pub async fn update_tool(&self, agent_id: &str, tool_name: &str) {
        let mut inner = self.inner.write().await;
        if let Some(entry) = inner.agents.get_mut(agent_id) {
            entry.last_tool = Some(tool_name.to_string());
            if !entry.tools_used.contains(&tool_name.to_string()) {
                entry.tools_used.push(tool_name.to_string());
            }
        }
    }

    pub async fn finish(&self, agent_id: &str, status: AgentStatus, tokens: u64) {
        let mut inner = self.inner.write().await;
        if let Some(entry) = inner.agents.get_mut(agent_id) {
            entry.status = status;
            entry.tokens = tokens;
            entry.finished_at = Some(Instant::now());
        }
        inner.cancels.remove(agent_id);
    }

    pub async fn cancel(&self, agent_id: &str) -> bool {
        let inner = self.inner.read().await;
        if let Some(cancel) = inner.cancels.get(agent_id) {
            cancel.cancel();
            true
        } else {
            false
        }
    }

    pub async fn get(&self, agent_id: &str) -> Option<AgentEntry> {
        self.inner.read().await.agents.get(agent_id).cloned()
    }

    pub async fn list_running(&self) -> Vec<AgentEntry> {
        self.inner
            .read()
            .await
            .agents
            .values()
            .filter(|e| e.status == AgentStatus::Running)
            .cloned()
            .collect()
    }

    pub async fn list_all(&self) -> Vec<AgentEntry> {
        self.inner.read().await.agents.values().cloned().collect()
    }

    /// Remove completed/failed entries older than `max_age`.
    pub async fn gc(&self, max_age: std::time::Duration) {
        let mut inner = self.inner.write().await;
        inner.agents.retain(|_, e| {
            if e.status == AgentStatus::Running {
                return true;
            }
            e.finished_at.map(|f| f.elapsed() < max_age).unwrap_or(true)
        });
    }
}

impl Default for AgentRegistry {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn register_and_list() {
        let reg = AgentRegistry::new();
        let cancel = CancellationToken::new();
        reg.register("sa-1", "test task", "explore", cancel).await;

        let running = reg.list_running().await;
        assert_eq!(running.len(), 1);
        assert_eq!(running[0].agent_id, "sa-1");
        assert_eq!(running[0].status, AgentStatus::Running);
    }

    #[tokio::test]
    async fn update_tool_and_finish() {
        let reg = AgentRegistry::new();
        let cancel = CancellationToken::new();
        reg.register("sa-2", "research", "general", cancel).await;

        reg.update_tool("sa-2", "web_search").await;
        reg.update_tool("sa-2", "bash").await;
        reg.update_tool("sa-2", "web_search").await;

        let entry = reg.get("sa-2").await.unwrap();
        assert_eq!(entry.last_tool.as_deref(), Some("web_search"));
        assert_eq!(entry.tools_used, vec!["web_search", "bash"]);

        reg.finish("sa-2", AgentStatus::Completed, 5000).await;
        let entry = reg.get("sa-2").await.unwrap();
        assert_eq!(entry.status, AgentStatus::Completed);
        assert_eq!(entry.tokens, 5000);
        assert!(reg.list_running().await.is_empty());
    }

    #[tokio::test]
    async fn cancel_agent() {
        let reg = AgentRegistry::new();
        let cancel = CancellationToken::new();
        let c2 = cancel.clone();
        reg.register("sa-3", "long task", "explore", cancel).await;

        assert!(!c2.is_cancelled());
        let ok = reg.cancel("sa-3").await;
        assert!(ok);
        assert!(c2.is_cancelled());

        let not_found = reg.cancel("sa-999").await;
        assert!(!not_found);
    }

    #[tokio::test]
    async fn gc_removes_old() {
        let reg = AgentRegistry::new();
        let cancel = CancellationToken::new();
        reg.register("sa-4", "old", "explore", cancel).await;
        reg.finish("sa-4", AgentStatus::Completed, 100).await;

        reg.gc(std::time::Duration::from_secs(0)).await;
        assert!(reg.get("sa-4").await.is_none());
    }
}
