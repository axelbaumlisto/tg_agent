//! B7: Remote execution context.
//!
//! When active, tools route operations through SSH to a remote host.
//! Set via `/remote <host>` command, cleared via `/remote off`.

use std::sync::Arc;
use tokio::sync::RwLock;

use super::ops::{SshOps, ToolOps, LocalOps};

/// Global remote context — shared across all tools in a session.
#[derive(Clone)]
pub struct RemoteContext {
    inner: Arc<RwLock<Option<Arc<dyn ToolOps>>>>,
}

impl RemoteContext {
    pub fn new() -> Self {
        Self {
            inner: Arc::new(RwLock::new(None)),
        }
    }

    /// Get the active ops (remote or local fallback).
    pub async fn ops(&self) -> Arc<dyn ToolOps> {
        self.inner
            .read()
            .await
            .clone()
            .unwrap_or_else(|| Arc::new(LocalOps) as Arc<dyn ToolOps>)
    }

    /// Switch to SSH remote host.
    pub async fn set_ssh(&self, host: String, key: Option<String>) {
        *self.inner.write().await = Some(Arc::new(SshOps { host, key }));
    }

    /// Switch back to local.
    pub async fn set_local(&self) {
        *self.inner.write().await = None;
    }

    /// Check if remote is active.
    pub async fn is_remote(&self) -> bool {
        self.inner.read().await.is_some()
    }

    /// Get the label of the active ops.
    pub async fn label(&self) -> String {
        match self.inner.read().await.as_ref() {
            Some(ops) => ops.label().to_string(),
            None => "local".to_string(),
        }
    }
}

impl Default for RemoteContext {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn default_is_local() {
        let ctx = RemoteContext::new();
        assert!(!ctx.is_remote().await);
        assert_eq!(ctx.label().await, "local");
    }

    #[tokio::test]
    async fn switch_to_ssh_and_back() {
        let ctx = RemoteContext::new();

        ctx.set_ssh("nova-1".to_string(), None).await;
        assert!(ctx.is_remote().await);
        assert_eq!(ctx.label().await, "nova-1");

        ctx.set_local().await;
        assert!(!ctx.is_remote().await);
        assert_eq!(ctx.label().await, "local");
    }
}
