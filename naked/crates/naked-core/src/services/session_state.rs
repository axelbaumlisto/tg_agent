//! SessionState — owns all session-related runtime data.
//!
//! Extracted from AgentCore so that session queries (is_active, list, usage)
//! can be served without touching provider/research/tool state.
//! Complex methods (dispatch_turn, create_session) stay on AgentCore
//! but access session data through `Arc<SessionState>`.

use std::collections::HashMap;
use std::sync::Arc;

use tokio::sync::RwLock;
use tokio_util::sync::CancellationToken;

use crate::session::store::SessionStore;
use crate::session::{Session, SessionState as SState, SessionSummary};
use crate::types::{ContentBlock, TurnUsage};

/// Runtime session data: the authoritative source for in-memory sessions,
/// cancellation tokens, persistence, and per-session sender tracking.
pub struct SessionState {
    /// In-memory sessions, keyed by session ID.
    pub sessions: Arc<RwLock<HashMap<String, Session>>>,
    /// Per-session cancellation tokens (for abort).
    pub cancels: RwLock<HashMap<String, CancellationToken>>,
    /// Durable session store (JSONL on disk).
    pub store: Arc<dyn SessionStore>,
    /// Per-session "current author" (e.g. Telegram user id).
    pub session_senders: RwLock<HashMap<String, String>>,
}

impl SessionState {
    pub fn new(store: Arc<dyn SessionStore>) -> Self {
        Self {
            sessions: Arc::new(RwLock::new(HashMap::new())),
            cancels: RwLock::new(HashMap::new()),
            store,
            session_senders: RwLock::new(HashMap::new()),
        }
    }

    // ── Queries (read-only) ─────────────────────────────────────────

    pub async fn is_session_active(&self, session_id: &str) -> bool {
        self.sessions
            .read()
            .await
            .get(session_id)
            .is_some_and(|s| s.state == SState::Active)
    }

    pub async fn list_sessions(&self) -> Vec<SessionSummary> {
        let sessions = self.sessions.read().await;
        let mut out: Vec<SessionSummary> = sessions
            .values()
            .map(|s| SessionSummary {
                id: s.id.clone(),
                name: s.metadata.name.clone(),
                created_at: s.created_at,
                updated_at: s.updated_at,
                message_count: s.history.message_count(),
                state: s.state.clone(),
                provider: Some(s.metadata.provider.clone()),
                model: Some(s.metadata.model.clone()),
                image_artifacts_total_bytes: None,
            })
            .collect();
        out.sort_by_key(|s| std::cmp::Reverse(s.updated_at));
        out
    }

    pub async fn list_sessions_paged(&self, skip: usize, limit: usize) -> Vec<SessionSummary> {
        let all = self.list_sessions().await;
        all.into_iter().skip(skip).take(limit).collect()
    }

    pub async fn session_workspace(&self, session_id: &str) -> Option<std::path::PathBuf> {
        self.sessions
            .read()
            .await
            .get(session_id)
            .map(|s| s.workspace.clone())
    }

    pub async fn session_total_usage(&self, session_id: &str) -> TurnUsage {
        let sessions = self.sessions.read().await;
        let mut total = TurnUsage::default();
        if let Some(session) = sessions.get(session_id) {
            for msg in session.history.messages() {
                if let Some(u) = &msg.usage {
                    total.input_tokens += u.input_tokens;
                    total.output_tokens += u.output_tokens;
                    total.cache_read_tokens += u.cache_read_tokens;
                    total.cache_write_tokens += u.cache_write_tokens;
                }
            }
        }
        total
    }

    pub async fn session_file_stats(&self, session_id: &str) -> (Vec<String>, Vec<String>) {
        let sessions = self.sessions.read().await;
        if let Some(session) = sessions.get(session_id) {
            let read = session
                .files
                .read_only()
                .into_iter()
                .map(|s| s.to_string())
                .collect();
            let modified = session
                .files
                .modified()
                .into_iter()
                .map(|s| s.to_string())
                .collect();
            (read, modified)
        } else {
            (Vec::new(), Vec::new())
        }
    }

    pub async fn session_context_usage(&self, session_id: &str) -> Option<(usize, u32)> {
        self.sessions.read().await.get(session_id).map(|s| {
            (
                s.history.estimated_tokens(),
                s.history.context_window_tokens(),
            )
        })
    }

    // ── Mutations ───────────────────────────────────────────────────

    pub async fn abort(&self, session_id: &str) {
        if let Some(cancel) = self.cancels.read().await.get(session_id) {
            cancel.cancel();
        }
    }

    pub async fn queue_message(&self, session_id: &str, text: &str) {
        let mut sessions = self.sessions.write().await;
        if let Some(session) = sessions.get_mut(session_id) {
            session.history.push_user(text);
            session.updated_at = chrono::Utc::now();
            tracing::info!("queued message for active session [{session_id}]");
        }
    }

    pub async fn queue_message_multimodal(
        &self,
        session_id: &str,
        blocks: Vec<ContentBlock>,
    ) {
        let mut sessions = self.sessions.write().await;
        if let Some(session) = sessions.get_mut(session_id) {
            session.history.push_user_multimodal(blocks);
            session.updated_at = chrono::Utc::now();
        }
    }

    pub async fn close_session_summary(&self, session_id: &str) {
        let mut sessions = self.sessions.write().await;
        if let Some(session) = sessions.get_mut(session_id) {
            session.state = SState::Idle;
            session.updated_at = chrono::Utc::now();
            if let Err(e) = self.store.save(session).await {
                tracing::error!("failed to save session [{session_id}] on close: {e}");
            }
        }
    }

    // ── Sender tracking ─────────────────────────────────────────────

    pub async fn set_session_sender(&self, session_id: &str, sender_id: Option<String>) {
        match sender_id {
            Some(id) => {
                self.session_senders
                    .write()
                    .await
                    .insert(session_id.to_string(), id);
            }
            None => {
                self.session_senders.write().await.remove(session_id);
            }
        }
    }

    pub async fn session_sender(&self, session_id: &str) -> Option<String> {
        self.session_senders.read().await.get(session_id).cloned()
    }

    // ── Channel mapping ─────────────────────────────────────────────

    pub async fn set_session_channel_id(&self, session_id: &str, channel_id: &str) {
        let mut sessions = self.sessions.write().await;
        if let Some(session) = sessions.get_mut(session_id) {
            session.metadata.channel_id = Some(channel_id.to_string());
            if let Err(e) = self.store.save(session).await {
                tracing::error!("failed to persist channel_id for {session_id}: {e}");
            }
        }
    }

    pub async fn channel_session_mappings(&self) -> Vec<(String, String)> {
        let sessions = self.sessions.read().await;
        let mut best: HashMap<String, (&str, chrono::DateTime<chrono::Utc>)> = HashMap::new();
        for s in sessions.values() {
            if let Some(cid) = &s.metadata.channel_id {
                let entry = best.entry(cid.clone()).or_insert((&s.id, s.updated_at));
                if s.updated_at > entry.1 {
                    *entry = (&s.id, s.updated_at);
                }
            }
        }
        best.into_iter()
            .map(|(cid, (sid, _))| (cid, sid.to_string()))
            .collect()
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::session::jsonl_store::JsonlSessionStore;

    fn test_state() -> SessionState {
        let dir = tempfile::tempdir().unwrap();
        let store = Arc::new(JsonlSessionStore::new(dir.into_path()));
        SessionState::new(store)
    }

    #[tokio::test]
    async fn is_active_empty() {
        let ss = test_state();
        assert!(!ss.is_session_active("nonexistent").await);
    }

    #[tokio::test]
    async fn list_sessions_empty() {
        let ss = test_state();
        assert!(ss.list_sessions().await.is_empty());
    }

    #[tokio::test]
    async fn sender_roundtrip() {
        let ss = test_state();
        ss.set_session_sender("s1", Some("user42".into())).await;
        assert_eq!(ss.session_sender("s1").await, Some("user42".into()));
        ss.set_session_sender("s1", None).await;
        assert_eq!(ss.session_sender("s1").await, None);
    }

    #[tokio::test]
    async fn abort_nonexistent_is_noop() {
        let ss = test_state();
        ss.abort("nonexistent").await; // should not panic
    }
}
