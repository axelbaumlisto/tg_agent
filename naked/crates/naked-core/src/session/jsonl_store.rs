use std::path::PathBuf;

use async_trait::async_trait;
use chrono::Utc;
use tokio::io::AsyncWriteExt;

use super::artifacts::{
    extern_image_blocks, gc_old_image_artifacts_impl, gc_orphan_image_artifacts_impl,
    intern_image_blocks, rotate_if_needed,
};
use crate::error::{AgentError, Result};
use crate::history::ConversationHistory;
use crate::types::ConversationMessage;

use super::store::SessionStore;
use super::usage::UsageTracker;
use super::{Session, SessionMetadata, SessionState, SessionSummary};

/// Sentinel prefix used to externalize `ContentBlock::Image` payloads to disk
/// instead of bloating `session.jsonl` with inline base64. The constant lives
/// in `types::IMAGE_REF_SENTINEL_PREFIX` so the API-request guard in
/// `history::to_api_messages` can reference the same value.
///
/// Why a Text-block sentinel rather than a new `ContentBlock` variant?
/// Keeping the in-memory enum unchanged means provider request builders, token
/// estimators, and downstream consumers don't need to know about a "ref" form
/// — externalization is a pure storage concern owned by the JSONL store.
pub struct JsonlSessionStore {
    base_dir: PathBuf,
}

impl JsonlSessionStore {
    pub fn new(base_dir: PathBuf) -> Self {
        Self { base_dir }
    }

    /// `{base_dir}/{session_id}/` — each session is a directory.
    fn session_dir(&self, session_id: &str) -> PathBuf {
        self.base_dir.join(session_id)
    }

    /// `{base_dir}/{session_id}/session.jsonl`
    fn session_path(&self, session_id: &str) -> PathBuf {
        self.session_dir(session_id).join("session.jsonl")
    }

    /// Legacy flat path for migration: `{base_dir}/{session_id}.jsonl`
    fn legacy_path(&self, session_id: &str) -> PathBuf {
        self.base_dir.join(format!("{session_id}.jsonl"))
    }

    async fn ensure_session_dir(&self, session_id: &str) -> Result<()> {
        let dir = self.session_dir(session_id);
        tokio::fs::create_dir_all(&dir).await?;
        tokio::fs::create_dir_all(dir.join("artifacts")).await?;
        Ok(())
    }

    /// Migrate legacy flat .jsonl file into directory layout.
    async fn migrate_if_needed(&self, session_id: &str) -> Result<()> {
        let legacy = self.legacy_path(session_id);
        if legacy.exists() && legacy.is_file() {
            self.ensure_session_dir(session_id).await?;
            let new_path = self.session_path(session_id);
            if !new_path.exists() {
                tokio::fs::rename(&legacy, &new_path).await?;
            }
        }
        Ok(())
    }
}

#[async_trait]
impl SessionStore for JsonlSessionStore {
    async fn save(&self, session: &Session) -> Result<()> {
        self.ensure_session_dir(&session.id).await?;
        let path = self.session_path(&session.id);
        let msg_count = session.history.message_count();
        let artifacts_dir = self.artifacts_dir(&session.id);

        // (applied: KISS) Incremental save: if only new messages were added
        // since last persist, append them instead of rewriting everything.
        // Full rewrite on: first save, compaction (msg count shrank), or
        // if the file doesn't exist yet.
        let can_append = session.persisted_msg_count > 0
            && msg_count > session.persisted_msg_count
            && path.exists();

        if can_append {
            // Append only the new messages (fast path: O(delta)).
            let new_msgs = &session.history.messages()[session.persisted_msg_count..];
            let mut buf = String::new();
            for msg in new_msgs {
                let externalized = extern_image_blocks(&artifacts_dir, msg).await?;
                let record = serde_json::json!({ "type": "message", "message": externalized });
                buf.push_str(&serde_json::to_string(&record).map_err(|e| {
                    crate::error::AgentError::ProviderTyped(
                        crate::provider::error::ProviderError::Serialize {
                            context: "message".into(),
                            source: e.to_string(),
                        },
                    )
                })?);
                buf.push('\n');
            }
            use tokio::io::AsyncWriteExt;
            let mut file = tokio::fs::OpenOptions::new()
                .append(true)
                .open(&path)
                .await?;
            file.write_all(buf.as_bytes()).await?;
            file.flush().await?;
            return Ok(());
        }

        // Full rewrite (compaction, first save, or corruption recovery).
        rotate_if_needed(&path).await?;

        let mut lines = Vec::new();

        let meta = serde_json::json!({
            "type": "session_meta",
            "session_id": session.id,
            "created_at": session.created_at,
            "updated_at": session.updated_at,
            "workspace": session.workspace,
            "state": session.state,
            "metadata": session.metadata,
            "fork": session.fork_info,
            "system_prompt": session.history.system_prompt(),
            "files": serde_json::to_value(&session.files).unwrap_or_default(),
        });
        lines.push(serde_json::to_string(&meta).map_err(|e| {
            crate::error::AgentError::ProviderTyped(
                crate::provider::error::ProviderError::Serialize {
                    context: "meta".into(),
                    source: e.to_string(),
                },
            )
        })?);

        for msg in session.history.messages() {
            let externalized = extern_image_blocks(&artifacts_dir, msg).await?;
            let record = serde_json::json!({ "type": "message", "message": externalized });
            lines.push(serde_json::to_string(&record).map_err(|e| {
                crate::error::AgentError::ProviderTyped(
                    crate::provider::error::ProviderError::Serialize {
                        context: "message".into(),
                        source: e.to_string(),
                    },
                )
            })?);
        }

        let content = lines.join("\n") + "\n";
        let tmp = path.with_extension("tmp");
        tokio::fs::write(&tmp, &content).await?;
        tokio::fs::rename(&tmp, &path).await?;

        Ok(())
    }

    async fn load(&self, session_id: &str) -> Result<Option<Session>> {
        self.migrate_if_needed(session_id).await?;
        let path = self.session_path(session_id);
        if !path.exists() {
            return Ok(None);
        }

        let content = tokio::fs::read_to_string(&path).await?;
        let mut meta_json: Option<serde_json::Value> = None;
        let mut messages: Vec<ConversationMessage> = Vec::new();

        for line in content.lines() {
            let line = line.trim();
            if line.is_empty() {
                continue;
            }
            let record: serde_json::Value = serde_json::from_str(line)
                .map_err(|e| AgentError::Session(format!("JSONL parse: {e}")))?;

            match record.get("type").and_then(|v| v.as_str()) {
                Some("session_meta") => meta_json = Some(record),
                Some("message") => {
                    if let Some(msg_val) = record.get("message") {
                        let msg: ConversationMessage = serde_json::from_value(msg_val.clone())
                            .map_err(|e| AgentError::Session(format!("message parse: {e}")))?;
                        let msg = intern_image_blocks(&self.artifacts_dir(session_id), msg).await;
                        messages.push(msg);
                    }
                }
                _ => {}
            }
        }

        let meta = meta_json.ok_or_else(|| AgentError::Session("no session_meta found".into()))?;

        let system_prompt = meta
            .get("system_prompt")
            .and_then(|v| v.as_str())
            .map(|s| s.to_string())
            .or_else(|| {
                messages
                    .first()
                    .filter(|m| m.role == crate::types::Role::System)
                    .map(|m| m.text_content())
            })
            .unwrap_or_default();

        let mut history = ConversationHistory::new(system_prompt);
        for msg in &messages {
            history.push_raw(msg.clone());
        }

        let mut usage = UsageTracker::default();
        for msg in &messages {
            if let Some(u) = &msg.usage {
                usage.record_turn(u);
            }
        }

        let session_metadata: SessionMetadata = serde_json::from_value(meta["metadata"].clone())
            .unwrap_or(SessionMetadata {
                name: None,
                provider: String::new(),
                model: String::new(),
                channel: String::new(),
                channel_id: None,
            });

        let fork_info = meta
            .get("fork")
            .and_then(|v| serde_json::from_value(v.clone()).ok());

        let msg_count = history.message_count();
        let session = Session {
            id: meta["session_id"]
                .as_str()
                .unwrap_or(session_id)
                .to_string(),
            created_at: meta["created_at"]
                .as_str()
                .and_then(|s| s.parse().ok())
                .unwrap_or_else(Utc::now),
            updated_at: meta["updated_at"]
                .as_str()
                .and_then(|s| s.parse().ok())
                .unwrap_or_else(Utc::now),
            workspace: meta["workspace"]
                .as_str()
                .map(PathBuf::from)
                .unwrap_or_default(),
            history,
            usage,
            fork_info,
            state: SessionState::Sleeping,
            metadata: session_metadata,
            files: meta
                .get("files")
                .and_then(|v| serde_json::from_value(v.clone()).ok())
                .unwrap_or_default(),
            working_set: meta
                .get("working_set")
                .and_then(|v| serde_json::from_value(v.clone()).ok())
                .unwrap_or_default(),
            persisted_msg_count: msg_count,
        };

        Ok(Some(session))
    }

    async fn append_message(&self, session_id: &str, msg: &ConversationMessage) -> Result<()> {
        self.ensure_session_dir(session_id).await?;
        let path = self.session_path(session_id);

        let externalized = extern_image_blocks(&self.artifacts_dir(session_id), msg).await?;
        let record = serde_json::json!({
            "type": "message",
            "message": externalized,
        });
        let line = serde_json::to_string(&record).unwrap_or_else(|e| {
            tracing::error!("failed to serialize session record: {e}");
            String::new()
        }) + "\n";

        let mut file = tokio::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(&path)
            .await?;
        file.write_all(line.as_bytes()).await?;
        file.flush().await?;

        Ok(())
    }

    async fn list(&self) -> Result<Vec<SessionSummary>> {
        if !self.base_dir.exists() {
            return Ok(Vec::new());
        }

        let mut summaries = Vec::new();
        let mut entries = tokio::fs::read_dir(&self.base_dir).await?;
        while let Some(entry) = entries.next_entry().await? {
            let path = entry.path();
            let name = entry.file_name().to_str().unwrap_or("").to_string();

            // Directory-based session: {id}/session.jsonl
            if path.is_dir() && path.join("session.jsonl").exists() {
                if let Ok(Some(session)) = self.load(&name).await {
                    summaries.push(session.summary_with_root(&self.base_dir));
                }
                continue;
            }
            // Legacy flat file: {id}.jsonl — migrate on access
            if path.extension().and_then(|e| e.to_str()) == Some("jsonl") {
                let stem = path
                    .file_stem()
                    .and_then(|s| s.to_str())
                    .unwrap_or("")
                    .to_string();
                if let Ok(Some(session)) = self.load(&stem).await {
                    summaries.push(session.summary_with_root(&self.base_dir));
                }
            }
        }

        summaries.sort_by_key(|s| std::cmp::Reverse(s.updated_at));
        Ok(summaries)
    }

    async fn delete(&self, session_id: &str) -> Result<()> {
        let dir = self.session_dir(session_id);
        if dir.exists() && dir.is_dir() {
            tokio::fs::remove_dir_all(&dir).await?;
        }
        let legacy = self.legacy_path(session_id);
        if legacy.exists() {
            tokio::fs::remove_file(&legacy).await?;
        }
        Ok(())
    }

    fn artifacts_dir(&self, session_id: &str) -> PathBuf {
        self.session_dir(session_id).join("artifacts")
    }

    async fn gc_orphan_image_artifacts(&self, session_id: &str) -> Result<usize> {
        gc_orphan_image_artifacts_impl(
            &self.session_path(session_id),
            &self.artifacts_dir(session_id),
        )
        .await
    }

    async fn gc_old_image_artifacts(&self, session_id: &str, max_age_secs: u64) -> Result<usize> {
        gc_old_image_artifacts_impl(&self.artifacts_dir(session_id), max_age_secs).await
    }

    fn session_root(&self, session_id: &str) -> PathBuf {
        self.session_dir(session_id)
    }

    async fn mark_active(&self, session_id: &str) -> Result<()> {
        let path = self.base_dir.join(".active_sessions");
        let mut content = tokio::fs::read_to_string(&path).await.unwrap_or_default();
        if !content.lines().any(|l| l.trim() == session_id) {
            content.push_str(session_id);
            content.push('\n');
            tokio::fs::write(&path, &content).await?;
        }
        Ok(())
    }

    async fn mark_idle(&self, session_id: &str) -> Result<()> {
        let path = self.base_dir.join(".active_sessions");
        if let Ok(content) = tokio::fs::read_to_string(&path).await {
            let filtered: String = content
                .lines()
                .filter(|l| l.trim() != session_id)
                .map(|l| format!("{l}\n"))
                .collect();
            tokio::fs::write(&path, &filtered).await?;
        }
        Ok(())
    }

    async fn drain_interrupted(&self) -> Vec<String> {
        let path = self.base_dir.join(".active_sessions");
        let content = tokio::fs::read_to_string(&path).await.unwrap_or_default();
        let _ = tokio::fs::remove_file(&path).await;
        content
            .lines()
            .filter(|l| !l.trim().is_empty())
            .map(|l| l.trim().to_string())
            .collect()
    }
}

#[cfg(test)]
#[path = "jsonl_store_tests.rs"]
mod tests;
