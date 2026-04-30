use std::path::{Path, PathBuf};

use async_trait::async_trait;
use base64::Engine as _;
use chrono::Utc;
use tokio::io::AsyncWriteExt;

use crate::error::{AgentError, Result};
use crate::history::ConversationHistory;
use crate::types::{ContentBlock, ConversationMessage, IMAGE_REF_SENTINEL_PREFIX};

use super::store::SessionStore;
use super::usage::UsageTracker;
use super::{Session, SessionMetadata, SessionState, SessionSummary};

const ROTATE_AFTER_BYTES: u64 = 256 * 1024;
const MAX_ROTATED_FILES: usize = 3;

/// Sentinel prefix used to externalize `ContentBlock::Image` payloads to disk
/// instead of bloating `session.jsonl` with inline base64. The constant lives
/// in `types::IMAGE_REF_SENTINEL_PREFIX` so the API-request guard in
/// `history::to_api_messages` can reference the same value.
///
/// Why a Text-block sentinel rather than a new `ContentBlock` variant?
/// Keeping the in-memory enum unchanged means provider request builders, token
/// estimators, and downstream consumers don't need to know about a "ref" form
/// — externalization is a pure storage concern owned by the JSONL store.
const IMAGE_REF_PREFIX: &str = IMAGE_REF_SENTINEL_PREFIX;

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
        });
        lines.push(
            serde_json::to_string(&meta)
                .map_err(|e| crate::error::AgentError::Provider(format!("serialize meta: {e}")))?,
        );

        let artifacts_dir = self.artifacts_dir(&session.id);
        for msg in session.history.messages() {
            let externalized = extern_image_blocks(&artifacts_dir, msg).await?;
            let record = serde_json::json!({
                "type": "message",
                "message": externalized,
            });
            lines.push(serde_json::to_string(&record).map_err(|e| {
                crate::error::AgentError::Provider(format!("serialize message: {e}"))
            })?);
        }

        let content = lines.join("\n") + "\n";

        // Atomic write
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

        summaries.sort_by(|a, b| b.updated_at.cmp(&a.updated_at));
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
}

/// Pick a file extension from a MIME type. Falls back to `.bin` for unknown
/// types — we never trust the wire MIME blindly, but we do want artifact files
/// to be openable by the user with a sensible default association.
fn ext_for_mime(mime: &str) -> &'static str {
    match mime.to_ascii_lowercase().as_str() {
        "image/png" => "png",
        "image/jpeg" | "image/jpg" => "jpg",
        "image/webp" => "webp",
        "image/gif" => "gif",
        "image/heic" | "image/heif" => "heic",
        _ => "bin",
    }
}

/// Replace every `ContentBlock::Image` in `msg` with a sentinel `Text` block
/// that points to the externalized artifact on disk. This keeps `session.jsonl`
/// small (kilobytes, not megabytes) and lets users inspect attachments with
/// normal tools (`file`, `xdg-open`, `feh`, …).
///
/// Returns a clone of `msg` with image blocks substituted; the original is
/// untouched. Filenames are derived from a deterministic hash of the bytes so
/// the same image attached twice deduplicates on disk.
async fn extern_image_blocks(
    artifacts_dir: &Path,
    msg: &ConversationMessage,
) -> Result<ConversationMessage> {
    let needs_extern = msg
        .blocks
        .iter()
        .any(|b| matches!(b, ContentBlock::Image { .. }));
    if !needs_extern {
        return Ok(msg.clone());
    }

    // Async fs from the start: callers are async and a session can carry
    // multi-MB images. Even short blocking writes hurt the runtime under
    // concurrent saves (multiple chats appending in parallel).
    tokio::fs::create_dir_all(artifacts_dir)
        .await
        .map_err(|e| AgentError::Session(format!("create artifacts dir: {e}")))?;

    let mut out = msg.clone();
    for block in &mut out.blocks {
        if let ContentBlock::Image {
            mime,
            data_base64,
            detail,
        } = block
        {
            let bytes = base64::engine::general_purpose::STANDARD
                .decode(data_base64.as_bytes())
                .map_err(|e| AgentError::Session(format!("decode image base64: {e}")))?;
            // blake3 over (mime || 0x00 || bytes) — content-addressable so the
            // same image attached twice (or the same image across sessions
            // restored from backup) produces the same filename. Switched away
            // from `DefaultHasher` because that's a stdlib implementation
            // detail (SipHash today, may change), and a long-lived on-disk
            // identifier deserves a stable cryptographic digest.
            let mut hasher = blake3::Hasher::new();
            hasher.update(mime.as_bytes());
            hasher.update(&[0u8]);
            hasher.update(&bytes);
            let digest = hasher.finalize();
            let ext = ext_for_mime(mime);
            // 16 hex chars (64 bits) is plenty for collision resistance per
            // session — one session would need ~2^32 distinct images before
            // a single collision is likely. Keeps filenames human-typable.
            let fname = format!("img_{}.{ext}", &digest.to_hex().as_str()[..16]);
            let abs_path = artifacts_dir.join(&fname);
            // tokio::fs::try_exists avoids racing with another task that's
            // writing the same artifact; if it returns Err just attempt the
            // write — the file system is the final source of truth.
            let exists = tokio::fs::try_exists(&abs_path).await.unwrap_or(false);
            if !exists {
                tokio::fs::write(&abs_path, &bytes)
                    .await
                    .map_err(|e| AgentError::Session(format!("write artifact: {e}")))?;
            }
            // Relative to artifacts_dir; `intern_image_blocks` rejoins it.
            let mut payload = serde_json::json!({
                "mime": mime,
                "path": fname,
                "bytes": bytes.len(),
            });
            if let Some(d) = detail {
                payload["detail"] = serde_json::Value::String(d.as_str().to_string());
            }
            *block = ContentBlock::Text {
                text: format!("{IMAGE_REF_PREFIX}{payload}"),
            };
        }
    }
    Ok(out)
}

/// Inverse of `extern_image_blocks`: rehydrate sentinel Text blocks back into
/// `ContentBlock::Image` by reading bytes from the artifacts directory and
/// re-encoding to base64. Sentinel blocks whose artifact file is missing or
/// whose JSON is malformed are kept as-is (degraded but visible) rather than
/// dropped — losing user attachments silently is worse than showing a marker.
async fn intern_image_blocks(
    artifacts_dir: &Path,
    mut msg: ConversationMessage,
) -> ConversationMessage {
    for block in &mut msg.blocks {
        if let ContentBlock::Text { text } = block
            && let Some(rest) = text.strip_prefix(IMAGE_REF_PREFIX)
            && let Ok(payload) = serde_json::from_str::<serde_json::Value>(rest)
            && let (Some(mime), Some(rel_path)) = (
                payload.get("mime").and_then(|v| v.as_str()),
                payload.get("path").and_then(|v| v.as_str()),
            )
        {
            let abs = artifacts_dir.join(rel_path);
            match tokio::fs::read(&abs).await {
                Ok(bytes) => {
                    let b64 = base64::engine::general_purpose::STANDARD.encode(&bytes);
                    let detail =
                        payload
                            .get("detail")
                            .and_then(|v| v.as_str())
                            .and_then(|s| match s {
                                "low" => Some(crate::types::ImageDetail::Low),
                                "high" => Some(crate::types::ImageDetail::High),
                                "auto" => Some(crate::types::ImageDetail::Auto),
                                _ => None,
                            });
                    *block = ContentBlock::Image {
                        mime: mime.to_string(),
                        data_base64: b64,
                        detail,
                    };
                }
                Err(_) => {
                    // Artifact missing — keep the sentinel so the user can see
                    // that an image was here and locate the broken reference.
                }
            }
        }
    }
    msg
}

/// Walk `session.jsonl` to collect every artifact filename mentioned by a
/// sentinel marker, then delete files in `artifacts_dir` that aren't in that
/// set. Conservative on parse errors: a malformed line aborts the walk
/// (returning Ok(0)) so we never delete artifacts based on partial knowledge.
async fn gc_orphan_image_artifacts_impl(
    session_jsonl: &Path,
    artifacts_dir: &Path,
) -> Result<usize> {
    if !session_jsonl.exists() || !artifacts_dir.is_dir() {
        return Ok(0);
    }
    let content = tokio::fs::read_to_string(session_jsonl).await?;
    let mut referenced: std::collections::HashSet<String> = std::collections::HashSet::new();
    for line in content.lines() {
        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }
        let v: serde_json::Value = match serde_json::from_str(trimmed) {
            Ok(v) => v,
            Err(_) => return Ok(0),
        };
        // Walk the JSON tree looking for any string value starting with the
        // sentinel marker. The marker carries `{"path": "img_..."}` JSON
        // tail — extracting the path is a substring scan rather than a
        // structural walk because we don't know which content-block schema
        // emitted the marker (current is Text, but future may differ).
        collect_referenced_artifacts(&v, &mut referenced);
    }

    let mut entries = tokio::fs::read_dir(artifacts_dir).await?;
    let mut removed = 0usize;
    while let Some(entry) = entries.next_entry().await? {
        let name = entry.file_name();
        let name_s = name.to_string_lossy().to_string();
        if !name_s.starts_with("img_") {
            continue;
        }
        if !referenced.contains(&name_s) {
            // try_exists is not strictly required — if the file disappeared
            // between read_dir and remove, ignore the NotFound.
            match tokio::fs::remove_file(entry.path()).await {
                Ok(_) => removed += 1,
                Err(e) if e.kind() == std::io::ErrorKind::NotFound => {}
                Err(e) => return Err(e.into()),
            }
        }
    }
    Ok(removed)
}

/// Delete every `img_*` file in `artifacts_dir` whose mtime is older than
/// `max_age_secs` seconds relative to now. Files younger than the cutoff
/// are left alone. Missing directory → Ok(0); we never create it.
///
/// Note: this is a *reachability-free* sweep. The caller is responsible
/// for ensuring the sessions whose artifacts get culled have been
/// compacted or archived first, otherwise history references may become
/// stale (but will still render as `[image missing]` placeholders —
/// never a crash).
async fn gc_old_image_artifacts_impl(artifacts_dir: &Path, max_age_secs: u64) -> Result<usize> {
    if !artifacts_dir.is_dir() {
        return Ok(0);
    }
    let now = std::time::SystemTime::now();
    let cutoff = std::time::Duration::from_secs(max_age_secs);
    let mut entries = tokio::fs::read_dir(artifacts_dir).await?;
    let mut removed = 0usize;
    while let Some(entry) = entries.next_entry().await? {
        let name = entry.file_name();
        let name_s = name.to_string_lossy();
        if !name_s.starts_with("img_") {
            continue;
        }
        let meta = match entry.metadata().await {
            Ok(m) => m,
            Err(_) => continue,
        };
        let mtime = meta.modified().unwrap_or(now);
        let age = now.duration_since(mtime).unwrap_or_default();
        if age > cutoff {
            match tokio::fs::remove_file(entry.path()).await {
                Ok(_) => removed += 1,
                Err(e) if e.kind() == std::io::ErrorKind::NotFound => {}
                Err(e) => return Err(e.into()),
            }
        }
    }
    Ok(removed)
}

fn collect_referenced_artifacts(
    v: &serde_json::Value,
    out: &mut std::collections::HashSet<String>,
) {
    match v {
        serde_json::Value::String(s) => {
            if let Some(rest) = s.strip_prefix(IMAGE_REF_PREFIX)
                && let Ok(payload) = serde_json::from_str::<serde_json::Value>(rest)
                && let Some(path) = payload.get("path").and_then(|p| p.as_str())
            {
                out.insert(path.to_string());
            }
        }
        serde_json::Value::Array(arr) => {
            for x in arr {
                collect_referenced_artifacts(x, out);
            }
        }
        serde_json::Value::Object(map) => {
            for (_, x) in map {
                collect_referenced_artifacts(x, out);
            }
        }
        _ => {}
    }
}

async fn rotate_if_needed(path: &Path) -> Result<()> {
    if !path.exists() {
        return Ok(());
    }
    let meta = tokio::fs::metadata(path).await?;
    if meta.len() < ROTATE_AFTER_BYTES {
        return Ok(());
    }

    for i in (1..MAX_ROTATED_FILES).rev() {
        let from = path.with_extension(format!("jsonl.{i}"));
        let to = path.with_extension(format!("jsonl.{}", i + 1));
        if from.exists() {
            let _ = tokio::fs::rename(&from, &to).await;
        }
    }

    let rotated = path.with_extension("jsonl.1");
    let _ = tokio::fs::rename(path, &rotated).await;

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::session::SessionMetadata;

    fn test_metadata() -> SessionMetadata {
        SessionMetadata {
            name: None,
            provider: "anthropic".into(),
            model: "claude-sonnet-4".into(),
            channel: "cli".into(),
            channel_id: None,
        }
    }

    fn test_session(dir: &Path) -> Session {
        Session::new(dir.to_path_buf(), "system prompt".into(), test_metadata())
    }

    #[tokio::test]
    async fn save_and_load_round_trip() {
        let dir = tempfile::tempdir().unwrap();
        let store = JsonlSessionStore::new(dir.path().to_path_buf());

        let mut session = test_session(dir.path());
        session.history.push_user("hello");

        store.save(&session).await.unwrap();
        let loaded = store.load(&session.id).await.unwrap().unwrap();

        assert_eq!(loaded.id, session.id);
        assert_eq!(loaded.history.system_prompt(), "system prompt");
        assert_eq!(loaded.history.message_count(), 1);
        assert_eq!(loaded.history.messages()[0].text_content(), "hello");
        assert_eq!(loaded.state, SessionState::Sleeping);
    }

    #[tokio::test]
    async fn load_nonexistent_returns_none() {
        let dir = tempfile::tempdir().unwrap();
        let store = JsonlSessionStore::new(dir.path().to_path_buf());
        let result = store.load("nonexistent").await.unwrap();
        assert!(result.is_none());
    }

    #[tokio::test]
    async fn list_returns_saved_sessions() {
        let dir = tempfile::tempdir().unwrap();
        let store = JsonlSessionStore::new(dir.path().to_path_buf());

        let s1 = test_session(dir.path());
        let s2 = test_session(dir.path());
        store.save(&s1).await.unwrap();
        store.save(&s2).await.unwrap();

        let list = store.list().await.unwrap();
        assert_eq!(list.len(), 2);
    }

    #[tokio::test]
    async fn list_empty_dir_returns_empty() {
        let dir = tempfile::tempdir().unwrap();
        let store = JsonlSessionStore::new(dir.path().join("nonexistent"));
        let list = store.list().await.unwrap();
        assert!(list.is_empty());
    }

    #[tokio::test]
    async fn delete_removes_session() {
        let dir = tempfile::tempdir().unwrap();
        let store = JsonlSessionStore::new(dir.path().to_path_buf());

        let session = test_session(dir.path());
        store.save(&session).await.unwrap();
        assert!(store.load(&session.id).await.unwrap().is_some());

        store.delete(&session.id).await.unwrap();
        assert!(store.load(&session.id).await.unwrap().is_none());
    }

    #[tokio::test]
    async fn delete_nonexistent_is_ok() {
        let dir = tempfile::tempdir().unwrap();
        let store = JsonlSessionStore::new(dir.path().to_path_buf());
        store.delete("no-such-session").await.unwrap();
    }

    #[tokio::test]
    async fn append_message_adds_to_file() {
        let dir = tempfile::tempdir().unwrap();
        let store = JsonlSessionStore::new(dir.path().to_path_buf());

        let session = test_session(dir.path());
        store.save(&session).await.unwrap();

        let msg = ConversationMessage::user("appended msg");
        store.append_message(&session.id, &msg).await.unwrap();

        let loaded = store.load(&session.id).await.unwrap().unwrap();
        assert_eq!(loaded.history.message_count(), 1);
    }

    #[tokio::test]
    async fn save_with_multiple_messages() {
        let dir = tempfile::tempdir().unwrap();
        let store = JsonlSessionStore::new(dir.path().to_path_buf());

        let mut session = test_session(dir.path());
        session.history.push_user("q1");
        session.history.push_assistant(
            vec![crate::types::ContentBlock::Text { text: "a1".into() }],
            None,
        );
        session.history.push_user("q2");

        store.save(&session).await.unwrap();
        let loaded = store.load(&session.id).await.unwrap().unwrap();

        assert_eq!(loaded.history.message_count(), 3);
        assert_eq!(loaded.history.messages()[0].text_content(), "q1");
        assert_eq!(loaded.history.messages()[1].text_content(), "a1");
        assert_eq!(loaded.history.messages()[2].text_content(), "q2");
    }

    #[tokio::test]
    async fn save_preserves_fork_info() {
        let dir = tempfile::tempdir().unwrap();
        let store = JsonlSessionStore::new(dir.path().to_path_buf());

        let parent = test_session(dir.path());
        let forked = parent.fork(Some("experiment".into()));
        store.save(&forked).await.unwrap();

        let loaded = store.load(&forked.id).await.unwrap().unwrap();
        let fi = loaded.fork_info.unwrap();
        assert_eq!(fi.parent_session_id, parent.id);
        assert_eq!(fi.branch_name.as_deref(), Some("experiment"));
    }

    #[tokio::test]
    async fn save_creates_directory_layout() {
        let dir = tempfile::tempdir().unwrap();
        let store = JsonlSessionStore::new(dir.path().to_path_buf());

        let session = test_session(dir.path());
        store.save(&session).await.unwrap();

        let session_dir = dir.path().join(&session.id);
        assert!(session_dir.is_dir(), "session dir must exist");
        assert!(
            session_dir.join("session.jsonl").is_file(),
            "session.jsonl must exist"
        );
        assert!(
            session_dir.join("artifacts").is_dir(),
            "artifacts/ must exist"
        );
    }

    #[tokio::test]
    async fn artifacts_dir_is_isolated() {
        let dir = tempfile::tempdir().unwrap();
        let store = JsonlSessionStore::new(dir.path().to_path_buf());

        let s1 = test_session(dir.path());
        let s2 = test_session(dir.path());
        store.save(&s1).await.unwrap();
        store.save(&s2).await.unwrap();

        let a1 = store.artifacts_dir(&s1.id);
        let a2 = store.artifacts_dir(&s2.id);
        assert_ne!(a1, a2, "each session must have its own artifacts dir");

        std::fs::write(a1.join("file.txt"), "session1").unwrap();
        assert!(!a2.join("file.txt").exists(), "artifacts must be isolated");
    }

    #[tokio::test]
    async fn delete_removes_entire_directory() {
        let dir = tempfile::tempdir().unwrap();
        let store = JsonlSessionStore::new(dir.path().to_path_buf());

        let session = test_session(dir.path());
        store.save(&session).await.unwrap();

        let a = store.artifacts_dir(&session.id);
        std::fs::write(a.join("data.csv"), "1,2,3").unwrap();
        assert!(a.join("data.csv").exists());

        store.delete(&session.id).await.unwrap();
        assert!(
            !dir.path().join(&session.id).exists(),
            "session dir must be gone"
        );
    }

    #[tokio::test]
    async fn migrate_legacy_flat_file() {
        let dir = tempfile::tempdir().unwrap();
        let store = JsonlSessionStore::new(dir.path().to_path_buf());

        // Create a legacy flat file manually
        let session = test_session(dir.path());
        let legacy_path = dir.path().join(format!("{}.jsonl", session.id));
        let meta = serde_json::json!({
            "type": "session_meta",
            "session_id": session.id,
            "created_at": session.created_at,
            "updated_at": session.updated_at,
            "workspace": session.workspace,
            "state": "idle",
            "metadata": { "provider": "anthropic", "model": "claude-sonnet-4", "channel": "cli" },
            "system_prompt": "system prompt",
        });
        std::fs::write(&legacy_path, serde_json::to_string(&meta).unwrap() + "\n").unwrap();
        assert!(legacy_path.exists(), "legacy file should exist before load");

        let loaded = store.load(&session.id).await.unwrap();
        assert!(loaded.is_some(), "should load from legacy file");
        assert!(!legacy_path.exists(), "legacy file should be migrated away");
        assert!(
            dir.path().join(&session.id).join("session.jsonl").exists(),
            "new layout should exist"
        );
    }

    #[tokio::test]
    async fn session_root_returns_correct_path() {
        let dir = tempfile::tempdir().unwrap();
        let store = JsonlSessionStore::new(dir.path().to_path_buf());
        let root = store.session_root("abc-123");
        assert_eq!(root, dir.path().join("abc-123"));
    }

    #[tokio::test]
    async fn image_blocks_externalize_to_artifacts_dir() {
        let dir = tempfile::tempdir().unwrap();
        let store = JsonlSessionStore::new(dir.path().to_path_buf());

        // 4-byte fake "image" — content doesn't matter, only the round-trip.
        let bytes = vec![0xDE, 0xAD, 0xBE, 0xEF];
        let b64 = base64::engine::general_purpose::STANDARD.encode(&bytes);

        let mut session = test_session(dir.path());
        session.history.push_raw(ConversationMessage {
            role: crate::types::Role::User,
            blocks: vec![
                crate::types::ContentBlock::Image {
                    mime: "image/png".into(),
                    data_base64: b64.clone(),
                    detail: None,
                },
                crate::types::ContentBlock::Text {
                    text: "describe".into(),
                },
            ],
            timestamp: chrono::Utc::now(),
            usage: None,
        });

        store.save(&session).await.unwrap();

        // session.jsonl must NOT contain the inline base64 — that's the whole
        // point of externalization (otherwise we're back to JSONL bloat).
        let raw =
            std::fs::read_to_string(dir.path().join(&session.id).join("session.jsonl")).unwrap();
        assert!(
            !raw.contains(&b64),
            "inline base64 must be stripped from session.jsonl"
        );
        assert!(
            raw.contains(IMAGE_REF_PREFIX),
            "sentinel marker must be present"
        );

        // The artifact file itself must exist on disk.
        let artifacts = store.artifacts_dir(&session.id);
        let mut any_artifact = false;
        for entry in std::fs::read_dir(&artifacts).unwrap() {
            let p = entry.unwrap().path();
            if p.file_name().unwrap().to_string_lossy().starts_with("img_") {
                any_artifact = true;
                assert_eq!(std::fs::read(&p).unwrap(), bytes);
            }
        }
        assert!(any_artifact, "an externalized image artifact must exist");

        // Round-trip: load must reconstruct the original Image block byte-for-byte.
        let loaded = store.load(&session.id).await.unwrap().unwrap();
        let msgs = loaded.history.messages();
        assert_eq!(msgs.len(), 1);
        match &msgs[0].blocks[0] {
            crate::types::ContentBlock::Image {
                mime, data_base64, ..
            } => {
                assert_eq!(mime, "image/png");
                assert_eq!(data_base64, &b64);
            }
            other => panic!("expected Image block after intern, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn gc_orphan_artifacts_removes_unreferenced_files() {
        let dir = tempfile::tempdir().unwrap();
        let store = JsonlSessionStore::new(dir.path().to_path_buf());

        let bytes = vec![0xAB, 0xCD];
        let b64 = base64::engine::general_purpose::STANDARD.encode(&bytes);

        let mut session = test_session(dir.path());
        session.history.push_raw(ConversationMessage {
            role: crate::types::Role::User,
            blocks: vec![crate::types::ContentBlock::Image {
                mime: "image/png".into(),
                data_base64: b64,
                detail: None,
            }],
            timestamp: chrono::Utc::now(),
            usage: None,
        });
        store.save(&session).await.unwrap();

        let artifacts_dir = store.artifacts_dir(&session.id);
        // Drop two extra orphan files that are NOT referenced by session.jsonl —
        // mimics the state after a /clear or compact() that dropped their messages.
        std::fs::write(artifacts_dir.join("img_dead0000deadbeef.png"), b"x").unwrap();
        std::fs::write(artifacts_dir.join("img_dead0000cafebabe.png"), b"y").unwrap();
        // And a non-img file that must be left untouched.
        std::fs::write(artifacts_dir.join("user_doc.txt"), "keep").unwrap();

        let removed = store.gc_orphan_image_artifacts(&session.id).await.unwrap();
        assert_eq!(removed, 2, "two orphan img_* files must be removed");

        // The referenced artifact and the user doc must survive.
        let mut surviving: Vec<String> = std::fs::read_dir(&artifacts_dir)
            .unwrap()
            .map(|e| e.unwrap().file_name().to_string_lossy().to_string())
            .collect();
        surviving.sort();
        assert!(surviving.iter().any(|n| n == "user_doc.txt"));
        assert!(
            surviving
                .iter()
                .any(|n| n.starts_with("img_") && !n.contains("dead")),
            "the live referenced artifact must remain: {surviving:?}"
        );
    }

    #[tokio::test]
    async fn gc_old_artifacts_removes_files_past_cutoff() {
        use std::time::{Duration, SystemTime};
        let dir = tempfile::tempdir().unwrap();
        let store = JsonlSessionStore::new(dir.path().to_path_buf());
        let session_id = "s-old";
        let artifacts_dir = store.artifacts_dir(session_id);
        std::fs::create_dir_all(&artifacts_dir).unwrap();

        // Create three img_* files. Backdate two to >1 day old; leave one
        // fresh. A non-img file must never be touched.
        let old_a = artifacts_dir.join("img_old0000aaaaaaaa.png");
        let old_b = artifacts_dir.join("img_old0000bbbbbbbb.jpg");
        let fresh = artifacts_dir.join("img_fresh0000cccccccc.png");
        let keep_non_img = artifacts_dir.join("user_doc.txt");
        std::fs::write(&old_a, b"x").unwrap();
        std::fs::write(&old_b, b"y").unwrap();
        std::fs::write(&fresh, b"z").unwrap();
        std::fs::write(&keep_non_img, b"keep").unwrap();

        // Set mtime on old files to 2 days ago.
        let two_days_ago = SystemTime::now() - Duration::from_secs(2 * 24 * 3600);
        let ft = filetime::FileTime::from_system_time(two_days_ago);
        filetime::set_file_mtime(&old_a, ft).unwrap();
        filetime::set_file_mtime(&old_b, ft).unwrap();

        // cutoff = 1 day → expect the two backdated files to go.
        let removed = store
            .gc_old_image_artifacts(session_id, 24 * 3600)
            .await
            .unwrap();
        assert_eq!(removed, 2, "two old img_* files must be removed");

        assert!(!old_a.exists());
        assert!(!old_b.exists());
        assert!(fresh.exists(), "fresh img artifact must survive");
        assert!(keep_non_img.exists(), "non-img files must never be touched");
    }

    #[tokio::test]
    async fn gc_old_artifacts_missing_dir_returns_zero() {
        let dir = tempfile::tempdir().unwrap();
        let store = JsonlSessionStore::new(dir.path().to_path_buf());
        let removed = store
            .gc_old_image_artifacts("nonexistent-session", 60)
            .await
            .unwrap();
        assert_eq!(removed, 0);
    }

    #[tokio::test]
    async fn duplicate_image_dedupes_on_disk() {
        // Same bytes attached twice must produce only one artifact file —
        // hash-based filenames give us free deduplication.
        let dir = tempfile::tempdir().unwrap();
        let store = JsonlSessionStore::new(dir.path().to_path_buf());

        let bytes = vec![1u8, 2, 3, 4, 5];
        let b64 = base64::engine::general_purpose::STANDARD.encode(&bytes);

        let mut session = test_session(dir.path());
        for _ in 0..3 {
            session.history.push_raw(ConversationMessage {
                role: crate::types::Role::User,
                blocks: vec![crate::types::ContentBlock::Image {
                    mime: "image/jpeg".into(),
                    data_base64: b64.clone(),
                    detail: None,
                }],
                timestamp: chrono::Utc::now(),
                usage: None,
            });
        }
        store.save(&session).await.unwrap();

        let count = std::fs::read_dir(store.artifacts_dir(&session.id))
            .unwrap()
            .filter(|e| {
                e.as_ref()
                    .unwrap()
                    .file_name()
                    .to_string_lossy()
                    .starts_with("img_")
            })
            .count();
        assert_eq!(count, 1, "identical images must dedupe to one artifact");
    }

    #[tokio::test]
    async fn compaction_summary_survives_save_load() {
        let dir = tempfile::tempdir().unwrap();
        let store = JsonlSessionStore::new(dir.path().to_path_buf());

        let mut session = test_session(dir.path());
        session.history.push_user("q1");
        session.history.push_assistant(
            vec![crate::types::ContentBlock::Text { text: "a1".into() }],
            None,
        );
        session.history.push_raw(ConversationMessage::system(
            "<summary>compacted context</summary>",
        ));
        session.history.push_user("q2");

        store.save(&session).await.unwrap();
        let loaded = store.load(&session.id).await.unwrap().unwrap();

        assert_eq!(loaded.history.system_prompt(), "system prompt");
        assert_eq!(loaded.history.message_count(), 4);
        let msgs = loaded.history.messages();
        assert_eq!(msgs[0].text_content(), "q1");
        assert_eq!(msgs[1].text_content(), "a1");
        assert_eq!(msgs[2].role, crate::types::Role::System);
        assert!(msgs[2].text_content().contains("compacted context"));
        assert_eq!(msgs[3].text_content(), "q2");
    }
}
