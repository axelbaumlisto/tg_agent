use std::path::PathBuf;

use async_trait::async_trait;

use crate::error::Result;
use crate::types::ConversationMessage;

use super::{Session, SessionSummary};

#[async_trait]
pub trait SessionStore: Send + Sync {
    async fn save(&self, session: &Session) -> Result<()>;
    async fn load(&self, session_id: &str) -> Result<Option<Session>>;
    async fn append_message(&self, session_id: &str, msg: &ConversationMessage) -> Result<()>;
    async fn list(&self) -> Result<Vec<SessionSummary>>;
    async fn delete(&self, session_id: &str) -> Result<()>;

    /// Mark a session as having an active turn (persists to disk).
    /// On crash recovery, sessions still marked active were interrupted.
    async fn mark_active(&self, session_id: &str) -> Result<()>;

    /// Mark a session as idle (turn finished). Clears the active marker.
    async fn mark_idle(&self, session_id: &str) -> Result<()>;

    /// Return session IDs that were active when the process last crashed.
    /// Clears the markers after reading (one-shot).
    async fn drain_interrupted(&self) -> Vec<String>;

    /// Returns the isolated artifacts directory for a session's tools to use as cwd.
    fn artifacts_dir(&self, session_id: &str) -> PathBuf;

    /// Returns the root directory for a session (containing session.jsonl, artifacts/, prompt.md, skills/).
    fn session_root(&self, session_id: &str) -> PathBuf;

    /// Sweep `<session>/artifacts/img_*` and remove any file no longer
    /// referenced by a sentinel marker in `session.jsonl`. Returns the number
    /// of files actually deleted. Default impl is a no-op so legacy/in-memory
    /// stores stay free; the JSONL store overrides with a real walk.
    ///
    /// Why a separate method instead of doing it inside `save`: GC needs to
    /// happen *after* compaction has potentially dropped messages, and `save`
    /// may be called many times per turn. A periodic sweep (every N saves, or
    /// on shutdown) keeps it cheap.
    async fn gc_orphan_image_artifacts(&self, _session_id: &str) -> Result<usize> {
        Ok(0)
    }

    /// Sweep `<session>/artifacts/img_*` and remove files whose
    /// modification time is older than `max_age_secs`, regardless of
    /// whether they're still referenced in `session.jsonl`. Intended for
    /// operator-driven retention (e.g. cron job) — never called from the
    /// hot path. Default impl is a no-op.
    ///
    /// Combining reachability-GC with age-GC would be a footgun: a long
    /// chat referencing a month-old image should *not* lose that image
    /// unless the operator explicitly requests it. This method is the
    /// explicit knob.
    async fn gc_old_image_artifacts(&self, _session_id: &str, _max_age_secs: u64) -> Result<usize> {
        Ok(0)
    }
}
