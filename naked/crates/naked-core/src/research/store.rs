//! On-disk research store. Each research lives in its own directory:
//!
//! ```text
//! $NAKED_HOME/research/<id>/
//!   spec.json       — atomic-write (tmp + rename) of the ResearchSpec
//!   findings.jsonl  — one `Finding` per line, append-only
//!   runs.jsonl      — one `RunRecord` per line, append-only
//!   cursor.json     — small JSON blob, atomic-write
//!   report.md       — regenerated each run from findings
//! ```
//!
//! The trait lets tests swap in a RAM-only implementation without touching the
//! disk. Production uses `FsResearchStore`.

use std::collections::HashSet;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use async_trait::async_trait;
use chrono::Utc;
use tokio::fs;
use tokio::io::AsyncWriteExt;
use tokio::sync::RwLock;

use crate::error::{AgentError, Result};

use super::inflight::Inflight;
use super::spec::{Cursor, Finding, ResearchSpec, RunRecord};

/// Resolve the root directory for research state.
///
/// Order: `$NAKED_HOME/research` → `~/.naked/research`. The runtime guarantees
/// this directory exists on first write; reads of a missing directory surface
/// as "no research found" rather than I/O errors.
pub fn research_root() -> PathBuf {
    if let Ok(v) = std::env::var("NAKED_HOME") {
        return PathBuf::from(v).join("research");
    }
    let home = std::env::var("HOME").unwrap_or_else(|_| ".".to_string());
    PathBuf::from(home).join(".naked").join("research")
}

/// Default destination for research run-completion log lines. Lives next
/// to per-spec dirs under `research/` so the durable `MEMORY.md` files
/// stay free of run-noise. Each line records `spec_id`, `topic`, `run_id`,
/// new/total finding counts, optional verification metrics, and a pointer
/// to the report on disk.
pub fn research_runlog_path() -> PathBuf {
    research_root().join("run_log.md")
}

// ── Sub-traits (ISP: Interface Segregation) ────────────────────────────────

/// CRUD for research specifications (topics).
#[async_trait]
pub trait SpecStore: Send + Sync {
    async fn create_spec(&self, spec: &ResearchSpec) -> Result<()>;
    async fn load_spec(&self, id: &str) -> Result<ResearchSpec>;
    async fn save_spec(&self, spec: &ResearchSpec) -> Result<()>;
    async fn list_specs(&self) -> Result<Vec<ResearchSpec>>;
    async fn delete_spec(&self, id: &str) -> Result<()>;
}

/// Storage for research findings (append-only, dedup by hash).
#[async_trait]
pub trait FindingStore: Send + Sync {
    /// Append a finding if it is not a duplicate. Returns `true` when stored,
    /// `false` when the `dedup_hash` was already present. MUST be atomic per
    /// research id — concurrent calls for the same id go through a mutex.
    async fn try_append_finding(&self, finding: &Finding) -> Result<bool>;

    /// Insert or update a finding. If a finding with the same `dedup_hash` already
    /// exists, it is replaced with the new data (atomic rewrite). Returns `true`
    /// when updated (i.e. the hash already existed), `false` when newly inserted.
    async fn upsert_finding(&self, finding: &Finding) -> Result<bool>;

    async fn list_findings(&self, id: &str, limit: Option<usize>) -> Result<Vec<Finding>>;
    async fn count_findings(&self, id: &str) -> Result<u32>;

    /// Remove findings whose `dedup_hash` is in the given set. Rewrites
    /// `findings.jsonl` atomically and updates the in-memory dedup cache.
    /// Returns the number of findings actually removed.
    async fn remove_findings_by_hash(&self, id: &str, hashes: &HashSet<String>) -> Result<u32>;
}

/// Runs and cursors.
#[async_trait]
pub trait RunStore: Send + Sync {
    async fn append_run(&self, run: &RunRecord) -> Result<()>;
    async fn list_runs(&self, id: &str, limit: Option<usize>) -> Result<Vec<RunRecord>>;
    async fn load_cursor(&self, id: &str) -> Result<Cursor>;
    async fn save_cursor(&self, id: &str, cursor: &Cursor) -> Result<()>;
}

/// Reports and agent briefs.
#[async_trait]
pub trait ReportStore: Send + Sync {
    async fn write_report(&self, id: &str, report: &str) -> Result<()>;
    async fn read_report(&self, id: &str) -> Result<Option<String>>;
    /// Filesystem path of the report file, when the backend is disk-backed.
    fn report_path(&self, _id: &str) -> Option<PathBuf> {
        None
    }
    async fn write_agent_brief(&self, id: &str, brief: &str) -> Result<()>;
    async fn read_agent_brief(&self, id: &str) -> Result<Option<String>>;
}

/// Inflight scheduler state-machine ledger.
#[async_trait]
pub trait InflightStore: Send + Sync {
    async fn save_inflight(&self, _id: &str, _infl: &Inflight) -> Result<()> {
        Ok(())
    }
    async fn load_inflight(&self, _id: &str) -> Result<Option<Inflight>> {
        Ok(None)
    }
    async fn clear_inflight(&self, _id: &str) -> Result<()> {
        Ok(())
    }
    async fn list_nonterminal_inflight(&self) -> Result<Vec<Inflight>> {
        Ok(Vec::new())
    }
    async fn purge_terminal_inflight(
        &self,
        _now: chrono::DateTime<chrono::Utc>,
        _retention: chrono::Duration,
    ) -> Result<u32> {
        Ok(0)
    }
    /// Filesystem root for backends that have one.
    fn fs_root(&self) -> Option<&Path> {
        None
    }
}

/// Composite: RunStore + ReportStore + InflightStore.
pub trait ArtifactStore: RunStore + ReportStore + InflightStore {}
impl<T: RunStore + ReportStore + InflightStore> ArtifactStore for T {}

// ── Composite trait (backward compat) ──────────────────────────────────

/// Full research store = SpecStore + FindingStore + ArtifactStore.
/// Existing code can keep using `dyn ResearchStore` unchanged.
pub trait ResearchStore: SpecStore + FindingStore + ArtifactStore {}

/// Blanket impl: anything implementing all three sub-traits is a ResearchStore.
impl<T: SpecStore + FindingStore + ArtifactStore> ResearchStore for T {}

/// Filesystem-backed research store. Holds a per-id mutex to serialize the
/// read-modify-write dance on `findings.jsonl` (dedup set lookup, then append).
pub struct FsResearchStore {
    root: PathBuf,
    /// Coarse lock per research id. Kept as `Arc<RwLock<HashSet<hash>>>` so
    /// lookups are cheap once warm and the set survives across append calls
    /// for the duration of the process.
    locks: RwLock<std::collections::HashMap<String, Arc<RwLock<HashSet<String>>>>>,
}

impl FsResearchStore {
    pub fn new(root: PathBuf) -> Self {
        Self {
            root,
            locks: RwLock::new(std::collections::HashMap::new()),
        }
    }

    pub fn with_default_root() -> Self {
        Self::new(research_root())
    }

    pub fn root(&self) -> &Path {
        &self.root
    }

    fn dir(&self, id: &str) -> PathBuf {
        self.root.join(id)
    }

    async fn ensure_dir(&self, id: &str) -> Result<()> {
        let d = self.dir(id);
        fs::create_dir_all(&d).await?;
        Ok(())
    }

    /// Get (or lazily load) the dedup set for a research id. On first use we
    /// scan `findings.jsonl` once — subsequent calls reuse the in-memory set.
    async fn dedup_set(&self, id: &str) -> Result<Arc<RwLock<HashSet<String>>>> {
        {
            let guard = self.locks.read().await;
            if let Some(set) = guard.get(id) {
                return Ok(set.clone());
            }
        }
        let mut guard = self.locks.write().await;
        // Re-check after upgrading — another task may have installed it.
        if let Some(set) = guard.get(id) {
            return Ok(set.clone());
        }
        let mut set = HashSet::new();
        let path = self.dir(id).join("findings.jsonl");
        if path.exists() {
            let content = fs::read_to_string(&path).await?;
            for line in content.lines() {
                let line = line.trim();
                if line.is_empty() {
                    continue;
                }
                if let Ok(f) = serde_json::from_str::<Finding>(line) {
                    set.insert(f.dedup_hash);
                }
            }
        }
        let arc = Arc::new(RwLock::new(set));
        guard.insert(id.to_string(), arc.clone());
        Ok(arc)
    }

    async fn atomic_write(&self, path: &Path, content: &[u8]) -> Result<()> {
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent).await?;
        }
        let tmp = path.with_extension("tmp");
        fs::write(&tmp, content).await?;
        fs::rename(&tmp, path).await?;
        Ok(())
    }

    async fn append_jsonl<T: serde::Serialize>(&self, path: &Path, value: &T) -> Result<()> {
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent).await?;
        }
        let mut line = serde_json::to_string(value).map_err(|e| {
            AgentError::ProviderTyped(crate::provider::error::ProviderError::Serialize {
                context: "jsonl".into(),
                source: e.to_string(),
            })
        })?;
        line.push('\n');
        let mut file = fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(path)
            .await?;
        file.write_all(line.as_bytes()).await?;
        file.flush().await?;
        Ok(())
    }

    async fn read_jsonl<T: serde::de::DeserializeOwned>(
        &self,
        path: &Path,
        limit: Option<usize>,
    ) -> Result<Vec<T>> {
        if !path.exists() {
            return Ok(Vec::new());
        }
        let content = fs::read_to_string(path).await?;
        let mut out = Vec::new();
        for line in content.lines() {
            let line = line.trim();
            if line.is_empty() {
                continue;
            }
            match serde_json::from_str::<T>(line) {
                Ok(v) => out.push(v),
                Err(e) => tracing::warn!("research store: skipping malformed jsonl row: {e}"),
            }
        }
        if let Some(n) = limit {
            let start = out.len().saturating_sub(n);
            out = out.split_off(start);
        }
        Ok(out)
    }
}

#[async_trait]
impl SpecStore for FsResearchStore {
    async fn create_spec(&self, spec: &ResearchSpec) -> Result<()> {
        self.ensure_dir(&spec.id).await?;
        let path = self.dir(&spec.id).join("spec.json");
        if path.exists() {
            return Err(AgentError::Config(format!(
                "research '{}' already exists",
                spec.id
            )));
        }
        let json = serde_json::to_vec_pretty(spec).map_err(|e| {
            AgentError::ProviderTyped(crate::provider::error::ProviderError::Serialize {
                context: "spec".into(),
                source: e.to_string(),
            })
        })?;
        self.atomic_write(&path, &json).await
    }

    async fn load_spec(&self, id: &str) -> Result<ResearchSpec> {
        let path = self.dir(id).join("spec.json");
        if !path.exists() {
            return Err(AgentError::Config(format!("research '{id}' not found")));
        }
        let data = fs::read_to_string(&path).await?;
        serde_json::from_str(&data)
            .map_err(|e| AgentError::Config(format!("bad spec for {id}: {e}")))
    }

    async fn save_spec(&self, spec: &ResearchSpec) -> Result<()> {
        self.ensure_dir(&spec.id).await?;
        let path = self.dir(&spec.id).join("spec.json");
        let json = serde_json::to_vec_pretty(spec).map_err(|e| {
            AgentError::ProviderTyped(crate::provider::error::ProviderError::Serialize {
                context: "spec".into(),
                source: e.to_string(),
            })
        })?;
        self.atomic_write(&path, &json).await
    }

    async fn list_specs(&self) -> Result<Vec<ResearchSpec>> {
        let mut out = Vec::new();
        if !self.root.exists() {
            return Ok(out);
        }
        let mut dir = fs::read_dir(&self.root).await?;
        while let Some(entry) = dir.next_entry().await? {
            if !entry.file_type().await?.is_dir() {
                continue;
            }
            let spec_path = entry.path().join("spec.json");
            if !spec_path.exists() {
                continue;
            }
            if let Ok(data) = fs::read_to_string(&spec_path).await
                && let Ok(spec) = serde_json::from_str::<ResearchSpec>(&data)
            {
                out.push(spec);
            }
        }
        out.sort_by_key(|r| std::cmp::Reverse(r.created_at));
        Ok(out)
    }

    async fn delete_spec(&self, id: &str) -> Result<()> {
        let dir = self.dir(id);
        if dir.exists() {
            fs::remove_dir_all(&dir).await?;
        }
        {
            let mut guard = self.locks.write().await;
            guard.remove(id);
        }
        Ok(())
    }
}

#[async_trait]
impl FindingStore for FsResearchStore {
    async fn try_append_finding(&self, finding: &Finding) -> Result<bool> {
        self.ensure_dir(&finding.research_id).await?;
        let set = self.dedup_set(&finding.research_id).await?;
        {
            let read = set.read().await;
            if read.contains(&finding.dedup_hash) {
                return Ok(false);
            }
        }
        // Secondary dedup pass: same physical listing reached via a different
        // canonical URL. We scan `findings.jsonl` once, looking for either
        // (a) the same `host_path_hash` (same site URL minus query string), or
        // (b) the same `content_hash` (same prose, possibly crossposted to a
        //     completely different domain).
        // Both cases are rejected outright — we keep the first arrival and let
        // the gatekeeper decide later whether the user wants the duplicate.
        // We also keep the existing soft fuzzy-title warning as a tertiary
        // signal when neither hash matches but the titles look similar.
        let path = self.dir(&finding.research_id).join("findings.jsonl");
        if path.exists()
            && let Ok(content) = tokio::fs::read_to_string(&path).await
        {
            let mut warned_title = false;
            for line in content.lines() {
                let existing: Finding = match serde_json::from_str(line) {
                    Ok(f) => f,
                    Err(_) => continue,
                };
                if existing.dedup_hash == finding.dedup_hash {
                    continue;
                }
                if !finding.host_path_hash.is_empty()
                    && existing.host_path_hash == finding.host_path_hash
                {
                    tracing::info!(
                        research_id = %finding.research_id,
                        new_url = %finding.url,
                        existing_url = %existing.url,
                        "host_path_hash dedup: same host+path, only query differs — rejecting"
                    );
                    return Ok(false);
                }
                if !finding.content_hash.is_empty() && existing.content_hash == finding.content_hash
                {
                    tracing::info!(
                        research_id = %finding.research_id,
                        new_url = %finding.url,
                        existing_url = %existing.url,
                        "content_hash dedup: same normalized excerpt across URLs — rejecting"
                    );
                    return Ok(false);
                }
                if !warned_title
                    && let (Some(new_title), Some(existing_title)) =
                        (finding.title.as_deref(), existing.title.as_deref())
                    && !new_title.trim().is_empty()
                    && crate::research::spec::titles_are_similar(new_title, existing_title)
                {
                    tracing::warn!(
                        research_id = %finding.research_id,
                        new_url = %finding.url,
                        existing_url = %existing.url,
                        new_title = new_title,
                        existing_title,
                        "fuzzy-dedup: near-duplicate title detected \
                         (different URL, similar title) — gatekeeper \
                         should decide whether to merge"
                    );
                    warned_title = true;
                }
            }
        }
        let mut write = set.write().await;
        // Double-check after acquiring the write lock.
        if !write.insert(finding.dedup_hash.clone()) {
            return Ok(false);
        }
        self.append_jsonl(&path, finding).await?;
        Ok(true)
    }

    async fn upsert_finding(&self, finding: &Finding) -> Result<bool> {
        self.ensure_dir(&finding.research_id).await?;
        let set = self.dedup_set(&finding.research_id).await?;
        let guard = set.read().await;
        let existed = guard.contains(&finding.dedup_hash);
        drop(guard);

        if !existed {
            // New finding — just append
            let mut write = set.write().await;
            write.insert(finding.dedup_hash.clone());
            drop(write);
            let path = self.dir(&finding.research_id).join("findings.jsonl");
            self.append_jsonl(&path, finding).await?;
            return Ok(false);
        }

        // Existing finding — rewrite the file, replacing the old entry
        let path = self.dir(&finding.research_id).join("findings.jsonl");
        let content = fs::read_to_string(&path).await.unwrap_or_default();
        let mut lines = Vec::new();
        for line in content.lines() {
            if line.trim().is_empty() {
                continue;
            }
            if let Ok(f) = serde_json::from_str::<Finding>(line)
                && f.dedup_hash == finding.dedup_hash
            {
                let new_line = serde_json::to_string(finding).map_err(|e| {
                    AgentError::ProviderTyped(crate::provider::error::ProviderError::Serialize {
                        context: "finding".into(),
                        source: e.to_string(),
                    })
                })?;
                lines.push(new_line);
                continue;
            }
            lines.push(line.to_string());
        }
        let new_content = lines.join("\n") + if lines.is_empty() { "" } else { "\n" };
        self.atomic_write(&path, new_content.as_bytes()).await?;
        Ok(true)
    }

    async fn list_findings(&self, id: &str, limit: Option<usize>) -> Result<Vec<Finding>> {
        let path = self.dir(id).join("findings.jsonl");
        self.read_jsonl(&path, limit).await
    }

    async fn count_findings(&self, id: &str) -> Result<u32> {
        let path = self.dir(id).join("findings.jsonl");
        if !path.exists() {
            return Ok(0);
        }
        let content = fs::read_to_string(&path).await?;
        Ok(content.lines().filter(|l| !l.trim().is_empty()).count() as u32)
    }

    async fn remove_findings_by_hash(&self, id: &str, hashes: &HashSet<String>) -> Result<u32> {
        if hashes.is_empty() {
            return Ok(0);
        }
        let path = self.dir(id).join("findings.jsonl");
        if !path.exists() {
            return Ok(0);
        }
        let content = fs::read_to_string(&path).await?;
        let mut kept = Vec::new();
        let mut removed = 0u32;
        for line in content.lines() {
            if line.trim().is_empty() {
                continue;
            }
            if let Ok(f) = serde_json::from_str::<Finding>(line)
                && hashes.contains(&f.dedup_hash)
            {
                removed += 1;
                continue;
            }
            kept.push(line);
        }
        if removed > 0 {
            let new_content = kept.join("\n") + if kept.is_empty() { "" } else { "\n" };
            self.atomic_write(&path, new_content.as_bytes()).await?;
            // Rebuild dedup set
            let set = self.dedup_set(id).await?;
            let mut guard = set.write().await;
            for h in hashes {
                guard.remove(h);
            }
        }
        Ok(removed)
    }
}

#[async_trait]
impl RunStore for FsResearchStore {
    async fn append_run(&self, run: &RunRecord) -> Result<()> {
        self.ensure_dir(&run.spec_id).await?;
        let path = self.dir(&run.spec_id).join("runs.jsonl");
        self.append_jsonl(&path, run).await
    }

    async fn list_runs(&self, id: &str, limit: Option<usize>) -> Result<Vec<RunRecord>> {
        let path = self.dir(id).join("runs.jsonl");
        self.read_jsonl(&path, limit).await
    }

    async fn load_cursor(&self, id: &str) -> Result<Cursor> {
        let path = self.dir(id).join("cursor.json");
        if !path.exists() {
            return Ok(Cursor::default());
        }
        let data = fs::read_to_string(&path).await?;
        serde_json::from_str(&data)
            .map_err(|e| AgentError::Config(format!("bad cursor for {id}: {e}")))
    }

    async fn save_cursor(&self, id: &str, cursor: &Cursor) -> Result<()> {
        self.ensure_dir(id).await?;
        let mut c = cursor.clone();
        c.updated_at = Some(Utc::now());
        let json = serde_json::to_vec_pretty(&c).map_err(|e| {
            AgentError::ProviderTyped(crate::provider::error::ProviderError::Serialize {
                context: "cursor".into(),
                source: e.to_string(),
            })
        })?;
        let path = self.dir(id).join("cursor.json");
        self.atomic_write(&path, &json).await
    }
}

#[async_trait]
impl ReportStore for FsResearchStore {
    async fn write_report(&self, id: &str, report: &str) -> Result<()> {
        self.ensure_dir(id).await?;
        let path = self.dir(id).join("report.md");
        self.atomic_write(&path, report.as_bytes()).await
    }

    async fn read_report(&self, id: &str) -> Result<Option<String>> {
        let path = self.dir(id).join("report.md");
        if !path.exists() {
            return Ok(None);
        }
        Ok(Some(fs::read_to_string(&path).await?))
    }

    fn report_path(&self, id: &str) -> Option<PathBuf> {
        Some(self.dir(id).join("report.md"))
    }

    async fn write_agent_brief(&self, id: &str, brief: &str) -> Result<()> {
        self.ensure_dir(id).await?;
        let path = self.dir(id).join("agent_brief.md");
        self.atomic_write(&path, brief.as_bytes()).await
    }

    async fn read_agent_brief(&self, id: &str) -> Result<Option<String>> {
        let path = self.dir(id).join("agent_brief.md");
        if !path.exists() {
            return Ok(None);
        }
        Ok(Some(fs::read_to_string(&path).await?))
    }
}

#[async_trait]
impl InflightStore for FsResearchStore {
    async fn save_inflight(&self, id: &str, infl: &Inflight) -> Result<()> {
        self.ensure_dir(id).await?;
        let path = self.dir(id).join("inflight.json");
        let json = serde_json::to_vec_pretty(infl).map_err(|e| {
            AgentError::ProviderTyped(crate::provider::error::ProviderError::Serialize {
                context: "inflight".into(),
                source: e.to_string(),
            })
        })?;
        self.atomic_write(&path, &json).await
    }

    async fn load_inflight(&self, id: &str) -> Result<Option<Inflight>> {
        let path = self.dir(id).join("inflight.json");
        if !path.exists() {
            return Ok(None);
        }
        let data = fs::read_to_string(&path).await?;
        match serde_json::from_str::<Inflight>(&data) {
            Ok(v) => Ok(Some(v)),
            Err(e) => {
                tracing::warn!(spec = %id, "ignoring malformed inflight.json: {e}");
                Ok(None)
            }
        }
    }

    async fn clear_inflight(&self, id: &str) -> Result<()> {
        let path = self.dir(id).join("inflight.json");
        if path.exists() {
            fs::remove_file(&path).await?;
        }
        Ok(())
    }

    async fn list_nonterminal_inflight(&self) -> Result<Vec<Inflight>> {
        let mut out = Vec::new();
        if !self.root.exists() {
            return Ok(out);
        }
        let mut dir = fs::read_dir(&self.root).await?;
        while let Some(entry) = dir.next_entry().await? {
            if !entry.file_type().await?.is_dir() {
                continue;
            }
            let inflight_path = entry.path().join("inflight.json");
            if !inflight_path.exists() {
                continue;
            }
            let Ok(data) = fs::read_to_string(&inflight_path).await else {
                continue;
            };
            match serde_json::from_str::<Inflight>(&data) {
                Ok(v) if !v.state.is_terminal() => out.push(v),
                Ok(_) => {}
                Err(e) => tracing::warn!(
                    path = %inflight_path.display(),
                    "ignoring malformed inflight.json: {e}"
                ),
            }
        }
        Ok(out)
    }

    async fn purge_terminal_inflight(
        &self,
        now: chrono::DateTime<chrono::Utc>,
        retention: chrono::Duration,
    ) -> Result<u32> {
        if !self.root.exists() {
            return Ok(0);
        }
        let mut removed: u32 = 0;
        let mut dir = fs::read_dir(&self.root).await?;
        while let Some(entry) = dir.next_entry().await? {
            if !entry.file_type().await?.is_dir() {
                continue;
            }
            let inflight_path = entry.path().join("inflight.json");
            if !inflight_path.exists() {
                continue;
            }
            let Ok(data) = fs::read_to_string(&inflight_path).await else {
                continue;
            };
            let infl = match serde_json::from_str::<Inflight>(&data) {
                Ok(v) => v,
                Err(e) => {
                    tracing::warn!(
                        path = %inflight_path.display(),
                        "ignoring malformed inflight.json during purge: {e}"
                    );
                    continue;
                }
            };
            if !infl.state.is_terminal() {
                continue;
            }
            let Some(finished_at) = infl.finished_at else {
                continue;
            };
            if now - finished_at <= retention {
                continue;
            }
            match fs::remove_file(&inflight_path).await {
                Ok(()) => {
                    removed = removed.saturating_add(1);
                    tracing::debug!(
                        path = %inflight_path.display(),
                        spec = %infl.spec_id,
                        attempt = infl.attempt,
                        "purged stale terminal inflight record"
                    );
                }
                Err(e) => tracing::warn!(
                    path = %inflight_path.display(),
                    "failed to remove stale inflight: {e}"
                ),
            }
        }
        Ok(removed)
    }

    fn fs_root(&self) -> Option<&Path> {
        Some(&self.root)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::research::spec::{Finding, ResearchSpec, RunRecord, dedup_hash};
    use chrono::Utc;
    use tempfile::tempdir;

    fn make_spec(id: &str, topic: &str) -> ResearchSpec {
        ResearchSpec {
            id: id.to_string(),
            topic: topic.to_string(),
            sources: vec!["https://example.com".into()],
            interval_seconds: None,
            run_at: None,
            cron: None,
            task_timeout_seconds: None,
            session_id: None,
            chat_id: None,
            thread_id: None,
            provider: None,
            model: None,
            max_iterations: None,
            max_wall_seconds: None,
            created_at: Utc::now(),
            paused: false,
            pause_reason: None,
        }
    }

    fn make_finding(spec_id: &str, url: &str) -> Finding {
        use crate::research::spec::host_path_hash;
        Finding {
            id: uuid::Uuid::new_v4().simple().to_string(),
            research_id: spec_id.to_string(),
            run_id: "r1".to_string(),
            url: url.to_string(),
            title: Some("title".into()),
            excerpt: Some("excerpt".into()),
            price: None,
            listing_date: None,
            source_content: None,
            dedup_hash: dedup_hash(url),
            host_path_hash: host_path_hash(url),
            // Empty content_hash so URL-only dedup tests aren't accidentally
            // tripped by the new content-based dedup. Content-hash dedup has
            // its own dedicated test.
            content_hash: String::new(),
            seen_at: Utc::now(),
        }
    }

    #[tokio::test]
    async fn create_load_save_roundtrip() {
        let tmp = tempdir().unwrap();
        let store = FsResearchStore::new(tmp.path().to_path_buf());
        let mut spec = make_spec("id-1", "topic");
        store.create_spec(&spec).await.unwrap();
        let loaded = store.load_spec("id-1").await.unwrap();
        assert_eq!(loaded, spec);
        spec.paused = true;
        store.save_spec(&spec).await.unwrap();
        let loaded2 = store.load_spec("id-1").await.unwrap();
        assert!(loaded2.paused);
    }

    #[tokio::test]
    async fn create_rejects_duplicate() {
        let tmp = tempdir().unwrap();
        let store = FsResearchStore::new(tmp.path().to_path_buf());
        let spec = make_spec("id-dup", "topic");
        store.create_spec(&spec).await.unwrap();
        let err = store.create_spec(&spec).await;
        assert!(err.is_err());
    }

    #[tokio::test]
    async fn list_specs_sorts_newest_first() {
        let tmp = tempdir().unwrap();
        let store = FsResearchStore::new(tmp.path().to_path_buf());
        let mut a = make_spec("a", "A");
        a.created_at = Utc::now() - chrono::Duration::hours(2);
        let b = make_spec("b", "B"); // newer
        store.create_spec(&a).await.unwrap();
        store.create_spec(&b).await.unwrap();
        let list = store.list_specs().await.unwrap();
        assert_eq!(list.len(), 2);
        assert_eq!(list[0].id, "b");
        assert_eq!(list[1].id, "a");
    }

    #[tokio::test]
    async fn findings_dedup_on_canonical_url() {
        let tmp = tempdir().unwrap();
        let store = FsResearchStore::new(tmp.path().to_path_buf());
        let spec = make_spec("id-2", "t");
        store.create_spec(&spec).await.unwrap();

        let f1 = make_finding("id-2", "https://x.com/ad/42?utm_source=a");
        let f2 = make_finding("id-2", "https://x.com/ad/42?utm_source=b");
        let f3 = make_finding("id-2", "https://x.com/ad/43");

        assert!(store.try_append_finding(&f1).await.unwrap());
        assert!(
            !store.try_append_finding(&f2).await.unwrap(),
            "should dedup"
        );
        assert!(store.try_append_finding(&f3).await.unwrap());

        assert_eq!(store.count_findings("id-2").await.unwrap(), 2);
        let list = store.list_findings("id-2", None).await.unwrap();
        assert_eq!(list.len(), 2);
    }

    #[tokio::test]
    async fn host_path_hash_dedup_rejects_same_listing_with_extra_query_params() {
        // T7: a listing reached via different non-tracking query strings
        // (`?sort=newest` vs `?sort=oldest`) has DIFFERENT `dedup_hash`
        // values but the SAME `host_path_hash`. The store must reject the
        // second arrival on the secondary check.
        let tmp = tempdir().unwrap();
        let store = FsResearchStore::new(tmp.path().to_path_buf());
        let spec = make_spec("id-hp", "t");
        store.create_spec(&spec).await.unwrap();

        let f1 = make_finding("id-hp", "https://batdongsan.com.vn/ad/42?sort=newest");
        let f2 = make_finding(
            "id-hp",
            "https://batdongsan.com.vn/ad/42?sort=oldest&page=3",
        );
        // Sanity: they pass the URL-canon dedup (different canonical URLs).
        assert_ne!(
            f1.dedup_hash, f2.dedup_hash,
            "different non-tracking params → different canonical URLs"
        );
        assert_eq!(
            f1.host_path_hash, f2.host_path_hash,
            "but same host+path → same secondary key"
        );

        assert!(store.try_append_finding(&f1).await.unwrap());
        assert!(
            !store.try_append_finding(&f2).await.unwrap(),
            "host_path_hash dedup must reject the second arrival"
        );
        assert_eq!(store.count_findings("id-hp").await.unwrap(), 1);
    }

    #[tokio::test]
    async fn content_hash_dedup_rejects_crosspost_to_different_domain() {
        // T7: same prose republished at a completely different URL
        // (different host AND different path) — neither URL-canon nor
        // host-path matches, but content_hash does. The store must reject.
        use crate::research::spec::content_hash;
        let tmp = tempdir().unwrap();
        let store = FsResearchStore::new(tmp.path().to_path_buf());
        let spec = make_spec("id-cp", "t");
        store.create_spec(&spec).await.unwrap();

        let body = "Cho thuê mặt bằng kinh doanh đường Ông Ích Khiêm \
                    quận Hải Châu, diện tích 140m². Liên hệ chính chủ.";

        let mut f1 = make_finding("id-cp", "https://batdongsan.com.vn/ad/100");
        f1.excerpt = Some(body.to_string());
        f1.content_hash = content_hash(body);

        let mut f2 = make_finding("id-cp", "https://facebook.com/marketplace/item/999");
        f2.excerpt = Some(body.to_string());
        f2.content_hash = content_hash(body);

        // Sanity: URL-level keys differ.
        assert_ne!(f1.dedup_hash, f2.dedup_hash);
        assert_ne!(f1.host_path_hash, f2.host_path_hash);
        assert_eq!(f1.content_hash, f2.content_hash);

        assert!(store.try_append_finding(&f1).await.unwrap());
        assert!(
            !store.try_append_finding(&f2).await.unwrap(),
            "content_hash dedup must reject the crosspost"
        );
        assert_eq!(store.count_findings("id-cp").await.unwrap(), 1);
    }

    #[tokio::test]
    async fn dedup_survives_reopen() {
        let tmp = tempdir().unwrap();
        {
            let store = FsResearchStore::new(tmp.path().to_path_buf());
            let spec = make_spec("id-3", "t");
            store.create_spec(&spec).await.unwrap();
            let f = make_finding("id-3", "https://x.com/ad/99");
            assert!(store.try_append_finding(&f).await.unwrap());
        }
        let store2 = FsResearchStore::new(tmp.path().to_path_buf());
        let dup = make_finding("id-3", "https://x.com/ad/99?utm_source=later");
        assert!(!store2.try_append_finding(&dup).await.unwrap());
    }

    #[tokio::test]
    async fn runs_are_append_only_and_limited() {
        let tmp = tempdir().unwrap();
        let store = FsResearchStore::new(tmp.path().to_path_buf());
        let spec = make_spec("id-4", "t");
        store.create_spec(&spec).await.unwrap();
        for i in 0..5 {
            let r = RunRecord {
                run_id: format!("r{i}"),
                spec_id: "id-4".to_string(),
                started_at: Utc::now(),
                finished_at: Utc::now(),
                new_findings: 0,
                total_findings_after: 0,
                stop_reason: "ok".to_string(),
                provider: "p".to_string(),
                model: "m".to_string(),
                verification_rounds: None,
                dead_removed: None,
                replacements_found: None,
                remaining_issues: None,
                elapsed_secs: None,
            };
            store.append_run(&r).await.unwrap();
        }
        let last3 = store.list_runs("id-4", Some(3)).await.unwrap();
        assert_eq!(last3.len(), 3);
        assert_eq!(last3[0].run_id, "r2");
        assert_eq!(last3[2].run_id, "r4");
    }

    #[tokio::test]
    async fn cursor_roundtrip_and_updated_at() {
        let tmp = tempdir().unwrap();
        let store = FsResearchStore::new(tmp.path().to_path_buf());
        let _ = store.create_spec(&make_spec("id-5", "t")).await;
        let empty = store.load_cursor("id-5").await.unwrap();
        assert!(empty.data.is_empty());
        let mut c = Cursor::default();
        c.data.insert("page".into(), serde_json::Value::from(3_i64));
        store.save_cursor("id-5", &c).await.unwrap();
        let reloaded = store.load_cursor("id-5").await.unwrap();
        assert_eq!(reloaded.data.get("page"), Some(&serde_json::json!(3)));
        assert!(reloaded.updated_at.is_some());
    }

    #[tokio::test]
    async fn purge_terminal_inflight_removes_only_old_terminals() {
        use crate::research::inflight::{Inflight, RunState};

        let tmp = tempdir().unwrap();
        let store = FsResearchStore::new(tmp.path().to_path_buf());
        let now = Utc::now();

        // Spec A: terminal Completed, finished_at = 30 days ago → must purge.
        store
            .create_spec(&make_spec("a-old-completed", "t"))
            .await
            .unwrap();
        let mut a = Inflight::scheduled("a-old-completed", 1);
        a.state = RunState::Completed;
        a.finished_at = Some(now - chrono::Duration::days(30));
        store.save_inflight("a-old-completed", &a).await.unwrap();

        // Spec B: terminal Failed, finished_at = 1 hour ago → must keep.
        store
            .create_spec(&make_spec("b-recent-failed", "t"))
            .await
            .unwrap();
        let mut b = Inflight::scheduled("b-recent-failed", 1);
        b.state = RunState::Failed;
        b.finished_at = Some(now - chrono::Duration::hours(1));
        store.save_inflight("b-recent-failed", &b).await.unwrap();

        // Spec C: Running → must keep regardless of age.
        store
            .create_spec(&make_spec("c-running", "t"))
            .await
            .unwrap();
        let mut c = Inflight::scheduled("c-running", 1);
        c.state = RunState::Running;
        c.started_at = Some(now - chrono::Duration::days(99));
        store.save_inflight("c-running", &c).await.unwrap();

        // Spec D: Scheduled → must keep regardless of age.
        store
            .create_spec(&make_spec("d-scheduled", "t"))
            .await
            .unwrap();
        let mut d = Inflight::scheduled("d-scheduled", 1);
        d.scheduled_at = now - chrono::Duration::days(99);
        store.save_inflight("d-scheduled", &d).await.unwrap();

        // Spec E: Completed but no finished_at → must keep (defensive).
        store
            .create_spec(&make_spec("e-no-finished-at", "t"))
            .await
            .unwrap();
        let mut e = Inflight::scheduled("e-no-finished-at", 1);
        e.state = RunState::Completed;
        e.finished_at = None;
        store.save_inflight("e-no-finished-at", &e).await.unwrap();

        let removed = store
            .purge_terminal_inflight(now, chrono::Duration::days(7))
            .await
            .unwrap();

        assert_eq!(removed, 1, "only the 30-day-old Completed should be purged");
        assert!(
            store
                .load_inflight("a-old-completed")
                .await
                .unwrap()
                .is_none()
        );
        assert!(
            store
                .load_inflight("b-recent-failed")
                .await
                .unwrap()
                .is_some()
        );
        assert!(store.load_inflight("c-running").await.unwrap().is_some());
        assert!(store.load_inflight("d-scheduled").await.unwrap().is_some());
        assert!(
            store
                .load_inflight("e-no-finished-at")
                .await
                .unwrap()
                .is_some()
        );
    }

    #[tokio::test]
    async fn purge_terminal_inflight_zero_retention_purges_all_terminals_with_finished_at() {
        use crate::research::inflight::{Inflight, RunState};

        let tmp = tempdir().unwrap();
        let store = FsResearchStore::new(tmp.path().to_path_buf());
        let now = Utc::now();

        for (id, state) in [("c1", RunState::Completed), ("c2", RunState::Failed)] {
            store.create_spec(&make_spec(id, "t")).await.unwrap();
            let mut i = Inflight::scheduled(id, 1);
            i.state = state;
            i.finished_at = Some(now - chrono::Duration::seconds(1));
            store.save_inflight(id, &i).await.unwrap();
        }

        let removed = store
            .purge_terminal_inflight(now, chrono::Duration::zero())
            .await
            .unwrap();
        assert_eq!(removed, 2);
    }

    #[tokio::test]
    async fn purge_terminal_inflight_handles_missing_root() {
        let tmp = tempdir().unwrap();
        let missing = tmp.path().join("does-not-exist");
        let store = FsResearchStore::new(missing);
        let removed = store
            .purge_terminal_inflight(Utc::now(), chrono::Duration::days(1))
            .await
            .unwrap();
        assert_eq!(removed, 0);
    }

    #[tokio::test]
    async fn purge_terminal_inflight_default_trait_impl_is_noop() {
        struct NoopStore;
        #[async_trait]
        impl SpecStore for NoopStore {
            async fn create_spec(&self, _spec: &ResearchSpec) -> Result<()> {
                Ok(())
            }
            async fn load_spec(&self, _id: &str) -> Result<ResearchSpec> {
                Err(AgentError::Provider("unused".into()))
            }
            async fn save_spec(&self, _spec: &ResearchSpec) -> Result<()> {
                Ok(())
            }
            async fn list_specs(&self) -> Result<Vec<ResearchSpec>> {
                Ok(vec![])
            }
            async fn delete_spec(&self, _id: &str) -> Result<()> {
                Ok(())
            }
        }
        #[async_trait]
        impl FindingStore for NoopStore {
            async fn try_append_finding(&self, _finding: &Finding) -> Result<bool> {
                Ok(false)
            }
            async fn upsert_finding(&self, _finding: &Finding) -> Result<bool> {
                Ok(false)
            }
            async fn list_findings(
                &self,
                _id: &str,
                _limit: Option<usize>,
            ) -> Result<Vec<Finding>> {
                Ok(vec![])
            }
            async fn count_findings(&self, _id: &str) -> Result<u32> {
                Ok(0)
            }
            async fn remove_findings_by_hash(
                &self,
                _id: &str,
                _hashes: &HashSet<String>,
            ) -> Result<u32> {
                Ok(0)
            }
        }
        #[async_trait]
        impl RunStore for NoopStore {
            async fn append_run(&self, _run: &RunRecord) -> Result<()> {
                Ok(())
            }
            async fn list_runs(&self, _id: &str, _limit: Option<usize>) -> Result<Vec<RunRecord>> {
                Ok(vec![])
            }
            async fn load_cursor(&self, _id: &str) -> Result<Cursor> {
                Ok(Cursor::default())
            }
            async fn save_cursor(&self, _id: &str, _cursor: &Cursor) -> Result<()> {
                Ok(())
            }
        }
        #[async_trait]
        impl ReportStore for NoopStore {
            async fn write_report(&self, _id: &str, _report: &str) -> Result<()> {
                Ok(())
            }
            async fn read_report(&self, _id: &str) -> Result<Option<String>> {
                Ok(None)
            }
            async fn write_agent_brief(&self, _id: &str, _brief: &str) -> Result<()> {
                Ok(())
            }
            async fn read_agent_brief(&self, _id: &str) -> Result<Option<String>> {
                Ok(None)
            }
        }
        impl InflightStore for NoopStore {}
        let store = NoopStore;
        let removed = store
            .purge_terminal_inflight(Utc::now(), chrono::Duration::days(1))
            .await
            .unwrap();
        assert_eq!(removed, 0);
    }

    #[tokio::test]
    async fn delete_removes_dir_and_locks() {
        let tmp = tempdir().unwrap();
        let store = FsResearchStore::new(tmp.path().to_path_buf());
        let spec = make_spec("id-6", "t");
        store.create_spec(&spec).await.unwrap();
        let f = make_finding("id-6", "https://x.com/a");
        store.try_append_finding(&f).await.unwrap();
        store.delete_spec("id-6").await.unwrap();
        assert!(store.load_spec("id-6").await.is_err());
        // Recreating is allowed after delete.
        store.create_spec(&spec).await.unwrap();
    }
}
