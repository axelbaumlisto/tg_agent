//! Filesystem-backed research store implementation.

use std::collections::HashSet;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use async_trait::async_trait;
use chrono::Utc;
use tokio::fs;
use tokio::io::AsyncWriteExt;
use tokio::sync::RwLock;

use super::inflight::Inflight;
use super::spec::{Cursor, Finding, ResearchSpec, RunRecord};
use super::store::*;
use crate::error::{AgentError, Result};

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
        Ok(self
            .list_all_inflight()
            .await?
            .into_iter()
            .filter(|i| !i.state.is_terminal())
            .collect())
    }

    async fn list_all_inflight(&self) -> Result<Vec<Inflight>> {
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
            // REGISTRY-WAIVE: intentional fallback: missing path → empty result
            let Ok(data) = fs::read_to_string(&inflight_path).await else {
                continue;
            };
            match serde_json::from_str::<Inflight>(&data) {
                Ok(v) => out.push(v),
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
            // REGISTRY-WAIVE: intentional fallback: missing path → empty result
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
#[path = "store_fs_tests.rs"]
mod tests;

#[cfg(test)]
mod store_trait_tests {
    use super::*;

    #[tokio::test]
    async fn fs_store_creates_spec_dir() {
        let tmp = tempfile::tempdir().unwrap();
        let store = FsResearchStore::new(tmp.path().to_path_buf());
        let spec = ResearchSpec {
            id: "test-1".into(),
            topic: "test topic".into(),
            sources: vec!["https://example.com".into()],
            ..Default::default()
        };
        store.save_spec(&spec).await.unwrap();
        let loaded = store.load_spec("test-1").await.unwrap();
        assert_eq!(loaded.topic, "test topic");
    }

    #[tokio::test]
    async fn fs_store_list_specs_empty() {
        let tmp = tempfile::tempdir().unwrap();
        let store = FsResearchStore::new(tmp.path().to_path_buf());
        let specs = store.list_specs().await.unwrap();
        assert!(specs.is_empty());
    }

    #[tokio::test]
    async fn fs_store_delete_spec() {
        let tmp = tempfile::tempdir().unwrap();
        let store = FsResearchStore::new(tmp.path().to_path_buf());
        let spec = ResearchSpec {
            id: "del-1".into(),
            topic: "delete me".into(),
            ..Default::default()
        };
        store.save_spec(&spec).await.unwrap();
        store.delete_spec("del-1").await.unwrap();
        assert!(store.load_spec("del-1").await.is_err());
    }
}
