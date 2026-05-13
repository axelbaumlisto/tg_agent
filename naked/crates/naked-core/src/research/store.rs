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

use async_trait::async_trait;

use crate::error::Result;

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
