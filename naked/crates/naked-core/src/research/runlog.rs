//! Research run log — append one-line summaries of completed runs.

use std::path::{Path, PathBuf};

use crate::error::Result;
use crate::research::{ResearchStore, RunRecord, VerifiedRunReport};

/// Append a run-completion line to the research run-log.
///
/// Run-completion lines go to the **research run-log**
/// (`research::store::research_runlog_path()` by default), *not* to any
/// `MEMORY.md`. The durable memory files are reserved for promoted rules
/// from the daily-digest pipeline.
///
/// If `memory_path_override` is `Some`, the line is appended to that
/// exact file (used by tests).
pub async fn write_research_memory_link_for(
    store: &dyn ResearchStore,
    _workspace: &Path,
    spec_id: &str,
    run_id: &str,
    verified: Option<&VerifiedRunReport>,
    memory_path_override: Option<&Path>,
) -> Result<()> {
    let spec = store.load_spec(spec_id).await?;
    let runs = store.list_runs(spec_id, Some(50)).await.unwrap_or_default();
    let record: Option<RunRecord> = runs.into_iter().find(|r| r.run_id == run_id);
    let total_after = record.as_ref().map(|r| r.total_findings_after).unwrap_or(0);
    let new_findings = record.as_ref().map(|r| r.new_findings).unwrap_or(0);
    let elapsed_secs = record.as_ref().and_then(|r| r.elapsed_secs).unwrap_or(0);

    let report_path = store
        .report_path(spec_id)
        .map(|p| p.display().to_string())
        .unwrap_or_else(|| "<unavailable>".to_string());

    let topic = spec.topic.replace('"', "'");
    let mut body = format!(
        "research:{spec_id} | topic=\"{topic}\" | run={run_id} | new={new_findings} total={total_after}",
    );
    if let Some(vr) = verified {
        body.push_str(&format!(
            " | verified={r} rounds, removed={d}, replaced={p}, remaining={rem}",
            r = vr.verification_rounds,
            d = vr.dead_removed,
            p = vr.replacements_found,
            rem = vr.remaining_issues.len(),
        ));
    }
    body.push_str(&format!(
        " | elapsed={elapsed_secs}s | report={report_path}"
    ));

    // UTF-8 safe truncation.
    let max = crate::memory::store::MAX_ENTRY_CHARS;
    let body = if body.chars().count() > max {
        let mut truncated: String = body.chars().take(max.saturating_sub(3)).collect();
        truncated.push_str("...");
        truncated
    } else {
        body
    };

    let path: PathBuf = match memory_path_override {
        Some(p) => p.to_path_buf(),
        None => crate::research::store::research_runlog_path(),
    };
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let stamp = chrono::Utc::now().format("%Y-%m-%dT%H:%M:%SZ");
    let line = format!("- {stamp} {body}\n");
    use std::io::Write;
    let mut f = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(&path)?;
    f.write_all(line.as_bytes())?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::research::store::{RunStore, SpecStore};
    use crate::research::{
        FsResearchStore, ResearchSpec, RunRecord, RunReport, StopReason, VerifiedRunReport,
    };
    use chrono::Utc;
    use std::time::Duration;
    use tempfile::TempDir;

    #[tokio::test]
    async fn test_write_research_memory_link_for() {
        let temp_dir = TempDir::new().unwrap();
        let store = FsResearchStore::new(temp_dir.path().to_path_buf());
        let workspace = temp_dir.path();
        let spec_id = "test-spec-123";
        let run_id = "run-456";

        // Create a test spec
        let spec = ResearchSpec {
            id: spec_id.to_string(),
            topic: "Test topic for runlog".to_string(),
            sources: vec!["https://example.com".to_string()],
            created_at: Utc::now(),
            ..Default::default()
        };
        store.create_spec(&spec).await.unwrap();

        // Create a test run record
        let now = Utc::now();
        let run_record = RunRecord {
            run_id: run_id.to_string(),
            spec_id: spec_id.to_string(),
            started_at: now,
            finished_at: now,
            new_findings: 12,
            total_findings_after: 42,
            stop_reason: "completed".to_string(),
            provider: "test-provider".to_string(),
            model: "test-model".to_string(),
            verification_rounds: None,
            dead_removed: None,
            replacements_found: None,
            remaining_issues: None,
            elapsed_secs: Some(120),
        };
        store.append_run(&run_record).await.unwrap();

        // Create a test verified report
        let verified = VerifiedRunReport {
            last_run: RunReport {
                spec_id: spec_id.to_string(),
                run_id: run_id.to_string(),
                new_findings: 12,
                total_findings_after: 42,
                stop_reason: StopReason::AgentIdle,
                elapsed: Duration::from_secs(120),
                provider: "test-provider".to_string(),
                model: "test-model".to_string(),
            },
            verification_rounds: 2,
            dead_removed: 3,
            replacements_found: 1,
            final_findings: 42,
            remaining_issues: vec![],
        };

        // Create a temporary output file
        let output_file = temp_dir.path().join("test_runlog.md");

        // Call the function under test
        write_research_memory_link_for(
            &store,
            workspace,
            spec_id,
            run_id,
            Some(&verified),
            Some(&output_file),
        )
        .await
        .unwrap();

        // Verify the output file was created and contains expected content
        assert!(output_file.exists());
        let content = std::fs::read_to_string(&output_file).unwrap();

        // Check that the line contains the expected components
        assert!(content.contains(spec_id));
        assert!(content.contains("Test topic for runlog"));
        assert!(content.contains(run_id));
        assert!(content.contains("new=12"));
        assert!(content.contains("total=42"));
        assert!(content.contains("verified=2 rounds"));
        assert!(content.contains("removed=3"));
        assert!(content.contains("replaced=1"));
        assert!(content.contains("elapsed=120s"));

        // Verify the line starts with a timestamp
        let lines: Vec<&str> = content.trim().split('\n').collect();
        assert_eq!(lines.len(), 1);
        assert!(lines[0].starts_with("- 20"));
        assert!(lines[0].contains("research:test-spec-123"));
    }

    #[tokio::test]
    async fn test_write_research_memory_link_without_verification() {
        let temp_dir = TempDir::new().unwrap();
        let store = FsResearchStore::new(temp_dir.path().to_path_buf());
        let workspace = temp_dir.path();
        let spec_id = "test-spec-456";
        let run_id = "run-789";

        // Create a minimal spec
        let spec = ResearchSpec {
            id: spec_id.to_string(),
            topic: "Simple test topic".to_string(),
            sources: vec![],
            created_at: Utc::now(),
            ..Default::default()
        };
        store.create_spec(&spec).await.unwrap();

        // Create a minimal run record
        let now = Utc::now();
        let run_record = RunRecord {
            run_id: run_id.to_string(),
            spec_id: spec_id.to_string(),
            started_at: now,
            finished_at: now,
            new_findings: 5,
            total_findings_after: 5,
            stop_reason: "completed".to_string(),
            provider: "test-provider".to_string(),
            model: "test-model".to_string(),
            verification_rounds: None,
            dead_removed: None,
            replacements_found: None,
            remaining_issues: None,
            elapsed_secs: Some(30),
        };
        store.append_run(&run_record).await.unwrap();

        let output_file = temp_dir.path().join("simple_runlog.md");

        // Call without verification report
        write_research_memory_link_for(
            &store,
            workspace,
            spec_id,
            run_id,
            None, // no verification
            Some(&output_file),
        )
        .await
        .unwrap();

        // Verify output
        let content = std::fs::read_to_string(&output_file).unwrap();
        assert!(content.contains("new=5 total=5"));
        assert!(content.contains("elapsed=30s"));

        // Should NOT contain verification info
        assert!(!content.contains("verified="));
        assert!(!content.contains("removed="));
        assert!(!content.contains("replaced="));
    }
}
