//! Phase 1 e2e for the research subsystem: back-compat of `RunRecord`
//! deserialization, verification metrics persistence through `run_verified`,
//! and the global memory link helper.
//!
//! These tests are intentionally pure-local — no LLM, no network. They exist
//! to pin down the JSONL/memory contract so the LLM control plane built on
//! top in subsequent phases has stable semantics.

use std::path::PathBuf;
use std::sync::Arc;

use chrono::Utc;
use naked_core::config::ResearchConfig;
use naked_core::error::Result as NakedResult;
use naked_core::memory::store::MarkdownMemoryStore;
use naked_core::research::coordinator::{AgentRunner, CoordinatorConfig, ResearchCoordinator};
use naked_core::research::ops_tool::{
    ResearchHelpTool, ResearchListSpecsTool, ResearchMetricsTool, ResearchSetScheduleTool,
};
use naked_core::research::spec::{Finding, ResearchSpec, RunRecord, dedup_hash};
use naked_core::research::store::{
    FsResearchStore, ReportStore, ResearchStore, RunStore, SpecStore, research_runlog_path,
};
use naked_core::tool::Tool;
use naked_core::types::{AgentEvent, AgentHandle, PermissionResponse};
use naked_core::{ResearchPatch, apply_research_patch, write_research_memory_link_for};
use serde_json::{Value, json};
use tempfile::tempdir;
use tokio::sync::{Mutex as AsyncMutex, mpsc};

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
        max_wall_seconds: Some(5),
        created_at: Utc::now(),
        paused: false,
        pause_reason: None,
    }
}

fn finding(spec_id: &str, url: &str, run_id: &str) -> Finding {
    use naked_core::research::spec::{content_hash, host_path_hash};
    Finding {
        id: uuid::Uuid::new_v4().simple().to_string(),
        research_id: spec_id.to_string(),
        run_id: run_id.to_string(),
        url: url.to_string(),
        title: Some("t".into()),
        excerpt: None,
        price: None,
        listing_date: None,
        source_content: None,
        dedup_hash: dedup_hash(url),
        host_path_hash: host_path_hash(url),
        content_hash: content_hash(""),
        seen_at: Utc::now(),
    }
}

/// Minimal scripted `AgentRunner` — drains a fixed list of events into the
/// coordinator's stream, then pushes a fixed list of findings into the store.
/// Always emits `Idle` so the coordinator records a clean stop.
struct ScriptedRunner {
    findings: AsyncMutex<Vec<Finding>>,
    store: Arc<dyn ResearchStore>,
}

#[async_trait::async_trait]
impl AgentRunner for ScriptedRunner {
    async fn start_research_turn(
        &self,
        _spec: &ResearchSpec,
        _prompt: &str,
        _config: &CoordinatorConfig,
        _run_id: &str,
    ) -> NakedResult<(AgentHandle, String, String)> {
        let (tx, rx) = mpsc::channel(8);
        let (perm_tx, _perm_rx) = mpsc::channel::<PermissionResponse>(4);
        let findings = std::mem::take(&mut *self.findings.lock().await);
        let store = self.store.clone();
        tokio::spawn(async move {
            for f in findings {
                let _ = store.try_append_finding(&f).await;
            }
            let _ = tx.send(AgentEvent::Idle).await;
        });
        Ok((
            {
                let (steer_tx, _) = tokio::sync::mpsc::channel(1);
                AgentHandle {
                    events: rx,
                    permissions: perm_tx,
                    steer: steer_tx,
                }
            },
            "test-provider".into(),
            "test-model".into(),
        ))
    }

    async fn cleanup_research_session(&self, _session_id: &str) {}
}

#[tokio::test]
async fn t_runrecord_back_compat_load_legacy_jsonl() {
    // Legacy JSONL row written before verification fields existed must still
    // load, with all new optional fields defaulting to `None`.
    let tmp = tempdir().unwrap();
    let store = FsResearchStore::new(tmp.path().to_path_buf());
    let spec = make_spec("legacy-1", "legacy");
    store.create_spec(&spec).await.unwrap();

    let legacy = serde_json::json!({
        "run_id": "old-run",
        "spec_id": "legacy-1",
        "started_at": "2026-01-01T00:00:00Z",
        "finished_at": "2026-01-01T00:00:05Z",
        "new_findings": 3,
        "total_findings_after": 7,
        "stop_reason": "agent_idle",
        "provider": "old-provider",
        "model": "old-model"
    });
    let runs_path = tmp.path().join("legacy-1").join("runs.jsonl");
    let mut line = legacy.to_string();
    line.push('\n');
    tokio::fs::write(&runs_path, line).await.unwrap();

    let runs = store.list_runs("legacy-1", None).await.unwrap();
    assert_eq!(runs.len(), 1);
    let r = &runs[0];
    assert_eq!(r.run_id, "old-run");
    assert_eq!(r.new_findings, 3);
    assert_eq!(r.total_findings_after, 7);
    assert_eq!(r.verification_rounds, None);
    assert_eq!(r.dead_removed, None);
    assert_eq!(r.replacements_found, None);
    assert_eq!(r.remaining_issues, None);
    assert_eq!(r.elapsed_secs, None);
}

#[tokio::test]
async fn t_runrecord_persists_verification_metrics() {
    // Driving `run_verified` with a stub runner must produce at least one
    // run record with verification stats populated.
    let tmp = tempdir().unwrap();
    let store: Arc<dyn ResearchStore> = Arc::new(FsResearchStore::new(tmp.path().to_path_buf()));
    let spec = make_spec("v-1", "verified");
    store.create_spec(&spec).await.unwrap();

    let runner = Arc::new(ScriptedRunner {
        findings: AsyncMutex::new(vec![finding("v-1", "https://example.com/a", "r")]),
        store: store.clone(),
    });
    let coord = ResearchCoordinator::new(store.clone(), runner, CoordinatorConfig::default());

    // 0 max_rounds → fall back to gatekeeper default; we only care that the
    // verification summary record exists at the end.
    let _ = coord
        .run_verified("v-1", 0)
        .await
        .expect("run_verified should succeed for stub runner");

    let runs = store.list_runs("v-1", None).await.unwrap();
    let summary = runs
        .iter()
        .find(|r| r.verification_rounds.is_some())
        .expect("at least one record should carry verification stats");
    assert!(summary.run_id.ends_with("-verified"));
    assert!(summary.verification_rounds.unwrap() >= 1);
    assert!(summary.elapsed_secs.is_some());
}

#[tokio::test]
async fn t_memory_entry_written_after_successful_run() {
    // Use the explicit `memory_path_override` so we don't have to mutate the
    // process-global `NAKED_HOME` env var (which is also `forbid(unsafe_code)`
    // territory in the modern Rust edition).
    let tmp = tempdir().unwrap();
    let workspace = tempdir().unwrap();
    let memory_path = tmp.path().join("memory/MEMORY.md");

    let store = FsResearchStore::new(tmp.path().join("research"));
    let spec = make_spec("mem-1", "memory link");
    store.create_spec(&spec).await.unwrap();

    let record = RunRecord {
        run_id: "run-xyz".to_string(),
        spec_id: "mem-1".to_string(),
        started_at: Utc::now() - chrono::Duration::seconds(12),
        finished_at: Utc::now(),
        new_findings: 4,
        total_findings_after: 9,
        stop_reason: "agent_idle".to_string(),
        provider: "p".to_string(),
        model: "m".to_string(),
        verification_rounds: None,
        dead_removed: None,
        replacements_found: None,
        remaining_issues: None,
        elapsed_secs: Some(12),
    };
    store.append_run(&record).await.unwrap();
    store
        .write_report("mem-1", "# stub report\n")
        .await
        .unwrap();

    write_research_memory_link_for(
        &store,
        workspace.path(),
        "mem-1",
        "run-xyz",
        None,
        Some(&memory_path),
    )
    .await
    .expect("memory link write should succeed");

    let content = tokio::fs::read_to_string(&memory_path)
        .await
        .expect("MEMORY.md should be created");
    assert!(
        content.contains("research:mem-1"),
        "memory entry missing spec id, got:\n{content}"
    );
    assert!(
        content.contains("new=4 total=9"),
        "memory entry missing run counts, got:\n{content}"
    );
    assert!(
        content.contains("report="),
        "memory entry missing report= line, got:\n{content}"
    );
    assert!(
        content.contains("report.md"),
        "memory entry should reference report.md, got:\n{content}"
    );
}

/// Regression: research run-completion lines must NOT pollute the durable
/// `MEMORY.md` files (Global, Project, or User). They belong in a
/// research-scoped run-log so the system prompt stays free of run noise.
#[tokio::test]
async fn t_research_run_does_not_pollute_global_memory() {
    // The default destination of `write_research_memory_link_for` (when the
    // override is `None`) must point at the research run-log, not at the
    // global `MEMORY.md`.
    let runlog = research_runlog_path();
    let global = MarkdownMemoryStore::global_memory_path();

    assert_ne!(
        runlog,
        global,
        "research run-log must not be the global MEMORY.md (got {})",
        runlog.display(),
    );
    assert!(
        runlog
            .components()
            .any(|c: std::path::Component| c.as_os_str() == "research"),
        "research run-log must live under .../research/, got {}",
        runlog.display(),
    );
    assert!(
        !runlog.to_string_lossy().contains("MEMORY.md"),
        "research run-log must not be named MEMORY.md, got {}",
        runlog.display(),
    );
}

#[tokio::test]
async fn t_research_metrics_tool_lists_runs_and_report() {
    // Pure-local exercise of `ResearchMetricsTool`: seed disk with a spec,
    // two run records (one verified summary), and a report.md, then assert
    // the JSON shape returned by `execute()`.
    let tmp = tempdir().unwrap();
    let store: Arc<dyn ResearchStore> = Arc::new(FsResearchStore::new(tmp.path().to_path_buf()));
    let mut spec = make_spec("rm-1", "metrics target");
    spec.interval_seconds = Some(3600);
    store.create_spec(&spec).await.unwrap();

    let r1 = RunRecord {
        run_id: "run-a".to_string(),
        spec_id: "rm-1".to_string(),
        started_at: Utc::now() - chrono::Duration::seconds(120),
        finished_at: Utc::now() - chrono::Duration::seconds(60),
        new_findings: 2,
        total_findings_after: 2,
        stop_reason: "agent_idle".to_string(),
        provider: "p".to_string(),
        model: "m".to_string(),
        verification_rounds: None,
        dead_removed: None,
        replacements_found: None,
        remaining_issues: None,
        elapsed_secs: Some(60),
    };
    let r2 = RunRecord {
        run_id: "run-a-verified".to_string(),
        spec_id: "rm-1".to_string(),
        started_at: Utc::now() - chrono::Duration::seconds(120),
        finished_at: Utc::now() - chrono::Duration::seconds(20),
        new_findings: 0,
        total_findings_after: 2,
        stop_reason: "agent_idle".to_string(),
        provider: "p".to_string(),
        model: "m".to_string(),
        verification_rounds: Some(2),
        dead_removed: Some(1),
        replacements_found: Some(1),
        remaining_issues: Some(0),
        elapsed_secs: Some(100),
    };
    store.append_run(&r1).await.unwrap();
    store.append_run(&r2).await.unwrap();
    store
        .try_append_finding(&finding("rm-1", "https://ex.com/x", "run-a"))
        .await
        .unwrap();
    store
        .write_report("rm-1", "# Report\nbody body body\n")
        .await
        .unwrap();

    let tool = ResearchMetricsTool::new(store.clone());
    let cwd = PathBuf::from(".");
    let result = tool.execute(json!({"spec_id": "rm-1"}), &cwd).await;
    assert!(!result.is_error, "tool errored: {}", result.output);

    let v: Value = serde_json::from_str(&result.output).expect("tool output is JSON");
    assert_eq!(v["spec_id"], "rm-1");
    assert_eq!(v["topic"], "metrics target");
    assert_eq!(v["paused"], false);
    assert_eq!(v["schedule"]["interval_seconds"], 3600);
    assert_eq!(v["total_findings"], 1);
    let runs = v["runs"].as_array().expect("runs is array");
    assert_eq!(runs.len(), 2);
    // Verified row should carry the gatekeeper stats.
    let verified = runs
        .iter()
        .find(|r| r["verification_rounds"].is_u64())
        .expect("at least one row with verification stats");
    assert_eq!(verified["verification_rounds"], 2);
    assert_eq!(verified["dead_removed"], 1);
    assert_eq!(verified["replacements_found"], 1);
    let excerpt = v["report_excerpt"].as_str().expect("report excerpt");
    assert!(excerpt.starts_with("# Report"));
}

#[tokio::test]
async fn t_research_list_specs_tool_includes_schedule_and_verification() {
    let tmp = tempdir().unwrap();
    let store: Arc<dyn ResearchStore> = Arc::new(FsResearchStore::new(tmp.path().to_path_buf()));
    let mut spec = make_spec("ls-1", "spec list test");
    spec.interval_seconds = Some(900);
    store.create_spec(&spec).await.unwrap();

    let summary = RunRecord {
        run_id: "run-z-verified".to_string(),
        spec_id: "ls-1".to_string(),
        started_at: Utc::now() - chrono::Duration::seconds(30),
        finished_at: Utc::now(),
        new_findings: 5,
        total_findings_after: 5,
        stop_reason: "agent_idle".to_string(),
        provider: "p".to_string(),
        model: "m".to_string(),
        verification_rounds: Some(1),
        dead_removed: Some(0),
        replacements_found: Some(0),
        remaining_issues: Some(0),
        elapsed_secs: Some(30),
    };
    store.append_run(&summary).await.unwrap();

    let cfg = ResearchConfig {
        verify_by_default: true,
        ..Default::default()
    };
    let tool = ResearchListSpecsTool::new(store.clone(), cfg);
    let result = tool.execute(json!({}), &PathBuf::from(".")).await;
    assert!(!result.is_error, "tool errored: {}", result.output);

    let arr: Value = serde_json::from_str(&result.output).expect("array json");
    let item = &arr.as_array().unwrap()[0];
    assert_eq!(item["id"], "ls-1");
    assert_eq!(item["interval_seconds"], 900);
    assert_eq!(item["verify_by_default"], true);
    assert_eq!(item["last_run"]["verification_rounds"], 1);
    assert_eq!(item["last_run"]["new"], 5);
}

// ─── Phase 3: ResearchPatch semantics ───────────────────────────────────────

#[test]
fn t_patch_topic_trims_and_skips_empty() {
    let mut spec = make_spec("p-1", "original");
    let patch = ResearchPatch {
        topic: Some("   ".into()),
        ..Default::default()
    };
    apply_research_patch(&mut spec, patch);
    assert_eq!(spec.topic, "original", "blank topic should be a no-op");

    let patch = ResearchPatch {
        topic: Some("  new topic  ".into()),
        ..Default::default()
    };
    apply_research_patch(&mut spec, patch);
    assert_eq!(spec.topic, "new topic", "topic should be trimmed");
}

#[test]
fn t_patch_sources_add_dedups_and_skips_blanks() {
    let mut spec = make_spec("p-2", "t");
    spec.sources = vec!["https://a.example".into()];

    let patch = ResearchPatch {
        sources_add: Some(vec![
            "https://a.example".into(),
            "https://b.example".into(),
            "  ".into(),
            "https://b.example".into(),
        ]),
        ..Default::default()
    };
    apply_research_patch(&mut spec, patch);
    assert_eq!(
        spec.sources,
        vec![
            "https://a.example".to_string(),
            "https://b.example".to_string(),
        ]
    );
}

#[test]
fn t_patch_sources_replace_dedups() {
    let mut spec = make_spec("p-3", "t");
    spec.sources = vec!["https://old.example".into()];

    let patch = ResearchPatch {
        sources_replace: Some(vec![
            "https://x.example".into(),
            "https://y.example".into(),
            "https://x.example".into(),
            "".into(),
        ]),
        ..Default::default()
    };
    apply_research_patch(&mut spec, patch);
    assert_eq!(
        spec.sources,
        vec![
            "https://x.example".to_string(),
            "https://y.example".to_string(),
        ],
        "replace should drop duplicates and blanks"
    );
}

#[test]
fn t_patch_interval_set_then_clear() {
    let mut spec = make_spec("p-4", "t");
    assert!(spec.interval_seconds.is_none());

    let patch = ResearchPatch {
        interval_seconds: Some(Some(1800)),
        ..Default::default()
    };
    apply_research_patch(&mut spec, patch);
    assert_eq!(spec.interval_seconds, Some(1800));

    let patch = ResearchPatch {
        interval_seconds: Some(None),
        ..Default::default()
    };
    apply_research_patch(&mut spec, patch);
    assert_eq!(spec.interval_seconds, None);

    // No-op when field is `None`.
    spec.interval_seconds = Some(60);
    let patch = ResearchPatch::default();
    apply_research_patch(&mut spec, patch);
    assert_eq!(spec.interval_seconds, Some(60));
}

#[test]
fn t_patch_provider_model_clear_via_empty_string() {
    let mut spec = make_spec("p-5", "t");
    spec.provider = Some("openrouter".into());
    spec.model = Some("anthropic/claude-3".into());

    let patch = ResearchPatch {
        provider: Some("".into()),
        model: Some("openai/gpt-4o-mini".into()),
        ..Default::default()
    };
    apply_research_patch(&mut spec, patch);
    assert_eq!(spec.provider, None);
    assert_eq!(spec.model.as_deref(), Some("openai/gpt-4o-mini"));
}

#[test]
fn t_patch_max_iterations_and_wall_seconds() {
    let mut spec = make_spec("p-6", "t");
    spec.max_iterations = Some(10);
    spec.max_wall_seconds = Some(60);

    let patch = ResearchPatch {
        max_iterations: Some(Some(25)),
        max_wall_seconds: Some(None),
        ..Default::default()
    };
    apply_research_patch(&mut spec, patch);
    assert_eq!(spec.max_iterations, Some(25));
    assert_eq!(spec.max_wall_seconds, None);
}

#[tokio::test]
async fn t_research_update_spec_tool_payload_round_trip() {
    // The tool's JSON parsing layer is tested indirectly: build a patch the
    // same way the tool does, apply it, and confirm the spec mutates as
    // expected. This pins the contract the LLM sees without spinning up
    // AgentCore.
    let tmp = tempdir().unwrap();
    let store: Arc<dyn ResearchStore> = Arc::new(FsResearchStore::new(tmp.path().to_path_buf()));
    let mut spec = make_spec("upd-1", "before");
    spec.sources = vec!["https://keep.example".into()];
    store.create_spec(&spec).await.unwrap();

    // Simulate the tool's JSON → ResearchPatch translation for the common
    // "add a source + set schedule + rename" case.
    let patch = ResearchPatch {
        topic: Some("after".into()),
        sources_add: Some(vec!["https://added.example".into()]),
        interval_seconds: Some(Some(7200)),
        ..Default::default()
    };

    apply_research_patch(&mut spec, patch);
    store.save_spec(&spec).await.unwrap();

    let reloaded = store.load_spec("upd-1").await.unwrap();
    assert_eq!(reloaded.topic, "after");
    assert_eq!(
        reloaded.sources,
        vec![
            "https://keep.example".to_string(),
            "https://added.example".to_string(),
        ]
    );
    assert_eq!(reloaded.interval_seconds, Some(7200));
}

/// Schema-shape contract for the dedicated `research_set_schedule` tool.
///
/// The LLM never sees Rust code — it only sees the JSON schema returned by
/// `Tool::spec()`. If we silently rename `interval_seconds`/`enabled`, drop
/// `spec_id` from `required`, or change the accepted types, the LLM will
/// stop calling this tool correctly. This test pins the public contract
/// without spinning up a live model.
#[tokio::test]
async fn t_research_set_schedule_tool_exposes_stable_schema() {
    use std::sync::Weak;

    use naked_core::AgentCore;

    // Pass a dangling Weak — we never `execute()` here, only inspect the
    // descriptor, so the upgrade path is never exercised.
    let weak: Weak<AgentCore> = Weak::new();
    let tool = ResearchSetScheduleTool::new(weak);
    let spec = tool.spec();

    assert_eq!(spec.name, "research_set_schedule");

    let params = &spec.parameters;
    assert_eq!(params["type"], "object", "schema must be an object");

    let props = params["properties"]
        .as_object()
        .expect("properties is an object");
    assert!(props.contains_key("spec_id"), "missing spec_id property");
    assert!(
        props.contains_key("interval_seconds"),
        "missing interval_seconds property"
    );
    assert!(props.contains_key("enabled"), "missing enabled property");

    let required: Vec<&str> = params["required"]
        .as_array()
        .expect("required is an array")
        .iter()
        .filter_map(|v| v.as_str())
        .collect();
    assert!(
        required.contains(&"spec_id"),
        "spec_id must be required; got {required:?}"
    );

    // interval_seconds must accept null (clear schedule) — the LLM relies
    // on this nullability to express "remove the schedule entirely".
    let interval_type = &props["interval_seconds"]["type"];
    let accepts_null = match interval_type {
        Value::Array(arr) => arr.iter().any(|v| v.as_str() == Some("null")),
        Value::String(s) => s == "null",
        _ => false,
    };
    assert!(
        accepts_null,
        "interval_seconds must accept null to allow clearing the schedule; got {interval_type}"
    );

    // enabled must be a plain boolean.
    assert_eq!(props["enabled"]["type"], "boolean");
}

/// `research_help` tool returns a self-contained markdown explanation of the
/// research subsystem so the LLM can answer "how does scheduling work" without
/// the user having to paste docs into chat.
///
/// Three things are pinned here:
///
/// 1. The schema declares no required parameters (the LLM must be able to
///    invoke it as `research_help({})` — any required field would block the
///    common case).
/// 2. The body mentions the architectural concepts a reasonable user
///    question maps onto: scheduler, semaphore (concurrency cap),
///    gatekeeper verification, storage layout.
/// 3. The body lists every other research-orchestration tool — this is the
///    contract that lets the LLM jump from the help page to the right
///    follow-up tool without guessing.
///
/// We intentionally do NOT regression-test the exact prose: it lives in
/// `research::briefing` and the unit tests there pin the substitutions.
/// This test pins only the surface that the LLM observes.
#[tokio::test]
async fn t_research_help_tool_returns_briefing_with_zero_required_inputs() {
    use naked_core::config::ResearchConfig;
    use naked_core::types::ToolResult;
    use std::path::Path;

    let cfg = ResearchConfig::default();
    let tool = ResearchHelpTool::new(cfg.clone());

    // 1. Schema contract.
    let spec = tool.spec();
    assert_eq!(spec.name, "research_help");
    let required = spec.parameters.get("required");
    let required_empty = match required {
        None => true,
        Some(Value::Array(a)) => a.is_empty(),
        _ => false,
    };
    assert!(
        required_empty,
        "research_help must accept zero required args so the LLM can call it \
         directly when the user asks 'how does this work'; got {required:?}"
    );

    // 2 + 3. Body contract.
    let ToolResult { output, is_error } = tool.execute(json!({}), Path::new("/tmp")).await;
    assert!(!is_error, "tool should not error on empty input: {output}");

    for concept in ["scheduler", "semaphore", "verification", "storage layout"] {
        assert!(
            output.to_lowercase().contains(concept),
            "briefing must mention `{concept}`; got: {output}"
        );
    }

    for sibling_tool in [
        "research_create",
        "research_list_specs",
        "research_metrics",
        "research_findings",
        "research_launch",
        "research_update_spec",
        "research_set_schedule",
        "research_pause",
        "research_resume",
    ] {
        assert!(
            output.contains(sibling_tool),
            "briefing must reference sibling tool `{sibling_tool}` so the LLM \
             knows what to call after reading the help; got: {output}"
        );
    }
}

/// The system prompt assembled by `AgentCore::create_session_with_channel`
/// must include the **short** research briefing whenever
/// `Config.research.enabled` is true. This is the "Option B" surface from
/// the design discussion: the LLM gets enough context to answer
/// "когда B запустится / как работает расписание" with at most one
/// `research_metrics` call, without needing a separate `research_help`
/// fetch.
///
/// We don't construct an `AgentCore` here (it pulls in providers, MCP, etc.)
/// — the prompt-assembly contract is tested at the `briefing::short` level
/// in `research/briefing.rs`. This test just verifies that the public
/// briefing API still produces something useful so the prompt path stays
/// honest: a passing `research_help` body but an empty `briefing::short`
/// would defeat the design.
#[test]
fn t_research_briefing_short_is_non_empty_and_mentions_concurrency_cap() {
    use naked_core::config::ResearchConfig;
    use naked_core::research::briefing;

    let cfg = ResearchConfig {
        max_concurrent_runs: 3,
        ..Default::default()
    };
    let s = briefing::short(&cfg);

    assert!(!s.is_empty(), "short briefing must not be empty");
    assert!(
        s.contains("at most 3 run"),
        "short briefing must reflect Config.research.max_concurrent_runs so \
         operators editing the config see their value mirrored in the LLM \
         prompt; got: {s}"
    );
    assert!(
        s.to_lowercase().contains("scheduler"),
        "short briefing must mention the scheduler so the LLM can answer \
         scheduling questions without an extra tool call; got: {s}"
    );
}
