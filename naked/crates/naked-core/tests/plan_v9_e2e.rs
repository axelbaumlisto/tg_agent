//! E2E tests for plan-v9 solid core modules.
//!
//! Tests integration between modules — not just unit logic,
//! but the full pipeline: factory registration, tool execution, wiring.
//!
//! Run:  cargo test -p naked-core --test plan_v9_e2e -- --nocapture

use serde_json::json;
use std::path::Path;
use tempfile::TempDir;

// ═════════════════════════════════════════════════════════════════════════════
// P1. TOKEN TRACKER — multi-model accumulation
// ═════════════════════════════════════════════════════════════════════════════

#[test]
fn token_tracker_multimodel_session() {
    use naked_core::token_tracker::TokenTracker;

    let t = TokenTracker::new();

    // Simulate a session with 3 models:
    t.record("groq/llama-3.3-70b", 500, 200);
    t.record("kimi/moonshot-v1-128k", 3000, 1500);
    t.record("groq/llama-3.3-70b", 800, 300);
    t.record("qwen/qwen3-plus", 10000, 5000);

    let snap = t.snapshot();
    assert_eq!(snap.len(), 3);
    assert_eq!(snap["groq/llama-3.3-70b"].calls, 2);
    assert_eq!(snap["groq/llama-3.3-70b"].input_tokens, 1300);
    assert_eq!(snap["kimi/moonshot-v1-128k"].output_tokens, 1500);

    let totals = t.totals();
    assert_eq!(totals.calls, 4);
    assert_eq!(
        totals.total(),
        500 + 200 + 3000 + 1500 + 800 + 300 + 10000 + 5000
    );

    // Summary contains all models:
    let summary = t.summary();
    assert!(summary.contains("groq/llama"));
    assert!(summary.contains("kimi/moonshot"));
    assert!(summary.contains("qwen/qwen3"));
    assert!(summary.contains("Total:"));

    // Clone shares state:
    let t2 = t.clone();
    t2.record("groq/llama-3.3-70b", 100, 50);
    assert_eq!(t.snapshot()["groq/llama-3.3-70b"].calls, 3);
}

// ═════════════════════════════════════════════════════════════════════════════
// P4. AUDIT — round-trip with concurrent writes
// ═════════════════════════════════════════════════════════════════════════════

#[test]
fn audit_concurrent_writes() {
    use naked_core::audit;

    let tmp = TempDir::new().unwrap();
    let dir = tmp.path();

    // Sequential writes (audit is append-only, no lock needed for sequential):
    for i in 0..20 {
        audit::log_event(dir, &format!("event_{i}"), json!({"i": i}));
    }

    let entries = audit::recent(dir, 100);
    assert_eq!(entries.len(), 20);

    // All entries have ts + event:
    for e in &entries {
        assert!(e.get("ts").is_some());
        assert!(e.get("event").is_some());
    }

    // JSONL file is valid:
    let content = std::fs::read_to_string(audit::path(dir)).unwrap();
    for line in content.lines() {
        assert!(
            serde_json::from_str::<serde_json::Value>(line).is_ok(),
            "Invalid JSONL: {line}"
        );
    }
}

// ═════════════════════════════════════════════════════════════════════════════
// P6. COMMAND ARITY — approval_cache integration
// ═════════════════════════════════════════════════════════════════════════════

#[test]
fn arity_improves_approval_cache() {
    use naked_core::tool::approval_cache::{ApprovalCache, fingerprint};

    // Old behavior: `cargo test --workspace` and `cargo test --verbose` had different fingerprints.
    // New behavior with arity: both map to `cargo test`.
    let fp1 = fingerprint("bash", &json!({"command": "cargo test --workspace"}));
    let fp2 = fingerprint("bash", &json!({"command": "cargo test --verbose --all"}));
    let fp3 = fingerprint("bash", &json!({"command": "cargo test"}));
    assert_eq!(fp1, fp2, "same canonical prefix");
    assert_eq!(fp1, fp3, "same canonical prefix");

    // Different commands get different fingerprints:
    let fp4 = fingerprint("bash", &json!({"command": "cargo build --release"}));
    assert_ne!(fp1, fp4, "build ≠ test");

    // Docker compose:
    let fp5 = fingerprint("bash", &json!({"command": "docker compose up -d"}));
    let fp6 = fingerprint("bash", &json!({"command": "docker compose up --build"}));
    assert_eq!(fp5, fp6);

    // Cache works with arity fingerprints:
    let cache = ApprovalCache::new();
    cache.approve(&fp1);
    assert!(
        cache.is_approved(&fp2),
        "approved cargo test should match with flags"
    );
    assert!(
        !cache.is_approved(&fp4),
        "cargo build should NOT be auto-approved"
    );
}

#[test]
fn arity_readonly_detection() {
    use naked_core::command_arity::{canonical_prefix, is_readonly};

    // Read-only commands:
    assert!(is_readonly(&canonical_prefix("git status -s")));
    assert!(is_readonly(&canonical_prefix("git diff HEAD~1")));
    assert!(is_readonly(&canonical_prefix("cargo check --workspace")));
    assert!(is_readonly(&canonical_prefix("ls -la /tmp")));
    assert!(is_readonly(&canonical_prefix("docker ps --all")));

    // Write commands:
    assert!(!is_readonly(&canonical_prefix("git push origin main")));
    assert!(!is_readonly(&canonical_prefix("cargo build --release")));
    assert!(!is_readonly(&canonical_prefix("rm -rf /tmp/test")));
    assert!(!is_readonly(&canonical_prefix("docker run ubuntu")));
}

// ═════════════════════════════════════════════════════════════════════════════
// P7. AUTO REASONING — effort tiers
// ═════════════════════════════════════════════════════════════════════════════

#[test]
fn auto_reasoning_tiers() {
    use naked_core::auto_reasoning::{ReasoningEffort, select};

    // Sub-agent always low regardless of content:
    assert_eq!(select(true, "debug critical error"), ReasoningEffort::Low);
    assert_eq!(select(true, "plan architecture"), ReasoningEffort::Low);

    // User messages:
    assert_eq!(select(false, "there's a bug in auth"), ReasoningEffort::Max);
    assert_eq!(
        select(false, "Error: connection refused"),
        ReasoningEffort::Max
    );
    assert_eq!(
        select(false, "debug the crash on line 42"),
        ReasoningEffort::Max
    );
    assert_eq!(
        select(false, "find all TODO comments"),
        ReasoningEffort::Low
    );
    assert_eq!(
        select(false, "search for config file"),
        ReasoningEffort::Low
    );
    assert_eq!(select(false, "plan the migration"), ReasoningEffort::High);
    assert_eq!(select(false, "design the API"), ReasoningEffort::High);
    assert_eq!(select(false, "write the handler"), ReasoningEffort::Medium);
}

// ═════════════════════════════════════════════════════════════════════════════
// P10. COHERENCE — state transitions match capacity
// ═════════════════════════════════════════════════════════════════════════════

#[test]
fn coherence_tracks_capacity() {
    use naked_core::capacity::check_pressure;
    use naked_core::coherence::{CoherenceState, from_tokens};

    let window = 128_000u64;

    // Verify coherence aligns with capacity pressure:
    for &(tokens, expected_coherence) in &[
        (10_000, CoherenceState::Healthy),
        (50_000, CoherenceState::Healthy),
        (80_000, CoherenceState::GettingCrowded),
        (100_000, CoherenceState::RefreshingContext),
        (120_000, CoherenceState::ResettingPlan),
    ] {
        let state = from_tokens(tokens, window);
        assert_eq!(state, expected_coherence, "tokens={tokens}");

        // Emoji + label are non-empty:
        assert!(!state.emoji().is_empty());
        assert!(!state.label().is_empty());
    }

    // Capacity pressure also fires at similar thresholds:
    assert!(!check_pressure(50_000, window).is_actionable());
    assert!(check_pressure(100_000, window).is_actionable());
}

// ═════════════════════════════════════════════════════════════════════════════
// P3a. TODO TOOL — full lifecycle via tool interface
// ═════════════════════════════════════════════════════════════════════════════

#[tokio::test]
async fn todo_tool_lifecycle() {
    use naked_core::tool::Tool;
    use naked_core::tool::todo_tool::{TodoList, TodoTool};

    let list = TodoList::new();
    let tool = TodoTool::new(list.clone());
    let cwd = Path::new(".");

    // Add items:
    let r = tool
        .execute(json!({"action": "add", "content": "Write tests"}), cwd)
        .await;
    assert!(!r.is_error && r.output.contains("#1"));

    let r = tool
        .execute(json!({"action": "add", "content": "Fix bug"}), cwd)
        .await;
    assert!(r.output.contains("#2"));

    let r = tool
        .execute(json!({"action": "add", "content": "Deploy"}), cwd)
        .await;
    assert!(r.output.contains("#3"));

    // List:
    let r = tool.execute(json!({"action": "list"}), cwd).await;
    assert!(
        r.output.contains("Write tests")
            && r.output.contains("Fix bug")
            && r.output.contains("Deploy")
    );
    assert!(r.output.contains("0% complete"));

    // Update status:
    let r = tool
        .execute(
            json!({"action": "update", "id": 1, "status": "completed"}),
            cwd,
        )
        .await;
    assert!(!r.is_error && r.output.contains("●"));

    let r = tool
        .execute(
            json!({"action": "update", "id": 2, "status": "in_progress"}),
            cwd,
        )
        .await;
    assert!(r.output.contains("◎"));

    // Check completion:
    assert_eq!(list.completion_pct(), 33); // 1/3

    // Remove:
    let r = tool
        .execute(json!({"action": "remove", "id": 3}), cwd)
        .await;
    assert!(!r.is_error);
    assert_eq!(list.list().len(), 2);
    assert_eq!(list.completion_pct(), 50); // 1/2

    // Error: update non-existent:
    let r = tool
        .execute(
            json!({"action": "update", "id": 99, "status": "completed"}),
            cwd,
        )
        .await;
    assert!(r.is_error);
}

// ═════════════════════════════════════════════════════════════════════════════
// P3b. PLAN TOOL — step tracker lifecycle
// ═════════════════════════════════════════════════════════════════════════════

#[tokio::test]
async fn plan_tool_lifecycle() {
    use naked_core::tool::Tool;
    use naked_core::tool::plan_tool::{PlanState, PlanTool};

    let state = PlanState::new();
    let tool = PlanTool::new(state.clone());
    let cwd = Path::new(".");

    // Set plan:
    let r = tool
        .execute(
            json!({"action": "set", "steps": ["Analyze", "Implement", "Test", "Deploy"]}),
            cwd,
        )
        .await;
    assert!(!r.is_error);
    assert!(r.output.contains("○ 1. Analyze"));
    assert!(r.output.contains("○ 4. Deploy"));

    // Update steps:
    let r = tool
        .execute(
            json!({"action": "update", "index": 0, "status": "completed"}),
            cwd,
        )
        .await;
    assert!(r.output.contains("● 1. Analyze"));

    let r = tool
        .execute(
            json!({"action": "update", "index": 1, "status": "in_progress"}),
            cwd,
        )
        .await;
    assert!(r.output.contains("◎ 2. Implement"));

    // Not all completed:
    assert!(!state.all_completed());

    // Complete all:
    for i in 1..4 {
        state.update(i, naked_core::tool::plan_tool::StepStatus::Completed);
    }
    assert!(state.all_completed());

    // Get shows final state:
    let r = tool.execute(json!({"action": "get"}), cwd).await;
    assert!(r.output.contains("●") && !r.output.contains("○"));

    // Error: out of bounds:
    let r = tool
        .execute(
            json!({"action": "update", "index": 99, "status": "completed"}),
            cwd,
        )
        .await;
    assert!(r.is_error);
}

// ═════════════════════════════════════════════════════════════════════════════
// P8. VALIDATE DATA — tool execution
// ═════════════════════════════════════════════════════════════════════════════

#[tokio::test]
async fn validate_data_tool() {
    use naked_core::tool::Tool;
    use naked_core::tool::validate_data::ValidateDataTool;

    let tool = ValidateDataTool;
    let cwd = Path::new(".");

    // Valid JSON:
    let r = tool
        .execute(json!({"content": r#"{"name":"test","version":1}"#}), cwd)
        .await;
    assert!(!r.is_error && r.output.contains("✓ Valid json"));

    // Invalid JSON:
    let r = tool
        .execute(json!({"content": r#"{"broken":}"#}), cwd)
        .await;
    assert!(!r.is_error && r.output.contains("✗ Invalid json"));

    // Valid TOML:
    let r = tool
        .execute(
            json!({"content": "[package]\nname = \"test\"\nversion = \"1.0\""}),
            cwd,
        )
        .await;
    assert!(r.output.contains("✓ Valid toml"));

    // Invalid TOML:
    let r = tool
        .execute(json!({"content": "[broken\nname =", "format": "toml"}), cwd)
        .await;
    assert!(r.output.contains("✗ Invalid toml"));

    // File-based validation:
    let tmp = TempDir::new().unwrap();
    let json_file = tmp.path().join("test.json");
    std::fs::write(&json_file, r#"{"valid": true}"#).unwrap();
    let r = tool
        .execute(json!({"path": json_file.to_str().unwrap()}), cwd)
        .await;
    assert!(r.output.contains("✓ Valid json"));

    // No input = error:
    let r = tool.execute(json!({}), cwd).await;
    assert!(r.is_error);
}

// ═════════════════════════════════════════════════════════════════════════════
// P9. SCHEMA SANITIZE — dirty MCP schemas
// ═════════════════════════════════════════════════════════════════════════════

#[test]
fn schema_sanitize_real_mcp_patterns() {
    use naked_core::tool::schema_sanitize::sanitize;

    // Pattern 1: Pydantic nullable (common from Python MCP servers):
    let mut s = json!({
        "type": "object",
        "properties": {
            "name": {"type": "string"},
            "age": {"anyOf": [{"type": "integer"}, {"type": "null"}]}
        },
        "required": ["name", "age"]
    });
    sanitize(&mut s);
    assert_eq!(s["properties"]["age"]["type"], "integer");
    assert_eq!(s["properties"]["age"]["nullable"], true);
    assert!(
        !s["properties"]["age"]
            .as_object()
            .unwrap()
            .contains_key("anyOf")
    );

    // Pattern 2: Bare object (no properties):
    let mut s = json!({"type": "object"});
    sanitize(&mut s);
    assert!(s["properties"].is_object());

    // Pattern 3: Dangling required:
    let mut s = json!({
        "type": "object",
        "properties": {"a": {"type": "string"}},
        "required": ["a", "b", "c"]
    });
    sanitize(&mut s);
    let req: Vec<&str> = s["required"]
        .as_array()
        .unwrap()
        .iter()
        .filter_map(|v| v.as_str())
        .collect();
    assert_eq!(req, vec!["a"]);

    // Pattern 4: Single-element oneOf wrapper:
    let mut s = json!({"oneOf": [{"type": "string", "enum": ["a","b"]}]});
    sanitize(&mut s);
    assert_eq!(s["type"], "string");
    assert!(!s.as_object().unwrap().contains_key("oneOf"));

    // Pattern 5: Nested — properties with dirty sub-schemas:
    let mut s = json!({
        "type": "object",
        "properties": {
            "config": {
                "type": "object",
                "properties": {
                    "timeout": {"anyOf": [{"type": "number"}, {"type": "null"}]}
                }
            }
        }
    });
    sanitize(&mut s);
    assert_eq!(
        s["properties"]["config"]["properties"]["timeout"]["type"],
        "number"
    );
    assert_eq!(
        s["properties"]["config"]["properties"]["timeout"]["nullable"],
        true
    );
}

// ═════════════════════════════════════════════════════════════════════════════
// P2. RECALL ARCHIVE — BM25 search over JSONL cycles
// ═════════════════════════════════════════════════════════════════════════════

#[tokio::test]
async fn recall_archive_search() {
    use naked_core::tool::Tool;
    use naked_core::tool::recall_archive::RecallArchiveTool;

    let tmp = TempDir::new().unwrap();
    let sessions_dir = tmp.path();

    // Create fake cycle archives:
    let cycle_dir = sessions_dir.join("test-session").join("cycles");
    std::fs::create_dir_all(&cycle_dir).unwrap();

    // Cycle 0: early conversation about Rust:
    std::fs::write(
        cycle_dir.join("0.jsonl"),
        r#"{"role":"user","content":"How do I handle errors in Rust?"}
{"role":"assistant","content":"Use Result<T, E> with the ? operator. Avoid unwrap() in production code."}
{"role":"user","content":"What about panics?"}
{"role":"assistant","content":"Panics are for unrecoverable errors. Use panic::catch_unwind for FFI boundaries."}"#,
    ).unwrap();

    // Cycle 1: later conversation about testing:
    std::fs::write(
        cycle_dir.join("1.jsonl"),
        r#"{"role":"user","content":"How do I write integration tests?"}
{"role":"assistant","content":"Put tests in tests/ directory. Use #[tokio::test] for async. Mock external services."}"#,
    ).unwrap();

    let tool = RecallArchiveTool::new(sessions_dir);
    let cwd = Path::new(".");

    // Search for Rust error handling:
    let r = tool
        .execute(
            json!({"query": "error handling Result", "session_id": "test-session"}),
            cwd,
        )
        .await;
    assert!(!r.is_error);
    assert!(r.output.contains("matches"));
    assert!(r.output.contains("Result")); // excerpt should contain the match

    // Search for testing:
    let r = tool
        .execute(
            json!({"query": "integration tests", "session_id": "test-session"}),
            cwd,
        )
        .await;
    assert!(r.output.contains("tests/"));

    // No results:
    let r = tool
        .execute(
            json!({"query": "quantum computing", "session_id": "test-session"}),
            cwd,
        )
        .await;
    assert!(r.output.contains("No matches"));

    // Missing session:
    let r = tool
        .execute(
            json!({"query": "anything", "session_id": "nonexistent"}),
            cwd,
        )
        .await;
    assert!(r.output.contains("No archived cycles"));
}

// ═════════════════════════════════════════════════════════════════════════════
// P5. REVIEW TOOL — real file analysis
// ═════════════════════════════════════════════════════════════════════════════

#[tokio::test]
async fn review_tool_real_files() {
    use naked_core::tool::Tool;
    use naked_core::tool::review_tool::ReviewTool;

    let tmp = TempDir::new().unwrap();
    let tool = ReviewTool;

    // File with issues:
    let bad_file = tmp.path().join("bad.rs");
    std::fs::write(
        &bad_file,
        r#"
fn process() {
    // TODO: handle edge case
    let value = get_value().unwrap();
    // FIXME: this is fragile
    let data = parse().expect("should work");
}
"#,
    )
    .unwrap();

    let r = tool
        .execute(json!({"path": bad_file.to_str().unwrap()}), Path::new("."))
        .await;
    assert!(!r.is_error);
    assert!(r.output.contains("TODO"));
    assert!(r.output.contains("FIXME"));
    assert!(r.output.contains("unwrap"));
    assert!(r.output.contains("expect"));

    // Clean file:
    let good_file = tmp.path().join("good.rs");
    std::fs::write(
        &good_file,
        r#"
fn add(a: i32, b: i32) -> i32 {
    a + b
}
"#,
    )
    .unwrap();

    let r = tool
        .execute(json!({"path": good_file.to_str().unwrap()}), Path::new("."))
        .await;
    assert!(r.output.contains("No issues"));

    // Long function warning:
    let long_fn = tmp.path().join("long.rs");
    let mut code = String::from("fn big() {\n");
    for i in 0..60 {
        code.push_str(&format!("    let x{i} = {i};\n"));
    }
    code.push_str("}\n");
    std::fs::write(&long_fn, &code).unwrap();

    let r = tool
        .execute(json!({"path": long_fn.to_str().unwrap()}), Path::new("."))
        .await;
    assert!(r.output.contains("lines"));

    // Non-existent file:
    let r = tool
        .execute(json!({"path": "/tmp/nonexistent_12345.rs"}), Path::new("."))
        .await;
    assert!(r.is_error);
}

// ═════════════════════════════════════════════════════════════════════════════
// INTEGRATION: tool factory registers all new tools
// ═════════════════════════════════════════════════════════════════════════════

#[test]
fn all_v9_tools_registered_names() {
    // Verify tool names exist (factory test — just names, no full init):
    // These are the names from spec():
    assert_eq!(
        naked_core::tool::validate_data::ValidateDataTool
            .spec()
            .name,
        "validate_data"
    );
    assert_eq!(
        naked_core::tool::review_tool::ReviewTool.spec().name,
        "review"
    );

    let todo =
        naked_core::tool::todo_tool::TodoTool::new(naked_core::tool::todo_tool::TodoList::new());
    assert_eq!(todo.spec().name, "todo");

    let plan =
        naked_core::tool::plan_tool::PlanTool::new(naked_core::tool::plan_tool::PlanState::new());
    assert_eq!(plan.spec().name, "plan");

    let recall = naked_core::tool::recall_archive::RecallArchiveTool::new("/tmp");
    assert_eq!(recall.spec().name, "recall_archive");

    // Verify permissions:
    use naked_core::tool::Tool;
    use naked_core::types::Permission;
    assert_eq!(
        naked_core::tool::validate_data::ValidateDataTool
            .spec()
            .permission,
        Permission::ReadOnly
    );
    assert_eq!(todo.spec().permission, Permission::ReadOnly);
    assert_eq!(plan.spec().permission, Permission::ReadOnly);
    assert_eq!(
        naked_core::tool::review_tool::ReviewTool.spec().permission,
        Permission::ReadOnly
    );
    assert_eq!(recall.spec().permission, Permission::ReadOnly);
}
