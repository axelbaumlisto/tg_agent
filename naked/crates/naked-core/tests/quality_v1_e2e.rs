//! Integration tests for PLAN_QUALITY_v1 modules that exercise real
//! filesystem + git binary (no network, no LLM).
//!
//! Covers the user-facing claim that's hardest to prove at the unit
//! level: "edit broke `cargo test` → `/restore N` rolls back". We
//! drive `SnapshotRepo` + `RevertTurnTool` against a real temp git
//! workspace and assert end-to-end behaviour.
//!
//! The other PLAN_QUALITY_v1 modules (T2 LSP, T5 permissions, T6
//! lifecycle hooks, T7 coherence ladder, T9 skills, T10 sub-agent
//! roles) are pure-logic / pure-data and fully covered by the
//! per-module unit tests in `naked-core/src/{lsp,permissions,
//! coherence,…}/`. They don't need a parallel integration suite.
//!
//! Why a separate file (vs adding to `loop_golden.rs`)? Loop-golden
//! is sealed via filenames `G1..G12` for the loop's own contract.
//! Quality-v1 tests cover STORAGE + TOOL surfaces, which is a
//! different stable identifier domain.

use std::path::Path;

use naked_core::snapshot::SnapshotRepo;
use naked_core::tool::Tool;
use naked_core::tool::revert_turn::RevertTurnTool;
use serde_json::json;

fn require_git() -> bool {
    std::process::Command::new("git")
        .arg("--version")
        .output()
        .map(|o| o.status.success())
        .unwrap_or(false)
}

#[tokio::test]
async fn t1_snapshot_capture_restore_roundtrip() {
    if !require_git() {
        eprintln!("skip: git binary not available");
        return;
    }
    let dir = tempfile::TempDir::new().unwrap();
    let ws = dir.path();
    std::fs::write(ws.join("foo.rs"), "fn main() {}").unwrap();

    let repo = SnapshotRepo::open_or_init(ws).expect("init");

    // Pre-turn snapshot.
    let pre = repo
        .capture("pre-turn:1")
        .expect("capture pre")
        .expect("non-empty");

    // Simulate an agent edit that broke the file.
    std::fs::write(ws.join("foo.rs"), "BROKEN GARBAGE").unwrap();
    std::fs::write(ws.join("new_file.txt"), "agent created me").unwrap();
    assert!(ws.join("new_file.txt").exists());

    // Restore.
    repo.restore(&pre).expect("restore");

    let restored = std::fs::read_to_string(ws.join("foo.rs")).expect("read");
    assert_eq!(restored, "fn main() {}", "pre-turn content must be back");
    // `restore <id> -- .` only writes paths that existed in the
    // snapshot. The new_file.txt created post-snapshot is left as-is
    // (this matches DeepSeek TUI's revert_turn semantics).
    assert!(
        ws.join("new_file.txt").exists(),
        "post-snapshot extras are not deleted"
    );
}

#[tokio::test]
async fn t1_revert_turn_tool_with_offset_one() {
    if !require_git() {
        eprintln!("skip: git binary not available");
        return;
    }
    let dir = tempfile::TempDir::new().unwrap();
    let ws = dir.path();
    std::fs::write(ws.join("a.txt"), "v1").unwrap();
    let repo = SnapshotRepo::open_or_init(ws).expect("init");
    repo.capture("pre-turn:1").unwrap().unwrap();

    // Modify after snapshot.
    std::fs::write(ws.join("a.txt"), "v2-broken").unwrap();

    // Run the tool.
    let tool = RevertTurnTool;
    let result = tool.execute(json!({ "turn_offset": 1 }), ws).await;

    assert!(!result.is_error, "tool reported error: {}", result.output);
    assert!(
        result.output.contains("Reverted to snapshot"),
        "missing summary: {}",
        result.output
    );
    let after = std::fs::read_to_string(ws.join("a.txt")).unwrap();
    assert_eq!(after, "v1", "v1 must be restored");
}

#[tokio::test]
async fn t1_revert_turn_offset_too_high_errors_gracefully() {
    if !require_git() {
        return;
    }
    let dir = tempfile::TempDir::new().unwrap();
    let ws = dir.path();
    std::fs::write(ws.join("a.txt"), "x").unwrap();
    let repo = SnapshotRepo::open_or_init(ws).expect("init");
    repo.capture("pre-turn:1").unwrap();
    let tool = RevertTurnTool;
    let r = tool.execute(json!({ "turn_offset": 99 }), ws).await;
    assert!(r.is_error, "must error on out-of-range offset");
    assert!(
        r.output.contains("turn_offset must be")
            || r.output.contains("out of range")
            || r.output.contains("snapshot(s)"),
        "must explain the failure: {}",
        r.output
    );
}

#[tokio::test]
async fn t1_capture_skips_when_no_changes() {
    if !require_git() {
        return;
    }
    let dir = tempfile::TempDir::new().unwrap();
    let ws = dir.path();
    std::fs::write(ws.join("a.txt"), "x").unwrap();
    let repo = SnapshotRepo::open_or_init(ws).expect("init");
    let first = repo.capture("first").unwrap();
    assert!(first.is_some(), "first capture must produce id");
    // No changes since.
    let second = repo.capture("second").unwrap();
    assert!(
        second.is_none(),
        "no-op capture must return None (got {second:?})"
    );
}

#[tokio::test]
async fn t1_list_orders_newest_first_real_git() {
    if !require_git() {
        return;
    }
    let dir = tempfile::TempDir::new().unwrap();
    let ws = dir.path();
    std::fs::write(ws.join("a.txt"), "v1").unwrap();
    let repo = SnapshotRepo::open_or_init(ws).expect("init");
    repo.capture("pre-turn:1").unwrap();
    std::fs::write(ws.join("a.txt"), "v2").unwrap();
    repo.capture("pre-turn:2").unwrap();
    std::fs::write(ws.join("a.txt"), "v3").unwrap();
    repo.capture("pre-turn:3").unwrap();

    let list = repo.list(10).unwrap();
    assert!(list.len() >= 3, "expected ≥3 snapshots");
    assert_eq!(list[0].label, "pre-turn:3");
    assert_eq!(list[1].label, "pre-turn:2");
    assert_eq!(list[2].label, "pre-turn:1");
}

#[tokio::test]
async fn t1_revert_with_offset_two_picks_older_snapshot() {
    if !require_git() {
        return;
    }
    let dir = tempfile::TempDir::new().unwrap();
    let ws = dir.path();
    std::fs::write(ws.join("a.txt"), "v1").unwrap();
    let repo = SnapshotRepo::open_or_init(ws).expect("init");
    repo.capture("pre-turn:1").unwrap();
    std::fs::write(ws.join("a.txt"), "v2").unwrap();
    repo.capture("pre-turn:2").unwrap();
    std::fs::write(ws.join("a.txt"), "v3").unwrap();

    let tool = RevertTurnTool;
    let r = tool.execute(json!({ "turn_offset": 2 }), ws).await;
    assert!(!r.is_error, "{}", r.output);

    let after = std::fs::read_to_string(ws.join("a.txt")).unwrap();
    assert_eq!(after, "v1", "offset=2 must restore the OLDER snapshot");
}

// ── T7 coherence ladder: signal-driven transitions end-to-end ────────

#[test]
fn t7_coherence_ladder_state_machine_e2e() {
    use naked_core::coherence::{CoherenceSignal, CoherenceState, next_coherence_state};

    // Simulate a session lifecycle: healthy → empty retries → compaction
    // → healthy → loop-guard halt → verifying.
    let mut s = CoherenceState::Healthy;
    s = next_coherence_state(s, CoherenceSignal::EmptyContentRetriesRising);
    assert_eq!(s, CoherenceState::GettingCrowded);
    s = next_coherence_state(s, CoherenceSignal::CompactionStarted);
    assert_eq!(s, CoherenceState::RefreshingContext);
    s = next_coherence_state(s, CoherenceSignal::CompactionCompleted);
    assert_eq!(s, CoherenceState::Healthy);
    s = next_coherence_state(s, CoherenceSignal::LoopGuardHalt);
    assert_eq!(s, CoherenceState::VerifyingRecentWork);
}

// ── T10 sub-agent role taxonomy: alias resolution end-to-end ─────────

#[test]
fn t10_canonicalize_role_full_alias_table() {
    use naked_core::agent_role::{CanonicalRole, canonicalize_role, role_for_canonical};

    // Every alias listed in the plan resolves correctly.
    for (input, expected) in [
        ("general", CanonicalRole::General),
        ("worker", CanonicalRole::General),
        ("default", CanonicalRole::General),
        ("explore", CanonicalRole::Explore),
        ("explorer", CanonicalRole::Explore),
        ("plan", CanonicalRole::Plan),
        ("planning", CanonicalRole::Plan),
        ("review", CanonicalRole::Review),
        ("code-review", CanonicalRole::Review),
        ("implementer", CanonicalRole::Implementer),
        ("builder", CanonicalRole::Implementer),
        ("verifier", CanonicalRole::Verifier),
        ("tester", CanonicalRole::Verifier),
        ("custom", CanonicalRole::Custom),
        // Case-insensitive
        ("EXPLORE", CanonicalRole::Explore),
        ("Code-Review", CanonicalRole::Review),
    ] {
        assert_eq!(
            canonicalize_role(input),
            Some(expected),
            "alias '{input}' must resolve to {expected:?}"
        );
    }
    assert_eq!(canonicalize_role("nonsense"), None);

    // Each role builds a non-empty AgentRole.
    let r = role_for_canonical(CanonicalRole::Explore);
    assert_eq!(r.name, "explore");
}

// ── T9 skill ecosystem paths: resolver discovers .claude/skills ──────

#[test]
fn t9_skill_resolver_discovers_claude_skills_in_parent() {
    use naked_core::skill::resolver::SkillResolver;

    let dir = tempfile::TempDir::new().unwrap();
    let claude_dir = dir.path().join(".claude").join("skills").join("test_skill");
    std::fs::create_dir_all(&claude_dir).unwrap();
    std::fs::write(
        claude_dir.join("SKILL.md"),
        "---\nname: test_skill\ndescription: a test\n---\n\nbody",
    )
    .unwrap();

    // Workspace == the temp dir (so .claude/skills is right there).
    let resolver = SkillResolver::with_ecosystem_paths(Vec::new(), dir.path());
    let resolved = resolver.resolve("test_skill");
    assert!(
        resolved.is_some(),
        "ecosystem-path skill must be discoverable"
    );
}

// ── T6 lifecycle hooks: real subprocess fires on PostToolUse ─────────

#[tokio::test]
async fn t6_lifecycle_hook_runner_fires_real_shell_command() {
    use naked_core::lifecycle_hooks::{
        FailMode, HookConfig, HookEvent, HookOutcome, LifecycleHookRunner,
    };

    let dir = tempfile::TempDir::new().unwrap();
    let marker = dir.path().join("hook_fired");
    let runner = LifecycleHookRunner::new();
    runner
        .install(HookConfig {
            event: HookEvent::PostToolUse,
            matcher: "write_file".into(),
            command: format!("touch {}", marker.display()),
            timeout_sec: 2,
            outcome_on_fail: FailMode::Continue,
        })
        .await;

    let outcome = runner
        .run(HookEvent::PostToolUse, "write_file:src/x.rs", &[])
        .await;
    assert_eq!(outcome, HookOutcome::Success);
    assert!(marker.exists(), "hook must have created marker file");
}

// ── T5 permissions: ruleset evaluates real workspace patterns ───────

#[test]
fn t5_permissions_ruleset_real_workspace_glob() {
    use naked_core::permissions::{Action, Rule, Ruleset};

    let mut rs = Ruleset::default();
    rs.push(Rule::new("read", "src/**", Action::Allow));
    rs.push(Rule::new("read", "src/secrets/*.key", Action::Deny));
    rs.push(Rule::new("write", ".env*", Action::Deny));

    // Allow rule.
    assert_eq!(rs.evaluate("read", "src/foo.rs"), Action::Allow);
    assert_eq!(rs.evaluate("read", "src/sub/dir/x.py"), Action::Allow);
    // More specific deny wins (last match).
    assert_eq!(rs.evaluate("read", "src/secrets/api.key"), Action::Deny);
    // Unmatched (read on .env) → Ask.
    assert_eq!(rs.evaluate("read", ".env"), Action::Ask);
    // Write deny.
    assert_eq!(rs.evaluate("write", ".env.local"), Action::Deny);
    // Out-of-scope tool → Ask.
    assert_eq!(rs.evaluate("shell", "anything"), Action::Ask);
}

// ── T2 LSP: smoke that path extraction handles all 3 tool shapes ────

#[test]
fn t2_lsp_edited_paths_three_tool_shapes_e2e() {
    use naked_core::loop_::lsp_hooks::edited_paths_for_tool;
    use serde_json::json;

    // Shape 1: edit_file with explicit path.
    let p1 = edited_paths_for_tool("edit_file", &json!({"path": "src/foo.rs"}));
    assert_eq!(p1, vec![std::path::PathBuf::from("src/foo.rs")]);

    // Shape 2: apply_patch with files array.
    let p2 = edited_paths_for_tool(
        "apply_patch",
        &json!({"files": [{"path": "a.rs", "content": "x"}, {"path": "b.rs", "content": "y"}]}),
    );
    assert_eq!(
        p2,
        vec![
            std::path::PathBuf::from("a.rs"),
            std::path::PathBuf::from("b.rs"),
        ]
    );

    // Shape 3: apply_patch with raw unified diff.
    let diff = "diff --git a/x.rs b/x.rs\n--- a/x.rs\n+++ b/x.rs\n@@ +1\n";
    let p3 = edited_paths_for_tool("apply_patch", &json!({"patch": diff}));
    assert_eq!(p3, vec![std::path::PathBuf::from("x.rs")]);

    // Non-edit tool returns empty.
    assert!(edited_paths_for_tool("read_file", &json!({"path": "x"})).is_empty());
}

// ── meta: ensure tests file itself imports succeed ──────────────────

#[test]
fn meta_module_paths_resolve() {
    // Compile-time guard: if any of the new modules got renamed
    // accidentally, this test fails to compile, which is the right
    // failure mode at the integration boundary.
    use naked_core::agent_role::CanonicalRole;
    use naked_core::coherence::CoherenceState;
    use naked_core::lifecycle_hooks::HookEvent;
    use naked_core::lsp::Severity;
    use naked_core::permissions::Action;
    use naked_core::snapshot::SnapshotRepo;
    let _ = (
        CanonicalRole::General,
        CoherenceState::Healthy,
        HookEvent::PreToolUse,
        Severity::Error,
        Action::Allow,
        std::any::type_name::<SnapshotRepo>(),
    );
    let _ = Path::new(""); // touch the import
}
