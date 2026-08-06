//! Deterministic mock-driven code-agent e2e: exercises the REAL optimized
//! tool hot-path (fff grep, fs_cache read, stale-guard + hashline edit,
//! atomic write, persistent bash) with all fast-backend flags ON, scripted by
//! a MockProvider so the measured time reflects OUR code — not LLM latency.
//!
//! Purpose: prove the full coding-agent tool loop works end-to-end through the
//! AgentLoop, and surface where our own work spends time (vs the provider).
//! We assert the optimization counters actually moved (fast_index grep, cache
//! hit, persistent bash) so a regression that silently disables an optimization
//! is caught.

use crate::loop_::AgentLoop;
use crate::provider::{ChatRequest, Provider};
use crate::tool::Tool;
use crate::tool::registry::ToolRegistry;
use crate::types::{AgentEvent, StreamChunk, TurnUsage};
use std::pin::Pin;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use tokio::sync::mpsc;
use tokio_util::sync::CancellationToken;

/// MockProvider that replays a scripted sequence of tool calls, one provider
/// "turn" per loop iteration. After the scripted steps it emits a final text
/// answer + Done so the loop terminates cleanly.
struct ScriptedProvider {
    responses: Vec<Vec<StreamChunk>>,
    call_count: AtomicUsize,
}

impl ScriptedProvider {
    fn new(responses: Vec<Vec<StreamChunk>>) -> Self {
        Self {
            responses,
            call_count: AtomicUsize::new(0),
        }
    }
}

#[async_trait::async_trait]
impl Provider for ScriptedProvider {
    fn name(&self) -> &str {
        "scripted-mock"
    }
    fn models(&self) -> Vec<crate::types::ModelInfo> {
        vec![]
    }
    async fn stream_chat(
        &self,
        _request: ChatRequest,
    ) -> crate::error::Result<Pin<Box<dyn tokio_stream::Stream<Item = StreamChunk> + Send>>> {
        let idx = self.call_count.fetch_add(1, Ordering::SeqCst);
        let chunks = if idx < self.responses.len() {
            self.responses[idx].clone()
        } else {
            vec![
                StreamChunk::Text("done".into()),
                StreamChunk::Usage(TurnUsage {
                    input_tokens: 10,
                    output_tokens: 2,
                    ..Default::default()
                }),
                StreamChunk::Done,
            ]
        };
        Ok(Box::pin(tokio_stream::iter(chunks)))
    }
}

fn tool_use(id: &str, name: &str, input: serde_json::Value) -> Vec<StreamChunk> {
    vec![
        StreamChunk::ToolUse {
            id: id.into(),
            name: name.into(),
            input,
        },
        StreamChunk::Done,
    ]
}

/// Build a realistic mini-workspace: a handful of source files containing a
/// known needle, plus an editable target file.
fn seed_workspace(root: &std::path::Path) {
    std::fs::create_dir_all(root.join("src")).unwrap();
    for i in 0..12 {
        std::fs::write(
            root.join(format!("src/module_{i:02}.rs")),
            format!("// module {i}\npub fn f_{i}() {{ let _ = NEEDLE_CODEAGENT; }}\n"),
        )
        .unwrap();
    }
    std::fs::write(
        root.join("src/target.rs"),
        "pub const OLD_VALUE: u32 = 1;\npub fn keep() {}\n",
    )
    .unwrap();
}

/// Build the REAL optimized core tool set (subset relevant to a coding agent),
/// with all fast-backend flags ON, exactly as `core_tools` wires them.
fn build_optimized_tools(workspace: &std::path::Path, session_id: &str) -> Vec<Box<dyn Tool>> {
    use crate::tool::bash::BashTool;
    use crate::tool::fff_registry::{FffPickerRegistry, FffRegistryConfig};
    use crate::tool::fff_tools::{FffGrepTool, FffState};
    use crate::tool::file_ops::{EditFileTool, ReadFileTool, WriteFileTool};
    use crate::tool::fs_cache::FsCache;
    use crate::tool::persistent_bash::PersistentBashManager;

    let fs_cache = Arc::new(FsCache::new(64 * 1024 * 1024));
    let fff_registry = Arc::new(FffPickerRegistry::new());
    let persistent = Arc::new(PersistentBashManager::new());

    // fff fast-index ON via the registry path (warm picker per workspace).
    let fff_cfg = FffRegistryConfig {
        enabled: true,
        max_workspaces: 4,
        cache_max_bytes: 256 * 1024 * 1024,
    };
    let fff_state = FffState::with_registry(fff_registry.clone(), workspace, fff_cfg);

    vec![
        Box::new(BashTool::new(30).with_persistent(persistent.clone(), session_id.to_string())),
        Box::new(ReadFileTool::new(Some(fs_cache.clone()))),
        Box::new(WriteFileTool::new(Some(fs_cache.clone()))),
        Box::new(
            EditFileTool::new(true) // stale_edit_guard ON
                .with_hashline_edit(true)
                .with_fs_cache(Some(fs_cache.clone())),
        ),
        Box::new(FffGrepTool::new(&fff_state)),
    ]
}

fn make_loop(
    provider: ScriptedProvider,
    tools: Vec<Box<dyn Tool>>,
    cwd: &std::path::Path,
) -> AgentLoop {
    AgentLoop::new(
        Box::new(provider),
        ToolRegistry::new(tools),
        crate::loop_::LoopConfig {
            max_iterations: 20,
            max_wall: None,
            cwd: cwd.to_path_buf(),
            model: "scripted-mock".into(),
            max_tokens: 1024,
            ..Default::default()
        },
    )
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn code_agent_full_tooling_hotpath_e2e() {
    let tmp = tempfile::tempdir().unwrap();
    let ws = tmp.path();
    seed_workspace(ws);
    let target = ws.join("src/target.rs");

    // Baseline optimization counters (process-global; assert deltas).
    use crate::types::{
        FFF_GREP_FALLBACK_COUNT, FFF_GREP_FAST_INDEX_COUNT, FS_CACHE_HIT_COUNT,
        FS_CACHE_MISS_COUNT, PERSISTENT_BASH_OK_COUNT,
    };
    let g_fast0 = FFF_GREP_FAST_INDEX_COUNT.load(Ordering::Relaxed);
    let g_fallback0 = FFF_GREP_FALLBACK_COUNT.load(Ordering::Relaxed);
    let c_hit0 = FS_CACHE_HIT_COUNT.load(Ordering::Relaxed);
    let c_miss0 = FS_CACHE_MISS_COUNT.load(Ordering::Relaxed);
    let pb_ok0 = PERSISTENT_BASH_OK_COUNT.load(Ordering::Relaxed);

    // Scripted coding-agent flow (one provider turn per tool):
    //  1. grep_search NEEDLE_CODEAGENT      (fff fast-index)
    //  2. read_file target.rs               (fs_cache miss)
    //  3. read_file target.rs AGAIN         (fs_cache HIT)
    //  4. edit_file OLD_VALUE -> NEW_VALUE  (stale-guard + atomic write)
    //  5. write_file new file               (atomic write)
    //  6. bash: grep the new value          (persistent bash)
    let target_s = target.to_string_lossy().to_string();
    let newfile_s = ws.join("src/created.rs").to_string_lossy().to_string();
    let provider = ScriptedProvider::new(vec![
        tool_use(
            "c1",
            "grep_search",
            serde_json::json!({"query": "NEEDLE_CODEAGENT"}),
        ),
        tool_use(
            "c2",
            "read_file",
            serde_json::json!({"file_path": target_s}),
        ),
        tool_use(
            "c3",
            "read_file",
            serde_json::json!({"file_path": target_s}),
        ),
        tool_use(
            "c4",
            "edit_file",
            serde_json::json!({"file_path": target_s, "old_string": "OLD_VALUE: u32 = 1", "new_string": "NEW_VALUE: u32 = 2"}),
        ),
        tool_use(
            "c5",
            "write_file",
            serde_json::json!({"file_path": newfile_s, "contents": "pub const CREATED: bool = true;\n"}),
        ),
        tool_use(
            "c6",
            "bash",
            serde_json::json!({"command": format!("grep -c NEW_VALUE {target_s}")}),
        ),
    ]);

    let tools = build_optimized_tools(ws, "code-agent-e2e");
    let agent = make_loop(provider, tools, ws);
    let mut history = crate::history::ConversationHistory::new("you are a coding agent".into());
    history
        .push_user("Refactor target.rs: rename OLD_VALUE to NEW_VALUE, create created.rs, verify.");

    let (tx, mut rx) = mpsc::channel(256);
    let cancel = CancellationToken::new();

    let t0 = std::time::Instant::now();
    let result = agent.run(&mut history, tx, cancel, None, None).await;
    let elapsed = t0.elapsed();

    assert!(result.is_ok(), "agent loop failed: {result:?}");

    // Collect tool-call events to prove every tool actually ran.
    let mut tool_names = Vec::new();
    let mut had_error = false;
    while let Ok(ev) = rx.try_recv() {
        match ev {
            AgentEvent::ToolStart { name, .. } => tool_names.push(name),
            AgentEvent::Error(_) => had_error = true,
            _ => {}
        }
    }
    assert!(!had_error, "agent emitted an error event");
    for expected in [
        "grep_search",
        "read_file",
        "edit_file",
        "write_file",
        "bash",
    ] {
        assert!(
            tool_names.iter().any(|n| n == expected),
            "tool {expected} never ran; saw {tool_names:?}"
        );
    }

    // The edit actually landed on disk (atomic write).
    let edited = std::fs::read_to_string(&target).unwrap();
    assert!(
        edited.contains("NEW_VALUE: u32 = 2"),
        "edit not applied: {edited}"
    );
    assert!(!edited.contains("OLD_VALUE"), "old value still present");
    // The new file was created atomically.
    assert!(
        ws.join("src/created.rs").exists(),
        "write_file did not create the file"
    );

    // Optimization counters moved as designed.
    let g_fast = FFF_GREP_FAST_INDEX_COUNT
        .load(Ordering::Relaxed)
        .saturating_sub(g_fast0);
    let g_fallback = FFF_GREP_FALLBACK_COUNT
        .load(Ordering::Relaxed)
        .saturating_sub(g_fallback0);
    let c_hit = FS_CACHE_HIT_COUNT
        .load(Ordering::Relaxed)
        .saturating_sub(c_hit0);
    let c_miss = FS_CACHE_MISS_COUNT
        .load(Ordering::Relaxed)
        .saturating_sub(c_miss0);
    let pb_ok = PERSISTENT_BASH_OK_COUNT
        .load(Ordering::Relaxed)
        .saturating_sub(pb_ok0);

    assert!(
        g_fast >= 1,
        "grep did not take the fff fast-index path (fast={g_fast} fallback={g_fallback})"
    );
    assert_eq!(g_fallback, 0, "grep fell back instead of using fast-index");
    assert!(
        c_miss >= 1,
        "first read_file should be a cache miss (miss={c_miss})"
    );
    assert!(
        c_hit >= 1,
        "second read_file of same file should be a cache HIT (hit={c_hit})"
    );
    assert!(
        pb_ok >= 1,
        "bash did not run via persistent shell (ok={pb_ok})"
    );

    // Whole 6-tool coding flow through the real loop is expected to be fast
    // (mock provider returns instantly), but wall-clock ceilings are flaky on
    // loaded hosts. Keep this as informational telemetry, not a hard gate.
    eprintln!(
        "code_agent_full_tooling_hotpath_e2e: 6-tool flow informational elapsed={:?} | \
         grep_fast={g_fast} cache_hit={c_hit} cache_miss={c_miss} persistent_bash_ok={pb_ok}",
        elapsed
    );
}
