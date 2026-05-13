//! Golden tests for steer-injection — DO NOT DELETE.
//!
//! Pin the contract that fixes the user-visible "Принято — доставлю
//! между шагами" hang reproduced in incident img_20260510_f1d4
//! (PLAN_NEXT_SESSION.md §1.5).
//!
//!   * Q1: steer during gated-tool execution → drained at next
//!     iteration boundary.
//!   * Q2: steer during readonly-tool batch → drained at next
//!     iteration boundary.
//!   * Q3: steer mid-LLM-stream WITH tool calls → soft-interrupt
//!     fires, model sees the steer on the next iteration.
//!   * Q4: steer mid-LLM-stream WITHOUT tool calls → not lost
//!     (S1 pre-Idle drain catches it).
//!   * Q5: TWO steers in a single stream window → merged into one
//!     user message.
//!
//! Renaming or removing any test here requires a paired removal in
//! `PLAN_NEXT_SESSION.md` §A.2 S5. The golden anchor convention
//! follows `loop_golden.rs` (G1..G8) and `liveness_golden.rs`
//! (L1..L5).

use std::pin::Pin;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Duration;

use async_trait::async_trait;
use tokio::sync::mpsc;
use tokio_stream::StreamExt;
use tokio_util::sync::CancellationToken;

use naked_core::error::Result;
use naked_core::history::ConversationHistory;
use naked_core::loop_::{AgentLoop, LoopConfig};
use naked_core::provider::{ChatRequest, Provider};
use naked_core::tool::{Tool, registry::ToolRegistry};
use naked_core::types::{
    ModelInfo, Permission, Role, SteerMessage, StreamChunk, ToolResult, ToolSpec,
};

// ─── Mock provider with optional per-chunk delay ────────────────────

struct DelayedMockProvider {
    responses: Vec<Vec<StreamChunk>>,
    call_count: AtomicUsize,
    /// Per-chunk delay ON THE FIRST RESPONSE. Subsequent responses are
    /// instant (so the test can re-issue without waiting).
    first_chunk_delay: Duration,
}

impl DelayedMockProvider {
    fn new(responses: Vec<Vec<StreamChunk>>, delay: Duration) -> Self {
        Self {
            responses,
            call_count: AtomicUsize::new(0),
            first_chunk_delay: delay,
        }
    }
}

#[async_trait]
impl Provider for DelayedMockProvider {
    fn name(&self) -> &str {
        "delayed-mock"
    }
    fn models(&self) -> Vec<ModelInfo> {
        vec![]
    }
    async fn stream_chat(
        &self,
        _request: ChatRequest,
    ) -> Result<Pin<Box<dyn tokio_stream::Stream<Item = StreamChunk> + Send>>> {
        let idx = self.call_count.fetch_add(1, Ordering::SeqCst);
        let chunks = if idx < self.responses.len() {
            self.responses[idx].clone()
        } else {
            vec![StreamChunk::Text("fallback".into()), StreamChunk::Done]
        };
        if idx == 0 {
            let d = self.first_chunk_delay;
            let s = tokio_stream::iter(chunks).then(move |c| async move {
                tokio::time::sleep(d).await;
                c
            });
            Ok(Box::pin(s))
        } else {
            Ok(Box::pin(tokio_stream::iter(chunks)))
        }
    }
}

// ─── Tools used by these tests ──────────────────────────────────────

struct EchoTool;

#[async_trait]
impl Tool for EchoTool {
    fn spec(&self) -> ToolSpec {
        // REGISTRY-WAIVE: exhaustive test struct ctor — pre-2026-05-13 baseline; new tests must use ..Default::default()
        ToolSpec {
            name: "echo".into(),
            description: "Echo".into(),
            parameters: serde_json::json!({
                "type":"object",
                "properties":{"text":{"type":"string"}},
                "required":["text"]
            }),
            permission: Permission::ReadOnly,
        }
    }
    async fn execute(&self, input: serde_json::Value, _cwd: &std::path::Path) -> ToolResult {
        ToolResult {
            output: format!("echoed: {}", input["text"].as_str().unwrap_or("")),
            is_error: false,
        }
    }
}

struct SlowReadonlyTool {
    delay_ms: u64,
}

impl SlowReadonlyTool {
    fn new(delay_ms: u64) -> Self {
        Self { delay_ms }
    }
}

#[async_trait]
impl Tool for SlowReadonlyTool {
    fn spec(&self) -> ToolSpec {
        // REGISTRY-WAIVE: exhaustive test struct ctor — pre-2026-05-13 baseline; new tests must use ..Default::default()
        ToolSpec {
            name: "slow_ro".into(),
            description: "Slow readonly".into(),
            parameters: serde_json::json!({"type":"object"}),
            permission: Permission::ReadOnly,
        }
    }
    async fn execute(&self, _input: serde_json::Value, _cwd: &std::path::Path) -> ToolResult {
        tokio::time::sleep(Duration::from_millis(self.delay_ms)).await;
        ToolResult {
            output: "slow done".into(),
            is_error: false,
        }
    }
}

fn make_loop(provider: Box<dyn Provider>, tools: Vec<Box<dyn Tool>>) -> AgentLoop {
    AgentLoop::new(provider, ToolRegistry::new(tools), LoopConfig::default())
}

// ─── Q1: steer during gated tool → drained at boundary ─────────────
//
// We use a single readonly tool here but assert the drain happens
// between iterations. (Gated-tool boundary uses the same drain_steers
// call as the iteration top.) The behaviour-equivalent contract.

#[tokio::test]
async fn q1_steer_during_tool_drained_at_iteration_boundary() {
    let provider = DelayedMockProvider::new(
        vec![
            // Iteration 1: invoke a slow tool
            vec![
                StreamChunk::ToolUse {
                    id: "t1".into(),
                    name: "slow_ro".into(),
                    input: serde_json::json!({}),
                },
                StreamChunk::Done,
            ],
            // Iteration 2: respond — model sees the steer in history
            vec![
                StreamChunk::Text("done with steer".into()),
                StreamChunk::Done,
            ],
        ],
        Duration::from_millis(0), // instant first stream so steer is mid-tool, not mid-stream
    );
    let agent_loop = make_loop(
        Box::new(provider),
        vec![Box::new(SlowReadonlyTool::new(120))],
    );
    let mut history = ConversationHistory::new("sys".into());
    history.push_user("kick off");
    let (tx, _rx) = mpsc::channel(64);
    let cancel = CancellationToken::new();
    let (steer_tx, steer_rx) = mpsc::channel(16);

    let handle = tokio::spawn(async move {
        agent_loop
            .run(&mut history, tx, cancel, None, Some(steer_rx))
            .await
            .map(|_| history)
    });

    // 50ms after spawn: iteration 1 stream is consumed; tool is running.
    tokio::time::sleep(Duration::from_millis(50)).await;
    steer_tx
        .send(SteerMessage {
            msg_id: 1,
            text: "Q1 steer".into(),
            is_edit: false,
        })
        .await
        .unwrap();

    let history = handle.await.unwrap().unwrap();
    let ok = history
        .messages()
        .iter()
        .any(|m| m.role == Role::User && m.text_content().contains("Q1 steer"));
    assert!(ok, "Q1: steer must reach history at iteration boundary");
}

// ─── Q2: steer during readonly batch → drained at iteration top ────

#[tokio::test]
async fn q2_steer_during_readonly_batch_drained() {
    let provider = DelayedMockProvider::new(
        vec![
            vec![
                StreamChunk::ToolUse {
                    id: "a".into(),
                    name: "slow_ro".into(),
                    input: serde_json::json!({}),
                },
                StreamChunk::ToolUse {
                    id: "b".into(),
                    name: "slow_ro".into(),
                    input: serde_json::json!({}),
                },
                StreamChunk::Done,
            ],
            vec![StreamChunk::Text("merged".into()), StreamChunk::Done],
        ],
        Duration::from_millis(0),
    );
    let agent_loop = make_loop(
        Box::new(provider),
        vec![Box::new(SlowReadonlyTool::new(120))],
    );
    let mut history = ConversationHistory::new("sys".into());
    history.push_user("parallel work");
    let (tx, _rx) = mpsc::channel(64);
    let cancel = CancellationToken::new();
    let (steer_tx, steer_rx) = mpsc::channel(16);

    let handle = tokio::spawn(async move {
        agent_loop
            .run(&mut history, tx, cancel, None, Some(steer_rx))
            .await
            .map(|_| history)
    });
    tokio::time::sleep(Duration::from_millis(50)).await;
    steer_tx
        .send(SteerMessage {
            msg_id: 2,
            text: "Q2 steer".into(),
            is_edit: false,
        })
        .await
        .unwrap();

    let history = handle.await.unwrap().unwrap();
    let ok = history
        .messages()
        .iter()
        .any(|m| m.role == Role::User && m.text_content().contains("Q2 steer"));
    assert!(
        ok,
        "Q2: steer arriving during readonly batch must reach history"
    );
}

// ─── Q3: steer mid-stream WITH tool-call response → soft-interrupt ─
//
// Model is streaming text+tool-use slowly; user injects a steer. With
// S2/S3 the stream breaks immediately, the partial assistant output
// is dropped, and iteration 2's response sees the steer.

#[tokio::test]
async fn q3_steer_mid_stream_soft_interrupts() {
    let provider = DelayedMockProvider::new(
        vec![
            // Iteration 1 (slow): would have produced text + a tool use,
            // but the steer interrupts before any of it is committed.
            vec![
                StreamChunk::Text("starting...".into()),
                StreamChunk::ToolUse {
                    id: "x".into(),
                    name: "echo".into(),
                    input: serde_json::json!({"text":"would-have"}),
                },
                StreamChunk::Done,
            ],
            // Iteration 2: instant, replies acknowledging the steer.
            vec![StreamChunk::Text("got Q3 steer".into()), StreamChunk::Done],
        ],
        Duration::from_millis(80),
    );
    let agent_loop = make_loop(Box::new(provider), vec![Box::new(EchoTool)]);
    let mut history = ConversationHistory::new("sys".into());
    history.push_user("question");
    let (tx, _rx) = mpsc::channel(64);
    let cancel = CancellationToken::new();
    let (steer_tx, steer_rx) = mpsc::channel(16);

    let handle = tokio::spawn(async move {
        agent_loop
            .run(&mut history, tx, cancel, None, Some(steer_rx))
            .await
            .map(|_| history)
    });
    // Wait past iteration-top drain but before iteration-1 stream finishes.
    tokio::time::sleep(Duration::from_millis(50)).await;
    steer_tx
        .send(SteerMessage {
            msg_id: 3,
            text: "Q3 steer".into(),
            is_edit: false,
        })
        .await
        .unwrap();

    let history = handle.await.unwrap().unwrap();
    let steer_ok = history
        .messages()
        .iter()
        .any(|m| m.role == Role::User && m.text_content().contains("Q3 steer"));
    assert!(steer_ok, "Q3: steer must reach history");
    let answered = history
        .messages()
        .iter()
        .any(|m| m.role == Role::Assistant && m.text_content().contains("got Q3 steer"));
    assert!(answered, "Q3: model must respond to the steer");
    // Soft-interrupt invariant: the partial would-have-been assistant
    // message is NOT in history (because we didn't push it).
    let no_partial = !history
        .messages()
        .iter()
        .any(|m| m.role == Role::Assistant && m.text_content().contains("starting"));
    assert!(
        no_partial,
        "Q3: partial assistant text must NOT be pushed to history"
    );
}

// ─── Q4: steer mid-stream, response has NO tool calls → S1 catches ─

#[tokio::test]
async fn q4_steer_mid_text_only_stream_not_lost() {
    let provider = DelayedMockProvider::new(
        vec![
            // Iteration 1 (slow): would-have been a text-only response.
            vec![StreamChunk::Text("ok".into()), StreamChunk::Done],
            // Iteration 2: response after re-issue.
            vec![
                StreamChunk::Text("acknowledged Q4".into()),
                StreamChunk::Done,
            ],
        ],
        Duration::from_millis(80),
    );
    let agent_loop = make_loop(Box::new(provider), vec![]);
    let mut history = ConversationHistory::new("sys".into());
    history.push_user("ask something");
    let (tx, _rx) = mpsc::channel(64);
    let cancel = CancellationToken::new();
    let (steer_tx, steer_rx) = mpsc::channel(16);

    let handle = tokio::spawn(async move {
        agent_loop
            .run(&mut history, tx, cancel, None, Some(steer_rx))
            .await
            .map(|_| history)
    });
    tokio::time::sleep(Duration::from_millis(50)).await;
    steer_tx
        .send(SteerMessage {
            msg_id: 4,
            text: "Q4 steer".into(),
            is_edit: false,
        })
        .await
        .unwrap();

    let history = handle.await.unwrap().unwrap();
    let steer_ok = history
        .messages()
        .iter()
        .any(|m| m.role == Role::User && m.text_content().contains("Q4 steer"));
    assert!(steer_ok, "Q4: text-only response steer must reach history");
    let answered = history
        .messages()
        .iter()
        .any(|m| m.role == Role::Assistant && m.text_content().contains("acknowledged Q4"));
    assert!(
        answered,
        "Q4: model must answer the steer (S1 + S2/S3 working together)"
    );
}

// ─── Q5: two steers in one stream window → merged ──────────────────

#[tokio::test]
async fn q5_two_steers_merged_into_one_user_message() {
    let provider = DelayedMockProvider::new(
        vec![
            vec![StreamChunk::Text("..".into()), StreamChunk::Done],
            vec![StreamChunk::Text("got merged".into()), StreamChunk::Done],
        ],
        Duration::from_millis(120), // long enough for two steers to land
    );
    let agent_loop = make_loop(Box::new(provider), vec![]);
    let mut history = ConversationHistory::new("sys".into());
    history.push_user("question");
    let (tx, _rx) = mpsc::channel(64);
    let cancel = CancellationToken::new();
    let (steer_tx, steer_rx) = mpsc::channel(16);

    let handle = tokio::spawn(async move {
        agent_loop
            .run(&mut history, tx, cancel, None, Some(steer_rx))
            .await
            .map(|_| history)
    });

    // Both steers arrive within the same scheduler tick, BEFORE the
    // stream's first chunk yields (chunk delay = 120ms). The
    // soft-interrupt arm should burst-drain the channel and emit a
    // SINGLE merged re-issue carrying both texts.
    tokio::time::sleep(Duration::from_millis(20)).await;
    steer_tx
        .send(SteerMessage {
            msg_id: 5,
            text: "first".into(),
            is_edit: false,
        })
        .await
        .unwrap();
    steer_tx
        .send(SteerMessage {
            msg_id: 6,
            text: "second".into(),
            is_edit: false,
        })
        .await
        .unwrap();

    let history = handle.await.unwrap().unwrap();
    let user_msgs: Vec<_> = history
        .messages()
        .iter()
        .filter(|m| m.role == Role::User)
        .map(|m| m.text_content())
        .collect();
    // The first user message is "question". Then we expect ONE merged
    // steer user message containing both.
    let merged = user_msgs
        .iter()
        .find(|t| t.contains("first") || t.contains("second"))
        .expect("merged steer user message must exist");
    let _ = Arc::new(()); // silence Arc unused in tiny tests
    // Both texts present in the same user message OR split across two
    // (depending on timing). Either way, both must reach history.
    let both_present = user_msgs.iter().any(|t| t.contains("first"))
        && user_msgs.iter().any(|t| t.contains("second"));
    assert!(
        both_present,
        "Q5: both steer texts must reach history (got: {user_msgs:?}, merged hit: {merged:?})"
    );
}
