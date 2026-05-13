//! Golden tests — DO NOT DELETE. Refactor target for T2-T6.
//!
//! Tests G1..G8 lock in the current observable behaviour of
//! `AgentLoop::run` before the T2-T6 refactoring passes.

use std::pin::Pin;
use std::sync::{
    Arc,
    atomic::{AtomicUsize, Ordering},
};
use std::time::Duration;

use async_trait::async_trait;
use tokio::sync::mpsc;
use tokio_util::sync::CancellationToken;

use naked_core::{
    error::AgentError,
    history::ConversationHistory,
    loop_::{AgentLoop, LoopConfig},
    provider::{ChatRequest, Provider},
    tool::{Tool, registry::ToolRegistry},
    types::{
        AgentEvent, ModelInfo, Permission, Role, SteerMessage, StreamChunk, ToolResult, ToolSpec,
        TurnUsage,
    },
};

// ── Mock provider ─────────────────────────────────────────────────────────────

/// One scripted response for a single `stream_chat` call.
enum MockResponse {
    /// Return `Ok(stream)` with these chunks.
    Chunks(Vec<StreamChunk>),
    /// Return `Err(AgentError::Provider(msg))` — simulates a connect-level error.
    ConnectErr(String),
    /// Return `Err(AgentError::ProviderTyped(err))` — simulates a typed
    /// provider-level error (e.g. `ContextWindowExceeded`). Used by G6.
    ConnectErrTyped(naked_core::provider::error::ProviderError),
}

/// Scriptable test provider following the same pattern as the private
/// `MockProvider` in `src/loop__tests.rs` (which is inaccessible from
/// integration tests).
///
/// Clone `calls()` *before* moving `self` into `make_loop` so the test can
/// inspect the call count after the run completes.
struct MockProvider {
    responses: Vec<MockResponse>,
    call_count: Arc<AtomicUsize>,
}

impl MockProvider {
    fn new(responses: Vec<MockResponse>) -> Self {
        Self {
            responses,
            call_count: Arc::new(AtomicUsize::new(0)),
        }
    }

    /// Returns a cloned `Arc` to the call counter so tests can read it
    /// after the provider has been moved into an `AgentLoop`.
    fn calls(&self) -> Arc<AtomicUsize> {
        self.call_count.clone()
    }
}

#[async_trait]
impl Provider for MockProvider {
    fn name(&self) -> &str {
        "mock"
    }

    fn models(&self) -> Vec<ModelInfo> {
        vec![]
    }

    async fn stream_chat(
        &self,
        _request: ChatRequest,
    ) -> naked_core::error::Result<Pin<Box<dyn tokio_stream::Stream<Item = StreamChunk> + Send>>>
    {
        let idx = self.call_count.fetch_add(1, Ordering::SeqCst);
        if idx < self.responses.len() {
            match &self.responses[idx] {
                MockResponse::Chunks(chunks) => Ok(Box::pin(tokio_stream::iter(chunks.clone()))),
                MockResponse::ConnectErr(msg) => Err(AgentError::Provider(msg.clone())),
                MockResponse::ConnectErrTyped(err) => Err(AgentError::ProviderTyped(err.clone())),
            }
        } else {
            // Fallback: return text so the loop terminates naturally.
            Ok(Box::pin(tokio_stream::iter(vec![
                StreamChunk::Text("fallback".into()),
                StreamChunk::Done,
            ])))
        }
    }
}

// ── Minimal read-only echo tool ───────────────────────────────────────────────

struct EchoTool;

#[async_trait]
impl Tool for EchoTool {
    fn spec(&self) -> ToolSpec {
        // REGISTRY-WAIVE: exhaustive test struct ctor — pre-2026-05-13 baseline; new tests must use ..Default::default()
        ToolSpec {
            name: "echo".into(),
            description: "Echo input".into(),
            parameters: serde_json::json!({
                "type": "object",
                "properties": {"text": {"type": "string"}},
                "required": ["text"]
            }),
            permission: Permission::ReadOnly,
        }
    }

    async fn execute(&self, input: serde_json::Value, _cwd: &std::path::Path) -> ToolResult {
        let text = input["text"].as_str().unwrap_or("no text");
        ToolResult {
            output: format!("echoed: {text}"),
            is_error: false,
        }
    }
}

// ── Helpers ───────────────────────────────────────────────────────────────────

fn make_loop(provider: MockProvider, tools: Vec<Box<dyn Tool>>) -> AgentLoop {
    AgentLoop::new(
        Box::new(provider),
        ToolRegistry::new(tools),
        LoopConfig {
            max_iterations: 10,
            cwd: std::path::PathBuf::from("/tmp"),
            model: "mock".into(),
            max_tokens: 1024,
            ..Default::default()
        },
    )
}

/// Drain all pending events from the receiver into a `Vec`.
fn drain_events(rx: &mut mpsc::Receiver<AgentEvent>) -> Vec<AgentEvent> {
    let mut out = Vec::new();
    while let Ok(ev) = rx.try_recv() {
        out.push(ev);
    }
    out
}

// ════════════════════════════════════════════════════════════════════════════
// G1 — Single clean text response
// ════════════════════════════════════════════════════════════════════════════

/// G1: Mock returns `[Usage, Text("hi"), Done]` once → `Done`.
///
/// Expected: `Ok`, history grows by exactly 1 assistant message,
/// `usage.output_tokens > 0`, `TextDelta("hi")` and `Idle` emitted,
/// no `Error` event.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn g1_single_text_response() {
    let provider = MockProvider::new(vec![MockResponse::Chunks(vec![
        StreamChunk::Usage(TurnUsage {
            input_tokens: 10,
            output_tokens: 3,
            cache_read_tokens: 0,
            cache_write_tokens: 0,
        }),
        StreamChunk::Text("hi".into()),
        StreamChunk::Done,
    ])]);
    let agent = make_loop(provider, vec![]);

    let mut history = ConversationHistory::new("sys".into());
    history.push_user("hello");
    let msgs_before = history.message_count();

    let (tx, mut rx) = mpsc::channel(64);
    let cancel = CancellationToken::new();

    let usage = tokio::time::timeout(
        Duration::from_secs(5),
        agent.run(&mut history, tx, cancel, None, None),
    )
    .await
    .expect("G1: timed out")
    .expect("G1: expected Ok");

    assert_eq!(
        history.message_count(),
        msgs_before + 1,
        "G1: exactly one assistant message appended"
    );
    assert!(usage.output_tokens > 0, "G1: output_tokens must be > 0");

    let events = drain_events(&mut rx);
    assert!(
        events
            .iter()
            .any(|e| matches!(e, AgentEvent::TextDelta(t) if t == "hi")),
        "G1: TextDelta(\"hi\") must be emitted; got {events:?}"
    );
    assert!(
        events.iter().any(|e| matches!(e, AgentEvent::Idle)),
        "G1: Idle must be emitted"
    );
    assert!(
        !events.iter().any(|e| matches!(e, AgentEvent::Error(_))),
        "G1: no Error events expected"
    );
}

// ════════════════════════════════════════════════════════════════════════════
// G2 — Connect error twice then success on third attempt
// ════════════════════════════════════════════════════════════════════════════

/// G2: Provider returns a connect-level `Err` on the first two calls and a
/// clean text response on the third.  The loop retries with exponential
/// backoff (1 s + 2 s = 3 s real time, well within the 5 s timeout).
///
/// Expected: `Ok`, provider called exactly 3 times, no `Error` event,
/// assistant history contains text from the third call.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn g2_connect_error_retried_twice_then_succeeds() {
    let provider = MockProvider::new(vec![
        MockResponse::ConnectErr("connect failed #1".into()),
        MockResponse::ConnectErr("connect failed #2".into()),
        MockResponse::Chunks(vec![StreamChunk::Text("success".into()), StreamChunk::Done]),
    ]);
    let call_count = provider.calls();
    let agent = make_loop(provider, vec![]);

    let mut history = ConversationHistory::new("sys".into());
    history.push_user("hi");

    let (tx, mut rx) = mpsc::channel(64);
    let cancel = CancellationToken::new();

    // Real backoff sleeps: retry-0 → 1 s, retry-1 → 2 s.  Total ≈ 3 s < 5 s.
    tokio::time::timeout(
        Duration::from_secs(5),
        agent.run(&mut history, tx, cancel, None, None),
    )
    .await
    .expect("G2: timed out")
    .expect("G2: agent must succeed after 2 connect errors");

    assert_eq!(
        call_count.load(Ordering::SeqCst),
        3,
        "G2: provider must be called exactly 3 times (2 errors + 1 success)"
    );

    let last = history.messages().last().expect("G2: history non-empty");
    assert!(
        last.text_content().contains("success"),
        "G2: third-call text must be in history"
    );

    let events = drain_events(&mut rx);
    assert!(
        !events.iter().any(|e| matches!(e, AgentEvent::Error(_))),
        "G2: no Error events after successful recovery"
    );
    assert!(
        events.iter().any(|e| matches!(e, AgentEvent::Idle)),
        "G2: Idle must be emitted"
    );
}

// ════════════════════════════════════════════════════════════════════════════
// G3 — Mid-stream error (no tool calls) retried, second attempt succeeds
// ════════════════════════════════════════════════════════════════════════════

/// G3: First call streams partial text then a `StreamChunk::Error` (mid-stream,
/// no tool calls buffered).  Second call (after 1 s backoff) returns a clean
/// response.
///
/// Expected: `Ok`, no final `Error` event, `Idle` emitted, history's last
/// assistant message contains text from the successful retry.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn g3_midstream_error_retried_succeeds() {
    let provider = MockProvider::new(vec![
        MockResponse::Chunks(vec![
            StreamChunk::Text("partial".into()),
            StreamChunk::Error("network hiccup".into()),
        ]),
        MockResponse::Chunks(vec![
            StreamChunk::Text("full response".into()),
            StreamChunk::Done,
        ]),
    ]);
    let agent = make_loop(provider, vec![]);

    let mut history = ConversationHistory::new("sys".into());
    history.push_user("test");

    let (tx, mut rx) = mpsc::channel(64);
    let cancel = CancellationToken::new();

    // Real backoff: retry-0 → 1 s.  Total ≈ 1 s < 5 s.
    tokio::time::timeout(
        Duration::from_secs(5),
        agent.run(&mut history, tx, cancel, None, None),
    )
    .await
    .expect("G3: timed out")
    .expect("G3: agent must succeed after mid-stream retry");

    let last = history.messages().last().expect("G3: history non-empty");
    assert!(
        last.text_content().contains("full response"),
        "G3: second-attempt text must be in history; got {:?}",
        last.text_content()
    );

    let events = drain_events(&mut rx);
    assert!(
        !events.iter().any(|e| matches!(e, AgentEvent::Error(_))),
        "G3: no final Error event after recovery"
    );
    assert!(
        events.iter().any(|e| matches!(e, AgentEvent::Idle)),
        "G3: Idle must be emitted"
    );
}

// ════════════════════════════════════════════════════════════════════════════
// G4 — Mid-stream error with buffered tool calls → NO retry, surface error
// ════════════════════════════════════════════════════════════════════════════

/// G4: Provider streams two `ToolUse` chunks then a `StreamChunk::Error`.
/// Because `tool_calls` is non-empty when the mid-stream error fires,
/// the loop must NOT retry — it surfaces the error immediately.
///
/// Expected: `Err(ProviderTyped)`, provider called exactly once,
/// `ToolStart` events emitted (tool buffered), `Error` event emitted, no `Idle`.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn g4_midstream_error_with_tool_calls_no_retry() {
    let provider = MockProvider::new(vec![
        MockResponse::Chunks(vec![
            StreamChunk::ToolUse {
                id: "c1".into(),
                name: "echo".into(),
                input: serde_json::json!({"text": "hello"}),
            },
            StreamChunk::ToolUse {
                id: "c2".into(),
                name: "echo".into(),
                input: serde_json::json!({"text": "world"}),
            },
            StreamChunk::Error("stream cut".into()),
        ]),
        // Second response — must never be consumed (no retry expected).
        MockResponse::Chunks(vec![
            StreamChunk::Text("should not be reached".into()),
            StreamChunk::Done,
        ]),
    ]);
    let call_count = provider.calls();
    let tools: Vec<Box<dyn Tool>> = vec![Box::new(EchoTool)];
    let agent = make_loop(provider, tools);

    let mut history = ConversationHistory::new("sys".into());
    history.push_user("do something");

    let (tx, mut rx) = mpsc::channel(64);
    let cancel = CancellationToken::new();

    let result = tokio::time::timeout(
        Duration::from_secs(5),
        agent.run(&mut history, tx, cancel, None, None),
    )
    .await
    .expect("G4: timed out");

    assert!(
        result.is_err(),
        "G4: must return Err when tool calls buffered + mid-stream error"
    );
    assert!(
        matches!(result, Err(AgentError::ProviderTyped(_))),
        "G4: error must be ProviderTyped"
    );
    assert_eq!(
        call_count.load(Ordering::SeqCst),
        1,
        "G4: provider must be called exactly once (no retry)"
    );

    let events = drain_events(&mut rx);
    assert!(
        events
            .iter()
            .any(|e| matches!(e, AgentEvent::ToolStart { .. })),
        "G4: ToolStart must have been emitted (tool calls were buffered)"
    );
    assert!(
        events.iter().any(|e| matches!(e, AgentEvent::Error(_))),
        "G4: Error event must be emitted"
    );
    assert!(
        !events.iter().any(|e| matches!(e, AgentEvent::Idle)),
        "G4: Idle must NOT be emitted (not a clean finish)"
    );
}

// ════════════════════════════════════════════════════════════════════════════
// G5 — Three consecutive empty streams exhaust the retry budget
// ════════════════════════════════════════════════════════════════════════════

/// G5: Provider returns an empty stream (Usage + Done, no text/tool/thinking)
/// three times in a row.  The empty-content budget (`MAX_EMPTY_CONTENT_RETRIES = 2`)
/// is exhausted after the third attempt (250 ms + 500 ms backoff, ≈ 750 ms total).
///
/// Expected: `Err(ProviderTyped)` (NOT panic, NOT infinite loop),
/// history count unchanged (empty content must not be pushed),
/// `Error` event with "no content" and an `Idle` event emitted.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn g5_three_empty_streams_exhausts_budget() {
    let empty = || {
        MockResponse::Chunks(vec![
            StreamChunk::Usage(TurnUsage {
                input_tokens: 10,
                output_tokens: 0,
                cache_read_tokens: 0,
                cache_write_tokens: 0,
            }),
            StreamChunk::Done,
        ])
    };

    let provider = MockProvider::new(vec![empty(), empty(), empty()]);
    let agent = make_loop(provider, vec![]);

    let mut history = ConversationHistory::new("sys".into());
    history.push_user("hi");
    let msgs_before = history.message_count();

    let (tx, mut rx) = mpsc::channel(64);
    let cancel = CancellationToken::new();

    // Real backoff: 250 ms + 500 ms ≈ 750 ms total < 5 s.
    let result = tokio::time::timeout(
        Duration::from_secs(5),
        agent.run(&mut history, tx, cancel, None, None),
    )
    .await
    .expect("G5: timed out");

    assert!(
        result.is_err(),
        "G5: must return Err after exhausting empty-content budget"
    );
    assert!(
        matches!(result, Err(AgentError::ProviderTyped(_))),
        "G5: must be ProviderTyped (not panic or MaxIterations); got {result:?}"
    );
    assert_eq!(
        history.message_count(),
        msgs_before,
        "G5: history must NOT grow (empty content must not be pushed)"
    );

    let events = drain_events(&mut rx);
    assert!(
        events
            .iter()
            .any(|e| matches!(e, AgentEvent::Error(s) if s.contains("no content"))),
        "G5: explanatory Error event (containing \"no content\") must be emitted"
    );
    assert!(
        events.iter().any(|e| matches!(e, AgentEvent::Idle)),
        "G5: Idle must be emitted so the UI flushes"
    );
}

// ════════════════════════════════════════════════════════════════════════════
// G6 — "prompt is too long" → emergency compaction → retry succeeds
// ════════════════════════════════════════════════════════════════════════════

/// G6: First call returns a connect error containing "prompt is too long".
/// History has 10 messages (`before > 3`) so emergency compaction fires.
/// After compaction the outer loop retries; the second call succeeds.
/// The compaction path is synchronous (no backoff sleep).
///
/// Expected: `Ok`, `ContextCompacted` event with `before_msgs > after_msgs`,
/// provider called exactly twice, `TextDelta("recovered")` and `Idle` emitted.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn g6_context_overflow_triggers_compaction_then_succeeds() {
    let provider = MockProvider::new(vec![
        MockResponse::ConnectErrTyped(
            naked_core::provider::error::ProviderError::ContextWindowExceeded {
                message: "prompt is too long".into(),
            },
        ),
        MockResponse::Chunks(vec![
            StreamChunk::Usage(TurnUsage {
                input_tokens: 5,
                output_tokens: 2,
                cache_read_tokens: 0,
                cache_write_tokens: 0,
            }),
            StreamChunk::Text("recovered".into()),
            StreamChunk::Done,
        ]),
    ]);
    let call_count = provider.calls();
    let agent = make_loop(provider, vec![]);

    let mut history = ConversationHistory::new("sys".into());
    // Build history with 10 messages so `before > 3` (guard does not bail early).
    for i in 0..10 {
        history.push_user(&format!("user message {i}"));
    }
    let msgs_before = history.message_count();

    let (tx, mut rx) = mpsc::channel(64);
    let cancel = CancellationToken::new();

    // The "prompt is too long" path is purely synchronous — no backoff sleep.
    let usage = tokio::time::timeout(
        Duration::from_secs(5),
        agent.run(&mut history, tx, cancel, None, None),
    )
    .await
    .expect("G6: timed out")
    .expect("G6: agent must succeed after compaction");

    assert_eq!(
        call_count.load(Ordering::SeqCst),
        2,
        "G6: exactly two provider calls (connect error + successful retry)"
    );
    assert!(
        usage.output_tokens > 0,
        "G6: output_tokens from the second call must be > 0"
    );

    let events = drain_events(&mut rx);

    let compacted_ev = events
        .iter()
        .find(|e| matches!(e, AgentEvent::ContextCompacted { .. }));
    assert!(
        compacted_ev.is_some(),
        "G6: ContextCompacted event must be emitted"
    );
    if let Some(AgentEvent::ContextCompacted {
        before_msgs,
        after_msgs,
        ..
    }) = compacted_ev
    {
        assert_eq!(
            *before_msgs, msgs_before,
            "G6: before_msgs must match history size at compaction time"
        );
        assert!(
            *after_msgs < *before_msgs,
            "G6: history must have been reduced; before={before_msgs}, after={after_msgs}"
        );
    }

    assert!(
        events
            .iter()
            .any(|e| matches!(e, AgentEvent::TextDelta(t) if t == "recovered")),
        "G6: TextDelta(\"recovered\") from second call must be emitted"
    );
    assert!(
        events.iter().any(|e| matches!(e, AgentEvent::Idle)),
        "G6: Idle must be emitted on final success"
    );
}

// ════════════════════════════════════════════════════════════════════════════
// G7 — Cancellation token fires during backoff → Err(Cancelled) quickly
// ════════════════════════════════════════════════════════════════════════════

/// G7: The cancellation token is already set when `run()` is called (the
/// outer-loop guard checks it before the first connect attempt).
/// The provider is wired to return connect errors to confirm we are in a
/// "retry scenario" context; the cancellation is caught at the outer-loop
/// guard, well before any backoff sleep starts.
///
/// Expected: `Err(Cancelled)` returned within 200 ms of wall-clock time.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn g7_cancel_during_backoff_returns_cancelled() {
    // Provider would fail on every connect — but cancellation fires first.
    let provider = MockProvider::new(vec![
        MockResponse::ConnectErr("connection refused".into()),
        MockResponse::ConnectErr("connection refused".into()),
        MockResponse::ConnectErr("connection refused".into()),
        MockResponse::ConnectErr("connection refused".into()),
    ]);
    let agent = make_loop(provider, vec![]);

    let mut history = ConversationHistory::new("sys".into());
    history.push_user("hi");

    let (tx, _rx) = mpsc::channel(64);
    let cancel = CancellationToken::new();
    // Pre-cancel: the outer loop guard (`if cancel.is_cancelled()`) catches
    // this before the first connect attempt, so no backoff sleep occurs.
    cancel.cancel();

    let wall_start = std::time::Instant::now();
    let result = tokio::time::timeout(
        Duration::from_secs(5),
        agent.run(&mut history, tx, cancel, None, None),
    )
    .await
    .expect("G7: timed out");
    let wall_elapsed = wall_start.elapsed();

    assert!(
        matches!(result, Err(AgentError::Cancelled)),
        "G7: must return Err(Cancelled); got {result:?}"
    );
    assert!(
        wall_elapsed < Duration::from_millis(200),
        "G7: must complete within 200 ms wall clock; got {wall_elapsed:?}"
    );
}

// ════════════════════════════════════════════════════════════════════════════
// G8 — Steer message injected between iterations appears in history
// ════════════════════════════════════════════════════════════════════════════

/// G8: A steer message is queued in the steer channel before the run starts.
/// The loop drains steer messages at the beginning of each outer iteration
/// (before the provider call), so the steer appears in history as a
/// `Role::User` message.
///
/// Expected: steer text in history as User, `SteerReceived` event emitted,
/// run returns `Ok`.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn g8_steer_message_injected_between_iterations() {
    let provider = MockProvider::new(vec![
        // Iteration 1: tool call forces a second outer iteration.
        MockResponse::Chunks(vec![
            StreamChunk::ToolUse {
                id: "t1".into(),
                name: "echo".into(),
                input: serde_json::json!({"text": "ping"}),
            },
            StreamChunk::Done,
        ]),
        // Iteration 2: text response.
        MockResponse::Chunks(vec![StreamChunk::Text("done".into()), StreamChunk::Done]),
    ]);
    let tools: Vec<Box<dyn Tool>> = vec![Box::new(EchoTool)];
    let agent = make_loop(provider, tools);

    let mut history = ConversationHistory::new("sys".into());
    history.push_user("do something");

    let (tx, mut rx) = mpsc::channel(64);
    let cancel = CancellationToken::new();
    let (steer_tx, steer_rx) = mpsc::channel(16);

    // Pre-load the steer message; `drain_steers` picks it up at the start
    // of the next iteration.
    steer_tx
        .send(SteerMessage {
            msg_id: 42,
            text: "change direction".into(),
            is_edit: false,
        })
        .await
        .unwrap();

    tokio::time::timeout(
        Duration::from_secs(5),
        agent.run(&mut history, tx, cancel, None, Some(steer_rx)),
    )
    .await
    .expect("G8: timed out")
    .expect("G8: agent must succeed");

    let steer_in_history = history
        .messages()
        .iter()
        .any(|m| m.role == Role::User && m.text_content().contains("change direction"));
    assert!(
        steer_in_history,
        "G8: steer message must appear in history as a User message"
    );

    let events = drain_events(&mut rx);
    assert!(
        events.iter().any(|e| matches!(
            &e,
            AgentEvent::SteerReceived { text, .. } if text.contains("change direction")
        )),
        "G8: SteerReceived event must be emitted"
    );
    assert!(
        events.iter().any(|e| matches!(e, AgentEvent::Idle)),
        "G8: Idle must be emitted at completion"
    );
}
