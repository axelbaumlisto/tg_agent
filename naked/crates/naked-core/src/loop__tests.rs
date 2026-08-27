use super::*;
use crate::provider::{ChatRequest, Provider};
use crate::tool::Tool;
use crate::types::{ContentBlock, Permission, Role, SteerMessage, ToolSpec, ToolState};
use std::pin::Pin;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

struct MockProvider {
    responses: Vec<Vec<StreamChunk>>,
    call_count: AtomicUsize,
}

impl MockProvider {
    fn new(responses: Vec<Vec<StreamChunk>>) -> Self {
        Self {
            responses,
            call_count: AtomicUsize::new(0),
        }
    }
}

#[async_trait::async_trait]
impl Provider for MockProvider {
    fn name(&self) -> &str {
        "mock"
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
            vec![StreamChunk::Text("fallback".into()), StreamChunk::Done]
        };
        Ok(Box::pin(tokio_stream::iter(chunks)))
    }
}

struct DoneAndQueueSteerProvider {
    steer_tx: mpsc::Sender<SteerMessage>,
    call_count: std::sync::Arc<AtomicUsize>,
}

struct DoneAndQueueSteerStream {
    steer_tx: mpsc::Sender<SteerMessage>,
    yielded: bool,
}

impl tokio_stream::Stream for DoneAndQueueSteerStream {
    type Item = StreamChunk;

    fn poll_next(
        mut self: Pin<&mut Self>,
        _cx: &mut std::task::Context<'_>,
    ) -> std::task::Poll<Option<Self::Item>> {
        if self.yielded {
            return std::task::Poll::Ready(None);
        }
        self.yielded = true;
        let _ = self.steer_tx.try_send(SteerMessage {
            msg_id: 70,
            text: "terminal empty steer".into(),
            is_edit: false,
        });
        std::task::Poll::Ready(Some(StreamChunk::Done))
    }
}

#[async_trait::async_trait]
impl Provider for DoneAndQueueSteerProvider {
    fn name(&self) -> &str {
        "done-and-queue-steer"
    }

    fn models(&self) -> Vec<crate::types::ModelInfo> {
        vec![]
    }

    async fn stream_chat(
        &self,
        _request: ChatRequest,
    ) -> crate::error::Result<Pin<Box<dyn tokio_stream::Stream<Item = StreamChunk> + Send>>> {
        self.call_count.fetch_add(1, Ordering::SeqCst);
        Ok(Box::pin(DoneAndQueueSteerStream {
            steer_tx: self.steer_tx.clone(),
            yielded: false,
        }))
    }
}

struct DelayedToolUseAndQueueSteerProvider {
    steer_tx: mpsc::Sender<SteerMessage>,
    delay: std::time::Duration,
    call_count: std::sync::Arc<AtomicUsize>,
}

struct DelayedTextProvider {
    delay: std::time::Duration,
    text: String,
}

struct HungStreamOpenProvider;

#[async_trait::async_trait]
impl Provider for HungStreamOpenProvider {
    fn name(&self) -> &str {
        "hung-stream-open"
    }

    fn models(&self) -> Vec<crate::types::ModelInfo> {
        vec![]
    }

    async fn stream_chat(
        &self,
        _request: ChatRequest,
    ) -> crate::error::Result<Pin<Box<dyn tokio_stream::Stream<Item = StreamChunk> + Send>>> {
        tokio::time::sleep(std::time::Duration::from_secs(60)).await;
        Ok(Box::pin(tokio_stream::iter(vec![StreamChunk::Done])))
    }
}

struct DelayedFailProvider {
    delay: std::time::Duration,
}

#[async_trait::async_trait]
impl Provider for DelayedFailProvider {
    fn name(&self) -> &str {
        "slow-qwen-fail"
    }

    fn models(&self) -> Vec<crate::types::ModelInfo> {
        vec![]
    }

    async fn stream_chat(
        &self,
        _request: ChatRequest,
    ) -> crate::error::Result<Pin<Box<dyn tokio_stream::Stream<Item = StreamChunk> + Send>>> {
        tokio::time::sleep(self.delay).await;
        Err(AgentError::ProviderTyped(
            crate::provider::error::ProviderError::Other {
                status: 503,
                body: "synthetic slow primary failure".into(),
            },
        ))
    }
}

#[async_trait::async_trait]
impl Provider for DelayedTextProvider {
    fn name(&self) -> &str {
        "delayed-text"
    }

    fn models(&self) -> Vec<crate::types::ModelInfo> {
        vec![]
    }

    async fn stream_chat(
        &self,
        _request: ChatRequest,
    ) -> crate::error::Result<Pin<Box<dyn tokio_stream::Stream<Item = StreamChunk> + Send>>> {
        let delay = self.delay;
        let text = self.text.clone();
        Ok(Box::pin(async_stream::stream! {
            tokio::time::sleep(delay).await;
            yield StreamChunk::Text(text);
            yield StreamChunk::Done;
        }))
    }
}

fn make_delayed_text_loop(delay: std::time::Duration, text: impl Into<String>) -> AgentLoop {
    AgentLoop::new(
        Box::new(DelayedTextProvider {
            delay,
            text: text.into(),
        }),
        crate::tool::registry::ToolRegistry::new(vec![]),
        LoopConfig {
            max_iterations: 10,
            max_wall: None,
            cwd: std::path::PathBuf::from("/tmp"),
            model: "delayed-text".into(),
            max_tokens: 1024,
            ..Default::default()
        },
    )
}

fn make_provider_backstop_loop(
    provider: Box<dyn Provider>,
    turn_backstop: std::time::Duration,
) -> AgentLoop {
    AgentLoop::new(
        provider,
        crate::tool::registry::ToolRegistry::new(vec![]),
        LoopConfig {
            max_iterations: 10,
            max_wall: Some(std::time::Duration::from_secs(60)),
            tool_deadline: Some(std::time::Duration::from_secs(60)),
            turn_backstop: Some(turn_backstop),
            cwd: std::path::PathBuf::from("/tmp"),
            model: "mock".into(),
            max_tokens: 1024,
            ..Default::default()
        },
    )
}

fn turn_bucket_delta_total(
    before: crate::metrics_hist::LatencySnapshot,
    after: crate::metrics_hist::LatencySnapshot,
) -> u64 {
    after.turn_under_1s.saturating_sub(before.turn_under_1s)
        + after.turn_1s_to_10s.saturating_sub(before.turn_1s_to_10s)
        + after.turn_10s_to_60s.saturating_sub(before.turn_10s_to_60s)
        + after.turn_over_60s.saturating_sub(before.turn_over_60s)
}

fn ttft_bucket_delta_total(
    before: crate::metrics_hist::LatencySnapshot,
    after: crate::metrics_hist::LatencySnapshot,
) -> u64 {
    after
        .ttft_under_500ms
        .saturating_sub(before.ttft_under_500ms)
        + after
            .ttft_500ms_to_2s
            .saturating_sub(before.ttft_500ms_to_2s)
        + after.ttft_2s_to_10s.saturating_sub(before.ttft_2s_to_10s)
        + after.ttft_over_10s.saturating_sub(before.ttft_over_10s)
}

#[async_trait::async_trait]
impl Provider for DelayedToolUseAndQueueSteerProvider {
    fn name(&self) -> &str {
        "delayed-tool-use-and-queue-steer"
    }

    fn models(&self) -> Vec<crate::types::ModelInfo> {
        vec![]
    }

    async fn stream_chat(
        &self,
        _request: ChatRequest,
    ) -> crate::error::Result<Pin<Box<dyn tokio_stream::Stream<Item = StreamChunk> + Send>>> {
        let idx = self.call_count.fetch_add(1, Ordering::SeqCst);
        if idx == 0 {
            let steer_tx = self.steer_tx.clone();
            let delay = self.delay;
            Ok(Box::pin(async_stream::stream! {
                tokio::time::sleep(delay).await;
                let _ = steer_tx.try_send(SteerMessage {
                    msg_id: 71,
                    text: "wall timeout steer".into(),
                    is_edit: false,
                });
                yield StreamChunk::ToolUse {
                    id: "call-timeout".into(),
                    name: "echo".into(),
                    input: serde_json::json!({"text": "force another iteration"}),
                };
                yield StreamChunk::Done;
            }))
        } else {
            Ok(Box::pin(tokio_stream::iter(vec![
                StreamChunk::Text("should not reach second provider call".into()),
                StreamChunk::Done,
            ])))
        }
    }
}

struct EchoTool;

#[async_trait::async_trait]
impl Tool for EchoTool {
    fn spec(&self) -> ToolSpec {
        // REGISTRY-WAIVE: exhaustive test struct ctor — pre-2026-05-13 baseline; new tests must use ..Default::default()
        ToolSpec {
            name: "echo".into(),
            description: "Echo input".into(),
            parameters: serde_json::json!({"type":"object","properties":{"text":{"type":"string"}},"required":["text"]}),
            permission: Permission::ReadOnly,
        }
    }

    async fn execute(
        &self,
        input: serde_json::Value,
        _cwd: &std::path::Path,
    ) -> crate::types::ToolResult {
        let text = input["text"].as_str().unwrap_or("no text");
        crate::types::ToolResult {
            output: format!("echoed: {text}"),
            is_error: false,
        }
    }
}

struct RecordingEchoTool {
    executions: std::sync::Arc<AtomicUsize>,
}

#[async_trait::async_trait]
impl Tool for RecordingEchoTool {
    fn spec(&self) -> ToolSpec {
        // REGISTRY-WAIVE: exhaustive test struct ctor — ToolSpec has no Default; this fixture must pin permission/name explicitly.
        ToolSpec {
            name: "echo".into(),
            description: "Recording echo input".into(),
            parameters: serde_json::json!({"type":"object","properties":{"text":{"type":"string"}},"required":["text"]}),
            permission: Permission::ReadOnly,
        }
    }

    async fn execute(
        &self,
        input: serde_json::Value,
        _cwd: &std::path::Path,
    ) -> crate::types::ToolResult {
        self.executions.fetch_add(1, Ordering::SeqCst);
        let text = input["text"].as_str().unwrap_or("no text");
        crate::types::ToolResult {
            output: format!("recorded: {text}"),
            is_error: false,
        }
    }
}

fn make_loop(provider: MockProvider, tools: Vec<Box<dyn Tool>>) -> AgentLoop {
    AgentLoop::new(
        Box::new(provider),
        crate::tool::registry::ToolRegistry::new(tools),
        LoopConfig {
            max_iterations: 10,
            max_wall: None,
            cwd: std::path::PathBuf::from("/tmp"),
            model: "mock".into(),
            max_tokens: 1024,
            ..Default::default()
        },
    )
}

#[tokio::test]
async fn loop_zero_token_turn_does_not_pollute_history() {
    // Regression for the "dirty-session 0-tok refusal" class of bugs.
    // Providers like glm-5-turbo can close a stream with no text, no
    // reasoning, and no tool calls. The loop tolerates a few of these
    // (see `loop_empty_then_text_retries_and_succeeds`) but if every
    // attempt comes back empty we must still surface an error WITHOUT
    // polluting history (otherwise every subsequent turn sees
    // `{role:assistant, content:[]}` and refuses in a loop).
    //
    // A clean terminal marker with zero content is genuine model silence
    // (B70), so it must fail once without retrying or hitting fallback.
    let provider = MockProvider::new(vec![vec![
        StreamChunk::Usage(TurnUsage {
            input_tokens: 42,
            output_tokens: 0,
            cache_read_tokens: 0,
            cache_write_tokens: 0,
        }),
        StreamChunk::Done,
    ]]);
    let agent_loop = make_loop(provider, vec![]);
    let mut history = ConversationHistory::new("sys".into());
    history.push_user("hi");
    let msgs_before = history.message_count();

    let (tx, mut rx) = mpsc::channel(64);
    let cancel = CancellationToken::new();

    let result = agent_loop.run(&mut history, tx, cancel, None, None).await;
    assert!(
        matches!(
            result,
            Err(AgentError::Provider(_)) | Err(AgentError::ProviderTyped(_))
        ),
        "exhausted-retry 0-token turn must surface as Provider error, not silent Ok; got {result:?}"
    );
    assert_eq!(
        history.message_count(),
        msgs_before,
        "empty assistant must NOT be appended to history"
    );

    let mut events = Vec::new();
    while let Ok(ev) = rx.try_recv() {
        events.push(ev);
    }
    assert!(
        events
            .iter()
            .any(|e| matches!(e, AgentEvent::Error(s) if s.contains("no content"))),
        "must emit an explanatory Error event; got events: {events:?}"
    );
    assert!(
        events.iter().any(|e| matches!(e, AgentEvent::Idle)),
        "must still emit Idle so the UI flushes"
    );
}

#[tokio::test]
async fn loop_empty_with_done_does_not_retry_and_drains_pending_steer() {
    let (steer_tx_for_provider, steer_rx) = mpsc::channel(16);
    let call_count = std::sync::Arc::new(AtomicUsize::new(0));
    let agent_loop = AgentLoop::new(
        Box::new(DoneAndQueueSteerProvider {
            steer_tx: steer_tx_for_provider,
            call_count: call_count.clone(),
        }),
        crate::tool::registry::ToolRegistry::new(vec![]),
        LoopConfig {
            max_iterations: 10,
            max_wall: None,
            cwd: std::path::PathBuf::from("/tmp"),
            model: "mock".into(),
            max_tokens: 1024,
            ..Default::default()
        },
    );
    let mut history = ConversationHistory::new("sys".into());
    history.push_user("ping");

    let (tx, mut rx) = mpsc::channel(64);
    let cancel = CancellationToken::new();

    let result = agent_loop
        .run(&mut history, tx, cancel, None, Some(steer_rx))
        .await;
    assert!(
        matches!(
            result,
            Err(AgentError::Provider(_)) | Err(AgentError::ProviderTyped(_))
        ),
        "clean empty-Done silence must fail terminally without retry; got {result:?}"
    );
    assert_eq!(
        call_count.load(Ordering::SeqCst),
        1,
        "empty + saw_done must not retry or hit fallback"
    );

    let steer_in_history = history
        .messages()
        .iter()
        .any(|m| m.role == Role::User && m.text_content().contains("terminal empty steer"));
    assert!(
        steer_in_history,
        "terminal empty cleanup must drain pending steer into history"
    );

    let mut events = Vec::new();
    while let Ok(ev) = rx.try_recv() {
        events.push(ev);
    }
    assert!(
        events
            .iter()
            .any(|e| matches!(e, AgentEvent::Error(s) if s.contains("no content"))),
        "terminal empty cleanup must emit Error; got events: {events:?}"
    );
    assert!(
        events.iter().any(|e| matches!(e, AgentEvent::Idle)),
        "terminal empty cleanup must emit Idle; got events: {events:?}"
    );
    assert!(
        events.iter().any(|e| matches!(e, AgentEvent::SteerReceived { text, .. } if text.contains("terminal empty steer"))),
        "terminal empty cleanup must emit SteerReceived while draining; got events: {events:?}"
    );
}

#[tokio::test]
async fn loop_empty_without_done_then_text_retries_and_succeeds() {
    // Defensive fallback for non-SSE/legacy providers: an empty stream
    // that ends without Done may be transport death, so the loop still
    // retries and surfaces the text from the second attempt.
    let provider = MockProvider::new(vec![
        // attempt 1: empty stream without terminal marker
        vec![],
        // attempt 2 (retry): real text
        vec![
            StreamChunk::Text("hi after retry".into()),
            StreamChunk::Done,
        ],
    ]);
    let agent_loop = make_loop(provider, vec![]);
    let mut history = ConversationHistory::new("sys".into());
    history.push_user("ping");

    let (tx, mut rx) = mpsc::channel(64);
    let cancel = CancellationToken::new();

    let result = agent_loop.run(&mut history, tx, cancel, None, None).await;
    assert!(
        result.is_ok(),
        "retry-after-empty must return Ok; got {result:?}"
    );

    let mut events = Vec::new();
    while let Ok(ev) = rx.try_recv() {
        events.push(ev);
    }
    let has_error = events
        .iter()
        .any(|e| matches!(e, AgentEvent::Error(s) if s.contains("no content")));
    assert!(
        !has_error,
        "recovered empty-content turn must NOT emit a final Error event; got events: {events:?}"
    );
    let text_combined: String = events
        .iter()
        .filter_map(|e| match e {
            AgentEvent::TextDelta(t) => Some(t.as_str()),
            _ => None,
        })
        .collect();
    assert_eq!(
        text_combined, "hi after retry",
        "second-attempt text must be delivered to the UI"
    );
}

#[tokio::test]
async fn loop_two_empty_then_text_succeeds_at_budget_edge() {
    // Verify the budget itself: `MAX_EMPTY_CONTENT_RETRIES = 2` means
    // we tolerate up to 2 retries (so 3 attempts total). Two empties
    // followed by text must still succeed — exhausting the retry
    // budget on the very last attempt.
    let provider = MockProvider::new(vec![
        vec![],
        vec![],
        vec![
            StreamChunk::Text("third time lucky".into()),
            StreamChunk::Done,
        ],
    ]);
    let agent_loop = make_loop(provider, vec![]);
    let mut history = ConversationHistory::new("sys".into());
    history.push_user("ping");

    let (tx, mut rx) = mpsc::channel(64);
    let cancel = CancellationToken::new();

    let result = agent_loop.run(&mut history, tx, cancel, None, None).await;
    assert!(
        result.is_ok(),
        "two empties + text must still succeed at the edge of the retry budget; got {result:?}"
    );

    let mut text_combined = String::new();
    while let Ok(ev) = rx.try_recv() {
        if let AgentEvent::TextDelta(t) = ev {
            text_combined.push_str(&t);
        }
    }
    assert_eq!(text_combined, "third time lucky");
}

#[tokio::test]
async fn loop_wall_timeout_at_iteration_boundary_drains_pending_steer() {
    let (steer_tx_for_provider, steer_rx) = mpsc::channel(16);
    let call_count = std::sync::Arc::new(AtomicUsize::new(0));
    let agent_loop = AgentLoop::new(
        Box::new(DelayedToolUseAndQueueSteerProvider {
            steer_tx: steer_tx_for_provider,
            delay: std::time::Duration::from_millis(5),
            call_count: call_count.clone(),
        }),
        crate::tool::registry::ToolRegistry::new(vec![Box::new(EchoTool)]),
        LoopConfig {
            max_iterations: 10,
            max_wall: Some(std::time::Duration::from_millis(1)),
            cwd: std::path::PathBuf::from("/tmp"),
            model: "mock".into(),
            max_tokens: 1024,
            ..Default::default()
        },
    );
    let mut history = ConversationHistory::new("sys".into());
    history.push_user("force tool loop");

    let (tx, mut rx) = mpsc::channel(64);
    let cancel = CancellationToken::new();

    let result = agent_loop
        .run(&mut history, tx, cancel, None, Some(steer_rx))
        .await;
    assert!(
        matches!(result, Err(AgentError::WallTimeout)),
        "wall budget must abort at the iteration boundary; got {result:?}"
    );
    assert_eq!(
        call_count.load(Ordering::SeqCst),
        1,
        "timeout must happen before the second provider call"
    );

    let steer_in_history = history
        .messages()
        .iter()
        .any(|m| m.role == Role::User && m.text_content().contains("wall timeout steer"));
    assert!(
        steer_in_history,
        "wall-timeout cleanup must drain pending steer into history"
    );

    let mut events = Vec::new();
    while let Ok(ev) = rx.try_recv() {
        events.push(ev);
    }
    assert!(
        events.iter().any(
            |e| matches!(e, AgentEvent::Error(s) if s.contains(crate::error::WALL_TIMEOUT_MESSAGE))
        ),
        "wall-timeout cleanup must emit Error; got events: {events:?}"
    );
    assert!(
        events.iter().any(|e| matches!(e, AgentEvent::Idle)),
        "wall-timeout cleanup must emit Idle; got events: {events:?}"
    );
    assert!(
        events.iter().any(|e| matches!(e, AgentEvent::SteerReceived { text, .. } if text.contains("wall timeout steer"))),
        "wall-timeout cleanup must emit SteerReceived while draining; got events: {events:?}"
    );
}

#[tokio::test]
async fn loop_max_wall_none_preserves_existing_tool_flow() {
    let provider = MockProvider::new(vec![
        vec![
            StreamChunk::ToolUse {
                id: "call1".into(),
                name: "echo".into(),
                input: serde_json::json!({"text": "ping"}),
            },
            StreamChunk::Done,
        ],
        vec![StreamChunk::Text("Done!".into()), StreamChunk::Done],
    ]);

    let tools: Vec<Box<dyn Tool>> = vec![Box::new(EchoTool)];
    let agent_loop = make_loop(provider, tools);
    let mut history = ConversationHistory::new("sys".into());
    history.push_user("test");

    let (tx, mut rx) = mpsc::channel(64);
    let cancel = CancellationToken::new();

    let result = agent_loop.run(&mut history, tx, cancel, None, None).await;
    assert!(
        result.is_ok(),
        "max_wall=None must preserve existing flow: {result:?}"
    );

    let mut events = Vec::new();
    while let Ok(ev) = rx.try_recv() {
        events.push(ev);
    }
    assert!(events.iter().any(|e| matches!(e, AgentEvent::Idle)));
    assert!(
        history
            .messages()
            .iter()
            .any(|m| m.text_content().contains("Done!"))
    );
}

#[tokio::test]
async fn loop_text_only_response() {
    let provider = MockProvider::new(vec![vec![
        StreamChunk::Text("Hello ".into()),
        StreamChunk::Text("world".into()),
        StreamChunk::Done,
    ]]);
    let agent_loop = make_loop(provider, vec![]);
    let mut history = ConversationHistory::new("sys".into());
    history.push_user("hi");

    let (tx, mut rx) = mpsc::channel(64);
    let cancel = CancellationToken::new();

    let result = agent_loop.run(&mut history, tx, cancel, None, None).await;
    assert!(result.is_ok());

    let mut events = Vec::new();
    while let Ok(ev) = rx.try_recv() {
        events.push(ev);
    }

    let text_events: Vec<_> = events
        .iter()
        .filter(|e| matches!(e, AgentEvent::TextDelta(_)))
        .collect();
    assert_eq!(text_events.len(), 2);

    assert!(events.iter().any(|e| matches!(e, AgentEvent::Idle)));

    assert_eq!(history.message_count(), 2);
    assert_eq!(history.messages()[1].text_content(), "Hello world");
}

#[tokio::test]
async fn loop_tool_use_flow() {
    let provider = MockProvider::new(vec![
        vec![
            StreamChunk::ToolUse {
                id: "call1".into(),
                name: "echo".into(),
                input: serde_json::json!({"text": "ping"}),
            },
            StreamChunk::Done,
        ],
        vec![StreamChunk::Text("Done!".into()), StreamChunk::Done],
    ]);

    let tools: Vec<Box<dyn Tool>> = vec![Box::new(EchoTool)];
    let agent_loop = make_loop(provider, tools);
    let mut history = ConversationHistory::new("sys".into());
    history.push_user("test");

    let (tx, mut rx) = mpsc::channel(64);
    let cancel = CancellationToken::new();

    let result = agent_loop.run(&mut history, tx, cancel, None, None).await;
    assert!(result.is_ok());

    let mut events = Vec::new();
    while let Ok(ev) = rx.try_recv() {
        events.push(ev);
    }

    assert!(
        events
            .iter()
            .any(|e| matches!(e, AgentEvent::ToolStart { .. }))
    );
    assert!(
        events
            .iter()
            .any(|e| matches!(e, AgentEvent::ToolEnd { .. }))
    );

    // 1:user, 2:assistant(tool_use), 3:tool_result, 4:assistant(text)
    assert_eq!(history.message_count(), 4);
}

#[tokio::test]
async fn loop_cancellation() {
    let provider = MockProvider::new(vec![vec![
        StreamChunk::Text("start".into()),
        StreamChunk::Done,
    ]]);
    let agent_loop = make_loop(provider, vec![]);
    let mut history = ConversationHistory::new("sys".into());
    history.push_user("hi");

    let (tx, _rx) = mpsc::channel(64);
    let cancel = CancellationToken::new();
    cancel.cancel();

    let result = agent_loop.run(&mut history, tx, cancel, None, None).await;
    assert!(matches!(result, Err(AgentError::Cancelled)));
}

#[tokio::test]
async fn loop_max_iterations() {
    // Provider always returns tool use, forcing infinite loop
    let responses: Vec<Vec<StreamChunk>> = (0..15)
        .map(|i| {
            vec![
                StreamChunk::ToolUse {
                    id: format!("c{i}"),
                    name: "echo".into(),
                    input: serde_json::json!({"text": "loop"}),
                },
                StreamChunk::Done,
            ]
        })
        .collect();

    let provider = MockProvider::new(responses);
    let tools: Vec<Box<dyn Tool>> = vec![Box::new(EchoTool)];
    let agent_loop = AgentLoop::new(
        Box::new(provider),
        crate::tool::registry::ToolRegistry::new(tools),
        LoopConfig {
            max_iterations: 3,
            max_wall: None,
            cwd: std::path::PathBuf::from("/tmp"),
            model: "mock".into(),
            max_tokens: 1024,
            ..Default::default()
        },
    );
    let mut history = ConversationHistory::new("sys".into());
    history.push_user("hi");

    let (tx, _rx) = mpsc::channel(64);
    let cancel = CancellationToken::new();

    let result = agent_loop.run(&mut history, tx, cancel, None, None).await;
    assert!(matches!(result, Err(AgentError::MaxIterations(3))));
}

#[tokio::test]
async fn agent_turn_duration_records_on_successful_text_turn() {
    let _guard = crate::metrics_hist::async_test_guard().await;
    let before = crate::metrics_hist::snapshot();
    let agent_loop = make_delayed_text_loop(std::time::Duration::from_millis(600), "delayed hello");
    let mut history = ConversationHistory::new("sys".into());
    history.push_user("hi");

    let (tx, mut rx) = mpsc::channel(64);
    let cancel = CancellationToken::new();

    let result = agent_loop.run(&mut history, tx, cancel, None, None).await;
    assert!(result.is_ok());

    let mut saw_text_delta = false;
    while let Ok(ev) = rx.try_recv() {
        if matches!(ev, AgentEvent::TextDelta(text) if text == "delayed hello") {
            saw_text_delta = true;
        }
    }
    assert!(saw_text_delta, "delayed text turn must emit TextDelta");

    let after = crate::metrics_hist::snapshot();
    let turn_sum_delta = after.turn_sum_ms.saturating_sub(before.turn_sum_ms);
    let turn_count_delta = after.turn_count.saturating_sub(before.turn_count);
    // `async_test_guard` serialises metrics tests that reset the process-global
    // counters. Other AgentLoop tests may still record additional turn samples,
    // so assert the sample landed instead of asserting nobody else existed.
    assert!(
        turn_count_delta >= 1,
        "this test's turn must be counted (delta {turn_count_delta})"
    );
    assert!(
        turn_sum_delta >= 600,
        "turn sum delta was {turn_sum_delta}ms"
    );
    // Do not pick the bucket from `turn_sum_delta`: that delta is the sum over
    // every turn recorded in the window, so a concurrent turn can select a
    // bucket this sample never landed in. This test's 600ms provider delay
    // normally lands in `turn_under_1s` (the edge is 1000ms), but scheduler
    // load can push the observed turn past that edge. Asserting the total
    // covers "our sample was bucketed" without guessing which band was used.
    assert!(
        turn_bucket_delta_total(before, after) >= 1,
        "turn bucket total must include at least this test's sample"
    );
}

#[tokio::test]
async fn agent_ttft_records_first_text_delta_after_provider_delay() {
    let _guard = crate::metrics_hist::async_test_guard().await;
    let before = crate::metrics_hist::snapshot();
    let agent_loop = make_delayed_text_loop(std::time::Duration::from_millis(600), "delayed ttft");
    let mut history = ConversationHistory::new("sys".into());
    history.push_user("hi");

    let (tx, _rx) = mpsc::channel(64);
    let cancel = CancellationToken::new();

    let result = agent_loop.run(&mut history, tx, cancel, None, None).await;
    assert!(result.is_ok());

    let after = crate::metrics_hist::snapshot();
    let ttft_sum_delta = after.ttft_sum_ms.saturating_sub(before.ttft_sum_ms);
    let ttft_count_delta = after.ttft_count.saturating_sub(before.ttft_count);
    assert!(
        ttft_count_delta >= 1,
        "ttft count must include at least this test's sample"
    );
    assert!(
        ttft_sum_delta >= 600,
        "ttft sum delta was {ttft_sum_delta}ms"
    );
    assert!(
        ttft_bucket_delta_total(before, after) >= 1,
        "ttft bucket total must include at least this test's sample"
    );
    let turn_count_delta = after.turn_count.saturating_sub(before.turn_count);
    assert!(
        turn_count_delta >= 1,
        "turn duration count should also increase during the successful turn"
    );
}

#[tokio::test]
async fn agent_turn_duration_records_on_provider_error() {
    let _guard = crate::metrics_hist::async_test_guard().await;
    let before = crate::metrics_hist::snapshot();
    let provider = MockProvider::new(vec![
        vec![StreamChunk::Error("API overloaded".into())],
        vec![StreamChunk::Error("API overloaded".into())],
        vec![StreamChunk::Error("API overloaded".into())],
        vec![StreamChunk::Error("API overloaded".into())],
    ]);
    let agent_loop = make_loop(provider, vec![]);
    let mut history = ConversationHistory::new("sys".into());
    history.push_user("hi");

    let (tx, _rx) = mpsc::channel(64);
    let cancel = CancellationToken::new();

    let result = agent_loop.run(&mut history, tx, cancel, None, None).await;
    assert!(matches!(
        result,
        Err(AgentError::Provider(_)) | Err(AgentError::ProviderTyped(_))
    ));

    let after = crate::metrics_hist::snapshot();
    let turn_count_delta = after.turn_count.saturating_sub(before.turn_count);
    assert!(
        turn_count_delta >= 1,
        "provider error path must record at least one turn duration sample"
    );
    assert!(
        turn_bucket_delta_total(before, after) >= 1,
        "turn bucket total must include at least this test's sample"
    );
}

#[tokio::test]
async fn loop_provider_error() {
    // Must provide enough error responses for all retry attempts (MAX_STREAM_RETRIES + 1)
    let provider = MockProvider::new(vec![
        vec![StreamChunk::Error("API overloaded".into())],
        vec![StreamChunk::Error("API overloaded".into())],
        vec![StreamChunk::Error("API overloaded".into())],
        vec![StreamChunk::Error("API overloaded".into())],
    ]);
    let agent_loop = make_loop(provider, vec![]);
    let mut history = ConversationHistory::new("sys".into());
    history.push_user("hi");

    let (tx, mut rx) = mpsc::channel(64);
    let cancel = CancellationToken::new();

    let result = agent_loop.run(&mut history, tx, cancel, None, None).await;
    assert!(matches!(
        result,
        Err(AgentError::Provider(_)) | Err(AgentError::ProviderTyped(_))
    ));

    let mut events = Vec::new();
    while let Ok(ev) = rx.try_recv() {
        events.push(ev);
    }
    assert!(events.iter().any(|e| matches!(e, AgentEvent::Error(_))));
}

#[tokio::test]
async fn loop_usage_accumulates() {
    let provider = MockProvider::new(vec![vec![
        StreamChunk::Usage(TurnUsage {
            input_tokens: 100,
            output_tokens: 50,
            cache_read_tokens: 0,
            cache_write_tokens: 0,
        }),
        StreamChunk::Text("ok".into()),
        StreamChunk::Done,
    ]]);
    let agent_loop = make_loop(provider, vec![]);
    let mut history = ConversationHistory::new("sys".into());
    history.push_user("hi");

    let (tx, _rx) = mpsc::channel(64);
    let cancel = CancellationToken::new();

    let result = agent_loop
        .run(&mut history, tx, cancel, None, None)
        .await
        .unwrap();
    assert_eq!(result.input_tokens, 100);
    assert_eq!(result.output_tokens, 50);
}

#[tokio::test]
async fn loop_thinking_events_emitted() {
    let provider = MockProvider::new(vec![vec![
        StreamChunk::Thinking("let me think...".into()),
        StreamChunk::Text("answer".into()),
        StreamChunk::Done,
    ]]);
    let agent_loop = make_loop(provider, vec![]);
    let mut history = ConversationHistory::new("sys".into());
    history.push_user("hi");

    let (tx, mut rx) = mpsc::channel(64);
    let cancel = CancellationToken::new();

    agent_loop
        .run(&mut history, tx, cancel, None, None)
        .await
        .unwrap();

    let mut events = Vec::new();
    while let Ok(ev) = rx.try_recv() {
        events.push(ev);
    }
    assert!(
        events
            .iter()
            .any(|e| matches!(e, AgentEvent::ThinkingDelta(t) if t == "let me think..."))
    );
}

#[tokio::test]
async fn loop_thinking_saved_to_history() {
    let provider = MockProvider::new(vec![vec![
        StreamChunk::Thinking("step 1\n".into()),
        StreamChunk::Thinking("step 2".into()),
        StreamChunk::Text("answer".into()),
        StreamChunk::Done,
    ]]);
    let agent_loop = make_loop(provider, vec![]);
    let mut history = ConversationHistory::new("sys".into());
    history.push_user("hi");

    let (tx, _rx) = mpsc::channel(64);
    let cancel = CancellationToken::new();

    agent_loop
        .run(&mut history, tx, cancel, None, None)
        .await
        .unwrap();

    assert_eq!(history.message_count(), 2);
    let assistant_msg = &history.messages()[1];
    assert_eq!(assistant_msg.blocks.len(), 2);
    assert!(matches!(
        &assistant_msg.blocks[0],
        ContentBlock::Thinking { text } if text == "step 1\nstep 2"
    ));
    assert!(matches!(
        &assistant_msg.blocks[1],
        ContentBlock::Text { text } if text == "answer"
    ));
}

#[tokio::test]
async fn loop_thinking_flushed_before_tool_use() {
    let provider = MockProvider::new(vec![
        vec![
            StreamChunk::Thinking("reasoning".into()),
            StreamChunk::ToolUse {
                id: "c1".into(),
                name: "echo".into(),
                input: serde_json::json!({"text": "hi"}),
            },
            StreamChunk::Done,
        ],
        vec![StreamChunk::Text("done".into()), StreamChunk::Done],
    ]);

    let tools: Vec<Box<dyn Tool>> = vec![Box::new(EchoTool)];
    let agent_loop = make_loop(provider, tools);
    let mut history = ConversationHistory::new("sys".into());
    history.push_user("test");

    let (tx, _rx) = mpsc::channel(64);
    let cancel = CancellationToken::new();

    agent_loop
        .run(&mut history, tx, cancel, None, None)
        .await
        .unwrap();

    // msg 0: user, msg 1: assistant(thinking + tool_use), msg 2: tool_result, msg 3: assistant(text)
    let first_assistant = &history.messages()[1];
    assert!(matches!(
        &first_assistant.blocks[0],
        ContentBlock::Thinking { text } if text == "reasoning"
    ));
    assert!(matches!(
        &first_assistant.blocks[1],
        ContentBlock::ToolUse { name, .. } if name == "echo"
    ));
}

#[tokio::test]
async fn loop_permission_denied_skips_tool() {
    let provider = MockProvider::new(vec![
        vec![
            StreamChunk::ToolUse {
                id: "call1".into(),
                name: "danger".into(),
                input: serde_json::json!({"cmd": "rm -rf"}),
            },
            StreamChunk::Done,
        ],
        vec![StreamChunk::Text("ok".into()), StreamChunk::Done],
    ]);

    struct DangerTool {
        executed: std::sync::Arc<AtomicBool>,
    }
    #[async_trait::async_trait]
    impl Tool for DangerTool {
        fn spec(&self) -> ToolSpec {
            // REGISTRY-WAIVE: exhaustive test struct ctor — pre-2026-05-13 baseline; new tests must use ..Default::default()
            ToolSpec {
                name: "danger".into(),
                description: "Dangerous".into(),
                parameters: serde_json::json!({"type":"object"}),
                permission: Permission::Dangerous,
            }
        }
        async fn execute(
            &self,
            _input: serde_json::Value,
            _cwd: &std::path::Path,
        ) -> crate::types::ToolResult {
            self.executed.store(true, Ordering::SeqCst);
            crate::types::ToolResult {
                output: "executed".into(),
                is_error: false,
            }
        }
    }

    let executed = std::sync::Arc::new(AtomicBool::new(false));
    let tools: Vec<Box<dyn Tool>> = vec![Box::new(DangerTool {
        executed: std::sync::Arc::clone(&executed),
    })];
    let agent_loop = make_loop(provider, tools);
    let mut history = ConversationHistory::new("sys".into());
    history.push_user("do it");

    let (tx, mut rx) = mpsc::channel(64);
    let cancel = CancellationToken::new();

    let (perm_tx, perm_rx) = mpsc::channel(4);

    let loop_handle = tokio::spawn(async move {
        let result = agent_loop
            .run(&mut history, tx, cancel, Some(perm_rx), None)
            .await;
        (result, history)
    });

    let mut saw_permission_request = false;
    let mut saw_success_tool_end = false;
    let mut saw_denial_tool_end = false;
    while let Some(ev) = rx.recv().await {
        if let AgentEvent::PermissionRequest { call_id, .. } = &ev {
            saw_permission_request = true;
            let _ = perm_tx
                .send(crate::types::PermissionResponse {
                    call_id: call_id.clone(),
                    allowed: false,
                })
                .await;
        }
        if let AgentEvent::ToolEnd { output, state, .. } = &ev {
            if *state == ToolState::Completed && output == "executed" {
                saw_success_tool_end = true;
            }
            if *state == ToolState::Error && output == "Permission denied by user" {
                saw_denial_tool_end = true;
            }
        }
        if matches!(ev, AgentEvent::Idle) {
            break;
        }
    }

    assert!(saw_permission_request);
    assert!(saw_denial_tool_end, "denial should be surfaced as ToolEnd");
    assert!(
        !saw_success_tool_end,
        "denied tool must not emit a successful ToolEnd"
    );
    assert!(
        !executed.load(Ordering::SeqCst),
        "denied tool executed despite explicit user refusal"
    );
    let (result, history) = loop_handle.await.unwrap();
    assert!(result.is_ok());
    assert!(
        history.messages().iter().any(|msg| {
            msg.blocks.iter().any(|block| {
                matches!(
                    block,
                    ContentBlock::ToolResult {
                        call_id,
                        output,
                        is_error: true,
                    } if call_id == "call1" && output == "Permission denied by user"
                )
            })
        }),
        "history must record the denial as an error tool_result"
    );
    assert!(
        !history.messages().iter().any(|msg| {
            msg.blocks.iter().any(|block| {
                matches!(
                    block,
                    ContentBlock::ToolResult {
                        output,
                        is_error: false,
                        ..
                    } if output == "executed"
                )
            })
        }),
        "history must not record successful output for a denied tool"
    );
}

#[tokio::test]
async fn loop_permission_allowed_executes_tool() {
    let provider = MockProvider::new(vec![
        vec![
            StreamChunk::ToolUse {
                id: "call1".into(),
                name: "echo".into(),
                input: serde_json::json!({"text": "safe"}),
            },
            StreamChunk::Done,
        ],
        vec![StreamChunk::Text("done".into()), StreamChunk::Done],
    ]);

    let tools: Vec<Box<dyn Tool>> = vec![Box::new(EchoTool)];
    let agent_loop = make_loop(provider, tools);
    let mut history = ConversationHistory::new("sys".into());
    history.push_user("test");

    let (tx, mut rx) = mpsc::channel(64);
    let cancel = CancellationToken::new();

    // EchoTool has Permission::ReadOnly, so no permission request should be emitted
    let (_, perm_rx) = mpsc::channel(4);

    let loop_handle = tokio::spawn(async move {
        agent_loop
            .run(&mut history, tx, cancel, Some(perm_rx), None)
            .await
    });

    let mut saw_perm_request = false;
    let mut saw_tool_end = false;
    while let Some(ev) = rx.recv().await {
        if matches!(ev, AgentEvent::PermissionRequest { .. }) {
            saw_perm_request = true;
        }
        if matches!(ev, AgentEvent::ToolEnd { .. }) {
            saw_tool_end = true;
        }
        if matches!(ev, AgentEvent::Idle) {
            break;
        }
    }

    assert!(
        !saw_perm_request,
        "ReadOnly tool should not ask for permission"
    );
    assert!(saw_tool_end, "tool should have been executed");
    let result = loop_handle.await.unwrap();
    assert!(result.is_ok());
}

// -- Step 2: ToolPolicy tests ------------------------------------------------

/// Policy that denies the echo tool.
struct DenyEchoPolicy;
impl crate::tool::policy::ToolPolicy for DenyEchoPolicy {
    fn classify(
        &self,
        name: &str,
        _input: &serde_json::Value,
        _cwd: &std::path::Path,
        permission: Permission,
    ) -> crate::tool::policy::ToolDecision {
        if name == "echo" {
            return crate::tool::policy::ToolDecision::Deny("echo denied by policy".into());
        }
        match permission {
            Permission::ReadOnly => crate::tool::policy::ToolDecision::Execute,
            p => crate::tool::policy::ToolDecision::AskUser(p),
        }
    }
}

#[tokio::test]
async fn loop_policy_deny_blocks_tool() {
    let provider = MockProvider::new(vec![
        vec![
            StreamChunk::ToolUse {
                id: "c1".into(),
                name: "echo".into(),
                input: serde_json::json!({"text": "hello"}),
            },
            StreamChunk::Done,
        ],
        vec![StreamChunk::Text("ok".into()), StreamChunk::Done],
    ]);

    let executions = std::sync::Arc::new(AtomicUsize::new(0));
    let agent_loop = AgentLoop::with_policy(
        Box::new(provider),
        crate::tool::registry::ToolRegistry::new(vec![Box::new(RecordingEchoTool {
            executions: std::sync::Arc::clone(&executions),
        })]),
        LoopConfig {
            max_iterations: 10,
            max_wall: None,
            cwd: std::path::PathBuf::from("/tmp"),
            model: "mock".into(),
            max_tokens: 1024,
            ..Default::default()
        },
        Box::new(DenyEchoPolicy),
    );

    let (tx, mut rx) = mpsc::channel(64);
    let cancel = CancellationToken::new();
    let mut history = ConversationHistory::new(String::new());
    history.push_user("test");

    let result = agent_loop.run(&mut history, tx, cancel, None, None).await;
    assert!(result.is_ok());

    let mut saw_deny = false;
    let mut saw_success_tool_end = false;
    while let Ok(ev) = rx.try_recv() {
        if let AgentEvent::ToolEnd {
            call_id,
            name,
            output,
            state,
        } = ev
        {
            if call_id == "c1"
                && name == "echo"
                && output.contains("denied by policy")
                && state == ToolState::Error
            {
                saw_deny = true;
            }
            if call_id == "c1"
                && name == "echo"
                && output == "recorded: hello"
                && state == ToolState::Completed
            {
                saw_success_tool_end = true;
            }
        }
    }
    assert!(saw_deny, "policy denial should emit ToolEnd with error");
    assert!(
        !saw_success_tool_end,
        "policy-denied tool must not emit a successful ToolEnd"
    );
    assert_eq!(
        executions.load(Ordering::SeqCst),
        0,
        "policy-denied tool executed despite policy denial"
    );
    assert!(
        history.messages().iter().any(|msg| {
            msg.blocks.iter().any(|block| {
                matches!(
                    block,
                    ContentBlock::ToolResult {
                        call_id,
                        output,
                        is_error: true,
                    } if call_id == "c1" && output.contains("denied by policy")
                )
            })
        }),
        "history must record the policy denial as an error tool_result"
    );
    assert!(
        !history.messages().iter().any(|msg| {
            msg.blocks.iter().any(|block| {
                matches!(
                    block,
                    ContentBlock::ToolResult {
                        call_id,
                        output,
                        is_error: false,
                    } if call_id == "c1" && output == "recorded: hello"
                )
            })
        }),
        "history must not record successful output for a policy-denied tool"
    );
}

#[tokio::test]
async fn loop_guard_denial_skips_blocked_tool() {
    let provider = MockProvider::new(vec![
        vec![
            StreamChunk::ToolUse {
                id: "c1".into(),
                name: "echo".into(),
                input: serde_json::json!({"text": "loop"}),
            },
            StreamChunk::ToolUse {
                id: "c2".into(),
                name: "echo".into(),
                input: serde_json::json!({"text": "loop"}),
            },
            StreamChunk::ToolUse {
                id: "c3".into(),
                name: "echo".into(),
                input: serde_json::json!({"text": "loop"}),
            },
            StreamChunk::Done,
        ],
        vec![StreamChunk::Text("ok".into()), StreamChunk::Done],
    ]);

    let executions = std::sync::Arc::new(AtomicUsize::new(0));
    let agent_loop = make_loop(
        provider,
        vec![Box::new(RecordingEchoTool {
            executions: std::sync::Arc::clone(&executions),
        })],
    );
    let mut history = ConversationHistory::new("sys".into());
    history.push_user("test");

    let (tx, mut rx) = mpsc::channel(64);
    let cancel = CancellationToken::new();

    let result = agent_loop.run(&mut history, tx, cancel, None, None).await;
    assert!(result.is_ok());

    let mut saw_guard_denial = false;
    let mut saw_blocked_success_tool_end = false;
    while let Ok(ev) = rx.try_recv() {
        if let AgentEvent::ToolEnd {
            call_id,
            name,
            output,
            state,
        } = ev
        {
            if call_id == "c3"
                && name == "echo"
                && output.contains("already ran 3 times")
                && state == ToolState::Error
            {
                saw_guard_denial = true;
            }
            if call_id == "c3"
                && name == "echo"
                && output == "recorded: loop"
                && state == ToolState::Completed
            {
                saw_blocked_success_tool_end = true;
            }
        }
    }
    assert!(
        saw_guard_denial,
        "loop guard denial should emit ToolEnd with error"
    );
    assert!(
        !saw_blocked_success_tool_end,
        "loop-guard-blocked tool must not emit a successful ToolEnd"
    );
    assert_eq!(
        executions.load(Ordering::SeqCst),
        2,
        "loop guard should skip the third identical tool call"
    );
    assert!(
        history.messages().iter().any(|msg| {
            msg.blocks.iter().any(|block| {
                matches!(
                    block,
                    ContentBlock::ToolResult {
                        call_id,
                        output,
                        is_error: true,
                    } if call_id == "c3" && output.contains("already ran 3 times")
                )
            })
        }),
        "history must record the loop guard denial as an error tool_result"
    );
    assert!(
        !history.messages().iter().any(|msg| {
            msg.blocks.iter().any(|block| {
                matches!(
                    block,
                    ContentBlock::ToolResult {
                        call_id,
                        output,
                        is_error: false,
                    } if call_id == "c3" && output == "recorded: loop"
                )
            })
        }),
        "history must not record successful output for a loop-guard-blocked tool"
    );
}

// ── Steer tests ──────────────────────────────────────────────────

#[tokio::test]
async fn steer_message_injected_between_iterations() {
    // Provider: iteration 1 calls a tool, iteration 2 returns text.
    // We send a steer message between them and verify it appears in history.
    let provider = MockProvider::new(vec![
        // Iteration 1: tool call
        vec![
            StreamChunk::ToolUse {
                id: "t1".into(),
                name: "echo".into(),
                input: serde_json::json!({"text": "hi"}),
            },
            StreamChunk::Done,
        ],
        // Iteration 2: text response (after steer)
        vec![
            StreamChunk::Text("got your steer".into()),
            StreamChunk::Done,
        ],
    ]);

    let agent_loop = make_loop(provider, vec![Box::new(EchoTool)]);
    let mut history = ConversationHistory::new("sys".into());
    history.push_user("do something");

    let (tx, mut rx) = mpsc::channel(64);
    let cancel = CancellationToken::new();
    let (steer_tx, steer_rx) = mpsc::channel(16);

    // Pre-load steer message (will be drained at iteration boundary).
    steer_tx
        .send(SteerMessage {
            msg_id: 42,
            text: "change direction".into(),
            is_edit: false,
        })
        .await
        .unwrap();

    let result = agent_loop
        .run(&mut history, tx, cancel, None, Some(steer_rx))
        .await;
    assert!(result.is_ok());

    // Verify steer text is in the history as a user message.
    let msgs = history.messages();
    let steer_in_history = msgs
        .iter()
        .any(|m| m.role == Role::User && m.text_content().contains("change direction"));
    assert!(steer_in_history, "steer message must appear in history");

    // Verify SteerReceived event was emitted.
    let mut saw_steer = false;
    while let Ok(ev) = rx.try_recv() {
        if matches!(&ev, AgentEvent::SteerReceived { text, .. } if text.contains("change direction"))
        {
            saw_steer = true;
        }
    }
    assert!(saw_steer, "SteerReceived event must be emitted");
}

#[tokio::test]
async fn steer_multiple_merged_into_one() {
    // Three steer messages should be merged into a single user message.
    let provider = MockProvider::new(vec![
        vec![
            StreamChunk::ToolUse {
                id: "t1".into(),
                name: "echo".into(),
                input: serde_json::json!({"text":"x"}),
            },
            StreamChunk::Done,
        ],
        vec![StreamChunk::Text("ok".into()), StreamChunk::Done],
    ]);

    let agent_loop = make_loop(provider, vec![Box::new(EchoTool)]);
    let mut history = ConversationHistory::new("sys".into());
    history.push_user("go");

    let (tx, _rx) = mpsc::channel(64);
    let cancel = CancellationToken::new();
    let (steer_tx, steer_rx) = mpsc::channel(16);

    for (id, text) in [(1, "msg one"), (2, "msg two"), (3, "msg three")] {
        steer_tx
            .send(SteerMessage {
                msg_id: id,
                text: text.into(),
                is_edit: false,
            })
            .await
            .unwrap();
    }

    let result = agent_loop
        .run(&mut history, tx, cancel, None, Some(steer_rx))
        .await;
    assert!(result.is_ok());

    // Count user messages that contain steer text.
    let steer_msgs: Vec<_> = history
        .messages()
        .iter()
        .filter(|m| m.role == Role::User && m.text_content().contains("msg one"))
        .collect();
    assert_eq!(
        steer_msgs.len(),
        1,
        "3 steer messages must be merged into 1 user message"
    );
    let combined = steer_msgs[0].text_content();
    assert!(combined.contains("msg one"));
    assert!(combined.contains("msg two"));
    assert!(combined.contains("msg three"));
}

#[tokio::test]
async fn steer_edit_replaces_in_pending_queue() {
    // Send msg_id=10, then edit msg_id=10 before drain.
    let provider = MockProvider::new(vec![
        vec![
            StreamChunk::ToolUse {
                id: "t1".into(),
                name: "echo".into(),
                input: serde_json::json!({"text":"x"}),
            },
            StreamChunk::Done,
        ],
        vec![StreamChunk::Text("done".into()), StreamChunk::Done],
    ]);

    let agent_loop = make_loop(provider, vec![Box::new(EchoTool)]);
    let mut history = ConversationHistory::new("sys".into());
    history.push_user("start");

    let (tx, _rx) = mpsc::channel(64);
    let cancel = CancellationToken::new();
    let (steer_tx, steer_rx) = mpsc::channel(16);

    // Original
    steer_tx
        .send(SteerMessage {
            msg_id: 10,
            text: "find cafes".into(),
            is_edit: false,
        })
        .await
        .unwrap();
    // Edit
    steer_tx
        .send(SteerMessage {
            msg_id: 10,
            text: "find bars".into(),
            is_edit: true,
        })
        .await
        .unwrap();

    let result = agent_loop
        .run(&mut history, tx, cancel, None, Some(steer_rx))
        .await;
    assert!(result.is_ok());

    let all_text: String = history
        .messages()
        .iter()
        .filter(|m| m.role == Role::User)
        .map(|m| m.text_content())
        .collect::<Vec<_>>()
        .join(" ");
    assert!(
        !all_text.contains("find cafes"),
        "original must be replaced by edit"
    );
    assert!(all_text.contains("find bars"), "edited text must appear");
}

#[tokio::test]
async fn steer_edit_after_drain_adds_correction() {
    // R1: this used to poke AgentLoop::drain_steers directly. The
    // unified `SteerPipeline` makes that signature private, so the
    // tightest equivalent is a pipeline-level test that mirrors the
    // exact scenario (drain → edit-of-delivered → second drain).
    use crate::loop_::steers::SteerPipeline;
    let (steer_tx, steer_rx) = mpsc::channel(16);
    let mut opt_rx = Some(steer_rx);
    let mut pipeline = SteerPipeline::new();
    let mut history = ConversationHistory::new("sys".into());
    history.push_user("go");
    let (tx, _rx) = mpsc::channel(64);

    steer_tx
        .send(SteerMessage {
            msg_id: 20,
            text: "original direction".into(),
            is_edit: false,
        })
        .await
        .unwrap();

    // Drain 1: delivers msg_id=20.
    let r1 = pipeline.drain(&mut opt_rx, &mut history, &tx).await;
    assert!(r1, "first drain must report rescue");
    let user_msgs: Vec<_> = history
        .messages()
        .iter()
        .filter(|m| m.role == Role::User)
        .map(|m| m.text_content())
        .collect();
    assert!(
        user_msgs.iter().any(|t| t.contains("original direction")),
        "original must be delivered"
    );

    // Edit of an already-delivered msg_id=20.
    steer_tx
        .send(SteerMessage {
            msg_id: 20,
            text: "corrected direction".into(),
            is_edit: true,
        })
        .await
        .unwrap();
    let r2 = pipeline.drain(&mut opt_rx, &mut history, &tx).await;
    assert!(r2, "second drain must also report rescue");

    let all_text: String = history
        .messages()
        .iter()
        .filter(|m| m.role == Role::User)
        .map(|m| m.text_content())
        .collect::<Vec<_>>()
        .join(" | ");
    assert!(
        all_text.contains("[correction] corrected direction"),
        "edit after drain must appear as correction; got: {all_text}"
    );
}

#[tokio::test]
async fn steer_no_channel_works() {
    // Passing None for steer_rx should work (backward compat).
    let provider = MockProvider::new(vec![vec![
        StreamChunk::Text("hello".into()),
        StreamChunk::Done,
    ]]);
    let agent_loop = make_loop(provider, vec![]);
    let mut history = ConversationHistory::new("sys".into());
    history.push_user("hi");
    let (tx, _rx) = mpsc::channel(64);
    let cancel = CancellationToken::new();
    let result = agent_loop.run(&mut history, tx, cancel, None, None).await;
    assert!(result.is_ok());
}

// ── Slow tool for mid-tool steer tests ───────────────────────

struct SlowTool {
    permission: Permission,
}

impl SlowTool {
    fn readonly() -> Self {
        Self {
            permission: Permission::ReadOnly,
        }
    }
}

#[async_trait::async_trait]
impl Tool for SlowTool {
    fn spec(&self) -> ToolSpec {
        // REGISTRY-WAIVE: exhaustive test struct ctor — pre-2026-05-13 baseline; new tests must use ..Default::default()
        ToolSpec {
            name: "slow".into(),
            description: "Sleeps 200ms then returns".into(),
            parameters: serde_json::json!({"type":"object","properties":{}}),
            permission: self.permission,
        }
    }

    async fn execute(
        &self,
        _input: serde_json::Value,
        _cwd: &std::path::Path,
    ) -> crate::types::ToolResult {
        tokio::time::sleep(std::time::Duration::from_millis(200)).await;
        crate::types::ToolResult {
            output: "done sleeping".into(),
            is_error: false,
        }
    }
}

#[tokio::test]
async fn steer_during_tool_execution_is_buffered() {
    // Verify that a steer message sent WHILE a tool is executing
    // gets buffered and then injected into history after the tool returns.
    let provider = MockProvider::new(vec![
        // Iteration 1: call slow tool
        vec![
            StreamChunk::ToolUse {
                id: "t1".into(),
                name: "slow".into(),
                input: serde_json::json!({}),
            },
            StreamChunk::Done,
        ],
        // Iteration 2: respond after seeing the steer
        vec![
            StreamChunk::Text("saw your redirect".into()),
            StreamChunk::Done,
        ],
    ]);

    let agent_loop = make_loop(provider, vec![Box::new(SlowTool::readonly())]);
    let mut history = ConversationHistory::new("sys".into());
    history.push_user("start task");

    let (tx, mut rx) = mpsc::channel(64);
    let cancel = CancellationToken::new();
    let (steer_tx, steer_rx) = mpsc::channel(16);

    // Spawn the agent loop:
    let handle = tokio::spawn(async move {
        agent_loop
            .run(&mut history, tx, cancel, None, Some(steer_rx))
            .await
            .map(|_| history)
    });

    // Wait for tool to start executing (50ms), then send steer:
    tokio::time::sleep(std::time::Duration::from_millis(50)).await;
    steer_tx
        .send(SteerMessage {
            msg_id: 99,
            text: "redirect to new task".into(),
            is_edit: false,
        })
        .await
        .unwrap();

    // Wait for completion:
    let history = handle.await.unwrap().unwrap();

    // Verify steer was injected into history:
    let steer_in_history = history
        .messages()
        .iter()
        .any(|m| m.role == Role::User && m.text_content().contains("redirect to new task"));
    assert!(
        steer_in_history,
        "steer sent during tool execution must appear in history"
    );

    // Verify SteerReceived event was emitted:
    let mut saw_steer = false;
    while let Ok(ev) = rx.try_recv() {
        if matches!(&ev, AgentEvent::SteerReceived { text, .. } if text.contains("redirect to new task"))
        {
            saw_steer = true;
        }
    }
    assert!(saw_steer, "SteerReceived event must be emitted");
}

#[tokio::test]
async fn steer_during_readonly_parallel_tools_is_buffered() {
    // Multiple readonly tools execute in parallel. Steer sent during
    // their execution must be buffered and appear in history.
    let provider = MockProvider::new(vec![
        // Iteration 1: two parallel readonly tools
        vec![
            StreamChunk::ToolUse {
                id: "t1".into(),
                name: "slow".into(),
                input: serde_json::json!({}),
            },
            StreamChunk::ToolUse {
                id: "t2".into(),
                name: "slow".into(),
                input: serde_json::json!({}),
            },
            StreamChunk::Done,
        ],
        // Iteration 2: response
        vec![StreamChunk::Text("acknowledged".into()), StreamChunk::Done],
    ]);

    let agent_loop = make_loop(provider, vec![Box::new(SlowTool::readonly())]);
    let mut history = ConversationHistory::new("sys".into());
    history.push_user("run parallel");

    let (tx, _rx) = mpsc::channel(64);
    let cancel = CancellationToken::new();
    let (steer_tx, steer_rx) = mpsc::channel(16);

    let handle = tokio::spawn(async move {
        agent_loop
            .run(&mut history, tx, cancel, None, Some(steer_rx))
            .await
            .map(|_| history)
    });

    tokio::time::sleep(std::time::Duration::from_millis(50)).await;
    steer_tx
        .send(SteerMessage {
            msg_id: 100,
            text: "parallel steer".into(),
            is_edit: false,
        })
        .await
        .unwrap();

    let history = handle.await.unwrap().unwrap();
    let found = history
        .messages()
        .iter()
        .any(|m| m.role == Role::User && m.text_content().contains("parallel steer"));
    assert!(found, "steer during parallel tools must appear in history");
}

/// MockProvider variant whose FIRST response yields chunks slowly so
/// the test can inject a steer between iteration-top drain and
/// pre-Idle drain. Subsequent responses are instant.
struct SlowFirstMockProvider {
    responses: Vec<Vec<StreamChunk>>,
    call_count: AtomicUsize,
    first_chunk_delay: std::time::Duration,
}

impl SlowFirstMockProvider {
    fn new(responses: Vec<Vec<StreamChunk>>, delay: std::time::Duration) -> Self {
        Self {
            responses,
            call_count: AtomicUsize::new(0),
            first_chunk_delay: delay,
        }
    }
}

#[async_trait::async_trait]
impl Provider for SlowFirstMockProvider {
    fn name(&self) -> &str {
        "slow-first-mock"
    }
    fn models(&self) -> Vec<crate::types::ModelInfo> {
        vec![]
    }
    async fn stream_chat(
        &self,
        _request: ChatRequest,
    ) -> crate::error::Result<Pin<Box<dyn tokio_stream::Stream<Item = StreamChunk> + Send>>> {
        use tokio_stream::StreamExt;
        let idx = self.call_count.fetch_add(1, Ordering::SeqCst);
        let chunks = if idx < self.responses.len() {
            self.responses[idx].clone()
        } else {
            vec![StreamChunk::Text("fallback".into()), StreamChunk::Done]
        };
        if idx == 0 {
            // Delay each chunk so a steer can land between
            // iteration-top drain and pre-Idle drain.
            let delay = self.first_chunk_delay;
            let stream = tokio_stream::iter(chunks).then(move |c| async move {
                tokio::time::sleep(delay).await;
                c
            });
            Ok(Box::pin(stream))
        } else {
            Ok(Box::pin(tokio_stream::iter(chunks)))
        }
    }
}

#[tokio::test]
async fn steer_after_idle_does_not_get_lost() {
    // S1 of PLAN_NEXT_SESSION: pinned regression for the user-visible
    // "Принято — доставлю между шагами" hang.
    //
    // Scenario: model produces a text-only response (no tool calls)
    // → pre-S1 the loop returned Idle BEFORE any drain_steers call,
    // so a steer arriving during the stream was silently dropped
    // (steer_rx is destroyed when run() returns).
    //
    // We use SlowFirstMockProvider so iteration 1's stream yields a
    // chunk every 80ms; we send the steer at t+50ms, after
    // iteration-top drain has already run (empty) but well before
    // the pre-Idle drain. With S1 the pre-Idle drain catches it and
    // continues to a second iteration.
    let provider = SlowFirstMockProvider::new(
        vec![
            // Iteration 1: text-only response, slow chunks
            vec![StreamChunk::Text("ok, doing X".into()), StreamChunk::Done],
            // Iteration 2: instant response to the steer (only reached
            // when S1 works)
            vec![
                StreamChunk::Text("acknowledged steer".into()),
                StreamChunk::Done,
            ],
        ],
        std::time::Duration::from_millis(80),
    );

    let agent_loop = AgentLoop::new(
        Box::new(provider),
        crate::tool::registry::ToolRegistry::new(vec![]),
        LoopConfig::default(),
    );
    let mut history = ConversationHistory::new("sys".into());
    history.push_user("do something");

    let (tx, _rx) = mpsc::channel(64);
    let cancel = CancellationToken::new();
    let (steer_tx, steer_rx) = mpsc::channel(16);

    let handle = tokio::spawn(async move {
        agent_loop
            .run(&mut history, tx, cancel, None, Some(steer_rx))
            .await
            .map(|_| history)
    });

    // Wait long enough that iteration-top drain has run with an empty
    // channel, but iteration 1 hasn't finished streaming.
    tokio::time::sleep(std::time::Duration::from_millis(50)).await;
    steer_tx
        .send(SteerMessage {
            msg_id: 7,
            text: "нет не надо".into(),
            is_edit: false,
        })
        .await
        .unwrap();

    let history = handle.await.unwrap().unwrap();
    // Steer must appear in history (drained on Idle exit).
    let steer_in_history = history
        .messages()
        .iter()
        .any(|m| m.role == Role::User && m.text_content().contains("нет не надо"));
    assert!(
        steer_in_history,
        "S1: steer arriving on text-only-response turn must reach history (was lost pre-S1)"
    );
    // The model's answer to the steer must appear too — proves the
    // loop did NOT return Idle before re-issuing.
    let steer_answered = history
        .messages()
        .iter()
        .any(|m| m.role == Role::Assistant && m.text_content().contains("acknowledged steer"));
    assert!(
        steer_answered,
        "S1: model must respond to the drained steer, not exit at Idle"
    );
}

// ── B80b: closed steer channel must not CPU-spin in execute_readonly_batch ──

/// A read-only tool whose execute-future parks on a real `tokio::time::sleep`
/// (an EXTERNAL timer waker — it does NOT self-wake). Every poll of the
/// wrapper is counted. A correctly-parking `select!` loop polls this future
/// ~twice (once initially, once when the timer wakes it). A busy-spinning
/// loop re-polls the in-flight `join` — and therefore this future — over and
/// over during the sleep window, inflating the count by orders of magnitude.
struct PollCountingTool {
    polls: std::sync::Arc<AtomicUsize>,
    delay: std::time::Duration,
}

struct SleepPollCounter {
    polls: std::sync::Arc<AtomicUsize>,
    sleep: Pin<Box<tokio::time::Sleep>>,
}

impl std::future::Future for SleepPollCounter {
    type Output = ();
    fn poll(mut self: Pin<&mut Self>, cx: &mut std::task::Context<'_>) -> std::task::Poll<()> {
        self.polls.fetch_add(1, Ordering::SeqCst);
        self.sleep.as_mut().poll(cx)
    }
}

#[async_trait::async_trait]
impl Tool for PollCountingTool {
    fn spec(&self) -> ToolSpec {
        // REGISTRY-WAIVE: B16 — exhaustive test ToolSpec ctor, mirrors EchoTool above
        ToolSpec {
            name: "echo".into(),
            description: "Poll-counting echo".into(),
            parameters: serde_json::json!({"type":"object","properties":{"text":{"type":"string"}}}),
            permission: Permission::ReadOnly,
        }
    }

    async fn execute(
        &self,
        _input: serde_json::Value,
        _cwd: &std::path::Path,
    ) -> crate::types::ToolResult {
        SleepPollCounter {
            polls: self.polls.clone(),
            sleep: Box::pin(tokio::time::sleep(self.delay)),
        }
        .await;
        crate::types::ToolResult {
            output: "echoed".into(),
            is_error: false,
        }
    }
}

struct WritePollCountingTool {
    polls: std::sync::Arc<AtomicUsize>,
    delay: std::time::Duration,
}

struct HangingWriteTool {
    delay: std::time::Duration,
}

#[async_trait::async_trait]
impl Tool for HangingWriteTool {
    fn spec(&self) -> ToolSpec {
        // REGISTRY-WAIVE: B16 — exhaustive test ToolSpec ctor, mirrors write_poll test tool above
        ToolSpec {
            name: "hang_write".into(),
            description: "Write-permission tool that sleeps past the in-turn deadline".into(),
            parameters: serde_json::json!({"type":"object","properties":{}}),
            permission: Permission::WorkspaceWrite,
        }
    }

    async fn execute(
        &self,
        _input: serde_json::Value,
        _cwd: &std::path::Path,
    ) -> crate::types::ToolResult {
        tokio::time::sleep(self.delay).await;
        crate::types::ToolResult {
            output: "unexpectedly completed".into(),
            is_error: false,
        }
    }
}

#[async_trait::async_trait]
impl Tool for WritePollCountingTool {
    fn spec(&self) -> ToolSpec {
        // REGISTRY-WAIVE: B16 — exhaustive test ToolSpec ctor, mirrors PollCountingTool above
        ToolSpec {
            name: "write_poll".into(),
            description: "Write-permission poll-counting tool".into(),
            parameters: serde_json::json!({"type":"object","properties":{"text":{"type":"string"}}}),
            permission: Permission::WorkspaceWrite,
        }
    }

    async fn execute(
        &self,
        _input: serde_json::Value,
        _cwd: &std::path::Path,
    ) -> crate::types::ToolResult {
        SleepPollCounter {
            polls: self.polls.clone(),
            sleep: Box::pin(tokio::time::sleep(self.delay)),
        }
        .await;
        crate::types::ToolResult {
            output: "write-poll done".into(),
            is_error: false,
        }
    }
}

/// B80b regression: when `steer_rx` is `Some` but its sender is CLOSED,
/// the `execute_readonly_batch` select! steer arm used to resolve `None`
/// instantly on every poll. Because the arm body did nothing on `None`,
/// the `loop { select! { ... } }` re-entered immediately and busy-spun:
/// every iteration re-polled the in-flight `join` (and therefore the
/// running tool future) while burning CPU, instead of parking until the
/// tool actually made progress.
///
/// Detection is deterministic via a poll counter on a timer-backed tool
/// future. A correct (parking) loop polls the tool future only a handful
/// of times (initial poll + the timer wake). A spinning loop re-polls the
/// in-flight join every iteration for the whole sleep window, inflating
/// the count into the thousands. We assert a tight upper budget.
#[tokio::test]
async fn b80b_closed_steer_channel_does_not_spin_in_readonly_batch() {
    let polls = std::sync::Arc::new(AtomicUsize::new(0));

    // iter 0: call the read-only `echo` tool; iter 1: finish.
    let provider = MockProvider::new(vec![
        vec![
            StreamChunk::ToolUse {
                id: "c1".into(),
                name: "echo".into(),
                input: serde_json::json!({"text": "hi"}),
            },
            StreamChunk::Done,
        ],
        vec![StreamChunk::Text("done".into()), StreamChunk::Done],
    ]);
    let agent_loop = make_loop(
        provider,
        vec![Box::new(PollCountingTool {
            polls: polls.clone(),
            delay: std::time::Duration::from_millis(200),
        })],
    );
    let mut history = ConversationHistory::new("sys".into());
    history.push_user("run a tool");

    // Build a steer channel and immediately drop the SENDER so the
    // receiver observes a closed channel (recv() -> None forever).
    let (steer_tx, steer_rx) = mpsc::channel::<SteerMessage>(16);
    drop(steer_tx);

    let (tx, _rx) = mpsc::channel(256);
    let cancel = CancellationToken::new();

    let result = tokio::time::timeout(
        std::time::Duration::from_secs(10),
        agent_loop.run(&mut history, tx, cancel, None, Some(steer_rx)),
    )
    .await;

    assert!(result.is_ok(), "B80b: turn must complete (no hang)");
    assert!(
        result.unwrap().is_ok(),
        "B80b: turn should complete successfully after the tool"
    );

    let observed = polls.load(Ordering::SeqCst);
    // Parking loop polls the timer-backed tool ~2-3 times over the 200ms
    // window. A spin re-polls it thousands of times. 100 is comfortably
    // above the parking count yet far below any spin.
    assert!(
        observed <= 100,
        "B80b: tool future polled {observed} times (budget 100); the closed \
         steer channel is busy-spinning the readonly-batch select! loop \
         instead of parking (regression)"
    );
}

/// B82 regression: same closed-steer spin class as B80b, but in the
/// gated single-tool heartbeat path (`execute_tool_with_heartbeat`). This
/// must NOT be a naive clone of the B80b test: the tool is
/// `WorkspaceWrite`, so `classify_tool_calls` puts it in `gated_calls`
/// rather than the read-only batch, and `permission_rx == None` exercises
/// the loop's non-interactive auto-approve path before execution.
#[tokio::test]
async fn b82_closed_steer_channel_does_not_spin_in_single_tool_heartbeat() {
    let polls = std::sync::Arc::new(AtomicUsize::new(0));

    // iter 0: call the non-readonly tool; iter 1: finish.
    let provider = MockProvider::new(vec![
        vec![
            StreamChunk::ToolUse {
                id: "c1".into(),
                name: "write_poll".into(),
                input: serde_json::json!({"text": "hi"}),
            },
            StreamChunk::Done,
        ],
        vec![StreamChunk::Text("done".into()), StreamChunk::Done],
    ]);
    let agent_loop = make_loop(
        provider,
        vec![Box::new(WritePollCountingTool {
            polls: polls.clone(),
            delay: std::time::Duration::from_millis(200),
        })],
    );
    let mut history = ConversationHistory::new("sys".into());
    history.push_user("run a write tool");

    // Build a steer channel and immediately drop the SENDER so the
    // receiver observes a closed channel (recv() -> None forever).
    let (steer_tx, steer_rx) = mpsc::channel::<SteerMessage>(16);
    drop(steer_tx);

    let (tx, mut rx) = mpsc::channel(256);
    let cancel = CancellationToken::new();

    let result = tokio::time::timeout(
        std::time::Duration::from_secs(10),
        agent_loop.run(&mut history, tx, cancel, None, Some(steer_rx)),
    )
    .await;

    assert!(result.is_ok(), "B82: turn must complete (no hang)");
    assert!(
        result.unwrap().is_ok(),
        "B82: turn should complete successfully after the gated tool"
    );

    let mut saw_write_tool_end = false;
    let mut saw_spurious_close_banner = false;
    while let Ok(ev) = rx.try_recv() {
        match ev {
            AgentEvent::ToolEnd { name, .. } if name == "write_poll" => {
                saw_write_tool_end = true;
            }
            AgentEvent::ToolOutput { chunk, .. } if chunk.contains("Steer queued") => {
                saw_spurious_close_banner = true;
            }
            _ => {}
        }
    }
    assert!(
        saw_write_tool_end,
        "B82: write_poll ToolEnd proves the non-readonly gated single-tool route executed"
    );
    assert!(
        !saw_spurious_close_banner,
        "B82: closed steer channel must not emit a spurious queued-steer banner"
    );

    let observed = polls.load(Ordering::SeqCst);
    assert!(
        observed <= 100,
        "B82: tool future polled {observed} times (budget 100); the closed \
         steer channel is busy-spinning execute_tool_with_heartbeat instead \
         of parking (regression)"
    );
}

/// B83 regression: a single hung gated tool used to outlive `max_wall`
/// because B73 checks only at the outer iteration boundary. The deadline
/// must cancel through the existing CancellationToken path (not drop the
/// tool future) so heartbeat/steer cleanup runs and the turn returns a
/// distinct wall-timeout error before the outer rerevert guard trips.
#[tokio::test]
async fn b83_in_turn_deadline_aborts_hung_tool() {
    let provider = MockProvider::new(vec![vec![
        StreamChunk::ToolUse {
            id: "c1".into(),
            name: "hang_write".into(),
            input: serde_json::json!({}),
        },
        StreamChunk::Done,
    ]]);
    let agent_loop = AgentLoop::new(
        Box::new(provider),
        crate::tool::registry::ToolRegistry::new(vec![Box::new(HangingWriteTool {
            delay: std::time::Duration::from_secs(60),
        })]),
        LoopConfig {
            max_iterations: 10,
            max_wall: Some(std::time::Duration::from_secs(60)),
            tool_deadline: Some(std::time::Duration::from_millis(200)),
            cwd: std::path::PathBuf::from("/tmp"),
            model: "mock".into(),
            max_tokens: 1024,
            ..Default::default()
        },
    );
    let mut history = ConversationHistory::new("sys".into());
    history.push_user("run a hung write tool");

    let (tx, mut rx) = mpsc::channel(256);
    let cancel = CancellationToken::new();
    let (_steer_tx, steer_rx) = mpsc::channel::<SteerMessage>(16);
    let started = std::time::Instant::now();

    let outer = tokio::time::timeout(
        std::time::Duration::from_secs(5),
        agent_loop.run(&mut history, tx, cancel, None, Some(steer_rx)),
    )
    .await;

    let result = outer.expect("B83: outer timeout tripped; in-turn tool deadline did not abort");
    assert!(
        matches!(result, Err(AgentError::WallTimeout)),
        "B83: deadline must return WallTimeout, not Cancelled/success; got {result:?}"
    );
    assert!(
        started.elapsed() < std::time::Duration::from_secs(2),
        "B83: tool deadline should fire well before the 60s tool sleep; elapsed {:?}",
        started.elapsed()
    );

    let mut events = Vec::new();
    while let Ok(ev) = rx.try_recv() {
        events.push(ev);
    }
    assert!(
        events.iter().any(
            |e| matches!(e, AgentEvent::Error(s) if s.contains(crate::error::WALL_TIMEOUT_MESSAGE))
        ),
        "B83: timeout cleanup must emit Error; got events: {events:?}"
    );
    assert!(
        events.iter().any(|e| matches!(e, AgentEvent::Idle)),
        "B83: timeout cleanup must emit Idle; got events: {events:?}"
    );
}

/// S2b/B83 regression: provider stream-open can hang inside one iteration.
/// The coarse turn backstop must cancel through the existing token and return
/// WallTimeout before the outer rerevert guard trips.
#[tokio::test]
async fn b83_turn_backstop_aborts_hung_provider_open() {
    let agent_loop = make_provider_backstop_loop(
        Box::new(HungStreamOpenProvider),
        std::time::Duration::from_millis(200),
    );
    let mut history = ConversationHistory::new("sys".into());
    history.push_user("open a provider stream that never returns");

    let (tx, mut rx) = mpsc::channel(256);
    let cancel = CancellationToken::new();
    let started = std::time::Instant::now();

    let outer = tokio::time::timeout(
        std::time::Duration::from_secs(5),
        agent_loop.run(&mut history, tx, cancel, None, None),
    )
    .await;

    let result = outer.expect("B83/S2b: outer timeout tripped; turn backstop did not abort");
    assert!(
        matches!(result, Err(AgentError::WallTimeout)),
        "B83/S2b: backstop must return WallTimeout; got {result:?}"
    );
    assert!(
        started.elapsed() < std::time::Duration::from_secs(2),
        "B83/S2b: backstop should fire well before the 60s provider-open sleep; elapsed {:?}",
        started.elapsed()
    );

    let mut events = Vec::new();
    while let Ok(ev) = rx.try_recv() {
        events.push(ev);
    }
    assert!(
        events.iter().any(
            |e| matches!(e, AgentEvent::Error(s) if s.contains(crate::error::WALL_TIMEOUT_MESSAGE))
        ),
        "B83/S2b: timeout cleanup must emit Error; got events: {events:?}"
    );
    assert!(
        events.iter().any(|e| matches!(e, AgentEvent::Idle)),
        "B83/S2b: timeout cleanup must emit Idle; got events: {events:?}"
    );
}

/// S2b/B83 false-positive guard: a slow but legitimate fallback chain must
/// complete when the ceiling is generously above its latency.
#[tokio::test]
async fn b83_legit_slow_fallback_turn_survives_backstop() {
    let provider = crate::provider::resilient::ResilientProvider::new(vec![
        Box::new(DelayedFailProvider {
            delay: std::time::Duration::from_millis(250),
        }),
        Box::new(DelayedTextProvider {
            delay: std::time::Duration::from_millis(500),
            text: "fallback ok".into(),
        }),
    ]);
    let agent_loop =
        make_provider_backstop_loop(Box::new(provider), std::time::Duration::from_secs(5));
    let mut history = ConversationHistory::new("sys".into());
    history.push_user("survive a slow fallback turn");

    let (tx, mut rx) = mpsc::channel(256);
    let cancel = CancellationToken::new();
    let result = tokio::time::timeout(
        std::time::Duration::from_secs(5),
        agent_loop.run(&mut history, tx, cancel, None, None),
    )
    .await
    .expect("B83/S2b false-positive guard should complete before outer timeout");

    assert!(
        result.is_ok(),
        "B83/S2b: generous backstop must not kill a legitimate slow fallback; got {result:?}"
    );

    let mut saw_text = false;
    let mut saw_idle = false;
    let mut saw_timeout_error = false;
    while let Ok(ev) = rx.try_recv() {
        match ev {
            AgentEvent::TextDelta(t) if t.contains("fallback ok") => saw_text = true,
            AgentEvent::Idle => saw_idle = true,
            AgentEvent::Error(e) if e.contains(crate::error::WALL_TIMEOUT_MESSAGE) => {
                saw_timeout_error = true;
            }
            _ => {}
        }
    }
    assert!(saw_text, "B83/S2b: fallback success text was not streamed");
    assert!(saw_idle, "B83/S2b: successful fallback turn must end Idle");
    assert!(
        !saw_timeout_error,
        "B83/S2b: false-positive guard saw timeout error despite successful fallback"
    );
}

// ── B80a: an inner AgentLoop whose event channel is not drained deadlocks ──

/// A read-only tool that emits more `ToolOutput` progress events than the
/// event channel can buffer. Mirrors a chatty inner research turn.
struct ChattyTool {
    bursts: usize,
}

#[async_trait::async_trait]
impl Tool for ChattyTool {
    fn spec(&self) -> ToolSpec {
        // REGISTRY-WAIVE: B16 — exhaustive test ToolSpec ctor, mirrors EchoTool above
        ToolSpec {
            name: "echo".into(),
            description: "Chatty echo".into(),
            parameters: serde_json::json!({"type":"object","properties":{"text":{"type":"string"}}}),
            permission: Permission::ReadOnly,
        }
    }

    async fn execute(
        &self,
        _input: serde_json::Value,
        _cwd: &std::path::Path,
    ) -> crate::types::ToolResult {
        crate::types::ToolResult {
            output: "echoed".into(),
            is_error: false,
        }
    }

    async fn execute_with_progress(
        &self,
        _input: serde_json::Value,
        _cwd: &std::path::Path,
        progress: mpsc::Sender<AgentEvent>,
    ) -> crate::types::ToolResult {
        // Each .send().await parks if the channel is full; with no drainer
        // this blocks forever (the B80a deadlock). With a drainer it
        // completes. This is exactly the inner research loop's behaviour.
        for i in 0..self.bursts {
            let _ = progress
                .send(AgentEvent::ToolOutput {
                    call_id: "c1".into(),
                    chunk: format!("chunk {i}"),
                })
                .await;
        }
        crate::types::ToolResult {
            output: "echoed".into(),
            is_error: false,
        }
    }
}

/// B80a regression: the inner research `AgentLoop` emits events into a
/// BOUNDED channel (`mpsc::channel(256)`). If the receiver is never
/// drained, the loop's own `tx.send().await` parks forever once the
/// buffer fills — the tool never returns and the parent turn hangs (the
/// live 30-minute CPU-burning hang). `research_run` fixes this by spawning
/// a forwarder that always drains the inner receiver. This test reproduces
/// the mechanism at the AgentLoop level: with a chatty tool emitting far
/// more than the buffer, the turn completes ONLY when a concurrent drainer
/// reads the events.
#[tokio::test]
async fn b80a_inner_loop_completes_only_when_events_are_drained() {
    // Tool emits 300 events; event channel buffers 8 -> must be drained.
    let provider = MockProvider::new(vec![
        vec![
            StreamChunk::ToolUse {
                id: "c1".into(),
                name: "echo".into(),
                input: serde_json::json!({"text": "hi"}),
            },
            StreamChunk::Done,
        ],
        vec![StreamChunk::Text("done".into()), StreamChunk::Done],
    ]);
    let agent_loop = make_loop(provider, vec![Box::new(ChattyTool { bursts: 300 })]);
    let mut history = ConversationHistory::new("sys".into());
    history.push_user("run a tool");

    let (tx, mut rx) = mpsc::channel::<AgentEvent>(8);
    let cancel = CancellationToken::new();

    // Concurrent drainer (the role research_run's forwarder plays).
    let drainer = tokio::spawn(async move {
        let mut n = 0usize;
        while rx.recv().await.is_some() {
            n += 1;
        }
        n
    });

    let result = tokio::time::timeout(
        std::time::Duration::from_secs(10),
        agent_loop.run(&mut history, tx, cancel, None, None),
    )
    .await;

    assert!(
        result.is_ok(),
        "B80a: inner loop must complete when its events are drained; it hung \
         (the undrained-bounded-channel deadlock that froze research_run)"
    );
    assert!(
        result.unwrap().is_ok(),
        "B80a: turn should complete successfully"
    );

    let drained = drainer.await.unwrap();
    assert!(
        drained >= 300,
        "B80a: drainer must observe all >256 emitted events; saw {drained}"
    );
}

/// B118a: the loop guard must accumulate ACROSS iterations, not just within one.
///
/// The guard's thresholds and its own wording say "this turn", but it used to
/// be constructed inside the iteration loop. A model that emits one identical
/// call per model round-trip therefore reset the counter every time and could
/// retry forever. This is the shape of the 2026-08-08 incident: 20 consecutive
/// reasoning-only iterations, each retrying a failing approach, 2.8M tokens,
/// no answer.
///
/// Three iterations each send the SAME call with the SAME arguments. With a
/// turn-scoped guard the third is blocked; with a per-iteration guard all three
/// execute.
#[tokio::test]
async fn loop_guard_counts_identical_calls_across_iterations() {
    let identical_call = |id: &str| {
        vec![
            StreamChunk::ToolUse {
                id: id.into(),
                name: "echo".into(),
                input: serde_json::json!({"text": "same"}),
            },
            StreamChunk::Done,
        ]
    };
    let provider = MockProvider::new(vec![
        identical_call("i1"),
        identical_call("i2"),
        identical_call("i3"),
        vec![StreamChunk::Text("done".into()), StreamChunk::Done],
    ]);

    let executions = std::sync::Arc::new(AtomicUsize::new(0));
    let agent_loop = make_loop(
        provider,
        vec![Box::new(RecordingEchoTool {
            executions: std::sync::Arc::clone(&executions),
        })],
    );
    let mut history = ConversationHistory::new("sys".into());
    history.push_user("test");

    let (tx, mut rx) = mpsc::channel(64);
    let cancel = CancellationToken::new();
    let result = agent_loop.run(&mut history, tx, cancel, None, None).await;
    assert!(result.is_ok());

    assert_eq!(
        executions.load(Ordering::SeqCst),
        2,
        "the third identical call — made in a LATER iteration — must be blocked; \
         a per-iteration guard would let all 3 through"
    );

    let mut saw_cross_iteration_denial = false;
    while let Ok(ev) = rx.try_recv() {
        if let AgentEvent::ToolEnd {
            call_id,
            output,
            state,
            ..
        } = ev
            && call_id == "i3"
            && output.contains("already ran 3 times")
            && state == ToolState::Error
        {
            saw_cross_iteration_denial = true;
        }
    }
    assert!(
        saw_cross_iteration_denial,
        "the denial must name the turn-wide count, proving state survived the iteration boundary"
    );
}

/// B119c: a turn that only reasoned must not be filed as a healthy success.
///
/// `TurnStreamOutcome::empty` is documented as "no text, no tools, no
/// thinking", so a reasoning-only turn is not empty and used to take the
/// success path unconditionally. That is how the 2026-08-08 turn — 20
/// reasoning-only iterations that never produced an answer — was recorded as
/// healthy, hiding the failure from the operator watching model health.
///
/// The user still sees `(no text — reasoning only)`, so this pins the HEALTH
/// signal, not the presence of output.
#[tokio::test]
async fn reasoning_only_turn_is_not_recorded_as_success() {
    let health = std::sync::Arc::new(crate::model_catalog::ModelHealth::new(
        crate::model_catalog::ModelHealthConfig::default(),
    ));
    let provider = MockProvider::new(vec![vec![
        StreamChunk::Thinking("рассуждаю, но ответа не дам".into()),
        StreamChunk::Done,
    ]]);
    let agent_loop = AgentLoop::new(
        Box::new(provider),
        crate::tool::registry::ToolRegistry::new(vec![]),
        LoopConfig {
            max_iterations: 10,
            cwd: std::path::PathBuf::from("/tmp"),
            model: "mock".into(),
            provider: "mockprov".into(),
            max_tokens: 1024,
            health: Some(std::sync::Arc::clone(&health)),
            ..Default::default()
        },
    );

    let mut history = ConversationHistory::new("sys".into());
    history.push_user("вопрос");
    let (tx, _rx) = mpsc::channel(64);
    let result = agent_loop
        .run(&mut history, tx, CancellationToken::new(), None, None)
        .await;
    assert!(result.is_ok(), "the turn itself still completes");

    let snap = health.snapshot();
    let window = snap
        .iter()
        .find(|((p, _), _)| p == "mockprov")
        .map(|(_, w)| w)
        .expect("health must have recorded the turn");
    assert_eq!(
        window.success, 0,
        "a turn with no answer text must not count as success: {window:?}"
    );
    assert_eq!(
        window.empty, 1,
        "it must be recorded as an empty/no-answer turn instead: {window:?}"
    );
}

/// B157: a 429 connect failure must not be retried at the blind 1s
/// exponential backoff — the rate-limit floor (2s) applies so a
/// concurrency-limited proxy has time to drain our previous stream's
/// slot. The provider first rejects with RateLimited, then answers
/// normally; we measure the wall time between the two calls.
#[tokio::test]
async fn b157_rate_limited_connect_retry_respects_two_second_floor() {
    use std::sync::Mutex;
    use std::time::Instant;

    struct RateLimitThenAnswerProvider {
        call_count: AtomicUsize,
        call_times: std::sync::Arc<Mutex<Vec<Instant>>>,
    }

    #[async_trait::async_trait]
    impl Provider for RateLimitThenAnswerProvider {
        fn name(&self) -> &str {
            "mock429"
        }
        fn models(&self) -> Vec<crate::types::ModelInfo> {
            vec![]
        }
        async fn stream_chat(
            &self,
            _request: ChatRequest,
        ) -> crate::error::Result<Pin<Box<dyn tokio_stream::Stream<Item = StreamChunk> + Send>>>
        {
            let idx = self.call_count.fetch_add(1, Ordering::SeqCst);
            self.call_times.lock().unwrap().push(Instant::now());
            if idx == 0 {
                return Err(AgentError::ProviderTyped(
                    crate::provider::error::ProviderError::RateLimited {
                        retry_after: None,
                        body: "Too many concurrent requests for this API key".into(),
                    },
                ));
            }
            Ok(Box::pin(tokio_stream::iter(vec![
                StreamChunk::Text("recovered".into()),
                StreamChunk::Done,
            ])))
        }
    }

    let call_times = std::sync::Arc::new(Mutex::new(Vec::<Instant>::new()));
    let provider = RateLimitThenAnswerProvider {
        call_count: AtomicUsize::new(0),
        call_times: call_times.clone(),
    };

    let agent_loop = AgentLoop::new(
        Box::new(provider),
        crate::tool::registry::ToolRegistry::new(vec![]),
        LoopConfig {
            max_iterations: 5,
            max_wall: None,
            cwd: std::path::PathBuf::from("/tmp"),
            model: "mock429".into(),
            max_tokens: 1024,
            ..Default::default()
        },
    );
    let mut history = ConversationHistory::new("sys".into());
    history.push_user("ping");
    let (tx, _rx) = mpsc::channel(64);

    let result = agent_loop
        .run(&mut history, tx, CancellationToken::new(), None, None)
        .await;
    assert!(
        result.is_ok(),
        "turn must recover after the 429: {result:?}"
    );

    // Call times were recorded through the shared Arc — no raw-pointer
    // access needed (the crate forbids `unsafe`).
    let times: Vec<std::time::Instant> = call_times.lock().unwrap().clone();
    assert_eq!(times.len(), 2, "exactly two provider calls expected");
    let gap = times[1].duration_since(times[0]);
    assert!(
        gap >= std::time::Duration::from_secs(2),
        "B157: rate-limit retry must wait >= 2s (self-collision floor), waited {gap:?}"
    );
}
