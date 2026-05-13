use super::*;
use crate::provider::{ChatRequest, Provider};
use crate::tool::Tool;
use crate::types::{ContentBlock, Permission, Role, SteerMessage, ToolSpec, ToolState};
use std::pin::Pin;
use std::sync::atomic::{AtomicUsize, Ordering};

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

fn make_loop(provider: MockProvider, tools: Vec<Box<dyn Tool>>) -> AgentLoop {
    AgentLoop::new(
        Box::new(provider),
        crate::tool::registry::ToolRegistry::new(tools),
        LoopConfig {
            max_iterations: 10,
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
    // The provider is wired to return three identical empty streams
    // (initial + `MAX_EMPTY_CONTENT_RETRIES` retries) so the budget is
    // exhausted before we hit the MockProvider fallback.
    let empty_stream = || {
        vec![
            StreamChunk::Usage(TurnUsage {
                input_tokens: 42,
                output_tokens: 0,
                cache_read_tokens: 0,
                cache_write_tokens: 0,
            }),
            StreamChunk::Done,
        ]
    };
    let provider = MockProvider::new(vec![empty_stream(), empty_stream(), empty_stream()]);
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
async fn loop_empty_then_text_retries_and_succeeds() {
    // Targeted regression for `glm-5-turbo`-style transient empty
    // responses: the loop must transparently retry and surface the
    // text from the second attempt, returning Ok without exposing
    // an Error event for the (recovered) hiccup.
    let provider = MockProvider::new(vec![
        // attempt 1: empty stream
        vec![StreamChunk::Done],
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
        vec![StreamChunk::Done],
        vec![StreamChunk::Done],
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

    struct DangerTool;
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
            crate::types::ToolResult {
                output: "executed".into(),
                is_error: false,
            }
        }
    }

    let tools: Vec<Box<dyn Tool>> = vec![Box::new(DangerTool)];
    let agent_loop = make_loop(provider, tools);
    let mut history = ConversationHistory::new("sys".into());
    history.push_user("do it");

    let (tx, mut rx) = mpsc::channel(64);
    let cancel = CancellationToken::new();

    let (perm_tx, perm_rx) = mpsc::channel(4);

    let loop_handle = tokio::spawn(async move {
        agent_loop
            .run(&mut history, tx, cancel, Some(perm_rx), None)
            .await
    });

    let mut saw_permission_request = false;
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
        if matches!(ev, AgentEvent::Idle) {
            break;
        }
    }

    assert!(saw_permission_request);
    let result = loop_handle.await.unwrap();
    assert!(result.is_ok());
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

    let agent_loop = AgentLoop::with_policy(
        Box::new(provider),
        crate::tool::registry::ToolRegistry::new(vec![Box::new(EchoTool)]),
        LoopConfig {
            max_iterations: 10,
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

    let _ = agent_loop.run(&mut history, tx, cancel, None, None).await;

    // Collect events
    let mut saw_deny = false;
    while let Ok(ev) = rx.try_recv() {
        if let AgentEvent::ToolEnd { output, state, .. } = ev
            && output.contains("denied by policy")
            && state == ToolState::Error
        {
            saw_deny = true;
        }
    }
    assert!(saw_deny, "policy denial should emit ToolEnd with error");
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
