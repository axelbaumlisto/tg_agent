use tokio::sync::mpsc;
use tokio_stream::StreamExt;
use tokio_util::sync::CancellationToken;

use crate::error::{AgentError, Result};
use crate::history::ConversationHistory;
use crate::provider::{ChatRequest, Provider};
use crate::tool::registry::ToolRegistry;
use crate::types::{
    AgentEvent, ContentBlock, Permission, PermissionResponse, StreamChunk, ToolState, TurnUsage,
};

const MAX_STREAM_RETRIES: usize = 3;
const BASE_RETRY_DELAY_MS: u64 = 1000;
const HEARTBEAT_INTERVAL: std::time::Duration = std::time::Duration::from_secs(5);

/// How many times to retry the *same* turn after the provider closes the
/// stream with zero text, zero reasoning and zero tool calls. Some
/// providers — notably `glm-5-turbo` and occasionally Alibaba routes —
/// hiccup like this for a single second under load. A bounded retry lets
/// the agent recover transparently instead of failing the turn and asking
/// the operator to re-issue the prompt.
const MAX_EMPTY_CONTENT_RETRIES: usize = 2;
/// Base back-off for empty-content retries. Kept short (250 ms) because
/// the symptom is a *closed-but-empty* stream, not rate-limit, so we want
/// to come back fast. Doubles per attempt (250 → 500).
const EMPTY_CONTENT_BASE_DELAY_MS: u64 = 250;

pub struct LoopConfig {
    pub max_iterations: usize,
    pub cwd: std::path::PathBuf,
    pub model: String,
    pub max_tokens: u32,
    pub temperature: Option<f32>,
    pub reasoning: Option<String>,
    /// Provider id the `model` lives under (e.g. `"zai"`, `"kimi-code"`).
    /// Required by the Phase 3 [`crate::model_catalog::ModelHealth`]
    /// tracker to key per-pair counters. Default `""` — when empty, the
    /// loop skips health recording (keeps tests with a raw
    /// `LoopConfig::default()` working unchanged).
    pub provider: String,
    /// Optional health tracker handle. When attached, the loop reports
    /// Success / Empty / Error events at the three termination branches
    /// so the selector can quarantine misbehaving pairs across
    /// restarts. `None` disables recording entirely.
    pub health: Option<std::sync::Arc<crate::model_catalog::ModelHealth>>,
}

impl Default for LoopConfig {
    fn default() -> Self {
        Self {
            max_iterations: 0,
            cwd: std::env::current_dir().unwrap_or_default(),
            model: String::new(),
            max_tokens: 16384,
            temperature: None,
            reasoning: None,
            provider: String::new(),
            health: None,
        }
    }
}

impl LoopConfig {
    /// Record one health event iff a tracker is attached and the
    /// provider id is non-empty. Centralised so the three call sites
    /// don't repeat the same guard.
    pub(crate) fn record_health(
        &self,
        kind: crate::model_catalog::HealthEventKind,
        latency_ms: Option<u64>,
        detail: Option<String>,
    ) {
        if self.provider.is_empty() {
            return;
        }
        if let Some(h) = &self.health {
            h.record(&self.provider, &self.model, kind, latency_ms, detail);
        }
    }
}

pub struct AgentLoop {
    provider: Box<dyn Provider>,
    tools: ToolRegistry,
    config: LoopConfig,
    policy: Box<dyn crate::tool::policy::ToolPolicy>,
}

impl AgentLoop {
    pub fn new(provider: Box<dyn Provider>, tools: ToolRegistry, config: LoopConfig) -> Self {
        Self {
            provider,
            tools,
            config,
            policy: Box::new(crate::tool::policy::DefaultPolicy),
        }
    }

    /// Create with a custom tool policy.
    pub fn with_policy(
        provider: Box<dyn Provider>,
        tools: ToolRegistry,
        config: LoopConfig,
        policy: Box<dyn crate::tool::policy::ToolPolicy>,
    ) -> Self {
        Self {
            provider,
            tools,
            config,
            policy,
        }
    }

    pub async fn run(
        &self,
        history: &mut ConversationHistory,
        tx: mpsc::Sender<AgentEvent>,
        cancel: CancellationToken,
        mut permission_rx: Option<mpsc::Receiver<PermissionResponse>>,
    ) -> Result<TurnUsage> {
        let mut cumulative_usage = TurnUsage::default();

        let limit = if self.config.max_iterations == 0 {
            usize::MAX
        } else {
            self.config.max_iterations
        };
        // Counts consecutive empty-content responses across iterations.
        // Reset whenever an iteration produces real content so a normal
        // tool-use loop never accidentally exhausts the budget; only bursts
        // of empty responses (the actual `glm-5-turbo` failure mode) are
        // capped.
        let mut empty_content_attempts: usize = 0;
        'outer: for _iteration in 0..limit {
            if cancel.is_cancelled() {
                return Err(AgentError::Cancelled);
            }

            let messages = history.to_api_messages();
            let system = history.system_prompt().to_string();

            let request = ChatRequest {
                model: self.config.model.clone(),
                system,
                messages,
                tools: self.tools.schemas_json(),
                max_tokens: self.config.max_tokens,
                temperature: self.config.temperature,
                reasoning: self.config.reasoning.clone(),
            };

            let mut text_acc = String::new();
            let mut thinking_acc = String::new();
            let mut blocks: Vec<ContentBlock> = Vec::new();
            let mut tool_calls: Vec<(String, String, serde_json::Value)> = Vec::new();
            let mut turn_usage: Option<TurnUsage> = None;
            let mut stream_ok = false;

            for retry in 0..=MAX_STREAM_RETRIES {
                let connect_result = tokio::select! {
                    _ = cancel.cancelled() => {
                        return Err(AgentError::Cancelled);
                    }
                    r = self.provider.stream_chat(request.clone()) => r,
                };
                let mut stream = match connect_result {
                    Ok(s) => s,
                    Err(e) => {
                        let err_str = e.to_string().to_lowercase();
                        if err_str.contains("prompt is too long")
                            || err_str.contains("context_length_exceeded")
                            || err_str.contains("maximum context length")
                        {
                            let before = history.message_count();
                            if before <= 3 {
                                return Err(e);
                            }
                            for round in 0..5 {
                                let keep = (history.message_count() / 3).clamp(2, 6);
                                tracing::warn!(
                                    round,
                                    keep,
                                    msgs = history.message_count(),
                                    est_tokens = history.estimated_tokens(),
                                    "emergency compaction round"
                                );
                                history.compact(keep);
                                if history.estimated_tokens()
                                    < history.context_window_tokens() as usize * 4 / 5
                                {
                                    break;
                                }
                            }
                            let after = history.message_count();
                            tracing::warn!("emergency compaction done: {before} -> {after} msgs");
                            let _ = tx
                                .send(AgentEvent::ContextCompacted {
                                    before_msgs: before,
                                    after_msgs: after,
                                    summary_hint: None,
                                    files_count: 0,
                                })
                                .await;
                            continue 'outer;
                        }
                        if retry < MAX_STREAM_RETRIES {
                            let delay = BASE_RETRY_DELAY_MS * 2u64.pow(retry as u32);
                            tracing::warn!(
                                "stream_chat connect error (retry {}/{}, backoff {delay}ms): {e}",
                                retry + 1,
                                MAX_STREAM_RETRIES
                            );
                            tokio::time::sleep(std::time::Duration::from_millis(delay)).await;
                            continue;
                        }
                        self.config.record_health(
                            crate::model_catalog::HealthEventKind::Error,
                            None,
                            Some(e.to_string()),
                        );
                        return Err(e);
                    }
                };

                text_acc.clear();
                thinking_acc.clear();
                blocks.clear();
                tool_calls.clear();
                turn_usage = None;
                let mut mid_stream_error = None;

                loop {
                    let chunk = tokio::select! {
                        _ = cancel.cancelled() => {
                            return Err(AgentError::Cancelled);
                        }
                        next = stream.next() => match next {
                            Some(c) => c,
                            None => break,
                        },
                    };

                    match chunk {
                        StreamChunk::Text(t) => {
                            let _ = tx.send(AgentEvent::TextDelta(t.clone())).await;
                            text_acc.push_str(&t);
                        }
                        StreamChunk::Thinking(t) => {
                            let _ = tx.send(AgentEvent::ThinkingDelta(t.clone())).await;
                            thinking_acc.push_str(&t);
                        }
                        StreamChunk::ToolUse { id, name, input } => {
                            if !thinking_acc.is_empty() {
                                blocks.push(ContentBlock::Thinking {
                                    text: std::mem::take(&mut thinking_acc),
                                });
                            }
                            if !text_acc.is_empty() {
                                blocks.push(ContentBlock::Text {
                                    text: std::mem::take(&mut text_acc),
                                });
                            }
                            let _ = tx
                                .send(AgentEvent::ToolStart {
                                    call_id: id.clone(),
                                    name: name.clone(),
                                    input: input.clone(),
                                })
                                .await;
                            blocks.push(ContentBlock::ToolUse {
                                id: id.clone(),
                                name: name.clone(),
                                input: input.clone(),
                            });
                            tool_calls.push((id, name, input));
                        }
                        StreamChunk::Usage(u) => {
                            cumulative_usage.input_tokens += u.input_tokens;
                            cumulative_usage.output_tokens += u.output_tokens;
                            cumulative_usage.cache_read_tokens += u.cache_read_tokens;
                            cumulative_usage.cache_write_tokens += u.cache_write_tokens;
                            turn_usage = Some(u.clone());
                            let _ = tx.send(AgentEvent::UsageUpdate(u)).await;
                        }
                        StreamChunk::Done => break,
                        StreamChunk::Error(e) => {
                            mid_stream_error = Some(e);
                            break;
                        }
                    }
                }

                if let Some(e) = mid_stream_error {
                    if tool_calls.is_empty() && retry < MAX_STREAM_RETRIES {
                        let delay = BASE_RETRY_DELAY_MS * 2u64.pow(retry as u32);
                        tracing::warn!(
                            "mid-stream error (retry {}/{}, backoff {delay}ms): {e}",
                            retry + 1,
                            MAX_STREAM_RETRIES
                        );
                        tokio::time::sleep(std::time::Duration::from_millis(delay)).await;
                        continue;
                    }
                    let _ = tx.send(AgentEvent::Error(e.clone())).await;
                    self.config.record_health(
                        crate::model_catalog::HealthEventKind::Error,
                        None,
                        Some(e.clone()),
                    );
                    return Err(AgentError::Provider(e));
                }

                stream_ok = true;
                break;
            }

            if !stream_ok {
                let _ = tx
                    .send(AgentEvent::Error("stream retries exhausted".into()))
                    .await;
                self.config.record_health(
                    crate::model_catalog::HealthEventKind::Error,
                    None,
                    Some("stream retries exhausted".into()),
                );
                return Err(AgentError::Provider("stream retries exhausted".into()));
            }

            if !thinking_acc.is_empty() {
                blocks.push(ContentBlock::Thinking { text: thinking_acc });
            }
            if !text_acc.is_empty() {
                blocks.push(ContentBlock::Text { text: text_acc });
            }

            // Guard against "provider returned nothing" turns.
            //
            // Some providers (notably `glm-cn`/`glm-5-turbo`) can close a
            // stream with zero output tokens, no tool calls, and no
            // reasoning chunks. The first occurrence is almost always
            // transient under load, so we retry the same request up to
            // `MAX_EMPTY_CONTENT_RETRIES` times with a short back-off.
            // Only when retries are exhausted do we surface an Error and
            // return without polluting history (which would cause the
            // "dirty-session 0-tok refusal" loop on subsequent turns).
            if blocks.is_empty() && tool_calls.is_empty() {
                if empty_content_attempts < MAX_EMPTY_CONTENT_RETRIES {
                    empty_content_attempts += 1;
                    crate::types::EMPTY_CONTENT_RETRY_COUNT
                        .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                    let delay =
                        EMPTY_CONTENT_BASE_DELAY_MS * 2u64.pow((empty_content_attempts - 1) as u32);
                    tracing::warn!(
                        "provider returned empty content (retry {}/{}, backoff {delay}ms)",
                        empty_content_attempts,
                        MAX_EMPTY_CONTENT_RETRIES,
                    );
                    tokio::time::sleep(std::time::Duration::from_millis(delay)).await;
                    continue 'outer;
                }
                let msg = format!(
                    "provider returned no content {} times in a row (no text, no reasoning, no tool calls)",
                    empty_content_attempts + 1,
                );
                tracing::warn!("{msg}");
                let _ = tx.send(AgentEvent::Error(msg.clone())).await;
                let _ = tx.send(AgentEvent::Idle).await;
                self.config.record_health(
                    crate::model_catalog::HealthEventKind::Empty,
                    None,
                    Some(msg.clone()),
                );
                return Err(AgentError::Provider(msg));
            }
            // Reset the empty-content budget once we got real content.
            // This means a transient hiccup at iteration N doesn't starve
            // retries at iteration N+5.
            empty_content_attempts = 0;

            history.push_assistant(blocks, turn_usage);

            if tool_calls.is_empty() {
                let _ = tx.send(AgentEvent::Idle).await;
                self.config.record_health(
                    crate::model_catalog::HealthEventKind::Success,
                    None,
                    None,
                );
                return Ok(cumulative_usage);
            }

            // Classify tool calls via policy (Step 2: no permission rules in loop).
            use crate::tool::policy::ToolDecision;
            let mut readonly_batch: Vec<(String, String, serde_json::Value)> = Vec::new();
            let mut gated_calls: Vec<(String, String, serde_json::Value)> = Vec::new();
            let mut denied_calls: Vec<(String, String, String)> = Vec::new(); // (id, name, reason)

            for (id, name, input) in tool_calls {
                let perm = self
                    .tools
                    .get(&name)
                    .map(|t| t.effective_permission(&input, &self.config.cwd))
                    .unwrap_or(Permission::Dangerous);

                match self.policy.classify(&name, &input, &self.config.cwd, perm) {
                    ToolDecision::Execute => readonly_batch.push((id, name, input)),
                    ToolDecision::AskUser(_) => gated_calls.push((id, name, input)),
                    ToolDecision::Deny(reason) => denied_calls.push((id, name, reason)),
                }
            }

            // Emit denied tool results.
            for (id, name, reason) in denied_calls {
                let _ = tx
                    .send(AgentEvent::ToolEnd {
                        call_id: id.clone(),
                        name,
                        state: ToolState::Error,
                        output: reason.clone(),
                    })
                    .await;
                history.push_tool_result(&id, &reason, true);
            }

            // Execute read-only tools in parallel (no permission needed).
            if !readonly_batch.is_empty() {
                let futs: Vec<_> = readonly_batch
                    .iter()
                    .map(|(id, name, input)| {
                        let id = id.clone();
                        let name = name.clone();
                        let input = input.clone();
                        let cwd = self.config.cwd.clone();
                        let tools = &self.tools;
                        let progress = tx.clone();
                        async move {
                            let result = tools
                                .execute_with_progress(&name, input, &cwd, progress)
                                .await;
                            (id, name, result)
                        }
                    })
                    .collect();

                let results = tokio::select! {
                    _ = cancel.cancelled() => {
                        return Err(AgentError::Cancelled);
                    }
                    r = futures_util::future::join_all(futs) => r,
                };
                for (id, name, result) in results {
                    let state = if result.is_error {
                        ToolState::Error
                    } else {
                        ToolState::Completed
                    };
                    let _ = tx
                        .send(AgentEvent::ToolEnd {
                            call_id: id.clone(),
                            name,
                            state,
                            output: result.output.clone(),
                        })
                        .await;
                    history.push_tool_result(&id, &result.output, result.is_error);
                    // Inject any images produced by the tool.
                    for (mime, b64) in crate::tool::image_result::drain_images() {
                        history.push_image(&mime, &b64);
                    }
                }
            }

            // Execute gated tools sequentially (require permission).
            for (id, name, input) in gated_calls {
                let perm = self
                    .tools
                    .get(&name)
                    .map(|t| t.effective_permission(&input, &self.config.cwd))
                    .unwrap_or(Permission::Dangerous);

                let allowed = if let Some(ref mut prx) = permission_rx {
                    let _ = tx
                        .send(AgentEvent::PermissionRequest {
                            call_id: id.clone(),
                            tool_name: name.clone(),
                            input: input.clone(),
                            permission: perm,
                        })
                        .await;
                    match prx.recv().await {
                        Some(resp) if resp.call_id == id => resp.allowed,
                        _ => false,
                    }
                } else {
                    true
                };

                if !allowed {
                    let _ = tx
                        .send(AgentEvent::ToolEnd {
                            call_id: id.clone(),
                            name,
                            state: ToolState::Error,
                            output: "Permission denied by user".into(),
                        })
                        .await;
                    history.push_tool_result(&id, "Permission denied by user", true);
                    continue;
                }

                let progress_tx = tx.clone();
                let heartbeat_tx = tx.clone();
                let heartbeat_cancel = CancellationToken::new();
                let hb_token = heartbeat_cancel.clone();

                let hb_handle = tokio::spawn(async move {
                    loop {
                        tokio::select! {
                            _ = hb_token.cancelled() => break,
                            _ = tokio::time::sleep(HEARTBEAT_INTERVAL) => {
                                let _ = heartbeat_tx.send(AgentEvent::Heartbeat).await;
                            }
                        }
                    }
                });

                let tool_fut =
                    self.tools
                        .execute_with_progress(&name, input, &self.config.cwd, progress_tx);

                let result = tokio::select! {
                    _ = cancel.cancelled() => {
                        heartbeat_cancel.cancel();
                        let _ = hb_handle.await;
                        return Err(AgentError::Cancelled);
                    }
                    r = tool_fut => r,
                };

                heartbeat_cancel.cancel();
                let _ = hb_handle.await;

                let state = if result.is_error {
                    ToolState::Error
                } else {
                    ToolState::Completed
                };
                let _ = tx
                    .send(AgentEvent::ToolEnd {
                        call_id: id.clone(),
                        name,
                        state,
                        output: result.output.clone(),
                    })
                    .await;
                history.push_tool_result(&id, &result.output, result.is_error);
                for (mime, b64) in crate::tool::image_result::drain_images() {
                    history.push_image(&mime, &b64);
                }
            }
        }

        let _ = tx
            .send(AgentEvent::Error("Max iterations exceeded".into()))
            .await;
        Err(AgentError::MaxIterations(self.config.max_iterations))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::provider::{ChatRequest, Provider};
    use crate::tool::Tool;
    use crate::types::{Permission, ToolSpec};
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
        ) -> crate::error::Result<Pin<Box<dyn tokio_stream::Stream<Item = StreamChunk> + Send>>>
        {
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
                temperature: None,
                reasoning: None,
                provider: String::new(),
                health: None,
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

        let result = agent_loop.run(&mut history, tx, cancel, None).await;
        assert!(
            matches!(result, Err(AgentError::Provider(_))),
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

        let result = agent_loop.run(&mut history, tx, cancel, None).await;
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

        let result = agent_loop.run(&mut history, tx, cancel, None).await;
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

        let result = agent_loop.run(&mut history, tx, cancel, None).await;
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

        let result = agent_loop.run(&mut history, tx, cancel, None).await;
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

        let result = agent_loop.run(&mut history, tx, cancel, None).await;
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
                temperature: None,
                reasoning: None,
                provider: String::new(),
                health: None,
            },
        );
        let mut history = ConversationHistory::new("sys".into());
        history.push_user("hi");

        let (tx, _rx) = mpsc::channel(64);
        let cancel = CancellationToken::new();

        let result = agent_loop.run(&mut history, tx, cancel, None).await;
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

        let result = agent_loop.run(&mut history, tx, cancel, None).await;
        assert!(matches!(result, Err(AgentError::Provider(_))));

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
            .run(&mut history, tx, cancel, None)
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
            .run(&mut history, tx, cancel, None)
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
            .run(&mut history, tx, cancel, None)
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
            .run(&mut history, tx, cancel, None)
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
                .run(&mut history, tx, cancel, Some(perm_rx))
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
                .run(&mut history, tx, cancel, Some(perm_rx))
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
                temperature: None,
                reasoning: None,
                provider: String::new(),
                health: None,
            },
            Box::new(DenyEchoPolicy),
        );

        let (tx, mut rx) = mpsc::channel(64);
        let cancel = CancellationToken::new();
        let mut history = ConversationHistory::new(String::new());
        history.push_user("test");

        let _ = agent_loop.run(&mut history, tx, cancel, None).await;

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
}
