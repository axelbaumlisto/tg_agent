use tokio::sync::mpsc;
use tokio_stream::StreamExt;
use tokio_util::sync::CancellationToken;

use crate::error::{AgentError, Result};
use crate::history::ConversationHistory;
use crate::provider::{ChatRequest, Provider};
use crate::retry::Backoff;
use crate::tool::registry::ToolRegistry;
use crate::types::{
    AgentEvent, ContentBlock, Permission, PermissionResponse, SteerMessage, StreamChunk, ToolState,
    TurnUsage,
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
    /// Optional token tracker — records (model, in, out) per turn.
    pub token_tracker: Option<crate::token_tracker::TokenTracker>,
    /// Optional audit directory — logs tool calls to JSONL.
    pub audit_dir: Option<std::path::PathBuf>,
    /// Checkpoint-restart cycle configuration.
    pub cycle_config: Option<crate::session::cycle::CycleConfig>,
    /// Session id for cycle archive naming.
    pub session_id: Option<String>,
    /// Data directory for cycle archives (e.g. state/data).
    pub data_dir: Option<std::path::PathBuf>,
    /// Shared working set for file tracking across turns.
    pub working_set: Option<std::sync::Arc<std::sync::Mutex<crate::working_set::WorkingSet>>>,
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
            token_tracker: None,
            audit_dir: None,
            cycle_config: None,
            session_id: None,
            data_dir: None,
            working_set: None,
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
    approval_cache: crate::tool::approval_cache::ApprovalCache,
}

impl AgentLoop {
    pub fn new(provider: Box<dyn Provider>, tools: ToolRegistry, config: LoopConfig) -> Self {
        let own_source = Self::detect_own_source_dir(&config.cwd);
        Self {
            provider,
            tools,
            config,
            policy: Box::new(crate::tool::policy::default_pipeline(own_source)),
            approval_cache: crate::tool::approval_cache::ApprovalCache::new(),
        }
    }

    /// Detect the bot's own source directory from the workspace path.
    /// If cwd contains a `crates/naked-core/` directory, that's our source tree.
    fn detect_own_source_dir(cwd: &std::path::Path) -> Option<std::path::PathBuf> {
        // Walk up from cwd looking for our Cargo workspace with naked-core
        let mut dir = cwd.to_path_buf();
        for _ in 0..5 {
            if dir.join("crates/naked-core/src").is_dir() {
                return Some(dir.join("crates"));
            }
            if !dir.pop() {
                break;
            }
        }
        None
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
            approval_cache: crate::tool::approval_cache::ApprovalCache::new(),
        }
    }

    pub async fn run(
        &self,
        history: &mut ConversationHistory,
        tx: mpsc::Sender<AgentEvent>,
        cancel: CancellationToken,
        mut permission_rx: Option<mpsc::Receiver<PermissionResponse>>,
        mut steer_rx: Option<mpsc::Receiver<SteerMessage>>,
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
        // Pending steer messages not yet flushed to history.
        // Kept as a vec so edits can replace by msg_id before drain.
        let mut pending_steers: Vec<SteerMessage> = Vec::new();
        // msg_ids already flushed to history (for edit-after-drain detection).
        let mut delivered_msg_ids: std::collections::HashSet<i32> =
            std::collections::HashSet::new();

        'outer: for _iteration in 0..limit {
            if cancel.is_cancelled() {
                return Err(AgentError::Cancelled);
            }

            // Per-iteration guard — blocks identical repeated calls and
            // halts after too many consecutive failures.
            let mut loop_guard = crate::loop_guard::LoopGuard::default();

            // Drain steer messages between iterations.
            Self::drain_steers(
                &mut steer_rx,
                &mut pending_steers,
                &mut delivered_msg_ids,
                history,
                &tx,
            )
            .await;

            // Checkpoint-restart cycle: if token usage exceeds threshold,
            // archive old messages and restart with fresh context.
            if let Some(ref cycle_cfg) = self.config.cycle_config {
                let est = history.estimated_tokens() as u64;
                if crate::session::cycle::should_advance_cycle(est, cycle_cfg)
                    && let Some(ref data_dir) = self.config.data_dir
                {
                    let session_id = self.config.session_id.as_deref().unwrap_or("unknown");
                    let cycle_num = history.cycle_count();
                    let checkpoint = crate::session::cycle::build_checkpoint(
                        cycle_num,
                        history.messages(),
                        est,
                        None, // TODO: pass working_set when available
                        cycle_cfg,
                    );
                    if let Ok(archive_path) = crate::session::cycle::write_archive(
                        data_dir,
                        session_id,
                        cycle_num,
                        history.messages(),
                    ) {
                        let restart_prompt = crate::session::cycle::build_restart_prompt(
                            &checkpoint,
                            history.system_prompt(),
                        );
                        let archived_count = history.message_count();
                        history.clear_for_cycle_restart(&restart_prompt);
                        tracing::info!(
                            cycle = cycle_num,
                            archived = archived_count,
                            path = %archive_path.display(),
                            "cycle restart: archived and restarted"
                        );
                        let _ = tx
                            .send(AgentEvent::CycleRestarted {
                                cycle_number: cycle_num,
                                archived_messages: archived_count,
                                archive_path: archive_path.display().to_string(),
                            })
                            .await;
                    }
                }
            }

            let request = self.build_chat_request(history);

            let mut text_acc = String::new();
            let mut thinking_acc = String::new();
            let mut in_fake_tool = false;
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
                            let delay = Backoff {
                                base_ms: BASE_RETRY_DELAY_MS,
                                max_attempts: MAX_STREAM_RETRIES + 1,
                            }
                            .delay(retry);
                            tracing::warn!(
                                "stream_chat connect error (retry {}/{}, backoff {}ms): {e}",
                                retry + 1,
                                MAX_STREAM_RETRIES,
                                delay.as_millis()
                            );
                            tokio::time::sleep(delay).await;
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
                            let cleaned =
                                crate::stream_filter::filter_fake_tool_delta(&t, &mut in_fake_tool);
                            if !cleaned.is_empty() {
                                let _ = tx.send(AgentEvent::TextDelta(cleaned.clone())).await;
                                text_acc.push_str(&cleaned);
                            } else if !t.is_empty() {
                                tracing::debug!("stream_filter: scrubbed fake tool wrapper");
                            }
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
                            if let Some(ref tracker) = self.config.token_tracker {
                                tracker.record(&self.config.model, u.input_tokens, u.output_tokens);
                            }
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
                        let delay = Backoff {
                            base_ms: BASE_RETRY_DELAY_MS,
                            max_attempts: MAX_STREAM_RETRIES + 1,
                        }
                        .delay(retry);
                        tracing::warn!(
                            "mid-stream error (retry {}/{}, backoff {}ms): {e}",
                            retry + 1,
                            MAX_STREAM_RETRIES,
                            delay.as_millis()
                        );
                        tokio::time::sleep(delay).await;
                        continue;
                    }
                    let _ = tx.send(AgentEvent::Error(e.clone())).await;
                    self.config.record_health(
                        crate::model_catalog::HealthEventKind::Error,
                        None,
                        Some(e.clone()),
                    );
                    return Err(AgentError::ProviderTyped(
                        crate::provider::error::ProviderError::Other { status: 0, body: e },
                    ));
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
                return Err(AgentError::ProviderTyped(
                    crate::provider::error::ProviderError::Other {
                        status: 0,
                        body: "stream retries exhausted".into(),
                    },
                ));
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
                    let backoff = crate::retry::Backoff {
                        base_ms: EMPTY_CONTENT_BASE_DELAY_MS,
                        max_attempts: MAX_EMPTY_CONTENT_RETRIES,
                    };
                    let delay = backoff.delay(empty_content_attempts - 1);
                    tracing::warn!(
                        "provider returned empty content (retry {}/{}, backoff {}ms)",
                        empty_content_attempts,
                        MAX_EMPTY_CONTENT_RETRIES,
                        delay.as_millis(),
                    );
                    tokio::time::sleep(delay).await;
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
                return Err(AgentError::ProviderTyped(
                    crate::provider::error::ProviderError::Other {
                        status: 0,
                        body: msg,
                    },
                ));
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

            let (readonly_batch, gated_calls, denied_calls) =
                self.classify_tool_calls(tool_calls, &mut loop_guard);

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
                let results = Self::execute_readonly_batch(
                    &self.tools,
                    &readonly_batch,
                    &self.config.cwd,
                    &cancel,
                    &tx,
                    &mut steer_rx,
                    &mut pending_steers,
                )
                .await?;
                for (id, name, result) in results {
                    let state = if result.is_error {
                        ToolState::Error
                    } else {
                        ToolState::Completed
                    };
                    let _ = tx
                        .send(AgentEvent::ToolEnd {
                            call_id: id.clone(),
                            name: name.clone(),
                            state,
                            output: result.output.clone(),
                        })
                        .await;
                    Self::push_tool_outcome(
                        history,
                        &mut loop_guard,
                        &id,
                        &name,
                        &result.output,
                        result.is_error,
                        history.context_window_tokens() as u64,
                    );
                }
            }

            // Execute gated tools sequentially (require permission).
            for (id, name, input) in gated_calls {
                // Drain steer messages between sequential tool calls.
                Self::drain_steers(
                    &mut steer_rx,
                    &mut pending_steers,
                    &mut delivered_msg_ids,
                    history,
                    &tx,
                )
                .await;

                let perm = self
                    .tools
                    .get(&name)
                    .map(|t| t.effective_permission(&input, &self.config.cwd))
                    .unwrap_or(Permission::Dangerous);

                // Check approval cache before prompting user:
                let fp = crate::tool::approval_cache::fingerprint(&name, &input);
                let cached = self.approval_cache.is_approved(&fp);

                let allowed = if cached {
                    // Previously approved fingerprint — auto-approve.
                    let _ = tx
                        .send(AgentEvent::ToolOutput {
                            call_id: id.clone(),
                            chunk: format!("\u{2705} auto-approved (cached: {fp})"),
                        })
                        .await;
                    true
                } else if let Some(ref mut prx) = permission_rx {
                    let _ = tx
                        .send(AgentEvent::PermissionRequest {
                            call_id: id.clone(),
                            tool_name: name.clone(),
                            input: input.clone(),
                            permission: perm,
                        })
                        .await;
                    match prx.recv().await {
                        Some(resp) if resp.call_id == id => {
                            if resp.allowed {
                                self.approval_cache.approve(&fp);
                            }
                            resp.allowed
                        }
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

                let result = self
                    .execute_tool_with_heartbeat(
                        &name,
                        input,
                        &id,
                        &cancel,
                        &tx,
                        &mut steer_rx,
                        &mut pending_steers,
                    )
                    .await?;

                let state = if result.is_error {
                    ToolState::Error
                } else {
                    ToolState::Completed
                };
                // Audit gated tool execution:
                if let Some(ref dir) = self.config.audit_dir {
                    crate::audit::log_event(
                        dir,
                        "tool_exec",
                        serde_json::json!({"tool": &name, "error": result.is_error}),
                    );
                }
                let _ = tx
                    .send(AgentEvent::ToolEnd {
                        call_id: id.clone(),
                        name: name.clone(),
                        state,
                        output: result.output.clone(),
                    })
                    .await;
                Self::push_tool_outcome(
                    history,
                    &mut loop_guard,
                    &id,
                    &name,
                    &result.output,
                    result.is_error,
                    history.context_window_tokens() as u64,
                );
            }
        }

        let _ = tx
            .send(AgentEvent::Error("Max iterations exceeded".into()))
            .await;
        Err(AgentError::MaxIterations(self.config.max_iterations))
    }

    /// Drain all pending steer messages from the channel, merge them into
    /// a single user message, and push it to history.
    ///
    /// Edit semantics: if `is_edit` and the msg_id is still in the pending
    /// queue, replace in-place. If already delivered to history, add a
    /// `[correction]` prefix so the LLM knows the user changed their mind.
    /// Classify tool calls into readonly / gated / denied batches.
    /// Execute read-only tools in parallel, cancellable.
    async fn execute_readonly_batch(
        tools: &ToolRegistry,
        batch: &[(String, String, serde_json::Value)],
        cwd: &std::path::Path,
        cancel: &CancellationToken,
        tx: &mpsc::Sender<AgentEvent>,
        steer_rx: &mut Option<mpsc::Receiver<SteerMessage>>,
        pending_steers: &mut Vec<SteerMessage>,
    ) -> Result<Vec<(String, String, crate::types::ToolResult)>> {
        let futs: Vec<_> = batch
            .iter()
            .map(|(id, name, input)| {
                let id = id.clone();
                let name = name.clone();
                let input = input.clone();
                let cwd = cwd.to_path_buf();
                let progress = tx.clone();
                async move {
                    let result = tools
                        .execute_with_progress(&name, input, &cwd, progress)
                        .await;
                    (id, name, result)
                }
            })
            .collect();

        let join = futures_util::future::join_all(futs);
        let mut join = std::pin::pin!(join);
        loop {
            tokio::select! {
                _ = cancel.cancelled() => return Err(AgentError::Cancelled),
                r = &mut join => return Ok(r),
                msg = async {
                    match steer_rx.as_mut() {
                        Some(rx) => rx.recv().await,
                        None => std::future::pending().await,
                    }
                } => {
                    if let Some(msg) = msg {
                        pending_steers.push(msg);
                    }
                }
            }
        }
    }

    #[allow(clippy::type_complexity)]
    fn classify_tool_calls(
        &self,
        tool_calls: Vec<(String, String, serde_json::Value)>,
        guard: &mut crate::loop_guard::LoopGuard,
    ) -> (
        Vec<(String, String, serde_json::Value)>,
        Vec<(String, String, serde_json::Value)>,
        Vec<(String, String, String)>,
    ) {
        use crate::tool::policy::ToolDecision;
        let mut readonly_batch = Vec::new();
        let mut gated_calls = Vec::new();
        let mut denied_calls = Vec::new();

        for (id, name, input) in tool_calls {
            if let crate::loop_guard::AttemptDecision::Block(reason) =
                guard.record_attempt(&name, &input)
            {
                denied_calls.push((id, name, reason));
                continue;
            }
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

        // WorkingSet: observe tool calls before execution.
        if let Some(ref ws) = self.config.working_set
            && let Ok(mut ws) = ws.lock()
        {
            for (_, name, input) in readonly_batch.iter().chain(gated_calls.iter()) {
                ws.observe_tool(name, input);
            }
        }

        (readonly_batch, gated_calls, denied_calls)
    }

    /// Build the ChatRequest for the current turn.
    fn build_chat_request(&self, history: &ConversationHistory) -> ChatRequest {
        let messages = history.to_api_messages();
        let system = history.system_prompt().to_string();
        let reasoning = {
            let base = self.config.reasoning.clone();
            if base.as_deref() == Some("auto") {
                let last_msg = history
                    .messages()
                    .iter()
                    .rev()
                    .find(|m| m.role == crate::types::Role::User)
                    .and_then(|m| m.blocks.first())
                    .and_then(|b| {
                        if let crate::types::ContentBlock::Text { text } = b {
                            Some(text.as_str())
                        } else {
                            None
                        }
                    })
                    .unwrap_or("");
                let effort = crate::auto_reasoning::select(false, last_msg);
                Some(effort.label().to_string())
            } else {
                base
            }
        };
        ChatRequest {
            model: self.config.model.clone(),
            system,
            messages,
            tools: self.tools.schemas_json(),
            max_tokens: self.config.max_tokens,
            temperature: self.config.temperature,
            reasoning,
        }
    }

    /// Record a tool result in history with context-aware truncation and
    /// loop-guard tracking.  Centralises the pattern that was duplicated
    /// across readonly-batch and gated-tool execution paths.
    fn push_tool_outcome(
        history: &mut crate::history::ConversationHistory,
        guard: &mut crate::loop_guard::LoopGuard,
        id: &str,
        name: &str,
        output: &str,
        is_error: bool,
        context_window: u64,
    ) {
        let compacted =
            crate::tool::large_output::route_large_output_aware(output, name, context_window);
        history.push_tool_result(id, &compacted, is_error);
        let ok = !is_error;
        if let crate::loop_guard::OutcomeDecision::Halt(msg) = guard.record_outcome(name, ok) {
            tracing::warn!("loop_guard halt ({name}): {msg}");
            history.push_tool_result(&format!("guard_{id}"), &msg, true);
        }
        for (mime, b64) in crate::tool::image_result::drain_images() {
            history.push_image(&mime, &b64);
        }
    }

    async fn drain_steers(
        steer_rx: &mut Option<mpsc::Receiver<SteerMessage>>,
        pending: &mut Vec<SteerMessage>,
        delivered: &mut std::collections::HashSet<i32>,
        history: &mut ConversationHistory,
        tx: &mpsc::Sender<AgentEvent>,
    ) {
        let rx = match steer_rx.as_mut() {
            Some(rx) => rx,
            None => return,
        };

        // Collect new messages from the channel.
        while let Ok(msg) = rx.try_recv() {
            if msg.is_edit {
                // Try to replace in pending queue.
                if let Some(existing) = pending.iter_mut().find(|m| m.msg_id == msg.msg_id) {
                    existing.text = msg.text;
                    continue;
                }
                // Already delivered to LLM — send as correction.
                if delivered.contains(&msg.msg_id) {
                    pending.push(SteerMessage {
                        msg_id: msg.msg_id,
                        text: format!("[correction] {}", msg.text),
                        is_edit: false,
                    });
                    continue;
                }
            }
            pending.push(msg);
        }

        if pending.is_empty() {
            return;
        }

        // Merge all pending into ONE user message.
        let combined: String = pending
            .iter()
            .map(|m| m.text.as_str())
            .collect::<Vec<_>>()
            .join("\n\n");

        // Track delivered msg_ids.
        for m in pending.iter() {
            delivered.insert(m.msg_id);
        }
        pending.clear();

        history.push_user(&combined);
        let _ = tx.send(AgentEvent::SteerReceived { text: combined }).await;
    }

    /// Execute a single tool call with heartbeat, cancellation, and steer handling.
    ///
    /// Encapsulates the heartbeat-spawn + tokio::select! + cleanup pattern
    /// that was previously inlined in `run()` at 10 levels of nesting.
    #[allow(clippy::too_many_arguments)]
    async fn execute_tool_with_heartbeat(
        &self,
        name: &str,
        input: serde_json::Value,
        call_id: &str,
        cancel: &CancellationToken,
        tx: &mpsc::Sender<AgentEvent>,
        steer_rx: &mut Option<mpsc::Receiver<SteerMessage>>,
        pending_steers: &mut Vec<SteerMessage>,
    ) -> Result<crate::types::ToolResult> {
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

        let progress_tx = tx.clone();
        let tool_fut = self
            .tools
            .execute_with_progress(name, input, &self.config.cwd, progress_tx);

        let result = {
            tokio::pin!(tool_fut);
            loop {
                tokio::select! {
                    _ = cancel.cancelled() => {
                        heartbeat_cancel.cancel();
                        let _ = hb_handle.await;
                        return Err(AgentError::Cancelled);
                    }
                    r = &mut tool_fut => break r,
                    msg = async {
                        match steer_rx.as_mut() {
                            Some(rx) => rx.recv().await,
                            None => std::future::pending().await,
                        }
                    } => {
                        if let Some(msg) = msg {
                            let _ = tx.send(AgentEvent::ToolOutput {
                                call_id: call_id.to_string(),
                                chunk: format!("\u{21a9}\u{fe0f} Steer queued: {}", &msg.text[..msg.text.len().min(60)]),
                            }).await;
                            pending_steers.push(msg);
                        }
                    }
                }
            }
        };

        heartbeat_cancel.cancel();
        let _ = hb_handle.await;
        Ok(result)
    }
}

#[cfg(test)]
#[path = "loop__tests.rs"]
mod tests;
