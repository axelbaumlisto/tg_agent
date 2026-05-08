//! Tool-dispatch and request-building helpers extracted from `AgentLoop::run`.

use crate::error::{AgentError, Result};
use crate::history::ConversationHistory;
use crate::provider::ChatRequest;
use crate::tool::registry::ToolRegistry;
use crate::types::{AgentEvent, Permission, SteerMessage};
use tokio::sync::mpsc;
use tokio_util::sync::CancellationToken;

use super::HEARTBEAT_INTERVAL;

impl super::AgentLoop {
    pub(super) async fn execute_readonly_batch(
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
    pub(super) fn classify_tool_calls(
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

    pub(super) fn build_chat_request(&self, history: &ConversationHistory) -> ChatRequest {
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
    /// loop-guard tracking. Centralises the pattern that was duplicated
    /// across readonly-batch and gated-tool execution paths.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn push_tool_outcome(
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

    /// Execute a single tool call with a heartbeat keepalive.
    #[allow(clippy::too_many_arguments)]
    pub(super) async fn execute_tool_with_heartbeat(
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
