//! Public top-level `AgentLoop::run` driver.

use crate::error::{AgentError, Result};
use crate::history::ConversationHistory;
use crate::loop_observer::RetryKind;
use crate::types::{
    AgentEvent, Permission, PermissionResponse, SteerMessage, ToolState, TurnUsage,
};
use tokio::sync::mpsc;
use tokio_util::sync::CancellationToken;

use super::MAX_EMPTY_CONTENT_RETRIES;
use super::budget::EmptyContentBudget;
use super::stream_turn::TurnStreamOutcome;

impl super::AgentLoop {
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
        // Tracks consecutive empty-content responses across iterations.
        // Reset whenever an iteration produces real content so a normal
        // tool-use loop never accidentally exhausts the budget; only bursts
        // of empty responses (the actual `glm-5-turbo` failure mode) are
        // capped.
        let mut empty_budget = EmptyContentBudget::new();
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
            self.advance_cycle_if_needed(history, &tx).await?;

            let request = self.build_chat_request(history);

            let outcome = match self
                .stream_one_turn(&request, history, &mut cumulative_usage, &cancel, &tx)
                .await
            {
                Ok(o) => o,
                Err(AgentError::ProviderTyped(
                    crate::provider::error::ProviderError::ContextWindowExceeded { .. },
                )) => continue 'outer,
                Err(e) => return Err(e),
            };
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
            if outcome.empty {
                if let Some(delay) = empty_budget.next_delay() {
                    crate::types::EMPTY_CONTENT_RETRY_COUNT
                        .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                    self.config.observer.on_retry(
                        RetryKind::EmptyContent,
                        empty_budget.count(),
                        MAX_EMPTY_CONTENT_RETRIES,
                        delay.as_millis() as u64,
                        "",
                    );
                    tokio::time::sleep(delay).await;
                    continue 'outer;
                }
                let msg = format!(
                    "provider returned no content {} times in a row (no text, no reasoning, no tool calls)",
                    empty_budget.count() + 1,
                );
                self.config.observer.on_giveup(&msg);
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
            empty_budget.reset();
            let TurnStreamOutcome {
                blocks,
                tool_calls,
                turn_usage,
                ..
            } = outcome;
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

                let fp = crate::tool::approval_cache::fingerprint(&name, &input);
                let allowed = super::permission::request_or_cached_approval(
                    &self.approval_cache,
                    &fp,
                    &mut permission_rx,
                    &id,
                    &name,
                    &input,
                    perm,
                    &tx,
                )
                .await;

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
}
