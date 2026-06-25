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
use super::steers::SteerPipeline;
use super::stream_turn::TurnStreamOutcome;

/// R1 of PLAN_NEXT_SESSION: every drain-on-error site bumps this
/// counter via `bump_drained_on_abort_if_rescued`. Lifted to a free
/// helper so the three Err-paths in `run()` keep their bodies
/// compact (and never accidentally fall out of sync about whether a
/// drain that produced output should be observable as a rescue).
#[inline]
fn bump_drained_on_abort_if_rescued(rescued: bool) {
    if rescued {
        crate::types::STEER_DRAINED_ON_ABORT_COUNT
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    }
}

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
        // R1: SteerPipeline owns pending vec + delivered set.
        let mut steer = SteerPipeline::new();
        let started = std::time::Instant::now();
        let mut first_token_at: Option<std::time::Instant> = None;

        let result: Result<TurnUsage> = async {
            'outer: for _iteration in 0..limit {
            if cancel.is_cancelled() {
                // PLAN_NEXT_SESSION §A.2 S6 follow-up: preserve any
                // in-flight user input across an abort. Without this
                // the next turn starts with an incomplete view (e.g.
                // the user typed a steer 50ms before /abort — it
                // would otherwise be silently dropped with the
                // dying channel).
                let rescued = steer.drain(&mut steer_rx, history, &tx).await;
                bump_drained_on_abort_if_rescued(rescued);
                return Err(AgentError::Cancelled);
            }

            // Per-iteration guard — blocks identical repeated calls and
            // halts after too many consecutive failures.
            let mut loop_guard = crate::loop_guard::LoopGuard::default();

            // Drain steer messages between iterations.
            steer.drain(&mut steer_rx, history, &tx).await;

            if let Some(max_wall) = self.config.max_wall
                && started.elapsed() >= max_wall
            {
                // B73: cooperative turn-level wall-clock budget. This check is
                // intentionally at the iteration boundary (not a tokio::timeout
                // wrapper) so streams/tools keep their own cancellation behavior
                // and the B70 cleanup contract is preserved.
                let msg = crate::error::WALL_TIMEOUT_MESSAGE.to_string();
                let _ = tx.send(AgentEvent::Error(msg.clone())).await;
                let _ = tx.send(AgentEvent::Idle).await;
                self.config.record_health(
                    crate::model_catalog::HealthEventKind::Error,
                    None,
                    Some("turn wall timeout".into()),
                );
                let rescued = steer.drain(&mut steer_rx, history, &tx).await;
                bump_drained_on_abort_if_rescued(rescued);
                return Err(AgentError::WallTimeout);
            }

            // Checkpoint-restart cycle: if token usage exceeds threshold,
            // archive old messages and restart with fresh context.
            self.advance_cycle_if_needed(history, &tx).await?;

            let request = self.build_chat_request(history);

            let outcome = match self
                .stream_one_turn(
                    &request,
                    history,
                    &mut cumulative_usage,
                    &cancel,
                    &tx,
                    &mut steer_rx,
                    &mut steer,
                    started,
                    &mut first_token_at,
                )
                .await
            {
                Ok(o) => o,
                Err(AgentError::ProviderTyped(
                    crate::provider::error::ProviderError::ContextWindowExceeded { .. },
                )) => continue 'outer,
                Err(AgentError::WallTimeout) => {
                    let msg = crate::error::WALL_TIMEOUT_MESSAGE.to_string();
                    let _ = tx.send(AgentEvent::Error(msg)).await;
                    let _ = tx.send(AgentEvent::Idle).await;
                    self.config.record_health(
                        crate::model_catalog::HealthEventKind::Error,
                        None,
                        Some("turn provider/stream backstop timeout".into()),
                    );
                    let rescued = steer.drain(&mut steer_rx, history, &tx).await;
                    bump_drained_on_abort_if_rescued(rescued);
                    return Err(AgentError::WallTimeout);
                }
                Err(e) => {
                    // Same context-preservation guarantee as the
                    // pre-iteration cancel check above. Cheap (drain
                    // is a non-blocking try_recv loop) and applies
                    // uniformly to provider errors so the user
                    // doesn't lose typed input on a transient failure.
                    let rescued = steer.drain(&mut steer_rx, history, &tx).await;
                    bump_drained_on_abort_if_rescued(rescued);
                    return Err(e);
                }
            };

            // S2/S3 of PLAN_NEXT_SESSION: stream was interrupted by a
            // mid-stream steer. Drain steers (the one in the
            // pipeline plus any that may have piled up since), do NOT push the
            // partial assistant message to history (the user already
            // saw it via TextDelta), and re-issue the iteration so the
            // model sees the augmented context.
            if outcome.mid_stream_steer {
                empty_budget.reset();
                steer.drain(&mut steer_rx, history, &tx).await;
                continue 'outer;
            }
            // Guard against "provider returned nothing" turns.
            //
            // Some providers (notably `glm-cn`/`glm-5-turbo`) can close a
            // stream cleanly with zero output tokens, no tool calls, and no
            // reasoning chunks. If we saw the protocol terminal marker, this
            // is genuine model silence and retrying the same context will not
            // help. Only defensive empty streams without Done use the retry
            // budget; B65 turns abnormal SSE EOF into `StreamChunk::Error`,
            // but keep this fallback for non-SSE/legacy providers.
            // The terminal path below must keep Error + Idle + steer drain.
            if outcome.empty {
                if !outcome.saw_done
                    && let Some(delay) = empty_budget.next_delay()
                {
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
                let rescued = steer.drain(&mut steer_rx, history, &tx).await;
                bump_drained_on_abort_if_rescued(rescued);
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
                // S1 of PLAN_NEXT_SESSION: before signalling Idle and
                // returning, drain any steer messages that arrived while
                // the model was streaming. If a steer is pending, the
                // user's clarification would otherwise be silently lost
                // (steer_rx is dropped when run() returns) — the exact
                // failure mode reproduced in incident img_20260510_f1d4.
                if steer.drain(&mut steer_rx, history, &tx).await {
                    // The drain just appended a fresh user message;
                    // continue the iteration loop instead of returning
                    // Idle so the model gets a chance to react.
                    self.config.observer.on_retry(
                        RetryKind::EmptyContent,
                        0,
                        0,
                        0,
                        "steer-drained-on-idle-exit",
                    );
                    continue 'outer;
                }
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
                    &mut steer,
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
                steer.drain(&mut steer_rx, history, &tx).await;

                // T6 of PLAN_QUALITY_v1: PreToolUse hook. If a hook
                // is configured to abort on this (tool, input) pair,
                // skip the call and emit a synthetic ToolEnd with
                // an error so the model sees what happened.
                if let Some(hooks) = &self.config.lifecycle_hooks {
                    let key = format!("{name}:{input}");
                    let path_var = input.get("path").and_then(|v| v.as_str()).unwrap_or("");
                    let outcome = hooks
                        .run(
                            crate::lifecycle_hooks::HookEvent::PreToolUse,
                            &key,
                            &[("tool", &name), ("path", path_var)],
                        )
                        .await;
                    if outcome == crate::lifecycle_hooks::HookOutcome::FailedAbort {
                        let _ = tx
                            .send(AgentEvent::ToolEnd {
                                call_id: id.clone(),
                                name: name.clone(),
                                state: ToolState::Error,
                                output: "PreToolUse hook aborted this call".into(),
                            })
                            .await;
                        history.push_tool_result(&id, "PreToolUse hook aborted this call", true);
                        continue;
                    }
                }

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
                    self.config.permissions.as_ref(),
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

                let result = match self
                    .execute_tool_with_heartbeat(
                        &name,
                        input.clone(),
                        &id,
                        &cancel,
                        &tx,
                        &mut steer_rx,
                        &mut steer,
                    )
                    .await
                {
                    Ok(result) => result,
                    Err(AgentError::WallTimeout) => {
                        let msg = crate::error::WALL_TIMEOUT_MESSAGE.to_string();
                        let _ = tx.send(AgentEvent::Error(msg)).await;
                        let _ = tx.send(AgentEvent::Idle).await;
                        self.config.record_health(
                            crate::model_catalog::HealthEventKind::Error,
                            None,
                            Some("tool wall timeout".into()),
                        );
                        let rescued = steer.drain(&mut steer_rx, history, &tx).await;
                        bump_drained_on_abort_if_rescued(rescued);
                        return Err(AgentError::WallTimeout);
                    }
                    Err(e) => return Err(e),
                };

                // T2 of PLAN_QUALITY_v1: post-edit LSP hook. Compute
                // the edited paths and ask the LSP manager for
                // diagnostics. The rendered block (if any) is
                // pushed to history as a synthetic system message
                // so the model sees compile errors before its next
                // reasoning step. No-op when lsp == None or the
                // tool isn't an edit.
                if !result.is_error
                    && let Some(mgr) = self.config.lsp.as_ref()
                {
                    let paths = super::lsp_hooks::edited_paths_for_tool(&name, &input);
                    for p in &paths {
                        let diags = mgr.diagnostics_for(&self.config.cwd, p).await;
                        if !diags.is_empty() {
                            // R5 of PLAN_RESILIENCE_v1: bump counter
                            // + info log so operators see LSP fire
                            // in journalctl, not just at debug.
                            crate::types::LSP_DIAGNOSTIC_EMITTED_COUNT
                                .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                            tracing::info!(
                                file = %p.display(),
                                count = diags.len(),
                                "lsp: emitted post-edit diagnostics"
                            );
                            let body = crate::lsp::render_for_model(p, &diags);
                            history.push_user(&body);
                        }
                    }
                }

                // T6 of PLAN_QUALITY_v1: PostToolUse hook. Fired
                // for every gated-tool exec regardless of
                // success/error so operator hooks (e.g. cargo fmt
                // after .rs writes) always run.
                if let Some(hooks) = &self.config.lifecycle_hooks {
                    let key = format!("{name}:{input}");
                    let path_var = input.get("path").and_then(|v| v.as_str()).unwrap_or("");
                    let _ = hooks
                        .run(
                            crate::lifecycle_hooks::HookEvent::PostToolUse,
                            &key,
                            &[("tool", &name), ("path", path_var)],
                        )
                        .await;
                }

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
        .await;

        crate::metrics_hist::record_turn_duration(started.elapsed().as_millis() as u64);
        if let Some(first) = first_token_at {
            crate::metrics_hist::record_ttft(first.duration_since(started).as_millis() as u64);
        }
        result
    }
}
