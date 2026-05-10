//! Single provider-turn streaming extracted from `AgentLoop::run`.

use crate::error::{AgentError, Result};
use crate::history::ConversationHistory;
use crate::loop_observer::RetryKind;
use crate::provider::ChatRequest;
use crate::retry::Backoff;
use crate::types::{AgentEvent, ContentBlock, SteerMessage, StreamChunk, TurnUsage};
use tokio::sync::mpsc;
use tokio_stream::StreamExt;
use tokio_util::sync::CancellationToken;

use super::{BASE_RETRY_DELAY_MS, MAX_STREAM_RETRIES};

/// Outcome of a single provider turn returned by [`AgentLoop::stream_one_turn`].
/// The outer loop in [`AgentLoop::run`] decides what to do with empty turns
/// and dispatches tool calls.
pub(super) struct TurnStreamOutcome {
    pub(super) blocks: Vec<ContentBlock>,
    pub(super) tool_calls: Vec<(String, String, serde_json::Value)>,
    pub(super) turn_usage: Option<TurnUsage>,
    /// True when the stream produced no content at all (no text, no tools, no thinking).
    /// The outer loop decides whether to retry via the empty-content-attempts budget.
    pub(super) empty: bool,
    /// S2/S3 of PLAN_NEXT_SESSION: true when the stream was broken
    /// out of mid-flight because a steer message arrived. The outer
    /// loop must drain steers and re-issue the iteration WITHOUT
    /// pushing the partial assistant message to history (the user
    /// already saw the partial via TextDelta events).
    pub(super) mid_stream_steer: bool,
}

impl super::AgentLoop {
    #[allow(clippy::too_many_arguments)]
    pub(super) async fn stream_one_turn(
        &self,
        request: &ChatRequest,
        history: &mut ConversationHistory,
        cumulative_usage: &mut TurnUsage,
        cancel: &CancellationToken,
        tx: &mpsc::Sender<AgentEvent>,
        steer_rx: &mut Option<mpsc::Receiver<SteerMessage>>,
        pending_steers: &mut Vec<SteerMessage>,
    ) -> Result<TurnStreamOutcome> {
        let mut text_acc = String::new();
        let mut thinking_acc = String::new();
        let mut in_fake_tool = false;
        let mut blocks: Vec<ContentBlock> = Vec::new();
        let mut tool_calls: Vec<(String, String, serde_json::Value)> = Vec::new();
        let mut turn_usage: Option<TurnUsage> = None;
        let mut stream_ok = false;
        // Hoisted out of the retry loop so the post-loop tail can
        // surface it on the TurnStreamOutcome.
        let mut mid_stream_steer_flag = false;

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
                    if matches!(
                        &e,
                        AgentError::ProviderTyped(
                            crate::provider::error::ProviderError::ContextWindowExceeded { .. }
                        )
                    ) {
                        let before = history.message_count();
                        if before <= 3 {
                            return Err(e);
                        }
                        for round in 0..5 {
                            let keep = (history.message_count() / 3).clamp(2, 6);
                            self.config.observer.on_compact_round(
                                round,
                                keep,
                                history.message_count(),
                                history.estimated_tokens(),
                            );
                            history.compact(keep);
                            if history.estimated_tokens()
                                < history.context_window_tokens() as usize * 4 / 5
                            {
                                break;
                            }
                        }
                        let after = history.message_count();
                        self.config.observer.on_compact_done(before, after);
                        let _ = tx
                            .send(AgentEvent::ContextCompacted {
                                before_msgs: before,
                                after_msgs: after,
                                summary_hint: None,
                                files_count: 0,
                            })
                            .await;
                        return Err(e);
                    }
                    if retry < MAX_STREAM_RETRIES {
                        let delay = Backoff {
                            base_ms: BASE_RETRY_DELAY_MS,
                            max_attempts: MAX_STREAM_RETRIES + 1,
                        }
                        .delay(retry);
                        self.config.observer.on_retry(
                            RetryKind::Connect,
                            retry + 1,
                            MAX_STREAM_RETRIES,
                            delay.as_millis() as u64,
                            &e.to_string(),
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
                // S2 of PLAN_NEXT_SESSION: 3-arm select! — cancel,
                // next chunk, OR a steer message landing while we
                // stream. The steer arm makes the polling task
                // observable to user input within ~one tokio yield,
                // matching pi-coding-agent's REPL feel and removing
                // the "Принято — доставлю между шагами" hang.
                let chunk = tokio::select! {
                    _ = cancel.cancelled() => {
                        return Err(AgentError::Cancelled);
                    }
                    next = stream.next() => match next {
                        Some(c) => c,
                        None => break,
                    },
                    maybe_msg = async {
                        match steer_rx.as_mut() {
                            Some(rx) => rx.recv().await,
                            // No steer channel: park forever so this
                            // arm never wins.
                            None => std::future::pending().await,
                        }
                    } => {
                        if let Some(msg) = maybe_msg {
                            // S3 — soft-interrupt policy: stash the
                            // winning steer in pending_steers, then
                            // BURST-drain the channel so any messages
                            // that arrived in the same scheduler tick
                            // are merged into one re-issue. Without
                            // this, two close-spaced steers would
                            // each trigger their own iteration with a
                            // single user message — not the merge
                            // semantics drain_steers expects.
                            pending_steers.push(msg);
                            if let Some(rx) = steer_rx.as_mut() {
                                while let Ok(more) = rx.try_recv() {
                                    pending_steers.push(more);
                                }
                            }
                            mid_stream_steer_flag = true;
                            break;
                        }
                        // Channel closed (sender dropped) — fall
                        // through and keep streaming.
                        continue;
                    }
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

            // S3: when steer interrupted, treat the iteration as
            // "complete enough" — no retry, no error — and let the
            // outer loop re-issue.
            if mid_stream_steer_flag {
                stream_ok = true;
                break;
            }

            if let Some(e) = mid_stream_error {
                if tool_calls.is_empty() && retry < MAX_STREAM_RETRIES {
                    let delay = Backoff {
                        base_ms: BASE_RETRY_DELAY_MS,
                        max_attempts: MAX_STREAM_RETRIES + 1,
                    }
                    .delay(retry);
                    self.config.observer.on_retry(
                        RetryKind::MidStream,
                        retry + 1,
                        MAX_STREAM_RETRIES,
                        delay.as_millis() as u64,
                        &e,
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

        let empty = blocks.is_empty() && tool_calls.is_empty();
        Ok(TurnStreamOutcome {
            blocks,
            tool_calls,
            turn_usage,
            empty,
            mid_stream_steer: mid_stream_steer_flag,
        })
    }
}
