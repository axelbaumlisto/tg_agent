use std::collections::HashMap;
use std::sync::Arc;
use std::sync::atomic::AtomicBool;

use naked_core::AgentCore;
use naked_core::config::Config;
use naked_core::research::StopReason;
use naked_core::types::{AgentEvent, AgentHandle, PermissionResponse};
use naked_tg::channel_map::ChannelSessionMap;
use naked_tg::synthetic::SyntheticDispatchOutcome;
use teloxide::prelude::*;
use tokio::sync::RwLock;

use crate::shared::{self, ChatCtx, research_scheduler};

/// Additional deps needed by `stream_response()` that are available at
/// `wiring::build()` time but not part of the scheduler's own config.
#[derive(Clone)]
pub(crate) struct StreamDeps {
    pub bot: Bot,
    pub config: Config,
    pub http_client: Arc<reqwest::Client>,
    pub base_url: Arc<String>,
    pub tg_attach_queue: naked_tg::tg_attach::AttachmentQueue,
    pub bot_token: Arc<String>,
    pub bot_identity: Arc<naked_tg::bot_identity::BotIdentity>,
}

pub(crate) fn start_research_scheduler(
    agent: &Arc<AgentCore>,
    config: &Config,
    channel_map: &Arc<ChannelSessionMap>,
    liveness: &Arc<naked_core::liveness::LivenessRegistry>,
    scheduler_lock_held: bool,
    stream_deps: Option<StreamDeps>,
) -> Option<Arc<research_scheduler::ResearchScheduler>> {
    // B64 fix (PLAN_SYNTHETIC_DISPATCH_FIX_v1 + SESSION_FIX_v1):
    // Dispatch closure creates a fresh research session, submits the prompt,
    // and streams the result to the operator's chat via `stream_response()`.
    // The operator sees a live bubble with progress + Abort button.
    // Uses the shared global `RATE_LIMITER` to prevent Telegram 429 errors.
    if !config.research.enabled || !scheduler_lock_held {
        return None;
    }

    if !config.research.fallback_models.is_empty() {
        tracing::warn!(
            fallback_models = config.research.fallback_models.len(),
            "B72: research.fallback_models is consumed only by the legacy ResearchCoordinator; scheduled/synthetic research uses the provider-level fallback chain",
        );
    }

    let dispatch_fn = synthetic_dispatch_fn(agent, channel_map, stream_deps);
    let scheduler_cfg = research_scheduler::SchedulerConfig {
        verify_by_default: config.research.verify_by_default,
        max_verification_rounds: config.research.gatekeeper.max_rounds,
        max_concurrent_runs: config.research.max_concurrent_runs.max(1),
        task_timeout: std::time::Duration::from_secs(config.research.task_timeout_seconds),
        max_retries_before_alert: config.research.max_retries_before_alert,
        liveness: Some(liveness.clone()),
        dispatch_fn: Some(dispatch_fn),
        ..Default::default()
    };
    let (sched_arc, hook) =
        research_scheduler::ResearchScheduler::start(Arc::downgrade(agent), scheduler_cfg);
    agent.set_scheduler_hook(hook);
    tracing::info!(
        "research scheduler online (synthetic dispatch wired; T2.6 PLAN_RESEARCH_AGENT_FLOW_v1)"
    );
    Some(sched_arc)
}

fn synthetic_dispatch_fn(
    agent: &Arc<AgentCore>,
    channel_map: &Arc<ChannelSessionMap>,
    stream_deps: Option<StreamDeps>,
) -> naked_tg::synthetic::SyntheticDispatchFn {
    let agent = agent.clone();
    let channel_map = channel_map.clone();
    Arc::new(
        move |msg: naked_tg::synthetic::SyntheticMessage,
              sid_latch: naked_tg::synthetic::SessionIdLatch| {
            let agent = agent.clone();
            let channel_map = channel_map.clone();
            let stream_deps = stream_deps.clone();
            Box::pin(async move {
                let spec_id_owned = msg.source.spec_id().map(str::to_string).unwrap_or_default();
                match naked_tg::synthetic::dispatch_for_chat(
                    &agent,
                    &channel_map,
                    msg.chat_id,
                    msg.thread_id,
                    &spec_id_owned,
                )
                .await
                {
                    Ok((sid, handle)) => {
                        // Publish session-id immediately so the scheduler's
                        // cancel branch can abort the turn via agent.abort(sid)
                        // while stream_response() is still blocking.
                        *sid_latch.lock().await = Some(sid.clone());

                        tracing::info!(
                            session_id = %sid,
                            spec_id = %spec_id_owned,
                            "synthetic dispatch: turn submitted"
                        );

                        let outcome = if let Some(sd) = stream_deps {
                            // Full streaming path: bubble + live edits + Abort.
                            stream_research_turn(
                                &agent,
                                &channel_map,
                                &sd,
                                &sid,
                                &spec_id_owned,
                                &msg,
                                handle,
                            )
                            .await
                        } else {
                            // Headless fallback (tests, no-bot contexts).
                            drain_synthetic_agent_handle(&sid, &spec_id_owned, handle).await
                        };

                        tracing::info!(
                            session_id = %sid,
                            spec_id = %spec_id_owned,
                            stop = ?outcome.stop_reason,
                            errors = outcome.errors.len(),
                            "synthetic dispatch: turn completed"
                        );
                        Ok(outcome)
                    }
                    Err(e) => {
                        tracing::warn!(
                            spec_id = %spec_id_owned,
                            error = %e,
                            "synthetic dispatch failed"
                        );
                        Err(e)
                    }
                }
            })
        },
    )
}

/// Stream a research turn to the operator's chat via `stream_response()`.
///
/// Constructs a minimal `BotDeps` from the captured `StreamDeps` fields.
/// Uses the global `RATE_LIMITER` (shared with normal chat streams) to
/// prevent Telegram 429 errors from concurrent message edits.
async fn stream_research_turn(
    agent: &Arc<AgentCore>,
    channel_map: &Arc<ChannelSessionMap>,
    sd: &StreamDeps,
    session_id: &str,
    spec_id: &str,
    msg: &naked_tg::synthetic::SyntheticMessage,
    handle: AgentHandle,
) -> SyntheticDispatchOutcome {
    use crate::message_handler::BotDeps;
    use crate::per_chat_locks::PerChatLocks;

    let research_deps = BotDeps {
        bot: sd.bot.clone(),
        agent: agent.clone(),
        channel_map: channel_map.clone(),
        config: sd.config.clone(),
        pending_perms: Arc::new(RwLock::new(HashMap::new())),
        http_client: sd.http_client.clone(),
        base_url: sd.base_url.clone(),
        // CRITICAL: use the GLOBAL rate limiter, not a new one.
        // RATE_LIMITER is Arc<Mutex<State>> — Clone shares the inner
        // state. Research bubble edits and normal chat edits share
        // the same per-chat rate bucket, preventing Telegram 429.
        rate_limiter: shared::RATE_LIMITER.clone(),
        attribution_flag: Arc::new(AtomicBool::new(false)),
        bot_token: sd.bot_token.clone(),
        bot_identity: sd.bot_identity.clone(),
        tg_attach_queue: sd.tg_attach_queue.clone(),
        research_scheduler: None, // not needed by stream_response
        per_chat_locks: Arc::new(PerChatLocks::new()),
    };

    let ctx = ChatCtx {
        chat_id: ChatId(msg.chat_id),
        thread_id: msg
            .thread_id
            .map(|t| teloxide::types::ThreadId(teloxide::types::MessageId(t))),
        reply_to: None,
    };

    // Determine model tag for the streaming UI header.
    let model_tag = agent
        .config()
        .research
        .model
        .clone()
        .unwrap_or_else(|| "research".into());

    tracing::info!(
        session_id = %session_id,
        spec_id = %spec_id,
        "streaming research turn to chat bubble"
    );

    let run_ctx = crate::streaming::StreamRunContext {
        requested_run_id: msg.run_id.clone(),
        session_id: session_id.to_string(),
        kind: naked_tg::run_registry::RunKind::Research {
            spec_id: spec_id.to_string(),
        },
        source_ref: Some(spec_id.to_string()),
    };
    let outcome =
        crate::streaming::stream_response(&research_deps, ctx, handle, model_tag, run_ctx).await;

    SyntheticDispatchOutcome {
        session_id: session_id.to_string(),
        stop_reason: if outcome.wall_timeout {
            StopReason::Timeout
        } else if outcome.got_idle {
            StopReason::AgentIdle
        } else {
            StopReason::StreamClosed
        },
        errors: vec![], // errors rendered in bubble by stream_response
    }
}

/// B64: Headless drain for a synthetic agent turn. Kept as fallback for
/// tests and no-bot contexts. `stream_research_turn()` is preferred
/// when `StreamDeps` is available.
///
/// One-consumer invariant: only this helper OR `stream_response` should
/// drain a given `AgentHandle.events` — never both.
pub(crate) async fn drain_synthetic_agent_handle(
    session_id: &str,
    spec_id: &str,
    handle: AgentHandle,
) -> SyntheticDispatchOutcome {
    let AgentHandle {
        mut events,
        permissions,
        steer: _,
        abort: _,
    } = handle;
    let mut errors: Vec<String> = Vec::new();
    let mut wall_timeout = false;

    loop {
        match events.recv().await {
            Some(AgentEvent::Idle) => {
                return SyntheticDispatchOutcome {
                    session_id: session_id.to_string(),
                    stop_reason: if wall_timeout {
                        StopReason::Timeout
                    } else {
                        StopReason::AgentIdle
                    },
                    errors,
                };
            }
            Some(AgentEvent::PermissionRequest { call_id, .. }) => {
                tracing::debug!(
                    session_id = %session_id,
                    spec_id = %spec_id,
                    call_id = %call_id,
                    "synthetic drain: auto-approving permission request"
                );
                let _ = permissions
                    .send(PermissionResponse {
                        call_id,
                        allowed: true,
                    })
                    .await;
            }
            Some(AgentEvent::Error(msg)) => {
                if msg == naked_core::error::WALL_TIMEOUT_MESSAGE {
                    wall_timeout = true;
                }
                tracing::warn!(
                    session_id = %session_id,
                    spec_id = %spec_id,
                    error = %msg,
                    "synthetic drain: agent error event"
                );
                errors.push(msg);
            }
            Some(_) => {}
            None => {
                tracing::warn!(
                    session_id = %session_id,
                    spec_id = %spec_id,
                    "synthetic drain: event channel closed before Idle"
                );
                return SyntheticDispatchOutcome {
                    session_id: session_id.to_string(),
                    stop_reason: if wall_timeout {
                        StopReason::Timeout
                    } else {
                        StopReason::StreamClosed
                    },
                    errors,
                };
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tokio::sync::mpsc;

    fn fake_handle(
        buf: usize,
    ) -> (
        mpsc::Sender<AgentEvent>,
        mpsc::Receiver<PermissionResponse>,
        AgentHandle,
    ) {
        let (ev_tx, ev_rx) = mpsc::channel(buf);
        let (perm_tx, perm_rx) = mpsc::channel(buf);
        let (steer_tx, _steer_rx) = mpsc::channel(buf);
        let handle = AgentHandle {
            events: ev_rx,
            permissions: perm_tx,
            steer: steer_tx,
            abort: tokio_util::sync::CancellationToken::new(),
        };
        (ev_tx, perm_rx, handle)
    }

    #[test]
    fn scheduler_gate_requires_enabled_config_and_lock() {
        let source = include_str!("research_scheduler.rs");
        assert!(source.contains("!config.research.enabled || !scheduler_lock_held"));
        assert!(source.contains("dispatch_fn: Some(dispatch_fn)"));
    }

    #[tokio::test]
    async fn drain_waits_for_idle() {
        let (ev_tx, _perm_rx, handle) = fake_handle(8);
        ev_tx
            .send(AgentEvent::TextDelta("hello".into()))
            .await
            .unwrap();
        ev_tx.send(AgentEvent::Heartbeat).await.unwrap();
        ev_tx.send(AgentEvent::Idle).await.unwrap();

        let outcome = drain_synthetic_agent_handle("sid", "spec", handle).await;
        assert_eq!(outcome.stop_reason, StopReason::AgentIdle);
        assert!(outcome.errors.is_empty());
        assert_eq!(outcome.session_id, "sid");
    }

    #[tokio::test]
    async fn drain_auto_approves_permission() {
        let (ev_tx, mut perm_rx, handle) = fake_handle(8);
        ev_tx
            .send(AgentEvent::PermissionRequest {
                call_id: "c1".into(),
                tool_name: "bash".into(),
                input: serde_json::json!({}),
                permission: naked_core::types::Permission::ReadOnly,
            })
            .await
            .unwrap();
        ev_tx.send(AgentEvent::Idle).await.unwrap();

        let outcome = drain_synthetic_agent_handle("sid", "spec", handle).await;
        assert_eq!(outcome.stop_reason, StopReason::AgentIdle);

        let resp = perm_rx
            .try_recv()
            .expect("should have received PermissionResponse");
        assert_eq!(resp.call_id, "c1");
        assert!(resp.allowed);
    }

    #[tokio::test]
    async fn drain_wall_timeout_error_maps_to_timeout_stop_reason() {
        let (ev_tx, _perm_rx, handle) = fake_handle(8);
        ev_tx
            .send(AgentEvent::Error(
                naked_core::error::WALL_TIMEOUT_MESSAGE.into(),
            ))
            .await
            .unwrap();
        ev_tx.send(AgentEvent::Idle).await.unwrap();

        let outcome = drain_synthetic_agent_handle("sid", "spec", handle).await;
        assert_eq!(outcome.stop_reason, StopReason::Timeout);
        assert_eq!(
            outcome.errors,
            vec![naked_core::error::WALL_TIMEOUT_MESSAGE]
        );
    }

    #[tokio::test]
    async fn drain_collects_errors() {
        let (ev_tx, _perm_rx, handle) = fake_handle(8);
        ev_tx
            .send(AgentEvent::Error("provider died".into()))
            .await
            .unwrap();
        ev_tx
            .send(AgentEvent::Error("retry failed".into()))
            .await
            .unwrap();
        ev_tx.send(AgentEvent::Idle).await.unwrap();

        let outcome = drain_synthetic_agent_handle("sid", "spec", handle).await;
        assert_eq!(outcome.stop_reason, StopReason::AgentIdle);
        assert_eq!(outcome.errors.len(), 2);
        assert_eq!(outcome.errors[0], "provider died");
        assert_eq!(outcome.errors[1], "retry failed");
    }

    #[tokio::test]
    async fn drain_channel_close_before_idle() {
        let (ev_tx, _perm_rx, handle) = fake_handle(8);
        ev_tx
            .send(AgentEvent::TextDelta("partial".into()))
            .await
            .unwrap();
        drop(ev_tx);

        let outcome = drain_synthetic_agent_handle("sid", "spec", handle).await;
        assert_ne!(outcome.stop_reason, StopReason::AgentIdle);
        assert_eq!(outcome.stop_reason, StopReason::StreamClosed);
    }
}
