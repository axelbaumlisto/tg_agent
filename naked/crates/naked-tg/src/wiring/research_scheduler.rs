use std::sync::Arc;

use naked_core::AgentCore;
use naked_core::config::Config;
use naked_tg::channel_map::ChannelSessionMap;

use crate::shared::research_scheduler;

pub(crate) fn start_research_scheduler(
    agent: &Arc<AgentCore>,
    config: &Config,
    channel_map: &Arc<ChannelSessionMap>,
    liveness: &Arc<naked_core::liveness::LivenessRegistry>,
    scheduler_lock_held: bool,
) -> Option<Arc<research_scheduler::ResearchScheduler>> {
    // T2.6 (PLAN_RESEARCH_AGENT_FLOW_v1): build the synthetic-message
    // dispatch closure now that Bot + channel_map + agent are all in scope.
    // The scheduler will call this when a spec is due AND has `chat_id`
    // configured — the synthetic message lands in the operator's chat thread,
    // gets a normal session via channel_map, and streams through the same
    // pipeline as user-typed messages. The standard ⏹ Abort button is attached
    // automatically; `/abort` command works the same way (B57 mitigation).
    if !config.research.enabled || !scheduler_lock_held {
        return None;
    }

    let dispatch_fn = synthetic_dispatch_fn(agent, channel_map);
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
) -> naked_tg::synthetic::SyntheticDispatchFn {
    // B1 (PLAN_RESEARCH_FLOW_CLOSURE_v1): the dispatch closure is now a thin
    // shim over `synthetic::dispatch_for_chat` so the scheduler path AND the
    // operator `/research run X` path share the same flow. The
    // session-creation policy lives in synthetic.rs.
    let agent = agent.clone();
    let channel_map = channel_map.clone();
    Arc::new(move |msg: naked_tg::synthetic::SyntheticMessage| {
        let agent = agent.clone();
        let channel_map = channel_map.clone();
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
                Ok((sid, _handle)) => {
                    tracing::info!(
                        session_id = %sid,
                        spec_id = %spec_id_owned,
                        "synthetic dispatch: turn submitted via wiring closure"
                    );
                }
                Err(e) => {
                    tracing::warn!(
                        spec_id = %spec_id_owned,
                        error = %e,
                        "synthetic dispatch failed; scheduler will retry next tick"
                    );
                }
            }
        })
    })
}

#[cfg(test)]
mod tests {
    #[test]
    fn scheduler_gate_requires_enabled_config_and_lock() {
        let source = include_str!("research_scheduler.rs");
        assert!(source.contains("!config.research.enabled || !scheduler_lock_held"));
        assert!(source.contains("dispatch_fn: Some(dispatch_fn)"));
    }
}
