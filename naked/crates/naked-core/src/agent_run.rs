//! Generic agent task executor + parallel batch runner.
//!
//! Bridges the type-only [`crate::agent_role`] primitives to the
//! live [`crate::AgentCore`] runtime. Two public entry points:
//!
//! * [`run_task`]   — execute a single [`Task`] under an [`AgentRole`],
//!   returning a [`TaskOutput`].
//! * [`run_batch`]  — execute many tasks concurrently, bounded by the
//!   process-wide research-run semaphore (so we never exceed the
//!   browser-MCP cap).
//!
//! Why this lives next to [`crate::agent_role`] but not inside it:
//! `agent_role` is type-only (no `AgentCore` dep), so its tests run
//! in microseconds. This module is the thin glue that depends on the
//! whole runtime — kept separate so refactors to either layer don't
//! cascade.
//!
//! Design notes:
//!
//! * The role's `system` text is currently injected as a prefix to
//!   the user prompt (with a `[ROLE: ...]` header). This is MVP — a
//!   future `AgentCore::create_session_with_system_extra` helper will
//!   let us put it in the actual system prompt without touching
//!   `dispatch_turn`. Functionally equivalent for now.
//! * `default_skills` are advertised in the prompt header so the LLM
//!   knows it can `Skill(...)` them. We do NOT auto-load them into
//!   the prompt — the agent decides per-task whether to spend the
//!   tokens. (Reduces idle prompt cost on tasks that don't need the
//!   skill.)
//! * `tool_filter` is currently advisory: the registered toolset is
//!   the AgentCore default, and the role's `system` text tells the
//!   model which subset to use. Hard filtering at the registry level
//!   is a follow-up — needs `dispatch_turn` to accept an override.
//! * Concurrency is gated by `AgentCore::research_run_permits()`. We
//!   reuse that semaphore on purpose: the binding constraint is the
//!   single Playwright browser, which both research runs and generic
//!   agent tasks share. A separate semaphore would let batch tasks
//!   trample research runs.

use std::sync::Arc;
use std::time::{Duration, Instant};

use futures_util::stream::{FuturesUnordered, StreamExt};
use tokio::time::timeout;

use crate::agent_role::{AgentRole, StopReason, Task, TaskOutput, TaskStats, expand_placeholders};
use crate::error::Result;
use crate::types::{AgentEvent, AgentHandle, PermissionResponse};
use crate::{AgentCore, acquire_research_permit};

/// Anti-bot markers worth flagging in [`TaskStats::captcha_hits`].
/// Mirrors `research::coordinator::DrainStats::CAPTCHA_MARKERS` so
/// `naked agent batch` and `naked research probe` produce comparable
/// numbers. When this list grows, update both call sites or extract
/// to a shared constant in `types.rs`.
const CAPTCHA_MARKERS: &[&str] = &[
    "xac-thuc-nguoi-dung",
    "Vui lòng xác minh",
    "Just a moment",
    "Attention Required",
    "Enable JavaScript and cookies",
    "cf_chl_rt_tk",
];

/// Default wall-clock cap when neither the task nor the role specifies
/// one. Aligned with `naked research probe` (90s) — enough for a
/// browser navigation + a few snapshots, not so long the user gives
/// up watching.
const DEFAULT_TASK_WALL_SECS: u64 = 90;

/// Concrete prompt the executor sends as the user message.
///
/// Exposed for tests only — production code never needs to inspect it.
#[doc(hidden)]
pub fn build_task_prompt(role: &AgentRole, task: &Task) -> String {
    let system = expand_placeholders(&role.system, &task.context);
    let prompt = expand_placeholders(&task.prompt, &task.context);

    let mut out = String::new();
    out.push_str(&format!("[ROLE: {}]\n", role.name));
    if !role.description.is_empty() {
        out.push_str(&format!("# Purpose\n{}\n\n", role.description));
    }
    out.push_str("# System guidance\n");
    out.push_str(system.trim());
    out.push_str("\n\n");

    if !role.default_skills.is_empty() {
        out.push_str("# Available skills\n");
        out.push_str("Load any of these on demand via the `Skill` tool when relevant:\n");
        for s in &role.default_skills {
            out.push_str(&format!("- `{s}`\n"));
        }
        out.push('\n');
    }

    if let Some(allowed) = match &role.tool_filter {
        crate::agent_role::ToolFilter::Allow { tools } => Some(tools.clone()),
        _ => None,
    } && !allowed.is_empty()
    {
        out.push_str("# Allowed tools\n");
        out.push_str(
            "You should restrict yourself to these tools (others are unavailable for this role):\n",
        );
        for t in &allowed {
            out.push_str(&format!("- `{t}`\n"));
        }
        out.push('\n');
    }

    out.push_str("# Task\n");
    out.push_str(prompt.trim());
    out.push('\n');
    out
}

/// Drain agent events into a [`TaskStats`] + concatenated text. Auto-
/// approves tool permissions (batch tasks are headless, like research
/// runs).
///
/// Every event is also logged on the `naked::agent_run::stream` target
/// at `info` (tools, errors, idle, permissions) or `debug` (per-token
/// thinking/text deltas) — that way operators can `RUST_LOG=naked=info`
/// for a tool-by-tool trace, or `RUST_LOG=naked=debug` to also stream
/// raw model output. The CLI helper `print_task_stream_setup` documents
/// the recommended env for full e2e visibility.
async fn drain_to_output(
    handle: &mut AgentHandle,
    stats: &mut TaskStats,
    text: &mut String,
    label: &str,
) -> StopReason {
    use std::time::Instant;
    let started = Instant::now();
    let mut text_buf = String::new();
    let mut think_buf = String::new();

    loop {
        match handle.events.recv().await {
            Some(AgentEvent::Idle) => {
                if !text_buf.is_empty() {
                    tracing::info!(target: "naked::agent_run::stream", task = %label, kind = "text",
                        "[text] {}", truncate_for_log(&text_buf, 800));
                }
                if !think_buf.is_empty() {
                    tracing::debug!(target: "naked::agent_run::stream", task = %label, kind = "thinking",
                        "[thinking] {}", truncate_for_log(&think_buf, 800));
                }
                tracing::info!(target: "naked::agent_run::stream", task = %label, elapsed_ms = started.elapsed().as_millis() as u64,
                    "[idle] agent loop finished");
                return StopReason::AgentIdle;
            }
            Some(AgentEvent::Error(e)) => {
                stats.errors += 1;
                tracing::warn!(target: "naked::agent_run::stream", task = %label, "[error] {e}");
            }
            Some(AgentEvent::ToolStart {
                call_id,
                name,
                input,
            }) => {
                *stats.tools.entry(name.clone()).or_default() += 1;
                if name == "Skill" {
                    stats.skill_loads += 1;
                }
                if !text_buf.is_empty() {
                    tracing::info!(target: "naked::agent_run::stream", task = %label, kind = "text",
                        "[text] {}", truncate_for_log(&text_buf, 800));
                    text_buf.clear();
                }
                if !think_buf.is_empty() {
                    tracing::debug!(target: "naked::agent_run::stream", task = %label, kind = "thinking",
                        "[thinking] {}", truncate_for_log(&think_buf, 800));
                    think_buf.clear();
                }
                let input_preview = truncate_for_log(&input.to_string(), 400);
                tracing::info!(target: "naked::agent_run::stream", task = %label, call_id = %call_id, tool = %name,
                    "[tool-start] {name}({input_preview})");
            }
            Some(AgentEvent::ToolEnd {
                call_id,
                name,
                state,
                output,
            }) => {
                if CAPTCHA_MARKERS.iter().any(|m| output.contains(m)) {
                    stats.captcha_hits += 1;
                }
                let output_preview = truncate_for_log(&output, 400);
                let captcha_tag = CAPTCHA_MARKERS
                    .iter()
                    .find(|m| output.contains(*m))
                    .map(|m| format!(" captcha={m:?}"))
                    .unwrap_or_default();
                tracing::info!(target: "naked::agent_run::stream", task = %label, call_id = %call_id, tool = %name, state = ?state,
                    "[tool-end] {name} -> {state:?}{captcha_tag} :: {output_preview}");
            }
            Some(AgentEvent::PermissionRequest {
                call_id, tool_name, ..
            }) => {
                tracing::info!(target: "naked::agent_run::stream", task = %label, call_id = %call_id, tool = %tool_name,
                    "[permission] auto-approving (batch headless)");
                let _ = handle
                    .permissions
                    .send(PermissionResponse {
                        call_id,
                        allowed: true,
                    })
                    .await;
            }
            Some(AgentEvent::TextDelta(t)) => {
                stats.text_deltas += 1;
                text.push_str(&t);
                text_buf.push_str(&t);
            }
            Some(AgentEvent::ThinkingDelta(t)) => {
                think_buf.push_str(&t);
            }
            Some(AgentEvent::ContextCompacted {
                before_msgs,
                after_msgs,
            }) => {
                tracing::info!(target: "naked::agent_run::stream", task = %label,
                    "[context-compact] {before_msgs} -> {after_msgs} msgs");
            }
            Some(AgentEvent::UsageUpdate(u)) => {
                tracing::debug!(target: "naked::agent_run::stream", task = %label,
                    "[usage] {u:?}");
            }
            Some(_) => continue,
            None => {
                if !text_buf.is_empty() {
                    tracing::info!(target: "naked::agent_run::stream", task = %label, kind = "text",
                        "[text] {}", truncate_for_log(&text_buf, 800));
                }
                tracing::info!(target: "naked::agent_run::stream", task = %label,
                    "[stream-closed]");
                return StopReason::StreamClosed;
            }
        }
    }
}

/// Single-line preview of a multi-line / large value for log lines.
/// Newlines collapsed, tabs collapsed, ellipsised when over `cap`.
fn truncate_for_log(s: &str, cap: usize) -> String {
    let cleaned: String = s
        .chars()
        .map(|c| match c {
            '\n' | '\r' => ' ',
            '\t' => ' ',
            other => other,
        })
        .collect();
    if cleaned.chars().count() <= cap {
        cleaned
    } else {
        let head: String = cleaned.chars().take(cap).collect();
        format!("{head}…[+{} chars]", cleaned.chars().count() - cap)
    }
}

/// Execute one [`Task`] under one [`AgentRole`]. Acquires a research
/// permit (so concurrent calls share the global cap), creates an
/// ephemeral session, applies model override, drains events with a
/// wall-clock cap, then returns the [`TaskOutput`].
///
/// The session is intentionally NOT deleted on exit — like research
/// runs, it serves as an audit breadcrumb under `~/.naked/sessions/`.
/// Operators clean up via the existing `vacuum-sessions` machinery.
pub async fn run_task(core: &Arc<AgentCore>, role: &AgentRole, task: &Task) -> Result<TaskOutput> {
    let started = Instant::now();
    let label = if task.id.is_empty() {
        format!("agent:{}", role.name)
    } else {
        format!("agent:{}:{}", role.name, task.id)
    };

    let _permit = acquire_research_permit(&core.research_run_permits(), &label).await?;

    let workspace = core.config().workspace.clone();
    let session_id = core
        .create_session_with_channel(&workspace, "agent-task")
        .await;

    if let Some(model) = &role.model
        && let Err(e) = core
            .set_session_provider(&session_id, None, Some(model))
            .await
    {
        tracing::warn!(role = %role.name, "agent role model override failed: {e}");
    }
    let now_ts = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs() as i64)
        .unwrap_or(0);
    if let Err(e) = core.set_session_yolo(&session_id, Some(now_ts)).await {
        tracing::warn!(role = %role.name, "agent role yolo failed: {e}");
    }

    let prompt = build_task_prompt(role, task);
    tracing::info!(
        target: "naked::agent_run::stream",
        task = %label,
        role = %role.name,
        session = %session_id,
        prompt_len = prompt.len(),
        "[task-start] role={} session={} wall={}s",
        role.name,
        session_id,
        task.max_wall_secs.unwrap_or(DEFAULT_TASK_WALL_SECS)
    );
    tracing::debug!(
        target: "naked::agent_run::stream",
        task = %label,
        "[task-prompt] {}",
        truncate_for_log(&prompt, 4_000)
    );
    let mut handle = core.send_prompt(&session_id, &prompt).await?;

    let wall_secs = task.max_wall_secs.unwrap_or(DEFAULT_TASK_WALL_SECS);
    let mut stats = TaskStats::default();
    let mut text = String::new();

    let stop_reason = match timeout(
        Duration::from_secs(wall_secs),
        drain_to_output(&mut handle, &mut stats, &mut text, &label),
    )
    .await
    {
        Ok(reason) => reason,
        Err(_) => {
            tracing::warn!(target: "naked::agent_run::stream", task = %label,
                "[timeout] wall={wall_secs}s exceeded");
            StopReason::Timeout
        }
    };

    let elapsed_secs = started.elapsed().as_secs_f64();
    let task_id_out = if task.id.is_empty() {
        session_id.clone()
    } else {
        task.id.clone()
    };
    Ok(TaskOutput {
        task_id: task_id_out,
        role_name: role.name.clone(),
        stop_reason,
        elapsed_secs,
        text,
        stats,
        artifacts: serde_json::Value::Null,
    })
}

/// Execute many tasks concurrently. Each task picks its role from
/// `roles` (must be non-empty when `tasks` references its name).
/// Parallelism is bounded by `AgentCore::research_run_permits()` —
/// passing a `concurrency` higher than the global cap is fine, the
/// extra tasks just queue.
///
/// Order of returned outputs matches the order of `tasks` so callers
/// can correlate by index.
pub async fn run_batch(
    core: Arc<AgentCore>,
    roles: &[Arc<AgentRole>],
    tasks: Vec<Task>,
) -> Vec<Result<TaskOutput>> {
    if tasks.is_empty() {
        return Vec::new();
    }

    let mut futs: FuturesUnordered<_> = tasks
        .into_iter()
        .enumerate()
        .map(|(idx, task)| {
            let core = core.clone();
            let role = roles.iter().find(|r| r.name == task.role).cloned();
            async move {
                let res = match role {
                    None => Err(crate::error::AgentError::Config(format!(
                        "task #{idx} (id={}) references unknown role `{}`",
                        task.id, task.role
                    ))),
                    Some(role) => run_task(&core, &role, &task).await,
                };
                (idx, res)
            }
        })
        .collect();

    let mut indexed: Vec<(usize, Result<TaskOutput>)> = Vec::new();
    while let Some((idx, res)) = futs.next().await {
        indexed.push((idx, res));
    }
    indexed.sort_by_key(|(i, _)| *i);
    indexed.into_iter().map(|(_, r)| r).collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::agent_role::ToolFilter;

    #[test]
    fn build_task_prompt_includes_role_header_and_task() {
        let role = AgentRole::new("browser_extractor", "Open URL, extract phone.")
            .with_description("Headless browser worker");
        let task = Task::new("browser_extractor", "Open https://x.com").with_id("t-1");

        let p = build_task_prompt(&role, &task);
        assert!(p.starts_with("[ROLE: browser_extractor]"));
        assert!(p.contains("# Purpose\nHeadless browser worker"));
        assert!(p.contains("# System guidance"));
        assert!(p.contains("Open URL, extract phone."));
        assert!(p.contains("# Task"));
        assert!(p.contains("Open https://x.com"));
    }

    #[test]
    fn build_task_prompt_expands_placeholders_in_both_system_and_prompt() {
        let role = AgentRole::new("rt", "Topic: {topic}");
        let task = Task::new("rt", "Open {url}").with_context(serde_json::json!({
            "topic": "real-estate",
            "url":   "https://nhaban.com/listing/123",
        }));
        let p = build_task_prompt(&role, &task);
        assert!(p.contains("Topic: real-estate"));
        assert!(p.contains("Open https://nhaban.com/listing/123"));
    }

    #[test]
    fn build_task_prompt_advertises_skills_when_set() {
        let role = AgentRole::new("rt", "sys")
            .with_skills(vec!["web-browser-playbook".into(), "wayback-bypass".into()]);
        let task = Task::new("rt", "do");
        let p = build_task_prompt(&role, &task);
        assert!(p.contains("# Available skills"));
        assert!(p.contains("`web-browser-playbook`"));
        assert!(p.contains("`wayback-bypass`"));
    }

    #[test]
    fn build_task_prompt_omits_skills_section_when_empty() {
        let role = AgentRole::new("rt", "sys");
        let task = Task::new("rt", "do");
        let p = build_task_prompt(&role, &task);
        assert!(!p.contains("# Available skills"));
    }

    #[test]
    fn build_task_prompt_lists_allowed_tools_for_allow_filter() {
        let role = AgentRole::new("rt", "sys").with_tool_filter(ToolFilter::Allow {
            tools: vec!["web_fetch".into(), "research_save".into()],
        });
        let task = Task::new("rt", "do");
        let p = build_task_prompt(&role, &task);
        assert!(p.contains("# Allowed tools"));
        assert!(p.contains("`web_fetch`"));
        assert!(p.contains("`research_save`"));
    }

    #[test]
    fn build_task_prompt_omits_allowed_tools_for_allow_all_or_deny() {
        // AllowAll: no restriction announced.
        let role = AgentRole::new("rt", "sys");
        let p = build_task_prompt(&role, &Task::new("rt", "do"));
        assert!(!p.contains("# Allowed tools"));

        // Deny: also no positive whitelist (the host enforces — model
        // doesn't need a list of forbiddens cluttering the prompt).
        let role = AgentRole::new("rt", "sys").with_tool_filter(ToolFilter::Deny {
            tools: vec!["bash".into()],
        });
        let p = build_task_prompt(&role, &Task::new("rt", "do"));
        assert!(!p.contains("# Allowed tools"));
    }
}
