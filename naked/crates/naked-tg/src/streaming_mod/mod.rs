//! Streaming response handler + CompositeView.

mod delta;
mod flush;
mod handlers;
mod helpers;
mod pipeline;
pub(crate) use delta::render_html_document;
pub(crate) use flush::{ask_permission, flush_live, send_final, send_long_text};
pub(crate) use handlers::ViewAction;
pub(crate) use helpers::*;
pub(crate) use pipeline::stream_response;

#[allow(unused_imports)]
use super::*;
// ── Streaming response ──────────────────────────────────────────────────────

pub(crate) struct SubAgentState {
    prompt: String,
    status: &'static str,
    last_tool: Option<String>,
    tool_count: u32,
}

pub(crate) struct CompositeView {
    thinking: String,
    in_thinking: bool,
    tool_lines: Vec<String>,
    sub_agents: std::collections::HashMap<String, SubAgentState>,
    sub_agent_order: Vec<String>,
    response_text: String,
    usage: Option<TurnUsage>,
    context_window: u64,
    model_tag: String,
    tick: usize,
    phase: &'static str,
    started_at: std::time::Instant,
    /// When the currently running tool started (for per-tool timer).
    tool_started_at: Option<std::time::Instant>,
    /// Name of the currently running tool.
    active_tool: Option<String>,
    /// Last few lines of stdout/stderr from the running tool.
    tool_output: Option<String>,
    /// Live count of messages queued while this turn is active.
    queue_counter: Arc<std::sync::atomic::AtomicUsize>,
    /// Set when a provider error occurs — used to show retry buttons after final.
    pub(crate) had_provider_error: bool,
}

const SPINNER: &[&str] = &["⏳", "⌛", "⏳", "⌛"];

impl CompositeView {
    fn new(model_tag: String, queue_counter: Arc<std::sync::atomic::AtomicUsize>) -> Self {
        Self {
            thinking: String::new(),
            in_thinking: false,
            tool_lines: Vec::new(),
            sub_agents: std::collections::HashMap::new(),
            sub_agent_order: Vec::new(),
            response_text: String::new(),
            usage: None,
            context_window: naked_core::history::model_context_window(&model_tag) as u64,
            tool_started_at: None,
            active_tool: None,
            tool_output: None,
            model_tag,
            tick: 0,
            phase: "thinking",
            started_at: std::time::Instant::now(),
            queue_counter,
            had_provider_error: false,
        }
    }

    fn elapsed_label(&self) -> String {
        let total = self.started_at.elapsed().as_secs();
        let m = total / 60;
        let s = total % 60;
        if m == 0 {
            format!("🕐 {s}s")
        } else {
            format!("🕐 {m}:{s:02}")
        }
    }

    fn spinner(&self) -> &'static str {
        SPINNER[self.tick % SPINNER.len()]
    }
}

// REGISTRY-WAIVE: too_many_arguments — refactor-defer, signature complexity acceptable
#[allow(clippy::too_many_arguments)]
/// Apply a sub-agent progress event to the composite view.
fn apply_sub_agent_event(
    view: &mut CompositeView,
    agent_id: String,
    event: naked_core::types::SubAgentEvent,
) {
    use naked_core::types::SubAgentEvent;
    match event {
        SubAgentEvent::Started { prompt_preview } => {
            view.phase = "sub_agent";
            if !view.sub_agent_order.contains(&agent_id) {
                view.sub_agent_order.push(agent_id.clone());
            }
            view.sub_agents.insert(
                agent_id,
                SubAgentState {
                    prompt: prompt_preview,
                    status: "running",
                    last_tool: None,
                    tool_count: 0,
                },
            );
        }
        SubAgentEvent::ToolUse { name, .. } => {
            if let Some(sa) = view.sub_agents.get_mut(&agent_id) {
                sa.last_tool = Some(name);
                sa.tool_count = sa.tool_count.saturating_add(1);
            }
        }
        SubAgentEvent::ToolDone { .. } | SubAgentEvent::TextDelta(_) => {}
        SubAgentEvent::Finished { .. } => {
            if let Some(sa) = view.sub_agents.get_mut(&agent_id) {
                sa.status = "done";
                sa.last_tool = None;
            }
        }
        SubAgentEvent::Error(_) => {
            if let Some(sa) = view.sub_agents.get_mut(&agent_id) {
                sa.status = "error";
            }
        }
    }
}

fn format_provider_error(err: &str, model_tag: &str) -> String {
    if err.contains("no content") {
        return "— модель закрыла ход без ответа (0 токенов).\n\
             Попробуй переформулировать или /new."
            .to_string();
    }

    let (icon, reason, hint) = if err.contains("429")
        || err.contains("Too Many Requests")
        || err.contains("rate limit")
    {
        (
            "⏳",
            "Rate limit (429)",
            "Подожди 1–2 мин или /model — переключи модель",
        )
    } else if err.contains("402") || err.contains("Payment Required") || err.contains("membership")
    {
        (
            "💳",
            "API ключ — оплата/подписка (402)",
            "/model — переключи провайдер",
        )
    } else if err.contains("401")
        || err.contains("Unauthorized")
        || err.contains("Invalid Authentication")
    {
        ("🔒", "Ключ невалиден (401)", "/model — переключи провайдер")
    } else if err.contains("500")
        || err.contains("502")
        || err.contains("503")
        || err.contains("Internal Server")
    {
        (
            "🔧",
            "Сервер провайдера упал (5xx)",
            "Повтори через минуту или /model",
        )
    } else if err.contains("timeout") || err.contains("Timeout") {
        ("⏱", "Timeout", "Попробуй короче или /model")
    } else if err.contains("reasoning_content") {
        (
            "🧠",
            "Модель требует reasoning format",
            "/new — новая сессия или /model",
        )
    } else {
        ("❌", "Ошибка провайдера", "/model — переключи модель")
    };

    // Extract short error (first sentence or 120 chars, no JSON blobs)
    let short_err = err
        .split('{')
        .next()
        .unwrap_or(err)
        .trim()
        .chars()
        .take(120)
        .collect::<String>();

    format!(
        "{icon} {reason}\n\
         Модель: {model_tag}\n\
         {short_err}\n\
         💡 {hint}"
    )
}

#[cfg(test)]
#[path = "tests.rs"]
mod streaming_tests;
