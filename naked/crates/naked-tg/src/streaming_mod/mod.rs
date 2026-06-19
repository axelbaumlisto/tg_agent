//! Streaming response handler + CompositeView.

mod delta;
mod flush;
mod handlers;
mod helpers;
mod pipeline;
pub(crate) use delta::render_html_document;
pub(crate) use flush::{ask_permission, flush_live, send_final};
pub(crate) use handlers::ViewAction;
pub(crate) use helpers::*;
pub(crate) use pipeline::{register_turn_routing, stream_response};

#[allow(unused_imports)]
use super::*;
// ── Streaming response ──────────────────────────────────────────────────────

pub(crate) struct SubAgentState {
    prompt: String,
    status: &'static str,
    last_tool: Option<String>,
    tool_count: u32,
}

/// One chronological entry in the turn's event timeline.
///
/// Events are pushed in arrival order; renderer iterates the `Vec`
/// strictly head-to-tail. Adjacent `ReasoningDelta` / `TextDelta`
/// events are coalesced at render time into single visual blocks.
///
/// Why this exists (PLAN_TG_INTERLEAVED_v1 §2): the previous model
/// stored `thinking: String`, `tool_lines: Vec<String>`, and
/// `response_text: String` as three parallel streams without
/// timestamps. Renderer was forced to emit "all tools first, all text
/// last", breaking chronology. This enum captures the true sequence.
#[derive(Debug, Clone)]
#[allow(dead_code)] // SubAgentReference + Note used optionally
pub(crate) enum TurnEvent {
    /// Reasoning fragment (model's hidden CoT). Coalesced with adjacent.
    ReasoningDelta(String),
    /// Assistant text fragment (markdown). Coalesced with adjacent.
    TextDelta(String),
    /// Tool invocation started. Pair with subsequent `ToolResult` by `idx`.
    ToolStart {
        name: String,
        args_preview: String,
        idx: u32,
    },
    /// Tool finished. `idx` links back to the matching `ToolStart`.
    ToolResult { idx: u32, ok: bool, output: String },
    /// Reference into the `sub_agents` map; renders as the sub-agent's
    /// current status line at this position in the timeline.
    SubAgentReference { agent_id: String },
    /// Out-of-band note (stall warnings, steer-received echoes, etc.).
    /// Rendered as plain italic line in chronology.
    Note(String),
}

pub(crate) struct CompositeView {
    thinking: String,
    in_thinking: bool,
    /// Chronological timeline. **The single source of truth for ordering.**
    pub(crate) events: Vec<TurnEvent>,
    /// Counter for next `ToolStart.idx` — monotonically increasing per turn.
    pub(crate) next_tool_idx: u32,
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
    /// Set when a provider error occurs — used to show retry buttons after final.
    pub(crate) had_provider_error: bool,
    /// PLAN_TG_INTERLEAVED_v1 Q4: count of events dropped by the
    /// most recent `render_final` head-truncation. flush.rs reads this
    /// after rendering; if > 0, it knows to attach the full-history
    /// HTML document so nothing is lost.
    pub(crate) last_dropped_events: std::sync::atomic::AtomicUsize,
}

const SPINNER: &[&str] = &["⏳", "⌛", "⏳", "⌛"];

impl CompositeView {
    fn new(model_tag: String) -> Self {
        Self {
            thinking: String::new(),
            in_thinking: false,
            events: Vec::new(),
            next_tool_idx: 0,
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
            had_provider_error: false,
            last_dropped_events: std::sync::atomic::AtomicUsize::new(0),
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
