//! Per-event view handlers — pure functions that mutate `CompositeView`.
//!
//! Each handler takes `&mut CompositeView` + event-specific data and returns `ViewAction`.
//! The dispatch loop calls the handler, then acts on `ViewAction`.
//! Adding a new `AgentEvent` variant = add one handler here. Zero changes to the loop.

use super::helpers::{format_input_preview, truncate_str};
use super::*;

/// What the dispatch loop should do after a handler runs.
#[derive(Debug, Clone, PartialEq, Eq)]
#[allow(dead_code)] // Break will be used when Idle is extracted
pub(crate) enum ViewAction {
    /// Mark view dirty (normal: will flush on next tick).
    Dirty,
    /// Mark dirty AND force immediate flush.
    DirtyFlush,
    /// No change needed.
    Clean,
    /// The turn is complete — break out of the event loop.
    Break,
}

// ---------------------------------------------------------------------------
// Pure view handlers
// ---------------------------------------------------------------------------

pub(crate) fn handle_thinking_delta(view: &mut CompositeView, text: &str) -> ViewAction {
    view.in_thinking = true;
    view.phase = "thinking";
    if view.thinking.len() < MAX_THINKING_BYTES {
        view.thinking.push_str(text);
    }
    ViewAction::Dirty
}

pub(crate) fn handle_text_delta(view: &mut CompositeView, text: &str) -> ViewAction {
    view.in_thinking = false;
    view.phase = "generating";
    if view.response_text.len() < MAX_RESPONSE_BYTES {
        view.response_text.push_str(text);
    }
    ViewAction::Dirty
}

pub(crate) fn handle_tool_start(
    view: &mut CompositeView,
    name: &str,
    input: &serde_json::Value,
) -> ViewAction {
    view.in_thinking = false;
    view.phase = "tool use";
    let preview = format_input_preview(input, 200);
    if view.tool_lines.len() >= TOOL_WINDOW * 4 {
        view.tool_lines.drain(..view.tool_lines.len() - TOOL_WINDOW);
    }
    view.tool_lines
        .push(format!("🔧 <b>{}</b>({preview})…", escape_html(name)));
    ViewAction::DirtyFlush
}

/// Returns (ViewAction, Option<detail_message_for_large_errors>).
pub(crate) fn handle_tool_end(
    view: &mut CompositeView,
    name: &str,
    is_error: bool,
    output: &str,
) -> (ViewAction, Option<String>) {
    let icon = if is_error { "❌" } else { "✅" };
    let budget = if is_error { 500 } else { 80 };
    let title = truncate_str(output, budget);
    view.tool_lines.push(format!(
        "{icon} <b>{}</b> — {}",
        escape_html(name),
        escape_html(&title)
    ));
    let detail_msg = if is_error && output.len() > 500 {
        let detail = truncate_str(output, 2000);
        Some(format!(
            "⚠️ <b>{}</b> error details:\n<pre>{}</pre>",
            escape_html(name),
            escape_html(&detail),
        ))
    } else {
        None
    };
    (ViewAction::DirtyFlush, detail_msg)
}

pub(crate) fn handle_compaction(
    before_msgs: usize,
    after_msgs: usize,
    files_count: usize,
    summary_hint: Option<&str>,
) -> String {
    let mut note = format!("📦 контекст сжат: {} → {}", before_msgs, after_msgs);
    if files_count > 0 {
        note.push_str(&format!(" | {} file(s)", files_count));
    }
    if let Some(hint) = summary_hint {
        note.push_str(&format!("\n🎯 {}", escape_html(hint)));
    }
    note
}

pub(crate) fn handle_heartbeat(view: &mut CompositeView) -> ViewAction {
    view.tick += 1;
    ViewAction::Dirty
}

pub(crate) fn handle_sub_agent(
    view: &mut CompositeView,
    agent_id: String,
    event: naked_core::types::SubAgentEvent,
) -> ViewAction {
    apply_sub_agent_event(view, agent_id, event);
    ViewAction::DirtyFlush
}

pub(crate) fn handle_usage(
    view: &mut CompositeView,
    usage: naked_core::types::TurnUsage,
) -> ViewAction {
    view.usage = Some(usage);
    ViewAction::Dirty
}

pub(crate) fn handle_error(view: &mut CompositeView, error: &str) -> ViewAction {
    let pretty = format_provider_error(error, &view.model_tag);
    if !view.response_text.is_empty() {
        view.response_text.push('\n');
    }
    view.response_text.push_str(&pretty);
    view.had_provider_error = true;
    ViewAction::DirtyFlush
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn new_view() -> CompositeView {
        CompositeView::new(
            "test-model".into(),
            Arc::new(std::sync::atomic::AtomicUsize::new(0)),
        )
    }

    #[test]
    fn thinking_delta_sets_phase() {
        let mut v = new_view();
        let action = handle_thinking_delta(&mut v, "hmm...");
        assert_eq!(action, ViewAction::Dirty);
        assert!(v.in_thinking);
        assert_eq!(v.phase, "thinking");
        assert_eq!(v.thinking, "hmm...");
    }

    #[test]
    fn text_delta_clears_thinking() {
        let mut v = new_view();
        v.in_thinking = true;
        let action = handle_text_delta(&mut v, "Hello");
        assert_eq!(action, ViewAction::Dirty);
        assert!(!v.in_thinking);
        assert_eq!(v.response_text, "Hello");
    }

    #[test]
    fn tool_start_formats_line() {
        let mut v = new_view();
        let action = handle_tool_start(&mut v, "bash", &serde_json::json!({"command": "ls"}));
        assert_eq!(action, ViewAction::DirtyFlush);
        assert_eq!(v.tool_lines.len(), 1);
        assert!(v.tool_lines[0].contains("bash"));
    }

    #[test]
    fn tool_end_success_compact() {
        let mut v = new_view();
        let (action, detail) = handle_tool_end(&mut v, "read_file", false, "file contents here");
        assert_eq!(action, ViewAction::DirtyFlush);
        assert!(detail.is_none());
        assert!(v.tool_lines[0].contains("✅"));
    }

    #[test]
    fn tool_end_error_with_detail() {
        let mut v = new_view();
        let long_error = "x".repeat(600);
        let (action, detail) = handle_tool_end(&mut v, "bash", true, &long_error);
        assert_eq!(action, ViewAction::DirtyFlush);
        assert!(detail.is_some());
        assert!(v.tool_lines[0].contains("❌"));
    }

    #[test]
    fn compaction_message_with_hint() {
        let msg = handle_compaction(100, 10, 5, Some("Fix Caddy routing"));
        assert!(msg.contains("100 → 10"));
        assert!(msg.contains("5 file(s)"));
        assert!(msg.contains("Fix Caddy routing"));
    }

    #[test]
    fn heartbeat_increments_tick() {
        let mut v = new_view();
        assert_eq!(v.tick, 0);
        handle_heartbeat(&mut v);
        assert_eq!(v.tick, 1);
    }

    #[test]
    fn error_handler_marks_provider_error() {
        let mut v = new_view();
        let action = handle_error(&mut v, "timeout");
        assert_eq!(action, ViewAction::DirtyFlush);
        assert!(v.had_provider_error);
    }
}
