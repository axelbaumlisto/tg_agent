//! Per-event view handlers — pure functions that mutate `CompositeView`.
//!
//! Each handler takes `&mut CompositeView` + event-specific data and returns `ViewAction`.
//! The dispatch loop calls the handler, then acts on `ViewAction`.
//! Adding a new `AgentEvent` variant = add one handler here. Zero changes to the loop.

use super::helpers::{format_input_preview, truncate_str};
use super::*;

/// Normalize multi-line bash command for `args_preview` rendering.
///
/// Collapse runs of blank lines into a single `\n` so 400-char preview
/// holds more usable content. Strip leading/trailing whitespace.
fn normalize_args_preview(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    let mut prev_was_blank = false;
    for line in s.lines() {
        let trimmed_end = line.trim_end();
        if trimmed_end.is_empty() {
            if !prev_was_blank && !out.is_empty() {
                out.push('\n');
                prev_was_blank = true;
            }
        } else {
            if !out.is_empty() && !prev_was_blank {
                out.push('\n');
            }
            out.push_str(trimmed_end);
            prev_was_blank = false;
        }
    }
    out.trim().to_string()
}

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
    // PLAN_TG_INTERLEAVED_v1 §2.2: also push event to timeline so the
    // renderer can interleave thinking with text/tools chronologically.
    view.events.push(TurnEvent::ReasoningDelta(text.to_string()));
    ViewAction::Dirty
}

pub(crate) fn handle_text_delta(view: &mut CompositeView, text: &str) -> ViewAction {
    view.in_thinking = false;
    view.phase = "generating";
    if view.response_text.len() < MAX_RESPONSE_BYTES {
        view.response_text.push_str(text);
    }
    view.events.push(TurnEvent::TextDelta(text.to_string()));
    ViewAction::Dirty
}

pub(crate) fn handle_tool_start(
    view: &mut CompositeView,
    name: &str,
    input: &serde_json::Value,
) -> ViewAction {
    view.in_thinking = false;
    view.phase = "tool use";
    // Q3: bump args preview from 200 → 400 + normalize blank lines.
    let raw_preview = format_input_preview(input, 400);
    let preview = normalize_args_preview(&raw_preview);
    let idx = view.next_tool_idx;
    view.next_tool_idx = view.next_tool_idx.saturating_add(1);
    view.events.push(TurnEvent::ToolStart {
        name: name.to_string(),
        args_preview: preview,
        idx,
    });
    // Track active tool for per-tool timer + stdout preview.
    view.active_tool = Some(name.to_string());
    view.tool_started_at = Some(std::time::Instant::now());
    view.tool_output = None;
    ViewAction::DirtyFlush
}

/// Returns (ViewAction, Option<detail_message_for_large_errors>).
pub(crate) fn handle_tool_end(
    view: &mut CompositeView,
    name: &str,
    is_error: bool,
    output: &str,
) -> (ViewAction, Option<String>) {
    // Clear active tool — it's done.
    view.active_tool = None;
    view.tool_started_at = None;
    view.tool_output = None;

    // Find matching ToolStart.idx (most recent ToolStart with this name
    // that hasn't been paired with a ToolResult yet). Falls back to 0 if
    // unmatchable, which is harmless — idx is only used for visual pairing.
    let mut matched_idx: u32 = 0;
    let mut seen_results: std::collections::HashSet<u32> =
        std::collections::HashSet::new();
    for ev in view.events.iter().rev() {
        match ev {
            TurnEvent::ToolResult { idx, .. } => {
                seen_results.insert(*idx);
            }
            TurnEvent::ToolStart { name: n, idx, .. }
                if n == name && !seen_results.contains(idx) =>
            {
                matched_idx = *idx;
                break;
            }
            _ => {}
        }
    }

    view.events.push(TurnEvent::ToolResult {
        idx: matched_idx,
        ok: !is_error,
        output: output.to_string(),
    });

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
    fn tool_start_pushes_event() {
        let mut v = new_view();
        let action = handle_tool_start(&mut v, "bash", &serde_json::json!({"command": "ls"}));
        assert_eq!(action, ViewAction::DirtyFlush);
        assert_eq!(v.events.len(), 1);
        assert!(matches!(
            &v.events[0],
            TurnEvent::ToolStart { name, .. } if name == "bash"
        ));
    }

    #[test]
    fn tool_end_success_pairs_with_start() {
        let mut v = new_view();
        let _ = handle_tool_start(&mut v, "read_file", &serde_json::json!({"path": "foo"}));
        let (action, detail) = handle_tool_end(&mut v, "read_file", false, "file contents here");
        assert_eq!(action, ViewAction::DirtyFlush);
        assert!(detail.is_none());
        // Two events: ToolStart + ToolResult; both with idx=0 paired.
        assert_eq!(v.events.len(), 2);
        let TurnEvent::ToolStart { idx: start_idx, .. } = &v.events[0] else {
            panic!("expected ToolStart")
        };
        let TurnEvent::ToolResult {
            idx: result_idx,
            ok,
            ..
        } = &v.events[1]
        else {
            panic!("expected ToolResult")
        };
        assert_eq!(start_idx, result_idx);
        assert!(*ok);
    }

    #[test]
    fn tool_end_error_keeps_detail() {
        let mut v = new_view();
        let _ = handle_tool_start(&mut v, "bash", &serde_json::json!({"command": "false"}));
        let long_error = "x".repeat(600);
        let (action, detail) = handle_tool_end(&mut v, "bash", true, &long_error);
        assert_eq!(action, ViewAction::DirtyFlush);
        assert!(detail.is_some());
        assert!(matches!(
            v.events.last(),
            Some(TurnEvent::ToolResult { ok: false, .. })
        ));
    }

    #[test]
    fn text_delta_pushes_event() {
        let mut v = new_view();
        let _ = handle_text_delta(&mut v, "Hello");
        let _ = handle_text_delta(&mut v, " world");
        assert_eq!(v.events.len(), 2);
        assert!(matches!(
            &v.events[0],
            TurnEvent::TextDelta(s) if s == "Hello"
        ));
    }

    #[test]
    fn normalize_args_preview_collapses_blanks() {
        let s = "line1\n\n\n\nline2\n\nline3";
        let got = normalize_args_preview(s);
        assert_eq!(got, "line1\nline2\nline3");
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
