//! Telegram-side rendering helpers for the live `/research run` UX.
//!
//! Kept in its own module so `main.rs` stays focused on message
//! dispatch. The functions here are pure — no bot / network
//! interaction — which lets us unit-test the waterfall layout without
//! standing up a `teloxide` fake.

use chrono::{DateTime, Utc};
use naked_core::research::{EventKind, RunEvent};
use teloxide::types::{InlineKeyboardButton, InlineKeyboardMarkup};

/// Number of events shown in the waterfall. The plan says 5; we honour
/// that here as a single constant so downstream tweaks are trivial.
pub const WATERFALL_EVENTS: usize = 5;

/// Inline keyboard attached to every live-progress message. Single
/// "Stop & clarify" button — the user wanted a minimal set. The
/// callback payload is `r:stop:<spec_id>` so [`handle_callback`] in
/// `main.rs` can resolve the run without scraping the message body.
pub fn keyboard_stop(spec_id: &str) -> InlineKeyboardMarkup {
    InlineKeyboardMarkup::new(vec![vec![InlineKeyboardButton::callback(
        "⏸ Stop & clarify",
        format!("r:stop:{spec_id}"),
    )]])
}

/// Keyboard shown after a run finishes: operator can kick off another
/// one without retyping the id.
pub fn keyboard_after_complete(spec_id: &str) -> InlineKeyboardMarkup {
    InlineKeyboardMarkup::new(vec![vec![InlineKeyboardButton::callback(
        "🔁 Run again",
        format!("r:restart:{spec_id}"),
    )]])
}

/// Keyboard shown after a Stop & clarify pause: operator is invited
/// to send a clarification message, which the TG handler accumulates
/// into the spec's topic before offering restart.
pub fn keyboard_paused_awaiting_clarification(spec_id: &str) -> InlineKeyboardMarkup {
    InlineKeyboardMarkup::new(vec![vec![InlineKeyboardButton::callback(
        "▶ Restart with clarification",
        format!("r:restart:{spec_id}"),
    )]])
}

/// Everything the waterfall renderer needs to know that *isn't* in
/// the events themselves. Cheap to assemble from the coordinator's
/// store between heartbeat ticks; passing it explicitly keeps the
/// renderer pure and trivial to snapshot-test.
#[derive(Debug, Clone)]
pub struct HeartbeatProgress {
    pub topic: String,
    pub started_at: DateTime<Utc>,
    pub findings_total: u32,
    /// Number of findings already in the store when the run kicked
    /// off. Used to compute "saved this run" without a second store
    /// round-trip.
    pub findings_baseline: u32,
    pub iteration_estimate: Option<u32>,
    pub iteration_cap: u32,
}

impl HeartbeatProgress {
    fn elapsed_label(&self, now: DateTime<Utc>) -> String {
        let secs = (now - self.started_at).num_seconds().max(0) as u64;
        format_duration(secs)
    }

    fn saved_this_run(&self) -> u32 {
        self.findings_total.saturating_sub(self.findings_baseline)
    }
}

/// Render a waterfall heartbeat. Output is plain text (Telegram
/// `text` mode — no parse_mode), intentionally — the waterfall is a
/// log, not formatted prose, and Telegram's HTML parser does not play
/// well with URLs in list items. All output is pre-escaped enough to
/// be safe to render without HTML / Markdown parsing.
pub fn render_waterfall(
    spec_id: &str,
    progress: &HeartbeatProgress,
    events: &[RunEvent],
    now: DateTime<Utc>,
) -> String {
    let mut out = String::with_capacity(256 + 96 * events.len());
    let topic_preview: String = progress
        .topic
        .chars()
        .take(80)
        .collect::<String>()
        .trim()
        .to_string();
    out.push_str("🔎 ");
    out.push_str(&topic_preview);
    out.push_str("  (");
    out.push_str(&progress.elapsed_label(now));
    out.push_str(")\n");

    out.push_str("spec ");
    out.push_str(spec_id);
    out.push_str(" · saved ");
    out.push_str(&progress.saved_this_run().to_string());
    out.push_str(" / total ");
    out.push_str(&progress.findings_total.to_string());
    if let Some(it) = progress.iteration_estimate {
        out.push_str(&format!(" · iter ≈{}/{}", it, progress.iteration_cap));
    } else {
        out.push_str(&format!(" · cap {}", progress.iteration_cap));
    }
    out.push_str("\n\n");

    if events.is_empty() {
        out.push_str("· waiting for first tool call…\n");
    } else {
        let tail: Vec<&RunEvent> = events.iter().rev().take(WATERFALL_EVENTS).collect();
        for ev in tail.into_iter().rev() {
            render_event_line(ev, &mut out);
        }
    }

    out.push_str("\nTap ⏸ Stop & clarify to pause and send a note.");
    out
}

fn render_event_line(ev: &RunEvent, out: &mut String) {
    let t = ev.at.format("%H:%M:%S");
    let (icon, verb) = match ev.kind {
        EventKind::IterationStart => ("·", "iter"),
        EventKind::ToolCallStart => ("▸", "start"),
        EventKind::ToolCallEnd => ("◂", "done"),
        EventKind::FindingSaved => ("★", "saved"),
        EventKind::BlockDetected => ("⚠", "blocked"),
        EventKind::SkillLoaded => ("🎒", "skill"),
        EventKind::Note => ("·", "note"),
    };
    // Clamp label so a pathological URL can't blow the 4096 budget.
    let label: String = ev.label.chars().take(120).collect();
    out.push_str(&format!("{icon} {t}  {verb:<6}  {label}\n"));
}

fn format_duration(secs: u64) -> String {
    if secs < 60 {
        format!("{secs}s")
    } else if secs < 3600 {
        let m = secs / 60;
        let s = secs % 60;
        if s == 0 {
            format!("{m}m")
        } else {
            format!("{m}m {s:02}s")
        }
    } else {
        let h = secs / 3600;
        let m = (secs % 3600) / 60;
        if m == 0 {
            format!("{h}h")
        } else {
            format!("{h}h {m:02}m")
        }
    }
}

/// Pending clarification state per chat. When the operator taps "Stop
/// & clarify" we stash the spec id here and wait for the next user
/// message in that chat/thread to append to the spec's topic.
///
/// The key is `(chat_id, thread_id)` — groups with multiple threads
/// legitimately run one research per thread, so clarifications must
/// scope to the conversation that initiated the pause. `i64` + raw
/// `i32` keys keep the map trivially hashable without threading
/// teloxide types into the public API.
#[derive(Debug, Clone)]
pub struct PendingClarification {
    pub spec_id: String,
    pub message_id: teloxide::types::MessageId,
    pub paused_at: DateTime<Utc>,
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::TimeZone;

    fn make_progress(topic: &str, baseline: u32, total: u32) -> HeartbeatProgress {
        HeartbeatProgress {
            topic: topic.to_string(),
            started_at: Utc.with_ymd_and_hms(2026, 4, 20, 16, 30, 0).unwrap(),
            findings_total: total,
            findings_baseline: baseline,
            iteration_estimate: None,
            iteration_cap: 30,
        }
    }

    #[test]
    fn waterfall_renders_topic_and_progress_without_events() {
        let progress = make_progress("restaurants in Da Nang", 0, 0);
        let now = Utc.with_ymd_and_hms(2026, 4, 20, 16, 32, 15).unwrap();
        let out = render_waterfall("abc123", &progress, &[], now);
        assert!(out.contains("🔎 restaurants in Da Nang"));
        assert!(out.contains("(2m 15s)"));
        assert!(out.contains("spec abc123"));
        assert!(out.contains("saved 0 / total 0"));
        assert!(out.contains("waiting for first tool call"));
    }

    #[test]
    fn waterfall_limits_to_last_five_events() {
        let progress = make_progress("shops", 2, 5);
        let now = Utc.with_ymd_and_hms(2026, 4, 20, 16, 40, 0).unwrap();
        let events: Vec<RunEvent> = (0..10)
            .map(|i| RunEvent::new(EventKind::ToolCallStart, format!("web_fetch #{i}")))
            .collect();
        let out = render_waterfall("s-1", &progress, &events, now);
        // Only the last 5 labels should appear.
        assert!(out.contains("#5"));
        assert!(out.contains("#6"));
        assert!(out.contains("#9"));
        assert!(!out.contains("#0"));
        assert!(!out.contains("#4"));
    }

    #[test]
    fn waterfall_reports_saved_delta() {
        let progress = make_progress("x", 3, 9);
        let now = Utc.with_ymd_and_hms(2026, 4, 20, 16, 31, 0).unwrap();
        let out = render_waterfall("s", &progress, &[], now);
        assert!(out.contains("saved 6 / total 9"));
    }

    #[test]
    fn format_duration_covers_boundaries() {
        assert_eq!(format_duration(0), "0s");
        assert_eq!(format_duration(59), "59s");
        assert_eq!(format_duration(60), "1m");
        assert_eq!(format_duration(75), "1m 15s");
        assert_eq!(format_duration(3600), "1h");
        assert_eq!(format_duration(3725), "1h 02m");
    }

    #[test]
    fn keyboard_stop_has_single_button_with_stable_prefix() {
        let kb = keyboard_stop("abc");
        assert_eq!(kb.inline_keyboard.len(), 1);
        assert_eq!(kb.inline_keyboard[0].len(), 1);
        let btn = &kb.inline_keyboard[0][0];
        assert_eq!(btn.text, "⏸ Stop & clarify");
        if let teloxide::types::InlineKeyboardButtonKind::CallbackData(data) = &btn.kind {
            assert_eq!(data, "r:stop:abc");
        } else {
            panic!("expected callback data");
        }
    }
}
