//! Pure ratatui renderer for the naked TUI.

use ratatui::layout::{Constraint, Layout};
use ratatui::widgets::{Block, Borders, Paragraph, Wrap};

use crate::tui::app::{App, RunState};

pub(crate) fn render(frame: &mut ratatui::Frame<'_>, app: &App) {
    let [transcript_area, status_area, input_area] = Layout::vertical([
        Constraint::Min(1),
        Constraint::Length(1),
        Constraint::Length(3),
    ])
    .areas(frame.area());

    let transcript = transcript_text(app);
    frame.render_widget(
        Paragraph::new(transcript)
            .block(
                Block::new()
                    .title(format!("naked tui · {}", app.status.model))
                    .borders(Borders::ALL),
            )
            .scroll((app.scroll, 0))
            .wrap(Wrap { trim: false }),
        transcript_area,
    );

    frame.render_widget(Paragraph::new(status_line_text(app)), status_area);

    frame.render_widget(
        Paragraph::new(format!("> {}▌", app.input))
            .block(
                Block::new()
                    .title("Input · Enter send · Alt+Enter newline")
                    .borders(Borders::ALL),
            )
            .wrap(Wrap { trim: false }),
        input_area,
    );
}

pub(crate) fn status_line_text(app: &App) -> String {
    let mut parts = vec![run_state_label(app.run_state).to_string()];

    if let Some(usage) = &app.status.usage {
        parts.push(format!(
            "tok: {}↑/{}↓",
            usage.input_tokens, usage.output_tokens
        ));
    }

    if app.run_state == RunState::AwaitingPermission {
        if let Some(pending) = &app.pending_permission {
            parts.push(format!("tool: {}", pending.tool_name));
        }
        parts.push("y approve / n deny".to_string());
    }

    parts.join(" · ")
}

fn transcript_text(app: &App) -> String {
    let mut lines: Vec<String> = app.transcript.iter().cloned().collect();
    if !app.current_line.is_empty() {
        lines.push(app.current_line.clone());
    }
    lines.join("\n")
}

fn run_state_label(state: RunState) -> &'static str {
    match state {
        RunState::Idle => "Idle",
        RunState::Streaming => "Streaming",
        RunState::Aborting => "Aborting",
        RunState::AwaitingPermission => "awaiting-permission",
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::event_map::TranscriptUpdate;
    use naked_core::types::{Permission, TurnUsage};
    use serde_json::json;

    fn app() -> App {
        App::new("provider/model".into(), 5_000)
    }

    #[test]
    fn view_status_line_reflects_run_state() {
        let mut app = app();
        assert_eq!(status_line_text(&app), "Idle");

        app.run_state = RunState::Aborting;
        assert_eq!(status_line_text(&app), "Aborting");

        app.run_state = RunState::Streaming;
        app.status.usage = Some(TurnUsage {
            input_tokens: 123,
            output_tokens: 456,
            cache_read_tokens: 0,
            cache_write_tokens: 0,
        });
        assert_eq!(status_line_text(&app), "Streaming · tok: 123↑/456↓");

        app.apply(TranscriptUpdate::Permission {
            call_id: "call-1".into(),
            tool_name: "edit_file".into(),
            permission: Permission::WorkspaceWrite,
            input: json!({"path": "src/lib.rs"}),
        });
        assert_eq!(
            status_line_text(&app),
            "awaiting-permission · tok: 123↑/456↓ · tool: edit_file · y approve / n deny"
        );
    }
}
