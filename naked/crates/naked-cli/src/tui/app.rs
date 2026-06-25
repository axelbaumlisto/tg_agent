#![allow(dead_code)] // U3 lands pure state before U4/U5 wire it into loop/view.

use std::collections::VecDeque;

use crate::event_map::TranscriptUpdate;
use crossterm::event::{KeyCode, KeyEvent, KeyModifiers};
use naked_core::types::{Permission, TurnUsage};

const SCROLL_PAGE: u16 = 10;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum RunState {
    Idle,
    Streaming,
    Aborting,
    AwaitingPermission,
}

pub(crate) struct PendingPermission {
    pub call_id: String,
    pub tool_name: String,
    pub permission: Permission,
}

#[derive(Default)]
pub(crate) struct Status {
    pub model: String,
    pub usage: Option<TurnUsage>,
}

pub(crate) struct App {
    pub transcript: VecDeque<String>,
    pub transcript_cap: usize,
    pub input: String,
    pub scroll: u16,
    pub status: Status,
    pub run_state: RunState,
    pub pending_permission: Option<PendingPermission>,
    pub should_quit: bool,
    pub current_line: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum KeyOutcome {
    None,
    SendPrompt(String),
    Quit,
    AbortTurn,
    PermissionResponse { call_id: String, allowed: bool },
    HardQuit,
}

impl App {
    pub(crate) fn new(model: String, transcript_cap: usize) -> Self {
        Self {
            transcript: VecDeque::new(),
            transcript_cap,
            input: String::new(),
            scroll: 0,
            status: Status { model, usage: None },
            run_state: RunState::Idle,
            pending_permission: None,
            should_quit: false,
            current_line: String::new(),
        }
    }

    pub(crate) fn push_line(&mut self, line: String) {
        if self.transcript_cap == 0 {
            return;
        }

        if self.transcript.len() == self.transcript_cap {
            self.transcript.pop_front();
        }
        self.transcript.push_back(line);
    }

    pub(crate) fn apply(&mut self, update: TranscriptUpdate) {
        match update {
            TranscriptUpdate::AppendText(text) => {
                self.mark_streaming();
                self.append_stream_text(&text);
            }
            TranscriptUpdate::AppendThinking(text) => {
                self.mark_streaming();
                self.append_stream_text(&text);
            }
            TranscriptUpdate::ToolStarted {
                call_id,
                name,
                input,
            } => {
                self.mark_streaming();
                self.push_event_line(format!("tool {name} started ({call_id}): {input}"));
            }
            TranscriptUpdate::ToolEnded {
                call_id,
                name,
                ok,
                output,
            } => {
                self.mark_streaming();
                let state = if ok { "ok" } else { "error" };
                self.push_event_line(format!("tool {name} ended ({call_id}) [{state}]: {output}"));
            }
            TranscriptUpdate::Permission {
                call_id,
                tool_name,
                permission,
                input: _,
            } => {
                self.flush_current_line();
                self.run_state = RunState::AwaitingPermission;
                self.pending_permission = Some(PendingPermission {
                    call_id,
                    tool_name,
                    permission,
                });
            }
            TranscriptUpdate::SubAgent { agent_id, event } => {
                self.mark_streaming();
                self.push_event_line(format!("sub-agent {agent_id}: {event:?}"));
            }
            TranscriptUpdate::ContextCompacted {
                before_msgs,
                after_msgs,
                summary_hint,
                files_count,
            } => {
                self.mark_streaming();
                let hint = summary_hint.unwrap_or_default();
                self.push_event_line(format!(
                    "context compacted: {before_msgs}->{after_msgs} messages, {files_count} files {hint}"
                ));
            }
            TranscriptUpdate::CycleRestarted {
                cycle_number,
                archived_messages,
                archive_path,
            } => {
                self.mark_streaming();
                self.push_event_line(format!(
                    "cycle {cycle_number} restarted: archived {archived_messages} messages to {archive_path}"
                ));
            }
            TranscriptUpdate::ToolOutput { call_id, chunk } => {
                self.mark_streaming();
                self.push_event_line(format!("tool output ({call_id}): {chunk}"));
            }
            TranscriptUpdate::SteerReceived { text } => {
                self.mark_streaming();
                self.push_event_line(format!("steer received: {text}"));
            }
            TranscriptUpdate::Usage(usage) => {
                self.status.usage = Some(usage);
            }
            TranscriptUpdate::Error(message) => {
                self.mark_streaming();
                self.push_event_line(format!("error: {message}"));
            }
            TranscriptUpdate::TurnDone => {
                self.flush_current_line();
                self.pending_permission = None;
                self.run_state = RunState::Idle;
            }
            TranscriptUpdate::Heartbeat => {}
        }
    }

    pub(crate) fn on_key(&mut self, key: KeyEvent) -> KeyOutcome {
        if matches!(key.code, KeyCode::PageUp | KeyCode::PageDown | KeyCode::End) {
            return self.on_scroll_key(key.code);
        }

        if self.run_state == RunState::AwaitingPermission {
            return self.on_permission_key(key);
        }

        if key.modifiers.contains(KeyModifiers::CONTROL) {
            match key.code {
                KeyCode::Char('c') | KeyCode::Char('C') => return self.on_ctrl_c(),
                KeyCode::Char('d') | KeyCode::Char('D') => {
                    if self.run_state == RunState::Idle && self.input.is_empty() {
                        self.should_quit = true;
                        return KeyOutcome::Quit;
                    }
                    return KeyOutcome::None;
                }
                _ => {}
            }
        }

        match key.code {
            KeyCode::Enter if key.modifiers.contains(KeyModifiers::ALT) => {
                self.input.push('\n');
                KeyOutcome::None
            }
            KeyCode::Enter => self.on_enter(),
            KeyCode::Esc => {
                if self.run_state == RunState::Idle && self.input.is_empty() {
                    self.should_quit = true;
                    KeyOutcome::Quit
                } else {
                    KeyOutcome::None
                }
            }
            KeyCode::Backspace if self.input_allowed() => {
                self.input.pop();
                KeyOutcome::None
            }
            KeyCode::Char(c) if self.input_allowed() => {
                self.input.push(c);
                KeyOutcome::None
            }
            _ => KeyOutcome::None,
        }
    }

    fn append_stream_text(&mut self, text: &str) {
        for segment in text.split_inclusive('\n') {
            self.current_line.push_str(segment.trim_end_matches('\n'));
            if segment.ends_with('\n') {
                self.flush_current_line();
            }
        }
    }

    fn flush_current_line(&mut self) {
        if self.current_line.is_empty() {
            return;
        }

        let line = std::mem::take(&mut self.current_line);
        self.push_line(line);
    }

    fn push_event_line(&mut self, line: String) {
        self.flush_current_line();
        self.push_line(line);
    }

    fn mark_streaming(&mut self) {
        if self.run_state == RunState::Idle {
            self.run_state = RunState::Streaming;
        }
    }

    fn input_allowed(&self) -> bool {
        matches!(self.run_state, RunState::Idle | RunState::Streaming)
    }

    fn on_enter(&mut self) -> KeyOutcome {
        if self.run_state != RunState::Idle || self.input.trim().is_empty() {
            return KeyOutcome::None;
        }

        let prompt = std::mem::take(&mut self.input);
        self.run_state = RunState::Streaming;
        KeyOutcome::SendPrompt(prompt)
    }

    fn on_ctrl_c(&mut self) -> KeyOutcome {
        match self.run_state {
            RunState::Streaming => {
                self.run_state = RunState::Aborting;
                KeyOutcome::AbortTurn
            }
            RunState::Aborting => {
                self.should_quit = true;
                KeyOutcome::HardQuit
            }
            RunState::Idle => {
                self.should_quit = true;
                KeyOutcome::Quit
            }
            RunState::AwaitingPermission => KeyOutcome::None,
        }
    }

    fn on_permission_key(&mut self, key: KeyEvent) -> KeyOutcome {
        // DEFAULT-SAFE: a MODIFIED key (Ctrl+Y / Alt+Y / etc.) must NEVER approve
        // — only an unmodified y/Y/n/N (or Esc) counts. Esc may carry no
        // modifiers either. Anything else → None (re-store pending below).
        let unmodified = !key
            .modifiers
            .intersects(KeyModifiers::CONTROL | KeyModifiers::ALT | KeyModifiers::SUPER);
        let allowed = match key.code {
            KeyCode::Char('y') | KeyCode::Char('Y') if unmodified => Some(true),
            KeyCode::Char('n') | KeyCode::Char('N') if unmodified => Some(false),
            KeyCode::Esc => Some(false),
            _ => None,
        };

        match (allowed, self.pending_permission.take()) {
            (Some(allowed), Some(pending)) => {
                self.run_state = RunState::Streaming;
                KeyOutcome::PermissionResponse {
                    call_id: pending.call_id,
                    allowed,
                }
            }
            (Some(_), None) => KeyOutcome::None,
            (None, pending) => {
                self.pending_permission = pending;
                KeyOutcome::None
            }
        }
    }

    fn on_scroll_key(&mut self, code: KeyCode) -> KeyOutcome {
        match code {
            KeyCode::PageUp => self.scroll = self.scroll.saturating_add(SCROLL_PAGE),
            KeyCode::PageDown => self.scroll = self.scroll.saturating_sub(SCROLL_PAGE),
            KeyCode::End => self.scroll = 0,
            _ => {}
        }
        KeyOutcome::None
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crossterm::event::KeyEventKind;
    use serde_json::json;

    fn app() -> App {
        App::new("test-model".into(), 5_000)
    }

    fn key(code: KeyCode) -> KeyEvent {
        key_with_modifiers(code, KeyModifiers::NONE)
    }

    fn key_with_modifiers(code: KeyCode, modifiers: KeyModifiers) -> KeyEvent {
        KeyEvent {
            code,
            modifiers,
            kind: KeyEventKind::Press,
            state: crossterm::event::KeyEventState::NONE,
        }
    }

    fn permission_update(call_id: &str) -> TranscriptUpdate {
        TranscriptUpdate::Permission {
            call_id: call_id.into(),
            tool_name: "edit_file".into(),
            permission: Permission::WorkspaceWrite,
            input: json!({"path": "src/lib.rs"}),
        }
    }

    fn usage() -> TurnUsage {
        TurnUsage {
            input_tokens: 1,
            output_tokens: 2,
            cache_read_tokens: 3,
            cache_write_tokens: 4,
        }
    }

    #[test]
    fn app_apply_text_appends_to_transcript() {
        let mut app = app();

        app.apply(TranscriptUpdate::AppendText("hello\n".into()));

        assert_eq!(app.transcript.len(), 1);
        assert_eq!(app.transcript[0], "hello");
        assert!(app.current_line.is_empty());
    }

    #[test]
    fn app_text_deltas_coalesce_into_current_streaming_line() {
        let mut app = app();

        app.apply(TranscriptUpdate::AppendText("hel".into()));
        app.apply(TranscriptUpdate::AppendText("lo".into()));
        app.apply(TranscriptUpdate::AppendText(" world".into()));

        assert!(app.transcript.is_empty());
        assert_eq!(app.current_line, "hello world");
        app.apply(TranscriptUpdate::TurnDone);
        assert_eq!(app.transcript.len(), 1);
        assert_eq!(app.transcript[0], "hello world");
    }

    #[test]
    fn app_transcript_ring_evicts_oldest_and_keeps_newest() {
        let mut app = App::new("test-model".into(), 3);

        for i in 0..5 {
            app.push_line(format!("line-{i}"));
        }

        assert_eq!(app.transcript.len(), 3);
        assert_eq!(app.transcript[0], "line-2");
        assert_eq!(app.transcript[2], "line-4");
    }

    #[test]
    fn app_usage_update_updates_status() {
        let mut app = app();
        let usage = usage();

        app.apply(TranscriptUpdate::Usage(usage.clone()));

        assert_eq!(app.status.usage, Some(usage));
    }

    #[test]
    fn app_turn_done_returns_to_idle_and_reenables_input() {
        let mut app = app();
        app.run_state = RunState::Streaming;
        app.current_line = "done".into();

        app.apply(TranscriptUpdate::TurnDone);
        let outcome = app.on_key(key(KeyCode::Enter));
        assert_eq!(outcome, KeyOutcome::None);
        app.input = "next prompt".into();
        let outcome = app.on_key(key(KeyCode::Enter));

        assert_eq!(app.run_state, RunState::Streaming);
        assert_eq!(outcome, KeyOutcome::SendPrompt("next prompt".into()));
    }

    #[test]
    fn app_turn_done_clears_pending_permission() {
        let mut app = app();
        app.apply(permission_update("call-1"));

        app.apply(TranscriptUpdate::TurnDone);

        assert_eq!(app.run_state, RunState::Idle);
        assert!(app.pending_permission.is_none());
    }

    #[test]
    fn app_permission_request_sets_awaiting_state() {
        let mut app = app();

        app.apply(permission_update("call-1"));

        assert_eq!(app.run_state, RunState::AwaitingPermission);
        let pending = app.pending_permission.as_ref().expect("pending permission");
        assert_eq!(pending.call_id, "call-1");
        assert_eq!(pending.tool_name, "edit_file");
        assert_eq!(pending.permission, Permission::WorkspaceWrite);
    }

    #[test]
    fn permission_key_y_allows_and_clears_pending() {
        let mut app = app();
        app.apply(permission_update("call-1"));

        let outcome = app.on_key(key(KeyCode::Char('y')));

        assert_eq!(
            outcome,
            KeyOutcome::PermissionResponse {
                call_id: "call-1".into(),
                allowed: true,
            }
        );
        assert!(app.pending_permission.is_none());
        assert_eq!(app.run_state, RunState::Streaming);
    }

    #[test]
    fn permission_key_n_denies_and_clears_pending() {
        let mut app = app();
        app.apply(permission_update("call-1"));

        let outcome = app.on_key(key(KeyCode::Char('n')));

        assert_eq!(
            outcome,
            KeyOutcome::PermissionResponse {
                call_id: "call-1".into(),
                allowed: false,
            }
        );
        assert!(app.pending_permission.is_none());
        assert_eq!(app.run_state, RunState::Streaming);
    }

    #[test]
    fn permission_default_safe_enter_and_stray_keys_do_not_allow() {
        let mut app = app();
        app.apply(permission_update("call-1"));

        assert_eq!(app.on_key(key(KeyCode::Enter)), KeyOutcome::None);
        assert!(app.pending_permission.is_some());
        assert_eq!(app.on_key(key(KeyCode::Char('x'))), KeyOutcome::None);
        assert!(app.pending_permission.is_some());
        // MODIFIED y/Y must NEVER approve (Ctrl+Y / Alt+Y are not consent).
        assert_eq!(
            app.on_key(key_with_modifiers(
                KeyCode::Char('y'),
                KeyModifiers::CONTROL
            )),
            KeyOutcome::None
        );
        assert!(app.pending_permission.is_some());
        assert_eq!(
            app.on_key(key_with_modifiers(KeyCode::Char('Y'), KeyModifiers::ALT)),
            KeyOutcome::None
        );
        assert!(app.pending_permission.is_some());
        let outcome = app.on_key(key(KeyCode::Esc));

        assert_eq!(
            outcome,
            KeyOutcome::PermissionResponse {
                call_id: "call-1".into(),
                allowed: false,
            }
        );
    }

    #[test]
    fn enter_sends_prompt_only_when_idle_and_non_empty() {
        let mut app = app();

        assert_eq!(app.on_key(key(KeyCode::Enter)), KeyOutcome::None);
        app.input = "   ".into();
        assert_eq!(app.on_key(key(KeyCode::Enter)), KeyOutcome::None);
        app.input = "hello".into();
        assert_eq!(
            app.on_key(key(KeyCode::Enter)),
            KeyOutcome::SendPrompt("hello".into())
        );
        assert!(app.input.is_empty());
        assert_eq!(app.run_state, RunState::Streaming);

        for state in [
            RunState::Streaming,
            RunState::Aborting,
            RunState::AwaitingPermission,
        ] {
            app.run_state = state;
            app.input = "blocked".into();
            assert_eq!(app.on_key(key(KeyCode::Enter)), KeyOutcome::None);
            assert_eq!(app.input, "blocked");
        }
    }

    #[test]
    fn multiline_key_inserts_newline_without_sending() {
        let mut app = app();
        app.input = "first".into();

        let outcome = app.on_key(key_with_modifiers(KeyCode::Enter, KeyModifiers::ALT));

        assert_eq!(outcome, KeyOutcome::None);
        assert_eq!(app.input, "first\n");
        assert_eq!(app.run_state, RunState::Idle);
    }

    #[test]
    fn scroll_keys_update_scroll_without_mutating_input_or_permission() {
        let mut app = app();
        app.input = "keep".into();
        app.apply(permission_update("call-1"));

        assert_eq!(app.on_key(key(KeyCode::PageUp)), KeyOutcome::None);
        assert_eq!(app.scroll, SCROLL_PAGE);
        assert_eq!(app.on_key(key(KeyCode::PageDown)), KeyOutcome::None);
        assert_eq!(app.scroll, 0);
        app.scroll = 5;
        assert_eq!(app.on_key(key(KeyCode::End)), KeyOutcome::None);

        assert_eq!(app.scroll, 0);
        assert_eq!(app.input, "keep");
        assert!(app.pending_permission.is_some());
        assert_eq!(app.run_state, RunState::AwaitingPermission);
    }

    #[test]
    fn ctrl_c_streaming_enters_aborting_and_cancels_token() {
        let mut app = app();
        app.run_state = RunState::Streaming;

        let outcome = app.on_key(key_with_modifiers(
            KeyCode::Char('c'),
            KeyModifiers::CONTROL,
        ));

        assert_eq!(outcome, KeyOutcome::AbortTurn);
        assert_eq!(app.run_state, RunState::Aborting);
    }

    #[test]
    fn ctrl_c_aborting_is_idempotent_no_second_send() {
        let mut app = app();
        app.run_state = RunState::Aborting;

        let outcome = app.on_key(key_with_modifiers(
            KeyCode::Char('c'),
            KeyModifiers::CONTROL,
        ));

        assert_eq!(outcome, KeyOutcome::HardQuit);
        assert_eq!(app.run_state, RunState::Aborting);
    }
}
