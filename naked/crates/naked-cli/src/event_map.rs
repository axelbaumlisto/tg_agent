//! Shared semantic event mapping for CLI frontends.
//!
//! This module intentionally contains display data only: no ANSI, stdin/stdout,
//! crossterm, or ratatui types. The REPL and future TUI own their own IO.

use naked_core::types::{AgentEvent, SubAgentEvent, ToolState, TurnUsage};

pub(crate) enum TranscriptUpdate {
    AppendText(String),
    AppendThinking(String),
    ToolStarted {
        call_id: String,
        name: String,
        input: serde_json::Value,
    },
    ToolEnded {
        call_id: String,
        name: String,
        ok: bool,
        output: String,
    },
    Permission {
        call_id: String,
        tool_name: String,
        permission: naked_core::types::Permission,
        input: serde_json::Value,
    },
    SubAgent {
        agent_id: String,
        event: SubAgentEvent,
    },
    ContextCompacted {
        before_msgs: usize,
        after_msgs: usize,
        summary_hint: Option<String>,
        files_count: usize,
    },
    CycleRestarted {
        cycle_number: u32,
        archived_messages: usize,
        archive_path: String,
    },
    ToolOutput {
        call_id: String,
        chunk: String,
    },
    SteerReceived {
        text: String,
    },
    Usage(TurnUsage),
    Error(String),
    TurnDone,
    Heartbeat,
}

pub(crate) fn map_event(event: AgentEvent) -> TranscriptUpdate {
    // Compile-time exhaustiveness guard: this match deliberately has no `_` arm.
    // Adding a new AgentEvent variant must fail here until every frontend has
    // semantic display data for it.
    match event {
        AgentEvent::TextDelta(t) => TranscriptUpdate::AppendText(t),
        AgentEvent::ThinkingDelta(t) => TranscriptUpdate::AppendThinking(t),
        AgentEvent::ToolStart {
            call_id,
            name,
            input,
        } => TranscriptUpdate::ToolStarted {
            call_id,
            name,
            input,
        },
        AgentEvent::ToolEnd {
            call_id,
            name,
            state,
            output,
        } => TranscriptUpdate::ToolEnded {
            call_id,
            name,
            ok: matches!(state, ToolState::Completed),
            output,
        },
        AgentEvent::PermissionRequest {
            call_id,
            tool_name,
            input,
            permission,
        } => TranscriptUpdate::Permission {
            call_id,
            tool_name,
            permission,
            input,
        },
        AgentEvent::Heartbeat => TranscriptUpdate::Heartbeat,
        AgentEvent::SubAgentProgress { agent_id, event } => {
            TranscriptUpdate::SubAgent { agent_id, event }
        }
        AgentEvent::UsageUpdate(usage) => TranscriptUpdate::Usage(usage),
        AgentEvent::ContextCompacted {
            before_msgs,
            after_msgs,
            summary_hint,
            files_count,
        } => TranscriptUpdate::ContextCompacted {
            before_msgs,
            after_msgs,
            summary_hint,
            files_count,
        },
        AgentEvent::CycleRestarted {
            cycle_number,
            archived_messages,
            archive_path,
        } => TranscriptUpdate::CycleRestarted {
            cycle_number,
            archived_messages,
            archive_path,
        },
        AgentEvent::ToolOutput { call_id, chunk } => {
            TranscriptUpdate::ToolOutput { call_id, chunk }
        }
        AgentEvent::SteerReceived { text, msg_ids: _ } => TranscriptUpdate::SteerReceived { text },
        AgentEvent::Error(e) => TranscriptUpdate::Error(e),
        AgentEvent::Idle => TranscriptUpdate::TurnDone,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use naked_core::types::{AgentEvent, Permission, SubAgentEvent, ToolState, TurnUsage};
    use serde_json::json;

    #[test]
    fn event_map_text_delta_appends() {
        match map_event(AgentEvent::TextDelta("hi".into())) {
            TranscriptUpdate::AppendText(text) => assert_eq!(text, "hi"),
            _ => panic!("TextDelta did not map to AppendText"),
        }

        match map_event(AgentEvent::ThinkingDelta("x".into())) {
            TranscriptUpdate::AppendThinking(text) => assert_eq!(text, "x"),
            _ => panic!("ThinkingDelta did not map to AppendThinking"),
        }
    }

    #[test]
    fn event_map_tool_start_end_roundtrip() {
        let input = json!({"cmd": "echo hi"});
        match map_event(AgentEvent::ToolStart {
            call_id: "call-1".into(),
            name: "bash".into(),
            input: input.clone(),
        }) {
            TranscriptUpdate::ToolStarted {
                call_id,
                name,
                input: mapped_input,
            } => {
                assert_eq!(call_id, "call-1");
                assert_eq!(name, "bash");
                assert_eq!(mapped_input, input);
            }
            _ => panic!("ToolStart did not map to ToolStarted"),
        }

        match map_event(AgentEvent::ToolEnd {
            call_id: "call-1".into(),
            name: "bash".into(),
            state: ToolState::Completed,
            output: "ok".into(),
        }) {
            TranscriptUpdate::ToolEnded {
                call_id,
                name,
                ok,
                output,
            } => {
                assert_eq!(call_id, "call-1");
                assert_eq!(name, "bash");
                assert!(ok);
                assert_eq!(output, "ok");
            }
            _ => panic!("completed ToolEnd did not map to ToolEnded"),
        }

        match map_event(AgentEvent::ToolEnd {
            call_id: "call-2".into(),
            name: "bash".into(),
            state: ToolState::Error,
            output: "boom".into(),
        }) {
            TranscriptUpdate::ToolEnded { ok, output, .. } => {
                assert!(!ok);
                assert_eq!(output, "boom");
            }
            _ => panic!("error ToolEnd did not map to ToolEnded"),
        }
    }

    #[test]
    fn event_map_idle_and_error_semantics() {
        match map_event(AgentEvent::Idle) {
            TranscriptUpdate::TurnDone => {}
            _ => panic!("Idle did not map to TurnDone"),
        }

        match map_event(AgentEvent::Error("boom".into())) {
            TranscriptUpdate::Error(message) => assert_eq!(message, "boom"),
            _ => panic!("Error did not map to Error"),
        }
    }

    #[test]
    fn event_map_maps_usage_context_cycle_tool_output_steer_subagent() {
        let usage = TurnUsage {
            input_tokens: 1,
            output_tokens: 2,
            cache_read_tokens: 3,
            cache_write_tokens: 4,
        };
        match map_event(AgentEvent::UsageUpdate(usage.clone())) {
            TranscriptUpdate::Usage(mapped) => assert_eq!(mapped, usage),
            _ => panic!("UsageUpdate did not map to Usage"),
        }

        match map_event(AgentEvent::ContextCompacted {
            before_msgs: 10,
            after_msgs: 3,
            summary_hint: Some("goal".into()),
            files_count: 2,
        }) {
            TranscriptUpdate::ContextCompacted {
                before_msgs,
                after_msgs,
                summary_hint,
                files_count,
            } => {
                assert_eq!(before_msgs, 10);
                assert_eq!(after_msgs, 3);
                assert_eq!(summary_hint.as_deref(), Some("goal"));
                assert_eq!(files_count, 2);
            }
            _ => panic!("ContextCompacted did not map"),
        }

        match map_event(AgentEvent::CycleRestarted {
            cycle_number: 7,
            archived_messages: 42,
            archive_path: "/tmp/archive.json".into(),
        }) {
            TranscriptUpdate::CycleRestarted {
                cycle_number,
                archived_messages,
                archive_path,
            } => {
                assert_eq!(cycle_number, 7);
                assert_eq!(archived_messages, 42);
                assert_eq!(archive_path, "/tmp/archive.json");
            }
            _ => panic!("CycleRestarted did not map"),
        }

        match map_event(AgentEvent::ToolOutput {
            call_id: "call-3".into(),
            chunk: "line".into(),
        }) {
            TranscriptUpdate::ToolOutput { call_id, chunk } => {
                assert_eq!(call_id, "call-3");
                assert_eq!(chunk, "line");
            }
            _ => panic!("ToolOutput did not map"),
        }

        match map_event(AgentEvent::SteerReceived {
            text: "adjust".into(),
            msg_ids: vec![1, 2],
        }) {
            TranscriptUpdate::SteerReceived { text } => assert_eq!(text, "adjust"),
            _ => panic!("SteerReceived did not map"),
        }

        let sub_event = SubAgentEvent::Finished { tokens: 99 };
        match map_event(AgentEvent::SubAgentProgress {
            agent_id: "agent-a".into(),
            event: sub_event,
        }) {
            TranscriptUpdate::SubAgent { agent_id, event } => {
                assert_eq!(agent_id, "agent-a");
                match event {
                    SubAgentEvent::Finished { tokens } => assert_eq!(tokens, 99),
                    _ => panic!("SubAgentProgress changed inner event"),
                }
            }
            _ => panic!("SubAgentProgress did not map"),
        }

        match map_event(AgentEvent::Heartbeat) {
            TranscriptUpdate::Heartbeat => {}
            _ => panic!("Heartbeat did not map"),
        }

        match map_event(AgentEvent::PermissionRequest {
            call_id: "call-4".into(),
            tool_name: "edit_file".into(),
            input: json!({"path": "x"}),
            permission: Permission::WorkspaceWrite,
        }) {
            TranscriptUpdate::Permission {
                call_id,
                tool_name,
                permission,
                input,
            } => {
                assert_eq!(call_id, "call-4");
                assert_eq!(tool_name, "edit_file");
                assert!(matches!(permission, Permission::WorkspaceWrite));
                assert_eq!(input, json!({"path": "x"}));
            }
            _ => panic!("PermissionRequest did not map"),
        }
    }

    #[test]
    fn chat_repl_uses_shared_event_map_no_inline_agent_event_match() {
        // Strip `SubAgentEvent::` first: rendering the INNER sub-agent event
        // (SubAgentEvent::Started/ToolUse/...) is legitimate REPL IO that stays
        // in chat.rs, and "SubAgentEvent::" contains the "AgentEvent::" substring.
        let source = include_str!("commands/chat.rs").replace("SubAgentEvent", "");
        assert!(source.contains("map_event("));
        // DRY guard: chat.rs must NOT re-introduce any inline AgentEvent::<Variant>
        // arm (a partial revert of even one variant would re-diverge from the
        // shared mapper). Check ALL 14 variants, not just one.
        for variant in [
            "AgentEvent::TextDelta",
            "AgentEvent::ThinkingDelta",
            "AgentEvent::ToolStart",
            "AgentEvent::ToolEnd",
            "AgentEvent::PermissionRequest",
            "AgentEvent::Heartbeat",
            "AgentEvent::SubAgentProgress",
            "AgentEvent::UsageUpdate",
            "AgentEvent::ContextCompacted",
            "AgentEvent::CycleRestarted",
            "AgentEvent::ToolOutput",
            "AgentEvent::SteerReceived",
            "AgentEvent::Error",
            "AgentEvent::Idle",
        ] {
            assert!(
                !source.contains(variant),
                "chat.rs re-introduced inline `{variant}` arm — must route through \
                 event_map::map_event instead (DRY)"
            );
        }
    }
}
