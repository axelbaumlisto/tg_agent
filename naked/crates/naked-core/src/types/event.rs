use super::session::*;
use super::tool::*;

// -- Streaming types (not persisted) -----------------------------------------

#[derive(Debug, Clone)]
pub enum StreamChunk {
    Text(String),
    Thinking(String),
    ToolUse {
        id: String,
        name: String,
        input: serde_json::Value,
    },
    Usage(TurnUsage),
    Done,
    Error(String),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ToolState {
    Completed,
    Error,
}

#[derive(Debug, Clone)]
pub enum AgentEvent {
    ThinkingDelta(String),
    TextDelta(String),
    ToolStart {
        call_id: String,
        name: String,
        input: serde_json::Value,
    },
    ToolEnd {
        call_id: String,
        name: String,
        state: ToolState,
        output: String,
    },
    PermissionRequest {
        call_id: String,
        tool_name: String,
        input: serde_json::Value,
        permission: Permission,
    },
    /// Periodic signal during long tool execution — keeps watchers alive.
    Heartbeat,
    /// Progress from a child sub-agent forwarded to the parent.
    SubAgentProgress {
        agent_id: String,
        event: SubAgentEvent,
    },
    UsageUpdate(TurnUsage),
    ContextCompacted {
        before_msgs: usize,
        after_msgs: usize,
        /// Short label from the compaction summary (e.g. first line of ## Goal).
        summary_hint: Option<String>,
        /// Number of tracked files (read + modified).
        files_count: usize,
    },
    /// Checkpoint-restart cycle triggered.
    CycleRestarted {
        cycle_number: u32,
        archived_messages: usize,
        archive_path: String,
    },
    /// Partial stdout/stderr from a running tool (e.g. bash).
    ToolOutput {
        call_id: String,
        chunk: String,
    },
    /// A steer message from the user was injected into the active turn.
    /// `text` is the merged combined text (joined with \n\n);
    /// `msg_ids` lists the Telegram message ids from the original
    /// `SteerMessage`s, in original arrival order. The bot uses
    /// `msg_ids` to delete the matching "↩️ Принято" temp
    /// confirmations once the steer is actually in history.
    SteerReceived {
        text: String,
        msg_ids: Vec<i32>,
    },
    Error(String),
    Idle,
}

/// Lightweight subset of child events forwarded to the parent agent.
#[derive(Debug, Clone)]
pub enum SubAgentEvent {
    Started { prompt_preview: String },
    ToolUse { name: String, input_preview: String },
    ToolDone { name: String, state: ToolState },
    TextDelta(String),
    Finished { tokens: u64 },
    Error(String),
}

#[derive(Debug, Clone)]
pub struct PermissionResponse {
    pub call_id: String,
    pub allowed: bool,
}

/// A user message injected into an active turn to steer the agent.
#[derive(Debug, Clone)]
pub struct SteerMessage {
    /// Telegram message id (used for edit-replacement in the pending queue).
    pub msg_id: i32,
    /// The user's text.
    pub text: String,
    /// `true` when this is an edit of a previously sent steer message.
    pub is_edit: bool,
}

/// Returned by `AgentCore::send_prompt` — events channel + permission/steer reply channels.
pub struct AgentHandle {
    pub events: tokio::sync::mpsc::Receiver<AgentEvent>,
    pub permissions: tokio::sync::mpsc::Sender<PermissionResponse>,
    pub steer: tokio::sync::mpsc::Sender<SteerMessage>,
}

// ---------------------------------------------------------------------------
// EventHandler — default no-op trait so new variants don't break consumers
// ---------------------------------------------------------------------------

/// Trait for handling agent events. All methods have default no-op
/// implementations, so adding a new `AgentEvent` variant only requires
/// adding a new method here — existing consumers compile unchanged.
#[allow(unused_variables)]
pub trait EventHandler {
    fn on_thinking_delta(&mut self, text: &str) {}
    fn on_text_delta(&mut self, text: &str) {}
    fn on_tool_start(&mut self, call_id: &str, name: &str, input: &serde_json::Value) {}
    fn on_tool_end(&mut self, call_id: &str, name: &str, state: ToolState, output: &str) {}
    fn on_permission_request(
        &mut self,
        call_id: &str,
        tool_name: &str,
        input: &serde_json::Value,
        permission: Permission,
    ) {
    }
    fn on_heartbeat(&mut self) {}
    fn on_sub_agent_progress(&mut self, agent_id: &str, event: &SubAgentEvent) {}
    fn on_usage_update(&mut self, usage: &TurnUsage) {}
    fn on_context_compacted(
        &mut self,
        before_msgs: usize,
        after_msgs: usize,
        summary_hint: Option<&str>,
        files_count: usize,
    ) {
    }
    fn on_cycle_restarted(
        &mut self,
        cycle_number: u32,
        archived_messages: usize,
        archive_path: &str,
    ) {
    }
    fn on_tool_output(&mut self, call_id: &str, chunk: &str) {}
    fn on_steer_received(&mut self, text: &str) {}
    fn on_error(&mut self, message: &str) {}
    fn on_idle(&mut self) {}
}

/// Dispatch an AgentEvent to the appropriate EventHandler method.
pub fn dispatch_event(handler: &mut dyn EventHandler, event: &AgentEvent) {
    match event {
        AgentEvent::ThinkingDelta(t) => handler.on_thinking_delta(t),
        AgentEvent::TextDelta(t) => handler.on_text_delta(t),
        AgentEvent::ToolStart {
            call_id,
            name,
            input,
        } => {
            handler.on_tool_start(call_id, name, input);
        }
        AgentEvent::ToolEnd {
            call_id,
            name,
            state,
            output,
        } => {
            handler.on_tool_end(call_id, name, *state, output);
        }
        AgentEvent::PermissionRequest {
            call_id,
            tool_name,
            input,
            permission,
        } => {
            handler.on_permission_request(call_id, tool_name, input, *permission);
        }
        AgentEvent::Heartbeat => handler.on_heartbeat(),
        AgentEvent::SubAgentProgress { agent_id, event } => {
            handler.on_sub_agent_progress(agent_id, event);
        }
        AgentEvent::UsageUpdate(u) => handler.on_usage_update(u),
        AgentEvent::ContextCompacted {
            before_msgs,
            after_msgs,
            summary_hint,
            files_count,
        } => {
            handler.on_context_compacted(
                *before_msgs,
                *after_msgs,
                summary_hint.as_deref(),
                *files_count,
            );
        }
        AgentEvent::CycleRestarted {
            cycle_number,
            archived_messages,
            archive_path,
        } => {
            handler.on_cycle_restarted(*cycle_number, *archived_messages, archive_path);
        }
        AgentEvent::ToolOutput { call_id, chunk } => handler.on_tool_output(call_id, chunk),
        AgentEvent::SteerReceived { text, msg_ids: _ } => handler.on_steer_received(text),
        AgentEvent::Error(msg) => handler.on_error(msg),
        AgentEvent::Idle => handler.on_idle(),
    }
}

#[cfg(test)]
mod handler_tests {
    use super::*;

    struct CountHandler {
        text_count: usize,
        idle_count: usize,
    }
    impl EventHandler for CountHandler {
        fn on_text_delta(&mut self, _text: &str) {
            self.text_count += 1;
        }
        fn on_idle(&mut self) {
            self.idle_count += 1;
        }
    }

    #[test]
    fn dispatch_routes_to_handler() {
        let mut h = CountHandler {
            text_count: 0,
            idle_count: 0,
        };
        dispatch_event(&mut h, &AgentEvent::TextDelta("hi".into()));
        dispatch_event(&mut h, &AgentEvent::Idle);
        dispatch_event(&mut h, &AgentEvent::Heartbeat); // no-op
        assert_eq!(h.text_count, 1);
        assert_eq!(h.idle_count, 1);
    }

    #[test]
    fn default_handler_compiles_with_no_overrides() {
        struct NoOp;
        impl EventHandler for NoOp {}
        let mut h = NoOp;
        dispatch_event(&mut h, &AgentEvent::Error("test".into()));
        // Should not panic — all defaults are no-op.
    }
}
