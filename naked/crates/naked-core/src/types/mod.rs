//! Core types — split into domain modules, re-exported flat for compatibility.

pub mod event;
pub mod message;
pub mod session;
pub mod tool;

// Re-export everything flat so `use crate::types::*` still works.
pub use event::*;
pub use message::*;
pub use session::*;
pub use tool::*;

// ── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn strip_sentinel_no_marker_is_borrowed() {
        let s = "plain text without marker";
        let out = strip_image_ref_sentinel(s);
        assert!(matches!(out, std::borrow::Cow::Borrowed(_)));
        assert_eq!(out, s);
    }

    #[test]
    fn strip_sentinel_hash_form() {
        let baseline = SENTINEL_LEAK_COUNT.load(std::sync::atomic::Ordering::Relaxed);
        let s = "before @@NAKED_IMG_REF@@/abc123def after";
        let out = strip_image_ref_sentinel(s);
        assert_eq!(out, "before  after");
        assert!(SENTINEL_LEAK_COUNT.load(std::sync::atomic::Ordering::Relaxed) > baseline);
    }

    #[test]
    fn strip_sentinel_legacy_json_form() {
        let s = r#"head @@NAKED_IMG_REF@@{"path":"a.png","mime":"image/png"} tail"#;
        let out = strip_image_ref_sentinel(s);
        assert_eq!(out, "head  tail");
    }

    #[test]
    fn strip_sentinel_multiple_markers() {
        let s = "a @@NAKED_IMG_REF@@/x b @@NAKED_IMG_REF@@/y c";
        let out = strip_image_ref_sentinel(s);
        assert_eq!(out, "a  b  c");
    }

    #[test]
    fn strip_sentinel_at_eol_no_trailing_garbage() {
        let s = "foo @@NAKED_IMG_REF@@/abc";
        let out = strip_image_ref_sentinel(s);
        assert_eq!(out, "foo ");
    }

    #[test]
    fn contains_image_ref_sentinel_detects_anywhere() {
        assert!(!contains_image_ref_sentinel("plain"));
        assert!(contains_image_ref_sentinel("@@NAKED_IMG_REF@@/x"));
        assert!(contains_image_ref_sentinel(
            "noise before @@NAKED_IMG_REF@@/x noise after"
        ));
    }

    #[test]
    fn conversation_message_user() {
        let msg = ConversationMessage::user("hello");
        assert_eq!(msg.role, Role::User);
        assert_eq!(msg.text_content(), "hello");
        assert!(msg.usage.is_none());
    }

    #[test]
    fn conversation_message_assistant_with_tool_use() {
        let blocks = vec![
            ContentBlock::Text {
                text: "let me check".into(),
            },
            ContentBlock::ToolUse {
                id: "c1".into(),
                name: "bash".into(),
                input: serde_json::json!({"command": "ls"}),
            },
        ];
        let msg = ConversationMessage::assistant(blocks, None);
        assert_eq!(msg.role, Role::Assistant);
        assert_eq!(msg.text_content(), "let me check");
        let uses = msg.tool_uses();
        assert_eq!(uses.len(), 1);
        assert_eq!(uses[0].1, "bash");
    }

    #[test]
    fn conversation_message_tool_result() {
        let msg = ConversationMessage::tool_result("c1", "file.txt", false);
        assert_eq!(msg.role, Role::Tool);
        assert!(msg.tool_uses().is_empty());
    }

    #[test]
    fn conversation_message_system() {
        let msg = ConversationMessage::system("You are helpful");
        assert_eq!(msg.role, Role::System);
        assert_eq!(msg.text_content(), "You are helpful");
    }

    #[test]
    fn text_content_joins_multiple_blocks() {
        let msg = ConversationMessage::assistant(
            vec![
                ContentBlock::Text { text: "a".into() },
                ContentBlock::ToolUse {
                    id: "x".into(),
                    name: "y".into(),
                    input: serde_json::json!({}),
                },
                ContentBlock::Text { text: "b".into() },
            ],
            None,
        );
        assert_eq!(msg.text_content(), "ab");
    }

    #[test]
    fn turn_usage_total_tokens() {
        let u = TurnUsage {
            input_tokens: 100,
            output_tokens: 50,
            cache_read_tokens: 10,
            cache_write_tokens: 5,
        };
        assert_eq!(u.total_tokens(), 165);
    }

    #[test]
    fn turn_usage_default_is_zero() {
        let u = TurnUsage::default();
        assert_eq!(u.total_tokens(), 0);
    }

    #[test]
    fn estimate_cost_sonnet() {
        let u = TurnUsage {
            input_tokens: 1_000_000,
            output_tokens: 100_000,
            ..Default::default()
        };
        let (inp, out, total) = u.estimate_cost("claude-3-5-sonnet");
        assert!((inp - 3.0).abs() < 0.01);
        assert!((out - 1.5).abs() < 0.01);
        assert!((total - 4.5).abs() < 0.01);
    }

    #[test]
    fn estimate_cost_unknown_model() {
        let u = TurnUsage {
            input_tokens: 1000,
            output_tokens: 500,
            ..Default::default()
        };
        let (_, _, total) = u.estimate_cost("some-unknown-model");
        assert!(total > 0.0);
    }

    #[test]
    fn content_block_serde_round_trip() {
        let blocks = vec![
            ContentBlock::Text { text: "hi".into() },
            ContentBlock::ToolUse {
                id: "1".into(),
                name: "bash".into(),
                input: serde_json::json!({"cmd": "ls"}),
            },
            ContentBlock::ToolResult {
                call_id: "1".into(),
                output: "ok".into(),
                is_error: false,
            },
        ];
        let json = serde_json::to_string(&blocks).unwrap();
        let parsed: Vec<ContentBlock> = serde_json::from_str(&json).unwrap();
        assert_eq!(blocks, parsed);
    }

    #[test]
    fn conversation_message_serde_round_trip() {
        let msg = ConversationMessage::user("test");
        let json = serde_json::to_string(&msg).unwrap();
        let parsed: ConversationMessage = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed.role, Role::User);
        assert_eq!(parsed.text_content(), "test");
    }

    #[test]
    fn role_serde_values() {
        assert_eq!(serde_json::to_string(&Role::User).unwrap(), "\"user\"");
        assert_eq!(
            serde_json::to_string(&Role::Assistant).unwrap(),
            "\"assistant\""
        );
        assert_eq!(serde_json::to_string(&Role::Tool).unwrap(), "\"tool\"");
        assert_eq!(serde_json::to_string(&Role::System).unwrap(), "\"system\"");
    }

    #[test]
    fn permission_serde_values() {
        assert_eq!(
            serde_json::to_string(&Permission::ReadOnly).unwrap(),
            "\"read_only\""
        );
        assert_eq!(
            serde_json::to_string(&Permission::Dangerous).unwrap(),
            "\"dangerous\""
        );
    }

    #[test]
    fn agent_event_heartbeat_is_clone() {
        let ev = AgentEvent::Heartbeat;
        let _cloned = ev.clone();
    }

    #[test]
    fn sub_agent_event_variants() {
        let started = SubAgentEvent::Started {
            prompt_preview: "research Rust".into(),
        };
        let tool = SubAgentEvent::ToolUse {
            name: "web_search".into(),
            input_preview: "Rust CLI 2025".into(),
        };
        let done = SubAgentEvent::ToolDone {
            name: "web_search".into(),
            state: ToolState::Completed,
        };
        let text = SubAgentEvent::TextDelta("partial output".into());
        let fin = SubAgentEvent::Finished { tokens: 5000 };
        let err = SubAgentEvent::Error("timeout".into());

        // All must be cloneable and debuggable
        for ev in [started, tool, done, text, fin, err] {
            let _ = format!("{:?}", ev.clone());
        }
    }

    #[test]
    fn agent_event_sub_agent_progress() {
        let ev = AgentEvent::SubAgentProgress {
            agent_id: "sa-001".into(),
            event: SubAgentEvent::ToolUse {
                name: "bash".into(),
                input_preview: "ls -la".into(),
            },
        };
        let cloned = ev.clone();
        assert!(format!("{cloned:?}").contains("sa-001"));
    }
}
