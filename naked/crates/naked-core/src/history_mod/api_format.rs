//! Convert conversation history to provider API message format.

#[allow(unused_imports)]
use super::ContentBlock;
#[allow(unused_imports)]
use super::ConversationHistory;
#[allow(unused_imports)]
use super::Role;

impl ConversationHistory {
    pub fn to_api_messages(&self) -> Vec<serde_json::Value> {
        let mut api_msgs = Vec::new();

        for msg in &self.messages {
            match msg.role {
                Role::User => {
                    let content: Vec<serde_json::Value> = msg
                        .blocks
                        .iter()
                        .filter_map(|b| match b {
                            ContentBlock::Text { text } => {
                                // Defence in depth (two layers):
                                //   1. If the entire block is just a sentinel
                                //      (intern failed for an externalised
                                //      image), drop it — the model has no
                                //      use for an empty marker.
                                //   2. Otherwise strip any embedded sentinel
                                //      segments mid-text (e.g. user pasted
                                //      a marker by accident, or an old
                                //      tool injected one). Bumps
                                //      `SENTINEL_LEAK_COUNT` per strip.
                                if text.starts_with(crate::types::IMAGE_REF_SENTINEL_PREFIX)
                                    && !text[crate::types::IMAGE_REF_SENTINEL_PREFIX.len()..]
                                        .contains(' ')
                                {
                                    crate::types::SENTINEL_LEAK_COUNT
                                        .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                                    return None;
                                }
                                let cleaned = crate::types::strip_image_ref_sentinel(text);
                                Some(serde_json::json!({"type": "text", "text": cleaned}))
                            }
                            ContentBlock::Image {
                                mime,
                                data_base64,
                                detail,
                            } => {
                                // Anthropic-style canonical form. The OpenAI-compat
                                // bridge reads `detail_hint` (mirrored at the top of
                                // the block, outside `source`) and translates it to
                                // `image_url.detail = "low"|"high"|"auto"`. Anthropic
                                // and other native vision providers simply ignore
                                // unknown top-level keys.
                                let mut v = serde_json::json!({
                                    "type": "image",
                                    "source": {
                                        "type": "base64",
                                        "media_type": mime,
                                        "data": data_base64,
                                    }
                                });
                                if let Some(d) = detail {
                                    v["detail_hint"] =
                                        serde_json::Value::String(d.as_str().to_string());
                                }
                                Some(v)
                            }
                            _ => None,
                        })
                        .collect();
                    api_msgs.push(serde_json::json!({
                        "role": "user",
                        "content": content,
                    }));
                }
                Role::Assistant => {
                    let content: Vec<serde_json::Value> = msg
                        .blocks
                        .iter()
                        .filter_map(|b| match b {
                            ContentBlock::Text { text } => {
                                let cleaned = crate::types::strip_image_ref_sentinel(text);
                                Some(serde_json::json!({"type": "text", "text": cleaned}))
                            }
                            ContentBlock::Thinking { text } => {
                                let cleaned = crate::types::strip_image_ref_sentinel(text);
                                Some(serde_json::json!({"type": "thinking", "thinking": cleaned}))
                            }
                            ContentBlock::ToolUse { id, name, input } => Some(serde_json::json!({
                                "type": "tool_use",
                                "id": id,
                                "name": name,
                                "input": input,
                            })),
                            _ => None,
                        })
                        .collect();
                    api_msgs.push(serde_json::json!({
                        "role": "assistant",
                        "content": content,
                    }));
                }
                Role::Tool => {
                    let content: Vec<serde_json::Value> = msg
                        .blocks
                        .iter()
                        .filter_map(|b| match b {
                            ContentBlock::ToolResult {
                                call_id,
                                output,
                                is_error,
                            } => {
                                let cleaned = crate::types::strip_image_ref_sentinel(output);
                                Some(serde_json::json!({
                                    "type": "tool_result",
                                    "tool_use_id": call_id,
                                    "content": cleaned,
                                    "is_error": is_error,
                                }))
                            }
                            _ => None,
                        })
                        .collect();
                    api_msgs.push(serde_json::json!({
                        "role": "user",
                        "content": content,
                    }));
                }
                Role::System => {
                    let content: Vec<serde_json::Value> = msg
                        .blocks
                        .iter()
                        .filter_map(|b| match b {
                            ContentBlock::Text { text } => {
                                let cleaned = crate::types::strip_image_ref_sentinel(text);
                                Some(serde_json::json!({"type": "text", "text": cleaned}))
                            }
                            _ => None,
                        })
                        .collect();
                    if !content.is_empty() {
                        api_msgs.push(serde_json::json!({
                            "role": "user",
                            "content": content,
                        }));
                    }
                }
            }
        }

        api_msgs
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn text_block(s: &str) -> ContentBlock {
        ContentBlock::Text {
            text: s.to_string(),
        }
    }

    #[test]
    fn text_user_message_to_api() {
        let mut h = ConversationHistory::new("sys".into());
        h.push_user("hello");
        let msgs = h.to_api_messages();
        assert_eq!(msgs.len(), 1);
        assert_eq!(msgs[0]["role"], "user");
        assert_eq!(msgs[0]["content"][0]["text"], "hello");
    }

    #[test]
    fn assistant_message_to_api() {
        let mut h = ConversationHistory::new("sys".into());
        h.push_user("hi");
        h.push_assistant(vec![text_block("reply")], None);
        let msgs = h.to_api_messages();
        assert_eq!(msgs.len(), 2);
        assert_eq!(msgs[1]["role"], "assistant");
    }

    #[test]
    fn empty_history_returns_empty_api() {
        let h = ConversationHistory::new("sys".into());
        assert!(h.to_api_messages().is_empty());
    }

    #[test]
    fn tool_use_and_tool_result_roundtrip() {
        let mut h = ConversationHistory::new("sys".into());
        h.push_user("run tests");
        h.push_assistant(
            vec![ContentBlock::ToolUse {
                id: "t1".into(),
                name: "bash".into(),
                input: serde_json::json!({"command": "echo hi"}),
            }],
            None,
        );
        h.push_tool_result("t1", "hi\n", false);
        let msgs = h.to_api_messages();
        assert!(msgs.len() >= 3, "expected 3+ msgs, got {}", msgs.len());
    }

    #[test]
    fn multiple_user_messages_interleaved() {
        let mut h = ConversationHistory::new("sys".into());
        h.push_user("first");
        h.push_assistant(vec![text_block("ok")], None);
        h.push_user("second");
        let msgs = h.to_api_messages();
        assert_eq!(msgs.len(), 3);
        assert_eq!(msgs[0]["content"][0]["text"], "first");
        assert_eq!(msgs[2]["content"][0]["text"], "second");
    }
}
