//! Pure rendering helpers for the Telegram live-message stream.
//!
//! Extracted from `main.rs::stream_response` so we can unit-test the
//! "thinking must not leak into user-visible text" invariant without
//! spinning up teloxide. The incident this guards against: 2026-04-21
//! 12:21 UTC (Income), when a raw internal reasoning paragraph shipped
//! as the first line of the bot's final message.

use naked_core::types::ContentBlock;

/// Build the live text the bot should edit into its outgoing Telegram
/// message, from the current assistant-turn content stream.
///
/// Only `ContentBlock::Text` contributes. `Thinking` is dropped on
/// purpose — it must never appear in front of the user. `ToolUse` /
/// `ToolResult` / `Image` produce their own rendering paths elsewhere
/// (tool-progress waterfall, artifact links) and are also skipped.
pub fn render_live_user_text(blocks: &[ContentBlock]) -> String {
    let mut out = String::new();
    for block in blocks {
        if let ContentBlock::Text { text } = block {
            out.push_str(text);
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn only_text_blocks_contribute() {
        let rendered = render_live_user_text(&[
            ContentBlock::Thinking {
                text: "user wants summary; rough plan: a, b, c".into(),
            },
            ContentBlock::Text {
                text: "Вот ответ".into(),
            },
            ContentBlock::ToolUse {
                id: "t1".into(),
                name: "bash".into(),
                input: serde_json::json!({}),
            },
        ]);
        assert_eq!(rendered, "Вот ответ");
    }

    #[test]
    fn empty_when_only_thinking() {
        let rendered = render_live_user_text(&[ContentBlock::Thinking {
            text: "I should not leak".into(),
        }]);
        assert_eq!(rendered, "");
    }

    #[test]
    fn concatenates_multiple_text_blocks() {
        let rendered = render_live_user_text(&[
            ContentBlock::Text {
                text: "Hello ".into(),
            },
            ContentBlock::Thinking {
                text: "pondering".into(),
            },
            ContentBlock::Text {
                text: "world".into(),
            },
        ]);
        assert_eq!(rendered, "Hello world");
    }
}
