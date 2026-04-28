//! Post-turn invariant checks applied to the assistant's reply.
//!
//! Motivated by the Income session 2026-04-21 13:34–13:36, where the
//! assistant emitted seven consecutive `tool_use` blocks and no final
//! `text` — leaving the user with no visible reply. We cannot easily
//! re-run the model from such a state, but we CAN surface the
//! invariant breach as a structured event so:
//!
//! 1. the coordinator can decide to ping the model for a closing text,
//! 2. operators get a signal in `session.jsonl` + metrics,
//! 3. the prompt rule "no silent tool-only turns" has a backing enforcement.
//!
//! Keep the module pure — no I/O, no history, just block classification.

use crate::types::ContentBlock;

/// Report from running invariants against an assistant turn's output.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct TurnInvariantReport {
    /// True when the turn ended without ANY `Text` block visible to the
    /// user. Silent tool-only runs almost always represent a wasted
    /// turn from the user's point of view.
    pub incomplete_turn: bool,
    /// True when a `Thinking` block's content appears inside a `Text`
    /// block — that's the B1 "thinking leaked into user output"
    /// failure mode. Catches the case where a provider stream
    /// concatenated the reasoning into the visible reply.
    pub thinking_leaked_into_text: bool,
    /// True when at least one `ToolUse` has no corresponding
    /// `ToolResult` later in the block stream. Indicates a broken
    /// pairing (shouldn't happen normally — but if it does, the loop
    /// has drifted). Checked for defence in depth.
    pub orphan_tool_use: bool,
}

impl TurnInvariantReport {
    /// True when any invariant was broken.
    pub fn is_violated(&self) -> bool {
        self.incomplete_turn || self.thinking_leaked_into_text || self.orphan_tool_use
    }
}

/// Namespacing handle — use the free function [`check_assistant_blocks`]
/// directly, or this struct for symmetry with other validators.
pub struct TurnInvariants;

impl TurnInvariants {
    /// Run every invariant over the assistant turn's content stream.
    pub fn check_assistant_blocks(blocks: &[ContentBlock]) -> TurnInvariantReport {
        check_assistant_blocks(blocks)
    }
}

/// Inspect the assistant turn's blocks and return a report. Pure, safe
/// to call on every turn.
pub fn check_assistant_blocks(blocks: &[ContentBlock]) -> TurnInvariantReport {
    let mut text_blocks: Vec<&str> = Vec::new();
    let mut thinking_blocks: Vec<&str> = Vec::new();
    let mut tool_uses: Vec<&str> = Vec::new();
    let mut tool_results: Vec<&str> = Vec::new();

    for b in blocks {
        match b {
            ContentBlock::Text { text } => text_blocks.push(text),
            ContentBlock::Thinking { text } => thinking_blocks.push(text),
            ContentBlock::ToolUse { id, .. } => tool_uses.push(id),
            ContentBlock::ToolResult { call_id, .. } => tool_results.push(call_id),
            ContentBlock::Image { .. } => {}
        }
    }

    let incomplete_turn = text_blocks.iter().all(|t| t.trim().is_empty());

    // Only consider thinking bodies with a meaningful minimum length;
    // otherwise a one-word match would produce false positives.
    let thinking_leaked_into_text = thinking_blocks
        .iter()
        .filter(|t| t.trim().len() >= 12)
        .any(|thought| {
            let needle = thought.trim();
            text_blocks.iter().any(|out| out.contains(needle))
        });

    let orphan_tool_use = tool_uses
        .iter()
        .any(|id| !tool_results.iter().any(|r| r == id));

    TurnInvariantReport {
        incomplete_turn,
        thinking_leaked_into_text,
        orphan_tool_use,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn tool_only_turn_is_incomplete() {
        let blocks = vec![
            ContentBlock::ToolUse {
                id: "c1".into(),
                name: "bash".into(),
                input: json!({}),
            },
            ContentBlock::ToolUse {
                id: "c2".into(),
                name: "bash".into(),
                input: json!({}),
            },
        ];
        let r = check_assistant_blocks(&blocks);
        assert!(r.incomplete_turn, "all tool_use, no text = incomplete");
        assert!(r.is_violated());
    }

    #[test]
    fn text_reply_is_complete() {
        let blocks = vec![
            ContentBlock::ToolUse {
                id: "c1".into(),
                name: "bash".into(),
                input: json!({}),
            },
            ContentBlock::Text {
                text: "Готово.".into(),
            },
        ];
        let r = check_assistant_blocks(&blocks);
        assert!(!r.incomplete_turn);
    }

    #[test]
    fn thinking_inside_text_is_flagged() {
        let thought = "The user wants summary; my plan is X Y Z";
        let blocks = vec![
            ContentBlock::Thinking {
                text: thought.into(),
            },
            ContentBlock::Text {
                text: format!("{thought}\n\nИтог: ..."),
            },
        ];
        let r = check_assistant_blocks(&blocks);
        assert!(r.thinking_leaked_into_text, "thinking body must not be echoed into text");
    }

    #[test]
    fn orphan_tool_use_flagged() {
        let blocks = vec![ContentBlock::ToolUse {
            id: "c1".into(),
            name: "bash".into(),
            input: json!({}),
        }];
        let r = check_assistant_blocks(&blocks);
        assert!(r.orphan_tool_use);
    }
}
