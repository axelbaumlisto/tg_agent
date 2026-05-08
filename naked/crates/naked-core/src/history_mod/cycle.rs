//! Cycle restart methods for ConversationHistory.

#[allow(unused_imports)]
use super::ContentBlock;
#[allow(unused_imports)]
use super::ConversationHistory;
#[allow(unused_imports)]
use super::ConversationMessage;
#[allow(unused_imports)]
use super::Role;
#[allow(unused_imports)]
use super::compaction;

impl ConversationHistory {
    /// How many checkpoint-restart cycles have been completed.
    pub fn cycle_count(&self) -> u32 {
        self.cycle_count
    }

    /// Clear messages for a cycle restart, replacing with a fresh system prompt.
    pub fn clear_for_cycle_restart(&mut self, restart_system_prompt: &str) {
        self.messages.clear();
        self.system_prompt = restart_system_prompt.to_string();
        self.last_input_tokens = None;
        self.cycle_count += 1;
    }

    pub fn cycle_restart(&mut self, working_set_block: &str) -> Option<String> {
        let est = self.estimated_tokens();
        let limit = self.context_window_tokens as usize;
        // Only trigger at >90%:
        if est <= limit * 9 / 10 {
            return None;
        }

        // Build briefing from recent messages:
        let recent_count = self.messages.len().min(6);
        let recent = &self.messages[self.messages.len() - recent_count..];
        let summary = compaction::summarize_messages(&self.messages);
        let recent_text: String = recent
            .iter()
            .filter_map(|m| {
                let text = m.text_content();
                if text.is_empty() { None } else { Some(text) }
            })
            .take(3)
            .collect::<Vec<_>>()
            .join("\n");

        let briefing = format!(
            "<cycle_restart>\n\
             Previous cycle archived ({} messages, ~{} tokens).\n\
             {summary}\n\
             \n\
             Last exchange:\n{recent_text}\n\
             {working_set_block}\
             </cycle_restart>",
            self.messages.len(),
            est,
        );

        // Reset: keep system prompt, add briefing as first user context:
        self.messages.clear();
        self.messages.push(ConversationMessage {
            role: Role::System,
            blocks: vec![ContentBlock::Text {
                text: briefing.clone(),
            }],
            timestamp: chrono::Utc::now(),
            usage: None,
        });
        self.last_input_tokens = None;
        self.last_compaction_summary = Some(summary);

        Some(briefing)
    }
}

#[cfg(test)]
mod cycle_tests {
    use super::*;

    #[test]
    fn cycle_restart_triggers_at_90pct() {
        let mut h = ConversationHistory::new("sys".into());
        h.set_context_window_tokens(100);
        for i in 0..30 {
            h.push_user(&format!("msg {i} with substantial content for tokens"));
            h.push_assistant(
                vec![ContentBlock::Text {
                    text: format!("reply {i} also with substantial content here"),
                }],
                Default::default(),
            );
        }
        let before = h.message_count();
        assert!(before > 10);

        let result = h.cycle_restart("[working set: main.rs]");
        assert!(result.is_some());
        let briefing = result.unwrap();
        assert!(briefing.contains("<cycle_restart>"));
        assert!(briefing.contains("archived"));
        // History should be minimal after restart:
        assert!(h.message_count() <= 2);
    }

    #[test]
    fn cycle_restart_noop_below_threshold() {
        let mut h = ConversationHistory::new("sys".into());
        h.set_context_window_tokens(100_000);
        h.push_user("hi");
        assert!(h.cycle_restart("").is_none());
    }

    #[test]
    fn cycle_restart_includes_working_set() {
        let mut h = ConversationHistory::new("sys".into());
        h.set_context_window_tokens(50);
        for i in 0..20 {
            h.push_user(&format!("message {i} padded content here"));
            h.push_assistant(
                vec![ContentBlock::Text {
                    text: format!("reply {i} padded content here"),
                }],
                Default::default(),
            );
        }
        let result = h.cycle_restart("[working set: foo.rs, bar.rs]");
        assert!(result.is_some());
        assert!(result.unwrap().contains("foo.rs"));
    }

    fn asst(text: &str) -> Vec<ContentBlock> {
        vec![ContentBlock::Text {
            text: text.to_string(),
        }]
    }

    #[test]
    fn compact_with_ws_pins_relevant_message() {
        use crate::working_set::WorkingSet;
        use std::path::Path;
        let mut h = ConversationHistory::new("sys".into());
        for i in 0..10 {
            if i == 2 {
                h.push_user("Please edit src/main.rs");
            } else {
                h.push_user(&format!("msg {i}"));
            }
            h.push_assistant(asst("ok"), None);
        }
        let mut ws = WorkingSet::new();
        ws.touch(Path::new("src/main.rs"));
        h.compact_with_working_set(4, Some(&ws));
        let all_text: String = h.messages().iter().map(|m| m.text_content()).collect();
        assert!(all_text.contains("src/main.rs"), "pinned should survive");
    }

    #[test]
    fn compact_with_ws_includes_working_set_in_summary() {
        use crate::working_set::WorkingSet;
        use std::path::Path;
        let mut h = ConversationHistory::new("sys".into());
        for i in 0..10 {
            h.push_user(&format!("msg {i}"));
            h.push_assistant(asst("ok"), None);
        }
        let mut ws = WorkingSet::new();
        ws.touch(Path::new("src/lib.rs"));
        ws.touch(Path::new("src/main.rs"));
        h.compact_with_working_set(4, Some(&ws));
        let summary = h.messages()[0].text_content();
        assert!(
            summary.contains("Active files") || summary.contains("src/main.rs"),
            "summary should mention working set"
        );
    }

    #[test]
    fn compact_with_ws_none_behaves_like_plain() {
        let mut h1 = ConversationHistory::new("sys".into());
        let mut h2 = ConversationHistory::new("sys".into());
        for i in 0..10 {
            h1.push_user(&format!("msg {i}"));
            h1.push_assistant(asst("ok"), None);
            h2.push_user(&format!("msg {i}"));
            h2.push_assistant(asst("ok"), None);
        }
        h1.compact(4);
        h2.compact_with_working_set(4, None);
        assert_eq!(h1.message_count(), h2.message_count());
    }
}
