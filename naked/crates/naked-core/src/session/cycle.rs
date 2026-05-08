//! Checkpoint-restart cycle management for long-running sessions.

use std::fs::{self, OpenOptions};
use std::io::Write;
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

use crate::types::ConversationMessage;
use crate::working_set::WorkingSet;

pub const DEFAULT_CYCLE_THRESHOLD_TOKENS: u64 = 768_000;
pub const DEFAULT_BRIEFING_MAX_TOKENS: usize = 3_000;
const CHARS_PER_TOKEN: usize = 4;

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CycleConfig {
    pub enabled: bool,
    pub threshold_tokens: u64,
    pub briefing_max_tokens: usize,
}

impl Default for CycleConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            threshold_tokens: DEFAULT_CYCLE_THRESHOLD_TOKENS,
            briefing_max_tokens: DEFAULT_BRIEFING_MAX_TOKENS,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CycleCheckpoint {
    pub cycle_number: u32,
    pub briefing: String,
    pub working_set_paths: Vec<String>,
    pub archived_at: String,
    pub messages_archived: usize,
    pub tokens_at_archive: u64,
}

pub fn should_advance_cycle(estimated_tokens: u64, config: &CycleConfig) -> bool {
    config.enabled && estimated_tokens >= config.threshold_tokens
}

pub fn write_archive(
    archive_dir: &Path,
    session_id: &str,
    cycle_number: u32,
    messages: &[ConversationMessage],
) -> std::io::Result<PathBuf> {
    let dir = archive_dir.join("cycle_archives").join(session_id);
    fs::create_dir_all(&dir)?;
    let path = dir.join(format!("cycle_{cycle_number}.jsonl"));
    let mut file = OpenOptions::new()
        .create(true)
        .truncate(true)
        .write(true)
        .open(&path)?;
    for msg in messages {
        writeln!(file, "{}", serde_json::to_string(msg).unwrap_or_default())?;
    }
    Ok(path)
}

pub fn read_archive(path: &Path) -> std::io::Result<Vec<ConversationMessage>> {
    let content = fs::read_to_string(path)?;
    Ok(content
        .lines()
        .filter(|l| !l.trim().is_empty())
        .filter_map(|l| serde_json::from_str(l).ok())
        .collect())
}

pub fn build_restart_prompt(checkpoint: &CycleCheckpoint, original_system: &str) -> String {
    let ws_block = if checkpoint.working_set_paths.is_empty() {
        String::new()
    } else {
        format!(
            "\n[Active files: {}]",
            checkpoint.working_set_paths.join(", ")
        )
    };
    format!(
        "{original_system}\n\n<cycle_restart cycle=\"{}\">\nPrevious cycle archived ({} messages, ~{} tokens at {}).\n{}\n{ws_block}\nContinue from this state. Do not recap — resume directly.\n</cycle_restart>",
        checkpoint.cycle_number,
        checkpoint.messages_archived,
        checkpoint.tokens_at_archive,
        checkpoint.archived_at,
        checkpoint.briefing,
    )
}

pub fn build_checkpoint(
    cycle_number: u32,
    messages: &[ConversationMessage],
    estimated_tokens: u64,
    ws: Option<&WorkingSet>,
    config: &CycleConfig,
) -> CycleCheckpoint {
    let budget = config.briefing_max_tokens * CHARS_PER_TOKEN;
    let mut briefing = String::new();
    for msg in messages.iter().rev().take(8).rev() {
        let text = msg.text_content();
        if text.is_empty() {
            continue;
        }
        let role = match msg.role {
            crate::types::Role::User => "user",
            crate::types::Role::Assistant => "assistant",
            crate::types::Role::Tool => "tool",
            crate::types::Role::System => continue,
        };
        let snippet = if text.len() > 400 {
            format!("{}...", &text[..400])
        } else {
            text
        };
        briefing.push_str(&format!("[{role}] {snippet}\n"));
        if briefing.len() >= budget {
            briefing.truncate(budget);
            briefing.push_str("...\n");
            break;
        }
    }
    CycleCheckpoint {
        cycle_number,
        briefing,
        working_set_paths: ws.map(|w| w.top_paths(8)).unwrap_or_default(),
        archived_at: chrono::Utc::now().to_rfc3339(),
        messages_archived: messages.len(),
        tokens_at_archive: estimated_tokens,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn below_threshold_no_advance() {
        assert!(!should_advance_cycle(
            50_000,
            &CycleConfig {
                threshold_tokens: 100_000,
                ..Default::default()
            }
        ));
    }

    #[test]
    fn above_threshold_advances() {
        let cfg = CycleConfig {
            threshold_tokens: 100_000,
            ..Default::default()
        };
        assert!(should_advance_cycle(100_000, &cfg));
    }

    #[test]
    fn disabled_never_advances() {
        assert!(!should_advance_cycle(
            999_999,
            &CycleConfig {
                enabled: false,
                threshold_tokens: 100,
                ..Default::default()
            }
        ));
    }

    #[test]
    fn archive_round_trip() {
        let tmp = tempfile::tempdir().unwrap();
        let msgs = vec![
            ConversationMessage::user("hello"),
            ConversationMessage::user("world"),
        ];
        let path = write_archive(tmp.path(), "s1", 0, &msgs).unwrap();
        let loaded = read_archive(&path).unwrap();
        assert_eq!(loaded.len(), 2);
        assert_eq!(loaded[0].text_content(), "hello");
    }

    #[test]
    fn restart_prompt_contains_key_fields() {
        let cp = CycleCheckpoint {
            cycle_number: 1,
            briefing: "Refactoring loop".into(),
            working_set_paths: vec!["src/loop_.rs".into()],
            archived_at: "2026-05-07T12:00:00Z".into(),
            messages_archived: 42,
            tokens_at_archive: 500_000,
        };
        let prompt = build_restart_prompt(&cp, "You are helpful.");
        assert!(
            prompt.contains("cycle=\"1\"")
                && prompt.contains("src/loop_.rs")
                && prompt.contains("42 messages")
        );
    }

    #[test]
    fn build_checkpoint_budget() {
        let msgs: Vec<ConversationMessage> = (0..20)
            .map(|i| ConversationMessage::user(format!("msg {i} content")))
            .collect();
        let cp = build_checkpoint(
            0,
            &msgs,
            100_000,
            None,
            &CycleConfig {
                briefing_max_tokens: 50,
                ..Default::default()
            },
        );
        assert!(cp.briefing.len() < 500);
        assert_eq!(cp.messages_archived, 20);
    }
}
