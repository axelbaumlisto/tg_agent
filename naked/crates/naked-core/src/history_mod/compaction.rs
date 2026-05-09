//! Compaction helpers — summary generation, file tracking, turn-safe cuts.

use super::ContentBlock;
use super::ConversationMessage;
use super::Role;
use super::{COMPACT_PREAMBLE, COMPACT_RECENT_NOTE, COMPACT_RESUME_INSTRUCTION};

/// Build the structured `<summary>` from a slice of removed messages.
pub(crate) fn summarize_messages(messages: &[ConversationMessage]) -> String {
    let user_count = messages.iter().filter(|m| m.role == Role::User).count();
    let asst_count = messages
        .iter()
        .filter(|m| m.role == Role::Assistant)
        .count();
    let tool_count = messages.iter().filter(|m| m.role == Role::Tool).count();
    // Image blocks carry no textual content so they vanish invisibly when
    // the summary drops older turns. Count them so the continuation
    // message at least *tells* the model: "there were 3 images earlier in
    // this conversation". Without this hint the assistant cannot reference
    // past vision context at all — a silent UX regression.
    let image_count: usize = messages
        .iter()
        .flat_map(|m| m.blocks.iter())
        .filter(|b| matches!(b, ContentBlock::Image { .. }))
        .count();

    let mut tool_names: Vec<&str> = messages
        .iter()
        .flat_map(|m| m.blocks.iter())
        .filter_map(|b| match b {
            ContentBlock::ToolUse { name, .. } => Some(name.as_str()),
            _ => None,
        })
        .collect();
    tool_names.sort_unstable();
    tool_names.dedup();

    let mut lines = vec![
        "<summary>".to_string(),
        "Conversation summary:".to_string(),
        format!(
            "- Scope: {} earlier messages compacted (user={user_count}, assistant={asst_count}, tool={tool_count}).",
            messages.len()
        ),
    ];

    if !tool_names.is_empty() {
        lines.push(format!("- Tools mentioned: {}.", tool_names.join(", ")));
    }

    if image_count > 0 {
        lines.push(format!(
            "- Images in compacted turns: {image_count} (pixels no longer available; \
             ask the user to re-send if you need to see them)."
        ));
    }

    // Recent user requests (last 3)
    let recent_user: Vec<String> = messages
        .iter()
        .filter(|m| m.role == Role::User)
        .rev()
        .filter_map(|m| {
            let t = m.text_content();
            if t.is_empty() {
                None
            } else {
                Some(safe_truncate(&t, 160))
            }
        })
        .take(3)
        .collect::<Vec<_>>()
        .into_iter()
        .rev()
        .collect();
    if !recent_user.is_empty() {
        lines.push("- Recent user requests:".to_string());
        for req in &recent_user {
            lines.push(format!("  - {req}"));
        }
    }

    // Pending work (messages containing todo/next/pending/remaining)
    let pending: Vec<String> = messages
        .iter()
        .rev()
        .filter_map(|m| {
            let t = m.text_content();
            let low = t.to_ascii_lowercase();
            if low.contains("todo")
                || low.contains("next")
                || low.contains("pending")
                || low.contains("remaining")
            {
                Some(safe_truncate(&t, 160))
            } else {
                None
            }
        })
        .take(3)
        .collect::<Vec<_>>()
        .into_iter()
        .rev()
        .collect();
    if !pending.is_empty() {
        lines.push("- Pending work:".to_string());
        for item in &pending {
            lines.push(format!("  - {item}"));
        }
    }

    // Key files (paths with known extensions)
    let key_files = collect_key_files(messages);
    if !key_files.is_empty() {
        lines.push(format!("- Key files referenced: {}.", key_files.join(", ")));
    }

    // Current work (last non-empty text from any role)
    if let Some(current) = messages
        .iter()
        .rev()
        .filter_map(|m| {
            let t = m.text_content();
            if t.trim().is_empty() {
                None
            } else {
                Some(safe_truncate(&t, 200))
            }
        })
        .next()
    {
        lines.push(format!("- Current work: {current}"));
    }

    // Key timeline
    lines.push("- Key timeline:".to_string());
    for msg in messages {
        let role = match msg.role {
            Role::System => "system",
            Role::User => "user",
            Role::Assistant => "assistant",
            Role::Tool => "tool",
        };
        let content: Vec<String> = msg.blocks.iter().map(summarize_block).collect();
        lines.push(format!("  - {role}: {}", content.join(" | ")));
    }
    lines.push("</summary>".to_string());
    lines.join("\n")
}

pub(crate) fn summarize_block(block: &ContentBlock) -> String {
    let raw = match block {
        ContentBlock::Text { text } | ContentBlock::Thinking { text } => text.clone(),
        ContentBlock::ToolUse { name, input, .. } => format!("tool_use {name}({input})"),
        ContentBlock::ToolResult {
            call_id: _,
            output,
            is_error,
        } => {
            let prefix = if *is_error { "error " } else { "" };
            format!("tool_result: {prefix}{output}")
        }
        ContentBlock::Image {
            mime, data_base64, ..
        } => {
            let bytes = data_base64.len() * 3 / 4;
            format!("[image {mime} ~{bytes}B]")
        }
    };
    safe_truncate(&raw, 160)
}

/// Extract file-like paths from message content.
pub(crate) fn collect_key_files(messages: &[ConversationMessage]) -> Vec<String> {
    let mut files: Vec<String> = messages
        .iter()
        .flat_map(|m| m.blocks.iter())
        .flat_map(|b| {
            let text = match b {
                ContentBlock::Text { text } | ContentBlock::Thinking { text } => text.as_str(),
                ContentBlock::ToolUse { input, .. } => {
                    return extract_file_candidates(&input.to_string());
                }
                ContentBlock::ToolResult { output, .. } => output.as_str(),
                ContentBlock::Image { .. } => return Vec::new(),
            };
            extract_file_candidates(text)
        })
        .collect();
    files.sort();
    files.dedup();
    files.into_iter().take(8).collect()
}

pub(crate) fn extract_file_candidates(content: &str) -> Vec<String> {
    const EXTENSIONS: &[&str] = &["rs", "ts", "tsx", "js", "json", "md", "toml", "py", "sh"];
    content
        .split_whitespace()
        .filter_map(|token| {
            let candidate = token.trim_matches(|c: char| {
                matches!(c, ',' | '.' | ':' | ';' | ')' | '(' | '"' | '\'' | '`')
            });
            if candidate.contains('/') {
                let ext = std::path::Path::new(candidate)
                    .extension()
                    .and_then(|e| e.to_str());
                if ext.is_some_and(|e| EXTENSIONS.iter().any(|x| e.eq_ignore_ascii_case(x))) {
                    return Some(candidate.to_string());
                }
            }
            None
        })
        .collect()
}

/// Merge an existing summary with a new one for re-compaction.
pub(crate) fn merge_summaries(existing: Option<&str>, new_summary: &str) -> String {
    let Some(existing) = existing else {
        return new_summary.to_string();
    };

    let prev_highlights = extract_highlights(existing);
    let new_formatted = format_summary(new_summary);
    let new_highlights = extract_highlights(&new_formatted);
    let new_timeline = extract_timeline(&new_formatted);

    let mut lines = vec!["<summary>".to_string(), "Conversation summary:".to_string()];

    if !prev_highlights.is_empty() {
        lines.push("- Previously compacted context:".to_string());
        for h in &prev_highlights {
            lines.push(format!("  {h}"));
        }
    }
    if !new_highlights.is_empty() {
        lines.push("- Newly compacted context:".to_string());
        for h in &new_highlights {
            lines.push(format!("  {h}"));
        }
    }
    if !new_timeline.is_empty() {
        lines.push("- Key timeline:".to_string());
        for t in &new_timeline {
            lines.push(format!("  {t}"));
        }
    }

    lines.push("</summary>".to_string());
    lines.join("\n")
}

/// Format a raw `<summary>` block into user-facing text.
pub(crate) fn format_summary(summary: &str) -> String {
    if let (Some(start), Some(end)) = (summary.find("<summary>"), summary.find("</summary>")) {
        let inner = &summary[start + 9..end];
        format!("Summary:\n{}", inner.trim())
    } else {
        summary.to_string()
    }
}

/// Build the continuation System message injected after compaction.
pub(crate) fn build_continuation_message(summary: &str, recent_preserved: bool) -> String {
    let mut text = format!("{COMPACT_PREAMBLE}{}", format_summary(summary));
    if recent_preserved {
        text.push_str("\n\n");
        text.push_str(COMPACT_RECENT_NOTE);
    }
    text.push('\n');
    text.push_str(COMPACT_RESUME_INSTRUCTION);
    text
}

/// Extract an existing summary from a System message (for re-compaction).
pub(crate) fn extract_existing_summary(message: &ConversationMessage) -> Option<String> {
    if message.role != Role::System {
        return None;
    }
    let text = message.text_content();
    let summary = text.strip_prefix(COMPACT_PREAMBLE)?;
    let summary = summary
        .split_once(&format!("\n\n{COMPACT_RECENT_NOTE}"))
        .map_or(summary, |(v, _)| v);
    let summary = summary
        .split_once(&format!("\n{COMPACT_RESUME_INSTRUCTION}"))
        .map_or(summary, |(v, _)| v);
    Some(summary.trim().to_string())
}

/// Extract highlight lines (everything except Key timeline) from formatted summary.
pub(crate) fn extract_highlights(summary: &str) -> Vec<String> {
    let mut lines = Vec::new();
    let mut in_timeline = false;
    for line in format_summary(summary).lines() {
        let trimmed = line.trim_end();
        if trimmed.is_empty() || trimmed == "Summary:" || trimmed == "Conversation summary:" {
            continue;
        }
        if trimmed == "- Key timeline:" {
            in_timeline = true;
            continue;
        }
        if in_timeline {
            continue;
        }
        lines.push(trimmed.to_string());
    }
    lines
}

/// Extract timeline lines from formatted summary.
pub(crate) fn extract_timeline(summary: &str) -> Vec<String> {
    let mut lines = Vec::new();
    let mut in_timeline = false;
    for line in format_summary(summary).lines() {
        let trimmed = line.trim_end();
        if trimmed == "- Key timeline:" {
            in_timeline = true;
            continue;
        }
        if !in_timeline {
            continue;
        }
        if trimmed.is_empty() {
            break;
        }
        lines.push(trimmed.to_string());
    }
    lines
}

/// Truncate a string at a char boundary, appending "..." if cut.
pub(crate) fn safe_truncate(s: &str, max_chars: usize) -> String {
    let char_count = s.chars().count();
    if char_count <= max_chars {
        return s.to_string();
    }
    let byte_end = s
        .char_indices()
        .nth(max_chars)
        .map(|(pos, _)| pos)
        .unwrap_or(s.len());
    format!("{}...", &s[..byte_end])
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn extract_file_candidates_finds_paths() {
        let text = "I read /home/user/file.rs and ./src/main.rs today";
        let files = extract_file_candidates(text);
        assert!(files.contains(&"/home/user/file.rs".to_string()));
        // ./src/main.rs: leading dot stripped by trim_matches
        assert!(files.iter().any(|f| f.contains("src/main.rs")));
    }

    #[test]
    fn extract_file_candidates_empty() {
        assert!(extract_file_candidates("no files here").is_empty());
    }

    #[test]
    fn merge_summaries_fresh() {
        let result = merge_summaries(None, "## Goal\nBuild something");
        assert!(result.contains("## Goal"));
        assert!(result.contains("Build something"));
    }

    #[test]
    fn merge_summaries_update() {
        let existing = "## Goal\nOld goal\n## Progress\n### Done\n- [x] step1";
        let new_info = "## Goal\nNew goal\n## Progress\n### Done\n- [x] step1\n- [x] step2";
        let result = merge_summaries(Some(existing), new_info);
        assert!(
            result.contains("step2"),
            "should include new info: {result}"
        );
    }

    #[test]
    fn safe_truncate_short() {
        assert_eq!(safe_truncate("hello", 100), "hello");
    }

    #[test]
    fn safe_truncate_cuts() {
        let result = safe_truncate("hello world", 5);
        assert!(result.len() <= 10); // 5 chars + "..."
        assert!(result.ends_with("..."));
    }

    #[test]
    fn safe_truncate_multibyte() {
        let result = safe_truncate("Привет мир", 3);
        assert!(result.ends_with("..."));
        // Should not panic on multi-byte boundary
    }

    #[test]
    fn extract_highlights_finds_done() {
        let summary = "## Progress\n### Done\n- [x] Built the thing\n- [x] Tested it\n### In Progress\n- [ ] Deploy";
        let highlights = extract_highlights(summary);
        assert!(highlights.iter().any(|h| h.contains("Built")));
    }

    #[test]
    fn extract_timeline_no_panic() {
        // extract_timeline looks for "- Key timeline:" in formatted output
        let summary = "## Goal\nTest\n## Progress\n### Done\n- [x] Step A";
        let _ = extract_timeline(summary); // just ensure no panic
    }

    #[test]
    fn format_summary_wraps() {
        let result = format_summary("test summary");
        assert!(result.contains("<summary>") || result.contains("test summary"));
    }

    #[test]
    fn build_continuation_has_summary() {
        let msg = build_continuation_message("my summary", false);
        assert!(msg.contains("my summary"));
    }
}
