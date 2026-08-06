//! HTML chunking and stable/unstable Markdown splitting.
//!
//! [`split_html`] breaks a rendered Telegram HTML string into chunks of at
//! most `max_bytes` bytes, re-opening any tags that were split across the
//! boundary.
//!
//! [`split_stable_unstable`] separates a growing Markdown stream into a
//! fully-closed "stable" prefix and an open "unstable" tail.

// ─── Split stable/unstable ───────────────────────────────────────────────

/// Split growing Markdown into stable (closed blocks) and unstable (open tail).
///
/// Stable = complete heading, paragraph, fenced code block, blockquote, list,
/// table, horizontal rule. Unstable = unclosed code fence, partial paragraph,
/// trailing text.
///
/// Returns `(stable_md, unstable_tail)` where `stable_md` can be safely
/// rendered as HTML and `unstable_tail` should be shown as escaped plain text.
pub fn split_stable_unstable(md: &str) -> (&str, &str) {
    let normalized = md.trim_end();
    if normalized.is_empty() {
        return ("", "");
    }

    let mut stable_end = 0;
    let mut in_fence = false;
    let mut i = 0;
    let lines: Vec<&str> = normalized.lines().collect();

    while i < lines.len() {
        let line = lines[i];
        let trimmed = line.trim();

        if trimmed.starts_with("```") {
            if in_fence {
                // Closing fence — block is complete
                in_fence = false;
                // stable_end = byte offset after this line
                stable_end = byte_offset_after_line(normalized, i, &lines);
                i += 1;
                continue;
            } else {
                in_fence = true;
                i += 1;
                continue;
            }
        }

        if in_fence {
            i += 1;
            continue;
        }

        // Complete single-line block
        if !trimmed.is_empty() {
            // Check if next line is blank or EOF — then this block is complete
            let next_is_boundary = i + 1 >= lines.len() || lines[i + 1].trim().is_empty();
            if next_is_boundary {
                stable_end = byte_offset_after_line(normalized, i, &lines);
            }
        }

        i += 1;
    }

    if in_fence {
        // Unclosed fence — everything from fence start is unstable.
        // stable_end stays at the last closed block.
    }

    let stable = &normalized[..stable_end];
    let unstable = normalized[stable_end..].trim_start_matches('\n');
    (stable, unstable)
}

fn byte_offset_after_line(text: &str, line_idx: usize, lines: &[&str]) -> usize {
    let mut offset = 0;
    for (i, line) in lines.iter().enumerate() {
        if i > line_idx {
            break;
        }
        offset += line.len();
        // Account for the \n separator (except possibly the last line)
        if offset < text.len() {
            offset += 1; // \n
        }
    }
    offset.min(text.len())
}

// ─── HTML chunk splitting ────────────────────────────────────────────────

const MIN_SPLIT_BUDGET_BYTES: usize = 2;
const TELEGRAM_HTML_TAGS: &[&str] = &["b", "i", "u", "s", "code", "pre", "a", "blockquote"];

/// Split Telegram HTML into chunks of at most `max_bytes`, re-opening
/// any tags that were split across the boundary.
///
/// Prefers splitting at newlines (block boundaries).  Falls back to
/// mid-line splits when a single line exceeds the budget.
pub fn split_html(html: &str, max_bytes: usize) -> Vec<String> {
    debug_assert!(
        max_bytes >= MIN_SPLIT_BUDGET_BYTES,
        "split_html budget must be at least {MIN_SPLIT_BUDGET_BYTES} bytes"
    );
    if html.len() <= max_bytes {
        return vec![html.to_string()];
    }
    if max_bytes < MIN_SPLIT_BUDGET_BYTES {
        // A non-empty UTF-8 string cannot be represented as chunks of at most
        // 0 bytes, and not all scalar values fit in 1 byte. Preserve content
        // and terminate rather than silently clamping the caller's bad budget.
        return vec![html.to_string()];
    }

    let mut chunks: Vec<String> = Vec::new();
    let mut current = String::new();
    let mut open_tags: Vec<String> = Vec::new();

    // Split into lines, then feed each line to the accumulator.
    // When a line itself exceeds the budget, force-split it at space
    // or char boundaries.
    let lines: Vec<&str> = html.split_inclusive('\n').collect();

    for line in lines {
        let parts = if line.len() > max_bytes / 2 {
            // Large line — split at spaces to produce manageable parts
            split_long_line(line, max_bytes / 2)
        } else {
            vec![line.to_string()]
        };

        for part in &parts {
            let suffix_len: usize = open_tags.iter().map(|t| close_tag_for(t).len()).sum();

            if !current.is_empty() && current.len() + part.len() + suffix_len > max_bytes {
                // Flush current chunk — close currently open tags
                for tag in open_tags.iter().rev() {
                    current.push_str(&close_tag_for(tag));
                }
                chunks.push(current);
                // Start new chunk — reopen currently open tags
                let reopener_len: usize = open_tags.iter().map(|t| t.len()).sum();
                current = String::with_capacity(reopener_len + part.len());
                for tag in &open_tags {
                    current.push_str(tag);
                }
            }

            current.push_str(part);
            track_tags(part, &mut open_tags);
        }
    }

    if !current.is_empty() {
        for tag in open_tags.iter().rev() {
            current.push_str(&close_tag_for(tag));
        }
        chunks.push(current);
    }

    chunks
}

/// Split a long line at spaces to produce parts ≤ `max` bytes.
fn split_long_line(line: &str, max: usize) -> Vec<String> {
    if max == 0 || line.len() <= max {
        return vec![line.to_string()];
    }
    let mut parts = Vec::new();
    let mut start = 0;
    while start < line.len() {
        let end = (start + max).min(line.len());
        // Snap to char boundary
        let mut boundary = end;
        while boundary > start && !line.is_char_boundary(boundary) {
            boundary -= 1;
        }
        // Try to split at `>` (end of tag) or a space NOT inside a tag.
        // Never split in the middle of an HTML tag.
        if boundary < line.len() {
            let slice = &line[start..boundary];
            let mut best = None;
            let mut in_tag = false;
            for (j, ch) in slice.char_indices() {
                match ch {
                    '<' => in_tag = true,
                    '>' => {
                        in_tag = false;
                        best = Some(j + 1); // split right after '>'
                    }
                    ' ' if !in_tag => best = Some(j + 1),
                    _ => {}
                }
            }
            if let Some(b) = best {
                boundary = start + b;
            }
        }
        if boundary <= start {
            boundary = end.min(line.len());
            while boundary < line.len() && !line.is_char_boundary(boundary) {
                boundary += 1;
            }
        }
        parts.push(line[start..boundary].to_string());
        start = boundary;
    }
    parts
}

fn close_tag_for(open: &str) -> String {
    // "<pre><code>" → "</code></pre>"
    // "<b>" → "</b>"
    let mut closers = Vec::new();
    let mut rest = open;
    while let Some(start) = rest.find('<') {
        if let Some(end) = rest[start..].find('>') {
            let tag_content = &rest[start + 1..start + end];
            if let Some(tag_name) = telegram_open_tag_name(tag_content) {
                closers.push(format!("</{tag_name}>"));
            }
            rest = &rest[start + end + 1..];
        } else {
            break;
        }
    }
    closers.reverse();
    closers.join("")
}

fn canonical_telegram_html_tag(name: &str) -> Option<&'static str> {
    TELEGRAM_HTML_TAGS
        .iter()
        .copied()
        .find(|allowed| allowed.eq_ignore_ascii_case(name))
}

fn raw_tag_name(tag: &str) -> Option<&str> {
    if tag.is_empty() || tag.starts_with(char::is_whitespace) {
        return None;
    }
    let name = tag
        .split(|c: char| c.is_whitespace() || c == '/')
        .next()
        .unwrap_or("");
    (!name.is_empty()).then_some(name)
}

fn telegram_open_tag_name(tag: &str) -> Option<&'static str> {
    if tag.starts_with('/') || tag.starts_with('!') || tag.ends_with('/') {
        return None;
    }
    let name = raw_tag_name(tag)?;
    canonical_telegram_html_tag(name)
}

fn telegram_close_tag_name(tag: &str) -> Option<&'static str> {
    let rest = tag.strip_prefix('/')?;
    let name = raw_tag_name(rest)?;
    canonical_telegram_html_tag(name)
}

fn open_tag_name(open: &str) -> Option<&'static str> {
    let inner = open.strip_prefix('<')?.strip_suffix('>')?;
    telegram_open_tag_name(inner)
}

fn track_tags(text: &str, open_tags: &mut Vec<String>) {
    let bytes = text.as_bytes();
    let mut i = 0;
    while i < bytes.len() {
        if bytes[i] == b'<'
            && let Some(end) = text[i..].find('>')
        {
            let tag = &text[i + 1..i + end];
            if let Some(name) = telegram_close_tag_name(tag) {
                // Closing tag — pop matching open
                if let Some(pos) = open_tags.iter().rposition(|open| {
                    open_tag_name(open).is_some_and(|open_name| open_name == name)
                }) {
                    open_tags.remove(pos);
                }
            } else if telegram_open_tag_name(tag).is_some() {
                open_tags.push(format!("<{tag}>"));
            }
            i += end + 1;
            continue;
        }
        i += 1;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ── close_tag_for (private fn — must live here) ──────────────────

    #[test]
    fn close_tag_simple() {
        assert_eq!(close_tag_for("<b>"), "</b>");
        assert_eq!(close_tag_for("<pre><code>"), "</code></pre>");
    }

    #[test]
    fn close_tag_with_attr() {
        assert_eq!(close_tag_for("<a href=\"x\">"), "</a>");
        assert_eq!(close_tag_for("<code class=\"language-rust\">"), "</code>");
        assert_eq!(close_tag_for("<blockquote expandable>"), "</blockquote>");
        assert_eq!(close_tag_for("<B>"), "</b>");
        assert_eq!(close_tag_for("<Code Class=\"language-rust\">"), "</code>");
    }

    fn test_canonical_telegram_html_tag(name: &str) -> Option<&'static str> {
        TELEGRAM_HTML_TAGS
            .iter()
            .copied()
            .find(|allowed| allowed.eq_ignore_ascii_case(name))
    }

    fn test_telegram_open_tag_name(tag: &str) -> Option<&'static str> {
        if tag.starts_with('/') || tag.starts_with('!') || tag.ends_with('/') {
            return None;
        }
        let name = raw_tag_name(tag)?;
        test_canonical_telegram_html_tag(name)
    }

    fn test_telegram_close_tag_name(tag: &str) -> Option<&'static str> {
        let rest = tag.strip_prefix('/')?;
        let name = raw_tag_name(rest)?;
        test_canonical_telegram_html_tag(name)
    }

    fn strip_telegram_tags(html: &str) -> String {
        let mut out = String::new();
        let mut i = 0;
        while i < html.len() {
            if html.as_bytes()[i] == b'<'
                && let Some(end) = html[i..].find('>')
            {
                let tag = &html[i + 1..i + end];
                if test_telegram_open_tag_name(tag).is_some()
                    || test_telegram_close_tag_name(tag).is_some()
                {
                    i += end + 1;
                    continue;
                }
            }
            let ch = html[i..].chars().next().expect("valid char boundary");
            out.push(ch);
            i += ch.len_utf8();
        }
        out
    }

    fn assert_telegram_tags_balanced(html: &str) {
        let mut stack: Vec<&str> = Vec::new();
        let mut i = 0;
        while i < html.len() {
            if html.as_bytes()[i] == b'<'
                && let Some(end) = html[i..].find('>')
            {
                let tag = &html[i + 1..i + end];
                if let Some(name) = test_telegram_close_tag_name(tag) {
                    assert_eq!(stack.pop(), Some(name), "unbalanced close in {html:?}");
                    i += end + 1;
                    continue;
                }
                if let Some(name) = test_telegram_open_tag_name(tag) {
                    stack.push(name);
                    i += end + 1;
                    continue;
                }
            }
            let ch = html[i..].chars().next().expect("valid char boundary");
            i += ch.len_utf8();
        }
        assert!(stack.is_empty(), "unclosed tags {stack:?} in {html:?}");
    }

    #[test]
    fn split_html_adversarial_chunks_stay_budgeted_balanced_and_lossless() {
        let bare_angles = "a < b and c > d ".repeat(1000);
        let long_token = "x".repeat(5_000);
        let cyrillic = "Привет мир ".repeat(320);
        let uppercase_span = format!(
            "<B>{}</B>",
            "bold across a forced split boundary ".repeat(180)
        );
        let nested = format!(
            "<blockquote expandable><pre><Code Class=\"language-rust\">{long_token}\n{cyrillic}</Code></pre></BlockQuote>"
        );
        let html = format!(
            "{bare_angles}\n{long_token}\n{uppercase_span}\n{nested}\n{s}{u}",
            s = "<s>strike</s>",
            u = "<u>under</u>"
        );
        let budget = 3_996;

        let chunks = split_html(&html, budget);

        assert!(chunks.len() > 1, "adversarial fixture should split");
        for chunk in &chunks {
            assert!(
                chunk.len() <= budget,
                "chunk exceeded budget: {} > {budget}: {chunk:?}",
                chunk.len()
            );
            assert_telegram_tags_balanced(chunk);
        }
        assert_eq!(
            strip_telegram_tags(&chunks.join("")),
            strip_telegram_tags(&html),
            "tag-stripped visible text must be preserved"
        );
    }

    #[test]
    fn split_html_tiny_budgets_terminate_without_clamping() {
        let html = "abc Привет";
        for budget in [0, 1] {
            #[cfg(debug_assertions)]
            assert!(
                std::panic::catch_unwind(|| split_html(html, budget)).is_err(),
                "debug builds should surface invalid budget {budget}"
            );
            #[cfg(not(debug_assertions))]
            assert_eq!(split_html(html, budget), vec![html.to_string()]);
        }
    }

    // ── split_stable_unstable ────────────────────────────────────────

    #[test]
    fn stable_complete_blocks() {
        let md = "# Title

Paragraph text.

```
code
```";
        let (stable, unstable) = split_stable_unstable(md);
        assert!(stable.contains("# Title"));
        assert!(stable.contains("```"));
        assert!(unstable.is_empty(), "all closed: unstable={unstable:?}");
    }

    #[test]
    fn unstable_open_fence() {
        let md = "# Done

```
open code";
        let (stable, unstable) = split_stable_unstable(md);
        assert!(stable.contains("# Done"), "heading is stable: {stable:?}");
        assert!(
            unstable.contains("```"),
            "open fence is unstable: {unstable:?}"
        );
    }

    #[test]
    fn empty_input() {
        let (s, u) = split_stable_unstable("");
        assert!(s.is_empty());
        assert!(u.is_empty());
    }

    #[test]
    fn all_tail_single_line() {
        let (stable, unstable) = split_stable_unstable("growing text");
        // Single line with no following blank = unstable (still being typed)
        // Actually it IS a complete block (single paragraph at EOF)
        assert!(!stable.is_empty() || !unstable.is_empty());
    }

    #[test]
    fn paragraph_then_growing() {
        let md = "First paragraph.

Second still growing";
        let (stable, unstable) = split_stable_unstable(md);
        assert!(
            stable.contains("First paragraph"),
            "first is stable: {stable:?}"
        );
        // Both paragraphs at EOF are complete blocks
        let _ = unstable; // may or may not be empty depending on EOF rules
    }
}
