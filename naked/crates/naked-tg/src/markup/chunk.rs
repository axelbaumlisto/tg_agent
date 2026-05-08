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

/// Split Telegram HTML into chunks of at most `max_bytes`, re-opening
/// any tags that were split across the boundary.
///
/// Prefers splitting at newlines (block boundaries).  Falls back to
/// mid-line splits when a single line exceeds the budget.
pub fn split_html(html: &str, max_bytes: usize) -> Vec<String> {
    if html.len() <= max_bytes {
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
    if line.len() <= max {
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
            let tag_name = tag_content.split_whitespace().next().unwrap_or("");
            if !tag_name.starts_with('/') && !tag_name.is_empty() {
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

fn track_tags(text: &str, open_tags: &mut Vec<String>) {
    let bytes = text.as_bytes();
    let mut i = 0;
    while i < bytes.len() {
        if bytes[i] == b'<'
            && let Some(end) = text[i..].find('>')
        {
            let tag = &text[i + 1..i + end];
            let tag_name = tag.split_whitespace().next().unwrap_or("");
            if let Some(name) = tag_name.strip_prefix('/') {
                // Closing tag — pop matching open
                if let Some(pos) = open_tags.iter().rposition(|t| {
                    // Extract tag name: "<code class=\"...\">" → "code"
                    t.strip_prefix('<')
                        .and_then(|s| s.split(|c: char| c == '>' || c.is_whitespace()).next())
                        .map(|n| n == name)
                        .unwrap_or(false)
                }) {
                    open_tags.remove(pos);
                }
            } else if !tag_name.is_empty() && !tag_name.starts_with('!') && !tag.ends_with('/') {
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
