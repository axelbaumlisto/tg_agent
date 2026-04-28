//! Markdown → Telegram HTML renderer.
//!
//! Converts a subset of Markdown into the HTML fragment that Telegram's
//! `parse_mode=HTML` understands. Telegram's supported tags are:
//! `<b>`, `<i>`, `<u>`, `<s>`, `<code>`, `<pre>`, `<a>`, `<blockquote>`.
//!
//! Design: pure functions, no I/O, no Telegram API dependency.
//! Reuses inline-formatting logic patterns from `research_html.rs`
//! but targets Telegram's constrained tag set.
//!
//! # Chunking
//!
//! [`split_html`] breaks a rendered Telegram HTML string into chunks
//! of at most `MAX_TG_MSG` bytes, re-opening any tags that were split
//! across the boundary.

/// Telegram's hard message-length limit (UTF-8 bytes).
pub const MAX_TG_MSG: usize = 4096;

// ─── HTML escaping ───────────────────────────────────────────────────────

/// Escape text for use inside Telegram HTML.
pub fn escape_html(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    for ch in s.chars() {
        match ch {
            '&' => out.push_str("&amp;"),
            '<' => out.push_str("&lt;"),
            '>' => out.push_str("&gt;"),
            '"' => out.push_str("&quot;"),
            _ => out.push(ch),
        }
    }
    out
}

// ─── Markdown → Telegram HTML ────────────────────────────────────────────

/// Convert Markdown text into Telegram-compatible HTML.
///
/// Handles: headings, fenced code, blockquotes, unordered/ordered lists,
/// tables (as monospace), horizontal rules, and inline formatting
/// (bold, italic, code, links).
pub fn md_to_tg_html(md: &str) -> String {
    let mut out = String::with_capacity(md.len() + 128);
    let mut lines = md.lines().peekable();
    let mut in_code = false;
    let mut prev_was_blank = true; // suppress leading blank

    while let Some(raw) = lines.next() {
        let trimmed = raw.trim_end();

        // ── Fenced code blocks ──────────────────────────────────────
        if trimmed.starts_with("```") {
            if in_code {
                // Close: strip trailing newline inside <pre> if present
                if out.ends_with('\n') {
                    out.pop();
                }
                out.push_str("</code></pre>\n");
                in_code = false;
            } else {
                if !prev_was_blank && !out.is_empty() {
                    out.push('\n');
                }
                out.push_str("<pre><code>");
                in_code = true;
            }
            prev_was_blank = false;
            continue;
        }
        if in_code {
            out.push_str(&escape_html(raw));
            out.push('\n');
            prev_was_blank = false;
            continue;
        }

        // ── Blank lines ─────────────────────────────────────────────
        if trimmed.is_empty() {
            if !prev_was_blank && !out.is_empty() {
                out.push('\n');
            }
            prev_was_blank = true;
            continue;
        }

        // ── Headings → bold ─────────────────────────────────────────
        if let Some(rest) = strip_heading(trimmed) {
            if !prev_was_blank && !out.is_empty() {
                out.push('\n');
            }
            out.push_str("<b>");
            out.push_str(&inline_md(&escape_html(rest)));
            out.push_str("</b>\n");
            prev_was_blank = false;
            continue;
        }

        // ── Horizontal rule ─────────────────────────────────────────
        if trimmed == "---" || trimmed == "***" || trimmed == "___" {
            out.push_str("———\n");
            prev_was_blank = false;
            continue;
        }

        // ── Blockquote ──────────────────────────────────────────────
        if let Some(rest) = trimmed
            .strip_prefix("> ")
            .or_else(|| trimmed.strip_prefix(">"))
        {
            // Collect consecutive quote lines
            let mut quote = String::new();
            quote.push_str(&inline_md(&escape_html(rest.trim())));
            while let Some(next) = lines.peek() {
                let nt = next.trim();
                if let Some(qr) = nt.strip_prefix("> ").or_else(|| nt.strip_prefix(">")) {
                    quote.push('\n');
                    quote.push_str(&inline_md(&escape_html(qr.trim())));
                    lines.next();
                } else {
                    break;
                }
            }
            out.push_str("<blockquote>");
            out.push_str(&quote);
            out.push_str("</blockquote>\n");
            prev_was_blank = false;
            continue;
        }

        // ── Tables → monospace ──────────────────────────────────────
        if trimmed.contains('|') && looks_like_table_row(trimmed) {
            if !prev_was_blank && !out.is_empty() {
                out.push('\n');
            }
            out.push_str("<pre>");
            // Emit this row
            out.push_str(&format_table_row(trimmed));
            out.push('\n');
            // Consume remaining table rows
            while let Some(next) = lines.peek() {
                let nt = next.trim();
                if nt.contains('|') && looks_like_table_row(nt) {
                    // Skip separator rows (---|---|---)
                    if !is_table_separator(nt) {
                        out.push_str(&format_table_row(nt));
                        out.push('\n');
                    }
                    lines.next();
                } else {
                    break;
                }
            }
            if out.ends_with('\n') {
                out.pop();
            }
            out.push_str("</pre>\n");
            prev_was_blank = false;
            continue;
        }

        // ── Lists ───────────────────────────────────────────────────
        let ltrim = trimmed.trim_start();
        if is_unordered_list_start(ltrim) || is_ordered_list_start(ltrim) {
            emit_list(&mut out, &mut lines, trimmed);
            prev_was_blank = false;
            continue;
        }

        // ── Regular paragraph ───────────────────────────────────────
        out.push_str(&inline_md(&escape_html(trimmed)));
        out.push('\n');
        prev_was_blank = false;
    }

    // Close unclosed code block
    if in_code {
        if out.ends_with('\n') {
            out.pop();
        }
        out.push_str("</code></pre>\n");
    }

    // Trim trailing whitespace
    while out.ends_with('\n') || out.ends_with(' ') {
        out.pop();
    }
    out
}

// ─── Heading helpers ─────────────────────────────────────────────────────

fn strip_heading(line: &str) -> Option<&str> {
    if let Some(rest) = line.strip_prefix("### ") {
        Some(rest)
    } else if let Some(rest) = line.strip_prefix("## ") {
        Some(rest)
    } else if let Some(rest) = line.strip_prefix("# ") {
        Some(rest)
    } else {
        None
    }
}

// ─── Table helpers ───────────────────────────────────────────────────────

fn looks_like_table_row(line: &str) -> bool {
    let trimmed = line.trim();
    trimmed.starts_with('|') || trimmed.matches('|').count() >= 2
}

fn is_table_separator(line: &str) -> bool {
    let trimmed = line.trim().trim_matches('|');
    !trimmed.is_empty()
        && trimmed
            .chars()
            .all(|c| c == '-' || c == ':' || c == '|' || c == ' ')
}

/// Format a table row: strip outer pipes, keep inner separators.
fn format_table_row(line: &str) -> String {
    let trimmed = line.trim();
    // Strip leading/trailing |
    let inner = trimmed
        .strip_prefix('|')
        .unwrap_or(trimmed)
        .strip_suffix('|')
        .unwrap_or(trimmed);
    escape_html(inner.trim())
}

// ─── List helpers ────────────────────────────────────────────────────────

fn is_unordered_list_start(s: &str) -> bool {
    s.starts_with("- ") || s.starts_with("* ") || s.starts_with("+ ")
}

fn is_ordered_list_start(s: &str) -> bool {
    let mut chars = s.chars();
    match chars.next() {
        Some(c) if c.is_ascii_digit() => {}
        _ => return false,
    }
    for c in chars {
        if c.is_ascii_digit() {
            continue;
        }
        return (c == '.' || c == ')') && s.len() > 2;
    }
    false
}

fn emit_list(out: &mut String, lines: &mut std::iter::Peekable<std::str::Lines<'_>>, first: &str) {
    emit_list_item(out, first);
    while let Some(next) = lines.peek() {
        let nt = next.trim_end();
        let ltrim = nt.trim_start();
        if is_unordered_list_start(ltrim) || is_ordered_list_start(ltrim) {
            emit_list_item(out, nt);
            lines.next();
        } else if ltrim.is_empty() {
            break;
        } else {
            // Continuation line — append to previous
            out.push(' ');
            out.push_str(&inline_md(&escape_html(ltrim)));
            out.push('\n');
            lines.next();
        }
    }
}

fn emit_list_item(out: &mut String, line: &str) {
    let ltrim = line.trim_start();
    let indent = line.len() - ltrim.len();
    let prefix = " ".repeat(indent);

    if let Some(rest) = ltrim
        .strip_prefix("- ")
        .or_else(|| ltrim.strip_prefix("* "))
        .or_else(|| ltrim.strip_prefix("+ "))
    {
        // Task list
        if let Some(task_rest) = rest
            .strip_prefix("[x] ")
            .or_else(|| rest.strip_prefix("[X] "))
        {
            out.push_str(&format!(
                "{prefix}☑ {}\n",
                inline_md(&escape_html(task_rest))
            ));
        } else if let Some(task_rest) = rest.strip_prefix("[ ] ") {
            out.push_str(&format!(
                "{prefix}☐ {}\n",
                inline_md(&escape_html(task_rest))
            ));
        } else {
            out.push_str(&format!("{prefix}• {}\n", inline_md(&escape_html(rest))));
        }
    } else if let Some(dot_pos) = ltrim.find(". ") {
        let marker = &ltrim[..dot_pos + 1];
        let rest = &ltrim[dot_pos + 2..];
        out.push_str(&format!(
            "{prefix}{marker} {}\n",
            inline_md(&escape_html(rest))
        ));
    } else if let Some(paren_pos) = ltrim.find(") ") {
        let marker = &ltrim[..paren_pos + 1];
        let rest = &ltrim[paren_pos + 2..];
        out.push_str(&format!(
            "{prefix}{marker} {}\n",
            inline_md(&escape_html(rest))
        ));
    } else {
        out.push_str(&inline_md(&escape_html(ltrim)));
        out.push('\n');
    }
}

// ─── Inline formatting ──────────────────────────────────────────────────

/// Apply inline Markdown formatting: `code`, **bold**, *italic*, [links](url).
fn inline_md(escaped: &str) -> String {
    let code = replace_delim(escaped, "`", "<code>", "</code>");
    let bold = replace_delim(&code, "**", "<b>", "</b>");
    let italic = replace_delim(&bold, "*", "<i>", "</i>");
    let strikethrough = replace_delim(&italic, "~~", "<s>", "</s>");
    linkify(&strikethrough)
}

fn replace_delim(input: &str, delim: &str, open: &str, close: &str) -> String {
    if !input.contains(delim) {
        return input.to_string();
    }
    let mut out = String::with_capacity(input.len());
    let mut rest = input;
    let mut next_is_open = true;
    while let Some(pos) = rest.find(delim) {
        out.push_str(&rest[..pos]);
        out.push_str(if next_is_open { open } else { close });
        next_is_open = !next_is_open;
        rest = &rest[pos + delim.len()..];
    }
    out.push_str(rest);
    if !next_is_open {
        out.push_str(delim);
    }
    out
}

/// Convert `[label](url)` and bare `https://...` into `<a>` tags.
fn linkify(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    let bytes = s.as_bytes();
    let mut i = 0;
    while i < bytes.len() {
        // [label](url)
        if bytes[i] == b'['
            && let Some((consumed, html)) = try_parse_md_link(&s[i..])
        {
            out.push_str(&html);
            i += consumed;
            continue;
        }
        // bare URL
        if s[i..].starts_with("http://") || s[i..].starts_with("https://") {
            let end = s[i..]
                .find(|c: char| c.is_whitespace() || "<>\"'".contains(c))
                .map(|p| i + p)
                .unwrap_or(s.len());
            // Strip trailing punctuation that's likely not part of URL
            let mut url_end = end;
            while url_end > i && matches!(s.as_bytes()[url_end - 1], b'.' | b',' | b')' | b';') {
                url_end -= 1;
            }
            let url = &s[i..url_end];
            out.push_str("<a href=\"");
            out.push_str(url);
            out.push_str("\">");
            out.push_str(url);
            out.push_str("</a>");
            i = url_end;
            continue;
        }
        out.push(bytes[i] as char);
        i += 1;
    }
    out
}

fn try_parse_md_link(s: &str) -> Option<(usize, String)> {
    if !s.starts_with('[') {
        return None;
    }
    let label_end = s[1..].find(']')? + 1;
    let after = s.get(label_end + 1..)?;
    if !after.starts_with('(') {
        return None;
    }
    let url_end_rel = after[1..].find(')')? + 1;
    let label = &s[1..label_end];
    let url = &after[1..url_end_rel];
    // Only linkify absolute URLs
    if url.starts_with("http://") || url.starts_with("https://") || url.starts_with("mailto:") {
        let consumed = label_end + 1 + url_end_rel + 1;
        let html = format!("<a href=\"{url}\">{label}</a>");
        Some((consumed, html))
    } else {
        // Relative/unknown — degrade to plain text
        None
    }
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

    for line in html.split_inclusive('\n') {
        let prefix_len: usize = open_tags.iter().map(|t| t.len()).sum();
        let suffix_len: usize = open_tags.iter().map(|t| close_tag_for(t).len()).sum();
        let overhead = prefix_len + suffix_len;

        if current.len() + line.len() + overhead > max_bytes && !current.is_empty() {
            // Close open tags, push chunk
            let mut chunk = current.clone();
            for tag in open_tags.iter().rev() {
                chunk.push_str(&close_tag_for(tag));
            }
            chunks.push(chunk);
            // Start new chunk with re-opened tags
            current = String::new();
            for tag in &open_tags {
                current.push_str(tag);
            }
        }

        // Track tags
        track_tags(line, &mut open_tags);
        current.push_str(line);
    }

    if !current.is_empty() {
        let mut chunk = current;
        for tag in open_tags.iter().rev() {
            chunk.push_str(&close_tag_for(tag));
        }
        chunks.push(chunk);
    }

    chunks
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
                    t.strip_prefix('<')
                        .and_then(|s| s.strip_suffix('>'))
                        .or_else(|| {
                            t.strip_prefix('<')
                                .and_then(|s| s.split('>').next())
                                .and_then(|s| s.split_whitespace().next())
                        })
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

// ─── Tests ──────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── escape_html ─────────────────────────────────────────────────

    #[test]
    fn escape_html_basic() {
        assert_eq!(escape_html("a < b & c > d"), "a &lt; b &amp; c &gt; d");
        assert_eq!(escape_html("\"hello\""), "&quot;hello&quot;");
    }

    #[test]
    fn escape_html_passthrough() {
        assert_eq!(escape_html("plain text"), "plain text");
    }

    // ── Headings ────────────────────────────────────────────────────

    #[test]
    fn heading_h1_becomes_bold() {
        assert_eq!(md_to_tg_html("# Title"), "<b>Title</b>");
    }

    #[test]
    fn heading_h2_becomes_bold() {
        assert_eq!(md_to_tg_html("## Section"), "<b>Section</b>");
    }

    #[test]
    fn heading_h3_becomes_bold() {
        assert_eq!(md_to_tg_html("### Sub"), "<b>Sub</b>");
    }

    #[test]
    fn heading_with_inline_formatting() {
        assert_eq!(
            md_to_tg_html("# Hello **world**"),
            "<b>Hello <b>world</b></b>"
        );
    }

    // ── Code blocks ─────────────────────────────────────────────────

    #[test]
    fn fenced_code_block() {
        let md = "```rust\nfn main() {}\n```";
        assert_eq!(md_to_tg_html(md), "<pre><code>fn main() {}</code></pre>");
    }

    #[test]
    fn fenced_code_escapes_html() {
        let md = "```\n<script>alert(1)</script>\n```";
        assert_eq!(
            md_to_tg_html(md),
            "<pre><code>&lt;script&gt;alert(1)&lt;/script&gt;</code></pre>"
        );
    }

    #[test]
    fn unclosed_code_block_auto_closes() {
        let md = "```\nsome code";
        assert_eq!(md_to_tg_html(md), "<pre><code>some code</code></pre>");
    }

    // ── Inline formatting ───────────────────────────────────────────

    #[test]
    fn inline_bold() {
        assert_eq!(md_to_tg_html("hello **world**"), "hello <b>world</b>");
    }

    #[test]
    fn inline_italic() {
        assert_eq!(md_to_tg_html("hello *world*"), "hello <i>world</i>");
    }

    #[test]
    fn inline_code() {
        assert_eq!(
            md_to_tg_html("use `cargo test`"),
            "use <code>cargo test</code>"
        );
    }

    #[test]
    fn inline_strikethrough() {
        assert_eq!(md_to_tg_html("~~deleted~~"), "<s>deleted</s>");
    }

    #[test]
    fn unbalanced_bold_degrades() {
        // Unbalanced ** gets partially consumed: bold sees open without
        // close, re-emits **; then italic pass treats the two * as a
        // pair. Either way the text content survives.
        let result = md_to_tg_html("hello **world");
        assert!(result.contains("world"), "text must survive: {result}");
    }

    // ── Links ───────────────────────────────────────────────────────

    #[test]
    fn inline_link() {
        assert_eq!(
            md_to_tg_html("[click](https://example.com)"),
            "<a href=\"https://example.com\">click</a>"
        );
    }

    #[test]
    fn bare_url() {
        assert_eq!(
            md_to_tg_html("visit https://example.com today"),
            "visit <a href=\"https://example.com\">https://example.com</a> today"
        );
    }

    #[test]
    fn relative_link_degrades_to_text() {
        // Should NOT produce <a> for relative URLs
        assert_eq!(md_to_tg_html("[file](./readme.md)"), "[file](./readme.md)");
    }

    #[test]
    fn bare_url_strips_trailing_punctuation() {
        assert_eq!(
            md_to_tg_html("see https://example.com."),
            "see <a href=\"https://example.com\">https://example.com</a>."
        );
    }

    // ── Blockquotes ─────────────────────────────────────────────────

    #[test]
    fn blockquote_single_line() {
        assert_eq!(
            md_to_tg_html("> quoted text"),
            "<blockquote>quoted text</blockquote>"
        );
    }

    #[test]
    fn blockquote_multi_line() {
        assert_eq!(
            md_to_tg_html("> line one\n> line two"),
            "<blockquote>line one\nline two</blockquote>"
        );
    }

    // ── Horizontal rules ────────────────────────────────────────────

    #[test]
    fn horizontal_rule() {
        assert_eq!(md_to_tg_html("---"), "———");
    }

    // ── Lists ───────────────────────────────────────────────────────

    #[test]
    fn unordered_list() {
        let md = "- one\n- two\n- three";
        let html = md_to_tg_html(md);
        assert!(html.contains("• one"));
        assert!(html.contains("• two"));
        assert!(html.contains("• three"));
    }

    #[test]
    fn ordered_list() {
        let md = "1. first\n2. second";
        let html = md_to_tg_html(md);
        assert!(html.contains("1. first"));
        assert!(html.contains("2. second"));
    }

    #[test]
    fn task_list_checked() {
        assert_eq!(md_to_tg_html("- [x] done"), "☑ done");
    }

    #[test]
    fn task_list_unchecked() {
        assert_eq!(md_to_tg_html("- [ ] todo"), "☐ todo");
    }

    // ── Tables ──────────────────────────────────────────────────────

    #[test]
    fn table_renders_monospace() {
        let md = "| Name | Price |\n|---|---|\n| A | $10 |";
        let html = md_to_tg_html(md);
        assert!(html.starts_with("<pre>"));
        assert!(html.contains("Name"));
        assert!(html.contains("Price"));
        assert!(!html.contains("---")); // separator stripped
        assert!(html.ends_with("</pre>"));
    }

    #[test]
    fn table_strips_outer_pipes() {
        let md = "| A | B |\n|---|---|\n| 1 | 2 |";
        let html = md_to_tg_html(md);
        // Inner content should not start/end with |
        let inside = html
            .strip_prefix("<pre>")
            .unwrap()
            .strip_suffix("</pre>")
            .unwrap();
        for line in inside.lines() {
            assert!(!line.starts_with('|'), "line starts with pipe: {line}");
            assert!(!line.ends_with('|'), "line ends with pipe: {line}");
        }
    }

    // ── Block spacing ───────────────────────────────────────────────

    #[test]
    fn heading_then_paragraph_has_blank_line() {
        let md = "# Title\nSome text";
        let html = md_to_tg_html(md);
        assert_eq!(html, "<b>Title</b>\nSome text");
    }

    #[test]
    fn paragraph_blank_paragraph() {
        let md = "first\n\nsecond";
        let html = md_to_tg_html(md);
        assert_eq!(html, "first\n\nsecond");
    }

    #[test]
    fn no_leading_blank_line() {
        let md = "\n\n# Title";
        let html = md_to_tg_html(md);
        assert_eq!(html, "<b>Title</b>");
    }

    // ── Composite ───────────────────────────────────────────────────

    #[test]
    fn realistic_agent_response() {
        let md = "\
# Summary

Here is the **result**:

```bash
echo hello
```

- Item one
- Item two

See [docs](https://docs.rs) for details.";

        let html = md_to_tg_html(md);
        assert!(html.contains("<b>Summary</b>"));
        assert!(html.contains("<b>result</b>"));
        assert!(html.contains("<pre><code>echo hello</code></pre>"));
        assert!(html.contains("• Item one"));
        assert!(html.contains("<a href=\"https://docs.rs\">docs</a>"));
    }

    // ── split_html ──────────────────────────────────────────────────

    #[test]
    fn short_message_no_split() {
        let html = "hello world";
        let chunks = split_html(html, MAX_TG_MSG);
        assert_eq!(chunks.len(), 1);
        assert_eq!(chunks[0], "hello world");
    }

    #[test]
    fn split_at_newline_boundary() {
        let line = "x".repeat(40);
        let html = format!("{line}\n{line}\n{line}");
        let chunks = split_html(&html, 90);
        assert!(chunks.len() >= 2);
        for chunk in &chunks {
            assert!(chunk.len() <= 90, "chunk too long: {}", chunk.len());
        }
    }

    #[test]
    fn split_reopens_pre_tag() {
        let code = "x\n".repeat(100);
        let html = format!("<pre><code>{code}</code></pre>");
        let chunks = split_html(&html, 200);
        assert!(chunks.len() >= 2);
        // Every chunk should have balanced pre/code
        for chunk in &chunks {
            assert!(
                chunk.contains("<pre>") || chunk.contains("<code>"),
                "chunk missing open tag: {chunk}"
            );
        }
    }

    #[test]
    fn split_preserves_all_content() {
        let line_a = "a".repeat(50);
        let line_b = "b".repeat(50);
        let line_c = "c".repeat(50);
        let html = format!("{line_a}\n{line_b}\n{line_c}");
        let chunks = split_html(&html, 60);
        let joined: String = chunks.join("");
        assert!(joined.contains(&line_a));
        assert!(joined.contains(&line_b));
        assert!(joined.contains(&line_c));
    }

    // ── close_tag_for ───────────────────────────────────────────────

    #[test]
    fn close_tag_simple() {
        assert_eq!(close_tag_for("<b>"), "</b>");
        assert_eq!(close_tag_for("<pre><code>"), "</code></pre>");
    }

    #[test]
    fn close_tag_with_attr() {
        assert_eq!(close_tag_for("<a href=\"x\">"), "</a>");
    }
}
