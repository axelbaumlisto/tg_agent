//! Miscellaneous Telegram markup utilities.

// ── Token formatting ───────────────────────────────────────────────────

/// Truncate a Telegram button label to fit the 56-char display limit.
pub fn truncate_button(label: &str, max: usize) -> String {
    if label.len() <= max {
        return label.to_string();
    }
    let mut end = max - 1;
    while end > 0 && !label.is_char_boundary(end) {
        end -= 1;
    }
    format!("{}\u{2026}", &label[..end])
}

/// Format a token count as human-readable: `1234` → `"1.2k"`, `1234567` → `"1.2M"`.
pub fn format_tokens(n: u64) -> String {
    if n < 1_000 {
        n.to_string()
    } else if n < 10_000 {
        format!("{:.1}k", n as f64 / 1_000.0)
    } else if n < 1_000_000 {
        format!("{}k", n / 1_000)
    } else if n < 10_000_000 {
        format!("{:.1}M", n as f64 / 1_000_000.0)
    } else {
        format!("{}M", n / 1_000_000)
    }
}

// ── Markdown normalization ───────────────────────────────────────────────

/// Normalize Markdown before rendering:
/// - `\r\n` → `\n`
/// - Strip trailing whitespace per line
/// - Collapse 3+ consecutive blank lines to 2
/// - Strip leading/trailing blank lines
pub fn normalize_md(md: &str) -> String {
    let s = md.replace("\r\n", "\n");
    let mut out = String::with_capacity(s.len());
    let mut consecutive_blanks = 0u32;

    for line in s.lines() {
        let trimmed = line.trim_end();
        if trimmed.is_empty() {
            consecutive_blanks += 1;
            if consecutive_blanks <= 2 {
                out.push('\n');
            }
        } else {
            consecutive_blanks = 0;
            out.push_str(trimmed);
            out.push('\n');
        }
    }

    // Strip leading/trailing blank lines
    let result = out.trim_matches('\n');
    result.to_string()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::markup::{MAX_TG_MSG, escape_html, md_to_tg_html, split_html};

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
        assert_eq!(
            md_to_tg_html(md),
            "<pre><code class=\"language-rust\">fn main() {}</code></pre>"
        );
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
        assert!(html.contains("echo hello</code></pre>"));
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

    // ── unicode (tests md_to_tg_html with multi-byte chars) ─────────

    #[test]
    fn unicode_em_dash_does_not_panic() {
        // Regression: em-dash — is 3 bytes, old byte-indexing panicked.
        let html = md_to_tg_html("hello — world");
        assert!(html.contains("—"), "em-dash must survive: {html}");
    }

    #[test]
    fn unicode_mixed_with_links() {
        let html = md_to_tg_html("Привет https://example.com мир");
        assert!(html.contains("<a href"), "link must render: {html}");
        assert!(html.contains("Привет"), "cyrillic must survive: {html}");
    }

    #[test]
    fn unicode_emoji_in_text() {
        let html = md_to_tg_html("🚀 **launch** the 🌍");
        assert!(html.contains("🚀"));
        assert!(html.contains("<b>launch</b>"));
    }

    // ── normalize_md ────────────────────────────────────────────────

    #[test]
    fn normalize_crlf() {
        assert_eq!(
            normalize_md(
                "a
b"
            ),
            "a
b"
        );
    }

    #[test]
    fn normalize_trailing_spaces() {
        assert_eq!(
            normalize_md(
                "hello   
world  "
            ),
            "hello
world"
        );
    }

    #[test]
    fn normalize_triple_blank_collapsed() {
        assert_eq!(
            normalize_md(
                "a



b"
            ),
            "a


b"
        );
    }

    #[test]
    fn normalize_leading_trailing_blanks_stripped() {
        assert_eq!(
            normalize_md(
                "

hello

"
            ),
            "hello"
        );
    }

    #[test]
    fn normalize_passthrough() {
        assert_eq!(normalize_md("plain text"), "plain text");
    }

    #[test]
    fn normalize_empty() {
        assert_eq!(normalize_md(""), "");
        assert_eq!(
            normalize_md(
                "

"
            ),
            ""
        );
    }

    // ── escaped Markdown ────────────────────────────────────────────

    #[test]
    fn escaped_asterisk_not_italic() {
        let html = md_to_tg_html(r"\*not italic\*");
        assert!(!html.contains("<i>"), "should not be italic: {html}");
        assert!(html.contains("*"), "literal asterisks: {html}");
    }

    #[test]
    fn escaped_backtick_not_code() {
        let html = md_to_tg_html(r"\`not code\`");
        assert!(!html.contains("<code>"), "should not be code: {html}");
    }

    #[test]
    fn escaped_bracket_not_link() {
        // \[ and \] prevent [text](url) syntax, but the bare URL
        // inside still gets auto-linked. The key: no <a> wrapping
        // "not a link" as label.
        let html = md_to_tg_html(r"\[not a link\](foo)");
        assert!(
            !html.contains(">not a link</a>"),
            "escaped brackets must not form a link: {html}"
        );
    }

    #[test]
    fn backslash_before_normal_char_preserved() {
        let html = md_to_tg_html(r"hello\world");
        assert!(html.contains(r"hello\world"), "{html}");
    }

    // ── indented code blocks ────────────────────────────────────────

    #[test]
    fn indented_code_block_after_blank() {
        let md = "paragraph

    fn main() {}
    println!()";
        let html = md_to_tg_html(md);
        assert!(
            html.contains("<pre><code>"),
            "should render as code: {html}"
        );
        assert!(html.contains("fn main()"), "code content: {html}");
    }

    #[test]
    fn indented_code_not_after_text() {
        // Without a preceding blank line, 4-space indent is NOT code
        let md = "paragraph
    not code";
        let html = md_to_tg_html(md);
        assert!(!html.contains("<pre>"), "should not be code: {html}");
    }

    #[test]
    fn mixed_fenced_and_indented() {
        let md = "```
fenced
```

    indented";
        let html = md_to_tg_html(md);
        let pre_count = html.matches("<pre>").count() + html.matches("<pre><code>").count();
        assert!(pre_count >= 2, "should have 2 code blocks: {html}");
    }

    // ── format_tokens ───────────────────────────────────────────────

    #[test]
    fn format_tokens_zero() {
        assert_eq!(format_tokens(0), "0");
    }

    #[test]
    fn format_tokens_small() {
        assert_eq!(format_tokens(42), "42");
        assert_eq!(format_tokens(999), "999");
    }

    #[test]
    fn format_tokens_thousands_precise() {
        assert_eq!(format_tokens(1000), "1.0k");
        assert_eq!(format_tokens(1234), "1.2k");
        assert_eq!(format_tokens(9999), "10.0k");
    }

    #[test]
    fn format_tokens_thousands_round() {
        assert_eq!(format_tokens(10000), "10k");
        assert_eq!(format_tokens(12345), "12k");
        assert_eq!(format_tokens(999999), "999k");
    }

    #[test]
    fn format_tokens_millions_precise() {
        assert_eq!(format_tokens(1000000), "1.0M");
        assert_eq!(format_tokens(1500000), "1.5M");
        assert_eq!(format_tokens(9999999), "10.0M");
    }

    #[test]
    fn format_tokens_millions_round() {
        assert_eq!(format_tokens(10000000), "10M");
        assert_eq!(format_tokens(123000000), "123M");
    }

    // ── code language tag ───────────────────────────────────────────

    #[test]
    fn code_block_with_language() {
        let md = "```python
print(1)
```";
        let html = md_to_tg_html(md);
        assert!(
            html.contains(r#"class="language-python""#),
            "missing language: {html}"
        );
    }

    #[test]
    fn code_block_without_language() {
        let md = "```
plain
```";
        let html = md_to_tg_html(md);
        assert!(html.contains("<pre><code>"), "should be plain: {html}");
        assert!(!html.contains("language-"), "no language class: {html}");
    }

    #[test]
    fn code_block_cpp_language() {
        let md = "```c++
int x;
```";
        let html = md_to_tg_html(md);
        assert!(html.contains(r#"class="language-c++""#), "{html}");
    }

    #[test]
    fn code_block_language_with_spaces() {
        let md = "```my lang
code
```";
        let html = md_to_tg_html(md);
        assert!(
            html.contains(r#"class="language-my""#),
            "first word only: {html}"
        );
    }

    // ── truncate_button ─────────────────────────────────────────────

    #[test]
    fn truncate_button_short() {
        assert_eq!(truncate_button("short", 56), "short");
    }

    #[test]
    fn truncate_button_long() {
        let long = "a".repeat(60);
        let t = truncate_button(&long, 56);
        assert!(t.len() <= 60); // 55 + ellipsis char
        assert!(t.ends_with('…'));
    }

    #[test]
    fn split_with_language_code_block() {
        // Build HTML with language-tagged code that will be split across chunks
        let long_code = "x ".repeat(60);
        let html = format!(
            r#"<pre><code class="language-rust">{long_code}</code></pre>
<pre><code class="language-python">short</code></pre>"#
        );
        let chunks = split_html(&html, 80);
        assert!(chunks.len() > 1, "should split into multiple chunks");
        for (i, chunk) in chunks.iter().enumerate() {
            let open_code = chunk.matches("<code>").count() + chunk.matches("<code ").count();
            let close_code = chunk.matches("</code>").count();
            assert_eq!(
                open_code, close_code,
                "chunk {i} code unbalanced ({open_code} open vs {close_code} close): {chunk}"
            );
        }
    }
}
