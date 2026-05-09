//! Markdown → Telegram HTML renderer.
//!
//! Converts a subset of Markdown into the HTML fragment that Telegram's
//! `parse_mode=HTML` understands. Telegram's supported tags are:
//! `<b>`, `<i>`, `<u>`, `<s>`, `<code>`, `<pre>`, `<a>`, `<blockquote>`.
//!
//! Design: pure functions, no I/O, no Telegram API dependency.

use super::util::normalize_md;

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
    let normalized = normalize_md(md);
    let mut out = String::with_capacity(normalized.len() + 128);
    let mut lines = normalized.lines().peekable();
    let mut in_code = false;
    let mut prev_was_blank = true; // suppress leading blank

    while let Some(raw) = lines.next() {
        let trimmed = raw.trim_end();

        // ── Fenced code blocks ──────────────────────────────────────
        if trimmed.starts_with("```") {
            emit_fence_toggle(&mut out, trimmed, &mut in_code, prev_was_blank);
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
            ensure_gap(&mut out, prev_was_blank);
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
            emit_blockquote(&mut out, &mut lines, rest);
            prev_was_blank = false;
            continue;
        }

        // ── Tables → monospace ──────────────────────────────────────
        if trimmed.contains('|') && looks_like_table_row(trimmed) {
            ensure_gap(&mut out, prev_was_blank);
            emit_table(&mut out, &mut lines, trimmed);
            prev_was_blank = false;
            continue;
        }

        // ── Indented code block (4+ spaces after blank line) ─────────
        if prev_was_blank && raw.starts_with("    ") && !raw.trim_start().is_empty() {
            emit_indented_code(&mut out, &mut lines, raw);
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
        close_pre(&mut out);
    }

    // Trim trailing whitespace
    while out.ends_with('\n') || out.ends_with(' ') {
        out.pop();
    }
    out
}

// ── Block-element helpers (extracted from md_to_tg_html) ────────────────────

/// Insert a blank line gap before a block element if not already present.
fn ensure_gap(out: &mut String, prev_was_blank: bool) {
    if !prev_was_blank && !out.is_empty() {
        out.push('\n');
    }
}

/// Strip trailing newline inside `<pre>` and close the block.
fn close_pre(out: &mut String) {
    if out.ends_with('\n') {
        out.pop();
    }
    out.push_str("</code></pre>\n");
}

/// Toggle a fenced code block open/close on a ` ``` ` line.
fn emit_fence_toggle(out: &mut String, trimmed: &str, in_code: &mut bool, prev_was_blank: bool) {
    if *in_code {
        close_pre(out);
        *in_code = false;
    } else {
        ensure_gap(out, prev_was_blank);
        let lang = sanitize_code_lang(trimmed.trim_start_matches('`'));
        if lang.is_empty() {
            out.push_str("<pre><code>");
        } else {
            out.push_str(&format!("<pre><code class=\"language-{lang}\">"));
        }
        *in_code = true;
    }
}

/// Consume consecutive `> ` lines into a `<blockquote>` block.
fn emit_blockquote<'a>(
    out: &mut String,
    lines: &mut std::iter::Peekable<impl Iterator<Item = &'a str>>,
    first_line: &str,
) {
    let mut quote = String::new();
    quote.push_str(&inline_md(&escape_html(first_line.trim())));
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
}

/// Consume a markdown table into a `<pre>` block.
fn emit_table<'a>(
    out: &mut String,
    lines: &mut std::iter::Peekable<impl Iterator<Item = &'a str>>,
    first_row: &str,
) {
    out.push_str("<pre>");
    out.push_str(&format_table_row(first_row));
    out.push('\n');
    while let Some(next) = lines.peek() {
        let nt = next.trim();
        if nt.contains('|') && looks_like_table_row(nt) {
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
}

/// Consume an indented code block (4+ leading spaces) into `<pre><code>`.
fn emit_indented_code<'a>(
    out: &mut String,
    lines: &mut std::iter::Peekable<impl Iterator<Item = &'a str>>,
    first_line: &str,
) {
    if !out.is_empty() && !out.ends_with('\n') {
        out.push('\n');
    }
    out.push_str("<pre><code>");
    out.push_str(&escape_html(
        first_line.strip_prefix("    ").unwrap_or(first_line),
    ));
    out.push('\n');
    while let Some(next) = lines.peek() {
        if let Some(stripped) = next.strip_prefix("    ") {
            out.push_str(&escape_html(stripped));
            out.push('\n');
            lines.next();
        } else if next.trim().is_empty() {
            out.push('\n');
            lines.next();
        } else {
            break;
        }
    }
    close_pre(out);
}

// ─── Heading helpers ─────────────────────────────────────────────────────

/// Extract and sanitize code language from fence opening (e.g. `python` from `python\n`).
fn sanitize_code_lang(s: &str) -> String {
    s.split_whitespace()
        .next()
        .unwrap_or("")
        .chars()
        .filter(|c| c.is_ascii_alphanumeric() || "_+.-".contains(*c))
        .collect()
}

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
    // Protect backslash-escaped Markdown characters: \* \[ \_ \` \~ \>
    // Replace with Unicode private-use placeholders, run formatting,
    // then restore the literal characters.
    let protected = protect_escapes(escaped);
    let code = replace_delim(&protected, "`", "<code>", "</code>");
    let bold = replace_delim(&code, "**", "<b>", "</b>");
    let italic = replace_delim(&bold, "*", "<i>", "</i>");
    let strikethrough = replace_delim(&italic, "~~", "<s>", "</s>");
    let linked = linkify(&strikethrough);
    restore_escapes(&linked)
}

/// Replace `\X` sequences with private-use area placeholders.
fn protect_escapes(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    let mut chars = s.chars().peekable();
    while let Some(ch) = chars.next() {
        if ch == '\\'
            && let Some(&next) = chars.peek()
            && "*_`~[]>!#+-".contains(next)
        {
            let idx = "*_`~[]>!#+-".find(next).unwrap_or(0);
            out.push(char::from_u32(0xE000 + idx as u32).expect("PUA codepoint"));
            chars.next();
            continue;
        }
        out.push(ch);
    }
    out
}

/// Restore placeholders to literal characters.
fn restore_escapes(s: &str) -> String {
    let map = "*_`~[]>!#+-";
    let mut out = String::with_capacity(s.len());
    for ch in s.chars() {
        let code = ch as u32;
        if (0xE000..0xE000 + map.len() as u32).contains(&code) {
            let idx = (code - 0xE000) as usize;
            out.push(map.as_bytes()[idx] as char);
        } else {
            out.push(ch);
        }
    }
    out
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
    let mut i = 0;
    while i < s.len() {
        debug_assert!(s.is_char_boundary(i), "linkify: i={i} not on char boundary");
        let rest = &s[i..];
        // [label](url)
        if rest.starts_with('[')
            && let Some((consumed, html)) = try_parse_md_link(rest)
        {
            out.push_str(&html);
            i += consumed;
            continue;
        }

        // bare URL
        if rest.starts_with("http://") || rest.starts_with("https://") {
            let end = rest
                .find(|c: char| c.is_whitespace() || "<>\"'".contains(c))
                .unwrap_or(rest.len());
            let mut url_end = end;
            // Strip trailing ASCII punctuation
            while url_end > 0 {
                let b = rest.as_bytes()[url_end - 1];
                if matches!(b, b'.' | b',' | b')' | b';') {
                    url_end -= 1;
                } else {
                    break;
                }
            }
            if url_end == 0 {
                url_end = end;
            }
            let url = &rest[..url_end];
            out.push_str("<a href=\"");
            out.push_str(url);
            out.push_str("\">");
            out.push_str(url);
            out.push_str("</a>");
            i += url_end;
            continue;
        }
        // Regular character (may be multi-byte)
        let Some(ch) = rest.chars().next() else { break };
        out.push(ch);
        i += ch.len_utf8();
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

#[cfg(test)]
mod tests {
    use super::*;

    // ── sanitize_code_lang (private fn — must live here) ─────────────

    #[test]
    fn sanitize_code_lang_special_chars() {
        assert_eq!(sanitize_code_lang("python"), "python");
        assert_eq!(sanitize_code_lang("c++"), "c++");
        assert_eq!(sanitize_code_lang(""), "");
        assert_eq!(sanitize_code_lang("a<b>c"), "abc");
    }
}

#[cfg(test)]
mod proptests {
    //! Property-based tests for `escape_html` (T13 of
    //! PLAN_CORE_HARDENING_v2). Telegram HTML parse mode is strict:
    //! any unescaped `<`/`>`/`&`/`"` in the output silently breaks
    //! rendering. Properties pin those invariants.

    use super::*;
    use proptest::prelude::*;

    fn has_raw_html_special(s: &str) -> bool {
        let mut chars = s.chars().peekable();
        while let Some(c) = chars.next() {
            match c {
                '<' | '>' | '"' => return true,
                '&' => {
                    let rest: String = chars.clone().take(5).collect();
                    let ok = rest.starts_with("lt;")
                        || rest.starts_with("gt;")
                        || rest.starts_with("amp;")
                        || rest.starts_with("quot;");
                    if !ok {
                        return true;
                    }
                }
                _ => {}
            }
        }
        false
    }

    proptest! {
        #![proptest_config(ProptestConfig {
            cases: 200,
            ..ProptestConfig::default()
        })]

        #[test]
        fn output_has_no_raw_specials(input in "\\PC{0,200}") {
            let escaped = escape_html(&input);
            prop_assert!(
                !has_raw_html_special(&escaped),
                "escape_html left raw HTML specials in output: {escaped:?}"
            );
        }

        #[test]
        fn double_escape_still_safe(input in "\\PC{0,200}") {
            let once = escape_html(&input);
            let twice = escape_html(&once);
            prop_assert!(
                !has_raw_html_special(&twice),
                "double escape produced raw specials: {twice:?}"
            );
            if once.contains("&lt;") || once.contains("&amp;") {
                prop_assert!(twice.contains("&amp;"));
            }
        }

        #[test]
        fn safe_alnum_passes_through(input in "[a-zA-Z0-9 ]{0,80}") {
            prop_assert_eq!(escape_html(&input), input);
        }

        #[test]
        fn no_tag_injection(
            tag in "[a-z]{1,8}",
            payload in "[a-zA-Z0-9 ]{0,40}",
        ) {
            let injected = format!("<{tag}>{payload}</{tag}>");
            let escaped = escape_html(&injected);
            prop_assert!(
                !escaped.contains('<'),
                "raw '<' survived in: {escaped:?}"
            );
            prop_assert!(
                !escaped.contains('>'),
                "raw '>' survived in: {escaped:?}"
            );
            if !payload.is_empty() {
                prop_assert!(
                    escaped.contains(&payload),
                    "payload {payload:?} lost in escape: {escaped:?}"
                );
            }
        }
    }
}

#[cfg(test)]
mod snapshot_tests {
    //! Insta snapshot tests for md_to_tg_html (T12 of
    //! PLAN_CORE_HARDENING_v2). Pinning the rendering shape so any
    //! future tweak to the converter trips a visible diff.

    use super::*;

    #[test]
    fn snapshot_plain_paragraphs() {
        let md = "First paragraph.\n\nSecond paragraph with *emphasis*.";
        insta::assert_snapshot!("plain_paragraphs", md_to_tg_html(md));
    }

    #[test]
    fn snapshot_fenced_code_block() {
        let md =
            "Here's some code:\n\n```rust\nfn main() {\n    println!(\"hi\");\n}\n```\n\nDone.";
        insta::assert_snapshot!("fenced_code_block", md_to_tg_html(md));
    }

    #[test]
    fn snapshot_unordered_list_with_emphasis() {
        let md = "Tasks:\n- buy *milk*\n- read **book**\n- write `code`";
        insta::assert_snapshot!("unordered_list_with_emphasis", md_to_tg_html(md));
    }

    #[test]
    fn snapshot_link_and_inline_code() {
        let md = "See [the docs](https://example.com/path?x=1) and run `cargo test`.";
        insta::assert_snapshot!("link_and_inline_code", md_to_tg_html(md));
    }

    #[test]
    fn snapshot_html_special_chars_escaped() {
        // `<script>` should be escaped, never survive raw.
        let md = "Beware: <script>alert(1)</script> & friends \"quoted\".";
        insta::assert_snapshot!("html_special_chars_escaped", md_to_tg_html(md));
    }

    #[test]
    fn snapshot_blockquote_then_heading() {
        let md = "> a quoted line\n> another quoted line\n\n## Heading\n\nbody text.";
        insta::assert_snapshot!("blockquote_then_heading", md_to_tg_html(md));
    }

    #[test]
    fn snapshot_mixed_russian_and_english() {
        let md = "Привет! Это **тест** с `кодом`.\n\nAlso English: *italic* word.";
        insta::assert_snapshot!("mixed_russian_and_english", md_to_tg_html(md));
    }

    #[test]
    fn snapshot_escape_html_direct() {
        let s = "5 < 10 && \"quote\" with > sign";
        insta::assert_snapshot!("escape_html_direct", escape_html(s));
    }
}
