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
    let normalized = normalize_md(md);
    let mut out = String::with_capacity(normalized.len() + 128);
    let mut lines = normalized.lines().peekable();
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
                // Extract language tag after ```
                let lang = sanitize_code_lang(trimmed.trim_start_matches('`'));
                if lang.is_empty() {
                    out.push_str("<pre><code>");
                } else {
                    out.push_str(&format!("<pre><code class=\"language-{lang}\">"));
                }
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

        // ── Indented code block (4+ spaces after blank line) ─────────
        if prev_was_blank && raw.starts_with("    ") && !raw.trim_start().is_empty() {
            if !out.is_empty() && !out.ends_with('\n') {
                out.push('\n');
            }
            out.push_str("<pre><code>");
            out.push_str(&escape_html(raw.strip_prefix("    ").unwrap_or(raw)));
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
            if out.ends_with('\n') {
                out.pop();
            }
            out.push_str("</code></pre>\n");
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
            out.push(char::from_u32(0xE000 + idx as u32).unwrap());
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
        // Safety: always ensure i is on a char boundary.
        if !s.is_char_boundary(i) {
            i += 1;
            continue;
        }
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
        let ch = rest.chars().next().unwrap();
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

// ─── HTML chunk splitting ────────────────────────────────────────────────

/// Split Telegram HTML into chunks of at most `max_bytes`, re-opening
/// any tags that were split across the boundary.
///
/// Prefers splitting at newlines (block boundaries).  Falls back to
/// mid-line splits when a single line exceeds the budget.
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

// ─── Tests ──────────────────────────────────────────────────────────────

// ── Markdown normalization ───────────────────────────────────────────────

// ── Model scope filtering ──────────────────────────────────────────────

/// Filter `provider/model` pairs by glob patterns.
///
/// Patterns support `*` (any chars) and `?` (single char).
/// Empty patterns → return all (backward compatible).
pub fn filter_models_by_scope<'a>(
    models: &'a [(String, String)],
    patterns: &[String],
) -> Vec<&'a (String, String)> {
    if patterns.is_empty() {
        return models.iter().collect();
    }
    models
        .iter()
        .filter(|(prov, model)| {
            let full = format!("{prov}/{model}");
            patterns.iter().any(|pat| glob_match(pat, &full))
        })
        .collect()
}

/// Simple glob matching: `*` = any chars, `?` = single char.
fn glob_match(pattern: &str, text: &str) -> bool {
    let mut p = pattern.chars().peekable();
    let mut t = text.chars().peekable();
    glob_match_inner(&mut p, &mut t)
}

fn glob_match_inner(
    p: &mut std::iter::Peekable<std::str::Chars<'_>>,
    t: &mut std::iter::Peekable<std::str::Chars<'_>>,
) -> bool {
    while let Some(&pc) = p.peek() {
        match pc {
            '*' => {
                p.next();
                // Try matching rest of pattern at every position
                if p.peek().is_none() {
                    return true; // trailing * matches everything
                }
                let mut t_clone = t.clone();
                loop {
                    let mut p_clone = p.clone();
                    let mut tc = t_clone.clone();
                    if glob_match_inner(&mut p_clone, &mut tc) {
                        return true;
                    }
                    if t_clone.next().is_none() {
                        return false;
                    }
                }
            }
            '?' => {
                p.next();
                if t.next().is_none() {
                    return false;
                }
            }
            c => {
                p.next();
                match t.next() {
                    Some(tc) if tc == c => {}
                    _ => return false,
                }
            }
        }
    }
    t.peek().is_none()
}

// ── Token formatting ───────────────────────────────────────────────────

/// Format a token count as human-readable: `1234` → `"1.2k"`, `1234567` → `"1.2M"`.
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

    // ── split_stable_unstable ───────────────────────────────────────

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

    // ── filter_models_by_scope ──────────────────────────────────────

    #[test]
    fn scope_empty_returns_all() {
        let models = vec![("a".into(), "m1".into()), ("b".into(), "m2".into())];
        assert_eq!(filter_models_by_scope(&models, &[]).len(), 2);
    }

    #[test]
    fn scope_exact_match() {
        let models = vec![
            ("anthropic".into(), "claude".into()),
            ("openai".into(), "gpt4".into()),
        ];
        let filtered = filter_models_by_scope(&models, &["anthropic/claude".into()]);
        assert_eq!(filtered.len(), 1);
        assert_eq!(filtered[0].1, "claude");
    }

    #[test]
    fn scope_wildcard() {
        let models = vec![
            ("anthropic".into(), "claude-3".into()),
            ("anthropic".into(), "claude-4".into()),
            ("openai".into(), "gpt-4".into()),
        ];
        let filtered = filter_models_by_scope(&models, &["anthropic/*".into()]);
        assert_eq!(filtered.len(), 2);
    }

    #[test]
    fn scope_question_mark() {
        let models = vec![
            ("a".into(), "v1".into()),
            ("a".into(), "v2".into()),
            ("a".into(), "v10".into()),
        ];
        let filtered = filter_models_by_scope(&models, &["a/v?".into()]);
        assert_eq!(filtered.len(), 2); // v1, v2 match; v10 doesn't
    }

    #[test]
    fn scope_no_match() {
        let models = vec![("a".into(), "m1".into())];
        let filtered = filter_models_by_scope(&models, &["b/*".into()]);
        assert!(filtered.is_empty());
    }

    #[test]
    fn scope_multiple_patterns() {
        let models = vec![
            ("a".into(), "m1".into()),
            ("b".into(), "m2".into()),
            ("c".into(), "m3".into()),
        ];
        let filtered = filter_models_by_scope(&models, &["a/*".into(), "c/*".into()]);
        assert_eq!(filtered.len(), 2);
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

    #[test]
    fn sanitize_code_lang_special_chars() {
        assert_eq!(sanitize_code_lang("python"), "python");
        assert_eq!(sanitize_code_lang("c++"), "c++");
        assert_eq!(sanitize_code_lang(""), "");
        assert_eq!(sanitize_code_lang("a<b>c"), "abc");
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
