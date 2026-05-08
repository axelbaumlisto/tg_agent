//! HTML → plain-text extraction for `web_fetch`.
//!
//! Minimal HTML → text stripper. Not a DOM parser — we explicitly
//! don't rely on one because the output only needs to be "readable by an
//! LLM", and a 300 KB page with nested `<script>` tags would otherwise
//! require `scraper` / `html5ever` (+1.5 MB binary bloat).
//!
//! Strategy:
//! 1. Drop `<script>...</script>` and `<style>...</style>` blocks wholesale.
//! 2. Extract `href` values from `<a>` tags into a separate list (preserved
//!    so the research agent can save them without re-fetching).
//! 3. Replace any remaining `<tag>` with a single space.
//! 4. Collapse runs of whitespace to one space / one newline.

/// Convert HTML to plain text + extracted link list.
///
/// Exposed `pub(crate)` so `web_fetch_tls` / `web_fetch_wayback` can reuse
/// it — consistent text shape across backends keeps the agent's prompt
/// handling trivial.
pub(crate) fn html_to_text(html: &str) -> (String, Vec<String>) {
    let cleaned = drop_block(html, "script");
    let cleaned = drop_block(&cleaned, "style");
    let links = extract_hrefs(&cleaned);
    let mut out = String::with_capacity(cleaned.len());
    let mut in_tag = false;
    for ch in cleaned.chars() {
        match ch {
            '<' => in_tag = true,
            '>' => {
                in_tag = false;
                out.push(' ');
            }
            _ if in_tag => {}
            _ => out.push(ch),
        }
    }
    let decoded = decode_basic_entities(&out);
    let collapsed = collapse_whitespace(&decoded);
    (collapsed, links)
}

/// Drop all `<tag ...>...</tag>` blocks, case-insensitive.
pub(super) fn drop_block(input: &str, tag: &str) -> String {
    let lower = input.to_ascii_lowercase();
    let open = format!("<{tag}");
    let close = format!("</{tag}>");
    let mut out = String::with_capacity(input.len());
    let mut i = 0;
    while i < input.len() {
        match lower[i..].find(&open) {
            Some(rel) => {
                let start = i + rel;
                out.push_str(&input[i..start]);
                // Find end of the opening `<tag ...>` then `</tag>`.
                let after_open = match input[start..].find('>') {
                    Some(p) => start + p + 1,
                    None => break,
                };
                match lower[after_open..].find(&close) {
                    Some(end_rel) => {
                        i = after_open + end_rel + close.len();
                    }
                    None => break,
                }
            }
            None => {
                out.push_str(&input[i..]);
                break;
            }
        }
    }
    out
}

/// Extract href values from `<a ... href="..."...>`. Naïve but good enough for
/// a stripped HTML body. Preserves order, deduplicates, skips fragments /
/// javascript: URLs.
pub(super) fn extract_hrefs(input: &str) -> Vec<String> {
    let lower = input.to_ascii_lowercase();
    let mut out = Vec::new();
    let mut seen = std::collections::HashSet::new();
    let mut i = 0;
    while i < input.len() {
        let Some(rel) = lower[i..].find("<a") else {
            break;
        };
        let start = i + rel;
        let Some(close) = input[start..].find('>') else {
            break;
        };
        let tag_slice = &input[start..start + close];
        if let Some(href) = find_attr(tag_slice, "href")
            && !href.is_empty()
            && !href.starts_with('#')
            && !href.to_ascii_lowercase().starts_with("javascript:")
            && seen.insert(href.clone())
        {
            out.push(href);
        }
        i = start + close + 1;
    }
    out
}

fn find_attr(tag: &str, name: &str) -> Option<String> {
    let lower = tag.to_ascii_lowercase();
    let needle = format!("{name}=");
    let pos = lower.find(&needle)?;
    let after = pos + needle.len();
    let tag_bytes = tag.as_bytes();
    let quote = *tag_bytes.get(after)?;
    let (open, end_at) = if quote == b'"' || quote == b'\'' {
        (
            after + 1,
            tag[after + 1..]
                .find(quote as char)
                .map(|p| after + 1 + p)?,
        )
    } else {
        let rest = &tag[after..];
        let stop: &[char] = &[' ', '\t', '>', '\n', '\r'];
        let stop_rel = rest.find(stop).unwrap_or(rest.len());
        (after, after + stop_rel)
    };
    Some(tag[open..end_at].to_string())
}

fn decode_basic_entities(s: &str) -> String {
    // Only the handful that break readability. A full entity decode would
    // require a dep table; leave anything exotic as-is.
    s.replace("&amp;", "&")
        .replace("&lt;", "<")
        .replace("&gt;", ">")
        .replace("&quot;", "\"")
        .replace("&apos;", "'")
        .replace("&nbsp;", " ")
}

fn collapse_whitespace(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    let mut last_newline = false;
    let mut last_space = false;
    for ch in s.chars() {
        if ch == '\n' || ch == '\r' {
            if !last_newline {
                out.push('\n');
                last_newline = true;
                last_space = true;
            }
        } else if ch.is_whitespace() {
            if !last_space {
                out.push(' ');
                last_space = true;
            }
        } else {
            out.push(ch);
            last_newline = false;
            last_space = false;
        }
    }
    out.trim().to_string()
}
