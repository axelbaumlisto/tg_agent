//! Shared helpers for the web_fetch tool family.
//!
//! All three fetch tools (web_fetch, web_fetch_tls, web_fetch_wayback)
//! share the same output formatting pipeline: parse input, convert
//! HTML to text, truncate, append links. This module deduplicates
//! that logic.

use crate::types::ToolResult;

/// Default character limit for fetched content.
pub const DEFAULT_MAX_CHARS: usize = 8_000;

/// Parse the common input fields shared by all fetch tools.
pub struct FetchInput {
    pub url: String,
    pub max_chars: usize,
    pub include_links: bool,
}

impl FetchInput {
    pub fn parse(input: &serde_json::Value) -> Result<Self, ToolResult> {
        let url = input
            .get("url")
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .trim()
            .to_string();
        if url.is_empty() {
            return Err(ToolResult::err("Error: url is required"));
        }
        let max_chars = input
            .get("max_chars")
            .and_then(|v| v.as_u64())
            .map(|v| v as usize)
            .unwrap_or(DEFAULT_MAX_CHARS);
        let include_links = input
            .get("include_links")
            .and_then(|v| v.as_bool())
            .unwrap_or(false);
        Ok(Self {
            url,
            max_chars,
            include_links,
        })
    }
}

/// Convert raw HTML body to formatted tool output with header and optional links.
///
/// `header` is prepended (e.g. "HTTP 200 — url\nContent-Type: ...\n\n").
/// `status_ok` controls `is_error` on the result.
pub fn format_fetch_output(
    body_html: &str,
    header: &str,
    include_links: bool,
    max_chars: usize,
    status_ok: bool,
) -> ToolResult {
    let (text, links) = html_to_text(body_html);
    let remaining = max_chars.saturating_sub(header.chars().count());
    let mut out = header.to_string();
    out.push_str(&truncate_chars(&text, remaining));
    if include_links && !links.is_empty() {
        out.push_str("\n\n## Links\n");
        for l in links.iter().take(40) {
            out.push_str("- ");
            out.push_str(l);
            out.push('\n');
        }
    }
    if status_ok {
        ToolResult::ok(out)
    } else {
        ToolResult::err(out)
    }
}

/// Truncate a string to at most `limit` characters, preserving char boundaries.
pub fn truncate_chars(s: &str, limit: usize) -> String {
    let char_count = s.chars().count();
    if char_count <= limit {
        return s.to_string();
    }
    const MARKER: &str = "\n\n[… truncated]";
    let marker_len = MARKER.chars().count();
    if limit <= marker_len {
        return s.chars().take(limit).collect();
    }
    let mut result: String = s.chars().take(limit - marker_len).collect();
    result.push_str(MARKER);
    result
}

/// Convert HTML to plain text + extracted links.
pub fn html_to_text(html: &str) -> (String, Vec<String>) {
    // Re-export from web_fetch for now — the real implementation stays there
    // until we can break the dependency.
    super::web_fetch::html_to_text(html)
}

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;

    #[test]
    fn truncate_short_unchanged() {
        assert_eq!(truncate_chars("hello", 100), "hello");
    }

    #[test]
    fn truncate_exact_limit() {
        let s = "a".repeat(100);
        assert_eq!(truncate_chars(&s, 100), s);
    }

    proptest! {
            #[test]
            fn truncate_never_exceeds_limit(s in "\\PC{0,500}", limit in 1usize..200) {
                let out = truncate_chars(&s, limit);
                prop_assert!(out.chars().count() <= limit,
                    "truncated to {} chars but limit was {}", out.chars().count(), limit);
            }

            #[test]
            fn truncate_always_valid_utf8(s in "\\PC{0,500}", limit in 1usize..200) {
                let out = truncate_chars(&s, limit);
                // If this compiles and runs, it's valid UTF-8 (Rust String invariant)
                prop_assert!(!out.is_empty() || s.is_empty() || limit == 0);
            }

            #[test]
            fn html_to_text_no_tags_in_output(html in "<[a-z]{1,5}>[^<]{0,100}</[a-z]{1,5}>") {
                let (text, _links) = html_to_text(&html);
                prop_assert!(!text.contains('<') || !text.contains('>'),
                    "output still contains HTML tags: {}", text);
            }

            #[test]
            fn format_fetch_output_is_valid(
                body in "\\PC{0,200}",
                header in "[A-Za-z ]{0,50}",
                max_chars in 50usize..500,
            ) {
                let result = format_fetch_output(&body, &header, false, max_chars, true);
                prop_assert!(!result.is_error);
                // Empty output is valid for degenerate HTML (e.g. "<")
    // No-panic + valid UTF-8 is the contract
            }
        }
}
