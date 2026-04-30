//! Self-contained HTML rendering for research reports.
//!
//! Telegram opens `.html` document attachments in its in-app browser,
//! which gives us a free "open the full report on any device"
//! affordance without standing up a separate web UI. The rendered
//! document has:
//!
//! * zero external assets (CSS is inlined, no JS at all),
//! * `prefers-color-scheme` theming so it looks native in both light
//!   and dark Telegram,
//! * a viewport meta so the in-app browser doesn't shrink the content.
//!
//! We accept a raw `report.md` (the coordinator regenerates this on
//! every run) plus a minimal amount of metadata and produce the
//! complete `<!DOCTYPE html>` document. The markdown→HTML step is a
//! conservative line-based converter shared with the chat pipeline's
//! `md_to_tg_html` — we only need headings, bullets, code spans,
//! links, and paragraph breaks; anything fancier is overkill for
//! structured finding dumps.

use chrono::{DateTime, Utc};

/// Minimum viable metadata to render the report header. `run_id` is
/// optional because the initial seed may not have been run yet — in
/// that case the header just lists the spec.
#[derive(Debug, Clone)]
pub struct ReportMeta<'a> {
    pub spec_id: &'a str,
    pub topic: &'a str,
    pub run_id: Option<&'a str>,
    pub findings_total: u32,
    pub new_findings: u32,
    pub generated_at: DateTime<Utc>,
}

/// Render a complete HTML document for `report_md`. Returns bytes so
/// the caller can feed it straight to `teloxide::types::InputFile::memory`.
pub fn render_report_html(meta: &ReportMeta<'_>, report_md: &str) -> Vec<u8> {
    let title = html_escape(meta.topic);
    let generated = meta.generated_at.format("%Y-%m-%d %H:%M UTC").to_string();
    let run_line = meta
        .run_id
        .map(|rid| format!("Run <code>{}</code> · ", html_escape(rid)))
        .unwrap_or_default();
    let body_html = md_to_html(report_md);

    let doc = format!(
        r#"<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{title} — research report</title>
<style>
:root {{ color-scheme: light dark; }}
* {{ box-sizing: border-box; }}
body {{
  font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
  line-height: 1.6;
  margin: 0 auto;
  max-width: 760px;
  padding: 24px 18px 48px;
  color: #111;
  background: #fafafa;
}}
@media (prefers-color-scheme: dark) {{
  body {{ color: #e8e8e8; background: #111; }}
  a {{ color: #8ab4f8; }}
  .meta {{ color: #999; }}
  code {{ background: #1d1d1d; color: #e8e8e8; }}
  pre {{ background: #1d1d1d; }}
  hr {{ border-color: #2a2a2a; }}
}}
h1, h2, h3 {{ line-height: 1.25; margin: 1.4em 0 0.5em; }}
h1 {{ font-size: 1.6rem; margin-top: 0; }}
h2 {{ font-size: 1.25rem; border-bottom: 1px solid rgba(128,128,128,0.25); padding-bottom: 0.25em; }}
h3 {{ font-size: 1.05rem; }}
ul, ol {{ padding-left: 1.4em; }}
li {{ margin: 0.25em 0; }}
a {{ color: #1565c0; text-decoration: none; }}
a:hover {{ text-decoration: underline; }}
code {{ background: #eee; padding: 1px 5px; border-radius: 3px; font-size: 0.92em; }}
pre {{ background: #eee; padding: 12px 14px; border-radius: 6px; overflow-x: auto; }}
pre code {{ background: transparent; padding: 0; }}
.meta {{ color: #666; font-size: 0.95em; margin-bottom: 1.6em; }}
hr {{ border: 0; border-top: 1px solid #ccc; margin: 1.8em 0; }}
blockquote {{
  border-left: 3px solid #888;
  margin: 0.8em 0;
  padding: 0.2em 1em;
  color: #555;
}}
</style>
</head>
<body>
<h1>{title}</h1>
<p class="meta">
  {run_line}spec <code>{spec}</code> · findings: <b>{total}</b> (new this run: <b>{new}</b>) · generated {generated}
</p>
<hr>
{body_html}
</body>
</html>
"#,
        title = title,
        spec = html_escape(meta.spec_id),
        total = meta.findings_total,
        new = meta.new_findings,
        generated = generated,
        run_line = run_line,
        body_html = body_html,
    );

    doc.into_bytes()
}

/// Minimal HTML entity escaping for text destined for the document
/// body or an attribute. Does not touch apostrophes — the output is
/// never embedded inside single-quoted attribute values.
fn html_escape(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    for c in s.chars() {
        match c {
            '<' => out.push_str("&lt;"),
            '>' => out.push_str("&gt;"),
            '&' => out.push_str("&amp;"),
            '"' => out.push_str("&quot;"),
            _ => out.push(c),
        }
    }
    out
}

/// Line-based markdown→HTML converter. Covers the subset the
/// research `report.md` actually uses: ATX headings, bullets,
/// numbered lists, fenced code blocks, inline code, links, bold /
/// italic, and blockquotes. Anything unrecognised falls through as
/// escaped text wrapped in a `<p>`.
fn md_to_html(md: &str) -> String {
    let mut out = String::with_capacity(md.len() + 64);
    let mut lines = md.lines().peekable();
    let mut in_code = false;

    while let Some(raw) = lines.next() {
        let trimmed = raw.trim_end();
        if trimmed.starts_with("```") {
            if in_code {
                out.push_str("</code></pre>\n");
                in_code = false;
            } else {
                out.push_str("<pre><code>");
                in_code = true;
            }
            continue;
        }
        if in_code {
            out.push_str(&html_escape(raw));
            out.push('\n');
            continue;
        }

        if trimmed.is_empty() {
            continue;
        }

        if let Some(rest) = trimmed.strip_prefix("### ") {
            out.push_str("<h3>");
            out.push_str(&inline_md(&html_escape(rest)));
            out.push_str("</h3>\n");
            continue;
        }
        if let Some(rest) = trimmed.strip_prefix("## ") {
            out.push_str("<h2>");
            out.push_str(&inline_md(&html_escape(rest)));
            out.push_str("</h2>\n");
            continue;
        }
        if let Some(rest) = trimmed.strip_prefix("# ") {
            out.push_str("<h1>");
            out.push_str(&inline_md(&html_escape(rest)));
            out.push_str("</h1>\n");
            continue;
        }
        if trimmed == "---" || trimmed == "***" {
            out.push_str("<hr>\n");
            continue;
        }
        if let Some(rest) = trimmed.strip_prefix("> ") {
            out.push_str("<blockquote>");
            out.push_str(&inline_md(&html_escape(rest)));
            out.push_str("</blockquote>\n");
            continue;
        }

        let ltrim = trimmed.trim_start();
        let leading_space = trimmed.len() - ltrim.len();
        if ltrim.starts_with("- ") || ltrim.starts_with("* ") || ltrim.starts_with("+ ") {
            emit_list_block(&mut out, &mut lines, trimmed, leading_space, false);
            continue;
        }
        if is_ordered_list_marker(ltrim) {
            emit_list_block(&mut out, &mut lines, trimmed, leading_space, true);
            continue;
        }

        out.push_str("<p>");
        out.push_str(&inline_md(&html_escape(trimmed)));
        out.push_str("</p>\n");
    }

    if in_code {
        out.push_str("</code></pre>\n");
    }

    out
}

fn is_ordered_list_marker(s: &str) -> bool {
    let mut chars = s.chars();
    match chars.next() {
        Some(c) if c.is_ascii_digit() => {}
        _ => return false,
    }
    for c in chars {
        if c.is_ascii_digit() {
            continue;
        }
        if c == '.' || c == ')' {
            return true;
        }
        return false;
    }
    false
}

fn emit_list_block<'a, I>(
    out: &mut String,
    lines: &mut std::iter::Peekable<I>,
    first: &str,
    leading: usize,
    ordered: bool,
) where
    I: Iterator<Item = &'a str>,
{
    let tag = if ordered { "ol" } else { "ul" };
    out.push('<');
    out.push_str(tag);
    out.push_str(">\n");
    emit_list_item(out, first);

    while let Some(peek) = lines.peek() {
        let t = peek.trim_end();
        if t.is_empty() {
            break;
        }
        let ltrim = t.trim_start();
        let this_leading = t.len() - ltrim.len();
        if this_leading < leading {
            break;
        }
        let matches_kind = if ordered {
            is_ordered_list_marker(ltrim)
        } else {
            ltrim.starts_with("- ") || ltrim.starts_with("* ") || ltrim.starts_with("+ ")
        };
        if !matches_kind {
            break;
        }
        let Some(line) = lines.next() else { break };
        emit_list_item(out, line.trim_end());
    }

    out.push_str("</");
    out.push_str(tag);
    out.push_str(">\n");
}

fn emit_list_item(out: &mut String, line: &str) {
    let trimmed = line.trim_start();
    let content = if let Some(rest) = trimmed
        .strip_prefix("- ")
        .or_else(|| trimmed.strip_prefix("* "))
        .or_else(|| trimmed.strip_prefix("+ "))
    {
        rest.to_string()
    } else if let Some(pos) = trimmed.find(['.', ')']) {
        trimmed[pos + 1..].trim_start().to_string()
    } else {
        trimmed.to_string()
    };
    out.push_str("<li>");
    out.push_str(&inline_md(&html_escape(&content)));
    out.push_str("</li>\n");
}

/// Inline markdown: autolinks, bold (`**x**`), italic (`*x*`), code
/// (`` `x` ``). The input is already HTML-escaped so we can safely
/// substitute tag pairs.
fn inline_md(escaped: &str) -> String {
    let code = replace_delim(escaped, "`", "<code>", "</code>");
    let bold = replace_delim(&code, "**", "<b>", "</b>");
    let italic = replace_delim(&bold, "*", "<i>", "</i>");
    linkify_md(&italic)
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
        // Unbalanced delim — put the stray marker back so the output
        // doesn't swallow text silently.
        out.push_str(delim);
    }
    out
}

/// Convert `[label](url)` and bare `https?://...` URLs into `<a>`
/// links. URLs inside already-rendered `<a>` tags are skipped (we
/// detect this cheaply by refusing to rewrite inside any pre-existing
/// `<a>` region — none exist at this point because the pipeline has
/// not emitted links yet, so a simple one-pass scanner suffices).
fn linkify_md(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    let bytes = s.as_bytes();
    let mut i = 0;
    while i < bytes.len() {
        if bytes[i] == b'['
            && let Some((consumed, html)) = try_parse_md_link(&s[i..])
        {
            out.push_str(&html);
            i += consumed;
            continue;
        }
        if s[i..].starts_with("http://") || s[i..].starts_with("https://") {
            let end = s[i..]
                .find(|c: char| c.is_whitespace() || c == ')' || c == ']' || c == '<' || c == '>')
                .map(|p| i + p)
                .unwrap_or(s.len());
            let url = &s[i..end];
            out.push_str(&format!("<a href=\"{url}\">{url}</a>"));
            i = end;
            continue;
        }
        let ch = bytes[i];
        out.push(ch as char);
        i += 1;
    }
    out
}

/// Attempt to parse `[label](url)` starting at the input. Returns
/// `Some((consumed_bytes, html))` on success, `None` otherwise.
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
    let consumed = label_end + 1 + url_end_rel + 1;
    let html = format!("<a href=\"{url}\">{label}</a>");
    Some((consumed, html))
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::TimeZone;

    fn stub_meta() -> ReportMeta<'static> {
        ReportMeta {
            spec_id: "abc",
            topic: "Da Nang commercial real estate",
            run_id: Some("run-xyz"),
            findings_total: 7,
            new_findings: 3,
            generated_at: Utc.with_ymd_and_hms(2026, 4, 20, 16, 30, 0).unwrap(),
        }
    }

    #[test]
    fn renders_doctype_and_core_metadata() {
        let html = render_report_html(&stub_meta(), "# Hello\n\nBody text.");
        let s = String::from_utf8(html).unwrap();
        assert!(s.starts_with("<!DOCTYPE html>"));
        assert!(s.contains("Da Nang commercial real estate"));
        assert!(s.contains("run-xyz"));
        assert!(s.contains("findings: <b>7</b>"));
        assert!(s.contains("<b>3</b>"));
        assert!(s.contains("<h1>Hello</h1>"));
        assert!(s.contains("<p>Body text.</p>"));
    }

    #[test]
    fn bullets_and_inline_link_survive_conversion() {
        let md = "## Findings\n\n- First item\n- Second with [label](https://example.com/a)\n\n## Notes\n\n- plain";
        let html = md_to_html(md);
        assert!(html.contains("<h2>Findings</h2>"));
        assert!(html.contains("<li>First item</li>"));
        assert!(html.contains("<a href=\"https://example.com/a\">label</a>"));
    }

    #[test]
    fn escapes_html_in_plain_paragraphs() {
        let md = "A <script> tag appears.";
        let html = md_to_html(md);
        assert!(html.contains("&lt;script&gt;"));
        assert!(!html.contains("<script>"));
    }

    #[test]
    fn autolink_bare_url_gets_wrapped() {
        let md = "See https://alonhadat.com.vn/foo for details.";
        let html = md_to_html(md);
        assert!(
            html.contains(
                "<a href=\"https://alonhadat.com.vn/foo\">https://alonhadat.com.vn/foo</a>"
            )
        );
    }

    #[test]
    fn bold_and_code_inline() {
        let md = "This is **important** and `code` together.";
        let html = md_to_html(md);
        assert!(html.contains("<b>important</b>"));
        assert!(html.contains("<code>code</code>"));
    }

    #[test]
    fn fenced_code_block_preserved() {
        let md = "```\nlet x = 1;\n```\n";
        let html = md_to_html(md);
        assert!(html.contains("<pre><code>"));
        assert!(html.contains("let x = 1;"));
        assert!(html.contains("</code></pre>"));
    }
}
