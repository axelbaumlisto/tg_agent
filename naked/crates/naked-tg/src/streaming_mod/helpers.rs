//! Pure helper functions for streaming.

pub(crate) fn format_input_preview(input: &serde_json::Value, max_len: usize) -> String {
    if let Some(map) = input.as_object() {
        if let Some((key, val)) = map.iter().next().filter(|_| map.len() == 1) {
            let fallback = val.to_string();
            let v = val.as_str().unwrap_or(&fallback);
            return format!("{key}: {}", truncate_str(v, max_len));
        }
        let parts: Vec<String> = map
            .iter()
            .map(|(k, v)| {
                let fallback = v.to_string();
                let s = v.as_str().unwrap_or(&fallback);
                format!("{k}: {}", truncate_str(s, 60))
            })
            .collect();
        truncate_str(&parts.join(", "), max_len)
    } else {
        truncate_str(&input.to_string(), max_len)
    }
}

pub(crate) fn truncate_str(s: &str, max_chars: usize) -> String {
    let mut last_boundary = 0;
    for (i, (byte_pos, _)) in s.char_indices().enumerate() {
        if i >= max_chars {
            return format!("{}…", &s[..last_boundary]);
        }
        last_boundary = byte_pos;
    }
    s.to_string()
}

/// Parse "Retry after Xs" from Telegram error string.
pub(crate) fn parse_retry_after(err: &str) -> Option<u64> {
    let s = err.to_lowercase();
    if let Some(pos) = s.find("retry after") {
        let after = &s[pos + 12..];
        let digits: String = after.chars().take_while(|c| c.is_ascii_digit()).collect();
        digits.parse().ok()
    } else {
        None
    }
}

/// Strip `class="language-..."` from `<code>` tags.
/// Telegram `editMessageText` rejects attributes during streaming preview
/// but accepts them in `sendMessage` (final render).
pub(crate) fn strip_code_class(html: &str) -> String {
    // Fast path: no class= at all
    if !html.contains("class=") {
        return html.to_string();
    }
    // Replace <code class="language-X"> with <code>
    let mut out = String::with_capacity(html.len());
    let mut rest = html;
    while let Some(pos) = rest.find("<code class=") {
        out.push_str(&rest[..pos]);
        out.push_str("<code>");
        // Skip past the closing >
        if let Some(close) = rest[pos..].find('>') {
            rest = &rest[pos + close + 1..];
        } else {
            rest = &rest[pos..];
            break;
        }
    }
    out.push_str(rest);
    out
}

/// Crude HTML tag stripper for plain-text fallback.
pub(crate) fn strip_html_tags(html: &str) -> String {
    let mut out = String::with_capacity(html.len());
    let mut in_tag = false;
    for ch in html.chars() {
        match ch {
            '<' => in_tag = true,
            '>' if in_tag => in_tag = false,
            _ if !in_tag => out.push(ch),
            _ => {}
        }
    }
    // Unescape HTML entities
    out.replace("&amp;", "&")
        .replace("&lt;", "<")
        .replace("&gt;", ">")
        .replace("&quot;", "\"")
}
