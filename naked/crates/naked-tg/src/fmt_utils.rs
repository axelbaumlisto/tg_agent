//! Shared formatting utilities used across commands, callbacks, media.

/// Render a polling interval as human-readable label.
pub(crate) fn format_interval(secs: u64) -> String {
    if secs == 0 {
        return "manual".to_string();
    }
    if secs.is_multiple_of(86_400) {
        format!("every {}d", secs / 86_400)
    } else if secs.is_multiple_of(3_600) {
        format!("every {}h", secs / 3_600)
    } else if secs.is_multiple_of(60) {
        format!("every {}m", secs / 60)
    } else {
        format!("every {secs}s")
    }
}

/// Render "time since" with one unit of precision.
pub(crate) fn format_age(secs: u64) -> String {
    if secs < 60 {
        format!("{secs}s")
    } else if secs < 3_600 {
        format!("{}m", secs / 60)
    } else if secs < 86_400 {
        format!("{}h", secs / 3_600)
    } else {
        format!("{}d", secs / 86_400)
    }
}

/// Minimal HTML escaper for Telegram messages.
pub(crate) fn escape_html_min(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    for c in s.chars() {
        match c {
            '<' => out.push_str("&lt;"),
            '>' => out.push_str("&gt;"),
            '&' => out.push_str("&amp;"),
            _ => out.push(c),
        }
    }
    out
}

/// Slugify a string for use as a research ID.
pub(crate) fn safe_slug(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    let mut last_dash = false;
    for c in s.chars() {
        if c.is_ascii_alphanumeric() || c == '-' || c == '_' {
            out.push(c);
            last_dash = false;
        } else if !last_dash {
            out.push('-');
            last_dash = true;
        }
    }
    if out.is_empty() {
        return "run".to_string();
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn interval_units() {
        assert_eq!(format_interval(0), "manual");
        assert_eq!(format_interval(30), "every 30s");
        assert_eq!(format_interval(60), "every 1m");
        assert_eq!(format_interval(3600), "every 1h");
        assert_eq!(format_interval(86400), "every 1d");
    }

    #[test]
    fn age_units() {
        assert_eq!(format_age(0), "0s");
        assert_eq!(format_age(59), "59s");
        assert_eq!(format_age(60), "1m");
        assert_eq!(format_age(3600), "1h");
        assert_eq!(format_age(86400), "1d");
    }

    #[test]
    fn html_escaping() {
        assert_eq!(escape_html_min("a < b & c > d"), "a &lt; b &amp; c &gt; d");
        assert_eq!(escape_html_min("\"quoted\""), "\"quoted\"");
    }

    #[test]
    fn slug_basic() {
        assert_eq!(safe_slug("hello world"), "hello-world");
        assert_eq!(safe_slug(""), "run");
        assert_eq!(safe_slug("my-topic_v2"), "my-topic_v2");
    }
}
