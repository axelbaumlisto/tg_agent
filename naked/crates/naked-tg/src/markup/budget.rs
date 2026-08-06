//! Telegram message-budget helpers.
//!
//! Telegram enforces message text length in UTF-16 code units after HTML
//! entities are decoded and markup tags are removed. MessageEntity offsets and
//! lengths use the same unit.

/// Count Telegram-visible text in UTF-16 code units after stripping HTML tags
/// and decoding HTML entities exactly once.
pub fn telegram_html_text_utf16_units(html: &str) -> u64 {
    telegram_html_visible_text(html).encode_utf16().count() as u64
}

/// Return the Telegram-visible text after stripping HTML tags and decoding HTML
/// entities exactly once.
///
/// This intentionally does not reuse streaming's legacy `strip_html_tags`: that
/// helper decodes repeatedly and under-measures `&amp;lt;`-heavy payloads.
///
/// Known limitation (PLAN_TG_LONG_ANSWERS_v2 S6 residual): this is a compact
/// Telegram-renderer measurement helper, not a full HTML tokenizer. A `>` inside
/// a quoted tag attribute would terminate the tag early. `md_to_tg_html` escapes
/// `>` in generated URLs before they reach attributes, so the current renderer
/// cannot produce that shape; do not broaden this scanner in S7.
pub fn telegram_html_visible_text(html: &str) -> String {
    let mut stripped = String::with_capacity(html.len());
    let mut rest = html;
    loop {
        let Some(open) = rest.find('<') else {
            stripped.push_str(rest);
            break;
        };
        stripped.push_str(&rest[..open]);
        let after_open = &rest[open + 1..];
        let Some(close_rel) = after_open.find('>') else {
            stripped.push_str(&rest[open..]);
            break;
        };
        rest = &after_open[close_rel + 1..];
    }
    decode_html_entities_once(&stripped)
}

fn decode_html_entities_once(text: &str) -> String {
    let mut decoded = String::with_capacity(text.len());
    let mut rest = text;
    while let Some(amp) = rest.find('&') {
        decoded.push_str(&rest[..amp]);
        let entity_tail = &rest[amp..];
        if let Some((ch, consumed)) = decode_one_html_entity(entity_tail) {
            decoded.push(ch);
            rest = &entity_tail[consumed..];
        } else {
            decoded.push('&');
            rest = &entity_tail['&'.len_utf8()..];
        }
    }
    decoded.push_str(rest);
    decoded
}

fn decode_one_html_entity(s: &str) -> Option<(char, usize)> {
    for (entity, decoded) in [
        ("&amp;", '&'),
        ("&lt;", '<'),
        ("&gt;", '>'),
        ("&quot;", '"'),
        ("&#39;", '\''),
        ("&apos;", '\''),
    ] {
        if s.starts_with(entity) {
            return Some((decoded, entity.len()));
        }
    }
    decode_numeric_html_entity(s)
}

fn decode_numeric_html_entity(s: &str) -> Option<(char, usize)> {
    let semi = s.find(';')?;
    if semi > 10 || !s.starts_with("&#") {
        return None;
    }
    let body = &s[2..semi];
    let codepoint = if let Some(hex) = body.strip_prefix('x').or_else(|| body.strip_prefix('X')) {
        u32::from_str_radix(hex, 16).ok()?
    } else {
        body.parse::<u32>().ok()?
    };
    char::from_u32(codepoint).map(|ch| (ch, semi + 1))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn telegram_html_text_utf16_units_strips_tags_decodes_entities_once() {
        assert_eq!(
            telegram_html_text_utf16_units("<b>Ж</b>&amp;&lt;💭"),
            1 + 1 + 1 + 2
        );
        assert_eq!(
            telegram_html_text_utf16_units("&amp;lt;"),
            "&lt;".encode_utf16().count() as u64
        );
        assert_eq!(telegram_html_text_utf16_units("&#x1F4AD;&#128173;"), 4);
    }

    #[test]
    fn telegram_html_visible_text_matches_budget_shape() {
        assert_eq!(
            telegram_html_visible_text("<b>A</b>&amp;lt;<i>Ж</i>"),
            "A&lt;Ж"
        );
        assert_eq!(
            telegram_html_visible_text("literal < without close"),
            "literal < without close"
        );
    }
}
