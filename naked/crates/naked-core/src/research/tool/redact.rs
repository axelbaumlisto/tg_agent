// B45-wired (2026-05-13): credential scrubber called from
// `streaming_mod/flush.rs::send_final` before every outgoing Telegram
// message. Patterns: api_key=X / Bearer X / Authorization: Token X /
// password=X / secret=X / token=X / separator-delimited compound keys like
// bot_token=X / client.secret=X / Telegram bot tokens in URL paths. Returns a
// borrowed string if no match — zero-allocation fast-path. The Pattern /
// BearerPattern internal matchers stay `pub(crate)`; module-level
// `#![allow(dead_code)]` still kept so test fixtures + private utility
// functions don't trigger lint.
#![allow(dead_code)]

//! Credential/secret redaction for research output.

use std::borrow::Cow;

/// Crude credential/secret redactor. Runs before we forward any research
/// output (reports, summaries, tool outputs) to Telegram / Discord so a
/// misconfigured proxy or a `curl -H "Authorization: ..."` snippet doesn't
/// leak by accident. Ported from zeroclaws `scan_and_redact_output`, but
/// scoped to the small handful of patterns we actually see.
pub fn scan_and_redact(text: &str) -> Cow<'_, str> {
    // No dependency, no regex, no lowercased full-text copy. The common case is
    // benign outgoing text; if it has none of the ASCII marker words that can
    // start a credential shape, return the caller's slice unchanged.
    if !has_redaction_candidate(text.as_bytes()) {
        return Cow::Borrowed(text);
    }

    // Cheap, layered matchers: no backtracking, case-insensitive ASCII where
    // needed. Matches "api_key=SOMETHING", "Bearer XXX",
    // "Authorization: Token YYY", long `$ALLCAPS=SECRETVAL` env exports,
    // separator-delimited compound secret-ish keys (`bot_token=...`,
    // `access_token=...`, `client.secret=...`), and Telegram Bot API URL path
    // tokens (`bot<digits>:<secret>`). All values collapse to `[redacted]`.
    let mut owned = match redact_telegram_bot_tokens(text) {
        Cow::Borrowed(_) => None,
        Cow::Owned(redacted) => Some(redacted),
    };
    for re in REDACTORS.iter() {
        let current = owned.as_deref().unwrap_or(text);
        if let Cow::Owned(redacted) = re.replace_all(current, "$key=[redacted]") {
            owned = Some(redacted);
        }
    }
    for re in BEARER_REDACTORS.iter() {
        let current = owned.as_deref().unwrap_or(text);
        if let Cow::Owned(redacted) = re.replace_all(current, "$prefix [redacted]") {
            owned = Some(redacted);
        }
    }
    match owned {
        Some(redacted) => Cow::Owned(redacted),
        None => Cow::Borrowed(text),
    }
}

/// Render a Display-able error/value through the shared credential redactor
/// before handing it to `tracing`. This is intentionally a thin wrapper over
/// [`scan_and_redact`]: B45 has exactly one place that knows credential shapes.
pub fn redact_for_log(value: impl std::fmt::Display) -> String {
    let rendered = value.to_string();
    scan_and_redact(&rendered).into_owned()
}

fn has_redaction_candidate(bytes: &[u8]) -> bool {
    let mut i = 0;
    while i < bytes.len() {
        match ascii_lower(bytes[i]) {
            b'a' if starts_with_ascii_ci(bytes, i, b"api")
                || starts_with_ascii_ci(bytes, i, b"authorization")
                || starts_with_ascii_ci(bytes, i, b"access") =>
            {
                return true;
            }
            b'b' if starts_with_ascii_ci(bytes, i, b"bearer")
                || starts_with_ascii_ci(bytes, i, b"bot") =>
            {
                return true;
            }
            b'k' if starts_with_ascii_ci(bytes, i, b"key") => {
                return true;
            }
            b'p' if starts_with_ascii_ci(bytes, i, b"password") => {
                return true;
            }
            b's' if starts_with_ascii_ci(bytes, i, b"secret") => {
                return true;
            }
            b't' if starts_with_ascii_ci(bytes, i, b"token") => {
                return true;
            }
            _ => {}
        }
        i += 1;
    }
    false
}

fn ascii_lower(b: u8) -> u8 {
    if b.is_ascii_uppercase() {
        b + (b'a' - b'A')
    } else {
        b
    }
}

fn starts_with_ascii_ci(hay: &[u8], start: usize, needle: &[u8]) -> bool {
    hay.get(start..start + needle.len())
        .is_some_and(|slice| slice.eq_ignore_ascii_case(needle))
}

fn redact_telegram_bot_tokens(text: &str) -> Cow<'_, str> {
    let bytes = text.as_bytes();
    let mut result: Option<String> = None;
    let mut last_copied = 0;
    let mut i = 0;
    while i < bytes.len() {
        if let Some(end) = telegram_bot_token_end(bytes, i) {
            let out = result.get_or_insert_with(|| String::with_capacity(text.len()));
            out.push_str(&text[last_copied..i]);
            out.push_str("bot[redacted]");
            i = end;
            last_copied = end;
            continue;
        }
        let ch_len = text[i..].chars().next().map(char::len_utf8).unwrap_or(1);
        i += ch_len;
    }

    match result {
        Some(mut out) => {
            out.push_str(&text[last_copied..]);
            Cow::Owned(out)
        }
        None => Cow::Borrowed(text),
    }
}

const MIN_TELEGRAM_BOT_SECRET_CHARS: usize = 4;

fn telegram_bot_token_end(bytes: &[u8], start: usize) -> Option<usize> {
    if start + 3 > bytes.len() || !bytes[start..start + 3].eq_ignore_ascii_case(b"bot") {
        return None;
    }
    if start > 0 {
        let prev = bytes[start - 1];
        if prev.is_ascii_alphanumeric() || matches!(prev, b'_' | b'-') {
            return None;
        }
    }

    let mut cursor = start + 3;
    let digit_start = cursor;
    while cursor < bytes.len() && bytes[cursor].is_ascii_digit() && cursor - digit_start < 10 {
        cursor += 1;
    }
    let digit_count = cursor - digit_start;
    if !(8..=10).contains(&digit_count) {
        return None;
    }
    if bytes.get(cursor).is_some_and(u8::is_ascii_digit) {
        return None;
    }
    if bytes.get(cursor) != Some(&b':') {
        return None;
    }
    cursor += 1;

    let secret_start = cursor;
    while cursor < bytes.len() && is_telegram_token_char(bytes[cursor]) {
        cursor += 1;
    }
    // `bot<8-10 digits>:` is a Telegram-credential-specific shape, so a short
    // URL-safe suffix is already sensitive. Four chars catches truncated log
    // leaks (`bot1234567890:AAFZ...`) without redacting accidental `bot...:x` prose.
    if cursor - secret_start < MIN_TELEGRAM_BOT_SECRET_CHARS {
        return None;
    }
    Some(cursor)
}

fn is_telegram_token_char(b: u8) -> bool {
    b.is_ascii_alphanumeric() || matches!(b, b'_' | b'-')
}

// Lazy-compiled matcher tables. Using the `regex` crate would add a dep;
// instead we write tiny inline matchers. The patterns are intentionally narrow.
static REDACTORS: once_cell_shim::Lazy<Vec<matcher::Pattern>> = once_cell_shim::Lazy::new(|| {
    vec![
        // `api_key = "abc"`, `api-key: abc`, `API_KEY=abc`
        matcher::Pattern::new("api[_-]?key", true),
        matcher::Pattern::new("key", true).with_key_prefixes(),
        matcher::Pattern::new("secret", true).with_key_prefixes(),
        matcher::Pattern::new("password", true).with_key_prefixes(),
        matcher::Pattern::new("token", true).with_key_prefixes(),
    ]
});

static BEARER_REDACTORS: once_cell_shim::Lazy<Vec<matcher::BearerPattern>> =
    once_cell_shim::Lazy::new(|| {
        // `Authorization: Bearer xxx` is covered by the bare bearer matcher;
        // `Authorization: Token xxx` is a real auth scheme but `token prose`
        // must remain readable, so Token requires the Authorization header.
        vec![
            matcher::BearerPattern::new("bearer"),
            matcher::BearerPattern::authorization_scheme("token"),
        ]
    });

mod once_cell_shim {
    use std::sync::OnceLock;

    pub struct Lazy<T: 'static> {
        init: fn() -> T,
        cell: OnceLock<T>,
    }

    impl<T: 'static> Lazy<T> {
        pub const fn new(init: fn() -> T) -> Self {
            Self {
                init,
                cell: OnceLock::new(),
            }
        }
    }

    impl<T: 'static> std::ops::Deref for Lazy<T> {
        type Target = T;
        fn deref(&self) -> &Self::Target {
            self.cell.get_or_init(self.init)
        }
    }
}

mod matcher {
    use std::borrow::Cow;

    use super::starts_with_ascii_ci;

    /// Very small "`key` = value" matcher. Finds case-insensitive `key` (as a
    /// word), optional whitespace, one of `:` / `=`, whitespace, then captures
    /// everything up to the next whitespace / quote / `,` / `;` / end.
    pub struct Pattern {
        key: String,
        case_insensitive: bool,
        allow_key_prefixes: bool,
    }

    impl Pattern {
        pub fn new(key: &str, case_insensitive: bool) -> Self {
            Self {
                key: key.to_string(),
                case_insensitive,
                allow_key_prefixes: false,
            }
        }

        pub fn with_key_prefixes(mut self) -> Self {
            self.allow_key_prefixes = true;
            self
        }

        pub fn replace_all<'a>(&self, haystack: &'a str, replacement: &str) -> Cow<'a, str> {
            let mut result: Option<String> = None;
            let bytes = haystack.as_bytes();
            let mut last_copied = 0;
            let mut i = 0;
            while i < bytes.len() {
                if let Some(end) = self.match_at(haystack, i) {
                    // `replacement` is `$key=[redacted]` — emit the original
                    // key substring verbatim, then the fixed redaction suffix.
                    let out = result.get_or_insert_with(|| String::with_capacity(haystack.len()));
                    out.push_str(&haystack[last_copied..i]);
                    let key_str = &haystack[i..end.key_end];
                    if let Some(suffix) = replacement.strip_prefix("$key") {
                        out.push_str(key_str);
                        out.push_str(suffix);
                    } else {
                        out.push_str(&replacement.replace("$key", key_str));
                    }
                    i = end.value_end;
                    last_copied = end.value_end;
                } else {
                    let ch_len = haystack[i..]
                        .chars()
                        .next()
                        .map(char::len_utf8)
                        .unwrap_or(1);
                    i += ch_len;
                }
            }

            match result {
                Some(mut out) => {
                    out.push_str(&haystack[last_copied..]);
                    Cow::Owned(out)
                }
                None => Cow::Borrowed(haystack),
            }
        }

        fn match_at(&self, hay: &str, start: usize) -> Option<Span> {
            // Match our regex-y key (allow [_-]? meta). Simplified: we support
            // the hardcoded `[_-]?` between tokens as written in the callers;
            // otherwise just literal equals.
            let key_end = match_key_literal(hay, start, &self.key, self.case_insensitive)
                .or_else(|| self.match_prefixed_key(hay, start))?;
            // Must be a key boundary at start (previous char non key-name char / start).
            if start > 0 {
                let prev = hay[..start].chars().next_back().unwrap_or(' ');
                if prev.is_ascii_alphanumeric() || matches!(prev, '_' | '-') {
                    return None;
                }
            }
            // Skip whitespace.
            let after_key = skip_ws(hay, key_end);
            let sep = hay.as_bytes().get(after_key).copied()?;
            if sep != b'=' && sep != b':' {
                return None;
            }
            let after_sep = skip_ws(hay, after_key + 1);
            if hay
                .get(after_sep..)
                .is_some_and(|rest| rest.starts_with("[redacted]"))
            {
                return None;
            }
            // Value: everything until whitespace / quote / `,` / `;` / `}` / `)`, preserving the start.
            let value_end = find_value_end(hay, after_sep);
            if value_end == after_sep {
                return None;
            }
            Some(Span { key_end, value_end })
        }

        fn match_prefixed_key(&self, hay: &str, start: usize) -> Option<usize> {
            if !self.allow_key_prefixes || self.key.contains("[_-]?") {
                return None;
            }
            let mut end = start;
            while hay
                .as_bytes()
                .get(end)
                .is_some_and(|b| is_key_name_char(*b))
            {
                end += 1;
            }
            if end == start {
                return None;
            }
            let candidate = hay.get(start..end)?;
            if ascii_eq(candidate, &self.key, self.case_insensitive)
                || has_separator_before_suffix(candidate, &self.key, self.case_insensitive)
            {
                return Some(end);
            }
            None
        }
    }

    pub(crate) struct BearerPattern {
        prefix: String,
        requires_authorization_header: bool,
    }

    impl BearerPattern {
        pub fn new(prefix: &str) -> Self {
            Self {
                prefix: prefix.to_string(),
                requires_authorization_header: false,
            }
        }

        pub fn authorization_scheme(prefix: &str) -> Self {
            Self {
                prefix: prefix.to_string(),
                requires_authorization_header: true,
            }
        }

        pub fn replace_all<'a>(&self, hay: &'a str, rep_template: &str) -> Cow<'a, str> {
            let mut result: Option<String> = None;
            let mut last_copied = 0;
            let mut i = 0;
            while i < hay.len() {
                if let Some(span) = self.match_at(hay, i) {
                    let out = result.get_or_insert_with(|| String::with_capacity(hay.len()));
                    out.push_str(&hay[last_copied..i]);
                    let prefix_str = &hay[i..span.prefix_end];
                    let rendered = rep_template.replace("$prefix", prefix_str.trim_end());
                    out.push_str(&rendered);
                    i = span.value_end;
                    last_copied = span.value_end;
                } else {
                    let ch_len = hay[i..].chars().next().map(char::len_utf8).unwrap_or(1);
                    i += ch_len;
                }
            }

            match result {
                Some(mut out) => {
                    out.push_str(&hay[last_copied..]);
                    Cow::Owned(out)
                }
                None => Cow::Borrowed(hay),
            }
        }

        fn match_at(&self, hay: &str, start: usize) -> Option<BearerSpan> {
            if !starts_with_ascii_ci(hay.as_bytes(), start, self.prefix.as_bytes()) {
                return None;
            }
            // Must be at word boundary (start-of-string or non-alnum/underscore before).
            if start > 0 {
                let prev = hay[..start].chars().next_back().unwrap_or(' ');
                if prev.is_ascii_alphanumeric() || prev == '_' {
                    return None;
                }
            }
            if self.requires_authorization_header && !has_authorization_header_before(hay, start) {
                return None;
            }

            let after_kw = start + self.prefix.len();
            let after_ws = skip_ws(hay, after_kw);
            if after_ws == after_kw {
                return None;
            }
            if hay
                .get(after_ws..)
                .is_some_and(|rest| rest.starts_with("[redacted]"))
            {
                return None;
            }
            let end = find_value_end(hay, after_ws);
            if end == after_ws {
                return None;
            }
            Some(BearerSpan {
                prefix_end: after_ws,
                value_end: end,
            })
        }
    }

    struct Span {
        key_end: usize,
        value_end: usize,
    }

    struct BearerSpan {
        prefix_end: usize,
        value_end: usize,
    }

    fn match_key_literal(
        hay: &str,
        start: usize,
        key: &str,
        case_insensitive: bool,
    ) -> Option<usize> {
        // Support `[_-]?` meta inside keys like `api[_-]?key` — interpret as:
        // try literal match with `_`, `-`, or none between the segments on `[`. B48:
        // use `str::get()` for safe slicing because callers walk UTF-8 text.
        if key.contains("[_-]?") {
            let parts: Vec<&str> = key.split("[_-]?").collect();
            let mut end = start;
            for (i, part) in parts.iter().enumerate() {
                let slice = hay.get(end..end + part.len())?;
                if !ascii_eq(slice, part, case_insensitive) {
                    return None;
                }
                end += part.len();
                if i + 1 < parts.len() {
                    match hay.as_bytes().get(end) {
                        Some(b'_') | Some(b'-') => end += 1,
                        _ => {}
                    }
                }
            }
            return Some(end);
        }
        let slice = hay.get(start..start + key.len())?;
        if ascii_eq(slice, key, case_insensitive) {
            return Some(start + key.len());
        }
        None
    }

    fn skip_ws(hay: &str, mut i: usize) -> usize {
        while let Some(b) = hay.as_bytes().get(i) {
            if matches!(*b, b' ' | b'\t') {
                i += 1;
            } else {
                break;
            }
        }
        i
    }

    fn is_key_name_char(b: u8) -> bool {
        b.is_ascii_alphanumeric() || matches!(b, b'_' | b'-' | b'.')
    }

    fn has_separator_before_suffix(candidate: &str, key: &str, case_insensitive: bool) -> bool {
        if candidate.len() <= key.len() {
            return false;
        }
        let suffix_start = candidate.len() - key.len();
        candidate
            .as_bytes()
            .get(suffix_start - 1)
            .is_some_and(|b| matches!(*b, b'_' | b'-' | b'.'))
            && candidate
                .get(suffix_start..)
                .is_some_and(|suffix| ascii_eq(suffix, key, case_insensitive))
    }

    fn has_authorization_header_before(hay: &str, scheme_start: usize) -> bool {
        let bytes = hay.as_bytes();
        let mut cursor = scheme_start;
        while cursor > 0 && matches!(bytes[cursor - 1], b' ' | b'\t') {
            cursor -= 1;
        }
        if cursor == 0 || bytes[cursor - 1] != b':' {
            return false;
        }
        cursor -= 1;
        while cursor > 0 && matches!(bytes[cursor - 1], b' ' | b'\t') {
            cursor -= 1;
        }
        let header = b"authorization";
        if cursor < header.len() {
            return false;
        }
        let header_start = cursor - header.len();
        if !starts_with_ascii_ci(bytes, header_start, header) {
            return false;
        }
        if header_start > 0 {
            let prev = bytes[header_start - 1];
            if prev.is_ascii_alphanumeric() || matches!(prev, b'_' | b'-') {
                return false;
            }
        }
        true
    }

    fn ascii_eq(left: &str, right: &str, case_insensitive: bool) -> bool {
        if case_insensitive {
            left.as_bytes().eq_ignore_ascii_case(right.as_bytes())
        } else {
            left == right
        }
    }

    fn find_value_end(hay: &str, start: usize) -> usize {
        // If the value starts with a quote, consume until matching quote.
        if let Some(b'"') = hay.as_bytes().get(start) {
            let after = start + 1;
            if let Some(pos) = hay[after..].find('"') {
                return after + pos + 1;
            }
        }
        let stop: &[char] = &[' ', '\t', '\n', '\r', '&', ',', ';', '"', '}', ')', ']'];
        match hay[start..].find(stop) {
            Some(p) => start + p,
            None => hay.len(),
        }
    }
}
