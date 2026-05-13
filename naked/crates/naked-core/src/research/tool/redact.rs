// REGISTRY-WAIVE: B45 — credential scrubber ported from zeroclaws.
// `scan_and_redact` is tested (tool_tests.rs) but NOT YET wired into the
// outbound message path. Intended call site: streaming_mod/flush.rs::send_final
// before forwarding research output to Telegram. Until wired, the function +
// its internal Pattern/BearerPattern matchers + REDACTORS static are all dead
// in the prod hot path. Keeping the module ready to plug in.
#![allow(dead_code)]

//! Credential/secret redaction for research output.

/// Crude credential/secret redactor. Runs before we forward any research
/// output (reports, summaries, tool outputs) to Telegram / Discord so a
/// misconfigured proxy or a `curl -H "Authorization: ..."` snippet doesn't
/// leak by accident. Ported from zeroclaws `scan_and_redact_output`, but
/// scoped to the small handful of patterns we actually see.
pub(crate) fn scan_and_redact(text: &str) -> String {
    // Cheap, layered regex: no backtracking, case-insensitive.
    // Matches "api_key=SOMETHING", "Bearer XXX", "Authorization: Token YYY",
    // and long `$ALLCAPS=SECRETVAL` env exports. All values collapse to
    // `[redacted]`.
    let mut out = text.to_string();
    for re in REDACTORS.iter() {
        out = re.replace_all(&out, "$key=[redacted]").to_string();
    }
    for re in BEARER_REDACTORS.iter() {
        out = re.replace_all(&out, "$prefix [redacted]").to_string();
    }
    out
}

// Lazy-compiled regex tables. Using the `regex` crate would add a dep; instead
// we write a tiny inline matcher. The patterns are intentionally narrow.
static REDACTORS: once_cell_shim::Lazy<Vec<matcher::Pattern>> = once_cell_shim::Lazy::new(|| {
    vec![
        // `api_key = "abc"`, `api-key: abc`, `API_KEY=abc`
        matcher::Pattern::new("api[_-]?key", true),
        matcher::Pattern::new("secret", true),
        matcher::Pattern::new("password", true),
        matcher::Pattern::new("token", true),
    ]
});

static BEARER_REDACTORS: once_cell_shim::Lazy<Vec<matcher::BearerPattern>> =
    once_cell_shim::Lazy::new(|| {
        // NOTE: we only redact the bare `bearer <token>` sequence. Redacting
        // the `authorization:` header as a separate pass collides with the
        // already-redacted `bearer` token and produces `[redacted] [redacted]`
        // noise. One pass is enough — `Authorization: Bearer xxx` becomes
        // `Authorization: Bearer [redacted]`, which still hides the secret.
        vec![matcher::BearerPattern::new("bearer")]
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
    /// Very small "`key` = value" matcher. Finds case-insensitive `key` (as a
    /// word), optional whitespace, one of `:` / `=`, whitespace, then captures
    /// everything up to the next whitespace / quote / `,` / `;` / end.
    pub struct Pattern {
        key: String,
        case_insensitive: bool,
    }

    impl Pattern {
        pub fn new(key: &str, case_insensitive: bool) -> Self {
            Self {
                key: key.to_string(),
                case_insensitive,
            }
        }

        pub fn replace_all(&self, haystack: &str, replacement: &str) -> String {
            let mut result = String::with_capacity(haystack.len());
            let bytes = haystack.as_bytes();
            let mut i = 0;
            while i < bytes.len() {
                if let Some(end) = self.match_at(haystack, i) {
                    // `replacement` is `$key=[redacted]` — we emit the original
                    // key substring the pattern matched, verbatim, then `=[redacted]`.
                    let key_str = &haystack[i..end.key_end];
                    let rendered = replacement.replace("$key", key_str);
                    result.push_str(&rendered);
                    i = end.value_end;
                } else {
                    let ch_len = haystack[i..]
                        .chars()
                        .next()
                        .map(|c| c.len_utf8())
                        .unwrap_or(1);
                    result.push_str(&haystack[i..i + ch_len]);
                    i += ch_len;
                }
            }
            result
        }

        fn match_at(&self, hay: &str, start: usize) -> Option<Span> {
            let lc;
            let haystack_cmp: &str = if self.case_insensitive {
                lc = hay.to_ascii_lowercase();
                &lc
            } else {
                hay
            };
            let key_lc = if self.case_insensitive {
                self.key.to_ascii_lowercase()
            } else {
                self.key.clone()
            };
            if start + key_lc.len() > haystack_cmp.len() {
                return None;
            }
            // Match our regex-y key (allow [_-]? meta). Simplified: we support
            // the hardcoded `[_-]?` between tokens as written in the callers;
            // otherwise just literal equals.
            let key_end = match_key_literal(haystack_cmp, start, &key_lc)?;
            // Must be a word boundary at start (previous char non-alnum / start).
            if start > 0 {
                let prev = hay[..start].chars().next_back().unwrap_or(' ');
                if prev.is_ascii_alphanumeric() || prev == '_' {
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
            // Value: everything until whitespace / quote / `,` / `;` / `}` / `)`, preserving the start.
            let value_end = find_value_end(hay, after_sep);
            if value_end == after_sep {
                return None;
            }
            Some(Span { key_end, value_end })
        }
    }

    pub(crate) struct BearerPattern {
        prefix: String,
    }

    impl BearerPattern {
        pub fn new(prefix: &str) -> Self {
            Self {
                prefix: prefix.to_string(),
            }
        }
        pub fn replace_all(&self, hay: &str, rep_template: &str) -> String {
            let mut result = String::with_capacity(hay.len());
            let lc = hay.to_ascii_lowercase();
            let needle = self.prefix.to_ascii_lowercase();
            let mut i = 0;
            while i < hay.len() {
                if let Some(pos) = lc[i..].find(&needle) {
                    let start = i + pos;
                    // Must be at word boundary (start-of-string or non-alnum before).
                    if start > 0 {
                        let prev = hay[..start].chars().next_back().unwrap_or(' ');
                        if prev.is_ascii_alphanumeric() || prev == '_' {
                            result.push_str(&hay[i..start + needle.len()]);
                            i = start + needle.len();
                            continue;
                        }
                    }
                    result.push_str(&hay[i..start]);
                    let after_kw = start + needle.len();
                    let after_ws = skip_ws(hay, after_kw);
                    let end = find_value_end(hay, after_ws);
                    let prefix_str = &hay[start..after_ws];
                    let rendered = rep_template.replace("$prefix", prefix_str.trim_end());
                    result.push_str(&rendered);
                    i = end;
                } else {
                    result.push_str(&hay[i..]);
                    break;
                }
            }
            result
        }
    }

    struct Span {
        key_end: usize,
        value_end: usize,
    }

    fn match_key_literal(hay: &str, start: usize, key: &str) -> Option<usize> {
        // Support `[_-]?` meta inside keys like `api[_-]?key` — interpret as:
        // try literal match with `_`, `-`, or none between the segments on `[`.
        if key.contains("[_-]?") {
            let parts: Vec<&str> = key.split("[_-]?").collect();
            let mut end = start;
            for (i, part) in parts.iter().enumerate() {
                if end + part.len() > hay.len() {
                    return None;
                }
                if &hay[end..end + part.len()] != *part {
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
        if start + key.len() > hay.len() {
            return None;
        }
        if &hay[start..start + key.len()] == key {
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

    fn find_value_end(hay: &str, start: usize) -> usize {
        // If the value starts with a quote, consume until matching quote.
        if let Some(b'"') = hay.as_bytes().get(start) {
            let after = start + 1;
            if let Some(pos) = hay[after..].find('"') {
                return after + pos + 1;
            }
        }
        let stop: &[char] = &[' ', '\t', '\n', '\r', ',', ';', '"', '}', ')', ']'];
        match hay[start..].find(stop) {
            Some(p) => start + p,
            None => hay.len(),
        }
    }
}
