//! URL canonicalization and title normalization (B5-1).
//!
//! These are the "strict-equality" normalization routines:
//! - [`canonicalize_url`] — dedup key for URL-based grouping
//! - [`normalize_title`] — dedup key for title-based grouping (not fuzzy)
use std::borrow::Cow;
use unicode_normalization::UnicodeNormalization;

// ── B5-1: URL canonicalisation ──────────────────────────────────────

/// Tracking / session params we always strip (case-insensitive).
/// Mirrors Python `_TRACKING_PARAMS`. Keep in sync — drift here
/// breaks differential tests.
const TRACKING_PARAMS: &[&str] = &[
    "utm_source",
    "utm_medium",
    "utm_campaign",
    "utm_term",
    "utm_content",
    "utm_id",
    "utm_name",
    "utm_referrer",
    "fbclid",
    "gclid",
    "yclid",
    "msclkid",
    "dclid",
    "twclid",
    "mc_cid",
    "mc_eid",
    "_ga",
    "_gl",
    "ref",
    "ref_",
    "referer",
    "referrer",
    "phpsessid",
    "sid",
    "ssid",
    "session",
    "sessionid",
    "spm",
    "from",
];

fn is_tracking(key: &str) -> bool {
    let lower = key.to_ascii_lowercase();
    TRACKING_PARAMS.iter().any(|t| *t == lower)
}

/// Pythonic URL parts. Field semantics match `urllib.parse.urlsplit`.
/// Owned `String`s are used only where canonicalisation must mutate
/// (scheme to lowercase); the rest are borrowed slices over the
/// input.
struct UrlParts<'a> {
    scheme: String,
    netloc: &'a str,
    path: &'a str,
    query: &'a str,
    fragment: &'a str,
}

/// Pythonic `urlsplit` — handles `scheme://netloc/path?q#f`,
/// scheme-relative `//host/p`, and opaque `scheme:opaque` (mailto).
/// Fields are filled in the same order as Python's `SplitResult`,
/// **except** `scheme` is auto-lowercased to mirror Python's
/// (`urlsplit` documents "always lowercase").
fn urlsplit(url: &str) -> UrlParts<'_> {
    // Detect explicit scheme (alpha+ followed by ':' before any of
    // '/', '?', '#'). Mirrors RFC 3986 grammar; Python's urlsplit
    // does the same check.
    let mut scheme_end: Option<usize> = None;
    for (i, c) in url.char_indices() {
        if i == 0 {
            if !c.is_ascii_alphabetic() {
                break;
            }
            continue;
        }
        if c == ':' {
            scheme_end = Some(i);
            break;
        }
        if c == '/' || c == '?' || c == '#' {
            break;
        }
        if !(c.is_ascii_alphanumeric() || c == '+' || c == '-' || c == '.') {
            break;
        }
    }

    let (scheme, after_scheme) = match scheme_end {
        Some(idx) => (url[..idx].to_ascii_lowercase(), &url[idx + 1..]),
        None => (String::new(), url),
    };

    // netloc only when "//" follows the scheme (or when no scheme
    // and the URL starts with "//"). Matches Python urlsplit.
    let (netloc, after_netloc) = if let Some(rest) = after_scheme.strip_prefix("//") {
        let end = rest.find(['/', '?', '#']).unwrap_or(rest.len());
        (&rest[..end], &rest[end..])
    } else {
        ("", after_scheme)
    };

    let (after_path, fragment) = match after_netloc.find('#') {
        Some(idx) => (&after_netloc[..idx], &after_netloc[idx + 1..]),
        None => (after_netloc, ""),
    };
    let (path, query) = match after_path.find('?') {
        Some(idx) => (&after_path[..idx], &after_path[idx + 1..]),
        None => (after_path, ""),
    };

    UrlParts {
        scheme,
        netloc,
        path,
        query,
        fragment,
    }
}

/// Pythonic `urlunsplit` — emits `scheme://netloc/path?query#fragment`
/// with the same semantics as `urllib.parse.urlunsplit`. The "//"
/// authority marker is emitted iff `netloc` is non-empty (matches
/// Python's behavior on schemes like mailto:).
fn urlunsplit(scheme: &str, netloc: &str, path: &str, query: &str, fragment: &str) -> String {
    let mut s = String::with_capacity(
        scheme.len() + netloc.len() + path.len() + query.len() + fragment.len() + 6,
    );
    if !scheme.is_empty() {
        s.push_str(scheme);
        s.push(':');
    }
    if !netloc.is_empty() {
        s.push_str("//");
        s.push_str(netloc);
    }
    s.push_str(path);
    if !query.is_empty() {
        s.push('?');
        s.push_str(query);
    }
    if !fragment.is_empty() {
        s.push('#');
        s.push_str(fragment);
    }
    s
}

/// Return a string-comparable canonical form of `url`. Bit-exact
/// mirror of Python `reconciler.canonicalize_url` — every rule is
/// frozen by `tests/fixtures/reconciler/url_title.json`.
///
/// Rules (in order):
///
/// 1. Strip fragment (`#…`) UNLESS path is empty or `"/"` — in that
///    case the fragment is most likely the only item discriminator
///    (single-page-extraction pattern from agent_research).
/// 2. Lowercase scheme and netloc (host).
/// 3. Drop tracking / session params (case-insensitive).
/// 4. Sort remaining params lexicographically by `(key, value)`.
/// 5. Drop trailing slash on path *unless* path == `"/"`.
///
/// Empty input returns `""`. On parse failure (non-ASCII scheme,
/// etc.) we lowercase netloc only and pass everything else through
/// — same defensive philosophy as Python.
pub fn canonicalize_url(url: &str) -> String {
    if url.is_empty() {
        return String::new();
    }
    let parts = urlsplit(url);

    // Lowercase netloc (scheme is already lowercased by urlsplit).
    let netloc_lc = parts.netloc.to_ascii_lowercase();

    // 1. Fragment policy.
    let fragment = if !parts.fragment.is_empty() && (parts.path.is_empty() || parts.path == "/") {
        parts.fragment
    } else {
        ""
    };

    // 5. Trailing slash policy. Root stays "/", everything else
    // drops a single trailing slash. Multi-trailing not collapsed
    // (B5.5 territory).
    let path: Cow<'_, str> = if parts.path.len() > 1 && parts.path.ends_with('/') {
        Cow::Owned(parts.path[..parts.path.len() - 1].to_string())
    } else {
        Cow::Borrowed(parts.path)
    };

    // 3 + 4. Param scrub + sort. Manual parse to preserve duplicate
    // keys and value encoding (Python comment notes parse_qs would
    // collapse and re-encode differently from raw scraper output).
    let query_out = if parts.query.is_empty() {
        String::new()
    } else {
        let mut kept: Vec<(&str, &str)> = parts
            .query
            .split('&')
            .filter_map(|kv| {
                if kv.is_empty() {
                    return None;
                }
                let (k, v) = match kv.find('=') {
                    Some(idx) => (&kv[..idx], &kv[idx + 1..]),
                    None => (kv, ""),
                };
                if is_tracking(k) { None } else { Some((k, v)) }
            })
            .collect();
        kept.sort_unstable();
        let mut q = String::with_capacity(parts.query.len());
        for (i, (k, v)) in kept.iter().enumerate() {
            if i > 0 {
                q.push('&');
            }
            q.push_str(k);
            q.push('=');
            q.push_str(v);
        }
        q
    };

    urlunsplit(&parts.scheme, &netloc_lc, &path, &query_out, fragment)
}

// ── B5-1: Title normalisation (strict-equality, NOT for Jaccard) ────

/// Translate a single non-ASCII character to its B5 surface-cleanup
/// equivalent. Mirrors Python `_QUOTE_TRANSLATIONS`. Returning
/// `Some("")` means "drop this char" (zero-width, BOM).
fn translate_punct(c: char) -> Option<&'static str> {
    match c as u32 {
        // Curly double quotes → ASCII "
        0x201C | 0x201D | 0x201E | 0x201F | 0x00AB | 0x00BB => Some("\""),
        // Curly single quotes → ASCII '
        0x2018..=0x201B => Some("'"),
        // Em-dash, en-dash, minus → ASCII hyphen-minus
        0x2014 | 0x2013 | 0x2212 => Some("-"),
        // Non-breaking space → regular space
        0x00A0 => Some(" "),
        // Zero-width / BOM → drop
        0x200B | 0x200C | 0x200D | 0xFEFF => Some(""),
        _ => None,
    }
}

/// Lowercase, NFKC-normalise unicode, normalise quotes/dashes/NBSP,
/// collapse whitespace. Bit-exact mirror of Python
/// `reconciler.normalize_title`.
///
/// **Not for fuzzy matching.** See module-level docstring; for
/// Jaccard similarity use
/// [`crate::research::spec::normalize_title_for_similarity`].
pub(crate) fn normalize_title(title: &str) -> String {
    if title.is_empty() {
        return String::new();
    }

    // 1. Punctuation translation (curly quotes, em-dash, NBSP, ZWJ…).
    let mut translated = String::with_capacity(title.len());
    for c in title.chars() {
        match translate_punct(c) {
            Some(rep) => translated.push_str(rep),
            None => translated.push(c),
        }
    }

    // 2. NFKC normalise. Order matches Python:
    //    `.translate(...)` then `unicodedata.normalize("NFKC", ...)`.
    let nfkc: String = translated.nfkc().collect();

    // 3. Lowercase. Python uses `str.lower()` which honours Unicode
    //    case mappings; Rust's `to_lowercase` does the same.
    let lowered = nfkc.to_lowercase();

    // 4. Collapse runs of whitespace and trim. Mirrors
    //    `re.sub(r"\s+", " ", s).strip()`.
    let mut out = String::with_capacity(lowered.len());
    let mut prev_ws = false;
    for c in lowered.chars() {
        if c.is_whitespace() {
            if !prev_ws && !out.is_empty() {
                out.push(' ');
            }
            prev_ws = true;
        } else {
            out.push(c);
            prev_ws = false;
        }
    }
    if out.ends_with(' ') {
        out.pop();
    }
    out
}
