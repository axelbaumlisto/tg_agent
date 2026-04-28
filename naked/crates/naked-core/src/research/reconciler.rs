//! `reconciler` — block B5 of the universal-research blueprint, Rust port.
//!
//! This module is the deterministic counterpart of Python
//! `scripts/reconciler.py`. The two implementations share **one
//! contract**, frozen in `tests/fixtures/reconciler/*.json`. Every
//! function here is gated against the same JSON cases as the Python
//! script — that's the differential test (group 33 in
//! `tests/run_all_offline.sh`).
//!
//! Why two `normalize_title` functions live in this crate
//! ──────────────────────────────────────────────────────
//! Yes, [`crate::research::spec::normalize_title_for_similarity`] also
//! exists. It's a different concept:
//!
//! * `spec::normalize_title_for_similarity` — aggressive **stemming**
//!   (drops digit tokens ≤ 4 chars, drops the `m2`/`sqm`/`tang`/`pn`
//!   stopword set, no quote/dash translation). Used by
//!   [`crate::research::spec::titles_are_similar`] for **fuzzy
//!   Jaccard matching** between near-duplicate listings on different
//!   crossposts.
//! * `reconciler::normalize_title` (this module) — conservative
//!   **surface cleanup** (lowercase, NFKC, quote/dash/NBSP translation,
//!   whitespace collapse; **no** stemming). Used by
//!   [`reconciler::reconcile`] for **strict-equality dedup** where
//!   "BMW X5" and "BMW X5 (urgent)" must NOT collapse.
//!
//! Both are kept; both are correct for their use case. The
//! cross-reference here and in `spec.rs` is the boy-scout fix to
//! prevent future contributors from "deduplicating" the duplicate.
//!
//! Port-now scope
//! ──────────────
//! * **R1.5-step-1 (2026-04-27)** — `canonicalize_url` + `normalize_title`.
//! * **R1.5-step-2 (2026-04-27 +1)** — `extract_price` + `to_usd`.
//! * **R1.5-step-3 (2026-04-27 +2)** — top-level `reconcile()`.
//!
//! With step-3 the entire deterministic CPU path of the Python
//! `scripts/reconciler.py` has a bit-exact Rust counterpart. Group
//! 33 (`reconciler_rust_differential_offline`) gates 6 Rust tests
//! against `tests/fixtures/reconciler/{url_title,price,dedup}.json`
//! — the same JSON the Python e2e groups (28/29/30) consume. Drift
//! on either side fails on either CI run.
//!
//! ## API note: the `reconcile()` shape
//!
//! Python's signature is
//! `reconcile(findings: list[dict], fx_rates=None, merge_by_title=False) -> list[dict]`.
//! Rust returns `Vec<serde_json::Value>` so callers don't have to
//! commit to a typed Finding struct (the cross-source pipeline
//! handles many shapes — `agent_research`, `playwright_generic`,
//! `discover` — and dedup must be schema-tolerant). The function is
//! still pure: input slice in, owned `Vec` out, no I/O, no globals.

use std::borrow::Cow;
use std::collections::HashMap;
use std::sync::LazyLock;
use regex::Regex;
use serde_json::{Map, Value, json};
use unicode_normalization::UnicodeNormalization;

// ── B5-1: URL canonicalisation ──────────────────────────────────────

/// Tracking / session params we always strip (case-insensitive).
/// Mirrors Python `_TRACKING_PARAMS`. Keep in sync — drift here
/// breaks differential tests.
const TRACKING_PARAMS: &[&str] = &[
    "utm_source", "utm_medium", "utm_campaign", "utm_term", "utm_content",
    "utm_id", "utm_name", "utm_referrer",
    "fbclid", "gclid", "yclid", "msclkid", "dclid", "twclid",
    "mc_cid", "mc_eid",
    "_ga", "_gl",
    "ref", "ref_", "referer", "referrer",
    "phpsessid",
    "sid", "ssid", "session", "sessionid",
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
            if !c.is_ascii_alphabetic() { break; }
            continue;
        }
        if c == ':' { scheme_end = Some(i); break; }
        if c == '/' || c == '?' || c == '#' { break; }
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

    UrlParts { scheme, netloc, path, query, fragment }
}

/// Pythonic `urlunsplit` — emits `scheme://netloc/path?query#fragment`
/// with the same semantics as `urllib.parse.urlunsplit`. The "//"
/// authority marker is emitted iff `netloc` is non-empty (matches
/// Python's behavior on schemes like mailto:).
fn urlunsplit(scheme: &str, netloc: &str, path: &str,
              query: &str, fragment: &str) -> String {
    let mut s = String::with_capacity(
        scheme.len() + netloc.len() + path.len()
        + query.len() + fragment.len() + 6
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
    let fragment = if !parts.fragment.is_empty()
        && (parts.path.is_empty() || parts.path == "/")
    {
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
        let mut kept: Vec<(&str, &str)> = parts.query
            .split('&')
            .filter_map(|kv| {
                if kv.is_empty() { return None; }
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
            if i > 0 { q.push('&'); }
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
pub fn normalize_title(title: &str) -> String {
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

// ── B5-3: Price extraction (R1.5-step-2) ────────────────────────────

/// Static FX table: 1 unit of `currency` → USD. Rates pinned
/// 2026-04-26. Mirrors Python `_FX_TO_USD`. Refresh policy is
/// quarterly — the absolute accuracy is irrelevant for cross-source
/// dedup; only consistent ordering matters. Callers needing live
/// rates pass `fx_rates: &[(&str, f64)]` to [`to_usd`].
const FX_TO_USD: &[(&str, f64)] = &[
    ("USD", 1.0),
    ("EUR", 1.08),
    ("GBP", 1.27),
    ("JPY", 0.0067),
    ("CNY", 0.138),
    ("RUB", 0.0107),
    ("KZT", 0.00194),
    ("THB", 0.0276),
    ("AMD", 0.00255),
    ("VND", 0.0000395),
    ("IDR", 0.0000595),
    ("MYR", 0.224),
    ("BYN", 0.305),
    ("UZS", 0.0000785),
    ("GEL", 0.36),
    ("UAH", 0.024),
];

fn lookup_rate(table: &[(&str, f64)], iso: &str) -> Option<f64> {
    let upper = iso.to_ascii_uppercase();
    table.iter()
        .find(|(c, _)| c.eq_ignore_ascii_case(&upper))
        .map(|(_, r)| *r)
}

/// Currency-suffix tokens (numeric appears BEFORE the token). Mirrors
/// Python `_CURRENCY_SUFFIX_TOKENS`. Order doesn't matter for the
/// dict lookup — alternation order in the regex is sorted by length
/// descending so longer matches (e.g. `руб.`) win over shorter
/// (`руб`).
const SUFFIX_TOKENS: &[(&str, &str)] = &[
    ("₽", "RUB"), ("руб", "RUB"), ("руб.", "RUB"), ("рубль", "RUB"),
    ("рубля", "RUB"), ("рублей", "RUB"),
    ("₸", "KZT"), ("тг", "KZT"), ("тг.", "KZT"), ("тенге", "KZT"),
    ("฿", "THB"), ("baht", "THB"), ("бат", "THB"),
    ("֏", "AMD"), ("драм", "AMD"), ("dram", "AMD"),
    ("₫", "VND"), ("đồng", "VND"),
    ("Rp", "IDR"), ("rp", "IDR"),
    ("RM", "MYR"), ("ringgit", "MYR"),
    ("₾", "GEL"), ("lari", "GEL"), ("лари", "GEL"),
    ("₴", "UAH"), ("грн", "UAH"),
    // ISO codes also valid as bare suffixes after a number.
    ("USD", "USD"), ("EUR", "EUR"), ("GBP", "GBP"), ("JPY", "JPY"),
    ("CNY", "CNY"),
    ("RUB", "RUB"), ("KZT", "KZT"), ("THB", "THB"), ("AMD", "AMD"),
    ("VND", "VND"),
    ("IDR", "IDR"), ("MYR", "MYR"), ("BYN", "BYN"), ("UZS", "UZS"),
    ("GEL", "GEL"), ("UAH", "UAH"),
];

/// Currency-prefix symbols (token appears BEFORE the number).
const PREFIX_TOKENS: &[(&str, &str)] = &[
    ("$", "USD"),
    ("€", "EUR"),
    ("£", "GBP"),
    ("¥", "JPY"),
];

/// Multiplier tokens. Sorted by length desc by `build_alternation`
/// so `миллиард` beats `миллион` etc.
const MULTIPLIERS: &[(&str, f64)] = &[
    ("млрд", 1e9), ("миллиард", 1e9), ("миллиардов", 1e9),
    ("млн", 1e6),  ("миллион", 1e6),  ("миллионов", 1e6),
    ("тыс", 1e3),  ("тысяч", 1e3),    ("тысячи", 1e3),
    ("billion", 1e9), ("million", 1e6), ("thousand", 1e3),
    ("tỷ", 1e9), ("ty", 1e9),
    ("triệu", 1e6), ("trieu", 1e6),
    ("nghìn", 1e3), ("nghin", 1e3),
];

/// Number pattern: optional thousand-grouping with space/NBSP/comma/dot,
/// optional decimal tail. Greedy so `1 234 567,89` parses as one
/// token. Mirrors Python `_NUMBER_RE`.
const NUMBER_RE: &str = r"\d{1,3}(?:[\s\u{00a0},.]\d{3})+(?:[.,]\d{1,2})?|\d+(?:[.,]\d{1,3})?";

/// Build an alternation regex source from a list of literal tokens,
/// sorted by character-count descending so longer tokens win the
/// regex engine's left-to-right alternation race. Mirrors Python
/// `_build_alternation`.
fn build_alternation<'a, I>(tokens: I) -> String
where
    I: IntoIterator<Item = &'a str>,
{
    let mut sorted: Vec<&str> = tokens.into_iter().collect();
    sorted.sort_by_key(|s| std::cmp::Reverse(s.chars().count()));
    sorted.iter().map(|t| regex::escape(t)).collect::<Vec<_>>().join("|")
}

/// `(amount [mult] curr)` regex — mirrors Python `_RE_SUFFIX` minus
/// the negative lookahead (Rust `regex` doesn't support lookaround;
/// we emulate it post-match in [`extract_price`]).
static RE_SUFFIX: LazyLock<Regex> = LazyLock::new(|| {
    let suffix_alt = build_alternation(SUFFIX_TOKENS.iter().map(|(t, _)| *t));
    let mult_alt   = build_alternation(MULTIPLIERS.iter().map(|(t, _)| *t));
    let pat = format!(
        r"(?i)(?P<amount>{NUMBER_RE})(?:\s*(?P<mult>{mult_alt}))?\s*(?P<curr>{suffix_alt})"
    );
    Regex::new(&pat).expect("RE_SUFFIX compile")
});

/// `(curr amount [mult])` regex — mirrors Python `_RE_PREFIX`.
static RE_PREFIX: LazyLock<Regex> = LazyLock::new(|| {
    let prefix_alt = build_alternation(PREFIX_TOKENS.iter().map(|(t, _)| *t));
    let mult_alt   = build_alternation(MULTIPLIERS.iter().map(|(t, _)| *t));
    let pat = format!(
        r"(?i)(?P<curr>{prefix_alt})\s*(?P<amount>{NUMBER_RE})(?:\s*(?P<mult>{mult_alt}))?"
    );
    Regex::new(&pat).expect("RE_PREFIX compile")
});

/// Bare `(amount [mult])` regex — mirrors Python `_RE_BARE`. Used
/// only when the caller passes `default_currency`.
static RE_BARE: LazyLock<Regex> = LazyLock::new(|| {
    let mult_alt = build_alternation(MULTIPLIERS.iter().map(|(t, _)| *t));
    let pat = format!(
        r"(?i)(?P<amount>{NUMBER_RE})(?:\s*(?P<mult>{mult_alt}))?"
    );
    Regex::new(&pat).expect("RE_BARE compile")
});

/// Convert a number-string with mixed thousand/decimal separators
/// into f64. Matches Python `_parse_amount` rules:
///
/// * strip whitespace and NBSP
/// * if both `,` and `.` present, the right-most is the decimal
/// * if only `,`: 1-2-digit tail = decimal, else thousand sep
/// * if only `.`: 1-2-digit tail = decimal, else thousand sep
/// * otherwise plain `parse::<f64>()`
fn parse_amount(s: &str) -> Option<f64> {
    if s.is_empty() { return None; }
    // Strip ASCII whitespace and U+00A0 NBSP.
    let stripped: String = s.chars()
        .filter(|c| !c.is_whitespace())
        .collect();
    if stripped.is_empty() { return None; }

    let has_comma = stripped.contains(',');
    let has_dot   = stripped.contains('.');

    let normalised: String = if has_comma && has_dot {
        let last_comma = stripped.rfind(',').unwrap();
        let last_dot   = stripped.rfind('.').unwrap();
        if last_comma > last_dot {
            stripped.replace('.', "").replace(',', ".")
        } else {
            stripped.replace(',', "")
        }
    } else if has_comma {
        let comma_count = stripped.matches(',').count();
        let tail_len = stripped.rsplit_once(',').map(|(_, t)| t.len()).unwrap_or(0);
        if comma_count == 1 && (1..=2).contains(&tail_len) {
            stripped.replace(',', ".")
        } else {
            stripped.replace(',', "")
        }
    } else if has_dot {
        let dot_count = stripped.matches('.').count();
        let tail_len  = stripped.rsplit_once('.').map(|(_, t)| t.len()).unwrap_or(0);
        if dot_count == 1 && (1..=2).contains(&tail_len) {
            stripped
        } else {
            stripped.replace('.', "")
        }
    } else {
        stripped
    };

    normalised.parse::<f64>().ok()
}

fn resolve_multiplier(mult: Option<&str>) -> f64 {
    let Some(m) = mult else { return 1.0; };
    if m.is_empty() { return 1.0; }
    let lower = m.to_lowercase();
    MULTIPLIERS.iter()
        .find(|(t, _)| *t == lower.as_str())
        .map(|(_, v)| *v)
        .unwrap_or(1.0)
}

/// Look up a currency suffix token in `SUFFIX_TOKENS` with the
/// 3-step fallback that matches Python:
///   1. exact case
///   2. uppercase
///   3. lowercase
fn resolve_suffix_currency(token: &str) -> Option<&'static str> {
    let lookup = |needle: &str| -> Option<&'static str> {
        SUFFIX_TOKENS.iter()
            .find(|(t, _)| *t == needle)
            .map(|(_, iso)| *iso)
    };
    lookup(token)
        .or_else(|| lookup(&token.to_uppercase()))
        .or_else(|| lookup(&token.to_lowercase()))
}

fn resolve_prefix_currency(token: &str) -> Option<&'static str> {
    PREFIX_TOKENS.iter()
        .find(|(t, _)| *t == token)
        .map(|(_, iso)| *iso)
}

/// Negative-lookahead emulation: returns true iff the char at
/// `byte_offset` in `text` exists AND is alphabetic. Used to filter
/// out matches like `рубля` inside `рублям` (the `я→м` boundary)
/// — that's the same defensive check Python encodes as
/// `(?![A-Za-zА-Яа-яёЁ])` in `_RE_SUFFIX`.
fn next_char_is_alpha(text: &str, byte_offset: usize) -> bool {
    text[byte_offset..]
        .chars()
        .next()
        .is_some_and(|c| c.is_alphabetic())
}

/// Extracted price record. Equivalent to Python's
/// `{amount, currency, raw}` dict — kept as a struct so callers can
/// pattern-match on missing fields without `serde_json::Value`
/// gymnastics.
#[derive(Debug, Clone, PartialEq)]
pub struct ExtractedPrice {
    pub amount: f64,
    pub currency: &'static str,
    pub raw: String,
}

/// Find the FIRST plausible price token in `text`. Bit-exact mirror
/// of Python `reconciler.extract_price`.
///
/// Strategy:
/// 1. Try the suffix pattern (`21 900 ₽`, `8.5 млн ₸`).
/// 2. Try the prefix pattern (`$250,000`).
/// 3. If `default_currency` is set, try a bare-amount match and
///    label the hit with that currency.
///
/// The amount is multiplied by the resolved multiplier (`млн`,
/// `tỷ`, `million`, …) so the returned `amount` is always in unit
/// currency.
pub fn extract_price(text: &str, default_currency: Option<&str>)
    -> Option<ExtractedPrice>
{
    if text.is_empty() { return None; }

    // Helper that scans `re` for the first match whose `curr` group
    // isn't followed by an alphabetic char. Returns (raw, amount,
    // currency-token).
    fn scan<'t>(re: &Regex, text: &'t str, is_prefix: bool)
        -> Option<(&'t str, f64, &'static str)>
    {
        for caps in re.captures_iter(text) {
            let m_full  = caps.get(0)?;
            let amt_grp = caps.name("amount")?;
            let mult    = caps.name("mult").map(|m| m.as_str());
            let curr_g  = caps.name("curr")?;

            // Defensive negative-lookahead emulation: the `curr`
            // token must NOT be a prefix of a longer alpha word.
            // (Skipped for the prefix pattern — there `curr` is a
            // single ASCII/symbol char like `$` and the constraint
            // is moot.)
            if !is_prefix && next_char_is_alpha(text, curr_g.end()) {
                continue;
            }

            let Some(amount_raw) = parse_amount(amt_grp.as_str()) else {
                continue;
            };
            let multiplier = resolve_multiplier(mult);

            let curr_token = curr_g.as_str().trim();
            let iso = if is_prefix {
                resolve_prefix_currency(curr_token)
            } else {
                resolve_suffix_currency(curr_token)
            };
            let Some(iso) = iso else { continue };

            return Some((m_full.as_str().trim(), amount_raw * multiplier, iso));
        }
        None
    }

    if let Some((raw, amount, iso)) = scan(&RE_SUFFIX, text, false) {
        return Some(ExtractedPrice {
            amount,
            currency: iso,
            raw: raw.to_string(),
        });
    }
    if let Some((raw, amount, iso)) = scan(&RE_PREFIX, text, true) {
        return Some(ExtractedPrice {
            amount,
            currency: iso,
            raw: raw.to_string(),
        });
    }

    // Bare-amount fallback only when the caller has explicitly
    // promised a currency hint. The default currency must match an
    // ISO code that we know how to convert to USD; if it doesn't, we
    // still tag the hit (FX missing is `to_usd`'s problem, not
    // `extract_price`'s) but log it via the canonical ISO list to
    // keep the `&'static str` field honest.
    if let Some(default) = default_currency
        && let Some(caps) = RE_BARE.captures(text)
        && let Some(amt_grp) = caps.name("amount")
        && let Some(amount_raw) = parse_amount(amt_grp.as_str())
        && let Some(m_full) = caps.get(0)
    {
        let multiplier = resolve_multiplier(caps.name("mult").map(|m| m.as_str()));
        let upper = default.to_ascii_uppercase();
        // Re-borrow the canonical ISO from the FX table so the
        // returned struct's `currency: &'static str` stays valid.
        // Unknown codes fall through to "XXX" (visible to callers,
        // matches Python's `default_currency.upper()` semantics
        // close enough — they then fail the `to_usd` lookup the
        // same way).
        let iso_static: &'static str = FX_TO_USD.iter()
            .find(|(c, _)| c.eq_ignore_ascii_case(&upper))
            .map(|(c, _)| *c)
            .unwrap_or("XXX");
        return Some(ExtractedPrice {
            amount: amount_raw * multiplier,
            currency: iso_static,
            raw: m_full.as_str().trim().to_string(),
        });
    }
    None
}

/// Convert a `(amount, currency)` pair to USD, using `fx_rates` if
/// supplied or [`FX_TO_USD`] otherwise. Returns None on
/// missing/unknown currency. Result is rounded to 2 decimal places
/// so cross-source ordering is stable.
///
/// Mirrors Python `to_usd(price: dict, fx_rates=None) -> float|None`
/// with the dict-input flattened into an explicit `(amount, currency)`
/// pair — no `Option` keys to second-guess.
pub fn to_usd(amount: f64, currency: &str,
              fx_rates: Option<&[(&str, f64)]>) -> Option<f64> {
    if !amount.is_finite() { return None; }
    let table = fx_rates.unwrap_or(FX_TO_USD);
    let rate = lookup_rate(table, currency)?;
    Some(((amount * rate) * 100.0).round() / 100.0)
}

// ── B5-2: top-level reconcile() (R1.5-step-3) ────────────────────────

/// Tie-breaker for winner pick. Prefers `_score`, falls back to
/// `_relevance`, falls back to 0.0. Bit-exact mirror of Python
/// `_winning_score`. `Value::Bool` is rejected by construction —
/// `Value::Number` cannot contain a bool, so the Python "bool is
/// a subclass of int" gotcha can't surface here.
fn winning_score(finding: &Value) -> f64 {
    fn pick(v: Option<&Value>) -> Option<f64> {
        match v {
            Some(Value::Number(n)) => n.as_f64(),
            _ => None,
        }
    }
    pick(finding.get("_score"))
        .or_else(|| pick(finding.get("_relevance")))
        .unwrap_or(0.0)
}

/// Read a string field, returning `""` for missing / non-string /
/// `null`. Matches Python `f.get(key) or ""` for the four text
/// fields used in dedup (`link`, `url`, `title`, `snippet`,
/// `description`).
fn finding_str<'a>(finding: &'a Value, key: &str) -> &'a str {
    finding.get(key).and_then(Value::as_str).unwrap_or("")
}

/// Combine the three text fields most likely to hold a price.
/// Order: title | snippet | description. Mirrors Python
/// `_price_signal_text`. The " · " separator matches byte-for-byte.
fn price_signal_text(finding: &Value) -> String {
    ["title", "snippet", "description"]
        .iter()
        .map(|k| finding_str(finding, k))
        .filter(|s| !s.is_empty())
        .collect::<Vec<_>>()
        .join(" · ")
}

/// Union-find `find` with path compression, applied to a flat
/// `parent` slice. Free function (no closures over outer state)
/// so the borrow checker stays out of the way.
fn uf_find(parent: &mut [usize], mut x: usize) -> usize {
    while parent[x] != x {
        parent[x] = parent[parent[x]];
        x = parent[x];
    }
    x
}

/// Union with **min-becomes-root**: the smallest index in any group
/// is always the root. This is what guarantees that
/// `_canonical_url` / `_normalised_title` bucket iteration order
/// does not affect the final group membership — Python relies on
/// the same property.
fn uf_union(parent: &mut [usize], a: usize, b: usize) {
    let ra = uf_find(parent, a);
    let rb = uf_find(parent, b);
    if ra != rb {
        let new_root = ra.min(rb);
        let old_root = ra.max(rb);
        parent[old_root] = new_root;
    }
}

/// Cross-source merge of findings. Bit-exact Rust port of Python
/// `reconciler.reconcile(findings, fx_rates, merge_by_title)`.
///
/// # Determinism contract
/// - Output preserves first-seen group order.
/// - Within a group: max `_score` → max `_relevance` → first-seen.
/// - `_sources` sorted by `_source_id` (missing/non-string → "").
///
/// # Schema-added keys (each survivor)
/// - `_canonical_url` — `canonicalize_url(winner.link or .url)`
/// - `_normalised_title` — `normalize_title(winner.title)`
/// - `_sources` — `Vec<{_source_id,_score,_relevance}>`, sorted
/// - `_price_struct` — `{amount,currency,raw}` or `null`
/// - `price_usd` — `f64` or `null`
///
/// # `merge_by_title` opt-in
/// Default is **URL-only dedup**. Title-only merge is risky on
/// marketplace corpora that share CTA-style titles ("узнать
/// наличие", "Apartment", "BMW M5"); see the Python module docstring
/// for the corpus survey behind the default.
pub fn reconcile(
    findings: &[Value],
    fx_rates: Option<&[(&str, f64)]>,
    merge_by_title: bool,
) -> Vec<Value> {
    // 1. Filter non-object entries; pre-compute url/title keys.
    struct Item<'a> {
        src: &'a Value,
        url_key: String,
        title_key: String,
    }
    let items: Vec<Item> = findings
        .iter()
        .filter(|f| f.is_object())
        .map(|f| {
            let link = finding_str(f, "link");
            let url = if link.is_empty() { finding_str(f, "url") } else { link };
            Item {
                src: f,
                url_key: canonicalize_url(url),
                title_key: normalize_title(finding_str(f, "title")),
            }
        })
        .collect();
    if items.is_empty() {
        return Vec::new();
    }

    let n = items.len();

    // 2. Bucket by each non-empty signal.
    //
    // We use HashMap (not BTreeMap) because bucket iteration order
    // does not affect the final union-find result — the
    // min-becomes-root invariant in `uf_union` makes the root of
    // every group the smallest index in it, regardless of the order
    // in which buckets are processed.
    let mut by_url: HashMap<&str, Vec<usize>> = HashMap::new();
    let mut by_title: HashMap<&str, Vec<usize>> = HashMap::new();
    for (i, it) in items.iter().enumerate() {
        if !it.url_key.is_empty() {
            by_url.entry(it.url_key.as_str()).or_default().push(i);
        }
        if !it.title_key.is_empty() {
            by_title.entry(it.title_key.as_str()).or_default().push(i);
        }
    }

    // 3. Union members of each bucket pairwise.
    let mut parent: Vec<usize> = (0..n).collect();
    for members in by_url.values() {
        for &j in members.iter().skip(1) {
            uf_union(&mut parent, members[0], j);
        }
    }
    if merge_by_title {
        for members in by_title.values() {
            for &j in members.iter().skip(1) {
                uf_union(&mut parent, members[0], j);
            }
        }
    }

    // 4. Collect groups preserving first-seen group order.
    let mut groups: HashMap<usize, Vec<usize>> = HashMap::new();
    let mut group_order: Vec<usize> = Vec::new();
    for i in 0..n {
        let root = uf_find(&mut parent, i);
        if !groups.contains_key(&root) {
            group_order.push(root);
        }
        groups.entry(root).or_default().push(i);
    }

    // 5. Per group: pick winner, build the merged record.
    let mut out: Vec<Value> = Vec::with_capacity(group_order.len());
    for root in group_order {
        let member_idx = &groups[&root];
        let bucket: Vec<&Value> = member_idx.iter().map(|&i| items[i].src).collect();

        // Winner: max score, ties broken by smaller in-bucket index.
        // Mirrors Python `max(range(len(bucket)), key=lambda i: (score, -i))`.
        let winner_local: usize = (0..bucket.len())
            .max_by(|&a, &b| {
                let sa = winning_score(bucket[a]);
                let sb = winning_score(bucket[b]);
                match sa.partial_cmp(&sb).unwrap_or(std::cmp::Ordering::Equal) {
                    std::cmp::Ordering::Equal => b.cmp(&a),
                    other => other,
                }
            })
            .unwrap_or(0);

        // Clone the winner so we can mutate without poisoning input.
        let mut winner: Map<String, Value> =
            bucket[winner_local].as_object().cloned().unwrap_or_default();

        // Recompute canonical fields from the winner — Python
        // explicitly recomputes here (rather than using the
        // first-seen's keys) so cross-group carries the survivor's
        // URL/title in the merged record.
        let winner_url = winner
            .get("link")
            .and_then(Value::as_str)
            .filter(|s| !s.is_empty())
            .or_else(|| winner.get("url").and_then(Value::as_str))
            .unwrap_or("");
        let canonical_url = canonicalize_url(winner_url);
        let normalised_title = normalize_title(
            winner.get("title").and_then(Value::as_str).unwrap_or(""),
        );

        // _sources: one entry per group member, sorted by _source_id.
        // Missing or non-string _source_id sorts as empty string —
        // mirrors Python `s.get('_source_id') or ''`.
        let mut sources_meta: Vec<Value> = bucket
            .iter()
            .map(|f| {
                let mut m = Map::with_capacity(3);
                m.insert("_source_id".into(), f.get("_source_id").cloned().unwrap_or(Value::Null));
                m.insert("_score".into(), f.get("_score").cloned().unwrap_or(Value::Null));
                m.insert("_relevance".into(), f.get("_relevance").cloned().unwrap_or(Value::Null));
                Value::Object(m)
            })
            .collect();
        sources_meta.sort_by(|a, b| {
            let sa = a.get("_source_id").and_then(Value::as_str).unwrap_or("");
            let sb = b.get("_source_id").and_then(Value::as_str).unwrap_or("");
            sa.cmp(sb)
        });

        // Price extraction: title | snippet | description joined by " · ".
        let signal = price_signal_text(&Value::Object(winner.clone()));
        let price_struct = extract_price(&signal, None);
        let price_usd: Option<f64> = price_struct
            .as_ref()
            .and_then(|p| to_usd(p.amount, p.currency, fx_rates));

        winner.insert("_canonical_url".into(), Value::String(canonical_url));
        winner.insert("_normalised_title".into(), Value::String(normalised_title));
        winner.insert("_sources".into(), Value::Array(sources_meta));
        winner.insert(
            "_price_struct".into(),
            match price_struct {
                Some(p) => json!({
                    "amount":   p.amount,
                    "currency": p.currency,
                    "raw":      p.raw,
                }),
                None => Value::Null,
            },
        );
        winner.insert(
            "price_usd".into(),
            match price_usd {
                Some(v) => json!(v),
                None => Value::Null,
            },
        );

        out.push(Value::Object(winner));
    }
    out
}

// ── tests ────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::Value;
    use std::path::PathBuf;

    /// Build a path under `tests/fixtures/reconciler/` relative to
    /// the workspace root. Mirrors the Python e2e harness which
    /// consumes the same JSON from the same place — that's the
    /// shared contract.
    fn fixture_path(name: &str) -> PathBuf {
        // CARGO_MANIFEST_DIR = .../naked/crates/naked-core
        // → walk up two levels to reach the workspace root
        //   (.../naked) which holds `tests/`.
        let mut p = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        p.pop(); // crates
        p.pop(); // naked
        p.push("tests/fixtures/reconciler");
        p.push(name);
        p
    }

    fn url_title_fixture_path() -> PathBuf { fixture_path("url_title.json") }
    fn price_fixture_path() -> PathBuf { fixture_path("price.json") }

    #[test]
    fn canonicalize_url_matches_python_fixture() {
        let raw = std::fs::read_to_string(url_title_fixture_path())
            .expect("read fixture");
        let v: Value = serde_json::from_str(&raw).expect("parse fixture");

        let mut failures: Vec<String> = Vec::new();
        for case in v["url_cases"].as_array().expect("url_cases array") {
            let name  = case["name"].as_str().unwrap_or("?");
            let input = case["input"].as_str().unwrap_or("");
            let exp   = case["expected"].as_str().unwrap_or("");
            let got   = canonicalize_url(input);
            if got != exp {
                failures.push(format!(
                    "[{name}] input={input:?}\n  got={got:?}\n  exp={exp:?}"
                ));
                eprintln!(
                    "[FAIL] {name}\n  input    = {input:?}\n  got      = {got:?}\n  expected = {exp:?}"
                );
            }
        }
        assert!(
            failures.is_empty(),
            "{} URL canonicalisation case(s) drift from Python contract",
            failures.len(),
        );
    }

    #[test]
    fn normalize_title_matches_python_fixture() {
        let raw = std::fs::read_to_string(url_title_fixture_path())
            .expect("read fixture");
        let v: Value = serde_json::from_str(&raw).expect("parse fixture");

        let mut failures = 0usize;
        for case in v["title_cases"].as_array().expect("title_cases array") {
            let name = case["name"].as_str().unwrap_or("?");
            let exp  = case["expected"].as_str().unwrap_or("");

            // Python's contract: non-string input → empty string.
            // Rust signature is `&str` already so a JSON `null` is
            // explicitly handled as "skip non-string check is moot
            // — pass empty &str instead".
            let input_owned: String = match &case["input"] {
                Value::String(s) => s.clone(),
                Value::Null      => String::new(),
                other => panic!(
                    "[{name}] non-string non-null input not supported in Rust impl: {other:?}"
                ),
            };
            let got = normalize_title(&input_owned);
            if got != exp {
                eprintln!(
                    "[FAIL] {name}\n  input    = {input_owned:?}\n  got      = {got:?}\n  expected = {exp:?}"
                );
                failures += 1;
            }
        }
        assert_eq!(failures, 0,
            "{failures} title-normalisation case(s) drift from Python contract"
        );
    }

    // ── R1.5-step-2: extract_price + to_usd ──────────────────────────

    /// Build the FX table from the fixture's `_fx_pin` block — a
    /// pinned snapshot is the only way the differential test stays
    /// stable across rate refreshes. Keeps the test independent of
    /// any drift in the in-code [`FX_TO_USD`] table.
    fn fixture_fx(v: &Value) -> Vec<(String, f64)> {
        v["_fx_pin"].as_object()
            .expect("_fx_pin object")
            .iter()
            .filter(|(k, _)| !k.starts_with('_'))
            .filter_map(|(k, v)| v.as_f64().map(|f| (k.clone(), f)))
            .collect()
    }

    #[test]
    fn extract_price_matches_python_fixture() {
        let raw = std::fs::read_to_string(price_fixture_path())
            .expect("read price fixture");
        let v: Value = serde_json::from_str(&raw).expect("parse fixture");

        let mut failures = 0usize;
        for case in v["extract_cases"].as_array().expect("extract_cases array") {
            let name  = case["name"].as_str().unwrap_or("?");
            let input = case["input"].as_str().unwrap_or("");
            let default_currency = case.get("default_currency")
                .and_then(|v| v.as_str());
            let got = extract_price(input, default_currency);
            let exp = &case["expected"];

            // Three branches: expected null, expected dict, mismatch.
            match (exp, &got) {
                (Value::Null, None) => continue,
                (Value::Null, Some(g)) => {
                    eprintln!(
                        "[FAIL] {name}: expected None, got {g:?}\n  input = {input:?}"
                    );
                    failures += 1;
                }
                (_, None) => {
                    eprintln!(
                        "[FAIL] {name}: expected {exp}, got None\n  input = {input:?}"
                    );
                    failures += 1;
                }
                (exp_obj, Some(g)) => {
                    let exp_amount = exp_obj["amount"].as_f64().unwrap_or(f64::NAN);
                    let exp_curr   = exp_obj["currency"].as_str().unwrap_or("");
                    if (g.amount - exp_amount).abs() > 1e-6 {
                        eprintln!(
                            "[FAIL] {name}: amount {} ≠ expected {}\n  input = {input:?}",
                            g.amount, exp_amount
                        );
                        failures += 1;
                    }
                    if g.currency != exp_curr {
                        eprintln!(
                            "[FAIL] {name}: currency {:?} ≠ expected {:?}\n  input = {input:?}",
                            g.currency, exp_curr
                        );
                        failures += 1;
                    }
                    // raw is asserted as substring-of-input only —
                    // mirrors the Python e2e contract (group 29). Exact
                    // byte-for-byte raw is too brittle (depends on
                    // internal regex alternation order, not on
                    // dedup contract).
                    if !input.contains(g.raw.as_str()) {
                        eprintln!(
                            "[FAIL] {name}: raw {:?} not a substring of input {:?}",
                            g.raw, input
                        );
                        failures += 1;
                    }
                }
            }
        }
        assert_eq!(failures, 0,
            "{failures} extract_price case(s) drift from Python contract"
        );
    }

    #[test]
    fn to_usd_matches_python_fixture() {
        let raw = std::fs::read_to_string(price_fixture_path())
            .expect("read price fixture");
        let v: Value = serde_json::from_str(&raw).expect("parse fixture");

        let fx_pinned = fixture_fx(&v);
        let fx_borrowed: Vec<(&str, f64)> = fx_pinned.iter()
            .map(|(k, v)| (k.as_str(), *v))
            .collect();

        let mut failures = 0usize;
        for case in v["to_usd_cases"].as_array().expect("to_usd_cases array") {
            let name = case["name"].as_str().unwrap_or("?");
            let input = &case["input"];
            let amount = input.get("amount").and_then(|v| v.as_f64());
            let currency = input.get("currency").and_then(|v| v.as_str());
            let exp = &case["expected"];

            let got: Option<f64> = match (amount, currency) {
                (Some(a), Some(c)) => to_usd(a, c, Some(&fx_borrowed)),
                _ => None,
            };

            match (exp, got) {
                (Value::Null, None) => continue,
                (Value::Null, Some(v)) => {
                    eprintln!("[FAIL] {name}: expected None, got {v:?}");
                    failures += 1;
                }
                (_, None) => {
                    eprintln!("[FAIL] {name}: expected {exp}, got None");
                    failures += 1;
                }
                (exp_v, Some(v)) => {
                    let exp_f = exp_v.as_f64().unwrap_or(f64::NAN);
                    if (v - exp_f).abs() > 0.01 {
                        eprintln!(
                            "[FAIL] {name}: got {v}, expected {exp_f} (±0.01)"
                        );
                        failures += 1;
                    }
                }
            }
        }
        assert_eq!(failures, 0,
            "{failures} to_usd case(s) drift from Python contract"
        );
    }

    /// Differential test for top-level `reconcile()` (R1.5-step-3).
    ///
    /// Consumes `tests/fixtures/reconciler/dedup.json`, the same
    /// JSON the Python e2e harness (group 30) reads. Asserts the
    /// same contract surface: count, winner, canonical_url,
    /// sources, prices, and stable winner order.
    #[test]
    fn reconcile_matches_python_fixture() {
        let path = fixture_path("dedup.json");
        let raw = std::fs::read_to_string(&path).expect("read dedup fixture");
        let v: Value = serde_json::from_str(&raw).expect("parse dedup fixture");

        let mut failures: Vec<String> = Vec::new();

        for case in v["cases"].as_array().expect("cases array") {
            let name = case["name"].as_str().unwrap_or("?").to_string();

            // Build (str, f64) FX pairs from the case's fx_rates map.
            let fx_pairs: Vec<(String, f64)> = case
                .get("fx_rates")
                .and_then(Value::as_object)
                .map(|m| {
                    m.iter()
                        .filter_map(|(k, v)| v.as_f64().map(|f| (k.clone(), f)))
                        .collect()
                })
                .unwrap_or_default();
            let fx_borrowed: Vec<(&str, f64)> = fx_pairs
                .iter()
                .map(|(k, v)| (k.as_str(), *v))
                .collect();
            let fx_arg: Option<&[(&str, f64)]> = if fx_pairs.is_empty() {
                None
            } else {
                Some(&fx_borrowed)
            };

            let merge_by_title = case
                .get("merge_by_title")
                .and_then(Value::as_bool)
                .unwrap_or(false);

            let input: Vec<Value> = case["input"]
                .as_array()
                .cloned()
                .unwrap_or_default();
            let out = reconcile(&input, fx_arg, merge_by_title);

            let exp_count = case["expected_count"].as_u64().unwrap_or(0) as usize;
            if out.len() != exp_count {
                failures.push(format!(
                    "[{name}] count: got {}, expected {exp_count}",
                    out.len()
                ));
                continue;
            }

            if let Some(exp_winner) = case.get("expected_winner_source")
                && let Some(first) = out.first()
            {
                let got = first.get("_source_id").and_then(Value::as_str);
                let exp = exp_winner.as_str();
                if got != exp {
                    failures.push(format!(
                        "[{name}] winner: got {got:?}, expected {exp:?}"
                    ));
                }
            }

            if let Some(exp_url) = case.get("expected_canonical_url")
                && let Some(first) = out.first()
            {
                let got = first.get("_canonical_url").and_then(Value::as_str);
                let exp = exp_url.as_str();
                if got != exp {
                    failures.push(format!(
                        "[{name}] canonical_url: got {got:?}, expected {exp:?}"
                    ));
                }
            }

            if let Some(exp_count_v) = case.get("expected_sources_count")
                && let Some(first) = out.first()
            {
                let got_n = first
                    .get("_sources")
                    .and_then(Value::as_array)
                    .map(|a| a.len())
                    .unwrap_or(0);
                let exp_n = exp_count_v.as_u64().unwrap_or(0) as usize;
                if got_n != exp_n {
                    failures.push(format!(
                        "[{name}] sources_count: got {got_n}, expected {exp_n}"
                    ));
                }
            }

            if let Some(exp_ids_v) = case.get("expected_sources_ids")
                && let Some(first) = out.first()
            {
                let mut got_ids: Vec<String> = first
                    .get("_sources")
                    .and_then(Value::as_array)
                    .map(|a| {
                        a.iter()
                            .map(|s| {
                                s.get("_source_id")
                                    .and_then(Value::as_str)
                                    .unwrap_or("")
                                    .to_string()
                            })
                            .collect()
                    })
                    .unwrap_or_default();
                let mut exp_ids: Vec<String> = exp_ids_v
                    .as_array()
                    .map(|a| {
                        a.iter()
                            .map(|s| s.as_str().unwrap_or("").to_string())
                            .collect()
                    })
                    .unwrap_or_default();
                got_ids.sort();
                exp_ids.sort();
                if got_ids != exp_ids {
                    failures.push(format!(
                        "[{name}] sources_ids: got {got_ids:?}, expected {exp_ids:?}"
                    ));
                }
            }

            if let Some(exp_prices_v) = case.get("expected_prices_usd")
                && let Some(exp_arr) = exp_prices_v.as_array()
            {
                let got_prices: Vec<Option<f64>> = out
                    .iter()
                    .map(|it| it.get("price_usd").and_then(Value::as_f64))
                    .collect();
                if got_prices.len() != exp_arr.len() {
                    failures.push(format!(
                        "[{name}] price list length: got {}, expected {}",
                        got_prices.len(),
                        exp_arr.len()
                    ));
                } else {
                    for (i, (g, e)) in got_prices.iter().zip(exp_arr.iter()).enumerate() {
                        match (e, g) {
                            (Value::Null, None) => {}
                            (Value::Null, Some(v)) => failures.push(format!(
                                "[{name}] price[{i}]: got {v:?}, expected None"
                            )),
                            (_, None) => failures.push(format!(
                                "[{name}] price[{i}]: got None, expected {e}"
                            )),
                            (ev, Some(v)) => {
                                let exp_f = ev.as_f64().unwrap_or(f64::NAN);
                                if (v - exp_f).abs() > 0.01 {
                                    failures.push(format!(
                                        "[{name}] price[{i}]: got {v}, expected {exp_f}"
                                    ));
                                }
                            }
                        }
                    }
                }
            }

            if let Some(exp_order_v) = case.get("expected_winner_order")
                && let Some(exp_arr) = exp_order_v.as_array()
            {
                let got_order: Vec<&str> = out
                    .iter()
                    .map(|it| it.get("_source_id").and_then(Value::as_str).unwrap_or(""))
                    .collect();
                let exp_order: Vec<&str> =
                    exp_arr.iter().map(|s| s.as_str().unwrap_or("")).collect();
                if got_order != exp_order {
                    failures.push(format!(
                        "[{name}] winner_order: got {got_order:?}, expected {exp_order:?}"
                    ));
                }
            }
        }

        assert!(
            failures.is_empty(),
            "reconcile() drift from Python contract:\n  - {}",
            failures.join("\n  - ")
        );
    }
}
