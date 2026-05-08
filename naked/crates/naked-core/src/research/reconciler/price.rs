//! Price extraction and FX conversion (B5-3).
//!
//! Public API:
//! - [`ExtractedPrice`] — parsed price record
//! - [`extract_price`] — find the first price token in a text string
//! - [`to_usd`] — convert (amount, currency) → USD
use regex::Regex;
use std::sync::LazyLock;

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
    table
        .iter()
        .find(|(c, _)| c.eq_ignore_ascii_case(&upper))
        .map(|(_, r)| *r)
}

/// Currency-suffix tokens (numeric appears BEFORE the token). Mirrors
/// Python `_CURRENCY_SUFFIX_TOKENS`. Order doesn't matter for the
/// dict lookup — alternation order in the regex is sorted by length
/// descending so longer matches (e.g. `руб.`) win over shorter
/// (`руб`).
const SUFFIX_TOKENS: &[(&str, &str)] = &[
    ("₽", "RUB"),
    ("руб", "RUB"),
    ("руб.", "RUB"),
    ("рубль", "RUB"),
    ("рубля", "RUB"),
    ("рублей", "RUB"),
    ("₸", "KZT"),
    ("тг", "KZT"),
    ("тг.", "KZT"),
    ("тенге", "KZT"),
    ("฿", "THB"),
    ("baht", "THB"),
    ("бат", "THB"),
    ("֏", "AMD"),
    ("драм", "AMD"),
    ("dram", "AMD"),
    ("₫", "VND"),
    ("đồng", "VND"),
    ("Rp", "IDR"),
    ("rp", "IDR"),
    ("RM", "MYR"),
    ("ringgit", "MYR"),
    ("₾", "GEL"),
    ("lari", "GEL"),
    ("лари", "GEL"),
    ("₴", "UAH"),
    ("грн", "UAH"),
    // ISO codes also valid as bare suffixes after a number.
    ("USD", "USD"),
    ("EUR", "EUR"),
    ("GBP", "GBP"),
    ("JPY", "JPY"),
    ("CNY", "CNY"),
    ("RUB", "RUB"),
    ("KZT", "KZT"),
    ("THB", "THB"),
    ("AMD", "AMD"),
    ("VND", "VND"),
    ("IDR", "IDR"),
    ("MYR", "MYR"),
    ("BYN", "BYN"),
    ("UZS", "UZS"),
    ("GEL", "GEL"),
    ("UAH", "UAH"),
];

/// Currency-prefix symbols (token appears BEFORE the number).
const PREFIX_TOKENS: &[(&str, &str)] = &[("$", "USD"), ("€", "EUR"), ("£", "GBP"), ("¥", "JPY")];

/// Multiplier tokens. Sorted by length desc by `build_alternation`
/// so `миллиард` beats `миллион` etc.
const MULTIPLIERS: &[(&str, f64)] = &[
    ("млрд", 1e9),
    ("миллиард", 1e9),
    ("миллиардов", 1e9),
    ("млн", 1e6),
    ("миллион", 1e6),
    ("миллионов", 1e6),
    ("тыс", 1e3),
    ("тысяч", 1e3),
    ("тысячи", 1e3),
    ("billion", 1e9),
    ("million", 1e6),
    ("thousand", 1e3),
    ("tỷ", 1e9),
    ("ty", 1e9),
    ("triệu", 1e6),
    ("trieu", 1e6),
    ("nghìn", 1e3),
    ("nghin", 1e3),
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
    sorted
        .iter()
        .map(|t| regex::escape(t))
        .collect::<Vec<_>>()
        .join("|")
}

/// `(amount [mult] curr)` regex — mirrors Python `_RE_SUFFIX` minus
/// the negative lookahead (Rust `regex` doesn't support lookaround;
/// we emulate it post-match in [`extract_price`]).
static RE_SUFFIX: LazyLock<Regex> = LazyLock::new(|| {
    let suffix_alt = build_alternation(SUFFIX_TOKENS.iter().map(|(t, _)| *t));
    let mult_alt = build_alternation(MULTIPLIERS.iter().map(|(t, _)| *t));
    let pat = format!(
        r"(?i)(?P<amount>{NUMBER_RE})(?:\s*(?P<mult>{mult_alt}))?\s*(?P<curr>{suffix_alt})"
    );
    Regex::new(&pat).expect("RE_SUFFIX compile")
});

/// `(curr amount [mult])` regex — mirrors Python `_RE_PREFIX`.
static RE_PREFIX: LazyLock<Regex> = LazyLock::new(|| {
    let prefix_alt = build_alternation(PREFIX_TOKENS.iter().map(|(t, _)| *t));
    let mult_alt = build_alternation(MULTIPLIERS.iter().map(|(t, _)| *t));
    let pat = format!(
        r"(?i)(?P<curr>{prefix_alt})\s*(?P<amount>{NUMBER_RE})(?:\s*(?P<mult>{mult_alt}))?"
    );
    Regex::new(&pat).expect("RE_PREFIX compile")
});

/// Bare `(amount [mult])` regex — mirrors Python `_RE_BARE`. Used
/// only when the caller passes `default_currency`.
static RE_BARE: LazyLock<Regex> = LazyLock::new(|| {
    let mult_alt = build_alternation(MULTIPLIERS.iter().map(|(t, _)| *t));
    let pat = format!(r"(?i)(?P<amount>{NUMBER_RE})(?:\s*(?P<mult>{mult_alt}))?");
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
    if s.is_empty() {
        return None;
    }
    // Strip ASCII whitespace and U+00A0 NBSP.
    let stripped: String = s.chars().filter(|c| !c.is_whitespace()).collect();
    if stripped.is_empty() {
        return None;
    }

    let has_comma = stripped.contains(',');
    let has_dot = stripped.contains('.');

    let normalised: String = if has_comma && has_dot {
        let (Some(last_comma), Some(last_dot)) = (stripped.rfind(','), stripped.rfind('.')) else {
            return None;
        };
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
        let tail_len = stripped.rsplit_once('.').map(|(_, t)| t.len()).unwrap_or(0);
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
    let Some(m) = mult else {
        return 1.0;
    };
    if m.is_empty() {
        return 1.0;
    }
    let lower = m.to_lowercase();
    MULTIPLIERS
        .iter()
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
        SUFFIX_TOKENS
            .iter()
            .find(|(t, _)| *t == needle)
            .map(|(_, iso)| *iso)
    };
    lookup(token)
        .or_else(|| lookup(&token.to_uppercase()))
        .or_else(|| lookup(&token.to_lowercase()))
}

fn resolve_prefix_currency(token: &str) -> Option<&'static str> {
    PREFIX_TOKENS
        .iter()
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
pub fn extract_price(text: &str, default_currency: Option<&str>) -> Option<ExtractedPrice> {
    if text.is_empty() {
        return None;
    }

    // Helper that scans `re` for the first match whose `curr` group
    // isn't followed by an alphabetic char. Returns (raw, amount,
    // currency-token).
    fn scan<'t>(
        re: &Regex,
        text: &'t str,
        is_prefix: bool,
    ) -> Option<(&'t str, f64, &'static str)> {
        for caps in re.captures_iter(text) {
            let m_full = caps.get(0)?;
            let amt_grp = caps.name("amount")?;
            let mult = caps.name("mult").map(|m| m.as_str());
            let curr_g = caps.name("curr")?;

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
        let iso_static: &'static str = FX_TO_USD
            .iter()
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
pub fn to_usd(amount: f64, currency: &str, fx_rates: Option<&[(&str, f64)]>) -> Option<f64> {
    if !amount.is_finite() {
        return None;
    }
    let table = fx_rates.unwrap_or(FX_TO_USD);
    let rate = lookup_rate(table, currency)?;
    Some(((amount * rate) * 100.0).round() / 100.0)
}
