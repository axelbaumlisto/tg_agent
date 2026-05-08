//! B4.5-1b filter-rules engine + intent-aware rule builder.
//!
//! Bit-exact Rust port of the **pure-CPU** path of:
//!
//!   * `scripts/validate_generic_brief.py::_apply_rule / _score_finding`
//!     (5 rule types over a single finding)
//!   * `scripts/brief_autopilot.py::_stem_keyword / _NOISE_TOKENS /
//!     _filter_rules_for_intent` (intent → rule list)
//!
//! Together these two pieces are the **scoring contract** that decides
//! shortlist / maybe / rejected for every finding the autopilot
//! pipeline emits. Pre-B4.5 the max achievable was 55 against
//! `shortlist_min=70` (structurally impossible); the rules below are
//! tuned so any signal pair (keyword × location, in any text field)
//! clears 70.
//!
//! ## Why one module
//!
//! The two halves live in **different** Python files but share one
//! contract — group 23 (`23_filter_rules_offline.sh`) tests them
//! end-to-end (`intent → rules → score → bucket`). Putting both
//! halves in one Rust module lets the differential test consume the
//! same fixture (`tests/fixtures/filter_rules/expected.json`)
//! verbatim, which is the cleanest cross-language guarantee.
//!
//! ## CLI / I/O scope
//!
//! Out of scope here: `validate_generic_brief.py`'s argparse / JSONL
//! writers / stdout summary. That's pure plumbing (no contract
//! surface) and lives in a separate port-step (next session). The
//! engine in this file is what `naked-cli` will eventually wrap.
//!
//! ## Contract pinning
//!
//! * Python: e2e group 23 (`23_filter_rules_offline.sh`)
//! * Rust:   e2e group 35 (`35_filter_rules_rust_differential_offline.sh`)
//!
//! Both consume `tests/fixtures/filter_rules/expected.json`. Drift
//! on either side trips both CI runs.

use regex::Regex;
use serde_json::{Map, Value, json};

// ── B4.5-1b: noise-token list (shared across Python and Rust) ────────

/// Strings that should knock a card out of `maybe`/`shortlist` when
/// they appear in the title. Mirrors `brief_autopilot._NOISE_TOKENS`
/// byte-for-byte — re-ordering or pruning this list will silently
/// drop the negative-score rule and "Sign Up | LinkedIn" cards will
/// resurface in `maybe.jsonl` (regression group 23 catches it).
pub const NOISE_TOKENS: &[&str] = &[
    "sign up",
    "sign in",
    "log in",
    "login",
    "register",
    "subscribe",
    "newsletter",
    "cookie",
    "cookies",
    "privacy policy",
    "terms of",
    "terms and",
    "404",
    "page not found",
    "all rights reserved",
    "skip to main",
    "skip to content",
];

// ── B4.5-1: keyword stemming (Russian + English plural-s) ────────────

/// Reduce a keyword to a prefix-stem so substring-search hits all
/// common inflections.
///
/// **Why stemming**
/// `contains_any` does plain `value in text` matching. For Russian
/// that misses every inflected form: "квартиры" (genitive) won't
/// match "квартира" (nominative) because the strings differ in the
/// final char. Karaganda's 157 raw findings rescored with the
/// un-stemmed lexicon hit shortlist on only 2.5 % of cards — the
/// same listings climb to 26 % once the stem "квартир" matches all
/// forms.
///
/// English handling is intentionally minimal: only the `-s` plural
/// is stripped, because verb inflections (`-ed`/`-ing`) are rare in
/// marketplace titles and stripping them has produced false
/// positives in development (`backend → back`).
///
/// Bit-exact mirror of `brief_autopilot._stem_keyword`.
pub fn stem_keyword(kw: &str) -> String {
    let len = kw.chars().count();
    if len < 4 {
        return kw.to_string();
    }
    let lower = kw.to_lowercase();
    if has_cyrillic_lower(&lower) {
        // Python uses `kw[:-2]` / `kw[:-1]` (slice on Python str =
        // codepoints). Rust strs are bytes, so we strip by char count.
        if len >= 7 {
            return chars_take(kw, len - 2);
        }
        if len >= 5 {
            return chars_take(kw, len - 1);
        }
        return kw.to_string();
    }
    if has_ascii_alpha_lower(&lower) && lower.ends_with('s') && len > 3 {
        return chars_take(kw, len - 1);
    }
    kw.to_string()
}

/// `True` iff any char in `s` (already lowercased) is in
/// `[а-яё]` — the Python regex `[а-яё]` Unicode range.
fn has_cyrillic_lower(s: &str) -> bool {
    s.chars().any(|c| matches!(c, 'а'..='я' | 'ё'))
}

/// `True` iff any char in `s` is an ASCII lowercase letter — Python's
/// `[a-z]` after `.lower()`.
fn has_ascii_alpha_lower(s: &str) -> bool {
    s.chars().any(|c| c.is_ascii_lowercase())
}

/// Take the first `n` chars (codepoints) from `s` and return them as
/// an owned String. Python's slice semantics on `str`.
fn chars_take(s: &str, n: usize) -> String {
    s.chars().take(n).collect()
}

// ── B4.5-1b: intent → rules ──────────────────────────────────────────

/// Generate scoring rules tailored to the user's intent.
///
/// Scoring math (post-B4.5):
///
/// | delta | trigger                                          |
/// |------:|--------------------------------------------------|
/// |  +20  | title length ≥ 5 (length gate, not relevance)    |
/// |  +20  | link well-formed (`^https?://`)                  |
/// |  +25  | ≥1 stemmed intent keyword in title (primary)     |
/// |  +15  | … in description (secondary)                     |
/// |  +15  | … in snippet (secondary)                         |
/// |  +25  | intent.location in title (primary)               |
/// |  +15  | … in description                                 |
/// |  +15  | … in snippet                                     |
/// |  +15  | marketplace_listing: `price` field non-empty     |
/// |  -30  | noise / nav copy in title (sign up, cookies, …)  |
///
/// Bit-exact mirror of `brief_autopilot._filter_rules_for_intent`.
pub fn rules_for_intent(intent: &Value) -> Vec<Value> {
    let mut rules: Vec<Value> = vec![
        json!({
            "name": "title not empty", "type": "regex_match",
            "field": "title", "pattern": ".{5,}", "score": 20
        }),
        json!({
            "name": "link present", "type": "regex_match",
            "field": "link", "pattern": "^https?://", "score": 20
        }),
    ];

    let slots = intent.get("slots").and_then(Value::as_object);

    // ── keyword stems ──────────────────────────────────────────────
    let keywords: Vec<String> = slots
        .and_then(|s| s.get("keywords"))
        .and_then(Value::as_array)
        .map(|a| {
            a.iter()
                .filter_map(|v| v.as_str())
                .filter(|k| k.chars().count() >= 3)
                .take(6)
                .map(|s| s.to_string())
                .collect()
        })
        .unwrap_or_default();

    if !keywords.is_empty() {
        let mut stems: Vec<String> = Vec::new();
        for kw in &keywords {
            let s = stem_keyword(kw);
            if !s.is_empty() && !stems.contains(&s) {
                stems.push(s);
            }
        }
        if !stems.is_empty() {
            rules.push(json!({
                "name": "intent keyword in title",
                "type": "contains_any",
                "field": "title",
                "values": stems,
                "score": 25,
            }));
            rules.push(json!({
                "name": "intent keyword in description",
                "type": "contains_any",
                "field": "description",
                "values": stems,
                "score": 15,
            }));
            rules.push(json!({
                "name": "intent keyword in snippet",
                "type": "contains_any",
                "field": "snippet",
                "values": stems,
                "score": 15,
            }));
        }
    }

    // ── location ───────────────────────────────────────────────────
    if let Some(loc) = slots
        .and_then(|s| s.get("location"))
        .and_then(Value::as_str)
        && loc.chars().count() >= 3
    {
        rules.push(json!({
            "name": "intent location in title",
            "type": "contains_any",
            "field": "title",
            "values": [loc],
            "score": 25,
        }));
        rules.push(json!({
            "name": "intent location in description",
            "type": "contains_any",
            "field": "description",
            "values": [loc],
            "score": 15,
        }));
        rules.push(json!({
            "name": "intent location in snippet",
            "type": "contains_any",
            "field": "snippet",
            "values": [loc],
            "score": 15,
        }));
    }

    // ── marketplace-only price gate ────────────────────────────────
    if intent.get("kind_hint").and_then(Value::as_str) == Some("marketplace_listing") {
        rules.push(json!({
            "name": "price present (marketplace)",
            "type": "regex_match",
            "field": "price",
            "pattern": ".+",
            "score": 15,
        }));
    }

    // ── noise penalty (always last) ────────────────────────────────
    rules.push(json!({
        "name": "noise / nav copy in title",
        "type": "contains_any",
        "field": "title",
        "values": NOISE_TOKENS,
        "score": -30,
    }));

    rules
}

// ── B4.5-1b: scoring engine (apply_rule + score_finding) ─────────────

/// Read the `score` field of a rule as f64. Defaults to 0.0 for
/// missing / non-numeric entries (Python `float(rule.get("score", 0))`).
fn rule_score(rule: &Value, key: &str) -> f64 {
    rule.get(key).and_then(Value::as_f64).unwrap_or(0.0)
}

/// Apply one rule to one finding. Returns `(delta, matched)`.
///
/// `matched=False` means the rule did not fire — `delta` is `0.0` and
/// the rule's name is **NOT** appended to `matched_rules` by the
/// caller. Mirrors `validate_generic_brief._apply_rule` exactly,
/// including the side-effect on `range_number` rules (which mutate
/// the finding by adding `_extracted[name] = num`).
pub fn apply_rule(rule: &Value, finding: &mut Value) -> (f64, bool) {
    let rtype = match rule.get("type").and_then(Value::as_str) {
        Some(t) => t,
        None => return (0.0, false),
    };
    let field_name = match rule.get("field").and_then(Value::as_str) {
        Some(f) => f,
        None => return (0.0, false),
    };

    let field_value: Option<Value> = finding.get(field_name).cloned();

    // Python: `if field_value is None: ...` covers Python None and the
    // case where the key isn't present at all. JSON nulls behave the
    // same — both end up as `Some(Value::Null)` here, which we treat
    // as "field missing".
    let field_present = matches!(&field_value, Some(v) if !v.is_null());
    if !field_present {
        // not_contains is trivially True on missing field → fires.
        if rtype == "not_contains" {
            return (rule_score(rule, "score"), true);
        }
        return (0.0, false);
    }

    let Some(raw_field) = field_value.as_ref() else {
        return (0.0, false);
    };
    let raw_str: String = match raw_field {
        Value::String(s) => s.clone(),
        // Python `str(field_value)` for any non-string value. Numbers,
        // bools, etc. get stringified for matching purposes.
        Value::Number(n) => n.to_string(),
        Value::Bool(b) => {
            if *b {
                "True".into()
            } else {
                "False".into()
            }
        }
        _ => raw_field.to_string(),
    };
    let text = raw_str.to_lowercase();

    match rtype {
        "contains_any" => {
            let values = rule_values_lower(rule);
            if values.iter().any(|v| text.contains(v.as_str())) {
                (rule_score(rule, "score"), true)
            } else {
                (0.0, false)
            }
        }
        "contains_all" => {
            let values = rule_values_lower(rule);
            if !values.is_empty() && values.iter().all(|v| text.contains(v.as_str())) {
                (rule_score(rule, "score"), true)
            } else {
                (0.0, false)
            }
        }
        "not_contains" => {
            let values = rule_values_lower(rule);
            if values.iter().all(|v| !text.contains(v.as_str())) {
                (rule_score(rule, "score"), true)
            } else {
                (0.0, false)
            }
        }
        "regex_match" => {
            let pattern = rule.get("pattern").and_then(Value::as_str).unwrap_or("");
            if pattern.is_empty() {
                return (0.0, false);
            }
            // Python: `re.search(pattern, str(field_value), flags=re.IGNORECASE)`.
            // Rust regex uses `(?i)` inline flag for case-insensitivity.
            // Compile failures fall back to non-match — Python would
            // raise; a brief with a malformed regex is operator error,
            // not a contract surface. We log via clippy's let-else so
            // it's obvious if we ever decide to surface the panic.
            let Ok(re) = Regex::new(&format!("(?i){pattern}")) else {
                return (0.0, false);
            };
            if re.is_match(&raw_str) {
                (rule_score(rule, "score"), true)
            } else {
                (0.0, false)
            }
        }
        "range_number" => {
            let pattern_default = "([\\d.,]+)";
            let pattern = rule
                .get("regex")
                .and_then(Value::as_str)
                .unwrap_or(pattern_default);
            let Ok(re) = Regex::new(pattern) else {
                return (0.0, false);
            };
            let Some(caps) = re.captures(&raw_str) else {
                return (0.0, false);
            };
            // Python: `m.group(1)`. We require an explicit capture
            // group — otherwise no number to compare against.
            let Some(grp) = caps.get(1) else {
                return (0.0, false);
            };
            // Strip thousands separators (commas) and parse.
            let cleaned = grp.as_str().replace(',', "");
            let Ok(num) = cleaned.parse::<f64>() else {
                return (0.0, false);
            };

            // Side-effect: store extracted number on the finding.
            // Python: `finding.setdefault("_extracted", {})[name] = num`.
            let name = rule
                .get("name")
                .and_then(Value::as_str)
                .unwrap_or("range_number")
                .to_string();
            if let Some(obj) = finding.as_object_mut() {
                let extracted = obj
                    .entry("_extracted")
                    .or_insert_with(|| Value::Object(Map::new()));
                if let Some(em) = extracted.as_object_mut() {
                    em.insert(name, json!(num));
                }
            }

            let mut in_range = true;
            if let Some(min_v) = rule.get("min").and_then(Value::as_f64)
                && num < min_v
            {
                in_range = false;
            }
            if let Some(max_v) = rule.get("max").and_then(Value::as_f64)
                && num > max_v
            {
                in_range = false;
            }

            let score_key = if in_range {
                "score_in_range"
            } else {
                "score_out_range"
            };
            let delta = rule_score(rule, score_key);

            // Python: `return delta, in_range or rule.get("score_out_range") is not None`.
            // Both rule branches register as "matched" when score_out_range is
            // explicitly defined — preserves matched_rules accuracy.
            let matched = in_range || rule.get("score_out_range").is_some();
            (delta, matched)
        }
        _ => (0.0, false),
    }
}

/// Lower-cased copy of `rule.values` (or empty if missing).
fn rule_values_lower(rule: &Value) -> Vec<String> {
    rule.get("values")
        .and_then(Value::as_array)
        .map(|a| {
            a.iter()
                .filter_map(|v| v.as_str())
                .map(|s| s.to_lowercase())
                .collect()
        })
        .unwrap_or_default()
}

/// Score one finding against a rule list. Mutates the finding in
/// place, adding `_score` (rounded to 1 decimal — Python
/// `round(score, 1)`) and `matched_rules` (list of `"<name> (±N)"`
/// strings). Mirrors `validate_generic_brief._score_finding`.
pub fn score_finding(rules: &[Value], finding: &mut Value) {
    let mut score = 0.0_f64;
    let mut matched: Vec<String> = Vec::new();
    for r in rules {
        let (delta, did_match) = apply_rule(r, finding);
        if did_match {
            score += delta;
            let default_name = format!(
                "{}/{}",
                r.get("type").and_then(Value::as_str).unwrap_or(""),
                r.get("field").and_then(Value::as_str).unwrap_or(""),
            );
            let name = r
                .get("name")
                .and_then(Value::as_str)
                .filter(|s| !s.is_empty())
                .map(|s| s.to_string())
                .unwrap_or(default_name);
            // Python: `f"{name} ({'+' if delta >= 0 else ''}{int(delta)})"`.
            // `int(delta)` is truncation toward zero; we mirror with
            // `as i64` (same behaviour for the values the rule set
            // produces — all integers).
            let sign = if delta >= 0.0 { "+" } else { "" };
            matched.push(format!("{name} ({sign}{})", delta as i64));
        }
    }
    if let Some(obj) = finding.as_object_mut() {
        // Python: `round(score, 1)`. f64 has no built-in 1-decimal
        // round; we implement explicitly. `(score*10).round()/10`
        // matches Python's banker's-rounding for the integer rule
        // deltas in this engine (no fractional cents to disagree on).
        let rounded = (score * 10.0).round() / 10.0;
        obj.insert("_score".to_string(), json!(rounded));
        obj.insert(
            "matched_rules".to_string(),
            Value::Array(matched.into_iter().map(Value::String).collect()),
        );
    }
}

// ── tests ────────────────────────────────────────────────────────────

#[cfg(test)]
#[path = "filter_rules_tests.rs"]
mod tests;
