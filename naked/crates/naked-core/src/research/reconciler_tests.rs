use super::price::to_usd; // pub(crate) at price.rs; reconciler/mod.rs no longer re-exports
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

fn url_title_fixture_path() -> PathBuf {
    fixture_path("url_title.json")
}
fn price_fixture_path() -> PathBuf {
    fixture_path("price.json")
}

#[test]
fn canonicalize_url_matches_python_fixture() {
    let raw = std::fs::read_to_string(url_title_fixture_path()).expect("read fixture");
    let v: Value = serde_json::from_str(&raw).expect("parse fixture");

    let mut failures: Vec<String> = Vec::new();
    for case in v["url_cases"].as_array().expect("url_cases array") {
        let name = case["name"].as_str().unwrap_or("?");
        let input = case["input"].as_str().unwrap_or("");
        let exp = case["expected"].as_str().unwrap_or("");
        let got = canonicalize_url(input);
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
    let raw = std::fs::read_to_string(url_title_fixture_path()).expect("read fixture");
    let v: Value = serde_json::from_str(&raw).expect("parse fixture");

    let mut failures = 0usize;
    for case in v["title_cases"].as_array().expect("title_cases array") {
        let name = case["name"].as_str().unwrap_or("?");
        let exp = case["expected"].as_str().unwrap_or("");

        // Python's contract: non-string input → empty string.
        // Rust signature is `&str` already so a JSON `null` is
        // explicitly handled as "skip non-string check is moot
        // — pass empty &str instead".
        let input_owned: String = match &case["input"] {
            Value::String(s) => s.clone(),
            Value::Null => String::new(),
            other => {
                panic!("[{name}] non-string non-null input not supported in Rust impl: {other:?}")
            }
        };
        let got = normalize_title(&input_owned);
        if got != exp {
            eprintln!(
                "[FAIL] {name}\n  input    = {input_owned:?}\n  got      = {got:?}\n  expected = {exp:?}"
            );
            failures += 1;
        }
    }
    assert_eq!(
        failures, 0,
        "{failures} title-normalisation case(s) drift from Python contract"
    );
}

// ── R1.5-step-2: extract_price + to_usd ──────────────────────────

/// Build the FX table from the fixture's `_fx_pin` block — a
/// pinned snapshot is the only way the differential test stays
/// stable across rate refreshes. Keeps the test independent of
/// any drift in the in-code [`FX_TO_USD`] table.
fn fixture_fx(v: &Value) -> Vec<(String, f64)> {
    v["_fx_pin"]
        .as_object()
        .expect("_fx_pin object")
        .iter()
        .filter(|(k, _)| !k.starts_with('_'))
        .filter_map(|(k, v)| v.as_f64().map(|f| (k.clone(), f)))
        .collect()
}

#[test]
fn extract_price_matches_python_fixture() {
    let raw = std::fs::read_to_string(price_fixture_path()).expect("read price fixture");
    let v: Value = serde_json::from_str(&raw).expect("parse fixture");

    let mut failures = 0usize;
    for case in v["extract_cases"].as_array().expect("extract_cases array") {
        let name = case["name"].as_str().unwrap_or("?");
        let input = case["input"].as_str().unwrap_or("");
        let default_currency = case.get("default_currency").and_then(|v| v.as_str());
        let got = extract_price(input, default_currency);
        let exp = &case["expected"];

        // Three branches: expected null, expected dict, mismatch.
        match (exp, &got) {
            (Value::Null, None) => continue,
            (Value::Null, Some(g)) => {
                eprintln!("[FAIL] {name}: expected None, got {g:?}\n  input = {input:?}");
                failures += 1;
            }
            (_, None) => {
                eprintln!("[FAIL] {name}: expected {exp}, got None\n  input = {input:?}");
                failures += 1;
            }
            (exp_obj, Some(g)) => {
                let exp_amount = exp_obj["amount"].as_f64().unwrap_or(f64::NAN);
                let exp_curr = exp_obj["currency"].as_str().unwrap_or("");
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
    assert_eq!(
        failures, 0,
        "{failures} extract_price case(s) drift from Python contract"
    );
}

#[test]
fn to_usd_matches_python_fixture() {
    let raw = std::fs::read_to_string(price_fixture_path()).expect("read price fixture");
    let v: Value = serde_json::from_str(&raw).expect("parse fixture");

    let fx_pinned = fixture_fx(&v);
    let fx_borrowed: Vec<(&str, f64)> = fx_pinned.iter().map(|(k, v)| (k.as_str(), *v)).collect();

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
                    eprintln!("[FAIL] {name}: got {v}, expected {exp_f} (±0.01)");
                    failures += 1;
                }
            }
        }
    }
    assert_eq!(
        failures, 0,
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
        let fx_borrowed: Vec<(&str, f64)> =
            fx_pairs.iter().map(|(k, v)| (k.as_str(), *v)).collect();
        let fx_arg: Option<&[(&str, f64)]> = if fx_pairs.is_empty() {
            None
        } else {
            Some(&fx_borrowed)
        };

        let merge_by_title = case
            .get("merge_by_title")
            .and_then(Value::as_bool)
            .unwrap_or(false);

        let input: Vec<Value> = case["input"].as_array().cloned().unwrap_or_default();
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
                failures.push(format!("[{name}] winner: got {got:?}, expected {exp:?}"));
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
                        (Value::Null, Some(v)) => {
                            failures.push(format!("[{name}] price[{i}]: got {v:?}, expected None"))
                        }
                        (_, None) => {
                            failures.push(format!("[{name}] price[{i}]: got None, expected {e}"))
                        }
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
            let exp_order: Vec<&str> = exp_arr.iter().map(|s| s.as_str().unwrap_or("")).collect();
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

mod prop_tests {
    use super::*;
    use proptest::prelude::*;

    proptest! {
        #[test]
        fn extract_price_never_panics(s in "\\PC{0,200}") {
            let _ = extract_price(&s, None);
        }

        #[test]
        fn extract_price_never_panics_with_currency(s in "\\PC{0,200}", curr in "(THB|USD|VND|EUR)") {
            let _ = extract_price(&s, Some(&curr));
        }

        #[test]
        fn extract_price_valid_result(s in "[0-9,.\\s]{1,20}(THB|฿|\\$|USD|VND|đ|EUR|€)?") {
            if let Some(p) = extract_price(&s, None) {
                prop_assert!(p.amount >= 0.0, "price must be non-negative: {}", p.amount);
                prop_assert!(!p.currency.is_empty(), "currency must not be empty");
            }
        }

        #[test]
        fn reconcile_never_panics(n in 0usize..10) {
            let findings: Vec<serde_json::Value> = (0..n)
                .map(|i| serde_json::json!({
                    "url": format!("https://example.com/{i}"),
                    "title": format!("Item {i}"),
                    "price": format!("{} THB", i * 1000 + 100),
                    "dedup_hash": format!("hash{i}"),
                }))
                .collect();
            let result = reconcile(&findings, None, false);
            // Must not panic; output ≤ input
            assert!(result.len() <= findings.len());
        }

        #[test]
        fn reconcile_merge_deduplicates(n in 2usize..8) {
            // All items with same URL should merge to 1
            let findings: Vec<serde_json::Value> = (0..n)
                .map(|i| serde_json::json!({
                    "url": "https://example.com/same",
                    "title": format!("Item {i}"),
                    "dedup_hash": format!("hash{i}"),
                }))
                .collect();
            let result = reconcile(&findings, None, true);
            prop_assert!(result.len() <= 1, "same URL should merge: got {}", result.len());
        }
    }
}
