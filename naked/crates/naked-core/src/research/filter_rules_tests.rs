use super::*;
use std::path::PathBuf;

fn fixture_path() -> PathBuf {
    let mut p = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    p.pop(); // crates
    p.pop(); // naked-core
    p.push("tests");
    p.push("fixtures");
    p.push("filter_rules");
    p.push("expected.json");
    p
}

/// Differential test for the full B4.5-1b contract:
/// `intent → rules_for_intent → score_finding → bucket`.
///
/// Consumes `tests/fixtures/filter_rules/expected.json`, the SAME
/// JSON the Python e2e harness (group 23) reads. For every
/// (case, finding) pair we verify the post-scoring bucket matches
/// the fixture `expected_bucket` field. The Python side runs
/// through `brief_autopilot._filter_rules_for_intent` plus
/// `validate_generic_brief._score_finding` — drift on either half
/// trips both CI runs.
#[test]
fn score_finding_matches_python_fixture() {
    let raw = std::fs::read_to_string(fixture_path()).expect("read filter_rules fixture");
    let v: Value = serde_json::from_str(&raw).expect("parse fixture");

    let shortlist_min = v["shortlist_min"].as_f64().unwrap_or(70.0);
    let maybe_min = v["maybe_min"].as_f64().unwrap_or(35.0);

    let mut failures: Vec<String> = Vec::new();
    let mut checked = 0usize;

    for case in v["cases"].as_array().expect("cases array") {
        let question = case["question"].as_str().unwrap_or("?").to_string();
        let intent = &case["intent"];
        let rules = rules_for_intent(intent);

        for f in case["findings"].as_array().expect("findings array") {
            let label = f["_label"].as_str().unwrap_or("?").to_string();
            let bucket_expected = f["expected_bucket"].as_str().unwrap_or("");

            // Strip private/expected-* keys; keep only real finding fields.
            let mut finding = Value::Object(
                f.as_object()
                    .expect("finding object")
                    .iter()
                    .filter(|(k, _)| !k.starts_with('_') && k.as_str() != "expected_bucket")
                    .map(|(k, v)| (k.clone(), v.clone()))
                    .collect(),
            );

            score_finding(&rules, &mut finding);
            let score = finding.get("_score").and_then(Value::as_f64).unwrap_or(0.0);
            let bucket = if score >= shortlist_min {
                "shortlist"
            } else if score >= maybe_min {
                "maybe"
            } else {
                "rejected"
            };

            checked += 1;
            if bucket != bucket_expected {
                let matched = finding
                    .get("matched_rules")
                    .map(|v| v.to_string())
                    .unwrap_or_default();
                failures.push(format!(
                    "[{question:?} :: {label:?}] score={score} → {bucket}, \
                     expected {bucket_expected}; matched={matched}"
                ));
            }
        }
    }

    assert!(
        failures.is_empty(),
        "filter_rules drift from Python contract ({checked} checked):\n  - {}",
        failures.join("\n  - ")
    );
}

#[test]
fn stem_keyword_russian_short() {
    // Python: len("кв")=2 < 4 → unchanged.
    assert_eq!(stem_keyword("кв"), "кв");
}

#[test]
fn stem_keyword_russian_long() {
    // Python `kw[:-2]` slices on codepoints: len("квартиру")=8 ≥ 7
    // → first 6 chars = "кварти". Note this is **less** than the
    // morpheme "квартир" — the docstring's "stem квартир" is a
    // semantic example, not the literal output. The contract is
    // "produces a prefix that matches all common inflections via
    // substring search", and "кварти" satisfies that for
    // квартира/квартиру/квартиры/квартире all four endings.
    assert_eq!(stem_keyword("квартиру"), "кварти");
}

#[test]
fn stem_keyword_english_plural() {
    // "jobs" len=4, ASCII, ends with 's' → strip 1 → "job".
    assert_eq!(stem_keyword("jobs"), "job");
}

#[test]
fn stem_keyword_english_no_strip() {
    // "java" len=4, ASCII, no trailing 's' → unchanged.
    assert_eq!(stem_keyword("java"), "java");
}

#[test]
fn rules_for_intent_marketplace_has_price_gate() {
    let intent = json!({
        "kind_hint": "marketplace_listing",
        "slots": {"location": "Алматы", "keywords": ["квартиру"]},
    });
    let rules = rules_for_intent(&intent);
    assert!(
        rules
            .iter()
            .any(|r| r.get("name").and_then(Value::as_str) == Some("price present (marketplace)"))
    );
}

#[test]
fn rules_for_intent_jobs_has_no_price_gate() {
    let intent = json!({
        "kind_hint": "jobs",
        "slots": {"location": "Yerevan", "keywords": ["java"]},
    });
    let rules = rules_for_intent(&intent);
    assert!(
        !rules
            .iter()
            .any(|r| r.get("name").and_then(Value::as_str) == Some("price present (marketplace)"))
    );
}

#[test]
fn apply_rule_range_number_mutates_extracted() {
    let rule = json!({
        "name": "price band",
        "type": "range_number",
        "field": "price",
        "regex": "(\\d+)",
        "min": 100, "max": 1000,
        "score_in_range": 30, "score_out_range": -10,
    });
    let mut finding = json!({"price": "Sale 500 USD"});
    let (delta, matched) = apply_rule(&rule, &mut finding);
    assert!(matched);
    assert!((delta - 30.0).abs() < 1e-9);
    assert_eq!(finding["_extracted"]["price band"].as_f64(), Some(500.0));
}

use proptest::prelude::*;

proptest! {
    #[test]
    fn apply_rule_never_panics_contains(
        field_val in "\\PC{0,100}",
        pattern in "\\PC{0,50}",
    ) {
        let rule = serde_json::json!({
            "type": "contains",
            "field": "title",
            "value": pattern,
            "score": 1.0,
        });
        let mut finding = serde_json::json!({
            "title": field_val,
            "url": "https://example.com",
        });
        let (score, fired) = apply_rule(&rule, &mut finding);
        prop_assert!(score >= 0.0);
        let _ = fired;
    }

    #[test]
    fn apply_rule_never_panics_range(
        price in 0.0f64..1_000_000.0,
        min in 0.0f64..500_000.0,
        max in 500_000.0f64..1_000_000.0,
    ) {
        let rule = serde_json::json!({
            "type": "range",
            "field": "price_thb",
            "min": min,
            "max": max,
            "score": -1.0,
        });
        let mut finding = serde_json::json!({
            "price_thb": price,
            "url": "https://example.com",
        });
        let (score, _) = apply_rule(&rule, &mut finding);
        prop_assert!(score <= 0.0 || score >= 0.0); // just no panic
    }

    #[test]
    fn rules_for_intent_never_panics(topic in "\\PC{0,100}") {
        let intent = serde_json::json!({
            "topic": topic,
            "sources": [],
        });
        let rules = rules_for_intent(&intent);
        let _ = rules; // just ensure no panic
    }
}
