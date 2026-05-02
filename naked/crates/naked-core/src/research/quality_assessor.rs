//! B4.6 quality_assessor — `honest-unknown` detector for the research pipeline.
//!
//! Bit-exact Rust port of `scripts/quality_assessor.py`. Runs the same
//! verdict ladder over `_score`/`_relevance`/`_relevance_reason` triples
//! and returns the same `{ok, verdict, n_*, mean/top, dominant_reject_reasons,
//! advice}` shape the Python e2e (group 24) and the autopilot/MCP
//! consume.
//!
//! ## Why two implementations
//!
//! The Python module ships the active pipeline today. The Rust port
//! sits next to it as a *contract surface* — same JSON fixture
//! (`tests/fixtures/quality_assessor/expected.json`) consumed by both:
//!
//! * Python: e2e group 24 (`24_quality_assessor_offline.sh`)
//! * Rust:   `assess_quality_matches_python_fixture` in this file
//!   (group 33, alongside the reconciler differential tests)
//!
//! Drift on either side trips both CI runs.
//!
//! ## Verdict ladder (mirrors Python)
//!
//! 1. `empty` — no findings.
//! 2. `green` — `shortlist_rate >= green_shortlist_rate` OR
//!    (`top_relevance >= green_top_relevance` AND
//!    `mean_relevance >= green_mean_relevance`).
//! 3. `yellow` — `shortlist_rate >= yellow_shortlist_rate` OR
//!    `top_relevance >= yellow_top_relevance`.
//! 4. `red` — everything else (the "honest unknown" branch).
//!
//! ## Determinism
//!
//! `dominant_reject_reasons` ties broken by first-occurrence — Python
//! uses `OrderedDict` + `sorted(...key=-count)` (stable sort). The
//! Rust port uses a `Vec<(String,usize)>` (linear lookup; N is tiny —
//! typically <30 distinct reasons) + `sort_by(|a,b| b.1.cmp(&a.1))`
//! which is also stable, so insertion order preserves identically.
//!
//! ## Pure / cheap
//!
//! `O(n)` over findings, no I/O, no globals, no LLM. Safe to call on
//! every brief and every CI run.

use serde_json::{Value, json};

// ── public types ─────────────────────────────────────────────────────

/// Threshold knobs. Defaults match Python `DEFAULT_THRESHOLDS` and the
/// pinned values in the e2e fixture's `_thresholds` block.
///
/// Calibrated on the B4.5 Tier-3 results (`shortlist_min=70`,
/// `green=30%`, `yellow=10%`); change with care — the Tier-3
/// regression baseline (group 25) gates against verdict drift.
#[derive(Debug, Clone, PartialEq)]
pub struct Thresholds {
    pub shortlist_min: f64,
    pub maybe_min: f64,
    pub green_shortlist_rate: f64,
    pub green_top_relevance: i64,
    pub green_mean_relevance: f64,
    pub yellow_shortlist_rate: f64,
    pub yellow_top_relevance: i64,
    pub low_relevance_max: i64,
    pub thin_sample_max: usize,
    pub min_graded_for_verdict: usize,
}

impl Default for Thresholds {
    fn default() -> Self {
        Self {
            shortlist_min: 70.0,
            maybe_min: 35.0,
            green_shortlist_rate: 0.30,
            green_top_relevance: 80,
            green_mean_relevance: 50.0,
            yellow_shortlist_rate: 0.10,
            yellow_top_relevance: 50,
            low_relevance_max: 35,
            thin_sample_max: 5,
            min_graded_for_verdict: 3,
        }
    }
}

impl Thresholds {
    /// Build a `Thresholds` from a JSON object, falling back to
    /// defaults for any missing key. Mirrors Python's
    /// `{**DEFAULT_THRESHOLDS, **(thresholds or {})}` merge — a
    /// caller can pass a partial dict and the rest is inherited.
    pub fn from_json(v: &Value) -> Self {
        let d = Self::default();
        let obj = match v.as_object() {
            Some(o) => o,
            None => return d,
        };
        let f = |k: &str, fallback: f64| -> f64 {
            obj.get(k).and_then(Value::as_f64).unwrap_or(fallback)
        };
        let i = |k: &str, fallback: i64| -> i64 {
            obj.get(k).and_then(Value::as_i64).unwrap_or(fallback)
        };
        let u = |k: &str, fallback: usize| -> usize {
            obj.get(k)
                .and_then(Value::as_u64)
                .map(|v| v as usize)
                .unwrap_or(fallback)
        };
        Self {
            shortlist_min: f("shortlist_min", d.shortlist_min),
            maybe_min: f("maybe_min", d.maybe_min),
            green_shortlist_rate: f("green_shortlist_rate", d.green_shortlist_rate),
            green_top_relevance: i("green_top_relevance", d.green_top_relevance),
            green_mean_relevance: f("green_mean_relevance", d.green_mean_relevance),
            yellow_shortlist_rate: f("yellow_shortlist_rate", d.yellow_shortlist_rate),
            yellow_top_relevance: i("yellow_top_relevance", d.yellow_top_relevance),
            low_relevance_max: i("low_relevance_max", d.low_relevance_max),
            thin_sample_max: u("thin_sample_max", d.thin_sample_max),
            min_graded_for_verdict: u("min_graded_for_verdict", d.min_graded_for_verdict),
        }
    }
}

/// One row of the `dominant_reject_reasons` list.
#[derive(Debug, Clone, PartialEq)]
pub struct ReasonCount {
    pub reason: String,
    pub count: usize,
}

/// Output of [`assess_quality`]. Field order matches Python's dict
/// keys and the JSON fixture's `_verdict_contract`.
#[derive(Debug, Clone, PartialEq)]
pub struct Verdict {
    pub ok: bool,
    pub verdict: String,
    pub n_total: usize,
    pub n_shortlist: usize,
    pub n_maybe: usize,
    pub n_rejected: usize,
    pub n_graded: usize,
    pub shortlist_rate: f64,
    pub mean_relevance: Option<f64>,
    pub top_relevance: Option<i64>,
    pub dominant_reject_reasons: Vec<ReasonCount>,
    pub advice: String,
}

impl Verdict {
    /// Serialize to a `serde_json::Value` matching Python's dict
    /// output byte-for-byte. Used by the differential test (group 33)
    /// and any caller that wants to emit the verdict as JSON.
    pub fn to_json(&self) -> Value {
        json!({
            "ok": self.ok,
            "verdict": self.verdict,
            "n_total": self.n_total,
            "n_shortlist": self.n_shortlist,
            "n_maybe": self.n_maybe,
            "n_rejected": self.n_rejected,
            "n_graded": self.n_graded,
            "shortlist_rate": self.shortlist_rate,
            "mean_relevance": match self.mean_relevance {
                Some(v) => json!(v),
                None => Value::Null,
            },
            "top_relevance": match self.top_relevance {
                Some(v) => json!(v),
                None => Value::Null,
            },
            "dominant_reject_reasons": self.dominant_reject_reasons
                .iter()
                .map(|r| json!({"reason": r.reason, "count": r.count}))
                .collect::<Vec<_>>(),
            "advice": self.advice,
        })
    }
}

// ── public api ───────────────────────────────────────────────────────

/// Compute verdict + advice for a batch of (already-scored) findings.
///
/// `findings` is a slice of `serde_json::Value` objects each carrying
/// `_score` (heuristic, after any LLM-relevance bonus) and optionally
/// `_relevance` / `_relevance_reason`. Cards without `_relevance` are
/// still counted in shortlist/maybe/rejected buckets but contribute
/// nothing to mean/top/dominant_reject_reasons.
///
/// Pass `None` for `thresholds` to use the defaults (matching
/// Python's `DEFAULT_THRESHOLDS`); passing `Some(Thresholds::default())`
/// is equivalent.
pub fn assess_quality(findings: &[Value], thresholds: Option<&Thresholds>) -> Verdict {
    let default;
    let th = match thresholds {
        Some(t) => t,
        None => {
            default = Thresholds::default();
            &default
        }
    };

    let n_total = findings.len();
    if n_total == 0 {
        return Verdict {
            ok: false,
            verdict: "empty".to_string(),
            n_total: 0,
            n_shortlist: 0,
            n_maybe: 0,
            n_rejected: 0,
            n_graded: 0,
            shortlist_rate: 0.0,
            mean_relevance: None,
            top_relevance: None,
            dominant_reject_reasons: Vec::new(),
            advice: "Findings пустой. Скорее всего scraper'ы упали или \
                     discovery не нашёл ни одного источника. Проверьте \
                     логи playwright_tick.sh и discover_sources."
                .to_string(),
        };
    }

    let (n_shortlist, n_maybe, n_rejected) =
        bucket_counts(findings, th.shortlist_min, th.maybe_min);
    let shortlist_rate = n_shortlist as f64 / n_total as f64;

    // Graded subset: cards where `_relevance` is a real number
    // (Python guards `not isinstance(_, bool)`; serde_json Numbers
    // can never hold a bool, so the check is implicit here).
    let graded: Vec<&Value> = findings
        .iter()
        .filter(|f| f.get("_relevance").and_then(Value::as_f64).is_some())
        .collect();
    let n_graded = graded.len();

    let (mean_relevance, top_relevance, dominant_reasons) = if n_graded >= th.min_graded_for_verdict
    {
        let rels: Vec<f64> = graded
            .iter()
            .map(|f| f.get("_relevance").and_then(Value::as_f64).unwrap_or(0.0))
            .collect();
        let sum: f64 = rels.iter().sum();
        let mean = sum / rels.len() as f64;
        // Python: `int(max(rels))` — floor on positive values; we
        // match by casting f64 → i64 (truncation toward zero,
        // identical to `int()` on non-negative reals).
        let top_f = rels.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let top = top_f as i64;
        let reasons = dominant_reasons(&graded, th.low_relevance_max);
        (Some(mean), Some(top), reasons)
    } else {
        (None, None, Vec::new())
    };

    let verdict = decide_verdict(shortlist_rate, mean_relevance, top_relevance, th);

    let advice = advice_for(
        &verdict,
        n_total,
        shortlist_rate,
        top_relevance,
        &dominant_reasons,
        th,
    );

    Verdict {
        ok: verdict == "green" || verdict == "yellow",
        verdict,
        n_total,
        n_shortlist,
        n_maybe,
        n_rejected,
        n_graded,
        shortlist_rate,
        mean_relevance,
        top_relevance,
        dominant_reject_reasons: dominant_reasons,
        advice,
    }
}

// ── helpers ──────────────────────────────────────────────────────────

fn bucket_counts(findings: &[Value], shortlist_min: f64, maybe_min: f64) -> (usize, usize, usize) {
    let mut n_short = 0usize;
    let mut n_maybe = 0usize;
    let mut n_rej = 0usize;
    for f in findings {
        let score = f.get("_score").and_then(Value::as_f64).unwrap_or(0.0);
        if score >= shortlist_min {
            n_short += 1;
        } else if score >= maybe_min {
            n_maybe += 1;
        } else {
            n_rej += 1;
        }
    }
    (n_short, n_maybe, n_rej)
}

/// Top-3 reject reasons among `_relevance <= low_relevance_max`,
/// sorted by count desc with ties broken by first-occurrence.
///
/// Insertion-order preservation: we use a `Vec<(String, usize)>` and
/// linear lookup. N is small in practice (typically <30 distinct
/// reasons per brief) so this is faster than a hash-map plus sort,
/// and it sidesteps the `IndexMap` dep that Python's `OrderedDict`
/// would naively translate to.
fn dominant_reasons(graded: &[&Value], low_relevance_max: i64) -> Vec<ReasonCount> {
    let cap = low_relevance_max as f64;
    let mut counts: Vec<(String, usize)> = Vec::new();
    for f in graded {
        let rel = f.get("_relevance").and_then(Value::as_f64).unwrap_or(0.0);
        // Python uses `>` (strict) — 35 ≤ 35 PASSES the gate; rel=36
        // is filtered out. Mirror exactly.
        if rel > cap {
            continue;
        }
        let reason = f
            .get("_relevance_reason")
            .and_then(Value::as_str)
            .map(|s| s.trim())
            .unwrap_or("");
        if reason.is_empty() {
            continue;
        }
        match counts.iter_mut().find(|(k, _)| k == reason) {
            Some((_, c)) => *c += 1,
            None => counts.push((reason.to_string(), 1)),
        }
    }
    // Stable sort by count desc — ties preserve insertion order
    // (this is what makes the differential output byte-stable).
    counts.sort_by_key(|a| std::cmp::Reverse(a.1));
    counts
        .into_iter()
        .take(3)
        .map(|(r, c)| ReasonCount {
            reason: r,
            count: c,
        })
        .collect()
}

fn decide_verdict(
    shortlist_rate: f64,
    mean_relevance: Option<f64>,
    top_relevance: Option<i64>,
    th: &Thresholds,
) -> String {
    if shortlist_rate >= th.green_shortlist_rate {
        return "green".to_string();
    }
    if let (Some(top), Some(mean)) = (top_relevance, mean_relevance)
        && top >= th.green_top_relevance
        && mean >= th.green_mean_relevance
    {
        return "green".to_string();
    }
    if shortlist_rate >= th.yellow_shortlist_rate {
        return "yellow".to_string();
    }
    if let Some(top) = top_relevance
        && top >= th.yellow_top_relevance
    {
        return "yellow".to_string();
    }
    "red".to_string()
}

/// Format a percentage via Python `int(round(x * 100))` semantics.
///
/// Note on rounding: Python 3's `round()` uses banker's rounding
/// (round-half-to-even) but f64 `.round()` rounds-half-away-from-zero.
/// In practice, the rates used here (10%, 30%, multiples of 5%) never
/// hit a half-cent boundary, so both rules agree on every fixture
/// case. If a future caller passes a rate like `0.005` we'll diverge
/// by one unit — flagged in module docs in case it ever bites.
fn pct(rate: f64) -> i64 {
    (rate * 100.0).round() as i64
}

fn advice_for(
    verdict: &str,
    n_total: usize,
    shortlist_rate: f64,
    top_relevance: Option<i64>,
    dominant_reasons: &[ReasonCount],
    th: &Thresholds,
) -> String {
    if verdict == "green" {
        return String::new();
    }

    let top_str = match top_relevance {
        Some(v) => v.to_string(),
        None => "—".to_string(),
    };

    if verdict == "yellow" {
        return format!(
            "Частичное покрытие: shortlist_rate={}% при пороге green={}%. \
             Top-карточка имеет relevance {}/100; есть пограничные \
             совпадения, но точных мало. Уточните вопрос дополнительными \
             ограничениями (бюджет, локация, год выпуска).",
            pct(shortlist_rate),
            pct(th.green_shortlist_rate),
            top_str,
        );
    }

    // verdict == "red"
    // Python takes the first 2 dominant reasons for the user-facing
    // string (the full top-3 is still in `dominant_reject_reasons`).
    let reasons_str = if dominant_reasons.is_empty() {
        "—".to_string()
    } else {
        dominant_reasons
            .iter()
            .take(2)
            .map(|r| format!("«{}» ({})", r.reason, r.count))
            .collect::<Vec<_>>()
            .join(", ")
    };

    if n_total < th.thin_sample_max {
        format!(
            "Релевантных карточек не найдено: shortlist_rate={}% и \
             top-карточка получила relevance {}/100. Доминирующие \
             причины отказа: {}. Малая выборка ({} карточек) — \
             попробуйте увеличить дискавери: --coordinator-max-iters 8.",
            pct(shortlist_rate),
            top_str,
            reasons_str,
            n_total,
        )
    } else {
        format!(
            "Релевантных карточек не найдено: shortlist_rate={}% и \
             top-карточка получила relevance {}/100. Доминирующие \
             причины отказа: {}. Скорее всего scraper выловил \
             нерелевантный блок — selector_synthesizer требует \
             переобучения для этих источников.",
            pct(shortlist_rate),
            top_str,
            reasons_str,
        )
    }
}

// ── tests ────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    fn fixture_path() -> PathBuf {
        let mut p = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        p.pop(); // crates
        p.pop(); // naked-core
        p.push("tests");
        p.push("fixtures");
        p.push("quality_assessor");
        p.push("expected.json");
        p
    }

    /// Differential test for the full B4.6 contract.
    ///
    /// Consumes `tests/fixtures/quality_assessor/expected.json`, the
    /// SAME JSON the Python e2e harness (group 24) reads. Compares
    /// every key the contract pins down — verdict, counts, rate,
    /// mean/top, dominant reasons (ordered), and the byte-for-byte
    /// advice string. Russian advice copy is asserted exactly because
    /// downstream HTML / MCP / autopilot consume it verbatim.
    #[test]
    fn assess_quality_matches_python_fixture() {
        let raw = std::fs::read_to_string(fixture_path()).expect("read quality_assessor fixture");
        let v: Value = serde_json::from_str(&raw).expect("parse fixture");

        let th = Thresholds::from_json(&v["_thresholds"]);

        let mut failures: Vec<String> = Vec::new();

        for case in v["cases"].as_array().expect("cases array") {
            let name = case["name"].as_str().unwrap_or("?").to_string();
            let findings: Vec<Value> = case["findings"].as_array().cloned().unwrap_or_default();
            let expected = &case["expected"];

            let got = assess_quality(&findings, Some(&th)).to_json();
            let exp_obj = expected.as_object().expect("expected object");

            for (k, exp_v) in exp_obj {
                let Some(got_v) = got.get(k) else {
                    failures.push(format!("[{name}] missing key {k:?}"));
                    continue;
                };
                if !json_eq(got_v, exp_v) {
                    failures.push(format!("[{name}] {k}: got {got_v}, expected {exp_v}"));
                }
            }
        }

        assert!(
            failures.is_empty(),
            "quality_assessor drift from Python contract:\n  - {}",
            failures.join("\n  - ")
        );
    }

    /// JSON equality with f64 tolerance.
    ///
    /// Strict `Value == Value` would require `mean_relevance` (e.g.
    /// `31.666666666666668`) to land on the exact same f64 bit
    /// pattern. IEEE 754 division of `95.0/3.0` is byte-identical
    /// across Python and Rust, so strict eq actually works — but we
    /// keep a 1e-9 epsilon as defensive cushion for future cases
    /// that go through a `mul/add/round` chain where reordering
    /// could cost a ULP.
    fn json_eq(a: &Value, b: &Value) -> bool {
        match (a, b) {
            (Value::Number(x), Value::Number(y)) => match (x.as_f64(), y.as_f64()) {
                (Some(xf), Some(yf)) => (xf - yf).abs() <= 1e-9,
                _ => x == y,
            },
            (Value::Array(xs), Value::Array(ys)) if xs.len() == ys.len() => {
                xs.iter().zip(ys.iter()).all(|(p, q)| json_eq(p, q))
            }
            (Value::Object(xm), Value::Object(ym)) if xm.len() == ym.len() => xm
                .iter()
                .all(|(k, v)| ym.get(k).is_some_and(|w| json_eq(v, w))),
            _ => a == b,
        }
    }
}
