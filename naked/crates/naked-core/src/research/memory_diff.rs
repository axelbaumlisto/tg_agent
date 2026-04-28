//! B6 memory_diff — state-of-knowledge tracker.
//!
//! Bit-exact Rust port of `scripts/memory_diff.py`. Compares two
//! snapshots of reconciled findings ("last week" vs "now") and tells
//! the user what changed:
//!
//! ```text
//! diff_findings(old, new) → {
//!     new:       [...],            // appeared since last snapshot
//!     gone:      [...],            // disappeared since last snapshot
//!     changed:   [{old, new, deltas}, ...],
//!     unchanged: [...],
//!     summary:   {n_new, n_gone, n_changed, n_unchanged,
//!                 n_total_old, n_total_new},
//! }
//! ```
//!
//! ## Design contract (mirrors the Python module)
//!
//! 1. **Identity** — items are joined by `_canonical_url` (the B5
//!    reconciler invariant). Defensive fallback to `link` so the
//!    function still works on raw scraper output.
//! 2. **Drift signals** (only these matter):
//!    - `title`     — string equality after `normalize_title`
//!      (the conservative B5 normaliser, NOT the fuzzy similarity one)
//!    - `price_usd` — abs delta > $1 AND relative delta > 5 %
//!    - `_score`    — abs delta > 5.0 (heuristic noise floor)
//!    - `_bucket`   — string equality (`shortlist`/`maybe` move)
//!
//!    Any other key change is ignored. This keeps signal-to-noise high.
//! 3. **Pure** — `diff_findings` is stdlib + reconciler-only; no I/O,
//!    no time, no globals. Determinism: input-preserving order for
//!    `new`/`unchanged`; old-preserving order for `gone`.
//!
//! ## Contract pinning
//!
//! * Python: e2e groups 31 (`diff_findings`) + 32 (`diff_reports`)
//! * Rust:   e2e group 37 (`37_memory_diff_rust_differential_offline.sh`)
//!
//! All three consume `tests/fixtures/memory_diff/diff_findings.json`
//! plus tmpdir-staged snapshots for the file-I/O wrapper. Drift on
//! either side trips both CI runs.

use std::fs;
use std::path::{Path, PathBuf};
use std::sync::LazyLock;

use regex::Regex;
use serde_json::{Value, json};

use super::reconciler::normalize_title;

// ── Drift thresholds ────────────────────────────────────────────────

const PRICE_ABS_NOISE: f64 = 1.0;
const PRICE_REL_NOISE: f64 = 0.05;
const SCORE_ABS_NOISE: f64 = 5.0;

// ── Pure: identity & drift detection ────────────────────────────────

/// Stable join key. Prefer `_canonical_url` (B5 invariant); fall back
/// to `link`; return `None` if neither is present so the caller drops
/// the item from the diff entirely. Bit-exact mirror of
/// `memory_diff._identity_key`.
fn identity_key(finding: &Value) -> Option<&str> {
    let obj = finding.as_object()?;
    if let Some(s) = obj.get("_canonical_url").and_then(Value::as_str)
        && !s.is_empty()
    {
        return Some(s);
    }
    if let Some(s) = obj.get("link").and_then(Value::as_str)
        && !s.is_empty()
    {
        return Some(s);
    }
    None
}

/// Treat `Value::Bool` as non-numeric even though Python `bool` is a
/// subclass of `int`. `serde_json::Value::is_number()` already
/// excludes booleans (separate variant), so this is one line —
/// documented for parity-readers.
fn as_number(v: Option<&Value>) -> Option<f64> {
    v.and_then(|x| if x.is_boolean() { None } else { x.as_f64() })
}

/// Both prices are numeric and the delta clears BOTH the absolute
/// and relative noise floors. One-sided None (price appeared /
/// disappeared) also counts as changed.
fn price_changed(old_v: Option<&Value>, new_v: Option<&Value>) -> bool {
    let a = as_number(old_v);
    let b = as_number(new_v);
    match (a, b) {
        (Some(x), Some(y)) => {
            let delta = (y - x).abs();
            if delta <= PRICE_ABS_NOISE {
                return false;
            }
            let denom = x.abs().max(y.abs()).max(1.0);
            delta / denom > PRICE_REL_NOISE
        }
        (Some(_), None) | (None, Some(_)) => true,
        (None, None) => false,
    }
}

fn score_changed(old_v: Option<&Value>, new_v: Option<&Value>) -> bool {
    match (as_number(old_v), as_number(new_v)) {
        (Some(x), Some(y)) => (y - x).abs() > SCORE_ABS_NOISE,
        (Some(_), None) | (None, Some(_)) => true,
        (None, None) => false,
    }
}

fn title_changed(old_v: Option<&Value>, new_v: Option<&Value>) -> bool {
    let a = old_v.and_then(Value::as_str).unwrap_or("");
    let b = new_v.and_then(Value::as_str).unwrap_or("");
    normalize_title(a) != normalize_title(b)
}

fn bucket_changed(old_v: Option<&Value>, new_v: Option<&Value>) -> bool {
    let a = old_v.and_then(Value::as_str).unwrap_or("").trim();
    let b = new_v.and_then(Value::as_str).unwrap_or("").trim();
    a != b
}

/// Compute the per-finding drift signal list. Order matters: the
/// fixture's `expected_changed_deltas` is sorted, so we just emit in
/// the same order as the Python reference (`_bucket`, `price_usd`,
/// `_score`, `title`) and the harness sorts both sides for compare.
fn detect_deltas(old_item: &Value, new_item: &Value) -> Vec<&'static str> {
    let mut deltas: Vec<&'static str> = Vec::new();
    if bucket_changed(old_item.get("_bucket"), new_item.get("_bucket")) {
        deltas.push("_bucket");
    }
    if price_changed(old_item.get("price_usd"), new_item.get("price_usd")) {
        deltas.push("price_usd");
    }
    if score_changed(old_item.get("_score"), new_item.get("_score")) {
        deltas.push("_score");
    }
    if title_changed(old_item.get("title"), new_item.get("title")) {
        deltas.push("title");
    }
    deltas
}

/// Compare two reconciled-findings lists. See module docstring.
///
/// Output is a JSON object with five keys (`new`, `gone`, `changed`,
/// `unchanged`, `summary`). The lists preserve input order:
/// `new`/`unchanged`/`changed` follow `new`'s first-seen order;
/// `gone` follows `old`'s first-seen order.
pub fn diff_findings(old: &[Value], new: &[Value]) -> Value {
    // Build identity → finding maps. Last-write-wins for duplicate
    // keys; first-seen order for the position vector. Python uses
    // dict insertion-order; we mimic with a parallel Vec<&str>.
    let mut old_order: Vec<String> = Vec::new();
    let mut old_by_key: std::collections::HashMap<String, &Value> =
        std::collections::HashMap::new();
    for f in old {
        let Some(k) = identity_key(f) else { continue };
        let k = k.to_string();
        if !old_by_key.contains_key(&k) {
            old_order.push(k.clone());
        }
        old_by_key.insert(k, f);
    }

    let mut new_order: Vec<String> = Vec::new();
    let mut new_by_key: std::collections::HashMap<String, &Value> =
        std::collections::HashMap::new();
    for f in new {
        let Some(k) = identity_key(f) else { continue };
        let k = k.to_string();
        if !new_by_key.contains_key(&k) {
            new_order.push(k.clone());
        }
        new_by_key.insert(k, f);
    }

    let mut new_items: Vec<Value> = Vec::new();
    let mut changed_items: Vec<Value> = Vec::new();
    let mut unchanged_items: Vec<Value> = Vec::new();
    for k in &new_order {
        let n = new_by_key[k];
        if let Some(o) = old_by_key.get(k) {
            let deltas = detect_deltas(o, n);
            if deltas.is_empty() {
                unchanged_items.push((*n).clone());
            } else {
                changed_items.push(json!({
                    "old": *o,
                    "new": n,
                    "deltas": deltas,
                }));
            }
        } else {
            new_items.push(n.clone());
        }
    }

    let mut gone_items: Vec<Value> = Vec::new();
    for k in &old_order {
        if !new_by_key.contains_key(k) {
            gone_items.push(old_by_key[k].clone());
        }
    }

    let summary = json!({
        "n_new": new_items.len(),
        "n_gone": gone_items.len(),
        "n_changed": changed_items.len(),
        "n_unchanged": unchanged_items.len(),
        "n_total_old": old_by_key.len(),
        "n_total_new": new_by_key.len(),
    });

    json!({
        "new": new_items,
        "gone": gone_items,
        "changed": changed_items,
        "unchanged": unchanged_items,
        "summary": summary,
    })
}

// ── File-IO wrapper ─────────────────────────────────────────────────

/// Load a JSONL findings snapshot. Empty / missing → `vec![]`.
/// Malformed lines are skipped silently (defensive — a single bad
/// line shouldn't tank a 7-day diff).
pub fn load_snapshot(path: &Path) -> Vec<Value> {
    let raw = match fs::read_to_string(path) {
        Ok(s) => s,
        Err(_) => return Vec::new(),
    };
    raw.lines()
        .map(str::trim)
        .filter(|l| !l.is_empty())
        .filter_map(|l| serde_json::from_str::<Value>(l).ok())
        .filter(Value::is_object)
        .collect()
}

/// Filename pattern emitted by `playwright_tick.sh`:
/// `report_YYYY-MM-DD_HHMM.findings.jsonl`.
static SNAPSHOT_NAME_RE: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"^report_(\d{4}-\d{2}-\d{2})_(\d{4})\.findings\.jsonl$")
        .expect("static snapshot-name regex")
});

/// Return all snapshots in `report_dir` sorted oldest → newest.
/// Sort is on the encoded `(date, time)` string (`YYYY-MM-DD_HHMM`),
/// which is lexicographically equivalent to chronological for the
/// canonical filename pattern. This matters across midnight: the
/// fixture-pinned scenario in group 32 verifies
/// `report_2026-04-25_2330` < `report_2026-04-26_0010`.
pub fn list_snapshots(report_dir: &Path) -> Vec<PathBuf> {
    let entries = match fs::read_dir(report_dir) {
        Ok(it) => it,
        Err(_) => return Vec::new(),
    };
    let mut snaps: Vec<(String, PathBuf)> = Vec::new();
    for entry in entries.flatten() {
        let path = entry.path();
        let Some(name) = path.file_name().and_then(|n| n.to_str()) else {
            continue;
        };
        if let Some(caps) = SNAPSHOT_NAME_RE.captures(name) {
            let key = format!("{}_{}", &caps[1], &caps[2]);
            snaps.push((key, path));
        }
    }
    snaps.sort_by(|a, b| a.0.cmp(&b.0));
    snaps.into_iter().map(|(_, p)| p).collect()
}

/// Compare the most recent snapshot in `report_dir` against the one
/// before it. If `since=YYYY-MM-DD` is provided, the OLD snapshot is
/// the most recent one whose **date portion is strictly before**
/// `since` (the trailing time slot is ignored — same semantics as
/// Python: `m.group(1) < target`).
///
/// Returns a `diff_findings`-shaped JSON object plus two extra
/// fields:
///
/// * `old_snapshot` — path of the older snapshot (or `null`)
/// * `new_snapshot` — path of the newer snapshot (or `null`)
pub fn diff_reports(report_dir: &Path, since: Option<&str>) -> Value {
    let snaps = list_snapshots(report_dir);
    if snaps.is_empty() {
        let mut out = diff_findings(&[], &[]);
        if let Some(o) = out.as_object_mut() {
            o.insert("old_snapshot".into(), Value::Null);
            o.insert("new_snapshot".into(), Value::Null);
        }
        return out;
    }

    let new_path = snaps.last().cloned().expect("non-empty checked above");
    let mut old_path: Option<PathBuf> = None;
    if let Some(s) = since {
        let target = s.trim();
        // Iterate the snapshots BEFORE the latest, newest-first.
        for p in snaps.iter().rev().skip(1) {
            let name = match p.file_name().and_then(|n| n.to_str()) {
                Some(n) => n,
                None => continue,
            };
            if let Some(caps) = SNAPSHOT_NAME_RE.captures(name)
                && &caps[1] < target
            {
                old_path = Some(p.clone());
                break;
            }
        }
    } else if snaps.len() >= 2 {
        old_path = Some(snaps[snaps.len() - 2].clone());
    }

    let old_findings = old_path.as_deref().map(load_snapshot).unwrap_or_default();
    let new_findings = load_snapshot(&new_path);

    let mut out = diff_findings(&old_findings, &new_findings);
    if let Some(o) = out.as_object_mut() {
        o.insert(
            "old_snapshot".into(),
            old_path
                .as_ref()
                .map(|p| Value::String(p.to_string_lossy().into()))
                .unwrap_or(Value::Null),
        );
        o.insert(
            "new_snapshot".into(),
            Value::String(new_path.to_string_lossy().into()),
        );
    }
    out
}

// ── tests ───────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn fixture_path() -> PathBuf {
        let mut p = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        p.pop(); // crates
        p.pop(); // naked-core
        p.push("tests");
        p.push("fixtures");
        p.push("memory_diff");
        p.push("diff_findings.json");
        p
    }

    /// Differential test for `diff_findings` against the same
    /// fixture Python e2e group 31 reads. For each case we verify
    /// (a) summary counters match `expected_summary`, (b)
    /// new/gone canonical URLs match `expected_new_urls`/
    /// `expected_gone_urls` if specified, (c) per-changed-item
    /// `deltas` match `expected_changed_deltas` (sorted both sides).
    #[test]
    fn diff_findings_matches_python_fixture() {
        let raw = fs::read_to_string(fixture_path()).expect("read fixture");
        let v: Value = serde_json::from_str(&raw).expect("parse fixture");

        let mut failures: Vec<String> = Vec::new();

        for case in v["cases"].as_array().expect("cases") {
            let name = case["name"].as_str().unwrap_or("?");
            let old: Vec<Value> = case["old"].as_array().cloned().unwrap_or_default();
            let new: Vec<Value> = case["new"].as_array().cloned().unwrap_or_default();
            let out = diff_findings(&old, &new);

            let summary = &out["summary"];
            for (k, expected) in case["expected_summary"]
                .as_object()
                .expect("expected_summary")
            {
                let got = &summary[k];
                if got != expected {
                    failures.push(format!(
                        "[{name}] summary.{k}: got {got}, expected {expected}"
                    ));
                }
            }

            if let Some(exp) = case.get("expected_new_urls").and_then(Value::as_array) {
                let mut got: Vec<String> = out["new"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .filter_map(|f| {
                        f.get("_canonical_url")
                            .or_else(|| f.get("link"))
                            .and_then(Value::as_str)
                            .map(str::to_string)
                    })
                    .collect();
                got.sort();
                let mut exp: Vec<String> = exp
                    .iter()
                    .filter_map(|v| v.as_str().map(str::to_string))
                    .collect();
                exp.sort();
                if got != exp {
                    failures.push(format!("[{name}] new urls: got {got:?}, expected {exp:?}"));
                }
            }

            if let Some(exp) = case.get("expected_gone_urls").and_then(Value::as_array) {
                let mut got: Vec<String> = out["gone"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .filter_map(|f| {
                        f.get("_canonical_url")
                            .or_else(|| f.get("link"))
                            .and_then(Value::as_str)
                            .map(str::to_string)
                    })
                    .collect();
                got.sort();
                let mut exp: Vec<String> = exp
                    .iter()
                    .filter_map(|v| v.as_str().map(str::to_string))
                    .collect();
                exp.sort();
                if got != exp {
                    failures.push(format!("[{name}] gone urls: got {got:?}, expected {exp:?}"));
                }
            }

            if let Some(exp) = case
                .get("expected_changed_deltas")
                .and_then(Value::as_array)
            {
                let mut got: Vec<Vec<String>> = out["changed"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .map(|ch| {
                        let mut d: Vec<String> = ch["deltas"]
                            .as_array()
                            .unwrap_or(&vec![])
                            .iter()
                            .filter_map(|v| v.as_str().map(str::to_string))
                            .collect();
                        d.sort();
                        d
                    })
                    .collect();
                got.sort();
                let mut exp: Vec<Vec<String>> = exp
                    .iter()
                    .map(|d| {
                        let mut v: Vec<String> = d
                            .as_array()
                            .unwrap_or(&vec![])
                            .iter()
                            .filter_map(|x| x.as_str().map(str::to_string))
                            .collect();
                        v.sort();
                        v
                    })
                    .collect();
                exp.sort();
                if got != exp {
                    failures.push(format!(
                        "[{name}] changed deltas: got {got:?}, expected {exp:?}"
                    ));
                }
            }
        }

        assert!(
            failures.is_empty(),
            "memory_diff drift from Python contract:\n  - {}",
            failures.join("\n  - ")
        );
    }

    /// Mirrors group 32 scenarios 1-9 in a single Rust unit test.
    /// Stages snapshot files in a tempdir, exercises
    /// `diff_reports` + `list_snapshots` + `load_snapshot`.
    #[test]
    fn diff_reports_stages_match_python_contract() {
        let tmp = std::env::temp_dir().join(format!("naked-memory-diff-{}", std::process::id()));
        let _ = fs::remove_dir_all(&tmp);
        fs::create_dir_all(&tmp).expect("mkdir tmp");

        // Helper: write JSONL.
        fn write_snap(d: &Path, name: &str, items: &[Value]) -> PathBuf {
            let p = d.join(name);
            let body: String = items
                .iter()
                .map(|it| serde_json::to_string(it).unwrap())
                .collect::<Vec<_>>()
                .join("\n");
            fs::write(&p, body).expect("write snap");
            p
        }

        // 1. Empty directory → empty diff.
        let out = diff_reports(&tmp, None);
        assert_eq!(out["summary"]["n_new"], 0);
        assert!(out["old_snapshot"].is_null());

        // 2. Single snapshot → first-run shape.
        let s1_items = vec![
            json!({"_canonical_url": "https://x.com/a", "title": "A",
                   "price_usd": 100.0, "_score": 80.0, "_bucket": "shortlist"}),
            json!({"_canonical_url": "https://x.com/b", "title": "B",
                   "price_usd": 200.0, "_score": 60.0, "_bucket": "maybe"}),
        ];
        let s1 = write_snap(&tmp, "report_2026-04-20_1200.findings.jsonl", &s1_items);
        let out = diff_reports(&tmp, None);
        assert_eq!(out["summary"]["n_new"], 2);
        assert_eq!(out["summary"]["n_total_old"], 0);
        assert!(out["old_snapshot"].is_null());
        assert_eq!(out["new_snapshot"].as_str().unwrap(), s1.to_string_lossy());

        // 3. Two snapshots.
        let s2_items = vec![
            json!({"_canonical_url": "https://x.com/a", "title": "A",
                   "price_usd": 150.0, "_score": 80.0, "_bucket": "shortlist"}),
            json!({"_canonical_url": "https://x.com/c", "title": "C",
                   "price_usd": 300.0, "_score": 75.0, "_bucket": "shortlist"}),
        ];
        let s2 = write_snap(&tmp, "report_2026-04-25_0930.findings.jsonl", &s2_items);
        let out = diff_reports(&tmp, None);
        let s = &out["summary"];
        assert_eq!(s["n_new"], 1);
        assert_eq!(s["n_gone"], 1);
        assert_eq!(s["n_changed"], 1);
        assert_eq!(s["n_unchanged"], 0);
        assert_eq!(out["old_snapshot"].as_str().unwrap(), s1.to_string_lossy());
        assert_eq!(out["new_snapshot"].as_str().unwrap(), s2.to_string_lossy());

        // 4. since older than every snapshot → no old.
        let out = diff_reports(&tmp, Some("2026-04-15"));
        assert!(out["old_snapshot"].is_null());

        // 5. since between snap1 and snap2 → old=s1.
        let out = diff_reports(&tmp, Some("2026-04-22"));
        assert_eq!(out["old_snapshot"].as_str().unwrap(), s1.to_string_lossy());

        // 6. since == new-date → strictly < → old=s1.
        let out = diff_reports(&tmp, Some("2026-04-25"));
        assert_eq!(out["old_snapshot"].as_str().unwrap(), s1.to_string_lossy());

        // 7. Snapshot crossing midnight sorts on (date, time).
        let s3 = write_snap(&tmp, "report_2026-04-25_2330.findings.jsonl", &s2_items);
        let s4 = write_snap(&tmp, "report_2026-04-26_0010.findings.jsonl", &s2_items);
        let snaps = list_snapshots(&tmp);
        let tail: Vec<String> = snaps
            .iter()
            .rev()
            .take(2)
            .rev()
            .map(|p| p.to_string_lossy().into_owned())
            .collect();
        assert_eq!(
            tail,
            vec![
                s3.to_string_lossy().into_owned(),
                s4.to_string_lossy().into_owned()
            ]
        );

        // 8. Malformed JSONL lines are skipped, valid ones survive.
        let bad_path = tmp.join("report_2026-04-27_1200.findings.jsonl");
        fs::write(
            &bad_path,
            format!(
                "{}\n{{this-is-not-json\n{}\n",
                serde_json::to_string(&json!({
                    "_canonical_url": "https://x.com/d", "title": "D"
                }))
                .unwrap(),
                serde_json::to_string(&json!({
                    "_canonical_url": "https://x.com/e", "title": "E"
                }))
                .unwrap(),
            ),
        )
        .unwrap();
        let rows = load_snapshot(&bad_path);
        assert_eq!(rows.len(), 2, "two valid lines must survive");

        // 9. Non-existent path returns [].
        assert!(load_snapshot(&tmp.join("nope.findings.jsonl")).is_empty());

        // Cleanup (best-effort — test harness handles leftover tmpdirs).
        let _ = fs::remove_dir_all(&tmp);
    }

    #[test]
    fn price_drift_below_noise_floor_is_ignored() {
        // 100 → 100.4 = 0.4 % → noise floor (5 %) blocks it.
        assert!(!price_changed(Some(&json!(100.0)), Some(&json!(100.4))));
    }

    #[test]
    fn price_drift_above_noise_floor_fires() {
        // 100 → 150 = +50 %, way over both abs and rel floors.
        assert!(price_changed(Some(&json!(100.0)), Some(&json!(150.0))));
    }

    #[test]
    fn one_sided_price_appears() {
        // None → 100 = price appeared.
        assert!(price_changed(None, Some(&json!(100.0))));
        assert!(price_changed(Some(&json!(100.0)), None));
    }

    #[test]
    fn boolean_is_not_numeric() {
        // Defensive: serde_json keeps bool/number separate, but the
        // semantics matter — a "price" of `true` must not be treated
        // as `1`.
        assert!(!price_changed(Some(&json!(true)), Some(&json!(false))));
    }
}
