//! `naked-validate-brief` — Rust port of `scripts/validate_generic_brief.py`.
//!
//! Reads a brief JSON + N JSONL "findings" files, runs the deterministic
//! CPU pipeline (dedup → since-filter → score → bucket → quality
//! assessment), and writes `shortlist.jsonl`, `maybe.jsonl`,
//! `rejected.jsonl`, `quality.json` into `--out-dir`. Optionally prints
//! a structured summary on stdout when `--json-out` is set.
//!
//! Wraps the three Rust modules in `naked-core`:
//!
//!   * `research::reconciler::reconcile` — cross-source dedup (B5)
//!   * `research::filter_rules::score_finding` — rule engine (B4.5-1b)
//!   * `research::quality_assessor::assess_quality` — verdict (B4.6)
//!
//! All three have bit-exact differential gates against the Python
//! reference (groups 33, 35, 34 respectively), so the CLI is **pure
//! plumbing** — no new contract surface.
//!
//! ## Scope vs the Python script
//!
//! Direct equivalents:
//!
//! | Python flag                  | Rust flag                  | Notes                                         |
//! |------------------------------|----------------------------|-----------------------------------------------|
//! | `--brief PATH`               | `--brief PATH`             | required                                      |
//! | `--findings A B C`           | `--findings A B C`         | repeatable; flat-merged in seen-order         |
//! | `--out-dir DIR`              | `--out-dir DIR`            | created if missing                            |
//! | `--since YYYY-MM-DD`         | `--since YYYY-MM-DD`       | drops findings older than cutoff              |
//! | `--json-out`                 | `--json-out`               | print summary as one-line JSON on stdout      |
//! | `--no-grade-relevance`       | implicit (LLM disabled)    | Rust always runs no-grade — see below         |
//! | `--grade-relevance` and \    | n/a (warn + ignore)        | LLM path stays Python-only                    |
//! | `--relevance-*` family       |                            |                                               |
//!
//! `--grade-relevance` and the four `--relevance-*` flags are accepted
//! for argparse-compatibility (so brief autopilot scripts can swap
//! `python3 validate_generic_brief.py` ⇄ `naked-validate-brief` with
//! no other changes), but logged as a one-line stderr warning and
//! otherwise no-op'd. The relevance grader is LLM-bound and out of
//! scope for the deterministic CPU port.
//!
//! ## CLI argument parser
//!
//! Hand-rolled parser, matching the rest of `naked-cli` (no `clap`
//! dep). Brief enough to fit inline; mirrors the Python `argparse`
//! surface exactly.

use std::collections::BTreeSet;
use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::ExitCode;

use anyhow::{Context, Result, bail};
use chrono::{DateTime, NaiveDate, Utc};
use serde_json::{Value, json};

use naked_core::research::filter_rules::score_finding;
use naked_core::research::quality_assessor::{Thresholds, assess_quality};
use naked_core::research::reconciler::reconcile;

#[derive(Default)]
struct Args {
    brief: Option<PathBuf>,
    findings: Vec<PathBuf>,
    out_dir: Option<PathBuf>,
    since: Option<String>,
    json_out: bool,
    grade_relevance: bool,
    no_grade_relevance: bool,
    help: bool,
}

const HELP: &str = "Usage: naked-validate-brief --brief PATH --findings PATH... \
                    --out-dir DIR [--since YYYY-MM-DD] [--json-out] \
                    [--no-grade-relevance] [--grade-relevance]\n\
                    \n\
                    Score findings against a brief's filter rules and \
                    partition into shortlist/maybe/rejected.\n\
                    \n\
                    --grade-relevance is accepted for argparse-compat \
                    with the Python script but the LLM grader is not \
                    available in the Rust port; the flag is no-op'd \
                    with a warning.\n";

fn parse_args() -> Result<Args> {
    let raw: Vec<String> = std::env::args().skip(1).collect();
    let mut args = Args::default();
    let mut i = 0;
    while i < raw.len() {
        let a = raw[i].as_str();
        match a {
            "-h" | "--help" => args.help = true,
            "--brief" => {
                i += 1;
                args.brief = raw.get(i).map(PathBuf::from);
            }
            "--out-dir" => {
                i += 1;
                args.out_dir = raw.get(i).map(PathBuf::from);
            }
            "--since" => {
                i += 1;
                args.since = raw.get(i).cloned();
            }
            "--json-out" => args.json_out = true,
            "--grade-relevance" => args.grade_relevance = true,
            "--no-grade-relevance" => args.no_grade_relevance = true,
            // argparse-compat no-ops (LLM grader is Python-only).
            "--relevance-question"
            | "--relevance-country"
            | "--relevance-batch-size"
            | "--relevance-max-findings"
            | "--relevance-bonus-weight" => {
                // consume the value too
                i += 1;
            }
            "--findings" => {
                // Python: nargs="+" + action="append". We accept all
                // following positional (non-`--…`) tokens until the
                // next flag. Repeated `--findings` blocks all merge.
                i += 1;
                while i < raw.len() && !raw[i].starts_with("--") {
                    args.findings.push(PathBuf::from(&raw[i]));
                    i += 1;
                }
                continue;
            }
            unknown => bail!("unknown arg `{unknown}` (try --help)"),
        }
        i += 1;
    }
    Ok(args)
}

fn read_jsonl(path: &Path) -> Vec<Value> {
    let mut out = Vec::new();
    let raw = match fs::read_to_string(path) {
        Ok(s) => s,
        Err(_) => {
            eprintln!("  WARN findings file missing: {}", path.display());
            return out;
        }
    };
    for line in raw.lines() {
        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }
        match serde_json::from_str::<Value>(trimmed) {
            Ok(v) => out.push(v),
            Err(e) => {
                eprintln!("  WARN bad json in {}: {e}", path.display());
            }
        }
    }
    out
}

fn parse_collected_at(raw: &str) -> Option<NaiveDate> {
    // Python: datetime.fromisoformat(ca.replace("Z", "+00:00")).date().
    let normalised = raw.replace('Z', "+00:00");
    if let Ok(dt) = DateTime::parse_from_rfc3339(&normalised) {
        return Some(dt.with_timezone(&Utc).date_naive());
    }
    if let Ok(dt) = DateTime::parse_from_str(&normalised, "%Y-%m-%dT%H:%M:%S%:z") {
        return Some(dt.with_timezone(&Utc).date_naive());
    }
    if let Ok(d) = NaiveDate::parse_from_str(raw, "%Y-%m-%d") {
        return Some(d);
    }
    None
}

/// Drop findings whose `_collected_at` precedes `since`.
/// Findings with missing or unparseable timestamps are kept (defensive
/// — better to over-include than to silently drop), matching the Python
/// `_filter_since` behaviour byte-for-byte.
fn filter_since(findings: Vec<Value>, since: Option<&str>) -> Vec<Value> {
    let cutoff = match since {
        Some(s) => match NaiveDate::parse_from_str(s, "%Y-%m-%d") {
            Ok(d) => d,
            Err(_) => return findings,
        },
        None => return findings,
    };
    findings
        .into_iter()
        .filter(|f| {
            let Some(ca) = f.get("_collected_at").and_then(Value::as_str) else {
                return true;
            };
            match parse_collected_at(ca) {
                Some(d) => d >= cutoff,
                None => true,
            }
        })
        .collect()
}

fn run(args: Args) -> Result<()> {
    let brief_path = args
        .brief
        .as_ref()
        .context("--brief is required (try --help)")?;
    let out_dir = args
        .out_dir
        .as_ref()
        .context("--out-dir is required")?;
    if args.findings.is_empty() {
        bail!("--findings is required (one or more JSONL paths)");
    }

    let brief: Value = serde_json::from_str(
        &fs::read_to_string(brief_path)
            .with_context(|| format!("reading brief {}", brief_path.display()))?,
    )
    .with_context(|| format!("parsing brief {}", brief_path.display()))?;

    let filters = brief.get("filters").cloned().unwrap_or_else(|| json!({}));
    let rules: Vec<Value> = filters
        .get("rules")
        .and_then(Value::as_array)
        .cloned()
        .unwrap_or_default();
    let thresholds = filters.get("score_thresholds").cloned().unwrap_or(json!({}));
    let shortlist_min = thresholds
        .get("shortlist_min")
        .and_then(Value::as_f64)
        .unwrap_or(0.0);
    let maybe_min = thresholds
        .get("maybe_min")
        .and_then(Value::as_f64)
        .unwrap_or(0.0);

    // ── --grade-relevance compat-warn ─────────────────────────────
    let autopilot_meta = brief.get("_autopilot_meta").cloned().unwrap_or(json!({}));
    let auto_grade = autopilot_meta
        .get("grade_relevance")
        .and_then(Value::as_bool)
        .unwrap_or(false);
    let grade_requested = !args.no_grade_relevance && (args.grade_relevance || auto_grade);
    if grade_requested {
        eprintln!(
            "  WARN relevance grading requested but the LLM grader is \
             Python-only; falling back to no-grade. Use \
             `python3 scripts/validate_generic_brief.py --grade-relevance` \
             when LLM grading is required."
        );
    }

    // ── load + dedup + since-filter ──────────────────────────────
    let mut flat_paths: Vec<PathBuf> = Vec::new();
    let mut seen: BTreeSet<PathBuf> = BTreeSet::new();
    for p in &args.findings {
        if seen.insert(p.clone()) {
            flat_paths.push(p.clone());
        }
    }
    let mut findings: Vec<Value> = flat_paths
        .iter()
        .flat_map(|p| read_jsonl(p))
        .collect();
    let n_raw = findings.len();
    findings = reconcile(&findings, None, false);
    let n_dedup = findings.len();
    findings = filter_since(findings, args.since.as_deref());
    let n_after_since = findings.len();

    // ── score ────────────────────────────────────────────────────
    for f in &mut findings {
        score_finding(&rules, f);
    }

    // ── sort desc by _score (None → 0.0); stable sort to preserve
    //   intra-tie order, mirroring Python's stable sort. ────────────
    findings.sort_by(|a, b| {
        let sa = a.get("_score").and_then(Value::as_f64).unwrap_or(0.0);
        let sb = b.get("_score").and_then(Value::as_f64).unwrap_or(0.0);
        sb.partial_cmp(&sa).unwrap_or(std::cmp::Ordering::Equal)
    });

    let mut shortlist: Vec<&Value> = Vec::new();
    let mut maybe: Vec<&Value> = Vec::new();
    let mut rejected: Vec<&Value> = Vec::new();
    for f in &findings {
        let s = f.get("_score").and_then(Value::as_f64).unwrap_or(0.0);
        if s >= shortlist_min {
            shortlist.push(f);
        } else if s >= maybe_min {
            maybe.push(f);
        } else {
            rejected.push(f);
        }
    }

    fs::create_dir_all(out_dir)
        .with_context(|| format!("creating out-dir {}", out_dir.display()))?;

    let paths = [
        ("shortlist", out_dir.join("shortlist.jsonl"), &shortlist),
        ("maybe", out_dir.join("maybe.jsonl"), &maybe),
        ("rejected", out_dir.join("rejected.jsonl"), &rejected),
    ];
    for (_label, p, items) in &paths {
        let mut fh = fs::File::create(p)
            .with_context(|| format!("creating {}", p.display()))?;
        for f in items.iter() {
            let line = serde_json::to_string(f)?;
            fh.write_all(line.as_bytes())?;
            fh.write_all(b"\n")?;
        }
    }

    // ── B4.6 quality assessment ──────────────────────────────────
    let t = Thresholds {
        shortlist_min,
        maybe_min,
        ..Thresholds::default()
    };
    let verdict = assess_quality(&findings, Some(&t));
    let quality = verdict.to_json();
    let quality_path = out_dir.join("quality.json");
    fs::write(
        &quality_path,
        serde_json::to_string_pretty(&quality)? + "\n",
    )
    .with_context(|| format!("writing {}", quality_path.display()))?;

    let summary = json!({
        "ok": true,
        "n_raw": n_raw,
        "n_after_dedup": n_dedup,
        "n_after_since": n_after_since,
        "shortlist_count": shortlist.len(),
        "maybe_count": maybe.len(),
        "rejected_count": rejected.len(),
        "thresholds": {
            "shortlist_min": shortlist_min,
            "maybe_min": maybe_min,
        },
        "out_dir": out_dir.to_string_lossy(),
        "paths": {
            "shortlist": paths[0].1.to_string_lossy(),
            "maybe": paths[1].1.to_string_lossy(),
            "rejected": paths[2].1.to_string_lossy(),
        },
        "relevance_graded": 0,
        "quality": quality,
        "quality_path": quality_path.to_string_lossy(),
    });

    if args.json_out {
        println!("{}", serde_json::to_string(&summary)?);
    } else {
        let v = quality
            .get("verdict")
            .and_then(Value::as_str)
            .unwrap_or("?");
        eprintln!(
            "  raw={n_raw} → dedup={n_dedup} → since={n_after_since} → \
             shortlist={} maybe={} rejected={} [quality={v}]",
            shortlist.len(),
            maybe.len(),
            rejected.len(),
        );
        if quality.get("ok").and_then(Value::as_bool) == Some(false) {
            let advice = quality
                .get("advice")
                .and_then(Value::as_str)
                .unwrap_or("")
                .trim();
            if !advice.is_empty() {
                eprintln!("  ⚠ {advice}");
            }
        }
    }
    Ok(())
}

fn main() -> ExitCode {
    let args = match parse_args() {
        Ok(a) => a,
        Err(e) => {
            eprintln!("naked-validate-brief: {e:#}");
            return ExitCode::FAILURE;
        }
    };
    if args.help {
        print!("{HELP}");
        return ExitCode::SUCCESS;
    }
    match run(args) {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("naked-validate-brief: {e:#}");
            ExitCode::FAILURE
        }
    }
}
