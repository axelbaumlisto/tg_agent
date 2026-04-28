//! Auto-enrichment loop for the Model Capabilities Catalog (Phase 4).
//!
//! # Goal
//!
//! Let the bot grow its own catalog over time without forcing the
//! operator to hand-tune `naked.json` entries after every quirk.
//!
//! # Pipeline
//!
//! ```text
//!  loop_.rs / ModelHealth --[state transition]--> observations.jsonl
//!                                                      │
//!                                                 (daily digest)
//!                                                      │
//!                                                      ▼
//!                         model_catalog_suggestions.json  { add, update, deprecate }
//!                                                      │
//!                                               operator review
//!                                                      │
//!                                                      ▼
//!                                               naked.json capabilities
//! ```
//!
//! Observations are append-only JSONL lines at
//! `~/.naked/model_catalog_observations.jsonl`. They are tiny,
//! grep-friendly, and survive restarts. The daily digest (promoted by
//! [`promote_observations`]) groups them by
//! `(provider, model, observation_kind)` and keeps only the kinds that
//! occurred on **≥ 2 distinct UTC days** — one-off flakes stay out of
//! the suggestion queue, repeated behaviour makes it through.
//!
//! The promotion output is a structured patch file at
//! `~/.naked/model_catalog_suggestions.json` with three slots:
//! `add`, `update`, `deprecate`. An operator review command (CLI or a
//! `/research catalog-review` bot command) can render, accept, or
//! reject the patches before they ever touch `naked.json`. The
//! promoter **never** writes to `naked.json` directly.
//!
//! # Back-compat
//!
//! The module is self-contained — tests, CLI, and older builds that
//! haven't plumbed it compile without any wiring changes. The
//! [`ObservationRecorder`] handle is optional everywhere it shows up.

use std::collections::BTreeSet;
use std::collections::HashMap;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::Mutex;

use chrono::{DateTime, NaiveDate, Utc};
use serde::{Deserialize, Serialize};

/// What kind of state transition we saw at a `(provider, model)` pair.
///
/// Deliberately small: the plan calls for four values and the promoter
/// applies simple per-kind rules.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ObservationKind {
    /// First recorded success for a pair that previously had none (or
    /// was marked `Unknown`). Promotes a `capabilities.status = Active`
    /// suggestion.
    FirstSuccess,
    /// A new failure mode surfaced (e.g. an empty-content pattern on a
    /// model that used to work). Promotes `known_failure_modes`
    /// additions.
    NewFailureMode,
    /// The pair quietly stopped working (errors crossed the
    /// `ModelHealthConfig.error_threshold`). Promotes
    /// `status = Deprecated` suggestions after two distinct days.
    KeyDied,
    /// A previously dead pair started working again (first success
    /// within the recovery window after a quarantine). Promotes
    /// `status = Active` if the pair was Deprecated.
    Resurrected,
}

impl ObservationKind {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::FirstSuccess => "first_success",
            Self::NewFailureMode => "new_failure_mode",
            Self::KeyDied => "key_died",
            Self::Resurrected => "resurrected",
        }
    }
}

/// One observation. Kept flat so `jq`, `rg`, and humans can inspect
/// the log without reaching for a schema viewer.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelObservation {
    pub ts: DateTime<Utc>,
    pub provider: String,
    pub model: String,
    pub kind: ObservationKind,
    /// Short human-readable context — e.g. the error string that
    /// triggered `NewFailureMode`. Truncated on write to keep the log
    /// tidy.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub detail: Option<String>,
}

/// Append-only recorder with a file-backed Mutex. Cheap to clone via
/// `Arc<ObservationRecorder>`. Safe to call concurrently.
pub struct ObservationRecorder {
    path: PathBuf,
    write_lock: Mutex<()>,
}

impl std::fmt::Debug for ObservationRecorder {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ObservationRecorder")
            .field("path", &self.path)
            .finish()
    }
}

impl ObservationRecorder {
    pub fn new(path: impl Into<PathBuf>) -> Self {
        Self {
            path: path.into(),
            write_lock: Mutex::new(()),
        }
    }

    /// Use the default `~/.naked/model_catalog_observations.jsonl`
    /// location. Returns `None` when `HOME` is unset (CI containers).
    pub fn default_location() -> Option<Self> {
        let home = std::env::var_os("HOME")?;
        let path = PathBuf::from(home)
            .join(".naked")
            .join("model_catalog_observations.jsonl");
        Some(Self::new(path))
    }

    pub fn path(&self) -> &Path {
        &self.path
    }

    /// Append one observation to the log. Any I/O error is logged but
    /// never fatal — the in-memory system keeps running.
    pub fn record(
        &self,
        provider: &str,
        model: &str,
        kind: ObservationKind,
        detail: Option<String>,
    ) {
        let event = ModelObservation {
            ts: Utc::now(),
            provider: provider.to_string(),
            model: model.to_string(),
            kind,
            detail: detail.map(|d| truncate_detail(&d)),
        };
        let _guard = match self.write_lock.lock() {
            Ok(g) => g,
            Err(p) => p.into_inner(),
        };
        if let Err(e) = append_jsonl(&self.path, &event) {
            tracing::warn!(
                path = %self.path.display(),
                error = %e,
                "model_catalog.enrichment: failed to append observation",
            );
        }
    }

    /// Read every observation currently in the log. Malformed lines
    /// are skipped silently (corrupt logs should not brick the daily
    /// digest).
    pub fn load(&self) -> Vec<ModelObservation> {
        let Ok(contents) = std::fs::read_to_string(&self.path) else {
            return Vec::new();
        };
        contents
            .lines()
            .filter_map(|l| {
                let l = l.trim();
                if l.is_empty() {
                    None
                } else {
                    serde_json::from_str::<ModelObservation>(l).ok()
                }
            })
            .collect()
    }
}

/// Structured patch file consumed by operator review tools. Emitted
/// by [`promote_observations`]; applied by a future CLI subcommand.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct CatalogSuggestions {
    /// Pairs we've never seen in `naked.json` and that produced
    /// evidence on ≥ 2 distinct days. Suggest adding them with
    /// `status = Active`.
    #[serde(default)]
    pub add: Vec<CatalogAdd>,
    /// Pairs already in `naked.json` that grew new failure modes or
    /// should be marked `Degraded`.
    #[serde(default)]
    pub update: Vec<CatalogUpdate>,
    /// Pairs that stopped working for ≥ 2 distinct days. Suggest
    /// `status = Deprecated`.
    #[serde(default)]
    pub deprecate: Vec<CatalogDeprecate>,
    /// When the digest last ran. Handy for the review tool.
    #[serde(default)]
    pub generated_at: Option<DateTime<Utc>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CatalogAdd {
    pub provider: String,
    pub model: String,
    /// Kind that triggered the suggestion (usually `FirstSuccess`).
    pub evidence_kind: ObservationKind,
    /// UTC dates on which evidence was observed.
    pub evidence_days: Vec<NaiveDate>,
    pub rationale: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CatalogUpdate {
    pub provider: String,
    pub model: String,
    /// Human-readable failure fingerprints (the observation details).
    pub known_failure_modes: Vec<String>,
    pub evidence_days: Vec<NaiveDate>,
    pub rationale: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CatalogDeprecate {
    pub provider: String,
    pub model: String,
    pub evidence_days: Vec<NaiveDate>,
    pub rationale: String,
}

/// Promote observations into the structured suggestions file.
///
/// Rules (from the plan):
/// - `FirstSuccess` seen on ≥ 2 distinct days, and pair is NOT yet in
///   `known_pairs` → propose an `add`.
/// - `NewFailureMode` seen on ≥ 2 distinct days for a pair in
///   `known_pairs` → propose an `update` with the unique detail list.
/// - `KeyDied` seen on ≥ 2 distinct days → propose `deprecate`.
/// - `Resurrected` is used to *remove* matching deprecate suggestions
///   and does not produce its own patch (operator can un-deprecate by
///   inspecting the observations log).
///
/// `known_pairs` is the current state of `naked.json` — typically built
/// from `Config.providers` keys. The caller passes it in so this
/// module stays free of a `Config` dependency and is easy to unit test.
pub fn promote_observations(
    observations: &[ModelObservation],
    known_pairs: &BTreeSet<(String, String)>,
) -> CatalogSuggestions {
    // Group observations by (provider, model, kind) -> distinct days + details.
    let mut buckets: HashMap<(String, String, ObservationKind), (BTreeSet<NaiveDate>, Vec<String>)> =
        HashMap::new();
    for obs in observations {
        let day = obs.ts.date_naive();
        let entry = buckets
            .entry((obs.provider.clone(), obs.model.clone(), obs.kind))
            .or_default();
        entry.0.insert(day);
        if let Some(detail) = obs.detail.as_ref()
            && !detail.is_empty()
            && !entry.1.iter().any(|d| d == detail)
        {
            entry.1.push(detail.clone());
        }
    }

    let mut out = CatalogSuggestions {
        generated_at: Some(Utc::now()),
        ..Default::default()
    };
    let mut resurrected_pairs: BTreeSet<(String, String)> = BTreeSet::new();

    // First pass: collect resurrections so we can filter out
    // deprecations for pairs that recovered.
    for ((provider, model, kind), (days, _)) in &buckets {
        if *kind == ObservationKind::Resurrected && days.len() >= 2 {
            resurrected_pairs.insert((provider.clone(), model.clone()));
        }
    }

    for ((provider, model, kind), (days, details)) in buckets {
        if days.len() < 2 {
            continue;
        }
        let days_sorted: Vec<NaiveDate> = days.iter().copied().collect();
        let pair = (provider.clone(), model.clone());

        match kind {
            ObservationKind::FirstSuccess => {
                if !known_pairs.contains(&pair) {
                    out.add.push(CatalogAdd {
                        provider,
                        model,
                        evidence_kind: kind,
                        evidence_days: days_sorted,
                        rationale:
                            "observed `FirstSuccess` on ≥ 2 distinct days; pair missing from catalog"
                                .to_string(),
                    });
                }
            }
            ObservationKind::NewFailureMode => {
                if known_pairs.contains(&pair) {
                    out.update.push(CatalogUpdate {
                        provider,
                        model,
                        known_failure_modes: details,
                        evidence_days: days_sorted,
                        rationale:
                            "observed repeated new failure mode across ≥ 2 distinct days"
                                .to_string(),
                    });
                }
            }
            ObservationKind::KeyDied => {
                if !resurrected_pairs.contains(&pair) {
                    out.deprecate.push(CatalogDeprecate {
                        provider,
                        model,
                        evidence_days: days_sorted,
                        rationale:
                            "quarantine fired on ≥ 2 distinct days with no subsequent resurrection"
                                .to_string(),
                    });
                }
            }
            ObservationKind::Resurrected => {
                // Resurrections are handled via `resurrected_pairs`
                // filtering above — no patch emitted.
            }
        }
    }

    // Keep output deterministic for diff-friendly reviews.
    out.add.sort_by(|a, b| {
        (a.provider.as_str(), a.model.as_str()).cmp(&(b.provider.as_str(), b.model.as_str()))
    });
    out.update.sort_by(|a, b| {
        (a.provider.as_str(), a.model.as_str()).cmp(&(b.provider.as_str(), b.model.as_str()))
    });
    out.deprecate.sort_by(|a, b| {
        (a.provider.as_str(), a.model.as_str()).cmp(&(b.provider.as_str(), b.model.as_str()))
    });
    out
}

/// Convenience: load observations, run [`promote_observations`], and
/// atomically write the result to `suggestions_path`. Returns the
/// written suggestions for callers that want to echo them.
pub fn run_daily_promotion(
    recorder: &ObservationRecorder,
    known_pairs: &BTreeSet<(String, String)>,
    suggestions_path: &Path,
) -> std::io::Result<CatalogSuggestions> {
    let observations = recorder.load();
    let suggestions = promote_observations(&observations, known_pairs);
    if let Some(parent) = suggestions_path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let json = serde_json::to_string_pretty(&suggestions)
        .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;
    let tmp = suggestions_path.with_extension("json.tmp");
    std::fs::write(&tmp, json)?;
    std::fs::rename(&tmp, suggestions_path)?;
    Ok(suggestions)
}

fn append_jsonl(path: &Path, event: &ModelObservation) -> std::io::Result<()> {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let line = serde_json::to_string(event)
        .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;
    let mut f = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(path)?;
    writeln!(f, "{line}")?;
    Ok(())
}

fn truncate_detail(s: &str) -> String {
    const MAX: usize = 200;
    if s.len() <= MAX {
        s.to_string()
    } else {
        let mut end = MAX;
        while end > 0 && !s.is_char_boundary(end) {
            end -= 1;
        }
        format!("{}…", &s[..end])
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::{Duration as ChronoDuration, TimeZone};
    use tempfile::tempdir;

    fn mk(
        provider: &str,
        model: &str,
        kind: ObservationKind,
        days_ago: i64,
        detail: Option<&str>,
    ) -> ModelObservation {
        ModelObservation {
            ts: Utc::now() - ChronoDuration::days(days_ago),
            provider: provider.into(),
            model: model.into(),
            kind,
            detail: detail.map(|s| s.to_string()),
        }
    }

    #[test]
    fn single_day_observations_do_not_promote() {
        let obs = vec![mk("p", "m", ObservationKind::FirstSuccess, 0, None)];
        let out = promote_observations(&obs, &BTreeSet::new());
        assert!(out.add.is_empty());
    }

    #[test]
    fn first_success_on_two_days_promotes_when_pair_unknown() {
        let obs = vec![
            mk("p", "m", ObservationKind::FirstSuccess, 0, None),
            mk("p", "m", ObservationKind::FirstSuccess, 1, None),
        ];
        let out = promote_observations(&obs, &BTreeSet::new());
        assert_eq!(out.add.len(), 1);
        assert_eq!(out.add[0].provider, "p");
        assert_eq!(out.add[0].model, "m");
        assert_eq!(out.add[0].evidence_days.len(), 2);
    }

    #[test]
    fn first_success_is_skipped_when_pair_already_known() {
        let obs = vec![
            mk("p", "m", ObservationKind::FirstSuccess, 0, None),
            mk("p", "m", ObservationKind::FirstSuccess, 1, None),
        ];
        let mut known = BTreeSet::new();
        known.insert(("p".to_string(), "m".to_string()));
        let out = promote_observations(&obs, &known);
        assert!(out.add.is_empty());
    }

    #[test]
    fn new_failure_mode_aggregates_unique_details() {
        let obs = vec![
            mk(
                "zai",
                "glm-5-turbo",
                ObservationKind::NewFailureMode,
                0,
                Some("empty_content"),
            ),
            mk(
                "zai",
                "glm-5-turbo",
                ObservationKind::NewFailureMode,
                1,
                Some("empty_content"),
            ),
            mk(
                "zai",
                "glm-5-turbo",
                ObservationKind::NewFailureMode,
                1,
                Some("tool_call_drop"),
            ),
        ];
        let mut known = BTreeSet::new();
        known.insert(("zai".to_string(), "glm-5-turbo".to_string()));
        let out = promote_observations(&obs, &known);
        assert_eq!(out.update.len(), 1);
        let u = &out.update[0];
        assert_eq!(u.known_failure_modes.len(), 2);
        assert!(u.known_failure_modes.iter().any(|d| d == "empty_content"));
        assert!(u.known_failure_modes.iter().any(|d| d == "tool_call_drop"));
    }

    #[test]
    fn key_died_promotes_to_deprecate_unless_resurrected() {
        let obs = vec![
            mk("p", "m", ObservationKind::KeyDied, 2, None),
            mk("p", "m", ObservationKind::KeyDied, 0, None),
        ];
        let out = promote_observations(&obs, &BTreeSet::new());
        assert_eq!(out.deprecate.len(), 1);

        let obs2 = vec![
            mk("p", "m", ObservationKind::KeyDied, 3, None),
            mk("p", "m", ObservationKind::KeyDied, 2, None),
            mk("p", "m", ObservationKind::Resurrected, 1, None),
            mk("p", "m", ObservationKind::Resurrected, 0, None),
        ];
        let out2 = promote_observations(&obs2, &BTreeSet::new());
        assert!(
            out2.deprecate.is_empty(),
            "resurrected pair must not appear in deprecate",
        );
    }

    #[test]
    fn recorder_roundtrip_persists_across_load() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("obs.jsonl");
        let rec = ObservationRecorder::new(path.clone());
        rec.record("p", "m", ObservationKind::FirstSuccess, None);
        rec.record(
            "p",
            "m",
            ObservationKind::NewFailureMode,
            Some("empty_content".into()),
        );
        let all = rec.load();
        assert_eq!(all.len(), 2);
        assert_eq!(all[0].provider, "p");
        assert_eq!(all[1].detail.as_deref(), Some("empty_content"));
    }

    #[test]
    fn run_daily_promotion_writes_suggestions_json() {
        let dir = tempdir().unwrap();
        let log = dir.path().join("obs.jsonl");
        let patch = dir.path().join("suggest.json");
        let rec = ObservationRecorder::new(log);
        // Two distinct days of FirstSuccess for an unknown pair.
        let ev1 = ModelObservation {
            ts: Utc.with_ymd_and_hms(2026, 4, 10, 10, 0, 0).unwrap(),
            provider: "anthropic".into(),
            model: "claude-new".into(),
            kind: ObservationKind::FirstSuccess,
            detail: None,
        };
        let ev2 = ModelObservation {
            ts: Utc.with_ymd_and_hms(2026, 4, 11, 10, 0, 0).unwrap(),
            provider: "anthropic".into(),
            model: "claude-new".into(),
            kind: ObservationKind::FirstSuccess,
            detail: None,
        };
        append_jsonl(rec.path(), &ev1).unwrap();
        append_jsonl(rec.path(), &ev2).unwrap();
        let out = run_daily_promotion(&rec, &BTreeSet::new(), &patch).unwrap();
        assert_eq!(out.add.len(), 1);
        let written = std::fs::read_to_string(&patch).unwrap();
        assert!(written.contains("\"add\""));
        assert!(written.contains("claude-new"));
    }

    #[test]
    fn promote_is_deterministic_sort_order() {
        let obs = vec![
            mk("b", "m", ObservationKind::FirstSuccess, 0, None),
            mk("b", "m", ObservationKind::FirstSuccess, 1, None),
            mk("a", "m", ObservationKind::FirstSuccess, 0, None),
            mk("a", "m", ObservationKind::FirstSuccess, 1, None),
        ];
        let out = promote_observations(&obs, &BTreeSet::new());
        assert_eq!(out.add.len(), 2);
        assert_eq!(out.add[0].provider, "a");
        assert_eq!(out.add[1].provider, "b");
    }

    #[test]
    fn truncate_detail_respects_char_boundaries() {
        let big = "ü".repeat(300);
        let t = truncate_detail(&big);
        assert!(t.ends_with("…"));
        assert!(t.len() <= 200 + "…".len() + 2);
    }
}
