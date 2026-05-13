//! Coordinator-wide types: configuration, run reports, gatekeeper verdict, fuzzy fingerprint.

use std::collections::{HashMap, HashSet};
use std::path::PathBuf;
use std::time::Duration;

/// Fuzzy fingerprint of a finding's title + price used by the gatekeeper's
/// second-pass dedup. Two listings are considered duplicates when:
///   * their normalized price digits are identical (after stripping currency
///     symbols, spaces, and thousands separators), and
///   * their title token sets overlap by Jaccard ≥ 0.7 (ignoring common
///     filler words and tokens shorter than 3 chars).
///
/// The exact-match `title||price` signature is still applied first; this only
/// catches near-duplicates the strict pass misses (e.g. one title carrying an
/// extra parenthetical like "có bếp+PN").
#[derive(Debug, Clone)]
pub(crate) struct FuzzyFingerprint {
    tokens: std::collections::BTreeSet<String>,
    price_digits: String,
}

impl FuzzyFingerprint {
    /// Stop-list of low-signal words common in real-estate listings (mostly
    /// Vietnamese and English). Tokens here never count toward the Jaccard
    /// similarity. Kept short on purpose — we'd rather miss a dup than wrongly
    /// merge two distinct listings.
    const FILLER: &'static [&'static str] = &[
        "cho",
        "thuê",
        "thue",
        "rent",
        "for",
        "the",
        "and",
        "căn",
        "can",
        "hộ",
        "ho",
        "apartment",
        "with",
        "near",
        "phòng",
        "phong",
        "trong",
        "tại",
        "tai",
        "có",
        "co",
        "đầy",
        "đủ",
        "day",
        "du",
        "fully",
        "furnished",
        "new",
        "mới",
        "moi",
    ];

    pub(crate) fn new(title: Option<&str>, price: Option<&str>) -> Option<Self> {
        let title = title?.trim();
        if title.len() < 10 {
            return None;
        }
        let lower = title.to_lowercase();
        let tokens: std::collections::BTreeSet<String> = lower
            .split(|c: char| !c.is_alphanumeric())
            .filter(|s| s.chars().count() >= 3 && !Self::FILLER.contains(s))
            .map(str::to_string)
            .collect();
        if tokens.len() < 3 {
            return None;
        }
        let price_digits: String = price
            .unwrap_or("")
            .chars()
            .filter(char::is_ascii_digit)
            .collect();
        Some(Self {
            tokens,
            price_digits,
        })
    }

    fn jaccard(&self, other: &Self) -> f32 {
        let intersection = self.tokens.intersection(&other.tokens).count();
        let union = self.tokens.union(&other.tokens).count();
        if union == 0 {
            0.0
        } else {
            intersection as f32 / union as f32
        }
    }

    pub(crate) fn is_duplicate_of(&self, other: &Self) -> bool {
        // Require both: identical normalized price AND high token overlap.
        // Either alone is too noisy.
        if self.price_digits.is_empty() || other.price_digits.is_empty() {
            return false;
        }
        if self.price_digits != other.price_digits {
            return false;
        }
        self.jaccard(other) >= 0.7
    }
}

/// Split a `"provider/model"` string into `(provider, model)`. Returns
/// `None` when the input doesn't carry a provider prefix — callers fall
/// back to whatever provider is already in scope.
///
/// Used by the research fallback chain so a JSON config can express
/// cross-provider fallbacks like:
///
/// ```json
/// "fallback_models": ["kimi-k2-thinking", "moonshot/kimi-k2-thinking", "qwen/qwen3.6-plus"]
/// ```
///
/// The split point is the FIRST `/`. Everything before it is the provider
/// name, everything after is the model id (model ids may legally contain
/// further slashes, e.g. `openrouter/anthropic/claude-3.5-sonnet`).
///
/// Trims whitespace on both halves and rejects empty halves.
pub fn parse_provider_model_pair(input: &str) -> Option<(String, String)> {
    let (p, m) = input.split_once('/')?;
    let p = p.trim();
    let m = m.trim();
    if p.is_empty() || m.is_empty() {
        return None;
    }
    Some((p.to_string(), m.to_string()))
}

/// Coordinator-wide defaults, folded in from `Config.research`.
#[derive(Debug, Clone)]
pub struct CoordinatorConfig {
    /// Provider override for all research runs. `None` means "use global default".
    pub default_provider: Option<String>,
    /// Model override for all research runs.
    pub default_model: Option<String>,
    /// Fallback models tried in order when the primary model is rejected.
    pub fallback_models: Vec<String>,
    /// Safety ceiling on agent iterations if the spec doesn't specify.
    pub default_max_iterations: u32,
    /// Wall-clock ceiling if the spec doesn't specify.
    pub default_max_wall_seconds: u64,
    /// Workspace root used to materialize the ephemeral research session.
    pub workspace: PathBuf,
    /// Gatekeeper quality gate settings, passed from the JSON config.
    pub gatekeeper: crate::config::GatekeeperConfig,
    /// Reasoning level applied to every research session via
    /// `set_session_reasoning`. `None` ⇒ inherit from session/global default.
    /// Accepts `"off"`, `"low"`, `"medium"`, `"high"`.
    pub reasoning: Option<String>,
    /// Provider map for capability-aware fallback filtering. Populated
    /// from `Config.providers` when the coordinator is built. Keyed by
    /// provider id; value holds the `capabilities` block (along with
    /// the `models` list so the selector can confirm the model is
    /// declared). Populated by [`AgentCore::build_coordinator`]; left
    /// empty in unit tests that don't care about capabilities (the
    /// selector then no-ops under the soft-mode contract).
    pub provider_capabilities: HashMap<String, crate::config::ProviderConfig>,
    /// Mirrors [`crate::config::Config::enforce_model_capabilities`]. When
    /// `false`, the fallback chain is passed through unchanged (Phase 2
    /// ships in soft mode by default). When `true`, entries rejected by
    /// the selector are dropped **before** an HTTP request is spent on
    /// them.
    pub enforce_model_capabilities: bool,
    /// Runtime health tracker (Phase 3). When present, the selector
    /// inside `try_start_with_fallback` consults per-pair quarantine
    /// state so a model that started hallucinating 10 min ago gets
    /// skipped instead of costing another 90 s dispatch.
    pub model_health: Option<std::sync::Arc<crate::model_catalog::ModelHealth>>,
    /// Per-run event registry (waterfall progress for TG). When
    /// present, the coordinator emits iteration/tool events into it
    /// so the TG heartbeat task can render a live summary. `None`
    /// leaves the push sites as no-ops — the CLI / tests neither
    /// need nor want the extra plumbing.
    pub run_events: Option<crate::research::run_events::RunEventRegistry>,
}

impl Default for CoordinatorConfig {
    fn default() -> Self {
        Self {
            default_provider: None,
            default_model: None,
            fallback_models: Vec::new(),
            default_max_iterations: 30,
            default_max_wall_seconds: 1200,
            workspace: std::env::current_dir().unwrap_or_else(|_| PathBuf::from(".")),
            gatekeeper: crate::config::GatekeeperConfig::default(),
            reasoning: None,
            provider_capabilities: HashMap::new(),
            enforce_model_capabilities: false,
            model_health: None,
            run_events: None,
        }
    }
}

/// Why a run ended. Opaque to the store (serialized as a string), but the
/// coordinator produces a fixed set we can reason about in tests.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StopReason {
    /// Agent reached `AgentEvent::Idle` naturally.
    AgentIdle,
    /// Hit wall-clock timeout (max_wall_seconds).
    Timeout,
    /// The agent's event stream closed before `Idle`. Usually an internal
    /// error — we still persist whatever was saved.
    StreamClosed,
    /// Spec was paused before we started.
    Paused,
    /// An error bubbled up from the runner.
    Error,
    /// Cooperative cancellation observed via the scheduler-supplied
    /// [`tokio_util::sync::CancellationToken`]. Distinct from `Timeout`
    /// (which is the coordinator's own wall-clock budget) — `Cancelled`
    /// means an external orchestrator (typically the scheduler's
    /// two-step `cancel → abort` on `task_timeout`) asked us to stop.
    Cancelled,
}

impl StopReason {
    pub fn as_str(self) -> &'static str {
        match self {
            StopReason::AgentIdle => "agent_idle",
            StopReason::Timeout => "timeout",
            StopReason::StreamClosed => "stream_closed",
            StopReason::Paused => "paused",
            StopReason::Error => "error",
            StopReason::Cancelled => "cancelled",
        }
    }
}

/// Summary of one `run_once`. Handed back to the caller for logging / TG
/// delivery; also mirrored into `runs.jsonl` via `RunRecord`.
#[derive(Debug, Clone)]
pub struct RunReport {
    pub spec_id: String,
    pub run_id: String,
    pub new_findings: u32,
    pub total_findings_after: u32,
    pub stop_reason: StopReason,
    pub elapsed: Duration,
    pub provider: String,
    pub model: String,
}

/// Result of a gatekeeper-verified research run.
#[derive(Debug, Clone)]
pub struct VerifiedRunReport {
    pub last_run: RunReport,
    pub verification_rounds: u32,
    pub dead_removed: u32,
    pub replacements_found: u32,
    pub final_findings: u32,
    pub remaining_issues: Vec<String>,
}

/// Gatekeeper verdict after verifying all findings.
#[derive(Debug, Default)]
pub(crate) struct GatekeeperVerdict {
    pub(crate) live_urls: u32,
    pub(crate) dead_urls: u32,
    pub(crate) missing_fields: u32,
    pub(crate) stale_dates: u32,
    /// Findings missing listing_date (None or "unknown").
    pub(crate) missing_dates: u32,
    /// Findings missing source_content entirely.
    pub(crate) missing_source_content: u32,
    /// Findings with excerpt < 200 chars (too short to be actionable).
    pub(crate) short_excerpts: u32,
    /// Semantic duplicates (same title+price at different URLs).
    pub(crate) semantic_dupes: u32,
    pub(crate) dead_hashes: HashSet<String>,
    pub(crate) dead_details: Vec<DeadFinding>,
    /// URLs of findings that need quality remediation (re-fetch for date/source).
    pub(crate) remediation_urls: Vec<String>,
    pub(crate) issues: Vec<String>,
}

#[derive(Debug)]
pub(crate) struct DeadFinding {
    pub(crate) url: String,
    pub(crate) title: String,
}

/// Verification block written into a [`RunRecord`] when emitted by `run_verified`.
#[derive(Debug, Clone, Copy)]
pub(crate) struct VerificationSummary {
    pub rounds: u32,
    pub dead_removed: u32,
    pub replacements_found: u32,
    pub remaining_issues: u32,
}
