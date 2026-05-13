//! Data model for the research subsystem.
//!
//! Two principal records:
//! - `ResearchSpec` — owned by the user (topic + sources + schedule). Persisted
//!   once at creation, mutated only via explicit `/research` commands.
//! - `Finding` — emitted by the agent via the `research_save` tool during a run.
//!   Append-only in `findings.jsonl`, deduplicated by `dedup_hash`.
//!
//! Plus: `RunRecord` (one per `coordinator.run_once`), `Cursor` (arbitrary
//! agent-owned JSON to resume between runs, e.g. page number on batdongsan).

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

/// User-owned definition of a research task.
///
/// Field rules:
/// - `id` — slug derived from `topic` + 8-char ULID/UUID suffix. URL-safe so it
///   can be used verbatim as a systemd template instance (`naked-research@ID.timer`).
/// - `sources` — seed URLs the agent is biased to start from. Agent may follow
///   links off-site; this is a hint, not a hard allow-list.
/// - `session_id` — the Telegram session this research was born from. Future
///   `/research ask` replies are posted back into this session so the answer
///   flows in the chat where the user asked, not a fresh thread.
///
/// ### Scheduling fields (priority, highest first)
/// 1. `paused` — short-circuits everything: a paused spec is never due.
/// 2. `run_at` — one-shot, fires once when `now >= run_at`. After firing the
///    scheduler clears it (writes `run_at = None` back to disk) so re-launches
///    require an explicit reschedule.
/// 3. `cron` — recurring, standard 5-field cron (`"min hour dom mon dow"`).
///    Min granularity is 60 s — for tighter schedules use `interval_seconds`.
/// 4. `interval_seconds` — legacy "every N seconds" trigger. Kept verbatim for
///    backward-compat with all existing JSONL specs; new specs should prefer
///    `cron`. The four triggers are evaluated in the order above; the first
///    one that says "due" wins.
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq)]
pub struct ResearchSpec {
    pub id: String,
    pub topic: String,
    #[serde(default)]
    pub sources: Vec<String>,
    /// Legacy "every N seconds" trigger. Prefer `cron` for new specs.
    #[serde(default)]
    pub interval_seconds: Option<u64>,
    /// One-shot launch: scheduler fires the spec once when `now >= run_at`,
    /// then clears the field. Survives restarts (persisted on disk).
    #[serde(default)]
    pub run_at: Option<DateTime<Utc>>,
    /// Recurring schedule in standard 5-field cron form. Examples:
    /// `"0 9 * * MON"` — every Monday at 09:00 UTC.
    /// Validation happens at parse time; an invalid expression makes the spec
    /// silently ineligible (logged once at warn level).
    #[serde(default)]
    pub cron: Option<String>,
    /// Per-spec wall-clock cap on a single scheduler-launched run. Falls back
    /// to `SchedulerConfig.task_timeout` when `None`.
    #[serde(default)]
    pub task_timeout_seconds: Option<u64>,
    #[serde(default)]
    pub session_id: Option<String>,
    /// Telegram chat id to post progress/delta to. If `None`, results are only
    /// written to disk — the systemd runner then has no one to announce to.
    #[serde(default)]
    pub chat_id: Option<i64>,
    /// Optional Telegram thread id for topic-based groups.
    #[serde(default)]
    pub thread_id: Option<i32>,
    /// Provider override for this research only. Falls back to
    /// `Config.research.provider` → `Config.default_provider`.
    #[serde(default)]
    pub provider: Option<String>,
    /// Model override for this research only.
    #[serde(default)]
    pub model: Option<String>,
    /// Per-run safety bound on tool navigations/iterations. Defaults applied
    /// by the coordinator if `None`.
    #[serde(default)]
    pub max_iterations: Option<u32>,
    /// Wall-clock budget for a single run (seconds). Defaults applied by the
    /// coordinator if `None`.
    #[serde(default)]
    pub max_wall_seconds: Option<u64>,
    pub created_at: DateTime<Utc>,
    /// Paused specs are skipped by `naked research run` and `/research run`
    /// but preserve their findings and cursor. Cleared by `/research resume`.
    #[serde(default)]
    pub paused: bool,
    /// Optional human-readable reason the spec is paused. Distinguishes a
    /// manual `/research pause` ("user pause") from an auto-pause hit by
    /// the scheduler after the configured failure threshold ("auto: 5
    /// consecutive failures — last error: ..."). Surfaced in `/research
    /// ls` and `/research state` so an operator can tell at a glance why
    /// a spec is silent. `None` for legacy specs predating this field
    /// and for manually-paused specs that were paused before this code
    /// landed; `Some` only when the current pause was set with a reason.
    /// Cleared back to `None` when `paused` flips to `false`.
    #[serde(default)]
    pub pause_reason: Option<String>,
}

/// A single deduplicated artefact discovered during a run.
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq)]
pub struct Finding {
    pub id: String,
    pub research_id: String,
    pub run_id: String,
    pub url: String,
    pub title: Option<String>,
    /// All actionable details extracted from the page: contacts, area, floor,
    /// conditions, amenities, neighbourhood — everything that makes this
    /// finding useful without revisiting the URL.
    pub excerpt: Option<String>,
    pub price: Option<String>,
    /// Publication / update date of the listing as seen on the page, free-form
    /// string (e.g. "2026-04-15", "15/04/2026", "hôm nay"). `None` when the
    /// page doesn't show a date.
    #[serde(default)]
    pub listing_date: Option<String>,
    /// Condensed text of the source page — enough to verify claims in `excerpt`
    /// without re-fetching the URL. Stripped of navigation/ads boilerplate.
    #[serde(default)]
    pub source_content: Option<String>,
    /// `blake3(canonicalize_url(url))` hex — stable across reruns so the store
    /// can reject duplicates atomically on append.
    pub dedup_hash: String,
    /// `blake3(host + path)` hex — secondary dedup key that ignores ALL query
    /// parameters and fragments, not just tracking ones. Catches the common
    /// case where the same listing is reached via different sort/page/filter
    /// query strings (e.g. `?sort=newest` vs `?sort=price`). Stored alongside
    /// `dedup_hash` so the store can warn on (or reject) re-saves where the
    /// canonical URL differs but the host+path is identical.
    #[serde(default)]
    pub host_path_hash: String,
    /// `blake3` of a normalized excerpt (digits/whitespace collapsed,
    /// lowercased) — collision-resistant content fingerprint that catches
    /// the same listing crossposted to different domains (e.g. Facebook +
    /// batdongsan). Empty when no excerpt was captured.
    #[serde(default)]
    pub content_hash: String,
    pub seen_at: DateTime<Utc>,
}

/// Audit trail for a single `coordinator.run_once` invocation. One row per run
/// is appended to `runs.jsonl` regardless of outcome.
///
/// The verification block (`verification_rounds`, `dead_removed`,
/// `replacements_found`, `remaining_issues`) is populated only when the run
/// went through `run_verified` (gatekeeper loop). Plain `run_once` rows leave
/// these `None`. All verification fields are `#[serde(default)]` so legacy
/// JSONL written before the field set existed loads unchanged.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct RunRecord {
    pub run_id: String,
    pub spec_id: String,
    pub started_at: DateTime<Utc>,
    pub finished_at: DateTime<Utc>,
    pub new_findings: u32,
    pub total_findings_after: u32,
    /// Coordinator-reported termination reason. Human-readable string, not
    /// an enum, because new reasons appear often (timeout, tool budget, api
    /// key exhaustion, etc.) and we don't want to version the JSONL on each.
    pub stop_reason: String,
    pub provider: String,
    pub model: String,
    /// Number of gatekeeper verification rounds executed. `None` for unverified runs.
    #[serde(default)]
    pub verification_rounds: Option<u32>,
    /// Findings removed by the gatekeeper (dead URLs, stale, dupes). `None` for unverified.
    #[serde(default)]
    pub dead_removed: Option<u32>,
    /// New findings collected by feedback re-runs after the first pass. `None` for unverified.
    #[serde(default)]
    pub replacements_found: Option<u32>,
    /// Total quality issues still flagged when the loop accepted the result. `None` for unverified.
    #[serde(default)]
    pub remaining_issues: Option<u32>,
    /// Wall-clock duration of the whole run (incl. verification rounds when applicable).
    #[serde(default)]
    pub elapsed_secs: Option<u64>,
}

/// Free-form JSON the agent may write via `research_save` (under a `cursor`
/// key) or the coordinator may pre-fill between runs. Typical contents:
/// `{ "batdongsan_page": 5, "last_post_id": "abc123" }`.
///
/// Kept as opaque `serde_json::Value` intentionally — every site needs a
/// different shape, and typing this eagerly would force a coupled enum.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct Cursor {
    #[serde(flatten)]
    pub data: serde_json::Map<String, serde_json::Value>,
    pub updated_at: Option<DateTime<Utc>>,
}

/// Canonicalize a URL for dedup purposes:
/// - Lowercase the scheme + host.
/// - Drop fragments (`#anything`).
/// - Drop tracking query parameters (`utm_*`, `gclid`, `fbclid`, `mc_cid`, `igshid`).
/// - Strip a single trailing slash on the path (but keep `/` as root).
/// - Preserve the remaining query parameters, but sort them for stability.
///
/// Falls back to the lowercased original on parse errors — the goal is "stable
/// string", not "RFC-perfect normalization".
pub fn canonicalize_url(url: &str) -> String {
    let trimmed = url.trim();
    let (scheme, rest) = match trimmed.split_once("://") {
        Some((s, r)) => (s.to_ascii_lowercase(), r),
        None => return trimmed.to_ascii_lowercase(),
    };
    let (authority_path, _frag) = rest.split_once('#').unwrap_or((rest, ""));
    let (authority_path, query) = authority_path
        .split_once('?')
        .map(|(ap, q)| (ap, Some(q)))
        .unwrap_or((authority_path, None));

    let (host, path) = authority_path
        .split_once('/')
        .map(|(h, p)| (h.to_ascii_lowercase(), format!("/{p}")))
        .unwrap_or((authority_path.to_ascii_lowercase(), String::from("/")));

    let path = if path.len() > 1 {
        path.trim_end_matches('/').to_string()
    } else {
        path
    };

    let mut result = format!("{scheme}://{host}{path}");
    if let Some(q) = query {
        let mut pairs: Vec<(&str, &str)> = q
            .split('&')
            .filter_map(|kv| {
                let (k, v) = kv.split_once('=').unwrap_or((kv, ""));
                if is_tracking_param(k) {
                    None
                } else {
                    Some((k, v))
                }
            })
            .collect();
        pairs.sort_by_key(|(k, _)| k.to_ascii_lowercase());
        if !pairs.is_empty() {
            result.push('?');
            for (i, (k, v)) in pairs.iter().enumerate() {
                if i > 0 {
                    result.push('&');
                }
                result.push_str(k);
                if !v.is_empty() {
                    result.push('=');
                    result.push_str(v);
                }
            }
        }
    }
    result
}

fn is_tracking_param(k: &str) -> bool {
    let k = k.to_ascii_lowercase();
    k.starts_with("utm_")
        || matches!(
            k.as_str(),
            "gclid" | "fbclid" | "mc_cid" | "mc_eid" | "igshid" | "yclid" | "dclid" | "msclkid"
        )
}

/// Hash used for dedup. Wraps blake3 so the caller doesn't need to depend on
/// the hash crate directly — we may swap algorithms later without touching
/// every call site.
pub fn dedup_hash(url: &str) -> String {
    let canon = canonicalize_url(url);
    blake3::hash(canon.as_bytes()).to_hex().to_string()
}

/// Secondary dedup key that ignores ALL query parameters / fragments,
/// not just the tracking subset that `canonicalize_url` strips. Two URLs
/// reaching the same listing through different sort/filter/page params
/// produce different `dedup_hash` values but the **same** `host_path_hash`,
/// so the store can warn about (or reject) such near-duplicates.
///
/// Falls back to the lowercased input on parse errors — same philosophy
/// as `canonicalize_url`.
pub fn host_path_hash(url: &str) -> String {
    let trimmed = url.trim();
    let (_scheme, rest) = match trimmed.split_once("://") {
        Some(v) => v,
        None => {
            return blake3::hash(trimmed.to_ascii_lowercase().as_bytes())
                .to_hex()
                .to_string();
        }
    };
    let (authority_path, _) = rest.split_once('#').unwrap_or((rest, ""));
    let (authority_path, _) = authority_path
        .split_once('?')
        .unwrap_or((authority_path, ""));
    let (host, path) = authority_path
        .split_once('/')
        .map(|(h, p)| (h.to_ascii_lowercase(), format!("/{p}")))
        .unwrap_or((authority_path.to_ascii_lowercase(), String::from("/")));
    let path = if path.len() > 1 {
        path.trim_end_matches('/').to_string()
    } else {
        path
    };
    blake3::hash(format!("{host}{path}").as_bytes())
        .to_hex()
        .to_string()
}

/// Content-based dedup key for catching the same listing crossposted to
/// different domains (e.g. Facebook + batdongsan + a Telegram channel).
/// Normalizes `excerpt`/`source_content` to lowercase, collapses runs of
/// whitespace, and drops digits before hashing — the digits are usually
/// the price / phone number which can vary on republish even when the
/// prose is verbatim. Returns an empty string for empty input so the
/// store can treat "no content" as "skip the content check".
pub fn content_hash(excerpt: &str) -> String {
    let trimmed = excerpt.trim();
    if trimmed.is_empty() {
        return String::new();
    }
    let normalized: String = trimmed
        .chars()
        .map(|c| {
            if c.is_alphabetic() {
                c.to_lowercase().next().unwrap_or(c)
            } else {
                ' '
            }
        })
        .collect();
    let collapsed: String = normalized.split_whitespace().collect::<Vec<_>>().join(" ");
    if collapsed.is_empty() {
        return String::new();
    }
    blake3::hash(collapsed.as_bytes()).to_hex().to_string()
}

/// Normalize a listing title for **soft / fuzzy** deduplication.
///
/// ⚠ Not the same as
/// [`crate::research::reconciler::normalize_title`]. This function
/// is **stem-aggressive** (drops short digit tokens, drops the
/// `m2`/`sqm`/`tầng`/`pn`/etc. stopword set, no quote/dash
/// translation) — used **only** by [`titles_are_similar`] for
/// Jaccard-similarity matching between near-duplicate listings.
/// For strict-equality dedup of B5 reconciler use
/// `reconciler::normalize_title` instead.
///
/// Removes:
/// - leading/trailing whitespace
/// - common area/floor tokens that vary between reposts of the same property
///   (m², sqm, tầng, floor, bedroom count digits)
/// - non-alphanumeric characters that function as separators
/// - runs of whitespace
///
/// The output is lowercased ASCII-folded Unicode text, so "70m² có bếp+PN"
/// and "70m² căn hộ 2PN" both normalize closer to each other than to a
/// completely different property. Not perfect — just good enough to catch
/// obvious same-property reposts with slightly different titles.
pub(crate) fn normalize_title_for_similarity(title: &str) -> String {
    // 1. Unicode to ASCII-ish via char-by-char lowercasing (no external crate).
    //    We keep only letters, digits, and spaces — everything else becomes a space.
    let normalized: String = title
        .chars()
        .map(|c| {
            if c.is_alphanumeric() {
                // Fold to lowercase; non-ASCII letters kept as-is for now.
                c.to_lowercase().next().unwrap_or(c)
            } else {
                ' '
            }
        })
        .collect();

    // 2. Remove pure-digit tokens ≤ 4 chars (prices, m², floor numbers change
    //    between reposts; longer numbers like apartment IDs are more stable).
    let tokens: Vec<&str> = normalized
        .split_whitespace()
        .filter(|tok| {
            if tok.chars().all(|c| c.is_ascii_digit()) {
                tok.len() > 4
            } else {
                true
            }
        })
        .collect();

    // 3. Remove very common Vietnamese/English area/unit tokens that add noise.
    let stopwords: &[&str] = &[
        "m2", "sqm", "tang", "floor", "phong", "pn", "wc", "toilet", "bedroom", "bdrm",
    ];
    let tokens: Vec<&&str> = tokens.iter().filter(|t| !stopwords.contains(t)).collect();

    tokens
        .iter()
        .map(|t| t.as_ref())
        .collect::<Vec<_>>()
        .join(" ")
}

/// Returns `true` when two titles are "soft-duplicate" — i.e. they likely
/// describe the same physical property even if worded differently.
///
/// Algorithm: token-Jaccard similarity on the normalized token sets.
/// Threshold = 0.65. That's conservative enough to avoid false positives
/// for distinct apartments on the same floor/building while catching the
/// same listing posted twice with slightly different wording.
///
/// Used by the store to warn about near-duplicates even when the URL differs.
pub(crate) fn titles_are_similar(a: &str, b: &str) -> bool {
    let a_norm = normalize_title_for_similarity(a);
    let b_norm = normalize_title_for_similarity(b);
    let a_tokens: std::collections::HashSet<&str> = a_norm.split_whitespace().collect();
    let b_tokens: std::collections::HashSet<&str> = b_norm.split_whitespace().collect();
    if a_tokens.is_empty() || b_tokens.is_empty() {
        return false;
    }
    let intersection = a_tokens.intersection(&b_tokens).count();
    let union = a_tokens.union(&b_tokens).count();
    if union == 0 {
        return false;
    }
    let jaccard = intersection as f64 / union as f64;
    jaccard >= 0.65
}

/// Slugify a topic into a filesystem-safe prefix, e.g.
/// `"Jaguar XF used cheap"` → `"jaguar-xf-used-cheap"`.
/// Conservative: ASCII-only, `[a-z0-9-]+`, collapses runs of separators.
pub(crate) fn slugify(topic: &str) -> String {
    let mut out = String::with_capacity(topic.len());
    let mut last_was_sep = true;
    for ch in topic.chars() {
        let c = ch.to_ascii_lowercase();
        if c.is_ascii_alphanumeric() {
            out.push(c);
            last_was_sep = false;
        } else if !last_was_sep {
            out.push('-');
            last_was_sep = true;
        }
    }
    let trimmed = out.trim_matches('-').to_string();
    if trimmed.is_empty() {
        "research".to_string()
    } else {
        // Cap slug length to keep filenames reasonable on fs's with 255-byte limits.
        trimmed.chars().take(48).collect()
    }
}

/// Stop-words that don't carry meaning when squeezed into a 2–3 word
/// human-readable id. We strip them in `short_slug` so the id reads like
/// `commercial-realty-a3f2`, not `the-best-list-of-a3f2`.
const SHORT_SLUG_STOP: &[&str] = &[
    // English
    "the", "a", "an", "and", "or", "of", "for", "in", "on", "at", "to", "by", "with", "is", "are",
    "was", "were", "be", "been", "being", "this", "that", "these", "those", "it", "its", "from",
    "as", "but", "if", "then", "than", "so", "into", "over", "under", "near",
    // Number / currency / unit noise common in research topics
    "vnd", "usd", "eur", "rub", "sqm", "m2", "m²", "k", "m", "mln", "bln",
];

/// Build a short, readable slug: 2–3 meaningful words from the topic
/// joined with `-`, each clipped to 12 chars. Numeric-only and short
/// stopword tokens are dropped so a topic like "Da Nang commercial real
/// estate 50–250m² 1500–3000USD" collapses to `danang-commercial-realty`.
///
/// Falls back to whatever `slugify` produces when the topic carries no
/// usable words (e.g. all-numeric or all-stopwords).
fn short_slug(topic: &str) -> String {
    let lowered = topic.to_lowercase();
    let words: Vec<String> = lowered
        .split(|c: char| !c.is_alphanumeric())
        .filter_map(|raw| {
            // Keep only ASCII letters/digits — non-ASCII is dropped because
            // file/URL targets need to stay short and unambiguous.
            let cleaned: String = raw.chars().filter(|c| c.is_ascii_alphanumeric()).collect();
            if cleaned.is_empty() {
                return None;
            }
            // Drop pure-digit tokens (prices, areas) and known stopwords.
            if cleaned.chars().all(|c| c.is_ascii_digit()) {
                return None;
            }
            if SHORT_SLUG_STOP.contains(&cleaned.as_str()) {
                return None;
            }
            // Drop very short non-alphabetic tokens left over from currency
            // / range markers ("k", "m") so they never end up as the only
            // word in the id.
            if cleaned.len() <= 1 {
                return None;
            }
            Some(cleaned.chars().take(12).collect::<String>())
        })
        .collect();

    let mut picked: Vec<String> = Vec::new();
    for w in words {
        if picked.iter().any(|p| p == &w) {
            continue;
        }
        picked.push(w);
        if picked.len() == 3 {
            break;
        }
    }
    if picked.is_empty() {
        return slugify(topic);
    }
    picked.join("-")
}

/// Generate a compact, URL/systemd-safe research id.
///
/// Format: `<2–3 meaningful topic words>-<4 hex>`.
/// Examples:
/// * `"Da Nang commercial real estate 50-250m² 1500-3000USD"` →
///   `danang-commercial-realty-a3f2`
/// * `"Jaguar XF cheap"` → `jaguar-xf-cheap-9b81`
/// * `""` (empty / all noise) → `research-3af2`
///
/// 4 hex characters give 65k unique suffixes per slug, which is plenty
/// for a per-user research store. Collision is handled at the persistence
/// layer (`ResearchStore::save_spec` rejects duplicates) — caller can
/// retry, the id is cheap to regenerate.
pub(crate) fn new_research_id(topic: &str) -> String {
    let slug = short_slug(topic);
    let uuid = uuid::Uuid::new_v4().simple().to_string();
    format!("{slug}-{}", &uuid[..4])
}

#[cfg(test)]
#[path = "spec_tests.rs"]
mod tests;
