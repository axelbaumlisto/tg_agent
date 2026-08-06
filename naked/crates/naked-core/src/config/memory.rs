//! Memory-enrichment configuration.

use serde::{Deserialize, Serialize};

/// Configuration for the lightweight memory-enrichment subsystem.
///
/// Two-tier storage: `MEMORY.md` holds active rules (shipped to every
/// system prompt), and per-day `memory/YYYY-MM-DD.md` files act as
/// short-lived drafts. A daily digest job merges yesterday's drafts
/// into `MEMORY.md` (LLM-summarized + scoring-based promotion), and
/// rejected candidates land in `DREAMS.md` for human review.
///
/// All fields are optional; defaults are tuned for "drop-in, no
/// config changes required" operation. Set `daily_enabled=false` to
/// disable the background job entirely.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct MemoryConfig {
    /// Master switch. `false` = no daily digest, no pre-compaction
    /// flush, no session-close summaries. The synchronous read APIs
    /// (`load_rules_for`, etc.) keep working unchanged.
    #[serde(default = "default_true")]
    pub daily_enabled: bool,
    /// Cron expression for the daily digest job. Defaults to `"0 4 * * *"`
    /// — 04:00 UTC, when most users are asleep.
    #[serde(default = "default_daily_cron")]
    pub daily_cron: String,
    /// What the digest does:
    /// * `"summarize_only"` — write a summary into `DREAMS.md`, never
    ///   touch `MEMORY.md`. Useful for inspecting the system before
    ///   trusting it with promotions.
    /// * `"summarize_and_promote"` (default) — also promote scoring
    ///   winners into `MEMORY.md`.
    #[serde(default = "default_daily_mode")]
    pub daily_mode: String,
    /// Provider/model used for digest LLM calls. `"<provider>/<model>"`
    /// or just `"<model>"` to use the default provider. `None` → use
    /// the global default. Pick a cheap model — the digest only sees
    /// short markdown lines.
    #[serde(default)]
    pub digest_provider: Option<String>,
    /// Maximum chars passed to the digest LLM in one call. Larger
    /// daily files are tail-truncated. Default: 8 KB.
    #[serde(default = "default_digest_max_chars")]
    pub digest_max_chars: usize,
    /// Maximum chars of the "Recent shift" block injected into every
    /// new turn's system prompt. Larger blobs are tail-truncated.
    #[serde(default = "default_recent_shift_max_chars")]
    pub recent_shift_max_chars: usize,
    /// How many days back the "Recent shift" block looks at. Default:
    /// 2 (today + yesterday).
    #[serde(default = "default_recent_shift_days")]
    pub recent_shift_days: u32,
    /// How many days of `memory/YYYY-MM-DD.md` files to keep on disk
    /// before the digest job rotates them out. Default: 30.
    #[serde(default = "default_daily_retention_days")]
    pub daily_retention_days: u32,
    /// How many days of `DREAMS.md` entries to keep before rotation.
    /// Default: 90.
    #[serde(default = "default_dreams_retention_days")]
    pub dreams_retention_days: u32,
    /// When `true`, `Session::reset/new` triggers a fire-and-forget
    /// summary into the current day's draft file.
    #[serde(default = "default_true")]
    pub session_close_summary: bool,
    /// Idle minutes before a session is treated as "closed" by the
    /// background sweep. `0` disables idle close. Default: 60.
    #[serde(default = "default_session_idle_close_minutes")]
    pub session_idle_close_minutes: u32,
    /// When `true`, `MemoryService::store` skips writes whose content
    /// hash already exists in the same scope. Default: `true`.
    #[serde(default = "default_true")]
    pub dedup_on_store: bool,
    /// When `true`, the conversation-history compactor performs a
    /// "silent turn" before sending the summary prompt to the LLM,
    /// asking it to flush rules-of-thumb / corrections to that day's
    /// draft file. Default: `true`.
    #[serde(default = "default_true")]
    pub pre_compaction_flush: bool,
    /// When `true`, the per-message memory classifier writes hits into
    /// today's daily draft file instead of `MEMORY.md`. The daily digest
    /// then decides whether to promote them based on
    /// `promote_min_repeat_days`. This prevents one-off mis-classifications
    /// from polluting durable rules. Default: `true`.
    ///
    /// Set to `false` to restore legacy behaviour (classifier writes
    /// straight to `MEMORY.md` — fast feedback, no scoring gate).
    #[serde(default = "default_true")]
    pub auto_classify_to_drafts: bool,
    /// "Forgetting" via compaction: when a `MEMORY.md` section grows
    /// past this many entries, the digest LLM is asked to merge the
    /// `compact_window` oldest items into a single summary line. The
    /// merged entry replaces them, with `source:compaction` and a
    /// `merged_from:` metadata trail. `0` disables compaction (default
    /// `12` — generous so small projects never hit it).
    #[serde(default = "default_compact_threshold")]
    pub compact_threshold_per_section: u32,
    /// How many of the oldest entries to roll into one when compaction
    /// fires. The current section must contain at least
    /// `compact_threshold_per_section` items. Default `5`.
    #[serde(default = "default_compact_window")]
    pub compact_window: u32,
    /// Hard upper bound on the LLM-generated merge summary. Anything
    /// longer is truncated server-side. Keep small so MEMORY.md never
    /// grows back faster than it shrinks. Default 240 chars.
    #[serde(default = "default_compact_max_chars")]
    pub compact_max_chars: usize,
    /// Don't compact entries younger than this many days — fresh rules
    /// are still earning their keep. Default 14.
    #[serde(default = "default_compact_min_age_days")]
    pub compact_min_age_days: u32,
    /// Spare entries from compaction if their **effective** recall
    /// score is at least this. The effective score applies an
    /// exponential half-life decay over `last_recalled_at` (see
    /// `compact_recall_half_life_days`) so a memory recalled 3 times
    /// last week is more protected than one recalled 3 times a year
    /// ago. Recently-used rules survive even when old. Default `2`.
    #[serde(default = "default_compact_spare_recall")]
    pub compact_spare_recall: u32,
    /// Half-life (in days) for the exponential decay applied to
    /// `recall_count` during compaction scoring. Operationally:
    /// `effective_recall = recall_count * 0.5^(age_in_days / half_life)`
    /// where `age_in_days = today - last_recalled_at` (or `today -
    /// created_at` if the entry was never recalled). `0` disables the
    /// decay entirely (back to the legacy "raw `recall_count`"
    /// behaviour). Default `30` days — a memory recalled once today
    /// fully counts; one recalled 30 days ago contributes 0.5 of its
    /// recall count; one recalled 90 days ago contributes 0.125.
    #[serde(default = "default_compact_recall_half_life_days")]
    pub compact_recall_half_life_days: u32,
    /// Promotion gate: a draft entry must appear on at least this many
    /// distinct daily files before it becomes a promotion candidate.
    /// Default: `2` (i.e. it stuck around for ≥2 days).
    #[serde(default = "default_promote_min_repeat_days")]
    pub promote_min_repeat_days: u32,
    /// Promotion gate: a draft entry must have been recalled (matched
    /// during context injection) at least this many times. `0` disables
    /// the gate (recall counts ignored). Default: `0` — recall
    /// tracking is best-effort and we don't want to block promotion
    /// on a rarely-instrumented signal.
    #[serde(default = "default_promote_min_recall_count")]
    pub promote_min_recall_count: u32,
    /// Default-off rollout flag for the S3 re-observation promotion road.
    /// When disabled, persisted re-observation counts are surfaced in
    /// `ScoringHints` but do not affect the reinforcement gate.
    #[serde(default)]
    pub memory_reobservation_promote_enabled: bool,
    /// Default-off rollout flag for S5 scope-priority prompt injection.
    /// When disabled, injection preserves the legacy Global → Project → User
    /// renderer exactly; when enabled, User → Project → Global gets the fixed
    /// budget first and truncation is logged/counted.
    #[serde(default)]
    pub memory_scope_priority_injection_enabled: bool,
    /// Minimum persisted re-observations before they are allowed to
    /// influence promotion. Default is deliberately 2 (not 1): the
    /// re-observation road is intentionally conservative even after S6
    /// made the shared dedup identity order-preserving.
    #[serde(default = "default_promote_min_reobservations")]
    pub promote_min_reobservations: u32,
}

impl Default for MemoryConfig {
    fn default() -> Self {
        Self {
            daily_enabled: true,
            daily_cron: default_daily_cron(),
            daily_mode: default_daily_mode(),
            digest_provider: None,
            digest_max_chars: default_digest_max_chars(),
            recent_shift_max_chars: default_recent_shift_max_chars(),
            recent_shift_days: default_recent_shift_days(),
            daily_retention_days: default_daily_retention_days(),
            dreams_retention_days: default_dreams_retention_days(),
            session_close_summary: true,
            session_idle_close_minutes: default_session_idle_close_minutes(),
            dedup_on_store: true,
            pre_compaction_flush: true,
            auto_classify_to_drafts: true,
            compact_threshold_per_section: default_compact_threshold(),
            compact_window: default_compact_window(),
            compact_max_chars: default_compact_max_chars(),
            compact_min_age_days: default_compact_min_age_days(),
            compact_spare_recall: default_compact_spare_recall(),
            compact_recall_half_life_days: default_compact_recall_half_life_days(),
            promote_min_repeat_days: default_promote_min_repeat_days(),
            promote_min_recall_count: default_promote_min_recall_count(),
            memory_reobservation_promote_enabled: false,
            memory_scope_priority_injection_enabled: false,
            promote_min_reobservations: default_promote_min_reobservations(),
        }
    }
}

fn default_daily_cron() -> String {
    "0 4 * * *".to_string()
}
fn default_daily_mode() -> String {
    "summarize_and_promote".to_string()
}
fn default_digest_max_chars() -> usize {
    8000
}
fn default_recent_shift_max_chars() -> usize {
    1500
}
fn default_recent_shift_days() -> u32 {
    7
}
fn default_daily_retention_days() -> u32 {
    30
}
fn default_dreams_retention_days() -> u32 {
    90
}
fn default_session_idle_close_minutes() -> u32 {
    60
}
fn default_promote_min_repeat_days() -> u32 {
    2
}
fn default_promote_min_recall_count() -> u32 {
    0
}
fn default_promote_min_reobservations() -> u32 {
    2
}
fn default_compact_threshold() -> u32 {
    12
}
fn default_compact_window() -> u32 {
    5
}
fn default_compact_max_chars() -> usize {
    240
}
fn default_compact_min_age_days() -> u32 {
    14
}
fn default_compact_spare_recall() -> u32 {
    2
}
fn default_compact_recall_half_life_days() -> u32 {
    30
}

fn default_true() -> bool {
    true
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_memory_config_sane() {
        let cfg = MemoryConfig::default();
        assert!(cfg.daily_enabled);
        assert!(cfg.digest_max_chars > 0);
        assert!(!cfg.memory_reobservation_promote_enabled);
        assert!(!cfg.memory_scope_priority_injection_enabled);
        assert_eq!(cfg.promote_min_reobservations, 2);
    }

    #[test]
    fn deserialize_empty_json() {
        let cfg: MemoryConfig = serde_json::from_str("{}").unwrap();
        assert!(cfg.daily_enabled);
        assert!(!cfg.memory_reobservation_promote_enabled);
        assert!(!cfg.memory_scope_priority_injection_enabled);
        assert_eq!(cfg.promote_min_reobservations, 2);
    }

    #[test]
    fn deserialize_override() {
        let cfg: MemoryConfig = serde_json::from_str(
            r#"{"daily_enabled": false, "digest_max_chars": 10, "memory_reobservation_promote_enabled": true, "memory_scope_priority_injection_enabled": true, "promote_min_reobservations": 3}"#,
        )
        .unwrap();
        assert!(!cfg.daily_enabled);
        assert_eq!(cfg.digest_max_chars, 10);
        assert!(cfg.memory_reobservation_promote_enabled);
        assert!(cfg.memory_scope_priority_injection_enabled);
        assert_eq!(cfg.promote_min_reobservations, 3);

        let encoded = serde_json::to_string(&cfg).unwrap();
        let roundtrip: MemoryConfig = serde_json::from_str(&encoded).unwrap();
        assert!(roundtrip.memory_reobservation_promote_enabled);
        assert!(roundtrip.memory_scope_priority_injection_enabled);
        assert_eq!(roundtrip.promote_min_reobservations, 3);
    }
}
