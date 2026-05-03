//! Research and gatekeeper configuration.

use std::path::PathBuf;

use serde::{Deserialize, Serialize};

fn default_true() -> bool {
    true
}
fn default_research_enabled() -> bool {
    true
}
fn default_research_max_iterations() -> u32 {
    30
}
fn default_research_max_wall_seconds() -> u64 {
    1200
}
fn default_schedule_interval() -> u64 {
    21600
}
fn default_task_timeout_seconds() -> u64 {
    1800
}
fn default_max_retries_before_alert() -> u32 {
    3
}

/// Configuration for the research subsystem. Defaults are safe but conservative:
/// 30 navigations per run, 20-minute wall-clock cap. Uses the same model family
/// as the main assistant (Qwen3.6-Plus by default) for best tool-use quality;
/// override with a cheaper model via `research.model` if needed.
///
/// Every field is optional. When the whole section is absent from JSON, the
/// default is "enabled, global provider/model, 30 navs, 1200 s". Unknown
/// `provider` or a `model` not listed under its provider surfaces as a
/// `tracing::warn!` via `Config::validate_and_warn` — not a hard error, so
/// existing configs keep booting unchanged.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ResearchConfig {
    /// When `false`, `/research *` commands return "research disabled" and
    /// the background runner is never spawned. Default: `true`.
    #[serde(default = "default_research_enabled")]
    pub enabled: bool,
    /// Override provider for research turns. `None` → `default_provider`.
    #[serde(default)]
    pub provider: Option<String>,
    /// Override model for research turns. `None` → `default_model`.
    #[serde(default)]
    pub model: Option<String>,
    /// Fallback models tried in order when the primary model is rejected by
    /// all provider keys (HTTP 400 "model not supported"). Empty = no fallback.
    #[serde(default)]
    pub fallback_models: Vec<String>,
    /// Max agent iterations per run. Each iteration = one LLM ↔ tool loop.
    #[serde(default = "default_research_max_iterations")]
    pub max_iterations: u32,
    /// Wall-clock budget per run, in seconds. Prevents runaway runs when a
    /// site hangs or a tool loops. Default: 1200 (20 min).
    #[serde(default = "default_research_max_wall_seconds")]
    pub max_wall_seconds: u64,
    /// Optional seed sources to pre-populate newly created specs. E.g.
    /// `["https://www.chotot.com/mua-ban-oto"]` for VN car research.
    #[serde(default)]
    pub default_sources: Vec<String>,
    /// Storage root. `None` → `$NAKED_HOME/research` → `~/.naked/research`.
    #[serde(default)]
    pub storage_dir: Option<PathBuf>,
    /// Tools the research agent is allowed to use during a run. `None` →
    /// registry default. Useful to force a tight loop that only uses
    /// `web_fetch` + `web_search` + `research_save`, forbidding shell.
    #[serde(default)]
    pub allowed_tools: Option<Vec<String>>,
    /// Allow `/research schedule` to manage systemd timers. Default: `true`.
    #[serde(default = "default_true")]
    pub schedule_enabled: bool,
    /// Default interval for scheduled runs when the spec doesn't override.
    /// Applied to new specs at creation time. Default: 21600 (6 hours).
    /// Set to 0 to create specs without a schedule (manual only).
    #[serde(default = "default_schedule_interval")]
    pub default_interval_seconds: u64,
    /// Default cron expression for new specs. Takes priority over
    /// `default_interval_seconds` when set. Example: `"0 10 * * *"`.
    #[serde(default)]
    pub default_cron: Option<String>,
    /// Trigger the first run immediately when a spec is created.
    /// Sets `run_at = now` so the scheduler picks it up on the next tick.
    /// Default: `true`.
    #[serde(default = "default_true")]
    pub auto_first_run: bool,
    /// Post a summary of new findings to the spec's `chat_id` after a
    /// scheduled run if `new_findings > 0`. Default: `true`.
    #[serde(default = "default_true")]
    pub notify_on_new_findings: bool,
    /// Run gatekeeper verification automatically for runs launched from
    /// Telegram (`/research run`) and the `research_launch` agent tool.
    /// Uses `gatekeeper.max_rounds` for the round budget.
    /// CLI `naked research run --verify` is unaffected (always explicit).
    /// Default: `true`.
    #[serde(default = "default_true")]
    pub verify_by_default: bool,
    /// Reasoning/thinking level applied to every research session via
    /// [`AgentCore::set_session_reasoning`]. Accepts `"off"`, `"low"`,
    /// `"medium"`, `"high"`. `None` ⇒ inherit from session/global default.
    ///
    /// Plumbed into the OpenAI-compatible `reasoning_effort` parameter for
    /// providers that accept it (kimi.com / moonshot.* / openai.com /
    /// fireworks / openrouter / deepseek) and into Anthropic-style
    /// extended-thinking budgets for Anthropic providers. See
    /// [`crate::provider::openai_compat::apply_reasoning_params`] and
    /// [`crate::provider::anthropic`] for the per-host gating rules.
    ///
    /// Recommended for `kimi-for-coding`: `"medium"` — same default as the
    /// official Roo Code integration guide.
    #[serde(default)]
    pub reasoning: Option<String>,
    /// Gatekeeper verification rules. Controls quality checks applied after
    /// each research run when using `--verify` / `run_verified()`.
    #[serde(default)]
    pub gatekeeper: GatekeeperConfig,
    /// Maximum number of research runs that may execute concurrently in this
    /// process. Applies to every entry point — manual `/research run`, the
    /// `research_launch` LLM tool, and the in-process scheduler. Default
    /// `5`: the scheduler subsystem treats this as the parallelism budget
    /// for at-time / cron / interval triggered runs. Set to `1` if your
    /// runs share a singleton (e.g. a single Playwright browser).
    #[serde(default = "default_max_concurrent_runs")]
    pub max_concurrent_runs: usize,
    /// Wall-clock cap for a single scheduler-launched run (seconds). When
    /// the cap is hit the scheduler cancels the run, increments the failure
    /// counter for that spec, and on the next tick may dispatch it again
    /// (subject to `max_retries_before_alert`). Per-spec override:
    /// `ResearchSpec.task_timeout_seconds`.
    #[serde(default = "default_task_timeout_seconds")]
    pub task_timeout_seconds: u64,
    /// Number of consecutive run failures (errors *or* timeouts) tolerated
    /// before the scheduler posts an alert into the spec's chat and stops
    /// retrying until reset. `0` disables the alert path entirely (legacy
    /// behaviour: keep retrying forever).
    #[serde(default = "default_max_retries_before_alert")]
    pub max_retries_before_alert: u32,
}

/// Quality gate configuration for the research verification loop.
///
/// All thresholds and prompts are configurable via JSON so you can tune
/// the quality bar and feedback wording without recompiling.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct GatekeeperConfig {
    /// Max verification rounds before accepting remaining issues.
    #[serde(default = "default_gk_max_rounds")]
    pub max_rounds: u32,
    /// Minimum excerpt length (chars) to be considered actionable.
    #[serde(default = "default_gk_min_excerpt")]
    pub min_excerpt_chars: usize,
    /// Minimum source_content length (chars).
    #[serde(default = "default_gk_min_source")]
    pub min_source_content_chars: usize,
    /// Maximum listing age in days. Older findings are removed.
    #[serde(default = "default_gk_max_age")]
    pub max_listing_age_days: i64,
    /// Require `listing_date` on every finding. When `true`, findings
    /// without a date trigger a feedback re-run.
    #[serde(default = "default_true")]
    pub require_listing_date: bool,
    /// Require `source_content` on every finding.
    #[serde(default = "default_true")]
    pub require_source_content: bool,
    /// Detect and remove semantic duplicates (same title+price, different URL).
    #[serde(default = "default_true")]
    pub detect_semantic_duplicates: bool,
    /// Stop the feedback loop when issue count doesn't decrease between rounds.
    #[serde(default = "default_true")]
    pub stop_on_stagnation: bool,
    /// HTTP timeout for URL liveness checks (seconds).
    #[serde(default = "default_gk_url_timeout")]
    pub url_check_timeout_secs: u64,

    // --- Prompt templates (use {placeholders}) ---
    /// Quality warnings returned inline by `research_save` when data is incomplete.
    /// Each entry is a condition→message pair checked at save time.
    /// Available placeholders: `{min_excerpt}`, `{min_source}`.
    #[serde(default)]
    pub save_warnings: GatekeeperSaveWarnings,

    /// Feedback prompt header sent to the agent when quality issues are found.
    /// Available placeholders: `{id}`, `{topic}`, `{today}`,
    /// `{dead_count}`, `{remediation_count}`, `{dead_list}`,
    /// `{quality_sections}`, `{dedup_list}`, `{min_excerpt}`, `{min_source}`.
    #[serde(default = "default_gk_feedback_header")]
    pub feedback_prompt_header: String,

    /// Section template for missing dates.
    /// Placeholders: `{count}`, `{today}`, `{list}`.
    #[serde(default = "default_gk_missing_date_section")]
    pub missing_date_section: String,

    /// Section template for missing source_content.
    /// Placeholders: `{count}`, `{min_source}`, `{list}`.
    #[serde(default = "default_gk_missing_source_section")]
    pub missing_source_section: String,

    /// Section template for short excerpts.
    /// Placeholders: `{count}`, `{min_excerpt}`, `{list}`.
    #[serde(default = "default_gk_short_excerpt_section")]
    pub short_excerpt_section: String,
}

/// Per-field quality warnings emitted by `research_save` so the agent
/// gets immediate feedback even during the first research pass.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct GatekeeperSaveWarnings {
    #[serde(default = "default_gk_warn_no_date")]
    pub no_listing_date: String,
    #[serde(default = "default_gk_warn_no_source")]
    pub no_source_content: String,
    #[serde(default = "default_gk_warn_short_excerpt")]
    pub short_excerpt: String,
}

impl Default for GatekeeperSaveWarnings {
    fn default() -> Self {
        Self {
            no_listing_date: default_gk_warn_no_date(),
            no_source_content: default_gk_warn_no_source(),
            short_excerpt: default_gk_warn_short_excerpt(),
        }
    }
}

impl Default for GatekeeperConfig {
    fn default() -> Self {
        Self {
            max_rounds: default_gk_max_rounds(),
            min_excerpt_chars: default_gk_min_excerpt(),
            min_source_content_chars: default_gk_min_source(),
            max_listing_age_days: default_gk_max_age(),
            require_listing_date: true,
            require_source_content: true,
            detect_semantic_duplicates: true,
            stop_on_stagnation: true,
            url_check_timeout_secs: default_gk_url_timeout(),
            save_warnings: GatekeeperSaveWarnings::default(),
            feedback_prompt_header: default_gk_feedback_header(),
            missing_date_section: default_gk_missing_date_section(),
            missing_source_section: default_gk_missing_source_section(),
            short_excerpt_section: default_gk_short_excerpt_section(),
        }
    }
}

fn default_gk_max_rounds() -> u32 {
    3
}
fn default_gk_min_excerpt() -> usize {
    300
}
fn default_gk_min_source() -> usize {
    100
}
fn default_gk_max_age() -> i64 {
    90
}
fn default_gk_url_timeout() -> u64 {
    10
}

fn default_gk_warn_no_date() -> String {
    "missing listing_date — look for ngày đăng/cập nhật on the page".into()
}
fn default_gk_warn_no_source() -> String {
    "missing source_content — paste the main page text (up to 8000 chars)".into()
}
fn default_gk_warn_short_excerpt() -> String {
    "excerpt too short (need ≥{min_excerpt} chars) — include contacts/phone, area m², price terms, deposit, address, condition".into()
}

fn default_gk_feedback_header() -> String {
    "You are an autonomous researcher. Research id: `{id}`.\n\
     \n\
     # GATEKEEPER FEEDBACK — verification round\n\
     \n\
     The previous research run has been verified. Some findings FAILED \
     quality checks. Your job now:\n\
     1. Find REPLACEMENTS for {dead_count} removed dead/stale links — \
        search for NEW listings using `web_search_exa` with different query \
        variations. Try at least 3 new search queries. Always OPEN the URL \
        with web_fetch BEFORE saving — verify it loads a real listing page.\n\
     2. FIX quality issues on {remediation_count} existing findings by \
        re-fetching each URL and calling `research_save` again with complete data.\n\
     \n\
     ## Removed findings (dead/stale — {dead_count})\n{dead_list}\n\
     \n\
     {remediation_section}\
     ## Quality issues breakdown\n\
     {quality_sections}\
     ## Rules\n\
     1. Every `research_save` call MUST include ALL of: title, price, \
        listing_date (today = {today}), excerpt (≥{min_excerpt} chars with contacts, \
        area, terms), source_content (≥{min_source} chars of page text).\n\
     2. To UPDATE an existing finding, call `research_save` with the SAME URL — \
        the system will overwrite the old data.\n\
     3. Do NOT re-save URLs that already have good data unless you are \
        specifically fixing an issue listed above.\n\
     4. For each remediation URL, actually OPEN the page (web_fetch or browser) \
        and re-extract the data. Do NOT guess or copy from memory.\n\
     5. If a page hides contacts behind login, note in excerpt: \
        \"Contacts hidden — requires site registration\".\n\
     6. Focus on the same topic: {topic}\n\
     \n\
     # Known findings (do NOT duplicate unless fixing)\n{dedup_list}\n"
        .into()
}

fn default_gk_missing_date_section() -> String {
    "### Missing listing_date ({count} findings)\n\
     Re-visit each URL below and look for the publication/update date. \
     Check: `ngày đăng`, `cập nhật`, `đăng ngày`, breadcrumbs, sidebar, \
     page footer near listing ID. Convert relative dates: `hôm nay` → {today}, \
     `hôm qua` → yesterday, `N ngày trước` → today minus N. \
     If truly absent, use `\"listing_date\": \"unknown\"`.\n{list}\n\n"
        .into()
}

fn default_gk_missing_source_section() -> String {
    "### Missing source_content ({count} findings)\n\
     Re-fetch each URL and paste the main page text (stripped of nav/ads/JS) \
     into the `source_content` field (up to 8000 chars). This is MANDATORY.\n{list}\n\n"
        .into()
}

fn default_gk_short_excerpt_section() -> String {
    "### Too-short excerpts ({count} findings)\n\
     Re-fetch each URL and expand the `excerpt` to ≥{min_excerpt} chars with ALL actionable \
     details: contacts (phone, Zalo, WhatsApp), area m², floor, conditions, \
     deposit, contract terms, amenities, neighbourhood.\n{list}\n\n"
        .into()
}

impl Default for ResearchConfig {
    fn default() -> Self {
        Self {
            enabled: default_research_enabled(),
            provider: None,
            model: None,
            fallback_models: Vec::new(),
            max_iterations: default_research_max_iterations(),
            max_wall_seconds: default_research_max_wall_seconds(),
            default_sources: Vec::new(),
            storage_dir: None,
            allowed_tools: None,
            schedule_enabled: true,
            default_interval_seconds: default_schedule_interval(),
            default_cron: None,
            auto_first_run: true,
            notify_on_new_findings: true,
            verify_by_default: true,
            reasoning: None,
            gatekeeper: GatekeeperConfig::default(),
            max_concurrent_runs: default_max_concurrent_runs(),
            task_timeout_seconds: default_task_timeout_seconds(),
            max_retries_before_alert: default_max_retries_before_alert(),
        }
    }
}

fn default_max_concurrent_runs() -> usize {
    5
}
