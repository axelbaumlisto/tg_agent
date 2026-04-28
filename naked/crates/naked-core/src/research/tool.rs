//! Tools the research agent calls during a run.
//!
//! - `research_save` — persist a new `Finding`. Idempotent thanks to
//!   `dedup_hash(url)`; duplicates return `duplicate:true` without an error.
//! - `research_list` — dedup-list of already-stored URLs so the model can skip
//!   them in the current turn (the coordinator prompt already carries 50, but
//!   an explicit tool call is useful for long runs).
//! - `research_status` — explicit-id lookup used by the `/research ask`
//!   conversational flow in Telegram: the model is given the research id and
//!   pulls spec + last findings + last run in one call.
//! - `research_save_cursor` — opaque JSON state for resumable pagination.
//!
//! All four mirror the ambient-context pattern from `tool::memory::MemoryTool`
//! (`ResearchContext::id` is set by the coordinator right before it kicks off
//! the turn, and cleared afterwards). Without a context set, save/list/cursor
//! return an error — never guess which research to write to.

use std::path::Path;
use std::sync::{Arc, RwLock};

use async_trait::async_trait;
use chrono::{NaiveDate, Utc};
use serde_json::{Value, json};

use crate::tool::Tool;
use crate::types::{Permission, ToolResult, ToolSpec};

use super::spec::{Cursor, Finding, content_hash, dedup_hash, host_path_hash};
use super::store::ResearchStore;

const MAX_LISTING_AGE_DAYS: i64 = 90;

/// Ambient "which research is this turn driving?" handle, installed by the
/// coordinator before it calls the agent and cleared after the turn is done.
/// Mirrors `tool::memory::MemoryContext` — same pattern, same invariants.
#[derive(Clone, Default)]
pub struct ResearchContext {
    inner: Arc<RwLock<Option<String>>>,
    run_id: Arc<RwLock<Option<String>>>,
    /// Optional waterfall sink (installed by `AgentCore` when the
    /// run-event registry is wired up). `None` in unit tests and CLI
    /// runs that don't care about the live TG progress stream.
    run_events: Arc<RwLock<Option<super::run_events::RunEventRegistry>>>,
    /// Successful `research_save` count for the current run. Reset to 0
    /// when the context is (re)bound to a fresh `(spec_id, run_id)` and
    /// inspected by `research_set_target` to refuse a "clear without
    /// saving anything" — the most common idle-failure mode where the
    /// agent visits 30+ pages but never persists a single finding.
    saves: Arc<RwLock<u32>>,
}

impl ResearchContext {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn set_id(&self, id: Option<String>) {
        if let Ok(mut g) = self.inner.write() {
            *g = id;
        }
    }

    pub fn id(&self) -> Option<String> {
        self.inner.read().ok().and_then(|g| g.clone())
    }

    pub fn set_run_id(&self, id: Option<String>) {
        if let Ok(mut g) = self.run_id.write() {
            *g = id;
        }
    }

    pub fn run_id(&self) -> Option<String> {
        self.run_id.read().ok().and_then(|g| g.clone())
    }

    /// Attach (or replace) the run-event registry used by tools to
    /// push waterfall events. Cheap — the registry is `Arc`-wrapped
    /// internally.
    pub fn set_run_events(&self, reg: Option<super::run_events::RunEventRegistry>) {
        if let Ok(mut g) = self.run_events.write() {
            *g = reg;
        }
    }

    pub fn run_events(&self) -> Option<super::run_events::RunEventRegistry> {
        self.run_events.read().ok().and_then(|g| g.clone())
    }

    /// Increment the per-run save counter. Called by `ResearchSaveTool`
    /// after a successful append (skipped saves do not count). Saturates
    /// at u32::MAX — we only ever check `> 0`.
    pub fn note_save(&self) {
        if let Ok(mut g) = self.saves.write() {
            *g = g.saturating_add(1);
        }
    }

    /// Number of successful saves recorded for the current bound run.
    pub fn save_count(&self) -> u32 {
        self.saves.read().map(|g| *g).unwrap_or(0)
    }

    /// Reset the per-run save counter to zero. Called from
    /// `set_id`/`set_run_id`-like rebind paths so a freshly-acquired
    /// context starts at 0 saves regardless of what the previous run
    /// observed.
    pub fn reset_saves(&self) {
        if let Ok(mut g) = self.saves.write() {
            *g = 0;
        }
    }
}

fn err(msg: impl Into<String>) -> ToolResult {
    ToolResult {
        output: msg.into(),
        is_error: true,
    }
}

fn ok(msg: impl Into<String>) -> ToolResult {
    ToolResult {
        output: msg.into(),
        is_error: false,
    }
}

/// Strip leading/trailing source-attribution lines an LLM tends to add to the
/// excerpt even when told not to. Targets the most common Vietnamese / English
/// trailers we see in the wild: `Nguồn: ...`, `Source: ...`, `Posted by ...`,
/// `đăng N ngày/giờ trước`, `Cập nhật ...`.
///
/// Conservative: only drops a *whole line* (or a final clause separated by `—`
/// / `-` / `|`) that matches one of the known prefixes case-insensitively. We
/// never edit the body of the text, so a legitimate phone number written
/// alongside `Source: foo` survives if it's on a different line.
pub(crate) fn strip_source_attribution(text: &str) -> String {
    const PREFIXES: &[&str] = &[
        "nguồn:",
        "nguon:",
        "source:",
        "источник:",
        "posted by",
        "đăng bởi",
        "dang boi",
        "đăng ngày",
        "cập nhật",
        "cap nhat",
    ];
    fn looks_like_attribution(line: &str) -> bool {
        let t = line.trim().to_lowercase();
        if t.is_empty() {
            return false;
        }
        if PREFIXES.iter().any(|p| t.starts_with(p)) {
            return true;
        }
        // Match relative-time trailers: "đăng 3 ngày trước", "5 giờ trước".
        (t.contains(" ngày trước") || t.contains(" giờ trước")) && t.split_whitespace().count() <= 6
    }

    let mut kept: Vec<String> = text
        .lines()
        .filter(|l| !looks_like_attribution(l))
        .map(|l| l.trim_end().to_string())
        .collect();
    // Strip a single trailing clause after the last separator if it looks like
    // attribution: "...rooms — Nguồn: alonhadat.com.vn".
    if let Some(last) = kept.pop() {
        let cleaned = ["—", " - ", " | "].iter().fold(last, |acc, sep| {
            if let Some((head, tail)) = acc.rsplit_once(sep)
                && looks_like_attribution(tail)
            {
                head.trim_end().to_string()
            } else {
                acc
            }
        });
        kept.push(cleaned);
    }
    kept.join("\n").trim().to_string()
}

/// Trim free-form agent-provided text to a max length so a single bad excerpt
/// can't balloon `findings.jsonl` to MB-per-line.
fn clip(text: &str, max: usize) -> Option<String> {
    let trimmed = text.trim();
    if trimmed.is_empty() {
        return None;
    }
    if trimmed.chars().count() <= max {
        Some(trimmed.to_string())
    } else {
        Some(trimmed.chars().take(max).collect::<String>() + "…")
    }
}

/// Try to parse a free-form listing date string into a NaiveDate.
/// Handles: `YYYY-MM-DD`, `DD/MM/YYYY`, `DD-MM-YYYY`, `DD.MM.YYYY`,
/// relative Vietnamese (`hôm nay`, `hôm qua`, `N ngày trước`), and `unknown`.
pub fn parse_listing_date(s: &str) -> Option<NaiveDate> {
    let s = s.trim().to_lowercase();
    if s.is_empty() || s == "unknown" {
        return None;
    }
    let today = Utc::now().date_naive();

    if s.contains("hôm nay") || s == "today" {
        return Some(today);
    }
    if s.contains("hôm qua") || s == "yesterday" {
        return Some(today - chrono::Duration::days(1));
    }
    // "N ngày trước" / "N days ago"
    if let Some(n) = extract_days_ago(&s) {
        return Some(today - chrono::Duration::days(n));
    }

    // ISO: 2026-04-18
    if let Ok(d) = NaiveDate::parse_from_str(&s, "%Y-%m-%d") {
        return Some(d);
    }
    // DD/MM/YYYY
    if let Ok(d) = NaiveDate::parse_from_str(&s, "%d/%m/%Y") {
        return Some(d);
    }
    // DD-MM-YYYY
    if let Ok(d) = NaiveDate::parse_from_str(&s, "%d-%m-%Y") {
        return Some(d);
    }
    // DD.MM.YYYY
    if let Ok(d) = NaiveDate::parse_from_str(&s, "%d.%m.%Y") {
        return Some(d);
    }
    None
}

fn extract_days_ago(s: &str) -> Option<i64> {
    // Match patterns like "3 ngày trước", "5 days ago"
    for word in s.split_whitespace() {
        if let Ok(n) = word.parse::<i64>()
            && (s.contains("ngày trước") || s.contains("days ago"))
        {
            return Some(n);
        }
    }
    None
}

/// Check if a listing date is too old (> MAX_LISTING_AGE_DAYS).
/// Returns `None` if date can't be parsed (let it through — benefit of the doubt).
fn is_stale(listing_date: Option<&str>) -> Option<bool> {
    let s = listing_date?;
    let d = parse_listing_date(s)?;
    let age = Utc::now().date_naive() - d;
    Some(age.num_days() > MAX_LISTING_AGE_DAYS)
}

/// `research_save` — save or update a finding, with configurable quality warnings.
pub struct ResearchSaveTool {
    store: Arc<dyn ResearchStore>,
    context: ResearchContext,
    gk: crate::config::GatekeeperConfig,
}

impl ResearchSaveTool {
    pub fn new(
        store: Arc<dyn ResearchStore>,
        context: ResearchContext,
        gk: crate::config::GatekeeperConfig,
    ) -> Self {
        Self { store, context, gk }
    }
}

#[async_trait]
impl Tool for ResearchSaveTool {
    fn spec(&self) -> ToolSpec {
        ToolSpec {
            name: "research_save".into(),
            description: "Save or update a finding in the active research. \
                If the URL already exists, the finding is UPDATED with the new data \
                (returned as `updated:true`). If the URL is new, a new finding is \
                created. Use this to fix quality issues on existing findings by \
                re-saving with the same URL and more complete data. Requires an \
                active research context (set by /research run or the coordinator)."
                .into(),
            parameters: json!({
                "type": "object",
                "properties": {
                    "url":          { "type": "string", "description": "Full URL of the item (required). Real URL only — never invent." },
                    "title":        { "type": "string", "description": "Short human title (≤200 chars)" },
                    "excerpt":      { "type": "string", "description": "ALL actionable details from the page: contacts (phone, name, messenger), area (m²), floor, conditions, amenities, neighbourhood, transport — everything that makes this finding useful without revisiting the URL. Up to 2000 chars." },
                    "price":        { "type": "string", "description": "Price or headline metric (e.g. `45 000 000 ₫`)" },
                    "listing_date": { "type": "string", "description": "Publication or update date of the listing as shown on the page (e.g. `2026-04-15`, `15/04/2026`, `hôm nay`)" },
                    "source_content": { "type": "string", "description": "Condensed text of the source page stripped of navigation/ads — enough to verify the excerpt claims without re-fetching the URL. Up to 8000 chars." }
                },
                "required": ["url"]
            }),
            permission: Permission::WorkspaceWrite,
        }
    }

    async fn execute(&self, input: Value, _cwd: &Path) -> ToolResult {
        let Some(id) = self.context.id() else {
            return err("no active research context — call this tool only inside /research run");
        };
        let url = match input.get("url").and_then(|v| v.as_str()) {
            Some(s) if !s.trim().is_empty() => s.trim().to_string(),
            _ => return err("`url` is required and must be non-empty"),
        };
        if !(url.starts_with("http://") || url.starts_with("https://")) {
            return err(format!(
                "`url` must be absolute http(s) — got `{}`",
                url.chars().take(80).collect::<String>()
            ));
        }

        // Reject stale listings (> 90 days old) at save time
        let listing_date_raw = input.get("listing_date").and_then(|v| v.as_str());
        if let Some(true) = is_stale(listing_date_raw) {
            let date_str = listing_date_raw.unwrap_or("?");
            return ok(format!(
                "{{\"stored\":false,\"skipped\":true,\"reason\":\"listing too old ({date_str}), max {MAX_LISTING_AGE_DAYS} days\"}}"
            ));
        }

        // Reject captcha-stub placeholder content. The agent is
        // explicitly told to use browser_navigate for JS-gated hosts;
        // when it ignores that and tries to save the raw HTML of a
        // Cloudflare interstitial, we catch it here instead of letting
        // it pollute findings.jsonl.
        let excerpt_raw = input.get("excerpt").and_then(|v| v.as_str()).unwrap_or("");
        let source_raw = input
            .get("source_content")
            .and_then(|v| v.as_str())
            .unwrap_or("");
        let combined = format!("{excerpt_raw}\n{source_raw}");
        if let Some(marker) = super::validators::looks_like_captcha_stub(&combined) {
            // Mirror the waterfall so the operator sees WHY the save
            // was dropped without grepping the log.
            if let Some(reg) = self.context.run_events() {
                reg.push(
                    &id,
                    super::run_events::RunEvent::new(
                        super::run_events::EventKind::BlockDetected,
                        format!("research_save rejected: captcha stub on {url}"),
                    ),
                )
                .await;
            }
            return ok(format!(
                "{{\"stored\":false,\"skipped\":true,\"reason\":\"captcha_stub (matched `{marker}`) — rerun via Skill(web-browser-playbook) + browser_navigate on {url}\"}}"
            ));
        }

        let run_id = self
            .context
            .run_id()
            .unwrap_or_else(|| "manual".to_string());

        let excerpt_str = input
            .get("excerpt")
            .and_then(|v| v.as_str())
            .map(strip_source_attribution)
            .and_then(|s| clip(&s, 2000));
        let source_content_str = input
            .get("source_content")
            .and_then(|v| v.as_str())
            .and_then(|s| clip(s, 8000));

        // Content fingerprint: prefer source_content (richer), fall back to
        // excerpt. When both are absent, leaves `content_hash` empty so the
        // store skips the content-based dedup check.
        let content_for_hash = content_hash(
            source_content_str
                .as_deref()
                .or(excerpt_str.as_deref())
                .unwrap_or(""),
        );

        let finding = Finding {
            id: uuid::Uuid::new_v4().simple().to_string(),
            research_id: id.clone(),
            run_id,
            url: url.clone(),
            title: input
                .get("title")
                .and_then(|v| v.as_str())
                .and_then(|s| clip(s, 200)),
            excerpt: excerpt_str,
            price: input
                .get("price")
                .and_then(|v| v.as_str())
                .and_then(|s| clip(s, 80)),
            listing_date: input
                .get("listing_date")
                .and_then(|v| v.as_str())
                .and_then(|s| clip(s, 40)),
            source_content: source_content_str,
            dedup_hash: dedup_hash(&url),
            host_path_hash: host_path_hash(&url),
            content_hash: content_for_hash,
            seen_at: Utc::now(),
        };

        let mut warnings = Vec::new();
        if finding.listing_date.is_none() && self.gk.require_listing_date {
            warnings.push(self.gk.save_warnings.no_listing_date.clone());
        }
        if finding.source_content.is_none() && self.gk.require_source_content {
            warnings.push(self.gk.save_warnings.no_source_content.clone());
        }
        if finding
            .excerpt
            .as_ref()
            .is_none_or(|e| e.len() < self.gk.min_excerpt_chars)
        {
            warnings.push(
                self.gk
                    .save_warnings
                    .short_excerpt
                    .replace("{min_excerpt}", &self.gk.min_excerpt_chars.to_string()),
            );
        }

        match self.store.upsert_finding(&finding).await {
            Ok(updated) => {
                // Bump the per-run save counter so `research_set_target`
                // can refuse a "clear without saving anything" cleanup.
                // Both inserts and updates count — agent did valid work.
                self.context.note_save();
                let total = self.store.count_findings(&id).await.unwrap_or(0);
                if let Some(reg) = self.context.run_events() {
                    let title_preview = finding
                        .title
                        .as_deref()
                        .unwrap_or("(untitled)")
                        .chars()
                        .take(40)
                        .collect::<String>();
                    let label = if updated {
                        format!("upsert {title_preview}")
                    } else {
                        format!("saved \"{title_preview}\"")
                    };
                    reg.push(
                        &id,
                        super::run_events::RunEvent::new(
                            super::run_events::EventKind::FindingSaved,
                            label,
                        ),
                    )
                    .await;
                }
                let warn_json = if warnings.is_empty() {
                    String::new()
                } else {
                    format!(
                        ",\"quality_warnings\":[{}]",
                        warnings
                            .iter()
                            .map(|w| format!("\"{w}\""))
                            .collect::<Vec<_>>()
                            .join(",")
                    )
                };
                if updated {
                    ok(format!(
                        "{{\"stored\":true,\"updated\":true,\"total_findings\":{total}{warn_json}}}"
                    ))
                } else {
                    ok(format!(
                        "{{\"stored\":true,\"duplicate\":false,\"total_findings\":{total}{warn_json}}}"
                    ))
                }
            }
            Err(e) => err(format!("store error: {e}")),
        }
    }
}

/// `research_list` — return already-known URLs so the agent can self-dedup
/// mid-run (the coordinator prompt carries the first 50, this tool is for
/// when the research grows beyond that).
pub struct ResearchListTool {
    store: Arc<dyn ResearchStore>,
    context: ResearchContext,
}

impl ResearchListTool {
    pub fn new(store: Arc<dyn ResearchStore>, context: ResearchContext) -> Self {
        Self { store, context }
    }
}

#[async_trait]
impl Tool for ResearchListTool {
    fn spec(&self) -> ToolSpec {
        ToolSpec {
            name: "research_list".into(),
            description: "List the most recent findings already stored for the \
                active research. Use before saving if you're unsure whether a URL \
                was seen earlier."
                .into(),
            parameters: json!({
                "type": "object",
                "properties": {
                    "limit": { "type": "integer", "default": 20, "minimum": 1, "maximum": 200 }
                }
            }),
            permission: Permission::ReadOnly,
        }
    }

    async fn execute(&self, input: Value, _cwd: &Path) -> ToolResult {
        let Some(id) = self.context.id() else {
            return err("no active research context");
        };
        let limit = input
            .get("limit")
            .and_then(|v| v.as_u64())
            .map(|n| n.min(200) as usize)
            .unwrap_or(20);
        match self.store.list_findings(&id, Some(limit)).await {
            Ok(list) if list.is_empty() => ok("(no findings yet)".to_string()),
            Ok(list) => {
                let mut out = String::new();
                for f in list.iter().rev() {
                    let title = f.title.as_deref().unwrap_or("(untitled)");
                    out.push_str(&format!("- {} — {}\n", title, f.url));
                }
                ok(out)
            }
            Err(e) => err(format!("store error: {e}")),
        }
    }
}

/// `research_save_cursor` — opaque state blob the agent writes to resume
/// pagination between runs. Replaces the previous cursor atomically.
pub struct ResearchSaveCursorTool {
    store: Arc<dyn ResearchStore>,
    context: ResearchContext,
}

impl ResearchSaveCursorTool {
    pub fn new(store: Arc<dyn ResearchStore>, context: ResearchContext) -> Self {
        Self { store, context }
    }
}

#[async_trait]
impl Tool for ResearchSaveCursorTool {
    fn spec(&self) -> ToolSpec {
        ToolSpec {
            name: "research_save_cursor".into(),
            description: "Persist a resume cursor for the active research. \
                Call at the end of a run with whatever pagination state you'd \
                need on the next pass (page number, last post id, etc.). Replaces \
                the previous cursor."
                .into(),
            parameters: json!({
                "type": "object",
                "properties": {
                    "cursor": {
                        "type": "object",
                        "description": "Free-form JSON object describing resume state",
                        "additionalProperties": true
                    }
                },
                "required": ["cursor"]
            }),
            permission: Permission::WorkspaceWrite,
        }
    }

    async fn execute(&self, input: Value, _cwd: &Path) -> ToolResult {
        let Some(id) = self.context.id() else {
            return err("no active research context");
        };
        let cursor_val = match input.get("cursor") {
            Some(Value::Object(m)) => m.clone(),
            _ => return err("`cursor` must be a JSON object"),
        };
        let cursor = Cursor {
            data: cursor_val,
            updated_at: None,
        };
        match self.store.save_cursor(&id, &cursor).await {
            Ok(()) => ok("cursor saved".to_string()),
            Err(e) => err(format!("store error: {e}")),
        }
    }
}

/// `research_status` — explicit-id lookup. Unlike the three tools above it does
/// NOT need an ambient context, so the Telegram `/research ask` flow can hand
/// a specific id to the agent and ask for a reasoned reply. Returns a compact
/// markdown block: topic, counts, last 10 findings, last 3 runs.
pub struct ResearchStatusTool {
    store: Arc<dyn ResearchStore>,
}

impl ResearchStatusTool {
    pub fn new(store: Arc<dyn ResearchStore>) -> Self {
        Self { store }
    }
}

#[async_trait]
impl Tool for ResearchStatusTool {
    fn spec(&self) -> ToolSpec {
        ToolSpec {
            name: "research_status".into(),
            description: "Return a concise status report for a research by id: \
                topic, counts, last ~10 findings, last ~3 runs. Use when the user \
                asks what you've found for a specific research."
                .into(),
            parameters: json!({
                "type": "object",
                "properties": {
                    "research_id": { "type": "string" },
                    "finding_limit": { "type": "integer", "default": 10, "minimum": 1, "maximum": 100 },
                    "fresh_only": { "type": "boolean", "default": false, "description": "If true, show only findings from the most recent run" },
                    "max_age_days": { "type": "integer", "default": 90, "description": "Hide findings with listing_date older than N days. 0 = show all." }
                },
                "required": ["research_id"]
            }),
            permission: Permission::ReadOnly,
        }
    }

    async fn execute(&self, input: Value, _cwd: &Path) -> ToolResult {
        let Some(id) = input.get("research_id").and_then(|v| v.as_str()) else {
            return err("`research_id` is required");
        };
        let limit = input
            .get("finding_limit")
            .and_then(|v| v.as_u64())
            .map(|n| n.min(100) as usize)
            .unwrap_or(10);
        let fresh_only = input
            .get("fresh_only")
            .and_then(|v| v.as_bool())
            .unwrap_or(false);
        let max_age_days = input
            .get("max_age_days")
            .and_then(|v| v.as_i64())
            .unwrap_or(90);

        let spec = match self.store.load_spec(id).await {
            Ok(s) => s,
            Err(e) => return err(format!("unknown research `{id}`: {e}")),
        };
        let runs = self.store.list_runs(id, Some(3)).await.unwrap_or_default();
        let total = self.store.count_findings(id).await.unwrap_or(0);

        let mut findings = self
            .store
            .list_findings(id, if fresh_only { None } else { Some(limit) })
            .await
            .unwrap_or_default();

        if fresh_only {
            if let Some(last_run) = runs.last() {
                let run_id = &last_run.run_id;
                findings.retain(|f| f.run_id == *run_id);
            }
            findings.truncate(limit);
        }

        // Filter out stale listings
        if max_age_days > 0 {
            let today = Utc::now().date_naive();
            findings.retain(|f| {
                if let Some(ref date_str) = f.listing_date
                    && let Some(d) = parse_listing_date(date_str)
                {
                    return (today - d).num_days() <= max_age_days;
                }
                true
            });
        }

        let mut out = String::new();
        out.push_str(&format!("# Research `{id}`\n"));
        out.push_str(&format!("**Topic:** {}\n\n", spec.topic));
        out.push_str(&format!("**Total findings:** {total}\n"));
        if fresh_only {
            out.push_str(&format!(
                "**Showing:** {} fresh (latest run only)\n",
                findings.len()
            ));
        }
        out.push_str(&format!("**Paused:** {}\n\n", spec.paused));

        out.push_str(if fresh_only {
            "## Fresh findings\n"
        } else {
            "## Latest findings\n"
        });
        if findings.is_empty() {
            out.push_str("(none)\n");
        } else {
            for f in findings.iter().rev() {
                let title = f.title.as_deref().unwrap_or("(untitled)");
                out.push_str(&format!("- [{title}]({})", f.url));
                if let Some(p) = &f.price {
                    out.push_str(&format!(" — {p}"));
                }
                if let Some(d) = &f.listing_date {
                    out.push_str(&format!(" _{d}_"));
                }
                out.push('\n');
            }
        }
        out.push_str("\n## Recent runs\n");
        if runs.is_empty() {
            out.push_str("(none)\n");
        } else {
            for r in runs.iter().rev() {
                out.push_str(&format!(
                    "- {}: +{} new (total {}) via {}/{} — {}\n",
                    r.started_at.format("%Y-%m-%d %H:%M UTC"),
                    r.new_findings,
                    r.total_findings_after,
                    r.provider,
                    r.model,
                    r.stop_reason,
                ));
            }
        }
        ok(out)
    }
}

/// Crude credential/secret redactor. Runs before we forward any research
/// output (reports, summaries, tool outputs) to Telegram / Discord so a
/// misconfigured proxy or a `curl -H "Authorization: ..."` snippet doesn't
/// leak by accident. Ported from zeroclaws `scan_and_redact_output`, but
/// scoped to the small handful of patterns we actually see.
pub fn scan_and_redact(text: &str) -> String {
    // Cheap, layered regex: no backtracking, case-insensitive.
    // Matches "api_key=SOMETHING", "Bearer XXX", "Authorization: Token YYY",
    // and long `$ALLCAPS=SECRETVAL` env exports. All values collapse to
    // `[redacted]`.
    let mut out = text.to_string();
    for re in REDACTORS.iter() {
        out = re.replace_all(&out, "$key=[redacted]").to_string();
    }
    for re in BEARER_REDACTORS.iter() {
        out = re.replace_all(&out, "$prefix [redacted]").to_string();
    }
    out
}

// Lazy-compiled regex tables. Using the `regex` crate would add a dep; instead
// we write a tiny inline matcher. The patterns are intentionally narrow.
static REDACTORS: once_cell_shim::Lazy<Vec<redact::Pattern>> = once_cell_shim::Lazy::new(|| {
    vec![
        // `api_key = "abc"`, `api-key: abc`, `API_KEY=abc`
        redact::Pattern::new("api[_-]?key", true),
        redact::Pattern::new("secret", true),
        redact::Pattern::new("password", true),
        redact::Pattern::new("token", true),
    ]
});

static BEARER_REDACTORS: once_cell_shim::Lazy<Vec<redact::BearerPattern>> =
    once_cell_shim::Lazy::new(|| {
        // NOTE: we only redact the bare `bearer <token>` sequence. Redacting
        // the `authorization:` header as a separate pass collides with the
        // already-redacted `bearer` token and produces `[redacted] [redacted]`
        // noise. One pass is enough — `Authorization: Bearer xxx` becomes
        // `Authorization: Bearer [redacted]`, which still hides the secret.
        vec![redact::BearerPattern::new("bearer")]
    });

/// Minimal std-only lazy helper so we can avoid pulling in `once_cell` just for
/// two tables. Mimics `OnceLock<Vec<_>>` behind a `Lazy`-like API.
mod once_cell_shim {
    use std::sync::OnceLock;

    pub struct Lazy<T: 'static> {
        init: fn() -> T,
        cell: OnceLock<T>,
    }

    impl<T: 'static> Lazy<T> {
        pub const fn new(init: fn() -> T) -> Self {
            Self {
                init,
                cell: OnceLock::new(),
            }
        }
    }

    impl<T: 'static> std::ops::Deref for Lazy<T> {
        type Target = T;
        fn deref(&self) -> &Self::Target {
            self.cell.get_or_init(self.init)
        }
    }
}

mod redact {
    /// Very small "`key` = value" matcher. Finds case-insensitive `key` (as a
    /// word), optional whitespace, one of `:` / `=`, whitespace, then captures
    /// everything up to the next whitespace / quote / `,` / `;` / end.
    pub struct Pattern {
        key: String,
        case_insensitive: bool,
    }

    impl Pattern {
        pub fn new(key: &str, case_insensitive: bool) -> Self {
            Self {
                key: key.to_string(),
                case_insensitive,
            }
        }

        pub fn replace_all(&self, haystack: &str, replacement: &str) -> String {
            let mut result = String::with_capacity(haystack.len());
            let bytes = haystack.as_bytes();
            let mut i = 0;
            while i < bytes.len() {
                if let Some(end) = self.match_at(haystack, i) {
                    // `replacement` is `$key=[redacted]` — we emit the original
                    // key substring the pattern matched, verbatim, then `=[redacted]`.
                    let key_str = &haystack[i..end.key_end];
                    let rendered = replacement.replace("$key", key_str);
                    result.push_str(&rendered);
                    i = end.value_end;
                } else {
                    let ch_len = haystack[i..]
                        .chars()
                        .next()
                        .map(|c| c.len_utf8())
                        .unwrap_or(1);
                    result.push_str(&haystack[i..i + ch_len]);
                    i += ch_len;
                }
            }
            result
        }

        fn match_at(&self, hay: &str, start: usize) -> Option<Span> {
            let lc;
            let haystack_cmp: &str = if self.case_insensitive {
                lc = hay.to_ascii_lowercase();
                &lc
            } else {
                hay
            };
            let key_lc = if self.case_insensitive {
                self.key.to_ascii_lowercase()
            } else {
                self.key.clone()
            };
            if start + key_lc.len() > haystack_cmp.len() {
                return None;
            }
            // Match our regex-y key (allow [_-]? meta). Simplified: we support
            // the hardcoded `[_-]?` between tokens as written in the callers;
            // otherwise just literal equals.
            let key_end = match_key_literal(haystack_cmp, start, &key_lc)?;
            // Must be a word boundary at start (previous char non-alnum / start).
            if start > 0 {
                let prev = hay[..start].chars().next_back().unwrap_or(' ');
                if prev.is_ascii_alphanumeric() || prev == '_' {
                    return None;
                }
            }
            // Skip whitespace.
            let after_key = skip_ws(hay, key_end);
            let sep = hay.as_bytes().get(after_key).copied()?;
            if sep != b'=' && sep != b':' {
                return None;
            }
            let after_sep = skip_ws(hay, after_key + 1);
            // Value: everything until whitespace / quote / `,` / `;` / `}` / `)`, preserving the start.
            let value_end = find_value_end(hay, after_sep);
            if value_end == after_sep {
                return None;
            }
            Some(Span { key_end, value_end })
        }
    }

    pub struct BearerPattern {
        prefix: String,
    }

    impl BearerPattern {
        pub fn new(prefix: &str) -> Self {
            Self {
                prefix: prefix.to_string(),
            }
        }
        pub fn replace_all(&self, hay: &str, rep_template: &str) -> String {
            let mut result = String::with_capacity(hay.len());
            let lc = hay.to_ascii_lowercase();
            let needle = self.prefix.to_ascii_lowercase();
            let mut i = 0;
            while i < hay.len() {
                if let Some(pos) = lc[i..].find(&needle) {
                    let start = i + pos;
                    // Must be at word boundary (start-of-string or non-alnum before).
                    if start > 0 {
                        let prev = hay[..start].chars().next_back().unwrap_or(' ');
                        if prev.is_ascii_alphanumeric() || prev == '_' {
                            result.push_str(&hay[i..start + needle.len()]);
                            i = start + needle.len();
                            continue;
                        }
                    }
                    result.push_str(&hay[i..start]);
                    let after_kw = start + needle.len();
                    let after_ws = skip_ws(hay, after_kw);
                    let end = find_value_end(hay, after_ws);
                    let prefix_str = &hay[start..after_ws];
                    let rendered = rep_template.replace("$prefix", prefix_str.trim_end());
                    result.push_str(&rendered);
                    i = end;
                } else {
                    result.push_str(&hay[i..]);
                    break;
                }
            }
            result
        }
    }

    struct Span {
        key_end: usize,
        value_end: usize,
    }

    fn match_key_literal(hay: &str, start: usize, key: &str) -> Option<usize> {
        // Support `[_-]?` meta inside keys like `api[_-]?key` — interpret as:
        // try literal match with `_`, `-`, or none between the segments on `[`.
        if key.contains("[_-]?") {
            let parts: Vec<&str> = key.split("[_-]?").collect();
            let mut end = start;
            for (i, part) in parts.iter().enumerate() {
                if end + part.len() > hay.len() {
                    return None;
                }
                if &hay[end..end + part.len()] != *part {
                    return None;
                }
                end += part.len();
                if i + 1 < parts.len() {
                    match hay.as_bytes().get(end) {
                        Some(b'_') | Some(b'-') => end += 1,
                        _ => {}
                    }
                }
            }
            return Some(end);
        }
        if start + key.len() > hay.len() {
            return None;
        }
        if &hay[start..start + key.len()] == key {
            return Some(start + key.len());
        }
        None
    }

    fn skip_ws(hay: &str, mut i: usize) -> usize {
        while let Some(b) = hay.as_bytes().get(i) {
            if matches!(*b, b' ' | b'\t') {
                i += 1;
            } else {
                break;
            }
        }
        i
    }

    fn find_value_end(hay: &str, start: usize) -> usize {
        // If the value starts with a quote, consume until matching quote.
        if let Some(b'"') = hay.as_bytes().get(start) {
            let after = start + 1;
            if let Some(pos) = hay[after..].find('"') {
                return after + pos + 1;
            }
        }
        let stop: &[char] = &[' ', '\t', '\n', '\r', ',', ';', '"', '}', ')', ']'];
        match hay[start..].find(stop) {
            Some(p) => start + p,
            None => hay.len(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::research::spec::ResearchSpec;
    use crate::research::store::FsResearchStore;
    use tempfile::tempdir;

    fn setup() -> (tempfile::TempDir, Arc<dyn ResearchStore>, ResearchContext) {
        let tmp = tempdir().unwrap();
        let store: Arc<dyn ResearchStore> =
            Arc::new(FsResearchStore::new(tmp.path().to_path_buf()));
        let ctx = ResearchContext::new();
        (tmp, store, ctx)
    }

    fn make_spec(id: &str) -> ResearchSpec {
        ResearchSpec {
            id: id.into(),
            topic: "t".into(),
            sources: vec![],
            interval_seconds: None,
            run_at: None,
            cron: None,
            task_timeout_seconds: None,
            session_id: None,
            chat_id: None,
            thread_id: None,
            provider: None,
            model: None,
            max_iterations: None,
            max_wall_seconds: None,
            created_at: Utc::now(),
            paused: false,
            pause_reason: None,
        }
    }

    #[tokio::test]
    async fn research_save_stores_and_dedups() {
        let (_tmp, store, ctx) = setup();
        store.create_spec(&make_spec("r1")).await.unwrap();
        ctx.set_id(Some("r1".into()));
        let tool = ResearchSaveTool::new(store.clone(), ctx.clone(), Default::default());

        let cwd = std::env::current_dir().unwrap();
        let first = tool
            .execute(json!({"url":"https://ex.com/a","title":"A"}), &cwd)
            .await;
        assert!(!first.is_error);
        assert!(first.output.contains("\"stored\":true"));

        let dup = tool
            .execute(
                json!({"url":"https://ex.com/a?utm_source=x","title":"A updated"}),
                &cwd,
            )
            .await;
        assert!(!dup.is_error);
        assert!(dup.output.contains("\"updated\":true"));
        assert_eq!(store.count_findings("r1").await.unwrap(), 1);
    }

    #[tokio::test]
    async fn research_save_rejects_missing_context() {
        let (_tmp, store, ctx) = setup();
        let tool = ResearchSaveTool::new(store.clone(), ctx, Default::default());
        let cwd = std::env::current_dir().unwrap();
        let r = tool.execute(json!({"url":"https://ex.com/a"}), &cwd).await;
        assert!(r.is_error);
    }

    #[tokio::test]
    async fn research_save_rejects_bad_url() {
        let (_tmp, store, ctx) = setup();
        store.create_spec(&make_spec("r1")).await.unwrap();
        ctx.set_id(Some("r1".into()));
        let tool = ResearchSaveTool::new(store.clone(), ctx, Default::default());
        let cwd = std::env::current_dir().unwrap();
        let r = tool.execute(json!({"url":"ftp://x/"}), &cwd).await;
        assert!(r.is_error);
        let r2 = tool.execute(json!({"url":""}), &cwd).await;
        assert!(r2.is_error);
    }

    #[tokio::test]
    async fn research_list_shows_recent_entries() {
        let (_tmp, store, ctx) = setup();
        store.create_spec(&make_spec("r2")).await.unwrap();
        ctx.set_id(Some("r2".into()));
        let save = ResearchSaveTool::new(store.clone(), ctx.clone(), Default::default());
        let cwd = std::env::current_dir().unwrap();
        save.execute(json!({"url":"https://ex.com/1","title":"One"}), &cwd)
            .await;
        save.execute(json!({"url":"https://ex.com/2","title":"Two"}), &cwd)
            .await;
        let list = ResearchListTool::new(store.clone(), ctx)
            .execute(json!({"limit":10}), &cwd)
            .await;
        assert!(!list.is_error);
        assert!(list.output.contains("https://ex.com/1"));
        assert!(list.output.contains("https://ex.com/2"));
    }

    #[tokio::test]
    async fn research_save_cursor_roundtrip() {
        let (_tmp, store, ctx) = setup();
        store.create_spec(&make_spec("r3")).await.unwrap();
        ctx.set_id(Some("r3".into()));
        let tool = ResearchSaveCursorTool::new(store.clone(), ctx);
        let cwd = std::env::current_dir().unwrap();
        let r = tool
            .execute(json!({"cursor":{"page":7,"anchor":"abc"}}), &cwd)
            .await;
        assert!(!r.is_error, "{}", r.output);
        let reloaded = store.load_cursor("r3").await.unwrap();
        assert_eq!(reloaded.data.get("page"), Some(&json!(7)));
    }

    #[tokio::test]
    async fn research_status_produces_markdown_for_known_id() {
        let (_tmp, store, _ctx) = setup();
        store.create_spec(&make_spec("r4")).await.unwrap();
        let tool = ResearchStatusTool::new(store.clone());
        let cwd = std::env::current_dir().unwrap();
        let r = tool.execute(json!({"research_id":"r4"}), &cwd).await;
        assert!(!r.is_error);
        assert!(r.output.contains("Research `r4`"));
        assert!(r.output.contains("**Total findings:** 0"));
    }

    #[tokio::test]
    async fn research_status_rejects_unknown_id() {
        let (_tmp, store, _ctx) = setup();
        let tool = ResearchStatusTool::new(store.clone());
        let cwd = std::env::current_dir().unwrap();
        let r = tool
            .execute(json!({"research_id":"does-not-exist"}), &cwd)
            .await;
        assert!(r.is_error);
    }

    #[test]
    fn strip_attribution_drops_leading_source_lines() {
        let s = "Nguồn: alonhadat.com.vn, đăng 18/04/2026\n\
                 2BR, 70m², District 7. Contact: 0912345678";
        let out = strip_source_attribution(s);
        assert!(!out.contains("alonhadat"), "got: {out}");
        assert!(out.contains("0912345678"));
        assert!(out.starts_with("2BR"));
    }

    #[test]
    fn strip_attribution_drops_trailing_clause_after_em_dash() {
        let s = "70m² fully furnished — Nguồn: chotot.com";
        let out = strip_source_attribution(s);
        assert!(!out.to_lowercase().contains("nguồn"));
        assert!(out.starts_with("70m²"));
        assert!(out.ends_with("furnished"));
    }

    #[test]
    fn strip_attribution_drops_relative_time_trailers() {
        let s = "Apartment listing\nđăng 3 ngày trước";
        let out = strip_source_attribution(s);
        assert!(!out.contains("ngày trước"), "got: {out}");
        assert_eq!(out.trim(), "Apartment listing");
    }

    #[test]
    fn strip_attribution_preserves_unrelated_text() {
        let s = "First line\nSecond line with phone 0987654321";
        let out = strip_source_attribution(s);
        assert_eq!(out, s);
    }

    #[tokio::test]
    async fn research_save_strips_source_attribution_from_excerpt() {
        let (_tmp, store, ctx) = setup();
        store.create_spec(&make_spec("rs1")).await.unwrap();
        ctx.set_id(Some("rs1".into()));
        let tool = ResearchSaveTool::new(store.clone(), ctx, Default::default());
        let cwd = std::env::current_dir().unwrap();
        let body = "70m², District 1, fully furnished. Contact Ms. Lan 0912345678 \
                    (Zalo). Available May 1.";
        let raw = format!("Nguồn: alonhadat.com.vn, đăng 18/04/2026\n{body}");
        let r = tool
            .execute(
                json!({"url":"https://ex.com/strip","title":"T","price":"$500","excerpt":raw}),
                &cwd,
            )
            .await;
        assert!(!r.is_error, "{}", r.output);
        let findings = store.list_findings("rs1", Some(10)).await.unwrap();
        let stored = findings[0].excerpt.as_deref().unwrap_or("");
        assert!(!stored.contains("alonhadat"), "got: {stored}");
        assert!(stored.contains("0912345678"));
    }

    #[test]
    fn redact_strips_api_key_and_bearer() {
        let s = "log: api_key=sk-abc123 done; Authorization: Bearer eyJhbGci OK";
        let out = scan_and_redact(s);
        assert!(out.contains("api_key=[redacted]"), "got: {out}");
        assert!(
            out.contains("Bearer [redacted]") || out.contains("bearer [redacted]"),
            "got: {out}"
        );
        assert!(!out.contains("sk-abc123"), "got: {out}");
        assert!(!out.contains("eyJhbGci"), "got: {out}");
    }

    #[test]
    fn redact_ignores_unrelated_text() {
        let s = "normal log line with https://example.com/path and nothing sensitive";
        let out = scan_and_redact(s);
        assert_eq!(out, s);
    }

    #[test]
    fn redact_handles_quoted_values() {
        let s = r#"config: secret="topsecret" and token = "xyz""#;
        let out = scan_and_redact(s);
        assert!(!out.contains("topsecret"), "got: {out}");
        assert!(!out.contains("xyz"), "got: {out}");
    }
}
