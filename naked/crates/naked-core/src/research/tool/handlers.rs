//! `Tool` trait implementations for all research tools.

use std::path::Path;

use async_trait::async_trait;
use chrono::Utc;
use serde_json::{Value, json};

use crate::research::run_events::{EventKind, RunEvent};
use crate::research::spec::Cursor;
use crate::research::validators;
use crate::tool::Tool;
use crate::types::{Permission, ToolResult, ToolSpec};

use super::output::{build_finding, is_stale, parse_listing_date};
use super::{
    MAX_LISTING_AGE_DAYS, ResearchListTool, ResearchSaveCursorTool, ResearchSaveTool,
    ResearchStatusTool,
};

// ── ResearchSaveTool ─────────────────────────────────────────────────────────

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
            return ToolResult::err(
                "no active research context — call this tool only inside /research run",
            );
        };
        let url = match input.get("url").and_then(|v| v.as_str()) {
            Some(s) if !s.trim().is_empty() => s.trim().to_string(),
            _ => return ToolResult::err("`url` is required and must be non-empty"),
        };
        if !(url.starts_with("http://") || url.starts_with("https://")) {
            return ToolResult::err(format!(
                "`url` must be absolute http(s) — got `{}`",
                url.chars().take(80).collect::<String>()
            ));
        }

        // Reject stale listings (> 90 days old) at save time
        let listing_date_raw = input.get("listing_date").and_then(|v| v.as_str());
        if let Some(true) = is_stale(listing_date_raw) {
            let date_str = listing_date_raw.unwrap_or("?");
            return ToolResult::ok(format!(
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
        if let Some(marker) = validators::looks_like_captcha_stub(&combined) {
            // Mirror the waterfall so the operator sees WHY the save
            // was dropped without grepping the log.
            if let Some(reg) = self.context.run_events() {
                reg.push(
                    &id,
                    RunEvent::new(
                        EventKind::BlockDetected,
                        format!("research_save rejected: captcha stub on {url}"),
                    ),
                )
                .await;
            }
            return ToolResult::ok(format!(
                "{{\"stored\":false,\"skipped\":true,\"reason\":\"captcha_stub (matched `{marker}`) — rerun via Skill(web-browser-playbook) + browser_navigate on {url}\"}}"
            ));
        }

        let run_id = self
            .context
            .run_id()
            .unwrap_or_else(|| "manual".to_string());
        let finding = build_finding(&id, &run_id, &url, &input);

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
                    reg.push(&id, RunEvent::new(EventKind::FindingSaved, label))
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
                    ToolResult::ok(format!(
                        "{{\"stored\":true,\"updated\":true,\"total_findings\":{total}{warn_json}}}"
                    ))
                } else {
                    ToolResult::ok(format!(
                        "{{\"stored\":true,\"duplicate\":false,\"total_findings\":{total}{warn_json}}}"
                    ))
                }
            }
            Err(e) => ToolResult::err(format!("store error: {e}")),
        }
    }
}

// ── ResearchListTool ──────────────────────────────────────────────────────────

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
            return ToolResult::err("no active research context");
        };
        let limit = input
            .get("limit")
            .and_then(|v| v.as_u64())
            .map(|n| n.min(200) as usize)
            .unwrap_or(20);
        match self.store.list_findings(&id, Some(limit)).await {
            Ok(list) if list.is_empty() => ToolResult::ok("(no findings yet)".to_string()),
            Ok(list) => {
                let mut out = String::new();
                for f in list.iter().rev() {
                    let title = f.title.as_deref().unwrap_or("(untitled)");
                    out.push_str(&format!("- {} — {}\n", title, f.url));
                }
                ToolResult::ok(out)
            }
            Err(e) => ToolResult::err(format!("store error: {e}")),
        }
    }
}

// ── ResearchSaveCursorTool ────────────────────────────────────────────────────

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
            return ToolResult::err("no active research context");
        };
        let cursor_val = match input.get("cursor") {
            Some(Value::Object(m)) => m.clone(),
            _ => return ToolResult::err("`cursor` must be a JSON object"),
        };
        let cursor = Cursor {
            data: cursor_val,
            updated_at: None,
        };
        match self.store.save_cursor(&id, &cursor).await {
            Ok(()) => ToolResult::ok("cursor saved".to_string()),
            Err(e) => ToolResult::err(format!("store error: {e}")),
        }
    }
}

// ── ResearchStatusTool ────────────────────────────────────────────────────────

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
            return ToolResult::err("`research_id` is required");
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
            Err(e) => return ToolResult::err(format!("unknown research `{id}`: {e}")),
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
        ToolResult::ok(out)
    }
}
