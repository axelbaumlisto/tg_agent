//! Prompt building, report generation, agent briefs, run records.

use super::ResearchCoordinator;
use super::ResearchSpec;
use super::RunRecord;
use super::RunReport;
use super::StopReason;
use super::VerificationSummary;
use crate::error::Result;
use crate::research::spec::{Cursor, Finding};
use chrono::Utc;

const RESEARCH_PROMPT_TEMPLATE: &str = include_str!("research_prompt.md");

struct PromptVars<'a> {
    id: &'a str,
    topic: &'a str,
    sources: &'a str,
    brief_section: &'a str,
    dedup_list: &'a str,
    cursor_json: &'a str,
    max_nav: u32,
    today: &'a str,
}

fn render_research_prompt(vars: PromptVars<'_>) -> String {
    RESEARCH_PROMPT_TEMPLATE
        .replace("{id}", vars.id)
        .replace("{topic}", vars.topic)
        .replace("{sources}", vars.sources)
        .replace("{brief_section}", vars.brief_section)
        .replace("{dedup_list}", vars.dedup_list)
        .replace("{cursor_json}", vars.cursor_json)
        .replace("{max_nav}", &vars.max_nav.to_string())
        .replace("{today}", vars.today)
}

fn format_dedup_list(known: &[Finding]) -> String {
    if known.is_empty() {
        return "  (none yet — first pass)".to_string();
    }
    known
        .iter()
        .rev()
        .take(50)
        .map(|f| {
            let title = f.title.as_deref().unwrap_or("(untitled)");
            let price = f.price.as_deref().unwrap_or("?");
            let date = f.listing_date.as_deref().unwrap_or("no date");
            format!("  - {title} | {price} | {date} | {}", f.url)
        })
        .collect::<Vec<_>>()
        .join("\n")
}

fn format_cursor_json(cursor: &Cursor) -> String {
    if cursor.data.is_empty() {
        "(empty)".to_string()
    } else {
        serde_json::to_string(&cursor.data).unwrap_or_else(|_| "(unparseable)".to_string())
    }
}

fn format_sources(sources: &[String]) -> String {
    if sources.is_empty() {
        "(no seeds — use web_search or your best guesses)".to_string()
    } else {
        sources
            .iter()
            .map(|s| format!("  - {s}"))
            .collect::<Vec<_>>()
            .join("\n")
    }
}

fn format_brief_section(agent_brief: Option<&str>) -> String {
    match agent_brief {
        Some(b) => format!(
            "# Brief from previous run\n\
             The following is a structured summary of the previous run. Use it to \
             focus on NEW listings, deeper pages, or sources that were not yet explored.\n\
             \n{b}\n"
        ),
        None => String::new(),
    }
}

pub(crate) struct WriteRecordArgs<'a> {
    pub(crate) spec: &'a ResearchSpec,
    pub(crate) run_id: &'a str,
    pub(crate) new_findings: u32,
    pub(crate) reason: StopReason,
    pub(crate) started: std::time::Instant,
    pub(crate) provider: &'a str,
    pub(crate) model: &'a str,
    pub(crate) verification: Option<VerificationSummary>,
}

impl ResearchCoordinator {
    /// Compose the agent-facing prompt. Short but dense: spec topic, source
    /// seeds, last-N known-finding URLs for dedup, cursor JSON, and a strict
    /// rule block describing the save/list/done tool contract.
    pub(crate) async fn build_prompt(&self, spec: &ResearchSpec) -> Result<String> {
        let known = self.store.list_findings(&spec.id, Some(50)).await?;
        let cursor = self.store.load_cursor(&spec.id).await?;
        let agent_brief = self.store.read_agent_brief(&spec.id).await.unwrap_or(None);

        let dedup_list = format_dedup_list(&known);
        let cursor_json = format_cursor_json(&cursor);
        let sources = format_sources(&spec.sources);

        let max_nav = spec
            .max_iterations
            .unwrap_or(self.config.default_max_iterations);

        let today = Utc::now().format("%d/%m/%Y").to_string();

        let brief_section = format_brief_section(agent_brief.as_deref());

        Ok(render_research_prompt(PromptVars {
            id: &spec.id,
            topic: &spec.topic,
            sources: &sources,
            brief_section: &brief_section,
            dedup_list: &dedup_list,
            cursor_json: &cursor_json,
            max_nav,
            today: &today,
        }))
    }

    pub(crate) async fn regenerate_report(&self, spec: &ResearchSpec) -> Result<()> {
        let findings = self.store.list_findings(&spec.id, None).await?;
        let runs = self.store.list_runs(&spec.id, Some(10)).await?;
        let mut md = String::new();
        md.push_str(&format!("# {}\n\n", spec.topic));
        md.push_str(&format!("_Research id: `{}`_\n\n", spec.id));
        md.push_str(&format!("**Total findings:** {}\n\n", findings.len()));

        md.push_str("## Latest findings\n\n");
        if findings.is_empty() {
            md.push_str("_(none yet)_\n\n");
        } else {
            for f in findings.iter().rev().take(25) {
                let title = f.title.as_deref().unwrap_or("(untitled)");
                md.push_str(&format!("- [{}]({})", title, f.url));
                if let Some(p) = &f.price {
                    md.push_str(&format!(" — **{p}**"));
                }
                if let Some(d) = &f.listing_date {
                    md.push_str(&format!(" _{d}_"));
                }
                md.push('\n');
                if let Some(ex) = &f.excerpt
                    && !ex.is_empty()
                {
                    md.push_str(&format!("  > {ex}\n"));
                }
            }
            md.push('\n');
        }

        md.push_str("## Run history (last 10)\n\n");
        if runs.is_empty() {
            md.push_str("_(none)_\n\n");
        } else {
            md.push_str("| Started | Provider | Model | New | Total | Reason |\n");
            md.push_str("|---------|----------|-------|----:|------:|--------|\n");
            for r in runs.iter().rev() {
                md.push_str(&format!(
                    "| {} | {} | {} | {} | {} | {} |\n",
                    r.started_at.format("%Y-%m-%d %H:%M UTC"),
                    r.provider,
                    r.model,
                    r.new_findings,
                    r.total_findings_after,
                    r.stop_reason,
                ));
            }
            md.push('\n');
        }

        self.store.write_report(&spec.id, &md).await
    }

    /// Generate a structured brief for the next agent run. Contains all findings
    /// with full detail, run history summary, and actionable guidance so the next
    /// agent knows what was already collected and where to focus.
    pub(crate) async fn regenerate_agent_brief(&self, spec: &ResearchSpec) -> Result<()> {
        let findings = self.store.list_findings(&spec.id, None).await?;
        let runs = self.store.list_runs(&spec.id, Some(5)).await?;

        let mut md = String::new();
        md.push_str(&format!(
            "## Collected findings ({} total)\n\n",
            findings.len()
        ));

        if findings.is_empty() {
            md.push_str("No findings yet.\n\n");
        } else {
            for (i, f) in findings.iter().rev().enumerate() {
                let title = f.title.as_deref().unwrap_or("(untitled)");
                let price = f.price.as_deref().unwrap_or("—");
                let date = f.listing_date.as_deref().unwrap_or("—");
                md.push_str(&format!("### {}. {} — {}\n", i + 1, title, price));
                md.push_str(&format!("- URL: {}\n", f.url));
                md.push_str(&format!("- Date: {}\n", date));
                if let Some(ex) = &f.excerpt {
                    md.push_str(&format!("- Details: {ex}\n"));
                }
                md.push('\n');
            }
        }

        if !runs.is_empty() {
            md.push_str("## Run history\n\n");
            for r in runs.iter().rev() {
                md.push_str(&format!(
                    "- {} via {}/{}: +{} new (total {}), stopped: {}\n",
                    r.started_at.format("%Y-%m-%d %H:%M UTC"),
                    r.provider,
                    r.model,
                    r.new_findings,
                    r.total_findings_after,
                    r.stop_reason,
                ));
            }
            md.push('\n');
        }

        md.push_str("## Guidance for this run\n\n");
        md.push_str(
            "- Skip all URLs already in the Known findings list.\n\
             - Focus on NEW listings posted since the last run.\n\
             - Explore deeper pages (page 2+) and alternative sources not yet tried.\n\
             - Prioritize listings with recent dates.\n",
        );

        self.store.write_agent_brief(&spec.id, &md).await
    }

    pub(crate) async fn write_record(&self, args: WriteRecordArgs<'_>) -> Result<RunReport> {
        let total = self.store.count_findings(&args.spec.id).await.unwrap_or(0);
        let elapsed = args.started.elapsed();
        let elapsed_secs = elapsed.as_secs();
        let record = RunRecord {
            run_id: args.run_id.to_string(),
            spec_id: args.spec.id.clone(),
            started_at: Utc::now() - chrono::Duration::from_std(elapsed).unwrap_or_default(),
            finished_at: Utc::now(),
            new_findings: args.new_findings,
            total_findings_after: total,
            stop_reason: args.reason.as_str().to_string(),
            provider: args.provider.to_string(),
            model: args.model.to_string(),
            verification_rounds: args.verification.as_ref().map(|v| v.rounds),
            dead_removed: args.verification.as_ref().map(|v| v.dead_removed),
            replacements_found: args.verification.as_ref().map(|v| v.replacements_found),
            remaining_issues: args.verification.as_ref().map(|v| v.remaining_issues),
            elapsed_secs: Some(elapsed_secs),
        };
        if let Err(e) = self.store.append_run(&record).await {
            tracing::warn!(spec = %args.spec.id, "failed to append run record: {e}");
        }
        Ok(RunReport {
            spec_id: args.spec.id.clone(),
            run_id: args.run_id.to_string(),
            new_findings: args.new_findings,
            total_findings_after: total,
            stop_reason: args.reason,
            elapsed,
            provider: args.provider.to_string(),
            model: args.model.to_string(),
        })
    }
}
