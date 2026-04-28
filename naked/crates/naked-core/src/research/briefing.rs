//! Source of truth for "how does the research subsystem work" prose.
//!
//! Two surfaces consume this:
//!
//! 1. The LLM tool [`super::ResearchHelpTool`] returns [`full`] verbatim —
//!    the user can ask "как работает исследование" and get the complete
//!    explanation back.
//! 2. The system-prompt assembler in [`crate::prompt`] embeds [`short`] in
//!    every session prompt when `config.research.enabled` so the LLM has
//!    enough context to answer follow-up questions without making a
//!    tool call (e.g. "когда B запустится?" can be answered after one
//!    `research_metrics` call instead of needing a separate help fetch).
//!
//! Keeping both versions here means a single edit propagates to both
//! surfaces. Don't duplicate this prose elsewhere.
//!
//! Both functions accept the live [`crate::config::ResearchConfig`] so the
//! defaults that appear in the text (`max_concurrent_runs`,
//! `verify_by_default`, `gatekeeper.max_rounds`,
//! `default_interval_seconds`) match the running process — operators have
//! tweaked these in the past and we don't want stale numbers in the prose.

use crate::config::ResearchConfig;

/// Compact briefing (≈800 chars) embedded in every research-enabled session
/// prompt. Designed to give the LLM the minimum it needs to answer
/// "how does scheduling / verification / concurrency work" without a
/// tool call, while still leaving the heavy detail to `research_help`.
pub fn short(cfg: &ResearchConfig) -> String {
    let interval_h = cfg.default_interval_seconds / 3600;
    format!(
        "## Research subsystem\n\
         You can manage long-running web research from this chat. Each \
         research has an `id`, a `topic`, optional `interval_seconds` for \
         periodic background runs, and a `paused` flag.\n\
         \n\
         - **Schedule**: stored as `interval_seconds` on the spec itself. \
           An in-process Tokio scheduler scans every 30s and launches due \
           specs (default cadence when set: {interval_h}h). Paused specs \
           never fire. Schedule changes take effect on the next tick \
           because [tools] notify the scheduler immediately.\n\
         - **Concurrency**: at most {max_concurrent} run(s) at a time across \
           the whole process — manual `/research run`, the `research_launch` \
           tool, and the scheduler share one semaphore. If a run is in \
           flight, others queue.\n\
         - **Verification**: `verify_by_default = {verify}`; when on, every \
           run goes through up to {max_rounds} gatekeeper rounds (URL \
           liveness, listing-date freshness, semantic dedup) before being \
           accepted. Bad findings are removed and a feedback prompt asks \
           the agent for replacements.\n\
         - **Where data lives**: `$NAKED_HOME/research/<id>/` with \
           `spec.json`, `findings.jsonl`, `runs.jsonl`, `report.md`, \
           `cursor.json`. After a successful run a one-line summary is \
           appended to global `MEMORY.md` so you can recall it later.\n\
         \n\
         To answer a user question about a specific research, prefer \
         calling `research_metrics` for the current `interval_seconds`, \
         `paused` state, run history, and `report.md` excerpt. Use \
         `research_help` only when the user asks how the system works in \
         general.\n",
        interval_h = interval_h,
        max_concurrent = cfg.max_concurrent_runs.max(1),
        verify = cfg.verify_by_default,
        max_rounds = cfg.gatekeeper.max_rounds,
    )
}

/// Full briefing (≈3 KB) returned verbatim by the `research_help` LLM tool.
/// Adds the storage layout, every LLM tool name with one-line purpose, and
/// the relationship between in-process and systemd scheduling.
pub fn full(cfg: &ResearchConfig) -> String {
    let interval_h = cfg.default_interval_seconds / 3600;
    format!(
        "# Research subsystem — operator and user guide\n\
         \n\
         The research subsystem performs long-running web investigations: \
         the agent walks the web, accumulates deduplicated findings, emits \
         a rolling `report.md`, and can be re-run on a schedule.\n\
         \n\
         ## Spec model\n\
         Each research is a `ResearchSpec` with:\n\
         - `id` (slug + 4-char suffix), `topic`, `sources` (seed URLs)\n\
         - `interval_seconds: Option<u64>` — when `Some(n)` the in-process \
           scheduler reruns every `n` seconds; `None` = manual only\n\
         - `paused: bool` — pauses scheduled runs, manual `/research run` \
           still works\n\
         - `provider`, `model`, `max_iterations`, `max_wall_seconds` — \
           per-spec overrides over `Config.research`\n\
         \n\
         ## Scheduling\n\
         - **In-process scheduler** (TG-bot path, default): a Tokio task \
           scans every 30s, launches due specs through the global \
           semaphore. The `SchedulerHook` fires on every spec mutation \
           (create / update / pause / resume / delete) so changes are \
           picked up immediately, not on the next tick. Default cadence \
           used by `default_interval_seconds`: **{interval_h}h**.\n\
         - **Systemd timer** (headless CLI path): `naked-research@<id>.timer` \
           in `ops/systemd/` runs `naked research run <id> --verify 3` on a \
           cron-style `OnCalendar`. Use only when running without the bot.\n\
         - Both paths share the same `max_concurrent_runs` semaphore on \
           `AgentCore`, so mixing them on one host is safe.\n\
         \n\
         ## Concurrency\n\
         A process-wide `tokio::sync::Semaphore` with **{max_concurrent}** \
         permit(s) gates every run. Default is 1 because Playwright \
         (browser MCP) cannot be shared across concurrent runs. Increase \
         only if your run loop avoids shared singletons. Configure via \
         `Config.research.max_concurrent_runs`.\n\
         \n\
         ## Verification (gatekeeper)\n\
         `verify_by_default = {verify}`. When on, every run is followed by \
         up to **{max_rounds}** gatekeeper rounds:\n\
         1. URL liveness probe (HEAD request, 5xx/4xx → dead).\n\
         2. Required-fields check (`title`, `price`, `excerpt ≥ N chars`, \
            `listing_date`, `source_content ≥ N chars`).\n\
         3. Listing-date freshness (max 90 days).\n\
         4. Exact + fuzzy semantic dedup (token Jaccard ≥ 0.7 + same \
            normalized price).\n\
         Dead findings are removed and the agent gets a feedback prompt \
         asking for replacements. Final stats (`verification_rounds`, \
         `dead_removed`, `replacements_found`, `remaining_issues`, \
         `elapsed_secs`) are written to `runs.jsonl`.\n\
         \n\
         ## Storage layout\n\
         `$NAKED_HOME/research/<id>/`:\n\
         - `spec.json` — the spec itself, updated by `research_update_spec`\n\
         - `findings.jsonl` — append-only, deduped by `blake3(canonicalize_url(url))`\n\
         - `runs.jsonl` — append-only run records\n\
         - `report.md` — regenerated after each run, latest 25 findings + \
           run-history table\n\
         - `cursor.json` — opaque agent state for resumable pagination\n\
         \n\
         After every successful run a `MemoryType::ProjectKnowledge` entry \
         is appended to `MEMORY.md` under `MemoryScope::Global` with run \
         id, finding counts, gatekeeper stats, elapsed secs, and a path \
         to `report.md`. Use `memory_recall` to find these summaries.\n\
         \n\
         ## LLM tools\n\
         - `research_create` — start a new research spec (topic + sources)\n\
         - `research_list_specs` — list all specs with their schedule + \
           latest run summary\n\
         - `research_metrics` — detailed metrics for one spec (recommended \
           for any 'how is X going / what's its schedule' question)\n\
         - `research_findings` — list saved findings for a spec\n\
         - `research_launch` — kick off a one-off run (uses verification \
           by default if `verify_by_default = true`)\n\
         - `research_update_spec` — partial mutation (topic / sources / \
           schedule / provider / model / paused). Sources can be appended \
           or replaced.\n\
         - `research_set_schedule` — focused tool just for `interval_seconds` \
           and pause toggling; re-arms the scheduler immediately.\n\
         - `research_pause` / `research_resume` — narrow wrappers\n\
         - `research_help` — return this guide\n\
         \n\
         ## Bot commands (CLI surface)\n\
         `/research help` lists every command. The most useful ones:\n\
         - `/research ls` — schedule + last-run summary per spec\n\
         - `/research show <id>` — full spec + recent findings\n\
         - `/research metrics <id>` — same data as `research_metrics`\n\
         - `/research run <id>` — manual run (verified by default)\n\
         - `/research ask <id> <q>` — LLM Q&A grounded in the findings\n\
         - `/research schedule <id> on <30s|15m|1h|1d|N>|off|status`\n\
         \n\
         For a specific research's current state always prefer the \
         `research_metrics` tool over guessing.\n",
        interval_h = interval_h,
        max_concurrent = cfg.max_concurrent_runs.max(1),
        verify = cfg.verify_by_default,
        max_rounds = cfg.gatekeeper.max_rounds,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::ResearchConfig;

    #[test]
    fn short_briefing_substitutes_config_values() {
        let cfg = ResearchConfig {
            max_concurrent_runs: 4,
            verify_by_default: false,
            default_interval_seconds: 7_200,
            ..Default::default()
        };
        let s = short(&cfg);
        assert!(s.contains("at most 4 run"), "max_concurrent: {s}");
        assert!(s.contains("verify_by_default = false"));
        assert!(s.contains("when set: 2h"), "default interval: {s}");
    }

    #[test]
    fn short_briefing_clamps_zero_concurrent_to_one() {
        let cfg = ResearchConfig {
            max_concurrent_runs: 0,
            ..Default::default()
        };
        let s = short(&cfg);
        assert!(
            s.contains("at most 1 run"),
            "must clamp 0 to 1 in display: {s}"
        );
    }

    #[test]
    fn full_briefing_lists_every_orchestration_tool() {
        let s = full(&ResearchConfig::default());
        for tool in [
            "research_create",
            "research_list_specs",
            "research_metrics",
            "research_findings",
            "research_launch",
            "research_update_spec",
            "research_set_schedule",
            "research_pause",
            "research_resume",
            "research_help",
        ] {
            assert!(s.contains(tool), "missing tool `{tool}` in full briefing");
        }
    }

    #[test]
    fn full_briefing_mentions_in_process_and_systemd_paths() {
        let s = full(&ResearchConfig::default());
        assert!(s.contains("In-process scheduler"));
        assert!(s.contains("Systemd timer"));
        assert!(s.contains("max_concurrent_runs"));
    }
}
