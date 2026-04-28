//! Pure rendering of the per-spec scheduler-state view.
//!
//! Powers the `/research state <id>` Telegram command and any other
//! introspection surface that wants a single human-readable snapshot
//! of *one* research spec — schedule, in-flight ledger, recent runs,
//! failure streak, pause status.
//!
//! Pure: no IO, no clocks taken implicitly. The caller passes the
//! `now` instant so the function is fully deterministic in tests.
//! Splitting render from data-loading keeps SOLID/SRP intact and lets
//! us snapshot-test the human format without spinning up a store.

use chrono::{DateTime, Utc};

use super::inflight::{Inflight, RunState};
use super::spec::{ResearchSpec, RunRecord};

/// Snapshot of all the per-spec data the state view wants to render.
///
/// Constructed by the bot command handler from the live store; the
/// pure [`render_state`] function does the formatting. Kept as a
/// flat struct (not separate function args) so adding a new field —
/// say, `paused_reason` later in Phase 2.2 — is a one-liner and the
/// signature stays small.
#[derive(Debug, Clone)]
pub struct StateView<'a> {
    /// The spec under inspection.
    pub spec: &'a ResearchSpec,
    /// Latest in-flight ledger record, if any. `None` for specs that
    /// have never been scheduled.
    pub inflight: Option<&'a Inflight>,
    /// Most-recent runs (newest first). The renderer prints up to
    /// `recent_runs_limit` of these — pass at least 5 in production.
    pub recent_runs: &'a [RunRecord],
    /// Cap on the number of `recent_runs` to render. Lets the caller
    /// cheaply choose between a compact (`/research state`) and a
    /// verbose (`/research metrics`) presentation while sharing the
    /// same render code.
    pub recent_runs_limit: usize,
    /// Current consecutive-failure streak from the in-memory
    /// scheduler state. `0` means the last attempt succeeded (or no
    /// runs yet).
    pub failure_streak: u32,
    /// `true` if the alert threshold has already fired for the
    /// current streak. Surfaces "alert sent" so an operator can tell
    /// silent-failing specs apart from never-noticed ones.
    pub alert_fired: bool,
    /// Total findings on disk for this spec. The view shows it as
    /// part of the summary header.
    pub total_findings: u64,
    /// Wall-clock anchor for relative-age formatting ("X min ago").
    /// Tests pass a fixed instant for determinism.
    pub now: DateTime<Utc>,
}

/// Render the state view as a Telegram-ready monospace string.
///
/// Format is intentionally human-skimmable in TG and grep-friendly in
/// `journalctl` (one fact per line, fixed prefixes). It is **not**
/// part of any on-disk contract — feel free to rephrase.
pub fn render_state(view: &StateView<'_>) -> String {
    let mut out = String::with_capacity(512);
    out.push_str("🔬 state for `");
    out.push_str(&view.spec.id);
    out.push_str("`\n");
    out.push_str("topic: ");
    out.push_str(&view.spec.topic);
    out.push('\n');
    out.push_str("paused: ");
    if view.spec.paused {
        out.push_str("yes");
        if let Some(reason) = view.spec.pause_reason.as_deref()
            && !reason.is_empty()
        {
            out.push_str(" (");
            out.push_str(reason);
            out.push(')');
        }
    } else {
        out.push_str("no");
    }
    out.push('\n');
    out.push_str("schedule: ");
    out.push_str(&render_schedule(view.spec));
    out.push('\n');
    out.push_str(&format!("findings: {}\n", view.total_findings));

    out.push('\n');
    out.push_str("── inflight ──\n");
    match view.inflight {
        None => out.push_str("(no inflight ledger — never scheduled)\n"),
        Some(infl) => out.push_str(&render_inflight(infl, view.now)),
    }

    out.push('\n');
    out.push_str("── failure streak ──\n");
    if view.failure_streak == 0 {
        out.push_str("0 (healthy)\n");
    } else {
        out.push_str(&format!(
            "{} consecutive failure(s){}\n",
            view.failure_streak,
            if view.alert_fired {
                " · alert already sent"
            } else {
                ""
            },
        ));
    }

    out.push('\n');
    out.push_str("── recent runs ──\n");
    if view.recent_runs.is_empty() {
        out.push_str("(none)\n");
    } else {
        for r in view.recent_runs.iter().take(view.recent_runs_limit) {
            out.push_str(&render_run_row(r, view.now));
            out.push('\n');
        }
    }
    out
}

fn render_schedule(spec: &ResearchSpec) -> String {
    if let Some(at) = spec.run_at {
        return format!("one-shot at {}", at.to_rfc3339());
    }
    if let Some(c) = &spec.cron {
        return format!("cron `{c}`");
    }
    if let Some(secs) = spec.interval_seconds {
        return format!("every {}s", secs);
    }
    "manual".to_string()
}

fn render_inflight(infl: &Inflight, now: DateTime<Utc>) -> String {
    let mut s = String::new();
    let icon = match infl.state {
        RunState::Scheduled => "🟡",
        RunState::Running => "🔵",
        RunState::Completed => "✅",
        RunState::Failed => "❌",
    };
    s.push_str(&format!(
        "{icon} {}  (attempt {})\n",
        infl.state.ru_label(),
        infl.attempt,
    ));
    s.push_str(&format!("attempt_id: {}\n", infl.attempt_id));
    s.push_str(&format!(
        "scheduled_at: {} ({} ago)\n",
        infl.scheduled_at.to_rfc3339(),
        format_age_secs(now, infl.scheduled_at),
    ));
    if let Some(t) = infl.started_at {
        s.push_str(&format!(
            "started_at:   {} ({} ago)\n",
            t.to_rfc3339(),
            format_age_secs(now, t),
        ));
    }
    if let Some(t) = infl.last_heartbeat {
        s.push_str(&format!(
            "heartbeat:    {} ({} ago)\n",
            t.to_rfc3339(),
            format_age_secs(now, t),
        ));
    }
    if let Some(t) = infl.finished_at {
        s.push_str(&format!(
            "finished_at:  {} ({} ago)\n",
            t.to_rfc3339(),
            format_age_secs(now, t),
        ));
    }
    if infl.scheduled_after_resurrection {
        s.push_str("flag: scheduled_after_resurrection\n");
    }
    if let Some(rid) = &infl.run_id {
        s.push_str(&format!("run_id: {rid}\n"));
    }
    if let Some(err) = &infl.error
        && !err.is_empty()
    {
        let short: String = err.chars().take(200).collect();
        s.push_str(&format!("error: {short}\n"));
    }
    s
}

fn render_run_row(r: &RunRecord, now: DateTime<Utc>) -> String {
    let mut s = format!(
        "· {}  (+{} new, total {})  stop={}",
        format_age_secs(now, r.finished_at),
        r.new_findings,
        r.total_findings_after,
        r.stop_reason,
    );
    if let Some(rounds) = r.verification_rounds {
        s.push_str(&format!(
            "  verify={rounds}rd, removed={}, replaced={}",
            r.dead_removed.unwrap_or(0),
            r.replacements_found.unwrap_or(0),
        ));
    }
    s
}

fn format_age_secs(now: DateTime<Utc>, t: DateTime<Utc>) -> String {
    let secs = (now - t).num_seconds();
    if secs < 0 {
        return format!("in {}s", -secs);
    }
    let secs = secs as u64;
    if secs < 60 {
        format!("{secs}s ago")
    } else if secs < 3600 {
        format!("{}m{}s ago", secs / 60, secs % 60)
    } else if secs < 86_400 {
        format!("{}h{}m ago", secs / 3600, (secs % 3600) / 60)
    } else {
        format!("{}d{}h ago", secs / 86_400, (secs % 86_400) / 3600)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::TimeZone;

    fn fixed_now() -> DateTime<Utc> {
        Utc.with_ymd_and_hms(2026, 4, 20, 12, 0, 0).unwrap()
    }

    fn base_spec(id: &str) -> ResearchSpec {
        ResearchSpec {
            id: id.into(),
            topic: format!("topic for {id}"),
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
            created_at: fixed_now(),
            paused: false,
            pause_reason: None,
        }
    }

    #[test]
    fn render_handles_spec_without_inflight() {
        let spec = base_spec("none-inflight");
        let view = StateView {
            spec: &spec,
            inflight: None,
            recent_runs: &[],
            recent_runs_limit: 5,
            failure_streak: 0,
            alert_fired: false,
            total_findings: 0,
            now: fixed_now(),
        };
        let s = render_state(&view);
        assert!(s.contains("none-inflight"), "must mention spec id");
        assert!(s.contains("topic for none-inflight"));
        assert!(s.contains("paused: no"));
        assert!(s.contains("schedule: manual"));
        assert!(s.contains("findings: 0"));
        assert!(
            s.contains("(no inflight ledger"),
            "missing inflight branch must be explicit"
        );
        assert!(s.contains("0 (healthy)"));
        assert!(s.contains("(none)"));
    }

    #[test]
    fn render_shows_paused_status_and_cron_schedule() {
        let mut spec = base_spec("paused-cron");
        spec.paused = true;
        spec.cron = Some("0 9 * * MON".into());
        let view = StateView {
            spec: &spec,
            inflight: None,
            recent_runs: &[],
            recent_runs_limit: 5,
            failure_streak: 0,
            alert_fired: false,
            total_findings: 7,
            now: fixed_now(),
        };
        let s = render_state(&view);
        assert!(s.contains("paused: yes"));
        assert!(s.contains("cron `0 9 * * MON`"));
        assert!(s.contains("findings: 7"));
    }

    #[test]
    fn render_renders_running_inflight_with_heartbeat_age() {
        let now = fixed_now();
        let spec = base_spec("running");
        let mut infl = Inflight::scheduled("running", 2);
        infl.scheduled_at = now - chrono::Duration::seconds(120);
        infl.mark_running();
        infl.started_at = Some(now - chrono::Duration::seconds(60));
        infl.last_heartbeat = Some(now - chrono::Duration::seconds(15));
        let view = StateView {
            spec: &spec,
            inflight: Some(&infl),
            recent_runs: &[],
            recent_runs_limit: 5,
            failure_streak: 0,
            alert_fired: false,
            total_findings: 0,
            now,
        };
        let s = render_state(&view);
        assert!(s.contains("В работе"), "must show ru label for Running");
        assert!(s.contains("attempt 2"));
        assert!(s.contains("15s ago"), "heartbeat age must render seconds");
        assert!(s.contains("1m0s ago"), "started_at age must render minutes");
    }

    #[test]
    fn render_includes_failure_streak_and_alert_flag() {
        let spec = base_spec("failing");
        let view = StateView {
            spec: &spec,
            inflight: None,
            recent_runs: &[],
            recent_runs_limit: 5,
            failure_streak: 3,
            alert_fired: true,
            total_findings: 0,
            now: fixed_now(),
        };
        let s = render_state(&view);
        assert!(s.contains("3 consecutive failure(s)"));
        assert!(s.contains("alert already sent"));
    }

    #[test]
    fn render_shows_scheduled_after_resurrection_flag() {
        let now = fixed_now();
        let spec = base_spec("res");
        let mut infl = Inflight::scheduled("res", 2);
        infl.scheduled_at = now - chrono::Duration::seconds(5);
        infl.scheduled_after_resurrection = true;
        let view = StateView {
            spec: &spec,
            inflight: Some(&infl),
            recent_runs: &[],
            recent_runs_limit: 5,
            failure_streak: 0,
            alert_fired: false,
            total_findings: 0,
            now,
        };
        let s = render_state(&view);
        assert!(s.contains("scheduled_after_resurrection"));
    }

    #[test]
    fn render_caps_recent_runs_at_limit() {
        let now = fixed_now();
        let spec = base_spec("many-runs");
        let runs: Vec<RunRecord> = (0..10)
            .map(|i| RunRecord {
                run_id: format!("r{i}"),
                spec_id: "many-runs".into(),
                started_at: now - chrono::Duration::seconds(i * 60 + 10),
                finished_at: now - chrono::Duration::seconds(i * 60),
                new_findings: i as u32,
                total_findings_after: i as u32,
                stop_reason: "agent_idle".into(),
                provider: "mock".into(),
                model: "mock".into(),
                verification_rounds: None,
                dead_removed: None,
                replacements_found: None,
                remaining_issues: None,
                elapsed_secs: Some(10),
            })
            .collect();
        let view = StateView {
            spec: &spec,
            inflight: None,
            recent_runs: &runs,
            recent_runs_limit: 3,
            failure_streak: 0,
            alert_fired: false,
            total_findings: 9,
            now,
        };
        let s = render_state(&view);
        // Should mention r0, r1, r2 (by their stop_reason+age) but NOT r9.
        assert_eq!(
            s.matches("stop=agent_idle").count(),
            3,
            "must cap at recent_runs_limit"
        );
    }

    #[test]
    fn render_failed_inflight_includes_error_message() {
        let now = fixed_now();
        let spec = base_spec("failed");
        let mut infl = Inflight::scheduled("failed", 1);
        infl.scheduled_at = now - chrono::Duration::seconds(60);
        infl.mark_failed("provider stream returned 503 after 4 retries");
        let view = StateView {
            spec: &spec,
            inflight: Some(&infl),
            recent_runs: &[],
            recent_runs_limit: 5,
            failure_streak: 1,
            alert_fired: false,
            total_findings: 0,
            now,
        };
        let s = render_state(&view);
        assert!(s.contains("Ошибка"));
        assert!(s.contains("provider stream returned 503"));
    }

    #[test]
    fn render_includes_pause_reason_when_present() {
        let mut spec = base_spec("auto-paused");
        spec.paused = true;
        spec.pause_reason = Some("auto: 5 consecutive failures — last error: stream closed".into());
        let view = StateView {
            spec: &spec,
            inflight: None,
            recent_runs: &[],
            recent_runs_limit: 5,
            failure_streak: 0,
            alert_fired: false,
            total_findings: 0,
            now: fixed_now(),
        };
        let s = render_state(&view);
        assert!(s.contains("paused: yes"));
        assert!(
            s.contains("auto: 5 consecutive failures"),
            "pause_reason must surface in the state view"
        );
    }

    #[test]
    fn render_paused_without_reason_shows_only_yes() {
        let mut spec = base_spec("manual-pause");
        spec.paused = true;
        spec.pause_reason = None;
        let view = StateView {
            spec: &spec,
            inflight: None,
            recent_runs: &[],
            recent_runs_limit: 5,
            failure_streak: 0,
            alert_fired: false,
            total_findings: 0,
            now: fixed_now(),
        };
        let s = render_state(&view);
        assert!(s.contains("paused: yes"));
        assert!(!s.contains("paused: yes ("), "no parens without a reason");
    }

    #[test]
    fn format_age_handles_future_negative_delta() {
        let now = fixed_now();
        let future = now + chrono::Duration::seconds(30);
        let s = format_age_secs(now, future);
        assert!(s.contains("in 30s"), "future timestamps render as 'in Ns'");
    }

    #[test]
    fn format_age_renders_units_correctly() {
        let now = fixed_now();
        assert_eq!(
            format_age_secs(now, now - chrono::Duration::seconds(45)),
            "45s ago"
        );
        assert_eq!(
            format_age_secs(now, now - chrono::Duration::seconds(125)),
            "2m5s ago"
        );
        assert_eq!(
            format_age_secs(now, now - chrono::Duration::seconds(3700)),
            "1h1m ago"
        );
        assert_eq!(
            format_age_secs(now, now - chrono::Duration::seconds(90_000)),
            "1d1h ago"
        );
    }
}
