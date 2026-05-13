// REGISTRY-WAIVE: B45 (dead-code revealed by Phase D' pub→pub(crate) flip).
// PLAN_SKILL_VS_CORE_v1 audit found 55 items in research/ never used inside
// the crate; they were hidden by `pub` visibility (dead_code lint exempts
// pub items). Audit + delete is queued as separate B45 cleanup task.
// Until then, this allow keeps the clippy gate green.
#![allow(dead_code)]

//! Event drain loop + per-run statistics for research coordinator.

use super::AgentEvent;
use super::AgentHandle;
use super::CancellationToken;
use super::StopReason;

#[derive(Default)]
pub struct DrainStats {
    /// Tool invocations grouped by tool name. `BTreeMap` so the printed
    /// summary is deterministic across runs.
    pub tool_counts: std::collections::BTreeMap<String, u32>,
    /// Tool-result bodies that contained one of the canonical
    /// captcha/anti-bot markers. Each occurrence is a "the agent saw a
    /// gated page" signal — high counts with low `Skill` invocations
    /// mean the prompt is not steering the agent to the playbook.
    pub captcha_hits: u32,
    /// Convenience counter pulled out of `tool_counts["Skill"]` so the
    /// summary line is grep-friendly without re-scanning the map.
    pub skill_loads: u32,
    pub text_deltas: u32,
    pub errors: u32,
}

impl DrainStats {
    /// Substrings we treat as "this page is anti-bot gated". Kept short and
    /// case-sensitive — false positives here would inflate the metric and
    /// dilute the signal we use to validate prompt changes.
    const CAPTCHA_MARKERS: &'static [&'static str] = &[
        "xac-thuc-nguoi-dung",           // alonhadat anti-bot interstitial path
        "Vui lòng xác minh",             // VN "please verify" string family
        "Just a moment",                 // Cloudflare interstitial title
        "Attention Required",            // Cloudflare 1020/blocked title
        "Enable JavaScript and cookies", // Cloudflare body line
        "cf_chl_rt_tk",                  // Cloudflare challenge token in URL
    ];

    pub(super) fn note_tool(&mut self, name: &str) {
        *self.tool_counts.entry(name.to_string()).or_default() += 1;
        if name == "Skill" {
            self.skill_loads += 1;
        }
    }

    pub(super) fn note_tool_output(&mut self, body: &str) {
        if Self::CAPTCHA_MARKERS.iter().any(|m| body.contains(m)) {
            self.captcha_hits += 1;
        }
    }

    /// Render the `[research-summary]` line. Stable, machine-parseable
    /// shape so `naked research probe` (and any future regression script)
    /// can `grep` for it without depending on tool ordering.
    pub fn summary_line(&self) -> String {
        let tools = self
            .tool_counts
            .iter()
            .map(|(k, v)| format!("{k}={v}"))
            .collect::<Vec<_>>()
            .join(",");
        format!(
            "[research-summary] tools={{{tools}}} captcha_hits={} skill_loads={} text_deltas={} errors={}",
            self.captcha_hits, self.skill_loads, self.text_deltas, self.errors
        )
    }
}

/// Build a short human-readable label for a `ToolStart` event — the
/// waterfall prefers "what is this tool doing" over "what are its raw
/// arguments". Keeps the total under ~60 chars so the heartbeat
/// message stays comfortably below Telegram's 4096-char budget even
/// when every one of the last 5 slots is a long URL.
fn tool_start_label(name: &str, input: &serde_json::Value) -> String {
    let hint = match name {
        "web_fetch" | "browser_navigate" => input
            .get("url")
            .and_then(|v| v.as_str())
            .map(short_url_host),
        "research_save" => input
            .get("url")
            .and_then(|v| v.as_str())
            .map(short_url_host),
        "web_search" | "web_search_exa" => input
            .get("query")
            .and_then(|v| v.as_str())
            .map(|q| q.chars().take(40).collect::<String>()),
        "Skill" => input
            .get("skill")
            .or_else(|| input.get("name"))
            .and_then(|v| v.as_str())
            .map(|s| s.to_string()),
        "browser_snapshot" | "browser_take_screenshot" => None,
        _ => None,
    };
    match hint {
        Some(h) if !h.is_empty() => format!("{name} {h}"),
        _ => name.to_string(),
    }
}

/// `https://www.alonhadat.com.vn/abc/xyz` → `alonhadat.com.vn/xyz`. Drops
/// scheme + `www.` and keeps only the host + last path segment so the
/// label stays readable in a 4096-char heartbeat message.
fn short_url_host(url: &str) -> String {
    let stripped = url
        .trim()
        .trim_start_matches("https://")
        .trim_start_matches("http://")
        .trim_start_matches("www.");
    let mut parts = stripped.splitn(2, '/');
    let host = parts.next().unwrap_or("").to_string();
    let rest = parts.next().unwrap_or("");
    let last_seg = rest.rsplit('/').find(|s| !s.is_empty()).unwrap_or("");
    if last_seg.is_empty() {
        host
    } else {
        let tail: String = last_seg.chars().take(28).collect();
        format!("{host}/{tail}")
    }
}

/// Drain agent events until the stream closes or the agent signals `Idle`.
/// Auto-approves every `PermissionRequest` — research runs are headless,
/// nobody is watching to click "allow".
///
/// `stats` is accumulated in place so the caller still sees partial telemetry
/// when the run is cancelled by the wall-clock timeout (the future is
/// dropped mid-loop in that case, but the borrow already wrote whatever it
/// observed).
pub async fn drain_events(
    handle: &mut AgentHandle,
    stats: &mut DrainStats,
    cancel: &CancellationToken,
    run_events: Option<&crate::research::run_events::RunEventRegistry>,
    run_id: &str,
) -> StopReason {
    let mut tool_calls = 0u32;
    loop {
        // Race the next agent event against the cancellation token. The
        // outer `select!` in `run_once`/`run_verified` already handles
        // cancellation at the run boundary, but we also check here so a
        // long-running provider stream doesn't make us hang on
        // `events.recv()` for the full wall-clock budget after the
        // scheduler asked us to stop.
        let event = tokio::select! {
            biased;
            _ = cancel.cancelled() => {
                tracing::info!(
                    tool_calls,
                    text_deltas = stats.text_deltas,
                    errors = stats.errors,
                    "drain_events: cancellation observed mid-stream"
                );
                eprintln!("  [research] cancelled mid-stream");
                return StopReason::Cancelled;
            }
            ev = handle.events.recv() => ev,
        };
        match event {
            Some(AgentEvent::Idle) => {
                tracing::info!(
                    tool_calls,
                    text_deltas = stats.text_deltas,
                    errors = stats.errors,
                    "research agent idle"
                );
                eprintln!("  {}", stats.summary_line());
                return StopReason::AgentIdle;
            }
            Some(AgentEvent::Error(e)) => {
                stats.errors += 1;
                tracing::warn!("research agent error #{}: {e}", stats.errors);
                eprintln!("  [research] error #{}: {e}", stats.errors);
            }
            Some(AgentEvent::ToolStart { name, input, .. }) => {
                tool_calls += 1;
                stats.note_tool(&name);
                eprintln!("  [research] tool #{tool_calls}: {name}");
                if let Some(reg) = run_events {
                    let label = tool_start_label(&name, &input);
                    let kind = if name == "Skill" {
                        crate::research::run_events::EventKind::SkillLoaded
                    } else {
                        crate::research::run_events::EventKind::ToolCallStart
                    };
                    reg.push(
                        run_id,
                        crate::research::run_events::RunEvent::new(kind, label),
                    )
                    .await;
                }
            }
            Some(AgentEvent::ToolEnd {
                name,
                state,
                output,
                ..
            }) => {
                stats.note_tool_output(&output);
                let preview: String = output.chars().take(200).collect();
                eprintln!("  [research] tool done: {name} state={state:?} → {preview}");
                if let Some(reg) = run_events {
                    let body_marker = DrainStats::CAPTCHA_MARKERS
                        .iter()
                        .any(|m| output.contains(m));
                    if body_marker {
                        reg.push(
                            run_id,
                            crate::research::run_events::RunEvent::new(
                                crate::research::run_events::EventKind::BlockDetected,
                                format!("{name}: captcha/anti-bot wall"),
                            ),
                        )
                        .await;
                    }
                    let out_preview: String = output
                        .chars()
                        .take(60)
                        .collect::<String>()
                        .replace('\n', " ")
                        .trim()
                        .to_string();
                    let label = if out_preview.is_empty() {
                        format!("{name} done")
                    } else {
                        format!("{name} → {out_preview}")
                    };
                    reg.push(
                        run_id,
                        crate::research::run_events::RunEvent::new(
                            crate::research::run_events::EventKind::ToolCallEnd,
                            label,
                        ),
                    )
                    .await;
                }
            }
            Some(AgentEvent::PermissionRequest {
                call_id, tool_name, ..
            }) => {
                tracing::debug!(tool = %tool_name, "auto-approving research tool");
                eprintln!("  [research] auto-approve: {tool_name}");
                let _ = handle
                    .permissions
                    .send(crate::types::PermissionResponse {
                        call_id,
                        allowed: true,
                    })
                    .await;
            }
            Some(AgentEvent::TextDelta(t)) => {
                stats.text_deltas += 1;
                if stats.text_deltas <= 3 || stats.text_deltas.is_multiple_of(50) {
                    let preview: String = t.chars().take(80).collect();
                    eprintln!("  [research] text delta #{}: {preview}", stats.text_deltas);
                }
            }
            Some(_) => continue,
            None => {
                tracing::info!(
                    tool_calls,
                    text_deltas = stats.text_deltas,
                    errors = stats.errors,
                    "research agent stream closed"
                );
                eprintln!(
                    "  [research] stream closed: {tool_calls} tools, {} text deltas, {} errors",
                    stats.text_deltas, stats.errors
                );
                eprintln!("  {}", stats.summary_line());
                return StopReason::StreamClosed;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn drain_stats_default_is_empty() {
        let s = DrainStats::default();
        assert!(s.tool_counts.is_empty());
        assert_eq!(s.captcha_hits, 0);
        assert_eq!(s.skill_loads, 0);
    }

    #[test]
    fn note_tool_counts_invocations() {
        let mut s = DrainStats::default();
        s.note_tool("bash");
        s.note_tool("bash");
        s.note_tool("read_file");
        assert_eq!(s.tool_counts["bash"], 2);
        assert_eq!(s.tool_counts["read_file"], 1);
    }

    #[test]
    fn note_tool_tracks_skill_loads() {
        let mut s = DrainStats::default();
        s.note_tool("Skill");
        s.note_tool("Skill");
        assert_eq!(s.skill_loads, 2);
    }

    #[test]
    fn note_tool_output_detects_captcha() {
        let mut s = DrainStats::default();
        s.note_tool_output("normal output");
        assert_eq!(s.captcha_hits, 0);
        s.note_tool_output("Just a moment");
        assert_eq!(s.captcha_hits, 1);
    }

    #[test]
    fn summary_line_is_nonempty() {
        let mut s = DrainStats::default();
        s.note_tool("bash");
        let line = s.summary_line();
        assert!(!line.is_empty());
        assert!(line.contains("bash"));
    }
}
