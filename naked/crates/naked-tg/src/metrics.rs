//! Lightweight in-process counters for media-routing decisions and other
//! interesting events. Exposed via `/metrics` operator command and also
//! emitted as `tracing::info!` events so existing log shippers pick them up.
//!
//! Why not `metrics`/`prometheus` crates? `naked-tg` ships as a single user
//! binary today and we want zero extra runtime overhead or registries to
//! configure. When we promote multimodal routing to production we can swap
//! these to `metrics::counter!` calls without changing call-sites.

use std::sync::atomic::{AtomicU64, Ordering};

/// Photos that took the **native** path (raw bytes attached to the chat
/// request). Increments once per processing batch (turn).
static NATIVE_ROUTE_CHOSEN: AtomicU64 = AtomicU64::new(0);

/// Batches where at least one image was over the per-provider cap and got
/// downgraded to the describer path. Useful for tuning `native_image_max_bytes`
/// and per-provider overrides.
static NATIVE_ROUTE_DOWNGRADED_OVERSIZE: AtomicU64 = AtomicU64::new(0);

/// Batches that fell back to the describer (vision-provider summary inlined as
/// text) — either because the active model isn't vision-capable, native routing
/// is disabled, or a downgrade triggered. Only counted when a describer is
/// configured (otherwise the photo would be path-only).
static DESCRIBER_FALLBACK: AtomicU64 = AtomicU64::new(0);

/// Outgoing TG messages that had to *wait* for rate-limit capacity to free
/// up (i.e. the 60-per-minute window was saturated). Does **not** count
/// every acquire — only the ones that actually delayed the caller.
/// Useful for spotting spam bursts or runaway retry loops in production.
static RATE_LIMIT_DELAYED: AtomicU64 = AtomicU64::new(0);

// ── F3 of PLAN_NEXT_SESSION (2026-05-10) ──────────────────
// Streaming-control-card button clicks. Two counters, one per
// callback action. A high `STREAM_BUTTON_CLICK_ABORT` rate (relative
// to active turns) signals UX friction; a high `_SENDNOW` rate
// means users frequently want to cut tool execution short.
static STREAM_BUTTON_CLICK_ABORT: AtomicU64 = AtomicU64::new(0);
static STREAM_BUTTON_CLICK_SENDNOW: AtomicU64 = AtomicU64::new(0);

// ── PLAN_MEDIA_UX_v1 M5 / BUG_REGISTRY B01 ────────────────────
// Audio-transcription observability. Two outcome buckets + six
// failure-reason buckets keep cardinality bounded.
static MEDIA_TRANSCRIPTION_OK: AtomicU64 = AtomicU64::new(0);
static MEDIA_TRANSCRIPTION_FAIL: AtomicU64 = AtomicU64::new(0);
static MEDIA_TRANSCRIPTION_FAIL_AUTH: AtomicU64 = AtomicU64::new(0);
static MEDIA_TRANSCRIPTION_FAIL_RATE: AtomicU64 = AtomicU64::new(0);
static MEDIA_TRANSCRIPTION_FAIL_PAYLOAD: AtomicU64 = AtomicU64::new(0);
static MEDIA_TRANSCRIPTION_FAIL_TIMEOUT: AtomicU64 = AtomicU64::new(0);
static MEDIA_TRANSCRIPTION_FAIL_NETWORK: AtomicU64 = AtomicU64::new(0);
static MEDIA_TRANSCRIPTION_FAIL_OTHER: AtomicU64 = AtomicU64::new(0);

/// Public bump for transcription outcome. `outcome` is "ok" or "fail";
/// `reason` is `Some(_)` only when `outcome == "fail"`. Counter labels
/// are cardinality-safe `&'static str` from [`crate::media::classify_media_error`].
pub fn record_transcription(outcome: &str, reason: Option<&str>) {
    match outcome {
        "ok" => {
            MEDIA_TRANSCRIPTION_OK.fetch_add(1, Ordering::Relaxed);
        }
        "fail" => {
            MEDIA_TRANSCRIPTION_FAIL.fetch_add(1, Ordering::Relaxed);
            match reason.unwrap_or("other") {
                "auth" => MEDIA_TRANSCRIPTION_FAIL_AUTH.fetch_add(1, Ordering::Relaxed),
                "rate_limit" => MEDIA_TRANSCRIPTION_FAIL_RATE.fetch_add(1, Ordering::Relaxed),
                "payload" => MEDIA_TRANSCRIPTION_FAIL_PAYLOAD.fetch_add(1, Ordering::Relaxed),
                "timeout" => MEDIA_TRANSCRIPTION_FAIL_TIMEOUT.fetch_add(1, Ordering::Relaxed),
                "network" => MEDIA_TRANSCRIPTION_FAIL_NETWORK.fetch_add(1, Ordering::Relaxed),
                _ => MEDIA_TRANSCRIPTION_FAIL_OTHER.fetch_add(1, Ordering::Relaxed),
            };
        }
        _ => {} // ignored — invariant: callers pass only "ok" or "fail"
    }
}

/// Bumped by [`crate::callbacks::handle_callback`] for every
/// `stream:abort` / `stream:sendnow` button press. The two-bucket
/// split (rather than one labelled counter) keeps the renderer
/// dead-simple — same convention as the rest of this module.
pub fn record_stream_button_click(action: &str) {
    match action {
        "abort" => {
            STREAM_BUTTON_CLICK_ABORT.fetch_add(1, Ordering::Relaxed);
        }
        "sendnow" => {
            STREAM_BUTTON_CLICK_SENDNOW.fetch_add(1, Ordering::Relaxed);
        }
        _ => {}
    }
}

/// Public bump for the rate-limiter. Kept in this module so all counters
/// stay in one place and the Prometheus renderer can see it directly.
#[allow(dead_code)] // used by tests + may be re-wired to adaptive limiter
pub fn record_rate_limit_delay() {
    RATE_LIMIT_DELAYED.fetch_add(1, Ordering::Relaxed);
}

/// Record a media-routing decision for one user turn (one `process_media_items`
/// call). Increments at most three counters depending on the chosen path.
pub fn record_media_routing(native: bool, has_oversize_hint: bool, describer_will_run: bool) {
    if native {
        NATIVE_ROUTE_CHOSEN.fetch_add(1, Ordering::Relaxed);
        if has_oversize_hint {
            NATIVE_ROUTE_DOWNGRADED_OVERSIZE.fetch_add(1, Ordering::Relaxed);
        }
    } else if describer_will_run {
        DESCRIBER_FALLBACK.fetch_add(1, Ordering::Relaxed);
    }
    tracing::info!(
        native,
        has_oversize_hint,
        describer_will_run,
        "metrics: media routing"
    );
}

/// Snapshot the counters for `/metrics` output.
pub fn snapshot() -> MediaRoutingSnapshot {
    MediaRoutingSnapshot {
        native_route_chosen: NATIVE_ROUTE_CHOSEN.load(Ordering::Relaxed),
        native_route_downgraded_oversize: NATIVE_ROUTE_DOWNGRADED_OVERSIZE.load(Ordering::Relaxed),
        describer_fallback: DESCRIBER_FALLBACK.load(Ordering::Relaxed),
        rate_limit_delayed: RATE_LIMIT_DELAYED.load(Ordering::Relaxed),
        stream_button_click_abort: STREAM_BUTTON_CLICK_ABORT.load(Ordering::Relaxed),
        stream_button_click_sendnow: STREAM_BUTTON_CLICK_SENDNOW.load(Ordering::Relaxed),
        transcription_ok: MEDIA_TRANSCRIPTION_OK.load(Ordering::Relaxed),
        transcription_fail: MEDIA_TRANSCRIPTION_FAIL.load(Ordering::Relaxed),
        transcription_fail_auth: MEDIA_TRANSCRIPTION_FAIL_AUTH.load(Ordering::Relaxed),
        transcription_fail_rate: MEDIA_TRANSCRIPTION_FAIL_RATE.load(Ordering::Relaxed),
        transcription_fail_payload: MEDIA_TRANSCRIPTION_FAIL_PAYLOAD.load(Ordering::Relaxed),
        transcription_fail_timeout: MEDIA_TRANSCRIPTION_FAIL_TIMEOUT.load(Ordering::Relaxed),
        transcription_fail_network: MEDIA_TRANSCRIPTION_FAIL_NETWORK.load(Ordering::Relaxed),
        transcription_fail_other: MEDIA_TRANSCRIPTION_FAIL_OTHER.load(Ordering::Relaxed),
    }
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct MediaRoutingSnapshot {
    pub native_route_chosen: u64,
    pub native_route_downgraded_oversize: u64,
    pub describer_fallback: u64,
    pub rate_limit_delayed: u64,
    pub stream_button_click_abort: u64,
    pub stream_button_click_sendnow: u64,
    // PLAN_MEDIA_UX_v1 M5 / BUG_REGISTRY B01
    pub transcription_ok: u64,
    pub transcription_fail: u64,
    pub transcription_fail_auth: u64,
    pub transcription_fail_rate: u64,
    pub transcription_fail_payload: u64,
    pub transcription_fail_timeout: u64,
    pub transcription_fail_network: u64,
    pub transcription_fail_other: u64,
}

impl MediaRoutingSnapshot {
    /// Human-readable summary suitable for `/metrics` Telegram replies.
    pub fn render_text(&self) -> String {
        format!(
            "Media routing counters (process lifetime):\n\
             • native route chosen: {}\n\
             • native downgraded (oversize hint): {}\n\
             • describer fallback: {}\n\
             • TG rate-limit delays: {}",
            self.native_route_chosen,
            self.native_route_downgraded_oversize,
            self.describer_fallback,
            self.rate_limit_delayed,
        )
    }

    /// Prometheus text-format (v0.0.4) rendering of all counters, including
    /// the `naked-core` sentinel-leak counter. Intended to be served over
    /// `/metrics` HTTP by [`serve_prometheus_if_enabled`].
    pub fn render_prometheus(&self) -> String {
        use std::sync::atomic::Ordering;
        let sentinel_leaks = naked_core::types::SENTINEL_LEAK_COUNT.load(Ordering::Relaxed);
        let empty_retries = naked_core::types::EMPTY_CONTENT_RETRY_COUNT.load(Ordering::Relaxed);
        let turn_ok = naked_core::types::TURN_COMPLETED_COUNT.load(Ordering::Relaxed);
        let turn_err = naked_core::types::TURN_ERROR_COUNT.load(Ordering::Relaxed);
        // F3: steer-pipeline + supervisor counters. Pin the recent
        // steer/abort UX work so a regression on either drain path
        // or the supervisor primitive shows up as a flat-zero series.
        let steer_delivered = naked_core::types::STEER_DELIVERED_COUNT.load(Ordering::Relaxed);
        let steer_soft_interrupted =
            naked_core::types::STEER_SOFT_INTERRUPTED_COUNT.load(Ordering::Relaxed);
        let steer_drained_on_abort =
            naked_core::types::STEER_DRAINED_ON_ABORT_COUNT.load(Ordering::Relaxed);
        // Note: `supervised` is in the lib crate (`naked_tg::`),
        // not the bin crate's `crate::*` namespace.
        let supervisor_restart =
            naked_tg::supervised::SUPERVISOR_PANIC_RESTART_COUNT.load(Ordering::Relaxed);
        // R5 of PLAN_RESILIENCE_v1: per-feature counters for the
        // PLAN_QUALITY_v1 modules. All start at 0 and stay there
        // unless the corresponding feature actually fires — a
        // flat-zero rate is a real signal the module isn't active.
        let snapshot_capture = naked_core::types::SNAPSHOT_CAPTURE_COUNT.load(Ordering::Relaxed);
        let lsp_emitted = naked_core::types::LSP_DIAGNOSTIC_EMITTED_COUNT.load(Ordering::Relaxed);
        let permission_match =
            naked_core::types::PERMISSION_RULE_MATCH_COUNT.load(Ordering::Relaxed);
        let hook_fire = naked_core::types::LIFECYCLE_HOOK_FIRE_COUNT.load(Ordering::Relaxed);
        let subagent_resolve =
            naked_core::types::SUBAGENT_ROLE_RESOLVE_COUNT.load(Ordering::Relaxed);
        let provider_perm_blacklist =
            naked_core::types::PROVIDER_PERMANENT_BLACKLIST_COUNT.load(Ordering::Relaxed);
        let crash_notified =
            naked_core::types::CRASH_RECOVERY_NOTIFIED_COUNT.load(Ordering::Relaxed);
        // Pollution sentinel from `naked-housekeep.timer`: 0 unless the
        // daily sweep found `research:*` lines in global MEMORY.md. Non-zero
        // means the research-leak fix regressed and operator should
        // investigate. Best-effort read; absent file → reported as 0.
        let memory_pollution = std::fs::read_to_string(
            std::env::var("HOME")
                .unwrap_or_else(|_| ".".into())
                .to_string()
                + "/.naked/.metrics/memory_pollution.prom",
        )
        .ok()
        .and_then(|s| {
            s.lines()
                .next()?
                .split_whitespace()
                .nth(1)?
                .parse::<u64>()
                .ok()
        })
        .unwrap_or(0);
        let model_health_body = std::fs::read_to_string(
            std::env::var("HOME")
                .unwrap_or_else(|_| ".".into())
                .to_string()
                + "/.naked/.metrics/model_health.prom",
        )
        .unwrap_or_default();
        format!(
            "# HELP naked_tg_native_route_chosen_total Photos sent via native path.\n\
             # TYPE naked_tg_native_route_chosen_total counter\n\
             naked_tg_native_route_chosen_total {native}\n\
             # HELP naked_tg_native_route_downgraded_oversize_total Native batches with oversize hint.\n\
             # TYPE naked_tg_native_route_downgraded_oversize_total counter\n\
             naked_tg_native_route_downgraded_oversize_total {oversize}\n\
             # HELP naked_tg_describer_fallback_total Batches that fell back to the describer.\n\
             # TYPE naked_tg_describer_fallback_total counter\n\
             naked_tg_describer_fallback_total {describer}\n\
             # HELP naked_core_sentinel_leak_stripped_total Image-ref sentinels stripped before reaching an LLM.\n\
             # TYPE naked_core_sentinel_leak_stripped_total counter\n\
             naked_core_sentinel_leak_stripped_total {leaks}\n\
             # HELP naked_tg_rate_limit_delayed_total TG sends that waited for the 60/min window to free up.\n\
             # TYPE naked_tg_rate_limit_delayed_total counter\n\
             naked_tg_rate_limit_delayed_total {rate_delayed}\n\
             # HELP naked_core_empty_content_retry_total Times the loop retried a turn after the provider closed the stream with no content (glm-5-turbo class).\n\
             # TYPE naked_core_empty_content_retry_total counter\n\
             naked_core_empty_content_retry_total {empty_retries}\n\
             # HELP naked_core_turn_completed_total Agent turns that finished successfully.\n\
             # TYPE naked_core_turn_completed_total counter\n\
             naked_core_turn_completed_total {turn_ok}\n\
             # HELP naked_core_turn_error_total Agent turns that ended with an error.\n\
             # TYPE naked_core_turn_error_total counter\n\
             naked_core_turn_error_total {turn_err}\n\
             # HELP naked_core_steer_delivered_total Steer messages successfully merged into history.\n\
             # TYPE naked_core_steer_delivered_total counter\n\
             naked_core_steer_delivered_total {steer_delivered}\n\
             # HELP naked_core_steer_soft_interrupted_total Times the LLM stream was soft-interrupted by a mid-stream steer (S2/S3 path).\n\
             # TYPE naked_core_steer_soft_interrupted_total counter\n\
             naked_core_steer_soft_interrupted_total {steer_soft_interrupted}\n\
             # HELP naked_core_steer_drained_on_abort_total Times the run-loop's drain-on-error rescued in-flight steers from a dying turn.\n\
             # TYPE naked_core_steer_drained_on_abort_total counter\n\
             naked_core_steer_drained_on_abort_total {steer_drained_on_abort}\n\
             # HELP naked_tg_supervisor_panic_restart_total Panic-triggered restarts inside spawn_supervised_with_opts.\n\
             # TYPE naked_tg_supervisor_panic_restart_total counter\n\
             naked_tg_supervisor_panic_restart_total {supervisor_restart}\n\
             # HELP naked_tg_stream_button_click_abort_total Clicks of the [⏹ Стоп] inline button on the streaming control card.\n\
             # TYPE naked_tg_stream_button_click_abort_total counter\n\
             naked_tg_stream_button_click_abort_total {btn_abort}\n\
             # HELP naked_tg_stream_button_click_sendnow_total Clicks of the [⏩ Send now] inline button on the streaming control card.\n\
             # TYPE naked_tg_stream_button_click_sendnow_total counter\n\
             naked_tg_stream_button_click_sendnow_total {btn_sendnow}\n\
             # HELP naked_core_snapshot_capture_total Side-git pre-turn workspace snapshots captured by `dispatch_turn`.\n\
             # TYPE naked_core_snapshot_capture_total counter\n\
             naked_core_snapshot_capture_total {snapshot_capture}\n\
             # HELP naked_core_lsp_diagnostic_emitted_total Non-empty LSP diagnostics blocks injected into history after edit tools.\n\
             # TYPE naked_core_lsp_diagnostic_emitted_total counter\n\
             naked_core_lsp_diagnostic_emitted_total {lsp_emitted}\n\
             # HELP naked_core_permission_rule_match_total Non-`Ask` decisions from the pattern-based permission ruleset.\n\
             # TYPE naked_core_permission_rule_match_total counter\n\
             naked_core_permission_rule_match_total {permission_match}\n\
             # HELP naked_core_lifecycle_hook_fire_total Lifecycle hooks that actually matched and executed.\n\
             # TYPE naked_core_lifecycle_hook_fire_total counter\n\
             naked_core_lifecycle_hook_fire_total {hook_fire}\n\
             # HELP naked_core_subagent_role_resolve_total Sub-agent calls whose `mode` parameter resolved to a canonical role.\n\
             # TYPE naked_core_subagent_role_resolve_total counter\n\
             naked_core_subagent_role_resolve_total {subagent_resolve}\n\
             # HELP naked_core_provider_permanent_blacklist_total Provider keys permanently removed from rotation due to auth/payment errors.\n\
             # TYPE naked_core_provider_permanent_blacklist_total counter\n\
             naked_core_provider_permanent_blacklist_total {provider_perm_blacklist}\n\
             # HELP naked_core_crash_recovery_notified_total Users notified about interrupted sessions on bot boot.\n\
             # TYPE naked_core_crash_recovery_notified_total counter\n\
             naked_core_crash_recovery_notified_total {crash_notified}\n\
             # HELP naked_memory_pollution_count research:* lines found in global MEMORY.md by the daily housekeep sweep (should stay 0).\n\
             # TYPE naked_memory_pollution_count gauge\n\
             naked_memory_pollution_count {memory_pollution}\n\
             # HELP naked_tg_media_transcription_total Audio transcription outcomes (label `outcome`: ok or fail).\n\
             # TYPE naked_tg_media_transcription_total counter\n\
             naked_tg_media_transcription_total{{outcome=\"ok\"}} {tr_ok}\n\
             naked_tg_media_transcription_total{{outcome=\"fail\"}} {tr_fail}\n\
             # HELP naked_tg_media_transcription_failure_total Audio transcription failures by classified reason (cardinality-safe).\n\
             # TYPE naked_tg_media_transcription_failure_total counter\n\
             naked_tg_media_transcription_failure_total{{reason=\"auth\"}} {tr_auth}\n\
             naked_tg_media_transcription_failure_total{{reason=\"rate_limit\"}} {tr_rate}\n\
             naked_tg_media_transcription_failure_total{{reason=\"payload\"}} {tr_payload}\n\
             naked_tg_media_transcription_failure_total{{reason=\"timeout\"}} {tr_timeout}\n\
             naked_tg_media_transcription_failure_total{{reason=\"network\"}} {tr_network}\n\
             naked_tg_media_transcription_failure_total{{reason=\"other\"}} {tr_other}\n\
             {model_health_body}",
            native = self.native_route_chosen,
            oversize = self.native_route_downgraded_oversize,
            describer = self.describer_fallback,
            leaks = sentinel_leaks,
            rate_delayed = self.rate_limit_delayed,
            empty_retries = empty_retries,
            turn_ok = turn_ok,
            turn_err = turn_err,
            steer_delivered = steer_delivered,
            steer_soft_interrupted = steer_soft_interrupted,
            steer_drained_on_abort = steer_drained_on_abort,
            supervisor_restart = supervisor_restart,
            btn_abort = self.stream_button_click_abort,
            btn_sendnow = self.stream_button_click_sendnow,
            snapshot_capture = snapshot_capture,
            lsp_emitted = lsp_emitted,
            permission_match = permission_match,
            hook_fire = hook_fire,
            subagent_resolve = subagent_resolve,
            provider_perm_blacklist = provider_perm_blacklist,
            crash_notified = crash_notified,
            memory_pollution = memory_pollution,
            tr_ok = self.transcription_ok,
            tr_fail = self.transcription_fail,
            tr_auth = self.transcription_fail_auth,
            tr_rate = self.transcription_fail_rate,
            tr_payload = self.transcription_fail_payload,
            tr_timeout = self.transcription_fail_timeout,
            tr_network = self.transcription_fail_network,
            tr_other = self.transcription_fail_other,
            model_health_body = model_health_body,
        )
    }
}

/// Build the HTTP response for a metrics request. Accepts only
/// `GET /metrics` — everything else returns `404 Not Found`. Unparseable
/// / empty requests also get a 404 (never a 500).
///
/// Split out for unit-testability: the listener loop is hard to exercise,
/// but the parse-and-respond logic is pure and can be hit directly.
pub fn build_metrics_response(request_bytes: &[u8]) -> String {
    // Parse the request line. It is always the first line of an HTTP/1.x
    // request and looks like `METHOD PATH HTTP/1.x`. Anything else → 404.
    let request_line = std::str::from_utf8(request_bytes)
        .ok()
        .and_then(|s| s.lines().next())
        .unwrap_or("");
    let mut parts = request_line.split_whitespace();
    let method = parts.next().unwrap_or("");
    let raw_path = parts.next().unwrap_or("");
    // Strip an optional query string — Prometheus sometimes adds one.
    let path = raw_path.split('?').next().unwrap_or("");

    if method == "GET" && path == "/metrics" {
        let body = snapshot().render_prometheus();
        return format!(
            "HTTP/1.1 200 OK\r\n\
             Content-Type: text/plain; version=0.0.4\r\n\
             Content-Length: {}\r\n\
             Connection: close\r\n\r\n{}",
            body.len(),
            body
        );
    }

    let body = "not found\n";
    format!(
        "HTTP/1.1 404 Not Found\r\n\
         Content-Type: text/plain; charset=utf-8\r\n\
         Content-Length: {}\r\n\
         Connection: close\r\n\r\n{}",
        body.len(),
        body
    )
}

/// If `NAKED_METRICS_ADDR` is set (e.g. `127.0.0.1:9898`), spawn a tiny
/// background task that serves the Prometheus text-format snapshot at `/metrics`.
///
/// Intentionally uses stdlib `TcpListener` + handwritten HTTP to avoid pulling
/// in a full web framework for one endpoint. Silent on misconfiguration —
/// `/metrics` is optional and we never want it to brick the bot.
pub fn serve_prometheus_if_enabled() {
    let addr = match std::env::var("NAKED_METRICS_ADDR") {
        Ok(a) if !a.trim().is_empty() => a,
        _ => return,
    };
    std::thread::Builder::new()
        .name("naked-tg-metrics".into())
        .spawn(move || {
            let listener = match std::net::TcpListener::bind(&addr) {
                Ok(l) => l,
                Err(e) => {
                    tracing::warn!("metrics listener failed to bind {addr}: {e}");
                    return;
                }
            };
            tracing::info!("Prometheus /metrics listening on {addr}");
            for stream in listener.incoming() {
                // REGISTRY-WAIVE: see BUG_REGISTRY B23 — verified intentional 2026-05-13
                let Ok(mut stream) = stream else { continue };
                use std::io::{Read, Write};
                let mut buf = [0u8; 1024];
                let n = stream.read(&mut buf).unwrap_or(0);
                let resp = build_metrics_response(&buf[..n]);
                let _ = stream.write_all(resp.as_bytes());
            }
        })
        .ok();
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn snapshot_is_monotonic_non_negative() {
        let s1 = snapshot();
        record_media_routing(true, false, false);
        let s2 = snapshot();
        assert!(s2.native_route_chosen > s1.native_route_chosen);
    }

    #[test]
    fn render_text_mentions_all_three_counters() {
        let s = MediaRoutingSnapshot {
            native_route_chosen: 1,
            native_route_downgraded_oversize: 2,
            describer_fallback: 3,
            rate_limit_delayed: 0,
            ..MediaRoutingSnapshot::default()
        };
        let t = s.render_text();
        assert!(t.contains("1"));
        assert!(t.contains("2"));
        assert!(t.contains("3"));
        assert!(t.contains("native"));
        assert!(t.contains("describer"));
    }

    #[test]
    fn fallback_counter_only_when_describer_configured() {
        let before = snapshot().describer_fallback;
        record_media_routing(false, false, false); // no describer → no bump
        assert_eq!(snapshot().describer_fallback, before);
        record_media_routing(false, false, true);
        assert_eq!(snapshot().describer_fallback, before + 1);
    }

    #[test]
    fn render_prometheus_has_help_type_and_counter_lines() {
        let s = MediaRoutingSnapshot {
            native_route_chosen: 7,
            native_route_downgraded_oversize: 2,
            describer_fallback: 1,
            rate_limit_delayed: 5,
            ..MediaRoutingSnapshot::default()
        };
        let p = s.render_prometheus();
        assert!(p.contains("# HELP naked_tg_native_route_chosen_total"));
        assert!(p.contains("# TYPE naked_tg_native_route_chosen_total counter"));
        assert!(p.contains("naked_tg_native_route_chosen_total 7"));
        assert!(p.contains("naked_tg_native_route_downgraded_oversize_total 2"));
        assert!(p.contains("naked_tg_describer_fallback_total 1"));
        assert!(p.contains("naked_core_sentinel_leak_stripped_total"));
    }

    #[test]
    fn render_prometheus_includes_steer_pipeline_counters_f3() {
        // F3: pin every metric introduced by the steer / abort /
        // button work so a removal in render_prometheus is loud.
        let p = snapshot().render_prometheus();
        for metric in [
            "naked_core_steer_delivered_total",
            "naked_core_steer_soft_interrupted_total",
            "naked_core_steer_drained_on_abort_total",
            "naked_tg_supervisor_panic_restart_total",
            "naked_tg_stream_button_click_abort_total",
            "naked_tg_stream_button_click_sendnow_total",
        ] {
            assert!(
                p.contains(metric),
                "render_prometheus must expose `{metric}`; full body:\n{p}",
            );
            // Each must declare its own HELP + TYPE pair (Prometheus
            // text format insists on it).
            assert!(
                p.contains(&format!("# HELP {metric}")),
                "missing HELP for {metric}",
            );
            assert!(
                p.contains(&format!("# TYPE {metric} counter")),
                "missing TYPE for {metric}",
            );
        }
    }

    #[test]
    fn stream_button_click_counter_splits_by_action() {
        let snap_before = snapshot();
        record_stream_button_click("abort");
        record_stream_button_click("abort");
        record_stream_button_click("sendnow");
        record_stream_button_click("unknown"); // ignored
        let snap_after = snapshot();
        assert_eq!(
            snap_after.stream_button_click_abort,
            snap_before.stream_button_click_abort + 2,
        );
        assert_eq!(
            snap_after.stream_button_click_sendnow,
            snap_before.stream_button_click_sendnow + 1,
        );
        // Unknown action must be a silent no-op (no panic, no
        // sneaking into either bucket).
    }

    #[test]
    fn render_prometheus_includes_new_resilience_counters() {
        // Verify the post-2026-04-22 sprint counters all surface in the
        // Prometheus body. We don't assert exact values (the underlying
        // atomics are process-global and may be bumped by other tests in
        // the same binary) — just that the metric NAMES are present so a
        // scraper can graph them.
        let p = snapshot().render_prometheus();
        for metric in [
            "naked_core_empty_content_retry_total",
            "naked_core_turn_completed_total",
            "naked_core_turn_error_total",
            "naked_memory_pollution_count",
        ] {
            assert!(
                p.contains(metric),
                "render_prometheus must expose `{metric}`; full body:\n{p}",
            );
        }
        // Spot-check the gauge HELP type so Prometheus parses it correctly.
        assert!(p.contains("# TYPE naked_memory_pollution_count gauge"));
    }

    #[test]
    fn empty_content_retry_counter_is_globally_observable() {
        use std::sync::atomic::Ordering;
        let before = naked_core::types::EMPTY_CONTENT_RETRY_COUNT.load(Ordering::Relaxed);
        naked_core::types::EMPTY_CONTENT_RETRY_COUNT.fetch_add(3, Ordering::Relaxed);
        let p = snapshot().render_prometheus();
        let after = naked_core::types::EMPTY_CONTENT_RETRY_COUNT.load(Ordering::Relaxed);
        assert_eq!(after, before + 3);
        // The rendered value must reflect the bumped counter.
        let line = p
            .lines()
            .find(|l| l.starts_with("naked_core_empty_content_retry_total "))
            .expect("counter line must be present");
        let val: u64 = line
            .split_whitespace()
            .nth(1)
            .and_then(|s| s.parse().ok())
            .expect("counter value must parse");
        assert!(
            val >= before + 3,
            "rendered value {val} must be >= {} (before+3)",
            before + 3
        );
    }

    // `serve_prometheus_without_env` test intentionally omitted: the
    // function reads `NAKED_METRICS_ADDR` via `std::env::var`, which has
    // well-defined "unset → return early" behavior. Verifying that would
    // require mutating process env (forbidden by workspace `unsafe-code`
    // lint) and is covered in practice by the bot start-up smoke test.

    /// Bind an ephemeral TCP listener, hand-roll a minimal HTTP client to
    /// hit it, and verify the response is a well-formed Prometheus
    /// text-format payload. This exercises the exact body-writer code
    /// used in `serve_prometheus_if_enabled` without reading env vars.
    #[test]
    fn http_listener_serves_prometheus_body() {
        use std::io::{Read, Write};
        use std::net::{TcpListener, TcpStream};

        let listener = TcpListener::bind("127.0.0.1:0").expect("bind");
        let addr = listener.local_addr().unwrap();

        let handle = std::thread::spawn(move || {
            let (mut stream, _) = listener.accept().expect("accept");
            let mut buf = [0u8; 1024];
            let n = stream.read(&mut buf).unwrap_or(0);
            let resp = build_metrics_response(&buf[..n]);
            stream.write_all(resp.as_bytes()).expect("write");
        });

        let mut client = TcpStream::connect(addr).expect("connect");
        client
            .write_all(b"GET /metrics HTTP/1.1\r\nHost: x\r\n\r\n")
            .unwrap();
        let mut out = String::new();
        client.read_to_string(&mut out).unwrap();
        handle.join().unwrap();

        assert!(out.starts_with("HTTP/1.1 200 OK"), "response: {out}");
        assert!(out.contains("text/plain; version=0.0.4"), "ct: {out}");
        assert!(out.contains("naked_tg_native_route_chosen_total"));
        assert!(out.contains("naked_core_sentinel_leak_stripped_total"));
    }

    #[test]
    fn metrics_response_rejects_non_get() {
        let resp = build_metrics_response(b"POST /metrics HTTP/1.1\r\n\r\n");
        assert!(resp.starts_with("HTTP/1.1 404 Not Found"), "resp: {resp}");
        assert!(!resp.contains("naked_tg_native_route_chosen_total"));
    }

    #[test]
    fn metrics_response_rejects_other_paths() {
        for path in ["/", "/admin", "/metrics/", "/healthz"] {
            let req = format!("GET {path} HTTP/1.1\r\n\r\n");
            let resp = build_metrics_response(req.as_bytes());
            assert!(
                resp.starts_with("HTTP/1.1 404 Not Found"),
                "expected 404 for {path}, got: {resp}"
            );
        }
    }

    #[test]
    fn metrics_response_accepts_query_string() {
        let resp = build_metrics_response(b"GET /metrics?debug=1 HTTP/1.1\r\n\r\n");
        assert!(resp.starts_with("HTTP/1.1 200 OK"), "resp: {resp}");
        assert!(resp.contains("naked_tg_native_route_chosen_total"));
    }

    #[test]
    fn rate_limit_counter_bumps_and_renders() {
        let before = snapshot().rate_limit_delayed;
        record_rate_limit_delay();
        record_rate_limit_delay();
        let after = snapshot().rate_limit_delayed;
        assert_eq!(after, before + 2);

        let prom = snapshot().render_prometheus();
        assert!(
            prom.contains("naked_tg_rate_limit_delayed_total"),
            "prom must include rate_limit counter: {prom}"
        );
        let text = snapshot().render_text();
        assert!(
            text.contains("rate-limit delays"),
            "text must mention rate-limit delays: {text}"
        );
    }

    #[test]
    fn metrics_response_handles_empty_or_garbage_request() {
        assert!(build_metrics_response(b"").starts_with("HTTP/1.1 404"));
        assert!(build_metrics_response(b"\x00\x01\x02\x03").starts_with("HTTP/1.1 404"));
        assert!(build_metrics_response(b"garbage").starts_with("HTTP/1.1 404"));
    }

    /// The Prometheus body must inline any pre-rendered
    /// `~/.naked/.metrics/model_health.prom` produced by the daily
    /// housekeep. We can't mutate `$HOME` safely in the test process
    /// (workspace `unsafe-code` lint + parallel-test pollution), so we
    /// just assert the render function does not panic and that, if the
    /// operator's live file happens to exist, its text is spliced in.
    #[test]
    fn render_prometheus_inlines_model_health_file_when_present() {
        let snap = MediaRoutingSnapshot::default();
        let body = snap.render_prometheus();
        // Required resilience counters always present.
        assert!(body.contains("naked_core_empty_content_retry_total"));
        // If the operator's file exists on this host, verify its first
        // token made it into the body (best-effort, no mutation).
        let home = std::env::var("HOME").unwrap_or_else(|_| ".".into());
        let path = format!("{home}/.naked/.metrics/model_health.prom");
        if let Ok(contents) = std::fs::read_to_string(&path)
            && !contents.is_empty()
        {
            // Splice happens verbatim at the end of the body.
            assert!(
                body.contains(contents.trim_end()),
                "model_health.prom contents must be inlined; body tail:\n{}",
                &body[body.len().saturating_sub(400)..]
            );
        }
    }
}
