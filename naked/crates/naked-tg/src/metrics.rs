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

/// B45 wire-up (2026-05-13): credential redactions applied before sending
/// outgoing HTML to Telegram. Increments each time `scan_and_redact` modified
/// the payload (no-op calls don't count). Visible leak budget should stay 0;
/// any non-zero value is a red flag that a tool / research output leaked an
/// `api_key=` / `Bearer X` / `Authorization:` token into the assistant turn.
static REDACTION_APPLIED: AtomicU64 = AtomicU64::new(0);

/// B61/TB2: same (chat,thread) dispatches that observed per-key lock contention.
static CONCURRENT_SAME_KEY_TURNS: AtomicU64 = AtomicU64::new(0);

/// B60 / FOLLOWUP F4: text bursts merged into one user turn.
static TEXT_COALESCED: AtomicU64 = AtomicU64::new(0);

/// B62 / Q4: SessionBusy follow-ups that could not be steered and got a soft Busy ack.
static SESSION_BUSY_ACK: AtomicU64 = AtomicU64::new(0);

/// B05: current boot-time config health for multimodal fallback.
/// State gauge: 1 when the default model is not vision-capable AND no
/// `tg_media.vision` describer fallback is configured, otherwise 0.
static CONFIG_DESCRIBER_MISSING: AtomicU64 = AtomicU64::new(0);

pub fn set_config_describer_missing(missing: bool) {
    CONFIG_DESCRIBER_MISSING.store(u64::from(missing), Ordering::Relaxed);
}

pub fn record_concurrent_same_key_turn_wait() {
    CONCURRENT_SAME_KEY_TURNS.fetch_add(1, Ordering::Relaxed);
}

pub fn record_text_coalesced() {
    TEXT_COALESCED.fetch_add(1, Ordering::Relaxed);
}

pub fn record_session_busy_ack() {
    SESSION_BUSY_ACK.fetch_add(1, Ordering::Relaxed);
}

pub fn record_redaction_applied() {
    REDACTION_APPLIED.fetch_add(1, Ordering::Relaxed);
}

// ── F3 of PLAN_NEXT_SESSION (2026-05-10) ──────────────────
// Streaming-control-card button clicks. Two counters, one per
// callback action. A high `STREAM_BUTTON_CLICK_ABORT` rate (relative
// to active turns) signals UX friction; a high `_SENDNOW` rate
// means users frequently want to cut tool execution short.
static STREAM_BUTTON_CLICK_ABORT: AtomicU64 = AtomicU64::new(0);
static STREAM_BUTTON_CLICK_SENDNOW: AtomicU64 = AtomicU64::new(0);
static RUN_REGISTRY_REGISTER: AtomicU64 = AtomicU64::new(0);
static RUN_REGISTRY_REMOVE: AtomicU64 = AtomicU64::new(0);
static RUN_REGISTRY_CAP_REJECT: AtomicU64 = AtomicU64::new(0);
static RUN_REGISTRY_CALLBACK_EXPIRED: AtomicU64 = AtomicU64::new(0);
static RUN_REGISTRY_CALLBACK_RESOLVED: AtomicU64 = AtomicU64::new(0);
static RUN_REGISTRY_CALLBACK_DENIED: AtomicU64 = AtomicU64::new(0);

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

// ── T11 (PLAN_RESEARCH_AGENT_FLOW_v1) ───────────────────────
// Research-cancel propagation latency. Tracks the time from
// `callbacks::handle_callback("r:stop:<id>")` receipt (or equivalent
// /abort command) to the moment the agent's tool loop actually exits.
//
// The 2026-05-16 incident (B56 PARALLEL-CANCEL-MECHANISMS) had
// effective propagation time of **infinity** — user pressed Stop
// ×3 over 32s and tools kept running for ~3 min after. With the
// unified `agent.abort(session_id)` path, propagation SHOULD be
// under 3s (next iteration boundary).
//
// Three buckets keep cardinality low while making outliers visible:
//   - under_3s   : nominal
//   - 3s_to_30s  : slow but recovers (likely tool mid-network call)
//   - over_30s   : alert-worthy (cancel got dropped somewhere)
//
// Sum + count expose the running average too (~1.2-1.5s expected).
static RESEARCH_CANCEL_PROPAGATION_UNDER_3S: AtomicU64 = AtomicU64::new(0);
static RESEARCH_CANCEL_PROPAGATION_3S_TO_30S: AtomicU64 = AtomicU64::new(0);
static RESEARCH_CANCEL_PROPAGATION_OVER_30S: AtomicU64 = AtomicU64::new(0);
static RESEARCH_CANCEL_PROPAGATION_SUM_MS: AtomicU64 = AtomicU64::new(0);
static RESEARCH_CANCEL_PROPAGATION_COUNT: AtomicU64 = AtomicU64::new(0);

/// Record one cancel-propagation observation. `elapsed_ms` is the wall
/// time between callback dispatch (or `/abort` command receipt) and the
/// tool loop's exit signal.
///
/// Anti-foot: callers MUST measure with `std::time::Instant::elapsed`
/// captured BEFORE invoking `agent.abort`, not after — we want the
/// operator-visible latency, not internal token-cancel overhead.
pub fn record_research_cancel_propagation(elapsed_ms: u64) {
    RESEARCH_CANCEL_PROPAGATION_COUNT.fetch_add(1, Ordering::Relaxed);
    RESEARCH_CANCEL_PROPAGATION_SUM_MS.fetch_add(elapsed_ms, Ordering::Relaxed);
    if elapsed_ms < 3_000 {
        RESEARCH_CANCEL_PROPAGATION_UNDER_3S.fetch_add(1, Ordering::Relaxed);
    } else if elapsed_ms < 30_000 {
        RESEARCH_CANCEL_PROPAGATION_3S_TO_30S.fetch_add(1, Ordering::Relaxed);
    } else {
        RESEARCH_CANCEL_PROPAGATION_OVER_30S.fetch_add(1, Ordering::Relaxed);
        tracing::warn!(
            elapsed_ms,
            "research cancel propagation > 30s — likely a B56-class regression \
             (a tool loop missed its CancellationToken check). See \
             postmortems/2026-05-16-research-split-brain.md."
        );
    }
    tracing::info!(elapsed_ms, "metrics: research cancel propagated");
}
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

pub fn record_run_registry_register() {
    RUN_REGISTRY_REGISTER.fetch_add(1, Ordering::Relaxed);
}

pub fn record_run_registry_remove() {
    RUN_REGISTRY_REMOVE.fetch_add(1, Ordering::Relaxed);
}

pub fn record_run_registry_cap_reject() {
    RUN_REGISTRY_CAP_REJECT.fetch_add(1, Ordering::Relaxed);
}

pub fn record_run_registry_callback_expired() {
    RUN_REGISTRY_CALLBACK_EXPIRED.fetch_add(1, Ordering::Relaxed);
}

pub fn record_run_registry_callback_resolved() {
    RUN_REGISTRY_CALLBACK_RESOLVED.fetch_add(1, Ordering::Relaxed);
}

pub fn record_run_registry_callback_denied() {
    RUN_REGISTRY_CALLBACK_DENIED.fetch_add(1, Ordering::Relaxed);
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
        run_registry_register: RUN_REGISTRY_REGISTER.load(Ordering::Relaxed),
        run_registry_remove: RUN_REGISTRY_REMOVE.load(Ordering::Relaxed),
        run_registry_cap_reject: RUN_REGISTRY_CAP_REJECT.load(Ordering::Relaxed),
        run_registry_callback_expired: RUN_REGISTRY_CALLBACK_EXPIRED.load(Ordering::Relaxed),
        run_registry_callback_resolved: RUN_REGISTRY_CALLBACK_RESOLVED.load(Ordering::Relaxed),
        run_registry_callback_denied: RUN_REGISTRY_CALLBACK_DENIED.load(Ordering::Relaxed),
        transcription_ok: MEDIA_TRANSCRIPTION_OK.load(Ordering::Relaxed),
        transcription_fail: MEDIA_TRANSCRIPTION_FAIL.load(Ordering::Relaxed),
        transcription_fail_auth: MEDIA_TRANSCRIPTION_FAIL_AUTH.load(Ordering::Relaxed),
        transcription_fail_rate: MEDIA_TRANSCRIPTION_FAIL_RATE.load(Ordering::Relaxed),
        transcription_fail_payload: MEDIA_TRANSCRIPTION_FAIL_PAYLOAD.load(Ordering::Relaxed),
        transcription_fail_timeout: MEDIA_TRANSCRIPTION_FAIL_TIMEOUT.load(Ordering::Relaxed),
        transcription_fail_network: MEDIA_TRANSCRIPTION_FAIL_NETWORK.load(Ordering::Relaxed),
        transcription_fail_other: MEDIA_TRANSCRIPTION_FAIL_OTHER.load(Ordering::Relaxed),
        redaction_applied: REDACTION_APPLIED.load(Ordering::Relaxed),
        concurrent_same_key_turns: CONCURRENT_SAME_KEY_TURNS.load(Ordering::Relaxed),
        text_coalesced: TEXT_COALESCED.load(Ordering::Relaxed),
        session_busy_ack: SESSION_BUSY_ACK.load(Ordering::Relaxed),
        config_describer_missing: CONFIG_DESCRIBER_MISSING.load(Ordering::Relaxed),
        research_cancel_propagation_under_3s: RESEARCH_CANCEL_PROPAGATION_UNDER_3S
            .load(Ordering::Relaxed),
        research_cancel_propagation_3s_to_30s: RESEARCH_CANCEL_PROPAGATION_3S_TO_30S
            .load(Ordering::Relaxed),
        research_cancel_propagation_over_30s: RESEARCH_CANCEL_PROPAGATION_OVER_30S
            .load(Ordering::Relaxed),
        research_cancel_propagation_sum_ms: RESEARCH_CANCEL_PROPAGATION_SUM_MS
            .load(Ordering::Relaxed),
        research_cancel_propagation_count: RESEARCH_CANCEL_PROPAGATION_COUNT
            .load(Ordering::Relaxed),
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
    pub run_registry_register: u64,
    pub run_registry_remove: u64,
    pub run_registry_cap_reject: u64,
    pub run_registry_callback_expired: u64,
    pub run_registry_callback_resolved: u64,
    pub run_registry_callback_denied: u64,
    // PLAN_MEDIA_UX_v1 M5 / BUG_REGISTRY B01
    pub transcription_ok: u64,
    pub transcription_fail: u64,
    pub transcription_fail_auth: u64,
    pub transcription_fail_rate: u64,
    pub transcription_fail_payload: u64,
    pub transcription_fail_timeout: u64,
    pub transcription_fail_network: u64,
    pub transcription_fail_other: u64,
    /// B45 wire-up: outgoing-msg credential redactions count.
    pub redaction_applied: u64,
    /// B61/TB2: same-key dispatches that had to wait on the per-chat lock.
    pub concurrent_same_key_turns: u64,
    /// B60/F4: text bursts merged into one turn.
    pub text_coalesced: u64,
    /// B62/Q4: SessionBusy follow-ups that got a soft Busy ack.
    pub session_busy_ack: u64,
    /// B05: 0/1 state gauge for missing multimodal describer fallback.
    pub config_describer_missing: u64,
    // T11 PLAN_RESEARCH_AGENT_FLOW_v1: cancel propagation latency.
    pub research_cancel_propagation_under_3s: u64,
    pub research_cancel_propagation_3s_to_30s: u64,
    pub research_cancel_propagation_over_30s: u64,
    pub research_cancel_propagation_sum_ms: u64,
    pub research_cancel_propagation_count: u64,
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
struct PrometheusInputs {
    media: MediaRoutingSnapshot,
    sentinel_leaks: u64,
    empty_retries: u64,
    turn_ok: u64,
    turn_err: u64,
    steer_delivered: u64,
    steer_soft_interrupted: u64,
    steer_drained_on_abort: u64,
    supervisor_restart: u64,
    snapshot_capture: u64,
    lsp_emitted: u64,
    permission_match: u64,
    hook_fire: u64,
    subagent_resolve: u64,
    provider_perm_blacklist: u64,
    crash_notified: u64,
    vision_mismatch: u64,
    provider_connect_timeout: u64,
    provider_inter_chunk_timeout: u64,
    provider_invalid_request: u64,
    scheduler_dispatch_skipped: u64,
    cfg_ext_write: u64,
    ip_hallucin: u64,
    research_store_corrupt_rows: u64,
    research_store_files_healed: u64,
    research_store_heal_failed: u64,
    research_store_file_lock_wait: u64,
    concurrent_same_key_turns: u64,
    text_coalesced: u64,
    session_busy_ack: u64,
    memory_pollution: u64,
    model_health_body: String,
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
        let inputs = PrometheusInputs {
            media: *self,
            sentinel_leaks: naked_core::types::SENTINEL_LEAK_COUNT.load(Ordering::Relaxed),
            empty_retries: naked_core::types::EMPTY_CONTENT_RETRY_COUNT.load(Ordering::Relaxed),
            turn_ok: naked_core::types::TURN_COMPLETED_COUNT.load(Ordering::Relaxed),
            turn_err: naked_core::types::TURN_ERROR_COUNT.load(Ordering::Relaxed),
            steer_delivered: naked_core::types::STEER_DELIVERED_COUNT.load(Ordering::Relaxed),
            steer_soft_interrupted: naked_core::types::STEER_SOFT_INTERRUPTED_COUNT
                .load(Ordering::Relaxed),
            steer_drained_on_abort: naked_core::types::STEER_DRAINED_ON_ABORT_COUNT
                .load(Ordering::Relaxed),
            supervisor_restart: naked_tg::supervised::SUPERVISOR_PANIC_RESTART_COUNT
                .load(Ordering::Relaxed),
            snapshot_capture: naked_core::types::SNAPSHOT_CAPTURE_COUNT.load(Ordering::Relaxed),
            lsp_emitted: naked_core::types::LSP_DIAGNOSTIC_EMITTED_COUNT.load(Ordering::Relaxed),
            permission_match: naked_core::types::PERMISSION_RULE_MATCH_COUNT
                .load(Ordering::Relaxed),
            hook_fire: naked_core::types::LIFECYCLE_HOOK_FIRE_COUNT.load(Ordering::Relaxed),
            subagent_resolve: naked_core::types::SUBAGENT_ROLE_RESOLVE_COUNT
                .load(Ordering::Relaxed),
            provider_perm_blacklist: naked_core::types::PROVIDER_PERMANENT_BLACKLIST_COUNT
                .load(Ordering::Relaxed),
            crash_notified: naked_core::types::CRASH_RECOVERY_NOTIFIED_COUNT
                .load(Ordering::Relaxed),
            vision_mismatch: naked_core::types::PROVIDER_VISION_CAP_MISMATCH_COUNT
                .load(Ordering::Relaxed),
            provider_connect_timeout: naked_core::types::PROVIDER_CONNECT_TIMEOUT_COUNT
                .load(Ordering::Relaxed),
            provider_inter_chunk_timeout: naked_core::types::PROVIDER_INTER_CHUNK_TIMEOUT_COUNT
                .load(Ordering::Relaxed),
            provider_invalid_request: naked_core::types::PROVIDER_INVALID_REQUEST_COUNT
                .load(Ordering::Relaxed),
            scheduler_dispatch_skipped: naked_core::types::SCHEDULER_DISPATCH_SKIPPED_COUNT
                .load(Ordering::Relaxed),
            cfg_ext_write: naked_core::types::CONFIG_EXTERNAL_WRITE_COUNT.load(Ordering::Relaxed),
            ip_hallucin: naked_core::types::IP_TOKEN_HALLUCINATION_COUNT.load(Ordering::Relaxed),
            research_store_corrupt_rows:
                naked_core::types::RESEARCH_STORE_CORRUPT_ROWS_DETECTED_COUNT
                    .load(Ordering::Relaxed),
            research_store_files_healed: naked_core::types::RESEARCH_STORE_FILES_HEALED_COUNT
                .load(Ordering::Relaxed),
            research_store_heal_failed: naked_core::types::RESEARCH_STORE_HEAL_FAILED_COUNT
                .load(Ordering::Relaxed),
            research_store_file_lock_wait: naked_core::types::RESEARCH_STORE_FILE_LOCK_WAIT_COUNT
                .load(Ordering::Relaxed),
            concurrent_same_key_turns: self.concurrent_same_key_turns,
            text_coalesced: self.text_coalesced,
            session_busy_ack: self.session_busy_ack,
            memory_pollution,
            model_health_body,
        };
        render_prometheus_from(&inputs)
    }
}

fn render_prometheus_from(inputs: &PrometheusInputs) -> String {
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
             # HELP naked_tg_run_registry_register_total RunRegistry run registrations.\n\
             # TYPE naked_tg_run_registry_register_total counter\n\
             naked_tg_run_registry_register_total {run_reg}\n\
             # HELP naked_tg_run_registry_remove_total RunRegistry run removals.\n\
             # TYPE naked_tg_run_registry_remove_total counter\n\
             naked_tg_run_registry_remove_total {run_remove}\n\
             # HELP naked_tg_run_registry_cap_reject_total RunRegistry thread cap rejections.\n\
             # TYPE naked_tg_run_registry_cap_reject_total counter\n\
             naked_tg_run_registry_cap_reject_total {run_cap_reject}\n\
             # HELP naked_tg_run_registry_callback_expired_total Expired legacy stream callbacks.\n\
             # TYPE naked_tg_run_registry_callback_expired_total counter\n\
             naked_tg_run_registry_callback_expired_total {run_cb_expired}\n\
             # HELP naked_tg_run_registry_callback_resolved_total Stream callbacks resolved to a live run.\n\
             # TYPE naked_tg_run_registry_callback_resolved_total counter\n\
             naked_tg_run_registry_callback_resolved_total {run_cb_resolved}\n\
             # HELP naked_tg_run_registry_callback_denied_total Stream callbacks denied by allowed_chat_ids.\n\
             # TYPE naked_tg_run_registry_callback_denied_total counter\n\
             naked_tg_run_registry_callback_denied_total {run_cb_denied}\n\
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
             # HELP naked_core_provider_vision_capability_mismatch_total (B06) caps.supports_vision=true but API rejects image_url content shape.\n\
             # TYPE naked_core_provider_vision_capability_mismatch_total counter\n\
             naked_core_provider_vision_capability_mismatch_total {vision_mismatch}\n\
             # HELP naked_core_provider_timeout_total (B68) Provider timeout events by timeout kind.\n\
             # TYPE naked_core_provider_timeout_total counter\n\
             naked_core_provider_timeout_total{{kind=\"connect\"}} {provider_connect_timeout}\n\
             naked_core_provider_timeout_total{{kind=\"inter_chunk\"}} {provider_inter_chunk_timeout}\n\
             # HELP naked_core_provider_invalid_request_total (B76) Provider HTTP 400 invalid_request_error responses (likely config/request-shape bugs).\n\
             # TYPE naked_core_provider_invalid_request_total counter\n\
             naked_core_provider_invalid_request_total {provider_invalid_request}\n\
             # HELP naked_core_scheduler_dispatch_skipped_total (B75) Scheduler dispatches skipped because all provider keys/providers were blacklisted.\n\
             # TYPE naked_core_scheduler_dispatch_skipped_total counter\n\
             naked_core_scheduler_dispatch_skipped_total {scheduler_dispatch_skipped}\n\
             # HELP naked_core_config_external_write_total (B42) state/naked.json modified by external process detected by D-CONFIG-MTIME-WATCH.\n\
             # TYPE naked_core_config_external_write_total counter\n\
             naked_core_config_external_write_total {cfg_ext_write}\n\
             # HELP naked_core_ip_token_hallucination_total (B37) Outgoing assistant messages mentioning noVNC with IP tokens not in the boot-cached allow-list.\n\
             # TYPE naked_core_ip_token_hallucination_total counter\n\
             naked_core_ip_token_hallucination_total {ip_hallucin}\n\
             # HELP naked_core_research_store_corrupt_rows_detected_total (B59) Invalid findings.jsonl rows detected by the research store healing path.\n\
             # TYPE naked_core_research_store_corrupt_rows_detected_total counter\n\
             naked_core_research_store_corrupt_rows_detected_total {research_store_corrupt_rows}\n\
             # HELP naked_core_research_store_files_healed_total (B59) findings.jsonl files successfully healed after corrupt-row detection.\n\
             # TYPE naked_core_research_store_files_healed_total counter\n\
             naked_core_research_store_files_healed_total {research_store_files_healed}\n\
             # HELP naked_core_research_store_heal_failed_total (B59) Research-store healing attempts that failed before replacing the live file.\n\
             # TYPE naked_core_research_store_heal_failed_total counter\n\
             naked_core_research_store_heal_failed_total {research_store_heal_failed}\n\
             # HELP naked_core_research_store_file_lock_wait_total (B59) findings.jsonl write-lock acquisitions that had to wait on contention.\n\
             # TYPE naked_core_research_store_file_lock_wait_total counter\n\
             naked_core_research_store_file_lock_wait_total {research_store_file_lock_wait}\n\
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
             # HELP naked_tg_redaction_applied_total (B45) Outgoing TG messages where scan_and_redact modified payload.\n\
             # TYPE naked_tg_redaction_applied_total counter\n\
             naked_tg_redaction_applied_total {redaction_applied}\n\
             # HELP naked_tg_concurrent_same_key_turns_total (B61) Same (chat,thread) dispatches that waited on the per-key short-section lock.\n\
             # TYPE naked_tg_concurrent_same_key_turns_total counter\n\
             naked_tg_concurrent_same_key_turns_total {concurrent_same_key_turns}\n\
             # HELP naked_tg_text_coalesced_total (B60) Text bursts merged (>=2 client-split messages joined into one turn).\n\
             # TYPE naked_tg_text_coalesced_total counter\n\
             naked_tg_text_coalesced_total {text_coalesced}\n\
             # HELP naked_tg_session_busy_ack_total (B62) SessionBusy follow-ups that got a soft busy ack (no steer sender / full channel); measures the F1/B3 drop window.\n\
             # TYPE naked_tg_session_busy_ack_total counter\n\
             naked_tg_session_busy_ack_total {session_busy_ack}\n\
             # HELP naked_tg_config_describer_missing (B05) Set to 1 when the default model is not vision-capable AND no tg_media.vision describer fallback is configured.\n\
             # TYPE naked_tg_config_describer_missing gauge\n\
             naked_tg_config_describer_missing {config_describer_missing}\n\
             # HELP naked_tg_research_cancel_propagation_total (T11/B56) Cancel propagation latency buckets (ms).\n\
             # TYPE naked_tg_research_cancel_propagation_total counter\n\
             naked_tg_research_cancel_propagation_total{{bucket=\"under_3s\"}} {cancel_u3s}\n\
             naked_tg_research_cancel_propagation_total{{bucket=\"3s_to_30s\"}} {cancel_3s_30s}\n\
             naked_tg_research_cancel_propagation_total{{bucket=\"over_30s\"}} {cancel_o30s}\n\
             # HELP naked_tg_research_cancel_propagation_sum_ms (T11) Sum of cancel propagation latencies in ms.\n\
             # TYPE naked_tg_research_cancel_propagation_sum_ms counter\n\
             naked_tg_research_cancel_propagation_sum_ms {cancel_sum_ms}\n\
             # HELP naked_tg_research_cancel_propagation_count (T11) Total cancel observations recorded.\n\
             # TYPE naked_tg_research_cancel_propagation_count counter\n\
             naked_tg_research_cancel_propagation_count {cancel_count}\n\
             {model_health_body}",
        native = inputs.media.native_route_chosen,
        oversize = inputs.media.native_route_downgraded_oversize,
        describer = inputs.media.describer_fallback,
        leaks = inputs.sentinel_leaks,
        rate_delayed = inputs.media.rate_limit_delayed,
        empty_retries = inputs.empty_retries,
        turn_ok = inputs.turn_ok,
        turn_err = inputs.turn_err,
        steer_delivered = inputs.steer_delivered,
        steer_soft_interrupted = inputs.steer_soft_interrupted,
        steer_drained_on_abort = inputs.steer_drained_on_abort,
        supervisor_restart = inputs.supervisor_restart,
        btn_abort = inputs.media.stream_button_click_abort,
        btn_sendnow = inputs.media.stream_button_click_sendnow,
        run_reg = inputs.media.run_registry_register,
        run_remove = inputs.media.run_registry_remove,
        run_cap_reject = inputs.media.run_registry_cap_reject,
        run_cb_expired = inputs.media.run_registry_callback_expired,
        run_cb_resolved = inputs.media.run_registry_callback_resolved,
        run_cb_denied = inputs.media.run_registry_callback_denied,
        snapshot_capture = inputs.snapshot_capture,
        lsp_emitted = inputs.lsp_emitted,
        permission_match = inputs.permission_match,
        hook_fire = inputs.hook_fire,
        subagent_resolve = inputs.subagent_resolve,
        provider_perm_blacklist = inputs.provider_perm_blacklist,
        crash_notified = inputs.crash_notified,
        vision_mismatch = inputs.vision_mismatch,
        provider_connect_timeout = inputs.provider_connect_timeout,
        provider_inter_chunk_timeout = inputs.provider_inter_chunk_timeout,
        provider_invalid_request = inputs.provider_invalid_request,
        scheduler_dispatch_skipped = inputs.scheduler_dispatch_skipped,
        cfg_ext_write = inputs.cfg_ext_write,
        ip_hallucin = inputs.ip_hallucin,
        research_store_corrupt_rows = inputs.research_store_corrupt_rows,
        research_store_files_healed = inputs.research_store_files_healed,
        research_store_heal_failed = inputs.research_store_heal_failed,
        research_store_file_lock_wait = inputs.research_store_file_lock_wait,
        memory_pollution = inputs.memory_pollution,
        tr_ok = inputs.media.transcription_ok,
        tr_fail = inputs.media.transcription_fail,
        tr_auth = inputs.media.transcription_fail_auth,
        tr_rate = inputs.media.transcription_fail_rate,
        tr_payload = inputs.media.transcription_fail_payload,
        tr_timeout = inputs.media.transcription_fail_timeout,
        tr_network = inputs.media.transcription_fail_network,
        tr_other = inputs.media.transcription_fail_other,
        redaction_applied = inputs.media.redaction_applied,
        concurrent_same_key_turns = inputs.concurrent_same_key_turns,
        text_coalesced = inputs.text_coalesced,
        session_busy_ack = inputs.session_busy_ack,
        config_describer_missing = inputs.media.config_describer_missing,
        cancel_u3s = inputs.media.research_cancel_propagation_under_3s,
        cancel_3s_30s = inputs.media.research_cancel_propagation_3s_to_30s,
        cancel_o30s = inputs.media.research_cancel_propagation_over_30s,
        cancel_sum_ms = inputs.media.research_cancel_propagation_sum_ms,
        cancel_count = inputs.media.research_cancel_propagation_count,
        model_health_body = inputs.model_health_body,
    )
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
    fn render_prometheus_from_golden() {
        let inputs = PrometheusInputs {
            media: MediaRoutingSnapshot {
                native_route_chosen: 1,
                native_route_downgraded_oversize: 2,
                describer_fallback: 3,
                rate_limit_delayed: 4,
                stream_button_click_abort: 5,
                stream_button_click_sendnow: 6,
                run_registry_register: 61,
                run_registry_remove: 62,
                run_registry_cap_reject: 63,
                run_registry_callback_expired: 64,
                run_registry_callback_resolved: 65,
                run_registry_callback_denied: 66,
                transcription_ok: 7,
                transcription_fail: 8,
                transcription_fail_auth: 9,
                transcription_fail_rate: 10,
                transcription_fail_payload: 11,
                transcription_fail_timeout: 12,
                transcription_fail_network: 13,
                transcription_fail_other: 14,
                redaction_applied: 15,
                concurrent_same_key_turns: 916,
                text_coalesced: 917,
                session_busy_ack: 918,
                config_describer_missing: 1,
                research_cancel_propagation_under_3s: 19,
                research_cancel_propagation_3s_to_30s: 20,
                research_cancel_propagation_over_30s: 21,
                research_cancel_propagation_sum_ms: 22,
                research_cancel_propagation_count: 23,
            },
            sentinel_leaks: 24,
            empty_retries: 25,
            turn_ok: 26,
            turn_err: 27,
            steer_delivered: 28,
            steer_soft_interrupted: 29,
            steer_drained_on_abort: 30,
            supervisor_restart: 31,
            snapshot_capture: 32,
            lsp_emitted: 33,
            permission_match: 34,
            hook_fire: 35,
            subagent_resolve: 36,
            provider_perm_blacklist: 37,
            crash_notified: 38,
            vision_mismatch: 39,
            provider_connect_timeout: 390,
            provider_inter_chunk_timeout: 391,
            provider_invalid_request: 392,
            scheduler_dispatch_skipped: 393,
            cfg_ext_write: 40,
            ip_hallucin: 41,
            research_store_corrupt_rows: 42,
            research_store_files_healed: 43,
            research_store_heal_failed: 44,
            research_store_file_lock_wait: 45,
            concurrent_same_key_turns: 46,
            text_coalesced: 47,
            session_busy_ack: 48,
            memory_pollution: 0,
            model_health_body: "# model health\nnaked_model_health_probe 777\n".to_string(),
        };
        let expected = concat!(
            "# HELP naked_tg_native_route_chosen_total Photos sent via native path.\n",
            "# TYPE naked_tg_native_route_chosen_total counter\n",
            "naked_tg_native_route_chosen_total 1\n",
            "# HELP naked_tg_native_route_downgraded_oversize_total Native batches with oversize hint.\n",
            "# TYPE naked_tg_native_route_downgraded_oversize_total counter\n",
            "naked_tg_native_route_downgraded_oversize_total 2\n",
            "# HELP naked_tg_describer_fallback_total Batches that fell back to the describer.\n",
            "# TYPE naked_tg_describer_fallback_total counter\n",
            "naked_tg_describer_fallback_total 3\n",
            "# HELP naked_core_sentinel_leak_stripped_total Image-ref sentinels stripped before reaching an LLM.\n",
            "# TYPE naked_core_sentinel_leak_stripped_total counter\n",
            "naked_core_sentinel_leak_stripped_total 24\n",
            "# HELP naked_tg_rate_limit_delayed_total TG sends that waited for the 60/min window to free up.\n",
            "# TYPE naked_tg_rate_limit_delayed_total counter\n",
            "naked_tg_rate_limit_delayed_total 4\n",
            "# HELP naked_core_empty_content_retry_total Times the loop retried a turn after the provider closed the stream with no content (glm-5-turbo class).\n",
            "# TYPE naked_core_empty_content_retry_total counter\n",
            "naked_core_empty_content_retry_total 25\n",
            "# HELP naked_core_turn_completed_total Agent turns that finished successfully.\n",
            "# TYPE naked_core_turn_completed_total counter\n",
            "naked_core_turn_completed_total 26\n",
            "# HELP naked_core_turn_error_total Agent turns that ended with an error.\n",
            "# TYPE naked_core_turn_error_total counter\n",
            "naked_core_turn_error_total 27\n",
            "# HELP naked_core_steer_delivered_total Steer messages successfully merged into history.\n",
            "# TYPE naked_core_steer_delivered_total counter\n",
            "naked_core_steer_delivered_total 28\n",
            "# HELP naked_core_steer_soft_interrupted_total Times the LLM stream was soft-interrupted by a mid-stream steer (S2/S3 path).\n",
            "# TYPE naked_core_steer_soft_interrupted_total counter\n",
            "naked_core_steer_soft_interrupted_total 29\n",
            "# HELP naked_core_steer_drained_on_abort_total Times the run-loop's drain-on-error rescued in-flight steers from a dying turn.\n",
            "# TYPE naked_core_steer_drained_on_abort_total counter\n",
            "naked_core_steer_drained_on_abort_total 30\n",
            "# HELP naked_tg_supervisor_panic_restart_total Panic-triggered restarts inside spawn_supervised_with_opts.\n",
            "# TYPE naked_tg_supervisor_panic_restart_total counter\n",
            "naked_tg_supervisor_panic_restart_total 31\n",
            "# HELP naked_tg_stream_button_click_abort_total Clicks of the [⏹ Стоп] inline button on the streaming control card.\n",
            "# TYPE naked_tg_stream_button_click_abort_total counter\n",
            "naked_tg_stream_button_click_abort_total 5\n",
            "# HELP naked_tg_stream_button_click_sendnow_total Clicks of the [⏩ Send now] inline button on the streaming control card.\n",
            "# TYPE naked_tg_stream_button_click_sendnow_total counter\n",
            "naked_tg_stream_button_click_sendnow_total 6\n",
            "# HELP naked_tg_run_registry_register_total RunRegistry run registrations.\n",
            "# TYPE naked_tg_run_registry_register_total counter\n",
            "naked_tg_run_registry_register_total 61\n",
            "# HELP naked_tg_run_registry_remove_total RunRegistry run removals.\n",
            "# TYPE naked_tg_run_registry_remove_total counter\n",
            "naked_tg_run_registry_remove_total 62\n",
            "# HELP naked_tg_run_registry_cap_reject_total RunRegistry thread cap rejections.\n",
            "# TYPE naked_tg_run_registry_cap_reject_total counter\n",
            "naked_tg_run_registry_cap_reject_total 63\n",
            "# HELP naked_tg_run_registry_callback_expired_total Expired legacy stream callbacks.\n",
            "# TYPE naked_tg_run_registry_callback_expired_total counter\n",
            "naked_tg_run_registry_callback_expired_total 64\n",
            "# HELP naked_tg_run_registry_callback_resolved_total Stream callbacks resolved to a live run.\n",
            "# TYPE naked_tg_run_registry_callback_resolved_total counter\n",
            "naked_tg_run_registry_callback_resolved_total 65\n",
            "# HELP naked_tg_run_registry_callback_denied_total Stream callbacks denied by allowed_chat_ids.\n",
            "# TYPE naked_tg_run_registry_callback_denied_total counter\n",
            "naked_tg_run_registry_callback_denied_total 66\n",
            "# HELP naked_core_snapshot_capture_total Side-git pre-turn workspace snapshots captured by `dispatch_turn`.\n",
            "# TYPE naked_core_snapshot_capture_total counter\n",
            "naked_core_snapshot_capture_total 32\n",
            "# HELP naked_core_lsp_diagnostic_emitted_total Non-empty LSP diagnostics blocks injected into history after edit tools.\n",
            "# TYPE naked_core_lsp_diagnostic_emitted_total counter\n",
            "naked_core_lsp_diagnostic_emitted_total 33\n",
            "# HELP naked_core_permission_rule_match_total Non-`Ask` decisions from the pattern-based permission ruleset.\n",
            "# TYPE naked_core_permission_rule_match_total counter\n",
            "naked_core_permission_rule_match_total 34\n",
            "# HELP naked_core_lifecycle_hook_fire_total Lifecycle hooks that actually matched and executed.\n",
            "# TYPE naked_core_lifecycle_hook_fire_total counter\n",
            "naked_core_lifecycle_hook_fire_total 35\n",
            "# HELP naked_core_subagent_role_resolve_total Sub-agent calls whose `mode` parameter resolved to a canonical role.\n",
            "# TYPE naked_core_subagent_role_resolve_total counter\n",
            "naked_core_subagent_role_resolve_total 36\n",
            "# HELP naked_core_provider_permanent_blacklist_total Provider keys permanently removed from rotation due to auth/payment errors.\n",
            "# TYPE naked_core_provider_permanent_blacklist_total counter\n",
            "naked_core_provider_permanent_blacklist_total 37\n",
            "# HELP naked_core_crash_recovery_notified_total Users notified about interrupted sessions on bot boot.\n",
            "# TYPE naked_core_crash_recovery_notified_total counter\n",
            "naked_core_crash_recovery_notified_total 38\n",
            "# HELP naked_core_provider_vision_capability_mismatch_total (B06) caps.supports_vision=true but API rejects image_url content shape.\n",
            "# TYPE naked_core_provider_vision_capability_mismatch_total counter\n",
            "naked_core_provider_vision_capability_mismatch_total 39\n",
            "# HELP naked_core_provider_timeout_total (B68) Provider timeout events by timeout kind.\n",
            "# TYPE naked_core_provider_timeout_total counter\n",
            "naked_core_provider_timeout_total{kind=\"connect\"} 390\n",
            "naked_core_provider_timeout_total{kind=\"inter_chunk\"} 391\n",
            "# HELP naked_core_provider_invalid_request_total (B76) Provider HTTP 400 invalid_request_error responses (likely config/request-shape bugs).\n",
            "# TYPE naked_core_provider_invalid_request_total counter\n",
            "naked_core_provider_invalid_request_total 392\n",
            "# HELP naked_core_scheduler_dispatch_skipped_total (B75) Scheduler dispatches skipped because all provider keys/providers were blacklisted.\n",
            "# TYPE naked_core_scheduler_dispatch_skipped_total counter\n",
            "naked_core_scheduler_dispatch_skipped_total 393\n",
            "# HELP naked_core_config_external_write_total (B42) state/naked.json modified by external process detected by D-CONFIG-MTIME-WATCH.\n",
            "# TYPE naked_core_config_external_write_total counter\n",
            "naked_core_config_external_write_total 40\n",
            "# HELP naked_core_ip_token_hallucination_total (B37) Outgoing assistant messages mentioning noVNC with IP tokens not in the boot-cached allow-list.\n",
            "# TYPE naked_core_ip_token_hallucination_total counter\n",
            "naked_core_ip_token_hallucination_total 41\n",
            "# HELP naked_core_research_store_corrupt_rows_detected_total (B59) Invalid findings.jsonl rows detected by the research store healing path.\n",
            "# TYPE naked_core_research_store_corrupt_rows_detected_total counter\n",
            "naked_core_research_store_corrupt_rows_detected_total 42\n",
            "# HELP naked_core_research_store_files_healed_total (B59) findings.jsonl files successfully healed after corrupt-row detection.\n",
            "# TYPE naked_core_research_store_files_healed_total counter\n",
            "naked_core_research_store_files_healed_total 43\n",
            "# HELP naked_core_research_store_heal_failed_total (B59) Research-store healing attempts that failed before replacing the live file.\n",
            "# TYPE naked_core_research_store_heal_failed_total counter\n",
            "naked_core_research_store_heal_failed_total 44\n",
            "# HELP naked_core_research_store_file_lock_wait_total (B59) findings.jsonl write-lock acquisitions that had to wait on contention.\n",
            "# TYPE naked_core_research_store_file_lock_wait_total counter\n",
            "naked_core_research_store_file_lock_wait_total 45\n",
            "# HELP naked_memory_pollution_count research:* lines found in global MEMORY.md by the daily housekeep sweep (should stay 0).\n",
            "# TYPE naked_memory_pollution_count gauge\n",
            "naked_memory_pollution_count 0\n",
            "# HELP naked_tg_media_transcription_total Audio transcription outcomes (label `outcome`: ok or fail).\n",
            "# TYPE naked_tg_media_transcription_total counter\n",
            "naked_tg_media_transcription_total{outcome=\"ok\"} 7\n",
            "naked_tg_media_transcription_total{outcome=\"fail\"} 8\n",
            "# HELP naked_tg_media_transcription_failure_total Audio transcription failures by classified reason (cardinality-safe).\n",
            "# TYPE naked_tg_media_transcription_failure_total counter\n",
            "naked_tg_media_transcription_failure_total{reason=\"auth\"} 9\n",
            "naked_tg_media_transcription_failure_total{reason=\"rate_limit\"} 10\n",
            "naked_tg_media_transcription_failure_total{reason=\"payload\"} 11\n",
            "naked_tg_media_transcription_failure_total{reason=\"timeout\"} 12\n",
            "naked_tg_media_transcription_failure_total{reason=\"network\"} 13\n",
            "naked_tg_media_transcription_failure_total{reason=\"other\"} 14\n",
            "# HELP naked_tg_redaction_applied_total (B45) Outgoing TG messages where scan_and_redact modified payload.\n",
            "# TYPE naked_tg_redaction_applied_total counter\n",
            "naked_tg_redaction_applied_total 15\n",
            "# HELP naked_tg_concurrent_same_key_turns_total (B61) Same (chat,thread) dispatches that waited on the per-key short-section lock.\n",
            "# TYPE naked_tg_concurrent_same_key_turns_total counter\n",
            "naked_tg_concurrent_same_key_turns_total 46\n",
            "# HELP naked_tg_text_coalesced_total (B60) Text bursts merged (>=2 client-split messages joined into one turn).\n",
            "# TYPE naked_tg_text_coalesced_total counter\n",
            "naked_tg_text_coalesced_total 47\n",
            "# HELP naked_tg_session_busy_ack_total (B62) SessionBusy follow-ups that got a soft busy ack (no steer sender / full channel); measures the F1/B3 drop window.\n",
            "# TYPE naked_tg_session_busy_ack_total counter\n",
            "naked_tg_session_busy_ack_total 48\n",
            "# HELP naked_tg_config_describer_missing (B05) Set to 1 when the default model is not vision-capable AND no tg_media.vision describer fallback is configured.\n",
            "# TYPE naked_tg_config_describer_missing gauge\n",
            "naked_tg_config_describer_missing 1\n",
            "# HELP naked_tg_research_cancel_propagation_total (T11/B56) Cancel propagation latency buckets (ms).\n",
            "# TYPE naked_tg_research_cancel_propagation_total counter\n",
            "naked_tg_research_cancel_propagation_total{bucket=\"under_3s\"} 19\n",
            "naked_tg_research_cancel_propagation_total{bucket=\"3s_to_30s\"} 20\n",
            "naked_tg_research_cancel_propagation_total{bucket=\"over_30s\"} 21\n",
            "# HELP naked_tg_research_cancel_propagation_sum_ms (T11) Sum of cancel propagation latencies in ms.\n",
            "# TYPE naked_tg_research_cancel_propagation_sum_ms counter\n",
            "naked_tg_research_cancel_propagation_sum_ms 22\n",
            "# HELP naked_tg_research_cancel_propagation_count (T11) Total cancel observations recorded.\n",
            "# TYPE naked_tg_research_cancel_propagation_count counter\n",
            "naked_tg_research_cancel_propagation_count 23\n",
            "# model health\n",
            "naked_model_health_probe 777\n",
        );
        assert_eq!(render_prometheus_from(&inputs), expected);
    }

    #[test]
    fn render_prometheus_includes_provider_timeout_labels() {
        let inputs = PrometheusInputs {
            provider_connect_timeout: 11,
            provider_inter_chunk_timeout: 12,
            ..PrometheusInputs::default()
        };
        let rendered = render_prometheus_from(&inputs);
        assert!(rendered.contains("# HELP naked_core_provider_timeout_total"));
        assert!(rendered.contains("# TYPE naked_core_provider_timeout_total counter"));
        assert!(rendered.contains("naked_core_provider_timeout_total{kind=\"connect\"} 11"));
        assert!(rendered.contains("naked_core_provider_timeout_total{kind=\"inter_chunk\"} 12"));
    }

    #[test]
    fn render_prometheus_includes_provider_invalid_request_counter() {
        let inputs = PrometheusInputs {
            provider_invalid_request: 13,
            ..PrometheusInputs::default()
        };
        let rendered = render_prometheus_from(&inputs);
        assert!(rendered.contains("# HELP naked_core_provider_invalid_request_total"));
        assert!(rendered.contains("# TYPE naked_core_provider_invalid_request_total counter"));
        assert!(rendered.contains("naked_core_provider_invalid_request_total 13"));
    }

    #[test]
    fn render_prometheus_includes_scheduler_dispatch_skipped_counter() {
        let inputs = PrometheusInputs {
            scheduler_dispatch_skipped: 14,
            ..PrometheusInputs::default()
        };
        let rendered = render_prometheus_from(&inputs);
        assert!(rendered.contains("# HELP naked_core_scheduler_dispatch_skipped_total"));
        assert!(rendered.contains("# TYPE naked_core_scheduler_dispatch_skipped_total counter"));
        assert!(rendered.contains("naked_core_scheduler_dispatch_skipped_total 14"));
    }

    #[test]
    fn render_prometheus_includes_config_describer_missing_gauge() {
        let inputs = PrometheusInputs {
            media: MediaRoutingSnapshot {
                config_describer_missing: 1,
                ..MediaRoutingSnapshot::default()
            },
            ..PrometheusInputs::default()
        };
        let rendered = render_prometheus_from(&inputs);
        assert!(rendered.contains("# HELP naked_tg_config_describer_missing (B05)"));
        assert!(rendered.contains("# TYPE naked_tg_config_describer_missing gauge"));
        assert!(rendered.contains("naked_tg_config_describer_missing 1"));
        assert!(!rendered.contains("naked_tg_config_describer_missing_total"));
    }

    #[test]
    fn set_config_describer_missing_updates_snapshot_gauge_state() {
        set_config_describer_missing(false);
        assert_eq!(snapshot().config_describer_missing, 0);
        set_config_describer_missing(true);
        assert_eq!(snapshot().config_describer_missing, 1);
        set_config_describer_missing(false);
        assert_eq!(snapshot().config_describer_missing, 0);
    }

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
        assert!(p.contains("# HELP naked_tg_concurrent_same_key_turns_total"));
        assert!(p.contains("# TYPE naked_tg_concurrent_same_key_turns_total counter"));
    }

    #[test]
    fn render_prometheus_includes_research_store_healing_counters() {
        let p = snapshot().render_prometheus();
        for metric in [
            "naked_core_research_store_corrupt_rows_detected_total",
            "naked_core_research_store_files_healed_total",
            "naked_core_research_store_heal_failed_total",
        ] {
            assert!(
                p.contains(metric),
                "render_prometheus must expose `{metric}`; full body:\n{p}",
            );
            assert!(
                p.contains(&format!("# TYPE {metric} counter")),
                "missing TYPE for {metric}",
            );
        }
    }

    #[test]
    fn render_prometheus_includes_text_coalesced_and_file_lock_wait_counters() {
        let p = snapshot().render_prometheus();
        for metric in [
            "naked_tg_text_coalesced_total",
            "naked_tg_session_busy_ack_total",
            "naked_core_research_store_file_lock_wait_total",
        ] {
            assert!(
                p.contains(metric),
                "render_prometheus must expose `{metric}`; full body:\n{p}",
            );
            assert!(
                p.contains(&format!("# TYPE {metric} counter")),
                "missing TYPE for {metric}",
            );
        }
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
    fn registry_metrics_render_and_bump() {
        let before = snapshot();
        record_run_registry_register();
        record_run_registry_remove();
        record_run_registry_cap_reject();
        record_run_registry_callback_expired();
        record_run_registry_callback_resolved();
        record_run_registry_callback_denied();
        let after = snapshot();
        assert_eq!(
            after.run_registry_register,
            before.run_registry_register + 1
        );
        assert_eq!(after.run_registry_remove, before.run_registry_remove + 1);
        assert_eq!(
            after.run_registry_cap_reject,
            before.run_registry_cap_reject + 1
        );
        assert_eq!(
            after.run_registry_callback_expired,
            before.run_registry_callback_expired + 1
        );
        assert_eq!(
            after.run_registry_callback_resolved,
            before.run_registry_callback_resolved + 1
        );
        assert_eq!(
            after.run_registry_callback_denied,
            before.run_registry_callback_denied + 1
        );
        let rendered = after.render_prometheus();
        assert!(rendered.contains("naked_tg_run_registry_register_total"));
        assert!(rendered.contains("naked_tg_run_registry_callback_expired_total"));
        assert!(rendered.contains("naked_tg_run_registry_callback_resolved_total"));
        assert!(rendered.contains("naked_tg_run_registry_callback_denied_total"));
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

    /// T11 (PLAN_RESEARCH_AGENT_FLOW_v1): cancel-propagation metric
    /// buckets are present in /metrics output. Three buckets keep
    /// cardinality low while flagging outliers (B56-class regressions
    /// would show up as `bucket="over_30s"` count > 0).
    #[test]
    fn research_cancel_propagation_renders_to_prometheus() {
        let snap = MediaRoutingSnapshot::default();
        let body = snap.render_prometheus();
        assert!(
            body.contains("naked_tg_research_cancel_propagation_total"),
            "propagation counter must be in /metrics"
        );
        assert!(
            body.contains("bucket=\"under_3s\""),
            "under_3s bucket must be exposed"
        );
        assert!(
            body.contains("bucket=\"over_30s\""),
            "over_30s bucket must be exposed (alert-worthy regressions)"
        );
        assert!(
            body.contains("naked_tg_research_cancel_propagation_sum_ms"),
            "sum_ms gauge must be exposed for avg latency calc"
        );
    }

    /// B3 (PLAN_RESEARCH_FLOW_CLOSURE_v1): both cancel paths (the r:stop
    /// inline button AND the /abort command) must record the
    /// propagation metric. Source-text sentinel so a refactor that
    /// drops one path surfaces here loudly instead of silently leaving
    /// the metric blind to half the cancels.
    #[test]
    fn both_cancel_paths_record_propagation_metric() {
        // The r:stop callback path lives in callbacks/research.rs.
        let callbacks_src = include_str!("callbacks/research.rs");
        assert!(
            callbacks_src.contains("record_research_cancel_propagation"),
            "r:stop callback must record cancel propagation (T11 wiring)"
        );
        // The /abort command path lives in commands/session.rs.
        let session_src = include_str!("commands/session.rs");
        assert!(
            session_src.contains("record_research_cancel_propagation"),
            "cmd_abort must record cancel propagation (B3 closure)"
        );
        // Both paths must take the timestamp BEFORE the abort call so
        // the elapsed includes the actual abort work, not zero.
        assert!(
            session_src.contains("cancel_started = std::time::Instant::now"),
            "cmd_abort must capture Instant BEFORE agent.abort"
        );
    }

    /// T11: record_research_cancel_propagation classifies into correct bucket.
    /// Note: this test mutates global atomics so it must NOT run in
    /// parallel with `prometheus_render` (which reads the same statics).
    /// In practice cargo test default jobs > 1, but the buckets are
    /// monotonic counters so multiple writers/readers just race on
    /// strictly-increasing counts — no torn writes, no false negatives.
    #[test]
    fn record_research_cancel_propagation_classifies_buckets() {
        let before_u3s = RESEARCH_CANCEL_PROPAGATION_UNDER_3S.load(Ordering::Relaxed);
        let before_3s_30s = RESEARCH_CANCEL_PROPAGATION_3S_TO_30S.load(Ordering::Relaxed);
        let before_o30s = RESEARCH_CANCEL_PROPAGATION_OVER_30S.load(Ordering::Relaxed);
        let before_count = RESEARCH_CANCEL_PROPAGATION_COUNT.load(Ordering::Relaxed);

        record_research_cancel_propagation(1_500); // under_3s
        record_research_cancel_propagation(5_000); // 3s_to_30s
        record_research_cancel_propagation(45_000); // over_30s

        let after_u3s = RESEARCH_CANCEL_PROPAGATION_UNDER_3S.load(Ordering::Relaxed);
        let after_3s_30s = RESEARCH_CANCEL_PROPAGATION_3S_TO_30S.load(Ordering::Relaxed);
        let after_o30s = RESEARCH_CANCEL_PROPAGATION_OVER_30S.load(Ordering::Relaxed);
        let after_count = RESEARCH_CANCEL_PROPAGATION_COUNT.load(Ordering::Relaxed);

        assert!(after_u3s > before_u3s, "1500ms must bump under_3s bucket");
        assert!(
            after_3s_30s > before_3s_30s,
            "5000ms must bump 3s_to_30s bucket"
        );
        assert!(
            after_o30s > before_o30s,
            "45000ms must bump over_30s bucket (alert-worthy)"
        );
        assert_eq!(
            after_count - before_count,
            3,
            "all 3 observations must increment count"
        );
    }
}
