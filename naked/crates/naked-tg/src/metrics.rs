//! Lightweight in-process counters for media-routing decisions and other
//! interesting events. Exposed via `/metrics` operator command and also
//! emitted as `tracing::info!` events so existing log shippers pick them up.
//!
//! Why not `metrics`/`prometheus` crates? `naked-tg` ships as a single user
//! binary today and we want zero extra runtime overhead or registries to
//! configure. When we promote multimodal routing to production we can swap
//! these to `metrics::counter!` calls without changing call-sites.

use naked_core::memory::daily::DigestStalenessSnapshot;
use naked_core::metrics_hist::{
    DurationHistogramSnapshot, LatencySnapshot, ToolClass, tool_snapshot,
};
use std::path::PathBuf;
use std::sync::OnceLock;
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

// ── PLAN_TG_LONG_ANSWERS_v2 S5 / BUG_REGISTRY B101+B102 ─────────────
// Final-answer observability. Histogram buckets are stored as exclusive
// atomics here, then rendered as cumulative Prometheus `_bucket{le=...}`
// series so `histogram_quantile` sees a real histogram.
static FINAL_ANSWER_UTF16_LE_1024: AtomicU64 = AtomicU64::new(0);
static FINAL_ANSWER_UTF16_1025_TO_2048: AtomicU64 = AtomicU64::new(0);
static FINAL_ANSWER_UTF16_2049_TO_4096: AtomicU64 = AtomicU64::new(0);
static FINAL_ANSWER_UTF16_4097_TO_8192: AtomicU64 = AtomicU64::new(0);
static FINAL_ANSWER_UTF16_8193_TO_16384: AtomicU64 = AtomicU64::new(0);
static FINAL_ANSWER_UTF16_16385_TO_32768: AtomicU64 = AtomicU64::new(0);
static FINAL_ANSWER_UTF16_OVER_32768: AtomicU64 = AtomicU64::new(0);
static FINAL_ANSWER_UTF16_SUM: AtomicU64 = AtomicU64::new(0);
static FINAL_ANSWER_UTF16_COUNT: AtomicU64 = AtomicU64::new(0);
static FINAL_ANSWER_DELIVERY_OK: AtomicU64 = AtomicU64::new(0);
static FINAL_ANSWER_DELIVERY_PARTIAL: AtomicU64 = AtomicU64::new(0);
static FINAL_ANSWER_DELIVERY_FAILED: AtomicU64 = AtomicU64::new(0);
static FINAL_ANSWER_TRUNCATED: AtomicU64 = AtomicU64::new(0);
static FINAL_ANSWER_ATTACHMENT_SEND_FAILED: AtomicU64 = AtomicU64::new(0);

/// B05: current boot-time config health for multimodal fallback.
/// State gauge: 1 when the default model is not vision-capable AND no
/// `tg_media.vision` describer fallback is configured, otherwise 0.
static CONFIG_DESCRIBER_MISSING: AtomicU64 = AtomicU64::new(0);

static MEMORY_METRICS_WORKSPACE: OnceLock<PathBuf> = OnceLock::new();

pub fn set_memory_metrics_workspace(workspace: PathBuf) {
    let _ = MEMORY_METRICS_WORKSPACE.set(workspace);
}

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

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum FinalAnswerDeliveryOutcome {
    Ok,
    Partial,
    Failed,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub(crate) struct FinalAnswerUtf16HistogramSnapshot {
    /// Exclusive bucket counts in `FINAL_ANSWER_UTF16_BUCKET_LABELS` order.
    buckets: [u64; FINAL_ANSWER_UTF16_BUCKET_LABELS.len()],
    sum: u64,
    count: u64,
}

const FINAL_ANSWER_UTF16_BUCKET_LABELS: [&str; 7] =
    ["1024", "2048", "4096", "8192", "16384", "32768", "+Inf"];
const FINAL_ANSWER_UTF16_BUCKET_EDGES: [u64; 6] = [1024, 2048, 4096, 8192, 16384, 32768];

pub(crate) fn record_final_answer_utf16_from_html(html: &str) -> u64 {
    let utf16_units = naked_tg::markup::telegram_html_text_utf16_units(html);
    record_final_answer_utf16_units(utf16_units);
    utf16_units
}

pub(crate) fn record_final_answer_delivery(outcome: FinalAnswerDeliveryOutcome) {
    match outcome {
        FinalAnswerDeliveryOutcome::Ok => FINAL_ANSWER_DELIVERY_OK.fetch_add(1, Ordering::Relaxed),
        FinalAnswerDeliveryOutcome::Partial => {
            FINAL_ANSWER_DELIVERY_PARTIAL.fetch_add(1, Ordering::Relaxed)
        }
        FinalAnswerDeliveryOutcome::Failed => {
            FINAL_ANSWER_DELIVERY_FAILED.fetch_add(1, Ordering::Relaxed)
        }
    };
}

pub(crate) fn record_final_answer_truncated() {
    FINAL_ANSWER_TRUNCATED.fetch_add(1, Ordering::Relaxed);
}

pub(crate) fn record_final_answer_attachment_send_failed() {
    FINAL_ANSWER_ATTACHMENT_SEND_FAILED.fetch_add(1, Ordering::Relaxed);
}

fn record_final_answer_utf16_units(utf16_units: u64) {
    let bucket = if utf16_units <= FINAL_ANSWER_UTF16_BUCKET_EDGES[0] {
        &FINAL_ANSWER_UTF16_LE_1024
    } else if utf16_units <= FINAL_ANSWER_UTF16_BUCKET_EDGES[1] {
        &FINAL_ANSWER_UTF16_1025_TO_2048
    } else if utf16_units <= FINAL_ANSWER_UTF16_BUCKET_EDGES[2] {
        &FINAL_ANSWER_UTF16_2049_TO_4096
    } else if utf16_units <= FINAL_ANSWER_UTF16_BUCKET_EDGES[3] {
        &FINAL_ANSWER_UTF16_4097_TO_8192
    } else if utf16_units <= FINAL_ANSWER_UTF16_BUCKET_EDGES[4] {
        &FINAL_ANSWER_UTF16_8193_TO_16384
    } else if utf16_units <= FINAL_ANSWER_UTF16_BUCKET_EDGES[5] {
        &FINAL_ANSWER_UTF16_16385_TO_32768
    } else {
        &FINAL_ANSWER_UTF16_OVER_32768
    };
    bucket.fetch_add(1, Ordering::Relaxed);
    FINAL_ANSWER_UTF16_SUM.fetch_add(utf16_units, Ordering::Relaxed);
    FINAL_ANSWER_UTF16_COUNT.fetch_add(1, Ordering::Relaxed);
}

fn final_answer_utf16_snapshot() -> FinalAnswerUtf16HistogramSnapshot {
    FinalAnswerUtf16HistogramSnapshot {
        buckets: [
            FINAL_ANSWER_UTF16_LE_1024.load(Ordering::Relaxed),
            FINAL_ANSWER_UTF16_1025_TO_2048.load(Ordering::Relaxed),
            FINAL_ANSWER_UTF16_2049_TO_4096.load(Ordering::Relaxed),
            FINAL_ANSWER_UTF16_4097_TO_8192.load(Ordering::Relaxed),
            FINAL_ANSWER_UTF16_8193_TO_16384.load(Ordering::Relaxed),
            FINAL_ANSWER_UTF16_16385_TO_32768.load(Ordering::Relaxed),
            FINAL_ANSWER_UTF16_OVER_32768.load(Ordering::Relaxed),
        ],
        sum: FINAL_ANSWER_UTF16_SUM.load(Ordering::Relaxed),
        count: FINAL_ANSWER_UTF16_COUNT.load(Ordering::Relaxed),
    }
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

#[cfg(test)]
pub(crate) static TRANSCRIPTION_TEST_LOCK: tokio::sync::Mutex<()> =
    tokio::sync::Mutex::const_new(());

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
        final_answer_utf16: final_answer_utf16_snapshot(),
        final_answer_delivery_ok: FINAL_ANSWER_DELIVERY_OK.load(Ordering::Relaxed),
        final_answer_delivery_partial: FINAL_ANSWER_DELIVERY_PARTIAL.load(Ordering::Relaxed),
        final_answer_delivery_failed: FINAL_ANSWER_DELIVERY_FAILED.load(Ordering::Relaxed),
        final_answer_truncated: FINAL_ANSWER_TRUNCATED.load(Ordering::Relaxed),
        final_answer_attachment_send_failed: FINAL_ANSWER_ATTACHMENT_SEND_FAILED
            .load(Ordering::Relaxed),
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
    /// PLAN_TG_LONG_ANSWERS_v2 S5: final-answer UTF-16 length histogram.
    pub(crate) final_answer_utf16: FinalAnswerUtf16HistogramSnapshot,
    pub final_answer_delivery_ok: u64,
    pub final_answer_delivery_partial: u64,
    pub final_answer_delivery_failed: u64,
    pub final_answer_truncated: u64,
    pub final_answer_attachment_send_failed: u64,
    /// B05: 0/1 state gauge for missing multimodal describer fallback.
    pub config_describer_missing: u64,
    // T11 PLAN_RESEARCH_AGENT_FLOW_v1: cancel propagation latency.
    pub research_cancel_propagation_under_3s: u64,
    pub research_cancel_propagation_3s_to_30s: u64,
    pub research_cancel_propagation_over_30s: u64,
    pub research_cancel_propagation_sum_ms: u64,
    pub research_cancel_propagation_count: u64,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct PrometheusInputs {
    media: MediaRoutingSnapshot,
    latency: LatencySnapshot,
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
    stale_edit_reject: u64,
    git_history_guard_block: u64,
    hashline_edit_applied: u64,
    hashline_edit_stale_anchor: u64,
    hashline_edit_overlap: u64,
    hashline_edit_out_of_bounds: u64,
    hashline_edit_disabled: u64,
    fs_cache_hit: u64,
    fs_cache_miss: u64,
    fs_cache_stale_bypass: u64,
    fs_cache_invalidate: u64,
    fs_cache_too_large: u64,
    persistent_bash_ok: u64,
    persistent_bash_timeout: u64,
    persistent_bash_killed: u64,
    persistent_bash_restart: u64,
    persistent_bash_error: u64,
    persistent_bash_disabled: u64,
    persistent_bash_busy: u64,
    fff_picker_created: u64,
    fff_picker_reused: u64,
    fff_picker_cap_fallback: u64,
    fff_grep_fast_index: u64,
    fff_grep_fallback: u64,
    ip_hallucin: u64,
    research_store_corrupt_rows: u64,
    research_store_files_healed: u64,
    research_store_heal_failed: u64,
    research_store_file_lock_wait: u64,
    memory_candidates: u64,
    memory_promoted_repeat_days: u64,
    memory_promoted_reinforce: u64,
    memory_promoted_reobs: u64,
    memory_injection_dropped_global: u64,
    memory_injection_dropped_project: u64,
    memory_injection_dropped_user: u64,
    memory_injection_failed: u64,
    memory_digest_staleness: DigestStalenessSnapshot,
    concurrent_same_key_turns: u64,
    text_coalesced: u64,
    session_busy_ack: u64,
    memory_pollution: u64,
    active_run_max_silent_seconds: u64,
    config_loaded_hash: Option<String>,
    model_health_body: String,
}

impl Default for PrometheusInputs {
    fn default() -> Self {
        Self {
            media: MediaRoutingSnapshot::default(),
            latency: LatencySnapshot {
                turn_under_1s: 0,
                turn_1s_to_10s: 0,
                turn_10s_to_60s: 0,
                turn_over_60s: 0,
                turn_sum_ms: 0,
                turn_count: 0,
                ttft_under_500ms: 0,
                ttft_500ms_to_2s: 0,
                ttft_2s_to_10s: 0,
                ttft_over_10s: 0,
                ttft_sum_ms: 0,
                ttft_count: 0,
                provider_under_500ms: 0,
                provider_500ms_to_2s: 0,
                provider_2s_to_10s: 0,
                provider_over_10s: 0,
                provider_sum_ms: 0,
                provider_count: 0,
                tool_grep: DurationHistogramSnapshot::default(),
                tool_read: DurationHistogramSnapshot::default(),
                tool_edit: DurationHistogramSnapshot::default(),
                tool_write: DurationHistogramSnapshot::default(),
                tool_bash: DurationHistogramSnapshot::default(),
                tool_apply_patch: DurationHistogramSnapshot::default(),
                tool_other: DurationHistogramSnapshot::default(),
                fff_cold_build: DurationHistogramSnapshot::default(),
                parent_fsync: DurationHistogramSnapshot::default(),
            },
            sentinel_leaks: 0,
            empty_retries: 0,
            turn_ok: 0,
            turn_err: 0,
            steer_delivered: 0,
            steer_soft_interrupted: 0,
            steer_drained_on_abort: 0,
            supervisor_restart: 0,
            snapshot_capture: 0,
            lsp_emitted: 0,
            permission_match: 0,
            hook_fire: 0,
            subagent_resolve: 0,
            provider_perm_blacklist: 0,
            crash_notified: 0,
            vision_mismatch: 0,
            provider_connect_timeout: 0,
            provider_inter_chunk_timeout: 0,
            provider_invalid_request: 0,
            scheduler_dispatch_skipped: 0,
            cfg_ext_write: 0,
            stale_edit_reject: 0,
            git_history_guard_block: 0,
            hashline_edit_applied: 0,
            hashline_edit_stale_anchor: 0,
            hashline_edit_overlap: 0,
            hashline_edit_out_of_bounds: 0,
            hashline_edit_disabled: 0,
            fs_cache_hit: 0,
            fs_cache_miss: 0,
            fs_cache_stale_bypass: 0,
            fs_cache_invalidate: 0,
            fs_cache_too_large: 0,
            persistent_bash_ok: 0,
            persistent_bash_timeout: 0,
            persistent_bash_killed: 0,
            persistent_bash_restart: 0,
            persistent_bash_error: 0,
            persistent_bash_disabled: 0,
            persistent_bash_busy: 0,
            fff_picker_created: 0,
            fff_picker_reused: 0,
            fff_picker_cap_fallback: 0,
            fff_grep_fast_index: 0,
            fff_grep_fallback: 0,
            ip_hallucin: 0,
            research_store_corrupt_rows: 0,
            research_store_files_healed: 0,
            research_store_heal_failed: 0,
            research_store_file_lock_wait: 0,
            memory_candidates: 0,
            memory_promoted_repeat_days: 0,
            memory_promoted_reinforce: 0,
            memory_promoted_reobs: 0,
            memory_injection_dropped_global: 0,
            memory_injection_dropped_project: 0,
            memory_injection_dropped_user: 0,
            memory_injection_failed: 0,
            memory_digest_staleness: DigestStalenessSnapshot::default(),
            concurrent_same_key_turns: 0,
            text_coalesced: 0,
            session_busy_ack: 0,
            memory_pollution: 0,
            active_run_max_silent_seconds: 0,
            config_loaded_hash: None,
            model_health_body: String::new(),
        }
    }
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
            latency: naked_core::metrics_hist::snapshot(),
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
            stale_edit_reject: naked_core::types::STALE_EDIT_REJECT_COUNT.load(Ordering::Relaxed),
            git_history_guard_block: naked_core::types::GIT_HISTORY_GUARD_BLOCK_COUNT
                .load(Ordering::Relaxed),
            hashline_edit_applied: naked_core::types::HASHLINE_EDIT_APPLIED_COUNT
                .load(Ordering::Relaxed),
            hashline_edit_stale_anchor: naked_core::types::HASHLINE_EDIT_STALE_ANCHOR_COUNT
                .load(Ordering::Relaxed),
            hashline_edit_overlap: naked_core::types::HASHLINE_EDIT_OVERLAP_COUNT
                .load(Ordering::Relaxed),
            hashline_edit_out_of_bounds: naked_core::types::HASHLINE_EDIT_OUT_OF_BOUNDS_COUNT
                .load(Ordering::Relaxed),
            hashline_edit_disabled: naked_core::types::HASHLINE_EDIT_DISABLED_COUNT
                .load(Ordering::Relaxed),
            fs_cache_hit: naked_core::types::FS_CACHE_HIT_COUNT.load(Ordering::Relaxed),
            fs_cache_miss: naked_core::types::FS_CACHE_MISS_COUNT.load(Ordering::Relaxed),
            fs_cache_stale_bypass: naked_core::types::FS_CACHE_STALE_BYPASS_COUNT
                .load(Ordering::Relaxed),
            fs_cache_invalidate: naked_core::types::FS_CACHE_INVALIDATE_COUNT
                .load(Ordering::Relaxed),
            fs_cache_too_large: naked_core::types::FS_CACHE_TOO_LARGE_COUNT.load(Ordering::Relaxed),
            persistent_bash_ok: naked_core::types::PERSISTENT_BASH_OK_COUNT.load(Ordering::Relaxed),
            persistent_bash_timeout: naked_core::types::PERSISTENT_BASH_TIMEOUT_COUNT
                .load(Ordering::Relaxed),
            persistent_bash_killed: naked_core::types::PERSISTENT_BASH_KILLED_COUNT
                .load(Ordering::Relaxed),
            persistent_bash_restart: naked_core::types::PERSISTENT_BASH_RESTART_COUNT
                .load(Ordering::Relaxed),
            persistent_bash_error: naked_core::types::PERSISTENT_BASH_ERROR_COUNT
                .load(Ordering::Relaxed),
            persistent_bash_disabled: naked_core::types::PERSISTENT_BASH_DISABLED_COUNT
                .load(Ordering::Relaxed),
            persistent_bash_busy: naked_core::types::PERSISTENT_BASH_BUSY_COUNT
                .load(Ordering::Relaxed),
            fff_picker_created: naked_core::types::FFF_PICKER_REGISTRY_CREATED_COUNT
                .load(Ordering::Relaxed),
            fff_picker_reused: naked_core::types::FFF_PICKER_REGISTRY_REUSED_COUNT
                .load(Ordering::Relaxed),
            fff_picker_cap_fallback: naked_core::types::FFF_PICKER_REGISTRY_CAP_FALLBACK_COUNT
                .load(Ordering::Relaxed),
            fff_grep_fast_index: naked_core::types::FFF_GREP_FAST_INDEX_COUNT
                .load(Ordering::Relaxed),
            fff_grep_fallback: naked_core::types::FFF_GREP_FALLBACK_COUNT.load(Ordering::Relaxed),
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
            memory_candidates: naked_core::types::MEMORY_CANDIDATES_COUNT.load(Ordering::Relaxed),
            memory_promoted_repeat_days: naked_core::types::MEMORY_PROMOTED_REPEAT_DAYS_COUNT
                .load(Ordering::Relaxed),
            memory_promoted_reinforce: naked_core::types::MEMORY_PROMOTED_REINFORCE_COUNT
                .load(Ordering::Relaxed),
            memory_promoted_reobs: naked_core::types::MEMORY_PROMOTED_REOBS_COUNT
                .load(Ordering::Relaxed),
            memory_injection_dropped_global:
                naked_core::types::MEMORY_INJECTION_DROPPED_GLOBAL_COUNT.load(Ordering::Relaxed),
            memory_injection_dropped_project:
                naked_core::types::MEMORY_INJECTION_DROPPED_PROJECT_COUNT.load(Ordering::Relaxed),
            memory_injection_dropped_user: naked_core::types::MEMORY_INJECTION_DROPPED_USER_COUNT
                .load(Ordering::Relaxed),
            memory_injection_failed: naked_core::types::MEMORY_INJECTION_FAILED_COUNT
                .load(Ordering::Relaxed),
            memory_digest_staleness: memory_digest_staleness_snapshot(),
            concurrent_same_key_turns: self.concurrent_same_key_turns,
            text_coalesced: self.text_coalesced,
            session_busy_ack: self.session_busy_ack,
            memory_pollution,
            active_run_max_silent_seconds: crate::shared::RUN_REGISTRY
                .active_run_max_silent_seconds(),
            config_loaded_hash: crate::shared::CONFIG_LOADED_HASH
                .get()
                .map(|hash| hash.hash_hex.clone()),
            model_health_body,
        };
        render_prometheus_from(&inputs)
    }
}

fn memory_digest_staleness_snapshot() -> DigestStalenessSnapshot {
    let workspace = MEMORY_METRICS_WORKSPACE
        .get()
        .cloned()
        .or_else(|| std::env::current_dir().ok())
        .unwrap_or_else(|| PathBuf::from("."));
    naked_core::memory::daily::digest_staleness_snapshot(&workspace)
}

fn render_prometheus_from(inputs: &PrometheusInputs) -> String {
    let latency_extra_metrics = render_latency_extra_metrics(&inputs.latency);
    let final_answer_metrics = render_final_answer_metrics(&inputs.media);
    let config_loaded_hash_metric = inputs
        .config_loaded_hash
        .as_deref()
        .map(|hash| {
            format!(
                "# HELP naked_tg_config_loaded_hash (B42/RC-14) Loaded config file hash captured immediately after Config::load.\n\
                 # TYPE naked_tg_config_loaded_hash gauge\n\
                 naked_tg_config_loaded_hash{{hash=\"{hash}\"}} 1\n"
            )
        })
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
             {config_loaded_hash_metric}\
             # HELP naked_core_stale_edit_reject_total Stale edit precondition failures rejected before writing.\n\
             # TYPE naked_core_stale_edit_reject_total counter\n\
             naked_core_stale_edit_reject_total {stale_edit_reject}\n\
             # HELP naked_core_git_history_guard_block_total Git history-destructive bash commands blocked (B85).\n\
             # TYPE naked_core_git_history_guard_block_total counter\n\
             naked_core_git_history_guard_block_total {git_history_guard_block}\n\
             # HELP naked_core_hashline_edit_total Hashline edit outcomes.\n\
             # TYPE naked_core_hashline_edit_total counter\n\
             naked_core_hashline_edit_total{{outcome=\"applied\"}} {hashline_edit_applied}\n\
             naked_core_hashline_edit_total{{outcome=\"stale_anchor\"}} {hashline_edit_stale_anchor}\n\
             naked_core_hashline_edit_total{{outcome=\"overlap\"}} {hashline_edit_overlap}\n\
             naked_core_hashline_edit_total{{outcome=\"out_of_bounds\"}} {hashline_edit_out_of_bounds}\n\
             naked_core_hashline_edit_total{{outcome=\"disabled\"}} {hashline_edit_disabled}\n\
             # HELP naked_core_fs_cache_total File-system content cache outcomes.\n\
             # TYPE naked_core_fs_cache_total counter\n\
             naked_core_fs_cache_total{{outcome=\"hit\"}} {fs_cache_hit}\n\
             naked_core_fs_cache_total{{outcome=\"miss\"}} {fs_cache_miss}\n\
             naked_core_fs_cache_total{{outcome=\"stale_bypass\"}} {fs_cache_stale_bypass}\n\
             naked_core_fs_cache_total{{outcome=\"invalidate\"}} {fs_cache_invalidate}\n\
             naked_core_fs_cache_total{{outcome=\"too_large\"}} {fs_cache_too_large}\n\
             # HELP naked_core_persistent_bash_total Persistent bash execution outcomes.\n\
             # TYPE naked_core_persistent_bash_total counter\n\
             naked_core_persistent_bash_total{{outcome=\"ok\"}} {persistent_bash_ok}\n\
             naked_core_persistent_bash_total{{outcome=\"timeout\"}} {persistent_bash_timeout}\n\
             naked_core_persistent_bash_total{{outcome=\"killed\"}} {persistent_bash_killed}\n\
             naked_core_persistent_bash_total{{outcome=\"restart\"}} {persistent_bash_restart}\n\
             naked_core_persistent_bash_total{{outcome=\"error\"}} {persistent_bash_error}\n\
             naked_core_persistent_bash_total{{outcome=\"disabled\"}} {persistent_bash_disabled}\n\
             naked_core_persistent_bash_total{{outcome=\"busy\"}} {persistent_bash_busy}\n\
             # HELP naked_core_fff_picker_registry_total fff fast-index picker registry outcomes.\n\
             # TYPE naked_core_fff_picker_registry_total counter\n\
             naked_core_fff_picker_registry_total{{outcome=\"created\"}} {fff_picker_created}\n\
             naked_core_fff_picker_registry_total{{outcome=\"reused\"}} {fff_picker_reused}\n\
             naked_core_fff_picker_registry_total{{outcome=\"cap_fallback\"}} {fff_picker_cap_fallback}\n\
             # HELP naked_core_fff_grep_total fff grep requests by backend.\n\
             # TYPE naked_core_fff_grep_total counter\n\
             naked_core_fff_grep_total{{backend=\"fast_index\"}} {fff_grep_fast_index}\n\
             naked_core_fff_grep_total{{backend=\"fallback\"}} {fff_grep_fallback}\n\
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
             # HELP naked_core_memory_candidates_total Memory digest candidates scored by the daily promotion pass.\n\
             # TYPE naked_core_memory_candidates_total counter\n\
             naked_core_memory_candidates_total {memory_candidates}\n\
             # HELP naked_core_memory_promoted_total Memory digest actual promotions by independent eligibility road; road series may overlap and sum may exceed promoted outcomes.\n\
             # TYPE naked_core_memory_promoted_total counter\n\
             naked_core_memory_promoted_total{{road=\"repeat_days\"}} {memory_promoted_repeat_days}\n\
             naked_core_memory_promoted_total{{road=\"reinforce\"}} {memory_promoted_reinforce}\n\
             naked_core_memory_promoted_total{{road=\"reobs\"}} {memory_promoted_reobs}\n\
             # HELP naked_core_memory_injection_dropped_total Memory entries dropped from prompt injection because MAX_INJECTION_CHARS was exhausted.\n\
             # TYPE naked_core_memory_injection_dropped_total counter\n\
             naked_core_memory_injection_dropped_total{{scope=\"global\"}} {memory_injection_dropped_global}\n\
             naked_core_memory_injection_dropped_total{{scope=\"project\"}} {memory_injection_dropped_project}\n\
             naked_core_memory_injection_dropped_total{{scope=\"user\"}} {memory_injection_dropped_user}\n\
             # HELP naked_core_memory_injection_failed_total Memory prompt injection failures degraded to no injected memory so turns can continue.\n\
             # TYPE naked_core_memory_injection_failed_total counter\n\
             naked_core_memory_injection_failed_total {memory_injection_failed}\n\
             # HELP naked_core_memory_days_since_last_digest Days since the persisted .last_digest marker by memory scope; 9999 means missing/invalid marker.\n\
             # TYPE naked_core_memory_days_since_last_digest gauge\n\
             naked_core_memory_days_since_last_digest{{scope=\"global\"}} {memory_days_since_last_digest_global}\n\
             naked_core_memory_days_since_last_digest{{scope=\"project\"}} {memory_days_since_last_digest_project}\n\
             naked_core_memory_days_since_last_digest{{scope=\"user\"}} {memory_days_since_last_digest_user}\n\
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
             {final_answer_metrics}\
             # HELP naked_tg_config_describer_missing (B05) Set to 1 when the default model is not vision-capable AND no tg_media.vision describer fallback is configured.\n\
             # TYPE naked_tg_config_describer_missing gauge\n\
             naked_tg_config_describer_missing {config_describer_missing}\n\
             # HELP naked_tg_active_run_max_silent_seconds Maximum age of non-heartbeat progress silence across active runs.\n\
             # TYPE naked_tg_active_run_max_silent_seconds gauge\n\
             naked_tg_active_run_max_silent_seconds {active_run_max_silent_seconds}\n\
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
             # HELP naked_core_agent_turn_duration_total Agent turn wall duration (ms)\n\
             # TYPE naked_core_agent_turn_duration_total counter\n\
             naked_core_agent_turn_duration_total{{bucket=\"under_1s\"}} {turn_duration_under_1s}\n\
             naked_core_agent_turn_duration_total{{bucket=\"1s_to_10s\"}} {turn_duration_1s_to_10s}\n\
             naked_core_agent_turn_duration_total{{bucket=\"10s_to_60s\"}} {turn_duration_10s_to_60s}\n\
             naked_core_agent_turn_duration_total{{bucket=\"over_60s\"}} {turn_duration_over_60s}\n\
             # HELP naked_core_agent_turn_duration_sum_ms Agent turn wall duration (ms)\n\
             # TYPE naked_core_agent_turn_duration_sum_ms counter\n\
             naked_core_agent_turn_duration_sum_ms {turn_duration_sum_ms}\n\
             # HELP naked_core_agent_turn_duration_count Agent turn wall duration observations\n\
             # TYPE naked_core_agent_turn_duration_count counter\n\
             naked_core_agent_turn_duration_count {turn_duration_count}\n\
             # HELP naked_core_agent_ttft_total Time to first text token (ms)\n\
             # TYPE naked_core_agent_ttft_total counter\n\
             naked_core_agent_ttft_total{{bucket=\"under_500ms\"}} {ttft_under_500ms}\n\
             naked_core_agent_ttft_total{{bucket=\"500ms_to_2s\"}} {ttft_500ms_to_2s}\n\
             naked_core_agent_ttft_total{{bucket=\"2s_to_10s\"}} {ttft_2s_to_10s}\n\
             naked_core_agent_ttft_total{{bucket=\"over_10s\"}} {ttft_over_10s}\n\
             # HELP naked_core_agent_ttft_sum_ms Time to first text token (ms)\n\
             # TYPE naked_core_agent_ttft_sum_ms counter\n\
             naked_core_agent_ttft_sum_ms {ttft_sum_ms}\n\
             # HELP naked_core_agent_ttft_count Time to first text token observations\n\
             # TYPE naked_core_agent_ttft_count counter\n\
             naked_core_agent_ttft_count {ttft_count}\n\
             # HELP naked_core_provider_stream_open_total Provider connect to first chunk (ms, successful streams only)\n\
             # TYPE naked_core_provider_stream_open_total counter\n\
             naked_core_provider_stream_open_total{{bucket=\"under_500ms\"}} {provider_stream_open_under_500ms}\n\
             naked_core_provider_stream_open_total{{bucket=\"500ms_to_2s\"}} {provider_stream_open_500ms_to_2s}\n\
             naked_core_provider_stream_open_total{{bucket=\"2s_to_10s\"}} {provider_stream_open_2s_to_10s}\n\
             naked_core_provider_stream_open_total{{bucket=\"over_10s\"}} {provider_stream_open_over_10s}\n\
             # HELP naked_core_provider_stream_open_sum_ms Provider connect to first chunk (ms, successful streams only)\n\
             # TYPE naked_core_provider_stream_open_sum_ms counter\n\
             naked_core_provider_stream_open_sum_ms {provider_stream_open_sum_ms}\n\
             # HELP naked_core_provider_stream_open_count Provider stream-open observations (successful streams only)\n\
             # TYPE naked_core_provider_stream_open_count counter\n\
             naked_core_provider_stream_open_count {provider_stream_open_count}\n\
             {latency_extra_metrics}\
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
        config_loaded_hash_metric = config_loaded_hash_metric,
        stale_edit_reject = inputs.stale_edit_reject,
        git_history_guard_block = inputs.git_history_guard_block,
        hashline_edit_applied = inputs.hashline_edit_applied,
        hashline_edit_stale_anchor = inputs.hashline_edit_stale_anchor,
        hashline_edit_overlap = inputs.hashline_edit_overlap,
        hashline_edit_out_of_bounds = inputs.hashline_edit_out_of_bounds,
        hashline_edit_disabled = inputs.hashline_edit_disabled,
        fs_cache_hit = inputs.fs_cache_hit,
        fs_cache_miss = inputs.fs_cache_miss,
        fs_cache_stale_bypass = inputs.fs_cache_stale_bypass,
        fs_cache_invalidate = inputs.fs_cache_invalidate,
        fs_cache_too_large = inputs.fs_cache_too_large,
        persistent_bash_ok = inputs.persistent_bash_ok,
        persistent_bash_timeout = inputs.persistent_bash_timeout,
        persistent_bash_killed = inputs.persistent_bash_killed,
        persistent_bash_restart = inputs.persistent_bash_restart,
        persistent_bash_error = inputs.persistent_bash_error,
        persistent_bash_disabled = inputs.persistent_bash_disabled,
        persistent_bash_busy = inputs.persistent_bash_busy,
        fff_picker_created = inputs.fff_picker_created,
        fff_picker_reused = inputs.fff_picker_reused,
        fff_picker_cap_fallback = inputs.fff_picker_cap_fallback,
        fff_grep_fast_index = inputs.fff_grep_fast_index,
        fff_grep_fallback = inputs.fff_grep_fallback,
        ip_hallucin = inputs.ip_hallucin,
        research_store_corrupt_rows = inputs.research_store_corrupt_rows,
        research_store_files_healed = inputs.research_store_files_healed,
        research_store_heal_failed = inputs.research_store_heal_failed,
        research_store_file_lock_wait = inputs.research_store_file_lock_wait,
        memory_candidates = inputs.memory_candidates,
        memory_promoted_repeat_days = inputs.memory_promoted_repeat_days,
        memory_promoted_reinforce = inputs.memory_promoted_reinforce,
        memory_promoted_reobs = inputs.memory_promoted_reobs,
        memory_injection_dropped_global = inputs.memory_injection_dropped_global,
        memory_injection_dropped_project = inputs.memory_injection_dropped_project,
        memory_injection_dropped_user = inputs.memory_injection_dropped_user,
        memory_injection_failed = inputs.memory_injection_failed,
        memory_days_since_last_digest_global = inputs.memory_digest_staleness.global,
        memory_days_since_last_digest_project = inputs.memory_digest_staleness.project,
        memory_days_since_last_digest_user = inputs.memory_digest_staleness.user,
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
        final_answer_metrics = final_answer_metrics,
        config_describer_missing = inputs.media.config_describer_missing,
        active_run_max_silent_seconds = inputs.active_run_max_silent_seconds,
        cancel_u3s = inputs.media.research_cancel_propagation_under_3s,
        cancel_3s_30s = inputs.media.research_cancel_propagation_3s_to_30s,
        cancel_o30s = inputs.media.research_cancel_propagation_over_30s,
        cancel_sum_ms = inputs.media.research_cancel_propagation_sum_ms,
        cancel_count = inputs.media.research_cancel_propagation_count,
        turn_duration_under_1s = inputs.latency.turn_under_1s,
        turn_duration_1s_to_10s = inputs.latency.turn_1s_to_10s,
        turn_duration_10s_to_60s = inputs.latency.turn_10s_to_60s,
        turn_duration_over_60s = inputs.latency.turn_over_60s,
        turn_duration_sum_ms = inputs.latency.turn_sum_ms,
        turn_duration_count = inputs.latency.turn_count,
        ttft_under_500ms = inputs.latency.ttft_under_500ms,
        ttft_500ms_to_2s = inputs.latency.ttft_500ms_to_2s,
        ttft_2s_to_10s = inputs.latency.ttft_2s_to_10s,
        ttft_over_10s = inputs.latency.ttft_over_10s,
        ttft_sum_ms = inputs.latency.ttft_sum_ms,
        ttft_count = inputs.latency.ttft_count,
        provider_stream_open_under_500ms = inputs.latency.provider_under_500ms,
        provider_stream_open_500ms_to_2s = inputs.latency.provider_500ms_to_2s,
        provider_stream_open_2s_to_10s = inputs.latency.provider_2s_to_10s,
        provider_stream_open_over_10s = inputs.latency.provider_over_10s,
        provider_stream_open_sum_ms = inputs.latency.provider_sum_ms,
        provider_stream_open_count = inputs.latency.provider_count,
        latency_extra_metrics = latency_extra_metrics,
        model_health_body = inputs.model_health_body,
    )
}

fn render_final_answer_metrics(media: &MediaRoutingSnapshot) -> String {
    let mut out = String::new();
    out.push_str("# HELP naked_tg_final_answer_utf16 Final answer visible text length in Telegram-enforced UTF-16 code units.\n");
    out.push_str("# TYPE naked_tg_final_answer_utf16 histogram\n");
    let mut cumulative = 0u64;
    for (label, count) in FINAL_ANSWER_UTF16_BUCKET_LABELS
        .iter()
        .zip(media.final_answer_utf16.buckets)
    {
        cumulative = cumulative.saturating_add(count);
        out.push_str(&format!(
            "naked_tg_final_answer_utf16_bucket{{le=\"{}\"}} {}\n",
            label, cumulative
        ));
    }
    out.push_str(&format!(
        "naked_tg_final_answer_utf16_sum {}\n\
         naked_tg_final_answer_utf16_count {}\n",
        media.final_answer_utf16.sum, media.final_answer_utf16.count
    ));
    out.push_str("# HELP naked_tg_final_answer_delivery_total Final answer delivery outcomes.\n");
    out.push_str("# TYPE naked_tg_final_answer_delivery_total counter\n");
    out.push_str(&format!(
        "naked_tg_final_answer_delivery_total{{outcome=\"ok\"}} {}\n\
         naked_tg_final_answer_delivery_total{{outcome=\"partial\"}} {}\n\
         naked_tg_final_answer_delivery_total{{outcome=\"failed\"}} {}\n",
        media.final_answer_delivery_ok,
        media.final_answer_delivery_partial,
        media.final_answer_delivery_failed
    ));
    out.push_str("# HELP naked_tg_final_answer_truncated_total Final answers rendered with inline truncation.\n");
    out.push_str("# TYPE naked_tg_final_answer_truncated_total counter\n");
    out.push_str(&format!(
        "naked_tg_final_answer_truncated_total {}\n",
        media.final_answer_truncated
    ));
    out.push_str("# HELP naked_tg_final_answer_attachment_send_failed_total Final-answer HTML attachment send failures.\n");
    out.push_str("# TYPE naked_tg_final_answer_attachment_send_failed_total counter\n");
    out.push_str(&format!(
        "naked_tg_final_answer_attachment_send_failed_total {}\n",
        media.final_answer_attachment_send_failed
    ));
    out
}

fn render_latency_extra_metrics(latency: &LatencySnapshot) -> String {
    let mut out = String::new();
    out.push_str("# HELP naked_core_tool_duration_total Tool execution wall duration (ms), by bounded tool class.\n");
    out.push_str("# TYPE naked_core_tool_duration_total counter\n");
    for class in ToolClass::ALL {
        push_labeled_duration_buckets(
            &mut out,
            "naked_core_tool_duration_total",
            "tool",
            class.label(),
            tool_snapshot(latency, class),
        );
    }
    out.push_str("# HELP naked_core_tool_duration_sum_ms Tool execution wall duration (ms), by bounded tool class.\n");
    out.push_str("# TYPE naked_core_tool_duration_sum_ms counter\n");
    for class in ToolClass::ALL {
        let hist = tool_snapshot(latency, class);
        out.push_str(&format!(
            "naked_core_tool_duration_sum_ms{{tool=\"{}\"}} {}\n",
            class.label(),
            hist.sum_ms
        ));
    }
    out.push_str("# HELP naked_core_tool_duration_count Tool execution wall duration (ms), by bounded tool class. observations\n");
    out.push_str("# TYPE naked_core_tool_duration_count counter\n");
    for class in ToolClass::ALL {
        let hist = tool_snapshot(latency, class);
        out.push_str(&format!(
            "naked_core_tool_duration_count{{tool=\"{}\"}} {}\n",
            class.label(),
            hist.count
        ));
    }

    push_single_duration_histogram(
        &mut out,
        "naked_core_fff_cold_build",
        "fff cold fast-index picker build duration (ms).",
        latency.fff_cold_build,
    );
    push_single_duration_histogram(
        &mut out,
        "naked_core_parent_fsync",
        "Parent directory fsync duration for atomic writes (ms).",
        latency.parent_fsync,
    );
    out
}

fn push_labeled_duration_buckets(
    out: &mut String,
    metric: &str,
    label_name: &str,
    label_value: &str,
    hist: DurationHistogramSnapshot,
) {
    out.push_str(&format!(
        "{metric}{{{label_name}=\"{label_value}\",bucket=\"under_10ms\"}} {}\n\
         {metric}{{{label_name}=\"{label_value}\",bucket=\"10ms_to_100ms\"}} {}\n\
         {metric}{{{label_name}=\"{label_value}\",bucket=\"100ms_to_1s\"}} {}\n\
         {metric}{{{label_name}=\"{label_value}\",bucket=\"over_1s\"}} {}\n",
        hist.under_10ms, hist.ms_10_to_100, hist.ms_100_to_1s, hist.over_1s
    ));
}

fn push_single_duration_histogram(
    out: &mut String,
    metric_base: &str,
    help: &str,
    hist: DurationHistogramSnapshot,
) {
    out.push_str(&format!("# HELP {metric_base}_total {help}\n"));
    out.push_str(&format!("# TYPE {metric_base}_total counter\n"));
    out.push_str(&format!(
        "{metric_base}_total{{bucket=\"under_10ms\"}} {}\n\
         {metric_base}_total{{bucket=\"10ms_to_100ms\"}} {}\n\
         {metric_base}_total{{bucket=\"100ms_to_1s\"}} {}\n\
         {metric_base}_total{{bucket=\"over_1s\"}} {}\n",
        hist.under_10ms, hist.ms_10_to_100, hist.ms_100_to_1s, hist.over_1s
    ));
    out.push_str(&format!("# HELP {metric_base}_sum_ms {help}\n"));
    out.push_str(&format!("# TYPE {metric_base}_sum_ms counter\n"));
    out.push_str(&format!("{metric_base}_sum_ms {}\n", hist.sum_ms));
    out.push_str(&format!("# HELP {metric_base}_count {help} observations\n"));
    out.push_str(&format!("# TYPE {metric_base}_count counter\n"));
    out.push_str(&format!("{metric_base}_count {}\n", hist.count));
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

    fn metric_value(body: &str, metric_line_prefix: &str) -> u64 {
        let line = body
            .lines()
            .find(|line| line.starts_with(metric_line_prefix))
            .unwrap_or_else(|| panic!("missing metric line `{metric_line_prefix}` in:\n{body}"));
        line.split_whitespace()
            .nth(1)
            .unwrap_or_else(|| panic!("missing value for metric line `{line}`"))
            .parse::<u64>()
            .unwrap_or_else(|err| panic!("invalid value for metric line `{line}`: {err}"))
    }

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
                final_answer_utf16: FinalAnswerUtf16HistogramSnapshot {
                    buckets: [1, 2, 3, 4, 5, 6, 7],
                    sum: 1234,
                    count: 28,
                },
                final_answer_delivery_ok: 24,
                final_answer_delivery_partial: 25,
                final_answer_delivery_failed: 26,
                final_answer_truncated: 27,
                final_answer_attachment_send_failed: 28,
                config_describer_missing: 1,
                research_cancel_propagation_under_3s: 19,
                research_cancel_propagation_3s_to_30s: 20,
                research_cancel_propagation_over_30s: 21,
                research_cancel_propagation_sum_ms: 22,
                research_cancel_propagation_count: 23,
            },
            latency: LatencySnapshot {
                turn_under_1s: 100,
                turn_1s_to_10s: 101,
                turn_10s_to_60s: 102,
                turn_over_60s: 103,
                turn_sum_ms: 104,
                turn_count: 105,
                ttft_under_500ms: 106,
                ttft_500ms_to_2s: 107,
                ttft_2s_to_10s: 108,
                ttft_over_10s: 109,
                ttft_sum_ms: 110,
                ttft_count: 111,
                provider_under_500ms: 112,
                provider_500ms_to_2s: 113,
                provider_2s_to_10s: 114,
                provider_over_10s: 115,
                provider_sum_ms: 116,
                provider_count: 117,
                tool_grep: DurationHistogramSnapshot {
                    under_10ms: 118,
                    ms_10_to_100: 119,
                    ms_100_to_1s: 120,
                    over_1s: 121,
                    sum_ms: 122,
                    count: 123,
                },
                tool_read: DurationHistogramSnapshot {
                    under_10ms: 124,
                    ms_10_to_100: 125,
                    ms_100_to_1s: 126,
                    over_1s: 127,
                    sum_ms: 128,
                    count: 129,
                },
                tool_edit: DurationHistogramSnapshot {
                    under_10ms: 130,
                    ms_10_to_100: 131,
                    ms_100_to_1s: 132,
                    over_1s: 133,
                    sum_ms: 134,
                    count: 135,
                },
                tool_write: DurationHistogramSnapshot {
                    under_10ms: 136,
                    ms_10_to_100: 137,
                    ms_100_to_1s: 138,
                    over_1s: 139,
                    sum_ms: 140,
                    count: 141,
                },
                tool_bash: DurationHistogramSnapshot {
                    under_10ms: 142,
                    ms_10_to_100: 143,
                    ms_100_to_1s: 144,
                    over_1s: 145,
                    sum_ms: 146,
                    count: 147,
                },
                tool_apply_patch: DurationHistogramSnapshot {
                    under_10ms: 148,
                    ms_10_to_100: 149,
                    ms_100_to_1s: 150,
                    over_1s: 151,
                    sum_ms: 152,
                    count: 153,
                },
                tool_other: DurationHistogramSnapshot {
                    under_10ms: 154,
                    ms_10_to_100: 155,
                    ms_100_to_1s: 156,
                    over_1s: 157,
                    sum_ms: 158,
                    count: 159,
                },
                fff_cold_build: DurationHistogramSnapshot {
                    under_10ms: 160,
                    ms_10_to_100: 161,
                    ms_100_to_1s: 162,
                    over_1s: 163,
                    sum_ms: 164,
                    count: 165,
                },
                parent_fsync: DurationHistogramSnapshot {
                    under_10ms: 166,
                    ms_10_to_100: 167,
                    ms_100_to_1s: 168,
                    over_1s: 169,
                    sum_ms: 170,
                    count: 171,
                },
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
            stale_edit_reject: 406,
            git_history_guard_block: 408,
            hashline_edit_applied: 407,
            hashline_edit_stale_anchor: 408,
            hashline_edit_overlap: 409,
            hashline_edit_out_of_bounds: 410,
            hashline_edit_disabled: 411,
            fs_cache_hit: 412,
            fs_cache_miss: 413,
            fs_cache_stale_bypass: 414,
            fs_cache_invalidate: 415,
            fs_cache_too_large: 416,
            persistent_bash_ok: 417,
            persistent_bash_timeout: 418,
            persistent_bash_killed: 419,
            persistent_bash_restart: 420,
            persistent_bash_error: 421,
            persistent_bash_disabled: 422,
            persistent_bash_busy: 423,
            fff_picker_created: 401,
            fff_picker_reused: 402,
            fff_picker_cap_fallback: 403,
            fff_grep_fast_index: 404,
            fff_grep_fallback: 405,
            ip_hallucin: 41,
            research_store_corrupt_rows: 42,
            research_store_files_healed: 43,
            research_store_heal_failed: 44,
            research_store_file_lock_wait: 45,
            memory_candidates: 424,
            memory_promoted_repeat_days: 425,
            memory_promoted_reinforce: 426,
            memory_promoted_reobs: 427,
            memory_injection_dropped_global: 428,
            memory_injection_dropped_project: 429,
            memory_injection_dropped_user: 430,
            memory_injection_failed: 431,
            memory_digest_staleness: DigestStalenessSnapshot {
                global: 2,
                project: 3,
                user: 4,
            },
            concurrent_same_key_turns: 46,
            text_coalesced: 47,
            session_busy_ack: 48,
            memory_pollution: 0,
            active_run_max_silent_seconds: 49,
            config_loaded_hash: None,
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
            "# HELP naked_core_stale_edit_reject_total Stale edit precondition failures rejected before writing.\n",
            "# TYPE naked_core_stale_edit_reject_total counter\n",
            "naked_core_stale_edit_reject_total 406\n",
            "# HELP naked_core_git_history_guard_block_total Git history-destructive bash commands blocked (B85).\n",
            "# TYPE naked_core_git_history_guard_block_total counter\n",
            "naked_core_git_history_guard_block_total 408\n",
            "# HELP naked_core_hashline_edit_total Hashline edit outcomes.\n",
            "# TYPE naked_core_hashline_edit_total counter\n",
            "naked_core_hashline_edit_total{outcome=\"applied\"} 407\n",
            "naked_core_hashline_edit_total{outcome=\"stale_anchor\"} 408\n",
            "naked_core_hashline_edit_total{outcome=\"overlap\"} 409\n",
            "naked_core_hashline_edit_total{outcome=\"out_of_bounds\"} 410\n",
            "naked_core_hashline_edit_total{outcome=\"disabled\"} 411\n",
            "# HELP naked_core_fs_cache_total File-system content cache outcomes.\n",
            "# TYPE naked_core_fs_cache_total counter\n",
            "naked_core_fs_cache_total{outcome=\"hit\"} 412\n",
            "naked_core_fs_cache_total{outcome=\"miss\"} 413\n",
            "naked_core_fs_cache_total{outcome=\"stale_bypass\"} 414\n",
            "naked_core_fs_cache_total{outcome=\"invalidate\"} 415\n",
            "naked_core_fs_cache_total{outcome=\"too_large\"} 416\n",
            "# HELP naked_core_persistent_bash_total Persistent bash execution outcomes.\n",
            "# TYPE naked_core_persistent_bash_total counter\n",
            "naked_core_persistent_bash_total{outcome=\"ok\"} 417\n",
            "naked_core_persistent_bash_total{outcome=\"timeout\"} 418\n",
            "naked_core_persistent_bash_total{outcome=\"killed\"} 419\n",
            "naked_core_persistent_bash_total{outcome=\"restart\"} 420\n",
            "naked_core_persistent_bash_total{outcome=\"error\"} 421\n",
            "naked_core_persistent_bash_total{outcome=\"disabled\"} 422\n",
            "naked_core_persistent_bash_total{outcome=\"busy\"} 423\n",
            "# HELP naked_core_fff_picker_registry_total fff fast-index picker registry outcomes.\n",
            "# TYPE naked_core_fff_picker_registry_total counter\n",
            "naked_core_fff_picker_registry_total{outcome=\"created\"} 401\n",
            "naked_core_fff_picker_registry_total{outcome=\"reused\"} 402\n",
            "naked_core_fff_picker_registry_total{outcome=\"cap_fallback\"} 403\n",
            "# HELP naked_core_fff_grep_total fff grep requests by backend.\n",
            "# TYPE naked_core_fff_grep_total counter\n",
            "naked_core_fff_grep_total{backend=\"fast_index\"} 404\n",
            "naked_core_fff_grep_total{backend=\"fallback\"} 405\n",
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
            "# HELP naked_core_memory_candidates_total Memory digest candidates scored by the daily promotion pass.\n",
            "# TYPE naked_core_memory_candidates_total counter\n",
            "naked_core_memory_candidates_total 424\n",
            "# HELP naked_core_memory_promoted_total Memory digest actual promotions by independent eligibility road; road series may overlap and sum may exceed promoted outcomes.\n",
            "# TYPE naked_core_memory_promoted_total counter\n",
            "naked_core_memory_promoted_total{road=\"repeat_days\"} 425\n",
            "naked_core_memory_promoted_total{road=\"reinforce\"} 426\n",
            "naked_core_memory_promoted_total{road=\"reobs\"} 427\n",
            "# HELP naked_core_memory_injection_dropped_total Memory entries dropped from prompt injection because MAX_INJECTION_CHARS was exhausted.\n",
            "# TYPE naked_core_memory_injection_dropped_total counter\n",
            "naked_core_memory_injection_dropped_total{scope=\"global\"} 428\n",
            "naked_core_memory_injection_dropped_total{scope=\"project\"} 429\n",
            "naked_core_memory_injection_dropped_total{scope=\"user\"} 430\n",
            "# HELP naked_core_memory_injection_failed_total Memory prompt injection failures degraded to no injected memory so turns can continue.\n",
            "# TYPE naked_core_memory_injection_failed_total counter\n",
            "naked_core_memory_injection_failed_total 431\n",
            "# HELP naked_core_memory_days_since_last_digest Days since the persisted .last_digest marker by memory scope; 9999 means missing/invalid marker.\n",
            "# TYPE naked_core_memory_days_since_last_digest gauge\n",
            "naked_core_memory_days_since_last_digest{scope=\"global\"} 2\n",
            "naked_core_memory_days_since_last_digest{scope=\"project\"} 3\n",
            "naked_core_memory_days_since_last_digest{scope=\"user\"} 4\n",
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
            "# HELP naked_tg_final_answer_utf16 Final answer visible text length in Telegram-enforced UTF-16 code units.\n",
            "# TYPE naked_tg_final_answer_utf16 histogram\n",
            "naked_tg_final_answer_utf16_bucket{le=\"1024\"} 1\n",
            "naked_tg_final_answer_utf16_bucket{le=\"2048\"} 3\n",
            "naked_tg_final_answer_utf16_bucket{le=\"4096\"} 6\n",
            "naked_tg_final_answer_utf16_bucket{le=\"8192\"} 10\n",
            "naked_tg_final_answer_utf16_bucket{le=\"16384\"} 15\n",
            "naked_tg_final_answer_utf16_bucket{le=\"32768\"} 21\n",
            "naked_tg_final_answer_utf16_bucket{le=\"+Inf\"} 28\n",
            "naked_tg_final_answer_utf16_sum 1234\n",
            "naked_tg_final_answer_utf16_count 28\n",
            "# HELP naked_tg_final_answer_delivery_total Final answer delivery outcomes.\n",
            "# TYPE naked_tg_final_answer_delivery_total counter\n",
            "naked_tg_final_answer_delivery_total{outcome=\"ok\"} 24\n",
            "naked_tg_final_answer_delivery_total{outcome=\"partial\"} 25\n",
            "naked_tg_final_answer_delivery_total{outcome=\"failed\"} 26\n",
            "# HELP naked_tg_final_answer_truncated_total Final answers rendered with inline truncation.\n",
            "# TYPE naked_tg_final_answer_truncated_total counter\n",
            "naked_tg_final_answer_truncated_total 27\n",
            "# HELP naked_tg_final_answer_attachment_send_failed_total Final-answer HTML attachment send failures.\n",
            "# TYPE naked_tg_final_answer_attachment_send_failed_total counter\n",
            "naked_tg_final_answer_attachment_send_failed_total 28\n",
            "# HELP naked_tg_config_describer_missing (B05) Set to 1 when the default model is not vision-capable AND no tg_media.vision describer fallback is configured.\n",
            "# TYPE naked_tg_config_describer_missing gauge\n",
            "naked_tg_config_describer_missing 1\n",
            "# HELP naked_tg_active_run_max_silent_seconds Maximum age of non-heartbeat progress silence across active runs.\n",
            "# TYPE naked_tg_active_run_max_silent_seconds gauge\n",
            "naked_tg_active_run_max_silent_seconds 49\n",
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
            "# HELP naked_core_agent_turn_duration_total Agent turn wall duration (ms)\n",
            "# TYPE naked_core_agent_turn_duration_total counter\n",
            "naked_core_agent_turn_duration_total{bucket=\"under_1s\"} 100\n",
            "naked_core_agent_turn_duration_total{bucket=\"1s_to_10s\"} 101\n",
            "naked_core_agent_turn_duration_total{bucket=\"10s_to_60s\"} 102\n",
            "naked_core_agent_turn_duration_total{bucket=\"over_60s\"} 103\n",
            "# HELP naked_core_agent_turn_duration_sum_ms Agent turn wall duration (ms)\n",
            "# TYPE naked_core_agent_turn_duration_sum_ms counter\n",
            "naked_core_agent_turn_duration_sum_ms 104\n",
            "# HELP naked_core_agent_turn_duration_count Agent turn wall duration observations\n",
            "# TYPE naked_core_agent_turn_duration_count counter\n",
            "naked_core_agent_turn_duration_count 105\n",
            "# HELP naked_core_agent_ttft_total Time to first text token (ms)\n",
            "# TYPE naked_core_agent_ttft_total counter\n",
            "naked_core_agent_ttft_total{bucket=\"under_500ms\"} 106\n",
            "naked_core_agent_ttft_total{bucket=\"500ms_to_2s\"} 107\n",
            "naked_core_agent_ttft_total{bucket=\"2s_to_10s\"} 108\n",
            "naked_core_agent_ttft_total{bucket=\"over_10s\"} 109\n",
            "# HELP naked_core_agent_ttft_sum_ms Time to first text token (ms)\n",
            "# TYPE naked_core_agent_ttft_sum_ms counter\n",
            "naked_core_agent_ttft_sum_ms 110\n",
            "# HELP naked_core_agent_ttft_count Time to first text token observations\n",
            "# TYPE naked_core_agent_ttft_count counter\n",
            "naked_core_agent_ttft_count 111\n",
            "# HELP naked_core_provider_stream_open_total Provider connect to first chunk (ms, successful streams only)\n",
            "# TYPE naked_core_provider_stream_open_total counter\n",
            "naked_core_provider_stream_open_total{bucket=\"under_500ms\"} 112\n",
            "naked_core_provider_stream_open_total{bucket=\"500ms_to_2s\"} 113\n",
            "naked_core_provider_stream_open_total{bucket=\"2s_to_10s\"} 114\n",
            "naked_core_provider_stream_open_total{bucket=\"over_10s\"} 115\n",
            "# HELP naked_core_provider_stream_open_sum_ms Provider connect to first chunk (ms, successful streams only)\n",
            "# TYPE naked_core_provider_stream_open_sum_ms counter\n",
            "naked_core_provider_stream_open_sum_ms 116\n",
            "# HELP naked_core_provider_stream_open_count Provider stream-open observations (successful streams only)\n",
            "# TYPE naked_core_provider_stream_open_count counter\n",
            "naked_core_provider_stream_open_count 117\n",
            "# HELP naked_core_tool_duration_total Tool execution wall duration (ms), by bounded tool class.\n",
            "# TYPE naked_core_tool_duration_total counter\n",
            "naked_core_tool_duration_total{tool=\"grep\",bucket=\"under_10ms\"} 118\n",
            "naked_core_tool_duration_total{tool=\"grep\",bucket=\"10ms_to_100ms\"} 119\n",
            "naked_core_tool_duration_total{tool=\"grep\",bucket=\"100ms_to_1s\"} 120\n",
            "naked_core_tool_duration_total{tool=\"grep\",bucket=\"over_1s\"} 121\n",
            "naked_core_tool_duration_total{tool=\"read\",bucket=\"under_10ms\"} 124\n",
            "naked_core_tool_duration_total{tool=\"read\",bucket=\"10ms_to_100ms\"} 125\n",
            "naked_core_tool_duration_total{tool=\"read\",bucket=\"100ms_to_1s\"} 126\n",
            "naked_core_tool_duration_total{tool=\"read\",bucket=\"over_1s\"} 127\n",
            "naked_core_tool_duration_total{tool=\"edit\",bucket=\"under_10ms\"} 130\n",
            "naked_core_tool_duration_total{tool=\"edit\",bucket=\"10ms_to_100ms\"} 131\n",
            "naked_core_tool_duration_total{tool=\"edit\",bucket=\"100ms_to_1s\"} 132\n",
            "naked_core_tool_duration_total{tool=\"edit\",bucket=\"over_1s\"} 133\n",
            "naked_core_tool_duration_total{tool=\"write\",bucket=\"under_10ms\"} 136\n",
            "naked_core_tool_duration_total{tool=\"write\",bucket=\"10ms_to_100ms\"} 137\n",
            "naked_core_tool_duration_total{tool=\"write\",bucket=\"100ms_to_1s\"} 138\n",
            "naked_core_tool_duration_total{tool=\"write\",bucket=\"over_1s\"} 139\n",
            "naked_core_tool_duration_total{tool=\"bash\",bucket=\"under_10ms\"} 142\n",
            "naked_core_tool_duration_total{tool=\"bash\",bucket=\"10ms_to_100ms\"} 143\n",
            "naked_core_tool_duration_total{tool=\"bash\",bucket=\"100ms_to_1s\"} 144\n",
            "naked_core_tool_duration_total{tool=\"bash\",bucket=\"over_1s\"} 145\n",
            "naked_core_tool_duration_total{tool=\"apply_patch\",bucket=\"under_10ms\"} 148\n",
            "naked_core_tool_duration_total{tool=\"apply_patch\",bucket=\"10ms_to_100ms\"} 149\n",
            "naked_core_tool_duration_total{tool=\"apply_patch\",bucket=\"100ms_to_1s\"} 150\n",
            "naked_core_tool_duration_total{tool=\"apply_patch\",bucket=\"over_1s\"} 151\n",
            "naked_core_tool_duration_total{tool=\"other\",bucket=\"under_10ms\"} 154\n",
            "naked_core_tool_duration_total{tool=\"other\",bucket=\"10ms_to_100ms\"} 155\n",
            "naked_core_tool_duration_total{tool=\"other\",bucket=\"100ms_to_1s\"} 156\n",
            "naked_core_tool_duration_total{tool=\"other\",bucket=\"over_1s\"} 157\n",
            "# HELP naked_core_tool_duration_sum_ms Tool execution wall duration (ms), by bounded tool class.\n",
            "# TYPE naked_core_tool_duration_sum_ms counter\n",
            "naked_core_tool_duration_sum_ms{tool=\"grep\"} 122\n",
            "naked_core_tool_duration_sum_ms{tool=\"read\"} 128\n",
            "naked_core_tool_duration_sum_ms{tool=\"edit\"} 134\n",
            "naked_core_tool_duration_sum_ms{tool=\"write\"} 140\n",
            "naked_core_tool_duration_sum_ms{tool=\"bash\"} 146\n",
            "naked_core_tool_duration_sum_ms{tool=\"apply_patch\"} 152\n",
            "naked_core_tool_duration_sum_ms{tool=\"other\"} 158\n",
            "# HELP naked_core_tool_duration_count Tool execution wall duration (ms), by bounded tool class. observations\n",
            "# TYPE naked_core_tool_duration_count counter\n",
            "naked_core_tool_duration_count{tool=\"grep\"} 123\n",
            "naked_core_tool_duration_count{tool=\"read\"} 129\n",
            "naked_core_tool_duration_count{tool=\"edit\"} 135\n",
            "naked_core_tool_duration_count{tool=\"write\"} 141\n",
            "naked_core_tool_duration_count{tool=\"bash\"} 147\n",
            "naked_core_tool_duration_count{tool=\"apply_patch\"} 153\n",
            "naked_core_tool_duration_count{tool=\"other\"} 159\n",
            "# HELP naked_core_fff_cold_build_total fff cold fast-index picker build duration (ms).\n",
            "# TYPE naked_core_fff_cold_build_total counter\n",
            "naked_core_fff_cold_build_total{bucket=\"under_10ms\"} 160\n",
            "naked_core_fff_cold_build_total{bucket=\"10ms_to_100ms\"} 161\n",
            "naked_core_fff_cold_build_total{bucket=\"100ms_to_1s\"} 162\n",
            "naked_core_fff_cold_build_total{bucket=\"over_1s\"} 163\n",
            "# HELP naked_core_fff_cold_build_sum_ms fff cold fast-index picker build duration (ms).\n",
            "# TYPE naked_core_fff_cold_build_sum_ms counter\n",
            "naked_core_fff_cold_build_sum_ms 164\n",
            "# HELP naked_core_fff_cold_build_count fff cold fast-index picker build duration (ms). observations\n",
            "# TYPE naked_core_fff_cold_build_count counter\n",
            "naked_core_fff_cold_build_count 165\n",
            "# HELP naked_core_parent_fsync_total Parent directory fsync duration for atomic writes (ms).\n",
            "# TYPE naked_core_parent_fsync_total counter\n",
            "naked_core_parent_fsync_total{bucket=\"under_10ms\"} 166\n",
            "naked_core_parent_fsync_total{bucket=\"10ms_to_100ms\"} 167\n",
            "naked_core_parent_fsync_total{bucket=\"100ms_to_1s\"} 168\n",
            "naked_core_parent_fsync_total{bucket=\"over_1s\"} 169\n",
            "# HELP naked_core_parent_fsync_sum_ms Parent directory fsync duration for atomic writes (ms).\n",
            "# TYPE naked_core_parent_fsync_sum_ms counter\n",
            "naked_core_parent_fsync_sum_ms 170\n",
            "# HELP naked_core_parent_fsync_count Parent directory fsync duration for atomic writes (ms). observations\n",
            "# TYPE naked_core_parent_fsync_count counter\n",
            "naked_core_parent_fsync_count 171\n",
            "# model health\n",
            "naked_model_health_probe 777\n",
        );
        assert_eq!(render_prometheus_from(&inputs), expected);
    }

    #[test]
    fn prometheus_latency_histograms_render_real_record_deltas() {
        let before = MediaRoutingSnapshot::default().render_prometheus();
        let turn_bucket = "naked_core_agent_turn_duration_total{bucket=\"1s_to_10s\"}";
        let turn_count = "naked_core_agent_turn_duration_count";
        let turn_sum = "naked_core_agent_turn_duration_sum_ms";
        let ttft_bucket = "naked_core_agent_ttft_total{bucket=\"500ms_to_2s\"}";
        let ttft_count = "naked_core_agent_ttft_count";
        let ttft_sum = "naked_core_agent_ttft_sum_ms";
        let provider_bucket = "naked_core_provider_stream_open_total{bucket=\"500ms_to_2s\"}";
        let provider_count = "naked_core_provider_stream_open_count";
        let provider_sum = "naked_core_provider_stream_open_sum_ms";

        let before_turn_bucket = metric_value(&before, turn_bucket);
        let before_turn_count = metric_value(&before, turn_count);
        let before_turn_sum = metric_value(&before, turn_sum);
        let before_ttft_bucket = metric_value(&before, ttft_bucket);
        let before_ttft_count = metric_value(&before, ttft_count);
        let before_ttft_sum = metric_value(&before, ttft_sum);
        let before_provider_bucket = metric_value(&before, provider_bucket);
        let before_provider_count = metric_value(&before, provider_count);
        let before_provider_sum = metric_value(&before, provider_sum);

        naked_core::metrics_hist::record_turn_duration(1_234);
        naked_core::metrics_hist::record_ttft(750);
        naked_core::metrics_hist::record_provider_stream_open(750);

        let after = MediaRoutingSnapshot::default().render_prometheus();
        assert_eq!(metric_value(&after, turn_bucket), before_turn_bucket + 1);
        assert_eq!(metric_value(&after, turn_count), before_turn_count + 1);
        assert_eq!(metric_value(&after, turn_sum), before_turn_sum + 1_234);
        assert_eq!(metric_value(&after, ttft_bucket), before_ttft_bucket + 1);
        assert_eq!(metric_value(&after, ttft_count), before_ttft_count + 1);
        assert_eq!(metric_value(&after, ttft_sum), before_ttft_sum + 750);
        assert_eq!(
            metric_value(&after, provider_bucket),
            before_provider_bucket + 1
        );
        assert_eq!(
            metric_value(&after, provider_count),
            before_provider_count + 1
        );
        assert_eq!(
            metric_value(&after, provider_sum),
            before_provider_sum + 750
        );
    }

    #[test]
    fn prometheus_per_tool_latency_render_real_record_deltas() {
        let before = MediaRoutingSnapshot::default().render_prometheus();
        let bucket = "naked_core_tool_duration_total{tool=\"grep\",bucket=\"over_1s\"}";
        let count = "naked_core_tool_duration_count{tool=\"grep\"}";
        let sum = "naked_core_tool_duration_sum_ms{tool=\"grep\"}";
        let before_bucket = metric_value(&before, bucket);
        let before_count = metric_value(&before, count);
        let before_sum = metric_value(&before, sum);

        naked_core::metrics_hist::record_tool_duration("grep_search", 1_234);

        let after = MediaRoutingSnapshot::default().render_prometheus();
        assert_eq!(metric_value(&after, bucket), before_bucket + 1);
        assert_eq!(metric_value(&after, count), before_count + 1);
        assert_eq!(metric_value(&after, sum), before_sum + 1_234);
    }

    #[test]
    fn per_tool_unknown_name_buckets_to_other() {
        let before = MediaRoutingSnapshot::default().render_prometheus();
        let other_bucket = "naked_core_tool_duration_total{tool=\"other\",bucket=\"100ms_to_1s\"}";
        let other_count = "naked_core_tool_duration_count{tool=\"other\"}";
        let other_sum = "naked_core_tool_duration_sum_ms{tool=\"other\"}";
        let forbidden_dynamic_bucket =
            "naked_core_tool_duration_total{tool=\"some_weird_tool\",bucket=\"100ms_to_1s\"}";
        let before_bucket = metric_value(&before, other_bucket);
        let before_count = metric_value(&before, other_count);
        let before_sum = metric_value(&before, other_sum);

        naked_core::metrics_hist::record_tool_duration("some_weird_tool", 250);

        let after = MediaRoutingSnapshot::default().render_prometheus();
        assert_eq!(metric_value(&after, other_bucket), before_bucket + 1);
        assert_eq!(metric_value(&after, other_count), before_count + 1);
        assert_eq!(metric_value(&after, other_sum), before_sum + 250);
        assert!(
            !after.contains(forbidden_dynamic_bucket),
            "unknown tool names must not create unbounded Prometheus labels"
        );
    }

    #[test]
    fn render_prometheus_includes_fff_fast_index_counters() {
        let inputs = PrometheusInputs {
            fff_picker_created: 1,
            fff_picker_reused: 2,
            fff_picker_cap_fallback: 3,
            fff_grep_fast_index: 4,
            fff_grep_fallback: 5,
            ..PrometheusInputs::default()
        };
        let rendered = render_prometheus_from(&inputs);
        assert!(rendered.contains("# HELP naked_core_fff_picker_registry_total"));
        assert!(rendered.contains("# TYPE naked_core_fff_picker_registry_total counter"));
        assert!(rendered.contains("naked_core_fff_picker_registry_total{outcome=\"created\"} 1"));
        assert!(rendered.contains("naked_core_fff_picker_registry_total{outcome=\"reused\"} 2"));
        assert!(
            rendered.contains("naked_core_fff_picker_registry_total{outcome=\"cap_fallback\"} 3")
        );
        assert!(rendered.contains("# HELP naked_core_fff_grep_total"));
        assert!(rendered.contains("# TYPE naked_core_fff_grep_total counter"));
        assert!(rendered.contains("naked_core_fff_grep_total{backend=\"fast_index\"} 4"));
        assert!(rendered.contains("naked_core_fff_grep_total{backend=\"fallback\"} 5"));
    }

    #[test]
    fn render_prometheus_includes_v2_ideal_tools_counters() {
        let inputs = PrometheusInputs {
            stale_edit_reject: 1,
            hashline_edit_applied: 2,
            hashline_edit_stale_anchor: 3,
            hashline_edit_overlap: 4,
            hashline_edit_out_of_bounds: 5,
            hashline_edit_disabled: 6,
            fs_cache_hit: 7,
            fs_cache_miss: 8,
            fs_cache_stale_bypass: 9,
            fs_cache_invalidate: 10,
            fs_cache_too_large: 11,
            persistent_bash_ok: 12,
            persistent_bash_timeout: 13,
            persistent_bash_killed: 14,
            persistent_bash_restart: 15,
            persistent_bash_error: 16,
            persistent_bash_disabled: 17,
            persistent_bash_busy: 18,
            ..PrometheusInputs::default()
        };
        let rendered = render_prometheus_from(&inputs);
        assert!(rendered.contains("# HELP naked_core_stale_edit_reject_total"));
        assert!(rendered.contains("# TYPE naked_core_stale_edit_reject_total counter"));
        assert!(rendered.contains("naked_core_stale_edit_reject_total 1"));
        assert!(rendered.contains("# HELP naked_core_hashline_edit_total"));
        assert!(rendered.contains("# TYPE naked_core_hashline_edit_total counter"));
        assert!(rendered.contains("naked_core_hashline_edit_total{outcome=\"applied\"} 2"));
        assert!(rendered.contains("naked_core_hashline_edit_total{outcome=\"stale_anchor\"} 3"));
        assert!(rendered.contains("naked_core_hashline_edit_total{outcome=\"overlap\"} 4"));
        assert!(rendered.contains("naked_core_hashline_edit_total{outcome=\"out_of_bounds\"} 5"));
        assert!(rendered.contains("naked_core_hashline_edit_total{outcome=\"disabled\"} 6"));
        assert!(rendered.contains("# HELP naked_core_fs_cache_total"));
        assert!(rendered.contains("# TYPE naked_core_fs_cache_total counter"));
        assert!(rendered.contains("naked_core_fs_cache_total{outcome=\"hit\"} 7"));
        assert!(rendered.contains("naked_core_fs_cache_total{outcome=\"miss\"} 8"));
        assert!(rendered.contains("naked_core_fs_cache_total{outcome=\"stale_bypass\"} 9"));
        assert!(rendered.contains("naked_core_fs_cache_total{outcome=\"invalidate\"} 10"));
        assert!(rendered.contains("naked_core_fs_cache_total{outcome=\"too_large\"} 11"));
        assert!(rendered.contains("# HELP naked_core_persistent_bash_total"));
        assert!(rendered.contains("# TYPE naked_core_persistent_bash_total counter"));
        assert!(rendered.contains("naked_core_persistent_bash_total{outcome=\"ok\"} 12"));
        assert!(rendered.contains("naked_core_persistent_bash_total{outcome=\"timeout\"} 13"));
        assert!(rendered.contains("naked_core_persistent_bash_total{outcome=\"killed\"} 14"));
        assert!(rendered.contains("naked_core_persistent_bash_total{outcome=\"restart\"} 15"));
        assert!(rendered.contains("naked_core_persistent_bash_total{outcome=\"error\"} 16"));
        assert!(rendered.contains("naked_core_persistent_bash_total{outcome=\"disabled\"} 17"));
        assert!(rendered.contains("naked_core_persistent_bash_total{outcome=\"busy\"} 18"));
    }

    #[test]
    fn render_prometheus_includes_memory_digest_metrics_with_zero_roads() {
        let inputs = PrometheusInputs {
            memory_candidates: 7,
            memory_digest_staleness: DigestStalenessSnapshot {
                global: 5,
                project: 6,
                user: 0,
            },
            ..PrometheusInputs::default()
        };
        let rendered = render_prometheus_from(&inputs);
        assert!(rendered.contains("# HELP naked_core_memory_candidates_total"));
        assert!(rendered.contains("# TYPE naked_core_memory_candidates_total counter"));
        assert!(rendered.contains("naked_core_memory_candidates_total 7"));
        assert!(rendered.contains("# HELP naked_core_memory_promoted_total"));
        assert!(rendered.contains("# TYPE naked_core_memory_promoted_total counter"));
        assert!(rendered.contains("naked_core_memory_promoted_total{road=\"repeat_days\"} 0"));
        assert!(rendered.contains("naked_core_memory_promoted_total{road=\"reinforce\"} 0"));
        assert!(rendered.contains("naked_core_memory_promoted_total{road=\"reobs\"} 0"));
        assert!(rendered.contains("# HELP naked_core_memory_injection_failed_total"));
        assert!(rendered.contains("# TYPE naked_core_memory_injection_failed_total counter"));
        assert!(rendered.contains("naked_core_memory_injection_failed_total 0"));
        assert!(rendered.contains("# HELP naked_core_memory_days_since_last_digest"));
        assert!(rendered.contains("# TYPE naked_core_memory_days_since_last_digest gauge"));
        assert!(rendered.contains("naked_core_memory_days_since_last_digest{scope=\"global\"} 5"));
        assert!(rendered.contains("naked_core_memory_days_since_last_digest{scope=\"project\"} 6"));
        assert!(rendered.contains("naked_core_memory_days_since_last_digest{scope=\"user\"} 0"));
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
    fn prometheus_includes_config_loaded_hash_info_gauge() {
        let rendered = render_prometheus_from(&PrometheusInputs {
            config_loaded_hash: Some("0123456789abcdef".to_string()),
            ..PrometheusInputs::default()
        });
        assert!(rendered.contains("# HELP naked_tg_config_loaded_hash"));
        assert!(rendered.contains("# TYPE naked_tg_config_loaded_hash gauge"));
        assert!(rendered.contains("naked_tg_config_loaded_hash{hash=\"0123456789abcdef\"} 1"));
    }

    #[test]
    fn prometheus_includes_active_run_max_silent_seconds_gauge() {
        use naked_tg::run_registry::{
            RegisterRunInput, RegisterRunOptions, RunKind, RunOrigin, RunRegistry,
        };
        use std::time::{Duration, Instant};
        use tokio::sync::mpsc;
        use tokio_util::sync::CancellationToken;

        let registry = RunRegistry::new();
        let (steer, _rx) = mpsc::channel(1);
        registry
            .register_run(
                RegisterRunInput {
                    requested_run_id: Some("run-silent".to_string()),
                    session_id: "sid-silent".to_string(),
                    origin: RunOrigin::new(42, None),
                    kind: RunKind::ChatTurn,
                    source_ref: None,
                    steer,
                    abort: CancellationToken::new(),
                },
                RegisterRunOptions::cap_three(),
            )
            .expect("register run");
        let base = Instant::now();
        registry.mark_run_progress_at("run-silent", base).unwrap();
        let silent_seconds =
            registry.active_run_max_silent_seconds_at(base + Duration::from_secs(73));

        let rendered = render_prometheus_from(&PrometheusInputs {
            active_run_max_silent_seconds: silent_seconds,
            ..PrometheusInputs::default()
        });
        assert!(rendered.contains("# HELP naked_tg_active_run_max_silent_seconds"));
        assert!(rendered.contains("# TYPE naked_tg_active_run_max_silent_seconds gauge"));
        assert!(rendered.contains("naked_tg_active_run_max_silent_seconds 73"));
    }

    #[test]
    fn prometheus_active_run_silent_gauge_reflects_shared_registry() {
        use naked_tg::run_registry::{RegisterRunInput, RegisterRunOptions, RunKind, RunOrigin};
        use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};
        use tokio::sync::mpsc;
        use tokio_util::sync::CancellationToken;

        let unique = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("system clock before epoch")
            .as_nanos();
        let run_id = format!("metrics-shared-silent-run-{unique}");
        let session_id = format!("metrics-shared-silent-sid-{unique}");
        let chat_id = 9_000_000_000_i64 + (unique % 1_000_000) as i64;
        let (steer, _rx) = mpsc::channel(1);
        let registry = &crate::shared::RUN_REGISTRY;
        let _ = registry.remove_run(&run_id);
        registry
            .register_run(
                RegisterRunInput {
                    requested_run_id: Some(run_id.clone()),
                    session_id,
                    origin: RunOrigin::new(chat_id, None),
                    kind: RunKind::ChatTurn,
                    source_ref: None,
                    steer,
                    abort: CancellationToken::new(),
                },
                RegisterRunOptions::cap_three(),
            )
            .expect("register shared run");
        registry
            .mark_run_progress_at(&run_id, Instant::now() - Duration::from_secs(3_600))
            .expect("backdate shared run progress");

        let rendered = MediaRoutingSnapshot::default().render_prometheus();
        let line = rendered
            .lines()
            .find(|line| line.starts_with("naked_tg_active_run_max_silent_seconds "))
            .unwrap_or_else(|| panic!("missing active-run silence gauge in:\n{rendered}"));
        let value = line
            .split_whitespace()
            .nth(1)
            .unwrap_or_else(|| panic!("missing gauge value in `{line}`"))
            .parse::<u64>()
            .unwrap_or_else(|err| panic!("invalid gauge value in `{line}`: {err}"));
        let _ = registry.remove_run(&run_id);

        assert!(
            value >= 3_600,
            "production render path must read shared RUN_REGISTRY silence age; got {value}"
        );
    }

    fn final_answer_bucket_value(rendered: &str, le: &str) -> u64 {
        metric_value(
            rendered,
            &format!("naked_tg_final_answer_utf16_bucket{{le=\"{le}\"}} "),
        )
    }

    #[test]
    fn final_answer_histogram_renders_cumulative_prometheus_buckets() {
        let inputs = PrometheusInputs {
            media: MediaRoutingSnapshot {
                final_answer_utf16: FinalAnswerUtf16HistogramSnapshot {
                    buckets: [2, 3, 5, 7, 11, 13, 17],
                    sum: 123_456,
                    count: 58,
                },
                ..MediaRoutingSnapshot::default()
            },
            ..PrometheusInputs::default()
        };
        let rendered = render_prometheus_from(&inputs);
        assert!(rendered.contains("# TYPE naked_tg_final_answer_utf16 histogram"));

        let mut prev = 0;
        for (le, expected) in [
            ("1024", 2),
            ("2048", 5),
            ("4096", 10),
            ("8192", 17),
            ("16384", 28),
            ("32768", 41),
            ("+Inf", 58),
        ] {
            let got = final_answer_bucket_value(&rendered, le);
            assert_eq!(got, expected, "unexpected cumulative bucket le={le}");
            assert!(
                got >= prev,
                "histogram buckets must be monotonic at le={le}"
            );
            prev = got;
        }
        assert_eq!(
            metric_value(&rendered, "naked_tg_final_answer_utf16_count "),
            58
        );
        assert_eq!(final_answer_bucket_value(&rendered, "+Inf"), 58);
        assert_eq!(
            metric_value(&rendered, "naked_tg_final_answer_utf16_sum "),
            123_456
        );
    }

    #[test]
    fn final_answer_metrics_zero_case_renders_every_series() {
        let rendered = render_prometheus_from(&PrometheusInputs::default());
        for le in FINAL_ANSWER_UTF16_BUCKET_LABELS {
            assert_eq!(final_answer_bucket_value(&rendered, le), 0);
        }
        for line in [
            "naked_tg_final_answer_utf16_sum 0",
            "naked_tg_final_answer_utf16_count 0",
            "naked_tg_final_answer_delivery_total{outcome=\"ok\"} 0",
            "naked_tg_final_answer_delivery_total{outcome=\"partial\"} 0",
            "naked_tg_final_answer_delivery_total{outcome=\"failed\"} 0",
            "naked_tg_final_answer_truncated_total 0",
            "naked_tg_final_answer_attachment_send_failed_total 0",
        ] {
            assert!(rendered.contains(line), "missing zero-series line `{line}`");
        }
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
