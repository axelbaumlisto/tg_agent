use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

use super::session::TurnUsage;

/// Sentinel prefix used by the JSONL session store to externalize image
/// payloads to disk (`<session>/artifacts/img_<hash>.<ext>`) instead of
/// bloating `session.jsonl` with inline base64. This constant is the source
/// of truth — both the externalize/internalize logic in `session::jsonl_store`
/// and the API-request guard in `history::to_api_messages` reference it.
///
/// Why public: `to_api_messages` MUST drop any Text block whose payload still
/// starts with this prefix (e.g. when an artifact file went missing and
/// `intern_image_blocks` couldn't rehydrate it). Otherwise the raw marker
/// leaks to the LLM and the model sees ugly internal JSON.
pub const IMAGE_REF_SENTINEL_PREFIX: &str = "@@NAKED_IMG_REF@@";

/// Returns true if the given text contains the image-ref sentinel anywhere
/// (not just at the start). Useful for defence-in-depth checks before
/// shipping text to an LLM.
pub fn contains_image_ref_sentinel(text: &str) -> bool {
    text.contains(IMAGE_REF_SENTINEL_PREFIX)
}

/// Process-wide counter incremented every time we strip / drop a sentinel
/// before it reaches an LLM. Surfaces in `/metrics` and tests so we can spot
/// silent extern-blob corruption (artifact files vanished, intern step
/// crashed, etc.).
pub static SENTINEL_LEAK_COUNT: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// Bumped each time the agent loop hits a "provider returned no content"
/// stream and successfully retries (or starts retrying) — see
/// `MAX_EMPTY_CONTENT_RETRIES` in `loop_.rs`. Exposed as
/// `naked_core_empty_content_retry_total` over Prometheus so operators can
/// quantify how flaky a given provider is in production.
pub static EMPTY_CONTENT_RETRY_COUNT: std::sync::atomic::AtomicU64 =
    std::sync::atomic::AtomicU64::new(0);

/// Total number of agent turns that completed successfully (`run` returned
/// `Ok`). Counted in `lib.rs` after the agent loop finishes. Pair with
/// `TURN_ERROR_COUNT` to compute a turn error rate.
pub static TURN_COMPLETED_COUNT: std::sync::atomic::AtomicU64 =
    std::sync::atomic::AtomicU64::new(0);

/// Total number of agent turns that failed (`run` returned `Err`). Bumped
/// in the same place as `TURN_COMPLETED_COUNT` so the two are sampled in
/// lockstep. The error reason is logged via `tracing::error!` for
/// distribution analysis (the in-process counter is intentionally
/// unlabelled to keep the renderer dead-simple).
pub static TURN_ERROR_COUNT: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// B62 / PLAN_RESEARCH_STORE_ATOMIC_v1 §R4: bumped each time
/// `dispatch_turn` rejects a second concurrent turn for the same
/// session before spawning an `AgentLoop`. Fixed unlabelled atomic so
/// renderers can expose `naked_core_session_double_turn_rejected_total`.
pub static SESSION_DOUBLE_TURN_REJECTED_COUNT: std::sync::atomic::AtomicU64 =
    std::sync::atomic::AtomicU64::new(0);

/// B59 / PLAN_RESEARCH_STORE_ATOMIC_v1 T4: malformed non-empty rows
/// detected while parsing `research/*/findings.jsonl` in the healing
/// read path. Incremented by bad row count, not per file.
pub static RESEARCH_STORE_CORRUPT_ROWS_DETECTED_COUNT: std::sync::atomic::AtomicU64 =
    std::sync::atomic::AtomicU64::new(0);

/// B59 / PLAN_RESEARCH_STORE_ATOMIC_v1 T4: `findings.jsonl` files
/// successfully rewritten from valid rows after a corrupt-row detection.
pub static RESEARCH_STORE_FILES_HEALED_COUNT: std::sync::atomic::AtomicU64 =
    std::sync::atomic::AtomicU64::new(0);

/// B59 / PLAN_RESEARCH_STORE_ATOMIC_v1 T4: healing attempts that aborted
/// before replacing the live file (for example, backup creation failed).
pub static RESEARCH_STORE_HEAL_FAILED_COUNT: std::sync::atomic::AtomicU64 =
    std::sync::atomic::AtomicU64::new(0);

/// B59 / FOLLOWUP F4: findings file-mutex acquisitions that had to wait
/// (contention on the per-research-id write lock).
pub static RESEARCH_STORE_FILE_LOCK_WAIT_COUNT: std::sync::atomic::AtomicU64 =
    std::sync::atomic::AtomicU64::new(0);

// ── Steer-pipeline counters (PLAN_NEXT_SESSION 2026-05-10) ─────────
//
// Three counters that pin the new steer behaviour. Operators can
// graph these to detect:
//   * `STEER_DELIVERED_COUNT` shrinking vs. `SteerMessage` channel
//     send rate → the bot is dropping user input.
//   * `STEER_SOFT_INTERRUPTED_COUNT` vs. delivered — fraction of
//     steers that hit during an in-flight LLM stream (S2/S3 path).
//   * `STEER_DRAINED_ON_ABORT_COUNT` rising = users frequently
//     abort with pending input, suggesting UX friction.
//
// All three are unlabelled `AtomicU64` for the same dead-simple
// renderer convention as the existing counters above.

/// Bumped once per successful drain in `loop_/steers.rs::drain_steers`
/// when at least one steer was merged into history.
pub static STEER_DELIVERED_COUNT: std::sync::atomic::AtomicU64 =
    std::sync::atomic::AtomicU64::new(0);

/// Bumped when the third arm of `stream_one_turn`'s select! fires
/// — a steer arrived mid-LLM-stream and triggered the soft-interrupt
/// path (S2/S3). Each burst-drain (multiple steers in one tick)
/// counts as one event.
pub static STEER_SOFT_INTERRUPTED_COUNT: std::sync::atomic::AtomicU64 =
    std::sync::atomic::AtomicU64::new(0);

/// Bumped when the run-loop's drain-on-error path (cancel / provider
/// error / empty-content giveup) actually found pending steers or
/// channel-buffered messages and rescued them into history. Zero
/// would mean the drain is a pure no-op safety net; non-zero means
/// it's actively saving user input.
pub static STEER_DRAINED_ON_ABORT_COUNT: std::sync::atomic::AtomicU64 =
    std::sync::atomic::AtomicU64::new(0);

// ── R5 of PLAN_RESILIENCE_v1 (2026-05-13) ────────────────────────
//
// Observability for PLAN_QUALITY_v1 modules. The 48h log audit
// on 2026-05-13 found zero journal hits for snapshot/LSP/hook/
// permission events because all fire-paths used `tracing::debug!`.
// These counters give operators a per-feature heartbeat in /metrics
// independent of the log level.

/// Bumped once per successful `SnapshotRepo::capture` invocation
/// in `session_ops::turn::dispatch_turn`. Phase label distinguishes
/// pre-turn from post-turn but for now we only fire pre-turn.
pub static SNAPSHOT_CAPTURE_COUNT: std::sync::atomic::AtomicU64 =
    std::sync::atomic::AtomicU64::new(0);

/// Bumped once per non-empty `LspManager::diagnostics_for` result.
/// The label dimension we'd want (language) is collapsed into a
/// single counter — dashboards already split by file extension via
/// the synthetic system message body.
pub static LSP_DIAGNOSTIC_EMITTED_COUNT: std::sync::atomic::AtomicU64 =
    std::sync::atomic::AtomicU64::new(0);

/// Bumped on every `Ruleset::evaluate` that produces a non-`Ask`
/// action (i.e. an actual rule hit). Stays at 0 until a user
/// installs a non-empty `~/.naked/permissions.json` or hits
/// `[✅ Always]` (when that UX lands).
pub static PERMISSION_RULE_MATCH_COUNT: std::sync::atomic::AtomicU64 =
    std::sync::atomic::AtomicU64::new(0);

/// Bumped each time `LifecycleHookRunner::run` actually fires a
/// configured hook (matcher matched the key). Currently 0 until a
/// user installs `~/.naked/hooks.json`.
pub static LIFECYCLE_HOOK_FIRE_COUNT: std::sync::atomic::AtomicU64 =
    std::sync::atomic::AtomicU64::new(0);

/// Bumped per call to `agent_role::canonicalize_role` that returns
/// `Some(...)` — i.e. the model invoked sub_agent with a valid role
/// alias. Zero would mean either no sub-agent calls or the model is
/// always passing unknown aliases (worth investigating).
pub static SUBAGENT_ROLE_RESOLVE_COUNT: std::sync::atomic::AtomicU64 =
    std::sync::atomic::AtomicU64::new(0);

/// R1 of PLAN_RESILIENCE_v1: bumped each time a key is
/// PERMANENTLY blacklisted by `ResilientProvider` (auth-failed /
/// payment-required errors). Distinct from the existing
/// `transient_blacklist` count (which is implicit, not exposed).
/// A flat-zero rate after the bot has run a while means every
/// provider key is healthy.
pub static PROVIDER_PERMANENT_BLACKLIST_COUNT: std::sync::atomic::AtomicU64 =
    std::sync::atomic::AtomicU64::new(0);

/// R4 of PLAN_RESILIENCE_v1: bumped on every successful
/// `notify_chat_about_crash` send. Distinct from the existing
/// log "notified N chat(s)" so the count is observable from
/// /metrics without grepping the journal.
pub static CRASH_RECOVERY_NOTIFIED_COUNT: std::sync::atomic::AtomicU64 =
    std::sync::atomic::AtomicU64::new(0);

/// BUG_REGISTRY D-BOOT-VISION-PROBE (B06): bumped each time a
/// (provider, model) pair declared as vision-capable in caps
/// (`supports_vision: true`) is found at boot to actually REJECT
/// the OpenAI-style `image_url` content shape. e.g. deepseek v4-pro
/// returns 400 "unknown variant `image_url`, expected `text`".
/// Operator should flip caps.supports_vision to false in naked.json.
pub static PROVIDER_VISION_CAP_MISMATCH_COUNT: std::sync::atomic::AtomicU64 =
    std::sync::atomic::AtomicU64::new(0);

/// B68: provider stream-open attempts that exceeded the connect-timeout
/// budget in `TimeoutProvider::stream_chat`. Rendered as
/// `naked_core_provider_timeout_total{kind="connect"}`.
pub static PROVIDER_CONNECT_TIMEOUT_COUNT: std::sync::atomic::AtomicU64 =
    std::sync::atomic::AtomicU64::new(0);

/// B68: provider streams that opened successfully but then exceeded the
/// inter-chunk timeout budget before the next chunk arrived. Rendered as
/// `naked_core_provider_timeout_total{kind="inter_chunk"}`.
pub static PROVIDER_INTER_CHUNK_TIMEOUT_COUNT: std::sync::atomic::AtomicU64 =
    std::sync::atomic::AtomicU64::new(0);

/// B76: upstream provider rejected the request as an HTTP 400
/// `invalid_request_error` (likely config/request-shape bug). Rendered as
/// `naked_core_provider_invalid_request_total`.
pub static PROVIDER_INVALID_REQUEST_COUNT: std::sync::atomic::AtomicU64 =
    std::sync::atomic::AtomicU64::new(0);

/// B75: scheduler skipped dispatch because the resolved research provider
/// chain reported every key/provider blacklisted. Rendered as
/// `naked_core_scheduler_dispatch_skipped_total`.
pub static SCHEDULER_DISPATCH_SKIPPED_COUNT: std::sync::atomic::AtomicU64 =
    std::sync::atomic::AtomicU64::new(0);

/// BUG_REGISTRY D-CONFIG-MTIME-WATCH (B42 detector): bumped each
/// time the bot detects that `state/naked.json` was modified by an
/// external process (mtime or sha256 differs from the boot snapshot).
/// Catches B42 — silent revert of config by recreate scripts or
/// concurrent agent sessions.
pub static CONFIG_EXTERNAL_WRITE_COUNT: std::sync::atomic::AtomicU64 =
    std::sync::atomic::AtomicU64::new(0);

/// PLAN_FAST_BACKEND_v2 Step B: explicit stale-edit precondition rejected a
/// write before atomic_replace_file, preserving the live file unchanged.
pub static STALE_EDIT_REJECT_COUNT: std::sync::atomic::AtomicU64 =
    std::sync::atomic::AtomicU64::new(0);

/// B85: production L1 policy-denies for bash commands containing a git
/// history-destructive operation (reset/checkout/switch/restore/clean/rebase/
/// revert/force-push) in a shared repo. Append-only git (add/commit/push) is
/// allowed and NOT counted here. The L2 execute-time backstop blocks safely but
/// is intentionally uncounted.
pub static GIT_HISTORY_GUARD_BLOCK_COUNT: std::sync::atomic::AtomicU64 =
    std::sync::atomic::AtomicU64::new(0);

/// PLAN_MEMORY_v05_UNFREEZE S4 / B97: memory digest candidates observed by
/// the daily scorer. Rendered as `naked_core_memory_candidates_total`.
pub static MEMORY_CANDIDATES_COUNT: std::sync::atomic::AtomicU64 =
    std::sync::atomic::AtomicU64::new(0);

/// PLAN_MEMORY_v05_UNFREEZE S4 / B97: promoted digest candidates by the road
/// that made them eligible. Rendered as `naked_core_memory_promoted_total{road}`.
pub static MEMORY_PROMOTED_REPEAT_DAYS_COUNT: std::sync::atomic::AtomicU64 =
    std::sync::atomic::AtomicU64::new(0);
pub static MEMORY_PROMOTED_REINFORCE_COUNT: std::sync::atomic::AtomicU64 =
    std::sync::atomic::AtomicU64::new(0);
pub static MEMORY_PROMOTED_REOBS_COUNT: std::sync::atomic::AtomicU64 =
    std::sync::atomic::AtomicU64::new(0);

/// PLAN_MEMORY_v05_UNFREEZE S5 / B100: memory entries omitted from prompt
/// injection because the fixed prompt budget was exhausted. Rendered as
/// `naked_core_memory_injection_dropped_total{scope}` with bounded scope
/// labels: global/project/user.
pub static MEMORY_INJECTION_DROPPED_GLOBAL_COUNT: std::sync::atomic::AtomicU64 =
    std::sync::atomic::AtomicU64::new(0);
pub static MEMORY_INJECTION_DROPPED_PROJECT_COUNT: std::sync::atomic::AtomicU64 =
    std::sync::atomic::AtomicU64::new(0);
pub static MEMORY_INJECTION_DROPPED_USER_COUNT: std::sync::atomic::AtomicU64 =
    std::sync::atomic::AtomicU64::new(0);

/// PLAN_MEMORY_v05_UNFREEZE S8 / F12: memory prompt injection failures that
/// were degraded to "inject nothing" so a memory bug cannot take down a turn.
/// Rendered as `naked_core_memory_injection_failed_total`.
pub static MEMORY_INJECTION_FAILED_COUNT: std::sync::atomic::AtomicU64 =
    std::sync::atomic::AtomicU64::new(0);

/// PLAN_FAST_BACKEND_v2 Step C: hashline edit outcomes. Intended Prometheus
/// shape: `naked_core_hashline_edit_total{outcome=...}` (rendering deferred).
pub static HASHLINE_EDIT_APPLIED_COUNT: std::sync::atomic::AtomicU64 =
    std::sync::atomic::AtomicU64::new(0);
pub static HASHLINE_EDIT_STALE_ANCHOR_COUNT: std::sync::atomic::AtomicU64 =
    std::sync::atomic::AtomicU64::new(0);
pub static HASHLINE_EDIT_OVERLAP_COUNT: std::sync::atomic::AtomicU64 =
    std::sync::atomic::AtomicU64::new(0);
pub static HASHLINE_EDIT_OUT_OF_BOUNDS_COUNT: std::sync::atomic::AtomicU64 =
    std::sync::atomic::AtomicU64::new(0);
pub static HASHLINE_EDIT_DISABLED_COUNT: std::sync::atomic::AtomicU64 =
    std::sync::atomic::AtomicU64::new(0);

/// PLAN_FAST_BACKEND_v2 Step D: bounded read_file/file_snapshot cache outcomes.
/// Intended Prometheus shape: `naked_core_fs_cache_total{outcome=...}`
/// (rendering deferred to Step F).
pub static FS_CACHE_HIT_COUNT: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
pub static FS_CACHE_MISS_COUNT: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
pub static FS_CACHE_STALE_BYPASS_COUNT: std::sync::atomic::AtomicU64 =
    std::sync::atomic::AtomicU64::new(0);
pub static FS_CACHE_INVALIDATE_COUNT: std::sync::atomic::AtomicU64 =
    std::sync::atomic::AtomicU64::new(0);
pub static FS_CACHE_TOO_LARGE_COUNT: std::sync::atomic::AtomicU64 =
    std::sync::atomic::AtomicU64::new(0);

/// PLAN_FAST_BACKEND_v2 Step E: persistent bash outcomes. Intended Prometheus
/// shape: `naked_core_persistent_bash_total{outcome=...}` (rendering deferred
/// to Step F).
pub static PERSISTENT_BASH_OK_COUNT: std::sync::atomic::AtomicU64 =
    std::sync::atomic::AtomicU64::new(0);
pub static PERSISTENT_BASH_TIMEOUT_COUNT: std::sync::atomic::AtomicU64 =
    std::sync::atomic::AtomicU64::new(0);
pub static PERSISTENT_BASH_KILLED_COUNT: std::sync::atomic::AtomicU64 =
    std::sync::atomic::AtomicU64::new(0);
pub static PERSISTENT_BASH_RESTART_COUNT: std::sync::atomic::AtomicU64 =
    std::sync::atomic::AtomicU64::new(0);
pub static PERSISTENT_BASH_ERROR_COUNT: std::sync::atomic::AtomicU64 =
    std::sync::atomic::AtomicU64::new(0);
pub static PERSISTENT_BASH_DISABLED_COUNT: std::sync::atomic::AtomicU64 =
    std::sync::atomic::AtomicU64::new(0);
pub static PERSISTENT_BASH_BUSY_COUNT: std::sync::atomic::AtomicU64 =
    std::sync::atomic::AtomicU64::new(0);

/// PLAN_FAST_BACKEND_v1 Step 2b: fff fast-index registry created a new
/// long-lived picker for a canonical workspace.
pub static FFF_PICKER_REGISTRY_CREATED_COUNT: std::sync::atomic::AtomicU64 =
    std::sync::atomic::AtomicU64::new(0);

/// PLAN_FAST_BACKEND_v1 Step 2b: fff fast-index registry reused an existing
/// long-lived picker for a canonical workspace.
pub static FFF_PICKER_REGISTRY_REUSED_COUNT: std::sync::atomic::AtomicU64 =
    std::sync::atomic::AtomicU64::new(0);

/// PLAN_FAST_BACKEND_v1 Step 2b: workspace cap/disabled mode forced legacy
/// fallback picker creation instead of spawning another watcher.
pub static FFF_PICKER_REGISTRY_CAP_FALLBACK_COUNT: std::sync::atomic::AtomicU64 =
    std::sync::atomic::AtomicU64::new(0);

/// PLAN_FAST_BACKEND_v1 Step 2b: grep requests served by the shared fast index.
pub static FFF_GREP_FAST_INDEX_COUNT: std::sync::atomic::AtomicU64 =
    std::sync::atomic::AtomicU64::new(0);

/// PLAN_FAST_BACKEND_v1 Step 2b: grep requests served by fallback legacy picker.
pub static FFF_GREP_FALLBACK_COUNT: std::sync::atomic::AtomicU64 =
    std::sync::atomic::AtomicU64::new(0);

/// BUG_REGISTRY D-VALIDATE-IP-TOKENS (B37 stream guard): bumped
/// each time an outgoing assistant message mentions noVNC/VNC
/// keyword AND contains an `IP:port` token that's NOT in the
/// boot-cached allow-list (from `novnc.sh url`). Catches B37 —
/// model hallucinating IP/port pairs when asked for noVNC creds.
pub static IP_TOKEN_HALLUCINATION_COUNT: std::sync::atomic::AtomicU64 =
    std::sync::atomic::AtomicU64::new(0);

/// Strip every embedded `@@NAKED_IMG_REF@@…` segment (sentinel + the
/// optional `/<hash>` or `{...json}` tail that follows it on the same line)
/// from `text`. Returns the cleaned string and bumps `SENTINEL_LEAK_COUNT`
/// once per strip.
///
/// Two payload shapes are handled:
///
/// * `@@NAKED_IMG_REF@@/abcd1234efgh…`   (newer hash form)
/// * `@@NAKED_IMG_REF@@{"path":"…"}`     (legacy JSON tail)
///
/// Anything up to the next whitespace, newline, or closing brace is stripped
/// together with the marker so we don't leave trailing garbage in the prompt.
pub fn strip_image_ref_sentinel(text: &str) -> std::borrow::Cow<'_, str> {
    if !text.contains(IMAGE_REF_SENTINEL_PREFIX) {
        return std::borrow::Cow::Borrowed(text);
    }
    let mut out = String::with_capacity(text.len());
    let mut rest = text;
    while let Some(pos) = rest.find(IMAGE_REF_SENTINEL_PREFIX) {
        out.push_str(&rest[..pos]);
        let after_marker = &rest[pos + IMAGE_REF_SENTINEL_PREFIX.len()..];
        // Determine where the marker payload ends. If the next char is `{`
        // we walk to the matching `}` (legacy JSON tail). Otherwise we eat
        // characters up to the next ASCII whitespace.
        let consumed = match after_marker.chars().next() {
            Some('{') => after_marker
                .find('}')
                .map(|i| i + 1)
                .unwrap_or(after_marker.len()),
            _ => after_marker
                .find(|c: char| c.is_ascii_whitespace())
                .unwrap_or(after_marker.len()),
        };
        rest = &after_marker[consumed..];
        SENTINEL_LEAK_COUNT.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    }
    out.push_str(rest);
    std::borrow::Cow::Owned(out)
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Role {
    System,
    User,
    Assistant,
    Tool,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ContentBlock {
    Text {
        text: String,
    },
    Thinking {
        text: String,
    },
    ToolUse {
        id: String,
        name: String,
        input: serde_json::Value,
    },
    ToolResult {
        call_id: String,
        output: String,
        is_error: bool,
    },
    /// Inline image attached to a user message. Stored as raw base64 + MIME so
    /// each provider can serialise it into its native multimodal format
    /// (Anthropic `image.source.base64`, OpenAI/Groq/xAI `image_url.url=data:`).
    Image {
        mime: String,
        data_base64: String,
        /// Per-image quality knob. OpenAI-compatible providers (gpt-4o,
        /// llama-4 vision) translate it to `image_url.detail = "low"|
        /// "high"|"auto"`, which controls token spend at inference time
        /// (`low` ~85 tokens, `high` ~tile-grid, `auto` lets the server
        /// decide). Anthropic, Gemini, and xAI ignore the field — they pick
        /// resolution server-side. `None` ⇒ "auto" / provider default.
        ///
        /// `#[serde(default)]` keeps existing session JSONL files
        /// deserializable; new images written today omit the field unless a
        /// non-default value is set.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        detail: Option<ImageDetail>,
    },
}

/// Quality preset for inline images. Mirrors OpenAI's `image_url.detail`
/// enum so we can pass it through unchanged.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ImageDetail {
    Auto,
    Low,
    High,
}

impl ImageDetail {
    /// String form for `image_url.detail`.
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Auto => "auto",
            Self::Low => "low",
            Self::High => "high",
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConversationMessage {
    pub role: Role,
    pub blocks: Vec<ContentBlock>,
    #[serde(default = "Utc::now")]
    pub timestamp: DateTime<Utc>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub usage: Option<TurnUsage>,
}

impl ConversationMessage {
    pub fn user(text: impl Into<String>) -> Self {
        Self {
            role: Role::User,
            blocks: vec![ContentBlock::Text { text: text.into() }],
            timestamp: Utc::now(),
            usage: None,
        }
    }

    pub fn assistant(blocks: Vec<ContentBlock>, usage: Option<TurnUsage>) -> Self {
        Self {
            role: Role::Assistant,
            blocks,
            timestamp: Utc::now(),
            usage,
        }
    }

    pub fn tool_result(
        call_id: impl Into<String>,
        output: impl Into<String>,
        is_error: bool,
    ) -> Self {
        Self {
            role: Role::Tool,
            blocks: vec![ContentBlock::ToolResult {
                call_id: call_id.into(),
                output: output.into(),
                is_error,
            }],
            timestamp: Utc::now(),
            usage: None,
        }
    }

    pub fn system(text: impl Into<String>) -> Self {
        Self {
            role: Role::System,
            blocks: vec![ContentBlock::Text { text: text.into() }],
            timestamp: Utc::now(),
            usage: None,
        }
    }

    pub fn text_content(&self) -> String {
        self.blocks
            .iter()
            .filter_map(|b| match b {
                ContentBlock::Text { text } => Some(text.as_str()),
                _ => None,
            })
            .collect::<Vec<_>>()
            .join("")
    }

    pub fn tool_uses(&self) -> Vec<(&str, &str, &serde_json::Value)> {
        self.blocks
            .iter()
            .filter_map(|b| match b {
                ContentBlock::ToolUse { id, name, input } => {
                    Some((id.as_str(), name.as_str(), input))
                }
                _ => None,
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn role_serde_roundtrip() {
        for role in [Role::System, Role::User, Role::Assistant] {
            let json = serde_json::to_string(&role).unwrap();
            let parsed: Role = serde_json::from_str(&json).unwrap();
            assert_eq!(parsed, role);
        }
    }

    #[test]
    fn content_block_text_serde() {
        let block = ContentBlock::Text {
            text: "hello".into(),
        };
        let json = serde_json::to_string(&block).unwrap();
        assert!(json.contains("hello"));
        let parsed: ContentBlock = serde_json::from_str(&json).unwrap();
        assert!(matches!(parsed, ContentBlock::Text { text } if text == "hello"));
    }

    #[test]
    fn conversation_message_roundtrip() {
        let msg = ConversationMessage {
            role: Role::User,
            blocks: vec![ContentBlock::Text {
                text: "test".into(),
            }],
            timestamp: chrono::Utc::now(),
            usage: None,
        };
        let json = serde_json::to_string(&msg).unwrap();
        let parsed: ConversationMessage = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed.role, Role::User);
        assert_eq!(parsed.blocks.len(), 1);
    }
}
