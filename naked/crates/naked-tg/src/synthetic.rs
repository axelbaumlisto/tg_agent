//! Synthetic message types for scheduler-driven research runs.
//!
//! # Problem
//!
//! Before this module the scheduler called `AgentCore::run_research_verified`
//! directly, creating an ephemeral session with `channel="research"` that has
//! no relation to the operator's Telegram chat. The operator cannot abort it
//! via their chat's `/abort` command because the session is not bound to any
//! `(chat_id, thread_id)` in the `ChannelSessionMap`.
//!
//! # Solution
//!
//! The scheduler injects a [`SyntheticMessage`] into the agent turn pipeline
//! via a [`SyntheticDispatchFn`] closure that wiring.rs provides. The closure
//! resolves (or creates) a session bound to the operator's `(chat_id,
//! thread_id)`, submits a prompt, and **drains the resulting `AgentHandle`
//! to completion** — auto-approving permission requests, collecting errors,
//! and returning a structured [`SyntheticDispatchOutcome`].
//!
//! When `StreamDeps` is wired (operator chat), the dispatch path renders a
//! **live Telegram streaming bubble** via the SAME `stream_response` engine
//! the chat path uses (`wiring/research_scheduler.rs:87-210`), with the Abort
//! button — it is NOT headless. The **headless drain**
//! (`drain_synthetic_agent_handle`, no streaming UI / no Abort button) remains
//! only as the `StreamDeps == None` fallback (CLI / tests / no-bot).
//! Either way, the scheduler awaits the outcome and maps it to ledger state
//! (Completed / Failed / Timeout), and the operator can `/abort` the session
//! because it is bound in the `ChannelSessionMap`.
//!
//! NB (B79): the synthetic gate that decides streaming-vs-headless must read
//! the effective chat (`infl.chat_id.or(spec.chat_id)`), not `spec.chat_id`
//! alone — see `scheduler/tasks.rs` and BUG_REGISTRY B79.
//!
//! # B57 (PLAN_RESEARCH_AGENT_FLOW_v1)
//!
//! `SCHEDULER-SPAWNS-PARALLEL-SESSION`: scheduler creates ephemeral session
//! with channel ≠ chat's. Fix: scheduler calls `dispatch_synthetic` which
//! routes through the normal message handler → same chat session → operator
//! can `/abort` like any other turn.
//!
//! # B64 (PLAN_SYNTHETIC_DISPATCH_FIX_v1)
//!
//! `FIRE-AND-FORGET`: dispatch closure discarded `AgentHandle`, returned
//! instantly. Fix: `SyntheticDispatchFn` now returns
//! `Result<SyntheticDispatchOutcome, String>` so the scheduler knows whether
//! the turn succeeded, failed, or was cancelled.

use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;

use naked_core::research::StopReason;

/// Outcome of a synthetic dispatch run. Carries enough information
/// for the scheduler to decide success vs failure.
#[derive(Debug, Clone)]
pub struct SyntheticDispatchOutcome {
    /// Session ID that was used for the turn.
    pub session_id: String,
    /// How the turn ended.
    pub stop_reason: StopReason,
    /// Collected `AgentEvent::Error` messages during drain.
    pub errors: Vec<String>,
}

/// Where a synthetic message originates. Used for logging + to decide
/// whether to suppress the outbound "echo" send_message to TG.
#[derive(Debug, Clone)]
pub enum SyntheticSource {
    /// Triggered by the in-process research scheduler for a periodic run.
    Scheduler { spec_id: String },
    /// Triggered by a cron job or external process (reserved for future).
    Cron { label: String },
    /// Triggered manually via an operator API (reserved for future).
    Manual,
}

impl SyntheticSource {
    /// Human-readable label used in tracing / logs.
    pub fn label(&self) -> &str {
        match self {
            SyntheticSource::Scheduler { .. } => "scheduler",
            SyntheticSource::Cron { .. } => "cron",
            SyntheticSource::Manual => "manual",
        }
    }

    /// For scheduler runs: the spec_id that triggered this message.
    pub fn spec_id(&self) -> Option<&str> {
        match self {
            SyntheticSource::Scheduler { spec_id } => Some(spec_id),
            _ => None,
        }
    }
}

/// A message that the scheduler (or another non-TG source) wants to inject
/// into the normal Telegram message handling pipeline.
///
/// The message is routed through `handle_message` exactly as a real user
/// message would be — the only difference is the `is_synthetic` flag, which
/// causes the message handler to skip outbound `send_message` for the
/// synthetic command text itself.
#[derive(Debug, Clone)]
pub struct SyntheticMessage {
    /// Target chat (matches `spec.chat_id`).
    pub chat_id: i64,
    /// Target topic / thread (matches `spec.thread_id`).
    pub thread_id: Option<i32>,
    /// The command text to inject, e.g. `"/research run da-nang-123"`.
    pub text: String,
    /// Origin of this synthetic message. Controls log labels and echo
    /// suppression policy.
    pub source: SyntheticSource,
    /// Optional RunRegistry identity supplied by an external launcher.
    /// Research scheduler maps `Inflight.attempt_id` here; other sources leave
    /// it empty and let the registry generate a kind-agnostic run id.
    pub run_id: Option<String>,
}

impl SyntheticMessage {
    /// Build a synthetic `/research run <spec_id>` message from a scheduler
    /// spec. Returns `None` if the spec has no `chat_id` configured (CLI /
    /// unattended runs fall back to the legacy direct-run path).
    pub fn from_scheduler_spec(spec_id: &str, chat_id: i64, thread_id: Option<i32>) -> Self {
        Self::from_scheduler_spec_with_run_id(spec_id, chat_id, thread_id, None)
    }

    pub fn from_scheduler_spec_with_run_id(
        spec_id: &str,
        chat_id: i64,
        thread_id: Option<i32>,
        run_id: Option<String>,
    ) -> Self {
        Self {
            chat_id,
            thread_id,
            text: format!("/research run {spec_id}"),
            source: SyntheticSource::Scheduler {
                spec_id: spec_id.to_string(),
            },
            run_id,
        }
    }
}

/// Opaque async function that the scheduler calls to inject a
/// [`SyntheticMessage`] into the agent turn pipeline.
///
/// `wiring.rs` constructs this as a closure that resolves/creates a session,
/// submits the prompt via `AgentCore::send_prompt`, drains the resulting
/// `AgentHandle` to completion, and returns a structured
/// [`SyntheticDispatchOutcome`]. The scheduler maps the outcome to ledger
/// state (Completed / Failed).
///
/// The scheduler only holds `Arc<SyntheticDispatchFn>` — no `Bot` in scope.
///
/// ## Why a boxed async fn instead of a trait?
///
/// The scheduler config needs to be `Clone` + `Default`. A trait object would
/// require `Arc<dyn …>` and async traits (which are not object-safe without
/// boxing). The boxed closure is simpler and achieves the same result.
/// Shared latch that lets the dispatch closure publish the session-id
/// *before* the blocking `stream_response()` call.  The scheduler's
/// cancel branch reads it to call `agent.abort(sid)`.
pub type SessionIdLatch = Arc<tokio::sync::Mutex<Option<String>>>;

pub type SyntheticDispatchFn = Arc<
    dyn Fn(
            SyntheticMessage,
            SessionIdLatch,
        )
            -> Pin<Box<dyn Future<Output = Result<SyntheticDispatchOutcome, String>> + Send>>
        + Send
        + Sync,
>;

// ────────────────────────────────────────────────────────────────────────
// Testing surface — MockDispatcher
//
// `MockDispatcher` records every [`SyntheticMessage`] handed to the
// `SyntheticDispatchFn` it produces. Lives in the main module (not
// `#[cfg(test)]`-gated) so integration tests and the `naked tg`
// CLI subcommand can both reuse it. Production code paths NEVER use
// it (the dispatch closure built in `wiring.rs` carries real Bot +
// AgentCore handles).
//
// Construction:
//   let mock = MockDispatcher::new();
//   let fn_handle = mock.as_fn();          // Arc<SyntheticDispatchFn>
//   scheduler_cfg.dispatch_fn = Some(fn_handle);
//   /* scheduler ticks, calls the closure */
//   let captured = mock.snapshot().await;  // Vec<SyntheticMessage>
//
// Thread-safety: backed by `tokio::sync::Mutex` because the closure
// returns a `Send` future; sync Mutex would deadlock if the same
// task polls + records.
// ────────────────────────────────────────────────────────────────────────

/// A test double for [`SyntheticDispatchFn`] that records every message
/// dispatched through it without contacting Telegram.
///
/// Cheap to clone (`Arc` inside). Use [`MockDispatcher::as_fn`] to plug
/// into [`crate::scheduler::SchedulerConfig::dispatch_fn`], then
/// [`MockDispatcher::snapshot`] to inspect captured messages.
///
/// Default delay: 0ms. Configure via [`MockDispatcher::with_delay_ms`]
/// to simulate slow dispatch for cancel-propagation tests.
#[derive(Clone, Default)]
pub struct MockDispatcher {
    captured: Arc<tokio::sync::Mutex<Vec<SyntheticMessage>>>,
    delay_ms: u64,
}

impl MockDispatcher {
    /// New mock with no artificial delay.
    pub fn new() -> Self {
        Self::default()
    }

    /// Tune the simulated dispatch latency. Useful for testing that
    /// `agent.abort(session_id)` interrupts an in-flight dispatch
    /// (B56 cancel-propagation regression guard).
    pub fn with_delay_ms(mut self, delay_ms: u64) -> Self {
        self.delay_ms = delay_ms;
        self
    }

    /// Construct the [`SyntheticDispatchFn`] suitable for
    /// [`crate::scheduler::SchedulerConfig::dispatch_fn`]. Each
    /// invocation appends the [`SyntheticMessage`] to the shared
    /// capture log and returns a default success outcome.
    pub fn as_fn(&self) -> SyntheticDispatchFn {
        let captured = self.captured.clone();
        let delay_ms = self.delay_ms;
        Arc::new(move |msg: SyntheticMessage, sid_latch: SessionIdLatch| {
            let captured = captured.clone();
            Box::pin(async move {
                if delay_ms > 0 {
                    tokio::time::sleep(std::time::Duration::from_millis(delay_ms)).await;
                }
                tracing::debug!(
                    chat_id = msg.chat_id,
                    source = %msg.source.label(),
                    spec_id = ?msg.source.spec_id(),
                    "MockDispatcher captured synthetic message",
                );
                // Latch a mock session-id so cancel tests can observe it.
                *sid_latch.lock().await = Some("mock".into());
                captured.lock().await.push(msg);
                Ok(SyntheticDispatchOutcome {
                    session_id: "mock".into(),
                    stop_reason: StopReason::AgentIdle,
                    errors: vec![],
                })
            })
        })
    }

    /// Snapshot of captured messages, in the order they were dispatched.
    /// Returns a *clone* — the internal log is preserved so the same
    /// mock can be inspected multiple times.
    pub async fn snapshot(&self) -> Vec<SyntheticMessage> {
        self.captured.lock().await.clone()
    }

    /// Number of messages captured so far. Cheap accessor for assertions.
    pub async fn count(&self) -> usize {
        self.captured.lock().await.len()
    }

    /// Clear the capture log (e.g. between sub-tests in a single fixture).
    pub async fn reset(&self) {
        self.captured.lock().await.clear();
    }
}

// ────────────────────────────────────────────────────────────────────────
// T6.2 (PLAN_RESEARCH_AGENT_FLOW_v1): dispatch_synthetic entry point.
//
// The scheduler holds an `Arc<SyntheticDispatchFn>` constructed by
// wiring.rs (which has ChannelSessionMap + AgentCore in scope).
// When the scheduler decides a spec is due, it calls the closure with
// a SyntheticMessage; the closure:
//   1. resolves/creates a session bound to (chat_id, thread_id)
//   2. submits the prompt via AgentCore::send_prompt
//   3. drives the AgentHandle to completion: streams a live bubble via
//      `stream_response` when `StreamDeps` is present, else headless drain
//   4. returns SyntheticDispatchOutcome to the scheduler
//
// When `StreamDeps` is wired (operator chat) the run renders a LIVE
// streaming bubble with an Abort button via the shared `stream_response`
// engine (`wiring/research_scheduler.rs:87-210`). The HEADLESS DRAIN
// (no streaming bubble, no Abort button callback) is only the
// `StreamDeps == None` fallback (CLI / tests). Either way the session IS
// bound in ChannelSessionMap, so the operator can `/abort` it from the chat.
//
// Echo suppression: the closure does NOT call `bot.send_message` for
// the synthetic command text itself — there is no operator-visible
// "/research run X" line in chat history.
//
// This module provides the BUILDER for the closure. wiring.rs calls
// `build_dispatch_fn(...)` once at boot, passes the result to
// `SchedulerConfig::dispatch_fn`. The closure body calls
// `submit_synthetic_prompt` + `drain_synthetic_agent_handle` — a
// structured drain that auto-approves permissions, collects errors,
// and returns a result-bearing outcome.
// ────────────────────────────────────────────────────────────────────────

/// Synthesize a turn for the given `(chat_id, thread_id)` and return the
/// resulting session id + agent handle. Used by the scheduler-facing
/// `SyntheticDispatchFn` closure (wired in `wiring.rs`).
///
/// This function isolates the **policy** of synthetic-turn dispatch from
/// the **wiring** of session resolution + prompt submission. The caller
/// is responsible for draining the returned `AgentHandle` — typically via
/// `drain_synthetic_agent_handle` (headless drain with permission
/// auto-approve).
///
/// Returns:
/// - `Ok((session_id, AgentHandle))` on success
/// - `Err(reason)` if `send_prompt` failed (e.g. session not found, provider error)
pub async fn submit_synthetic_prompt(
    agent: &std::sync::Arc<naked_core::AgentCore>,
    session_id: String,
    msg: &SyntheticMessage,
) -> Result<(String, naked_core::types::AgentHandle), String> {
    let handle = agent
        .send_prompt(&session_id, &msg.text)
        .await
        .map_err(|e| format!("send_prompt failed: {e}"))?;
    tracing::info!(
        session_id = %session_id,
        chat_id = msg.chat_id,
        thread_id = ?msg.thread_id,
        source = %msg.source.label(),
        spec_id = ?msg.source.spec_id(),
        text_preview = %msg.text.chars().take(80).collect::<String>(),
        "synthetic prompt submitted",
    );
    Ok((session_id, handle))
}

// ────────────────────────────────────────────────────────────────────────
// T1 (PLAN_v13_SOLID_AUDIT): session-aware dispatch helpers.
//
// Previously in `synthetic_dispatch.rs` (binary crate) due to the dual
// `mod channel_map` type mismatch.  Now that the binary uses the lib's
// `channel_map` module, these live here next to `submit_synthetic_prompt`.
// ────────────────────────────────────────────────────────────────────────

/// Resolve the session id for `(chat_id, thread_id)` from the
/// [`crate::channel_map::ChannelSessionMap`], creating a fresh
/// `channel="telegram"` session if none exists.  The new session is
/// persisted into the map so subsequent calls (including `/abort`)
/// can find it.
pub async fn resolve_or_create_session(
    agent: &std::sync::Arc<naked_core::AgentCore>,
    channel_map: &std::sync::Arc<crate::channel_map::ChannelSessionMap>,
    chat_id: i64,
    thread_id: Option<i32>,
) -> String {
    if let Some(sid) = channel_map.get(chat_id, thread_id).await {
        return sid;
    }
    let workspace =
        std::path::PathBuf::from(std::env::var("NAKED_WORKSPACE").unwrap_or_else(|_| ".".into()));
    let sid = agent
        .create_session_with_channel(&workspace, "telegram")
        .await;
    channel_map.set(chat_id, thread_id, sid.clone()).await;
    sid
}

/// One-shot synthetic dispatch for a `(chat_id, thread_id)` chat.
///
/// Resolves or creates a session, builds a
/// [`SyntheticMessage::from_scheduler_spec`], and calls
/// [`submit_synthetic_prompt`].  Returns `(session_id, AgentHandle)`.
pub async fn dispatch_for_chat(
    agent: &std::sync::Arc<naked_core::AgentCore>,
    _channel_map: &std::sync::Arc<crate::channel_map::ChannelSessionMap>,
    chat_id: i64,
    thread_id: Option<i32>,
    spec_id: &str,
) -> Result<(String, naked_core::types::AgentHandle), String> {
    // B64 follow-up: scheduler research runs use a FRESH session.
    //
    // Previously this called `resolve_or_create_session()` which returned
    // the operator's existing chat session (100+ messages of unrelated
    // context). The model then treated the synthetic prompt as a regular
    // chat message instead of dispatching research tools. Additionally,
    // reusing the chat session meant SessionBusy errors blocked the
    // operator from chatting while research ran.
    //
    // Fix: always create a fresh "research" channel session with empty
    // history. The model sees only the research command and acts
    // accordingly. The operator's chat session is not touched.
    //
    // NOTE (B72): scheduled/synthetic research uses the PROVIDER-level
    // fallback chain reached by the normal session provider resolution
    // (`AgentCore::send_prompt` → setup/provider_svc → create_provider_chain).
    // `research.fallback_models` is model-level fallback config consumed only
    // by the legacy ResearchCoordinator (`runner_dispatch.rs`) and is NOT
    // applied here. Wiring model-level fallback into synthetic dispatch is
    // tracked as future work.
    let workspace =
        std::path::PathBuf::from(std::env::var("NAKED_WORKSPACE").unwrap_or_else(|_| ".".into()));
    let session_id = agent
        .create_session_with_channel(&workspace, "research")
        .await;
    let msg = SyntheticMessage::from_scheduler_spec(spec_id, chat_id, thread_id);
    submit_synthetic_prompt(agent, session_id, &msg).await
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn synthetic_message_constructor() {
        let msg = SyntheticMessage::from_scheduler_spec("da-nang-abc", 12345, Some(42));
        assert_eq!(msg.chat_id, 12345);
        assert_eq!(msg.thread_id, Some(42));
        assert_eq!(msg.text, "/research run da-nang-abc");
        assert!(msg.text.contains("da-nang-abc"));
    }

    #[test]
    fn synthetic_source_label() {
        let s = SyntheticSource::Scheduler {
            spec_id: "test".into(),
        };
        assert_eq!(s.label(), "scheduler");
        assert_eq!(s.spec_id(), Some("test"));

        let c = SyntheticSource::Cron {
            label: "daily".into(),
        };
        assert_eq!(c.label(), "cron");
        assert_eq!(c.spec_id(), None);

        assert_eq!(SyntheticSource::Manual.label(), "manual");
    }

    #[test]
    fn synthetic_message_from_scheduler_spec_no_thread() {
        let msg = SyntheticMessage::from_scheduler_spec("spec-xyz", 99999, None);
        assert_eq!(msg.thread_id, None);
        assert_eq!(msg.text, "/research run spec-xyz");
    }

    fn new_latch() -> SessionIdLatch {
        std::sync::Arc::new(tokio::sync::Mutex::new(None))
    }

    // ── MockDispatcher tests (replaces the previous include_str! sentinel) ──

    /// MockDispatcher captures messages exactly once per dispatch call.
    #[tokio::test]
    async fn mock_dispatcher_captures_single_message() {
        let mock = MockDispatcher::new();
        let dispatch = mock.as_fn();

        let msg = SyntheticMessage::from_scheduler_spec("spec-1", 12345, Some(7));
        let result = dispatch(msg.clone(), new_latch()).await;
        assert!(result.is_ok(), "MockDispatcher should return Ok");
        let outcome = result.unwrap();
        assert_eq!(outcome.session_id, "mock");
        assert_eq!(outcome.stop_reason, StopReason::AgentIdle);
        assert!(outcome.errors.is_empty());

        let captured = mock.snapshot().await;
        assert_eq!(captured.len(), 1);
        assert_eq!(captured[0].chat_id, 12345);
        assert_eq!(captured[0].thread_id, Some(7));
        assert_eq!(captured[0].text, "/research run spec-1");
        assert_eq!(captured[0].source.spec_id(), Some("spec-1"));
    }

    /// MockDispatcher preserves dispatch order and accumulates multiple.
    #[tokio::test]
    async fn mock_dispatcher_records_multiple_in_order() {
        let mock = MockDispatcher::new();
        let dispatch = mock.as_fn();

        for i in 0..3 {
            let _ = dispatch(
                SyntheticMessage::from_scheduler_spec(&format!("spec-{i}"), 1000 + i, None),
                new_latch(),
            )
            .await;
        }

        let captured = mock.snapshot().await;
        assert_eq!(captured.len(), 3);
        assert_eq!(captured[0].chat_id, 1000);
        assert_eq!(captured[1].chat_id, 1001);
        assert_eq!(captured[2].chat_id, 1002);
        assert_eq!(mock.count().await, 3);
    }

    /// Cloning the mock yields independent handles sharing the same
    /// capture log — critical for the "closure captures Arc<dispatch>,
    /// test holds another Arc<dispatch>" pattern.
    #[tokio::test]
    async fn mock_dispatcher_clones_share_log() {
        let mock = MockDispatcher::new();
        let mock_b = mock.clone();

        let d1 = mock.as_fn();
        let d2 = mock_b.as_fn();

        let _ = d1(
            SyntheticMessage::from_scheduler_spec("a", 1, None),
            new_latch(),
        )
        .await;
        let _ = d2(
            SyntheticMessage::from_scheduler_spec("b", 2, None),
            new_latch(),
        )
        .await;

        // Either handle sees both messages.
        assert_eq!(mock.count().await, 2);
        assert_eq!(mock_b.count().await, 2);
    }

    /// `reset()` clears between sub-tests without rebuilding the handle.
    #[tokio::test]
    async fn mock_dispatcher_reset_clears_log() {
        let mock = MockDispatcher::new();
        let dispatch = mock.as_fn();

        let _ = dispatch(
            SyntheticMessage::from_scheduler_spec("x", 1, None),
            new_latch(),
        )
        .await;
        assert_eq!(mock.count().await, 1);

        mock.reset().await;
        assert_eq!(mock.count().await, 0);

        let _ = dispatch(
            SyntheticMessage::from_scheduler_spec("y", 2, None),
            new_latch(),
        )
        .await;
        assert_eq!(mock.count().await, 1);
        assert_eq!(mock.snapshot().await[0].source.spec_id(), Some("y"));
    }

    /// `with_delay_ms` actually delays — minimum observable latency check.
    /// Useful regression guard if the impl is accidentally changed to
    /// skip the sleep when `delay_ms == 0`.
    #[tokio::test]
    async fn mock_dispatcher_delay_observable() {
        let mock = MockDispatcher::new().with_delay_ms(20);
        let dispatch = mock.as_fn();
        let start = std::time::Instant::now();
        let _ = dispatch(
            SyntheticMessage::from_scheduler_spec("d", 1, None),
            new_latch(),
        )
        .await;
        assert!(
            start.elapsed() >= std::time::Duration::from_millis(15),
            "with_delay_ms(20) must produce ≥15ms latency, got {:?}",
            start.elapsed()
        );
    }

    /// T6.2 (PLAN_RESEARCH_AGENT_FLOW_v1): `submit_synthetic_prompt` API
    /// shape sentinel. We can't easily mock an in-process `AgentCore` for
    /// `send_prompt` without standing up a full provider stack, so the
    /// signature remains pinned via a coarse source-text grep. The mock
    /// tests above exercise the **scheduler→closure** seam end-to-end;
    /// `submit_synthetic_prompt` itself is exercised by the live
    /// `research_llm_control_e2e` suite.
    #[test]
    fn submit_synthetic_prompt_api_shape_locked() {
        let src = include_str!("synthetic.rs");
        assert!(
            src.contains("pub async fn submit_synthetic_prompt"),
            "submit_synthetic_prompt entry point must remain pub"
        );
        assert!(
            src.contains("AgentHandle"),
            "return type must include AgentHandle so caller can drive streaming"
        );
        assert!(
            src.contains("synthetic prompt submitted"),
            "INFO log tag required for boot-time observability (scheduler tick → turn)"
        );
    }

    /// B64: `SyntheticDispatchOutcome` must exist and carry session_id +
    /// stop_reason + errors for the scheduler to map to ledger state.
    #[test]
    fn synthetic_dispatch_outcome_shape() {
        let outcome = SyntheticDispatchOutcome {
            session_id: "test-session".into(),
            stop_reason: StopReason::AgentIdle,
            errors: vec!["some error".into()],
        };
        assert_eq!(outcome.session_id, "test-session");
        assert_eq!(outcome.stop_reason, StopReason::AgentIdle);
        assert_eq!(outcome.errors.len(), 1);
    }

    /// B64: `SyntheticDispatchFn` must return Result, not ().
    #[test]
    fn dispatch_fn_returns_result() {
        let src = include_str!("synthetic.rs");
        assert!(
            src.contains("Result<SyntheticDispatchOutcome, String>"),
            "SyntheticDispatchFn must return Result<SyntheticDispatchOutcome, String>"
        );
    }

    fn strip_line_comments(src: &str) -> String {
        src.lines()
            .map(|line| line.split_once("//").map_or(line, |(code, _)| code))
            .collect::<Vec<_>>()
            .join("\n")
    }

    /// B72: scheduled/synthetic research must be understood as the normal
    /// provider-chain path, not the legacy model-level fallback path.  This is
    /// a source-shape guard (like the API sentinels above): comments may mention
    /// the caveat, but executable code in the synthetic dispatch chain must not
    /// start consulting `research.fallback_models` without deliberately updating
    /// this test and the B72 registry entry.
    #[test]
    fn synthetic_research_does_not_consult_model_level_fallbacks() {
        let synthetic_src = strip_line_comments(
            include_str!("synthetic.rs")
                .split("\n#[cfg(test)]")
                .next()
                .expect("synthetic.rs has production section"),
        );
        let wiring_src = strip_line_comments(include_str!("wiring/research_scheduler.rs"));
        let session_control_src =
            strip_line_comments(include_str!("../../naked-core/src/session_ops/control.rs"));
        let legacy_dispatch_src = strip_line_comments(include_str!(
            "../../naked-core/src/research/coordinator_mod/runner_dispatch.rs"
        ));

        assert!(
            synthetic_src.contains("create_session_with_channel")
                && synthetic_src.contains("\"research\""),
            "synthetic dispatch must keep using a fresh research session"
        );
        assert!(
            synthetic_src.contains("submit_synthetic_prompt(agent, session_id, &msg).await"),
            "synthetic dispatch should enter the normal AgentCore::send_prompt path"
        );
        for (name, src) in [
            ("synthetic.rs", synthetic_src.as_str()),
            ("session_ops/control.rs", session_control_src.as_str()),
        ] {
            assert!(
                !src.contains("fallback_models"),
                "B72: {name} is on the synthetic path and must not silently start reading model-level research.fallback_models"
            );
        }
        assert!(
            wiring_src.contains("B72:") && wiring_src.contains("fallback_models.len()"),
            "wiring may read fallback_models only for the explicit B72 boot WARN"
        );
        let wiring_dispatch_impl = wiring_src
            .split("fn synthetic_dispatch_fn")
            .nth(1)
            .expect("research_scheduler.rs defines synthetic_dispatch_fn");
        assert!(
            !wiring_dispatch_impl.contains("fallback_models"),
            "B72: the synthetic dispatch closure/streaming path must not consume model-level fallback_models"
        );
        assert!(
            legacy_dispatch_src.contains("self.config.fallback_models"),
            "legacy ResearchCoordinator must remain the documented fallback_models consumer"
        );
        assert!(
            legacy_dispatch_src.contains("try_start_with_fallback"),
            "legacy runner_dispatch.rs should still expose the model-level fallback path"
        );
    }
}
