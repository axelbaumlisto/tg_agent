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
//! The scheduler injects a [`SyntheticMessage`] into the normal message
//! handling pipeline via a [`SyntheticDispatchFn`] closure that wiring.rs
//! provides. The closure has access to `Bot` + all `BotDeps`; the scheduler
//! only sees the opaque function type.
//!
//! A synthetic message is identical to a real Telegram message from the
//! operator's perspective — it lands in the same `(chat_id, thread_id)`,
//! creates a session in the `ChannelSessionMap`, and the operator can abort
//! it with `/abort`. The only difference is that the bot does NOT echo the
//! generated command text back to Telegram (to avoid polluting chat history
//! with `/research run <id>` lines).
//!
//! # B57 (PLAN_RESEARCH_AGENT_FLOW_v1)
//!
//! `SCHEDULER-SPAWNS-PARALLEL-SESSION`: scheduler creates ephemeral session
//! with channel ≠ chat's. Fix: scheduler calls `dispatch_synthetic` which
//! routes through the normal message handler → same chat session → operator
//! can `/abort` like any other turn.

use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;

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
}

impl SyntheticMessage {
    /// Build a synthetic `/research run <spec_id>` message from a scheduler
    /// spec. Returns `None` if the spec has no `chat_id` configured (CLI /
    /// unattended runs fall back to the legacy direct-run path).
    pub fn from_scheduler_spec(spec_id: &str, chat_id: i64, thread_id: Option<i32>) -> Self {
        Self {
            chat_id,
            thread_id,
            text: format!("/research run {spec_id}"),
            source: SyntheticSource::Scheduler {
                spec_id: spec_id.to_string(),
            },
        }
    }
}

/// Opaque async function that the scheduler calls to inject a
/// [`SyntheticMessage`] into the message-handling pipeline.
///
/// `wiring.rs` constructs this as a closure over `Bot` + `BotDeps`. The
/// scheduler only holds `Arc<SyntheticDispatchFn>` — no `Bot` in scope.
///
/// ## Why a boxed async fn instead of a trait?
///
/// The scheduler config needs to be `Clone` + `Default`. A trait object would
/// require `Arc<dyn …>` and async traits (which are not object-safe without
/// boxing). The boxed closure is simpler and achieves the same result.
pub type SyntheticDispatchFn =
    Arc<dyn Fn(SyntheticMessage) -> Pin<Box<dyn Future<Output = ()> + Send>> + Send + Sync>;

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
    /// capture log.
    pub fn as_fn(&self) -> SyntheticDispatchFn {
        let captured = self.captured.clone();
        let delay_ms = self.delay_ms;
        Arc::new(move |msg: SyntheticMessage| {
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
                captured.lock().await.push(msg);
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
// wiring.rs (which has Bot + ChannelSessionMap + AgentCore in scope).
// When the scheduler decides a spec is due, it calls the closure with
// a SyntheticMessage; the closure drives the message through the
// regular agent turn machinery so the operator sees:
//   - normal streaming bubble in their chat thread
//   - the same ⏹ Abort button that any chat turn has
//   - `/abort` works to cancel (no separate Stop callback path)
//
// Echo suppression: the closure does NOT call `bot.send_message` for
// the synthetic command text itself — there is no operator-visible
// "/research run X" line in chat history. The first visible message
// is the bot's own streaming bubble.
//
// This module provides the BUILDER for the closure. wiring.rs calls
// `build_dispatch_fn(...)` once at boot, passes the result to
// `SchedulerConfig::dispatch_fn`. The closure body is a thin wrapper
// around `agent.send_prompt` + `streaming::stream_response` (the same
// path message_handler.rs uses, minus addressing gates + media extraction
// since synthetic messages are pure-text by construction).
// ────────────────────────────────────────────────────────────────────────

/// Synthesize a turn for the given `(chat_id, thread_id)` and return the
/// resulting session id + agent handle. Used by the scheduler-facing
/// `SyntheticDispatchFn` closure (wired in `wiring.rs`).
///
/// This function isolates the **policy** of synthetic-turn dispatch from
/// the **wiring** of `Bot` + `BotDeps`. The streaming itself (HTTP edits,
/// Abort button, etc.) lives in `streaming::stream_response` — the caller
/// plugs that in. Echo suppression: the caller does NOT echo `msg.text`
/// as a separate `send_message` — the first operator-visible artifact is
/// the bot's own streaming bubble.
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
    channel_map: &std::sync::Arc<crate::channel_map::ChannelSessionMap>,
    chat_id: i64,
    thread_id: Option<i32>,
    spec_id: &str,
) -> Result<(String, naked_core::types::AgentHandle), String> {
    let session_id = resolve_or_create_session(agent, channel_map, chat_id, thread_id).await;
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

    // ── MockDispatcher tests (replaces the previous include_str! sentinel) ──

    /// MockDispatcher captures messages exactly once per dispatch call.
    #[tokio::test]
    async fn mock_dispatcher_captures_single_message() {
        let mock = MockDispatcher::new();
        let dispatch = mock.as_fn();

        let msg = SyntheticMessage::from_scheduler_spec("spec-1", 12345, Some(7));
        dispatch(msg.clone()).await;

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
            dispatch(SyntheticMessage::from_scheduler_spec(
                &format!("spec-{i}"),
                1000 + i,
                None,
            ))
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

        d1(SyntheticMessage::from_scheduler_spec("a", 1, None)).await;
        d2(SyntheticMessage::from_scheduler_spec("b", 2, None)).await;

        // Either handle sees both messages.
        assert_eq!(mock.count().await, 2);
        assert_eq!(mock_b.count().await, 2);
    }

    /// `reset()` clears between sub-tests without rebuilding the handle.
    #[tokio::test]
    async fn mock_dispatcher_reset_clears_log() {
        let mock = MockDispatcher::new();
        let dispatch = mock.as_fn();

        dispatch(SyntheticMessage::from_scheduler_spec("x", 1, None)).await;
        assert_eq!(mock.count().await, 1);

        mock.reset().await;
        assert_eq!(mock.count().await, 0);

        dispatch(SyntheticMessage::from_scheduler_spec("y", 2, None)).await;
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
        dispatch(SyntheticMessage::from_scheduler_spec("d", 1, None)).await;
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
}
