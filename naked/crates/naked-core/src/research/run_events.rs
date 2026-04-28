//! In-memory per-run event registry for live-progress UIs.
//!
//! Research runs live inside the coordinator's `drain_events` loop and
//! their internal signals (tool calls, finding saves, block-detection)
//! normally stay there — the outer caller only sees the terminal
//! `RunReport`. That's fine for the CLI, but the Telegram bot needs to
//! publish a "waterfall" of the last handful of actions every 20 s so
//! the operator can see that the run is actually making progress (as
//! opposed to wedged on a provider stream).
//!
//! This module keeps things intentionally small:
//!
//! * **Shape** — one [`RunEvent`] per interesting transition, a
//!   bounded [`RingBuffer`] per `run_id`, and a shared registry the
//!   coordinator/tools push into and the TG heartbeat reads from.
//! * **Lifetime** — in-memory only. If the process restarts the run is
//!   gone anyway, so there is no durability story here; the
//!   [`ModelHealth`](crate::model_catalog::ModelHealth) jsonl already
//!   owns the "survived a restart" stream.
//! * **Concurrency** — one `tokio::sync::RwLock` around a `HashMap`.
//!   Hot path is two inserts per drain-loop iteration, well under the
//!   coordinator's natural provider-stream throughput; no need for
//!   anything fancier.
//!
//! The registry is deliberately kept out of [`AgentRunner`] plumbing —
//! [`ResearchCoordinator`](crate::research::coordinator::ResearchCoordinator)
//! already owns the event stream, so we push from there (plus a handful
//! of in-tool emission points for moments the coordinator can't see,
//! e.g. `WebFetchTool` deciding a response is a captcha wall).

use std::collections::{HashMap, VecDeque};
use std::sync::Arc;

use chrono::{DateTime, Utc};
use tokio::sync::RwLock;

/// Hard ceiling on how many events we keep per run. TG asks for the
/// last 5; we keep a bit more so the heartbeat can show a little extra
/// context if the user opens `/research state` while the run is live.
const MAX_EVENTS_PER_RUN: usize = 32;

/// Kind of transition recorded by [`RunEventRegistry::push`]. Stable
/// string labels (via [`EventKind::as_str`]) are part of the bot's
/// rendering contract — don't rename them without updating
/// `render_waterfall` on the TG side.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EventKind {
    /// Coordinator crossed into a fresh agent turn (new `run_id`
    /// minted or feedback re-run kicked off).
    IterationStart,
    /// An `AgentEvent::ToolStart` was just observed.
    ToolCallStart,
    /// Paired with [`Self::ToolCallStart`]; emitted on
    /// `AgentEvent::ToolEnd`.
    ToolCallEnd,
    /// `research_save` persisted a new finding in the current run.
    FindingSaved,
    /// `web_fetch` (or a successor) detected a captcha / anti-bot
    /// wall. Emitted even when the agent elects to retry via the
    /// browser playbook — the operator cares about frequency, not
    /// resolution.
    BlockDetected,
    /// `Skill` tool loaded a skill (typically
    /// `web-browser-playbook`). Surfaces so the operator can see
    /// whether the prompt's captcha mandate is being honoured.
    SkillLoaded,
    /// Free-form breadcrumb — catch-all for events that don't fit the
    /// enum yet. Keep usage low; prefer adding a variant.
    Note,
}

impl EventKind {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::IterationStart => "iter",
            Self::ToolCallStart => "tool_start",
            Self::ToolCallEnd => "tool_end",
            Self::FindingSaved => "saved",
            Self::BlockDetected => "blocked",
            Self::SkillLoaded => "skill",
            Self::Note => "note",
        }
    }
}

/// One event in a run's waterfall. `label` is a short free-form
/// description (tool name + target, finding title, etc.) — the TG
/// renderer truncates it to whatever fits the 4096-char message
/// budget.
#[derive(Debug, Clone)]
pub struct RunEvent {
    pub at: DateTime<Utc>,
    pub kind: EventKind,
    pub label: String,
}

impl RunEvent {
    pub fn new(kind: EventKind, label: impl Into<String>) -> Self {
        Self {
            at: Utc::now(),
            kind,
            label: label.into(),
        }
    }
}

/// Fixed-size rolling buffer. Implements the `push_back` / drop-oldest
/// contract on top of `VecDeque` so `snapshot` stays O(n) in the cap.
#[derive(Debug, Default)]
struct RingBuffer {
    cap: usize,
    inner: VecDeque<RunEvent>,
}

impl RingBuffer {
    fn new(cap: usize) -> Self {
        Self {
            cap,
            inner: VecDeque::with_capacity(cap),
        }
    }

    fn push(&mut self, ev: RunEvent) {
        if self.inner.len() >= self.cap {
            self.inner.pop_front();
        }
        self.inner.push_back(ev);
    }

    fn tail(&self, n: usize) -> Vec<RunEvent> {
        let len = self.inner.len();
        let start = len.saturating_sub(n);
        self.inner.iter().skip(start).cloned().collect()
    }
}

/// Shared registry. Cheap to clone — the inner state is `Arc`-wrapped.
#[derive(Clone, Default, Debug)]
pub struct RunEventRegistry {
    inner: Arc<RwLock<HashMap<String, RingBuffer>>>,
}

impl RunEventRegistry {
    pub fn new() -> Self {
        Self::default()
    }

    /// Append a single event for `run_id`. Creates the ring buffer on
    /// demand so callers don't have to prime the map before the first
    /// event. Best-effort: if the lock is poisoned (impossible in
    /// practice, but tokio-rwlock is not panic-safe either way), the
    /// event is dropped and the run simply shows fewer items in its
    /// waterfall — no hard failure.
    pub async fn push(&self, run_id: &str, event: RunEvent) {
        let mut map = self.inner.write().await;
        map.entry(run_id.to_string())
            .or_insert_with(|| RingBuffer::new(MAX_EVENTS_PER_RUN))
            .push(event);
    }

    /// Synchronous variant for hot paths that don't have an async
    /// context — e.g. `Tool::execute` impls that are already inside a
    /// blocking scope. Uses `try_write`; silently drops the event if
    /// the lock is contended (the next tick will re-capture state
    /// anyway).
    pub fn push_blocking(&self, run_id: &str, event: RunEvent) {
        if let Ok(mut map) = self.inner.try_write() {
            map.entry(run_id.to_string())
                .or_insert_with(|| RingBuffer::new(MAX_EVENTS_PER_RUN))
                .push(event);
        }
    }

    /// Return up to `limit` most-recent events for `run_id`. Missing
    /// `run_id` (or an empty buffer) yields an empty vec.
    pub async fn snapshot(&self, run_id: &str, limit: usize) -> Vec<RunEvent> {
        let map = self.inner.read().await;
        map.get(run_id)
            .map(|buf| buf.tail(limit))
            .unwrap_or_default()
    }

    /// Forget everything we know about `run_id`. Called when the run
    /// reaches a terminal state so we don't leak in long-lived
    /// processes.
    pub async fn drop_run(&self, run_id: &str) {
        let mut map = self.inner.write().await;
        map.remove(run_id);
    }

    /// Number of live buffers in the registry. Test helper; exposed
    /// so integration tests can assert cleanup without poking
    /// private state.
    #[cfg(test)]
    pub async fn len(&self) -> usize {
        self.inner.read().await.len()
    }

    #[cfg(test)]
    pub async fn is_empty(&self) -> bool {
        self.inner.read().await.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn push_and_snapshot_roundtrip() {
        let reg = RunEventRegistry::new();
        reg.push("run-a", RunEvent::new(EventKind::IterationStart, "1/30"))
            .await;
        reg.push(
            "run-a",
            RunEvent::new(EventKind::ToolCallStart, "web_fetch"),
        )
        .await;
        let tail = reg.snapshot("run-a", 10).await;
        assert_eq!(tail.len(), 2);
        assert_eq!(tail[0].kind, EventKind::IterationStart);
        assert_eq!(tail[1].kind, EventKind::ToolCallStart);
        assert_eq!(tail[1].label, "web_fetch");
    }

    #[tokio::test]
    async fn snapshot_tail_respects_limit() {
        let reg = RunEventRegistry::new();
        for i in 0..10 {
            reg.push("r", RunEvent::new(EventKind::Note, format!("n{i}")))
                .await;
        }
        let tail = reg.snapshot("r", 3).await;
        assert_eq!(tail.len(), 3);
        assert_eq!(tail[0].label, "n7");
        assert_eq!(tail[2].label, "n9");
    }

    #[tokio::test]
    async fn ring_buffer_caps_at_max_per_run() {
        let reg = RunEventRegistry::new();
        for i in 0..(MAX_EVENTS_PER_RUN + 20) {
            reg.push("r", RunEvent::new(EventKind::Note, format!("n{i}")))
                .await;
        }
        let tail = reg.snapshot("r", 9999).await;
        assert_eq!(tail.len(), MAX_EVENTS_PER_RUN);
        // Oldest remaining is the (MAX)-th event we pushed (0-indexed = 20).
        let expected_first = format!("n{}", 20);
        assert_eq!(tail[0].label, expected_first);
    }

    #[tokio::test]
    async fn drop_run_clears_buffer() {
        let reg = RunEventRegistry::new();
        reg.push("r", RunEvent::new(EventKind::Note, "x")).await;
        assert_eq!(reg.len().await, 1);
        reg.drop_run("r").await;
        assert_eq!(reg.len().await, 0);
        assert!(reg.snapshot("r", 1).await.is_empty());
    }

    #[tokio::test]
    async fn missing_run_yields_empty_tail() {
        let reg = RunEventRegistry::new();
        let tail = reg.snapshot("nope", 10).await;
        assert!(tail.is_empty());
    }

    #[tokio::test]
    async fn push_blocking_is_ok_under_no_contention() {
        let reg = RunEventRegistry::new();
        reg.push_blocking("r", RunEvent::new(EventKind::Note, "sync"));
        let tail = reg.snapshot("r", 10).await;
        assert_eq!(tail.len(), 1);
        assert_eq!(tail[0].label, "sync");
    }

    #[test]
    fn event_kind_labels_are_stable() {
        // TG's `render_waterfall` depends on these strings — keep a
        // single-point reminder so a drive-by rename trips the test.
        assert_eq!(EventKind::IterationStart.as_str(), "iter");
        assert_eq!(EventKind::ToolCallStart.as_str(), "tool_start");
        assert_eq!(EventKind::ToolCallEnd.as_str(), "tool_end");
        assert_eq!(EventKind::FindingSaved.as_str(), "saved");
        assert_eq!(EventKind::BlockDetected.as_str(), "blocked");
        assert_eq!(EventKind::SkillLoaded.as_str(), "skill");
        assert_eq!(EventKind::Note.as_str(), "note");
    }
}
