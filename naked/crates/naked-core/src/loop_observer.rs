//! Observer interface for AgentLoop business events (DIP).

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RetryKind {
    Connect,
    MidStream,
    EmptyContent,
}

#[allow(unused_variables)]
pub trait LoopObserver: Send + Sync {
    /// Fired before each retry sleep. `err` is empty string for EmptyContent kind.
    fn on_retry(&self, kind: RetryKind, attempt: usize, max: usize, delay_ms: u64, err: &str) {}
    /// Fired once per emergency compaction round (0-indexed).
    fn on_compact_round(&self, round: usize, keep: usize, msgs: usize, est_tokens: usize) {}
    /// Fired after the compaction loop finishes.
    fn on_compact_done(&self, before: usize, after: usize) {}
    /// Fired on checkpoint-restart cycle.
    fn on_cycle_restart(&self, cycle: u64, archived: usize, path: &str) {}
    /// Fired when retries exhausted and the loop returns Err.
    fn on_giveup(&self, reason: &str) {}
}

pub struct TracingObserver;

impl LoopObserver for TracingObserver {
    fn on_retry(&self, kind: RetryKind, attempt: usize, max: usize, delay_ms: u64, err: &str) {
        let label = match kind {
            RetryKind::Connect => "stream_chat connect error",
            RetryKind::MidStream => "mid-stream error",
            RetryKind::EmptyContent => "provider returned empty content",
        };
        if matches!(kind, RetryKind::EmptyContent) {
            tracing::warn!("{label} (retry {attempt}/{max}, backoff {delay_ms}ms)");
        } else {
            tracing::warn!("{label} (retry {attempt}/{max}, backoff {delay_ms}ms): {err}");
        }
    }
    fn on_compact_round(&self, round: usize, keep: usize, msgs: usize, est_tokens: usize) {
        tracing::warn!(round, keep, msgs, est_tokens, "emergency compaction round");
    }
    fn on_compact_done(&self, before: usize, after: usize) {
        tracing::warn!("emergency compaction done: {before} -> {after} msgs");
    }
    fn on_cycle_restart(&self, cycle: u64, archived: usize, path: &str) {
        tracing::info!(
            cycle,
            archived,
            path,
            "cycle restart: archived and restarted"
        );
    }
    fn on_giveup(&self, reason: &str) {
        tracing::warn!("{reason}");
    }
}

pub struct NoopObserver;
impl LoopObserver for NoopObserver {}
