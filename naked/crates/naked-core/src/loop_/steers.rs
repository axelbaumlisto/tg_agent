//! Steer-injection pipeline (R1 of `PLAN_NEXT_SESSION.md`).
//!
//! Single-source-of-truth for steer state across an `AgentLoop::run`
//! invocation. Replaces the old free-function `drain_steers` that
//! had to be invoked from seven scattered sites in `run.rs` plus
//! one site in `stream_turn.rs`, every call passing the same three
//! `&mut`s (`steer_rx`, `pending`, `delivered`).
//!
//! Wins:
//!   * SRP — the pipeline OWNS the in-flight pending queue + the
//!     delivered-msg-id set. Outer code never reaches in to mutate
//!     either; it only calls `drain` / `record_winner_and_burst`.
//!   * DRY — one `drain` body; no chance of behavioural drift
//!     between the four "iteration / mid-stream / tool / pre-Idle"
//!     drains and the three "drain-on-error" rescue calls.
//!   * Testability — `SteerPipeline` is a plain struct without
//!     any of `AgentLoop`'s heavy state, so we can pin its
//!     edit-replacement / correction / burst-drain semantics in
//!     a tiny unit-test module without spinning up a full loop.
//!
//! The `Receiver<SteerMessage>` itself is intentionally NOT held by
//! the pipeline — `stream_one_turn`'s `tokio::select!` needs to own
//! the recv arm directly. The pipeline accepts the receiver as a
//! `&mut Option<...>` argument on the methods that talk to it. This
//! keeps the select! arm trivial while still routing every actual
//! mutation through the pipeline.
//!
//! See `SteerPipeline::drain` for the merge / edit / correction
//! contract.

use std::collections::HashSet;

use tokio::sync::mpsc;

use crate::history::ConversationHistory;
use crate::types::{AgentEvent, SteerMessage};

/// Owns the pending-steer vec + the delivered-msg-id set for one
/// agent-loop invocation. Cheap to construct (`Default::default`).
#[derive(Debug, Default)]
pub(super) struct SteerPipeline {
    /// Steers received but not yet merged into history.
    /// `Vec` so edits can replace a prior entry by `msg_id` before
    /// the next drain runs.
    pending: Vec<SteerMessage>,
    /// Telegram message-ids of steers that already reached history
    /// in a previous drain. Used to convert a late `is_edit` into a
    /// `[correction] …` follow-up rather than silently shadowing
    /// the original.
    delivered: HashSet<i32>,
}

impl SteerPipeline {
    pub fn new() -> Self {
        Self::default()
    }

    /// Called from `stream_one_turn`'s select! arm when a steer wins
    /// against the LLM-stream future. Stashes the winner in
    /// `pending`, then burst-drains any messages that arrived in the
    /// same scheduler tick so two close-spaced steers merge into one
    /// re-issue (matches the contract pinned by `steer_golden::Q5`).
    pub fn record_winner_and_burst(
        &mut self,
        winner: SteerMessage,
        rx: &mut Option<mpsc::Receiver<SteerMessage>>,
    ) {
        self.pending.push(winner);
        if let Some(r) = rx.as_mut() {
            while let Ok(more) = r.try_recv() {
                self.pending.push(more);
            }
        }
    }

    /// Single drain entry point. Called at iteration boundaries,
    /// after a mid-stream soft-interrupt, after gated-tool execution,
    /// before an Idle exit, and on every Err return path of
    /// `AgentLoop::run`.
    ///
    /// Behaviour:
    ///   * `try_recv` everything currently buffered in `rx`. Edits
    ///     replace a matching pending entry by `msg_id`; an edit of
    ///     an already-delivered steer is converted into a synthetic
    ///     `[correction] …` message (so the model sees the change
    ///     instead of having it silently dropped).
    ///   * If `pending` ends up empty after the drain pass, return
    ///     `false` immediately without touching `history` or `tx`.
    ///   * Otherwise merge every pending text with `\n\n` separators
    ///     into ONE user message, push it to `history`, mark every
    ///     contributing `msg_id` as delivered, bump
    ///     `STEER_DELIVERED_COUNT`, and emit `AgentEvent::SteerReceived`
    ///     with the merged text + the original msg-ids (the bot uses
    ///     them to clean up "↩️ Принято" temp confirmations).
    ///
    /// Returns `true` if at least one steer was merged into history,
    /// so callers can bump `STEER_DRAINED_ON_ABORT_COUNT` on the
    /// rescue paths without re-implementing the message-count delta
    /// they used to do inline.
    pub async fn drain(
        &mut self,
        rx: &mut Option<mpsc::Receiver<SteerMessage>>,
        history: &mut ConversationHistory,
        tx: &mpsc::Sender<AgentEvent>,
    ) -> bool {
        let rx = match rx.as_mut() {
            Some(rx) => rx,
            // No steer channel attached — pipeline is a no-op.
            None => return false,
        };

        // Collect newly-buffered messages, applying edit semantics.
        while let Ok(msg) = rx.try_recv() {
            if msg.is_edit {
                if let Some(existing) = self.pending.iter_mut().find(|m| m.msg_id == msg.msg_id) {
                    existing.text = msg.text;
                    continue;
                }
                if self.delivered.contains(&msg.msg_id) {
                    self.pending.push(SteerMessage {
                        msg_id: msg.msg_id,
                        text: format!("[correction] {}", msg.text),
                        is_edit: false,
                    });
                    continue;
                }
            }
            self.pending.push(msg);
        }

        if self.pending.is_empty() {
            return false;
        }

        let combined: String = self
            .pending
            .iter()
            .map(|m| m.text.as_str())
            .collect::<Vec<_>>()
            .join("\n\n");
        let msg_ids: Vec<i32> = self.pending.iter().map(|m| m.msg_id).collect();
        for m in self.pending.iter() {
            self.delivered.insert(m.msg_id);
        }
        self.pending.clear();

        history.push_user(&combined);
        crate::types::STEER_DELIVERED_COUNT.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let _ = tx
            .send(AgentEvent::SteerReceived {
                text: combined,
                msg_ids,
            })
            .await;
        true
    }
}

#[cfg(test)]
mod tests {
    //! Pin the pipeline contract independently of `AgentLoop` so
    //! future refactors of `run.rs` can't quietly change drain
    //! semantics. The full integration is still covered by
    //! `tests/loop_golden.rs::G8` and `tests/steer_golden.rs::Q1..Q5`.
    use super::*;
    use crate::history::ConversationHistory;
    use tokio::sync::mpsc;

    fn make() -> (
        SteerPipeline,
        ConversationHistory,
        mpsc::Sender<AgentEvent>,
        mpsc::Receiver<AgentEvent>,
    ) {
        let p = SteerPipeline::new();
        let h = ConversationHistory::new("sys".into());
        let (tx, rx) = mpsc::channel(16);
        (p, h, tx, rx)
    }

    #[tokio::test]
    async fn no_channel_drain_is_noop() {
        let (mut p, mut h, tx, _rx) = make();
        let mut none: Option<mpsc::Receiver<SteerMessage>> = None;
        assert!(!p.drain(&mut none, &mut h, &tx).await);
        assert_eq!(h.message_count(), 0);
    }

    #[tokio::test]
    async fn empty_channel_drain_returns_false() {
        let (mut p, mut h, tx, _rx) = make();
        let (_st, sr) = mpsc::channel::<SteerMessage>(4);
        let mut some = Some(sr);
        assert!(!p.drain(&mut some, &mut h, &tx).await);
        assert_eq!(h.message_count(), 0);
    }

    #[tokio::test]
    async fn single_steer_drains_and_returns_true() {
        let (mut p, mut h, tx, _rx) = make();
        let (st, sr) = mpsc::channel::<SteerMessage>(4);
        st.send(SteerMessage {
            msg_id: 1,
            text: "hi".into(),
            is_edit: false,
        })
        .await
        .unwrap();
        let mut some = Some(sr);
        assert!(p.drain(&mut some, &mut h, &tx).await);
        assert_eq!(h.message_count(), 1);
    }

    #[tokio::test]
    async fn two_steers_merge_into_one_message_with_double_newline() {
        let (mut p, mut h, tx, _rx) = make();
        let (st, sr) = mpsc::channel::<SteerMessage>(4);
        st.send(SteerMessage {
            msg_id: 1,
            text: "first".into(),
            is_edit: false,
        })
        .await
        .unwrap();
        st.send(SteerMessage {
            msg_id: 2,
            text: "second".into(),
            is_edit: false,
        })
        .await
        .unwrap();
        let mut some = Some(sr);
        assert!(p.drain(&mut some, &mut h, &tx).await);
        assert_eq!(h.message_count(), 1);
        // Single user message with both texts, separated by \n\n.
        let last = h.messages().last().unwrap();
        let body = last.text_content();
        assert!(body.contains("first"));
        assert!(body.contains("second"));
        assert!(body.contains("first\n\nsecond"));
    }

    #[tokio::test]
    async fn edit_of_pending_replaces_in_place_no_duplicate() {
        let (mut p, mut h, tx, _rx) = make();
        let (st, sr) = mpsc::channel::<SteerMessage>(4);
        st.send(SteerMessage {
            msg_id: 7,
            text: "old".into(),
            is_edit: false,
        })
        .await
        .unwrap();
        st.send(SteerMessage {
            msg_id: 7,
            text: "new".into(),
            is_edit: true,
        })
        .await
        .unwrap();
        let mut some = Some(sr);
        assert!(p.drain(&mut some, &mut h, &tx).await);
        let body = h.messages().last().unwrap().text_content();
        assert!(body.contains("new"), "edit must replace 'old'; got {body}");
        assert!(
            !body.contains("old"),
            "old text must not survive; got {body}"
        );
    }

    #[tokio::test]
    async fn edit_after_delivery_emits_correction() {
        let (mut p, mut h, tx, _rx) = make();
        let (st, sr) = mpsc::channel::<SteerMessage>(4);

        // First drain: deliver msg_id=42.
        st.send(SteerMessage {
            msg_id: 42,
            text: "original".into(),
            is_edit: false,
        })
        .await
        .unwrap();
        let mut some = Some(sr);
        assert!(p.drain(&mut some, &mut h, &tx).await);
        assert_eq!(h.message_count(), 1);

        // Second drain: edit of already-delivered msg_id=42 → correction.
        st.send(SteerMessage {
            msg_id: 42,
            text: "fixed".into(),
            is_edit: true,
        })
        .await
        .unwrap();
        assert!(p.drain(&mut some, &mut h, &tx).await);
        assert_eq!(h.message_count(), 2);
        let last = h.messages().last().unwrap().text_content();
        assert!(last.contains("[correction]"));
        assert!(last.contains("fixed"));
    }

    #[tokio::test]
    async fn record_winner_and_burst_picks_up_buffered_followers() {
        let (mut p, mut h, tx, _rx) = make();
        let (st, sr) = mpsc::channel::<SteerMessage>(4);
        // Buffer two extra messages BEFORE handing the rx to the pipeline.
        st.send(SteerMessage {
            msg_id: 2,
            text: "two".into(),
            is_edit: false,
        })
        .await
        .unwrap();
        st.send(SteerMessage {
            msg_id: 3,
            text: "three".into(),
            is_edit: false,
        })
        .await
        .unwrap();
        let mut some = Some(sr);

        // Simulate the select! arm: the FIRST message would have won
        // a race; here we just feed it directly.
        let winner = SteerMessage {
            msg_id: 1,
            text: "one".into(),
            is_edit: false,
        };
        p.record_winner_and_burst(winner, &mut some);

        // The drain merges all three.
        assert!(p.drain(&mut some, &mut h, &tx).await);
        let body = h.messages().last().unwrap().text_content();
        for needle in ["one", "two", "three"] {
            assert!(
                body.contains(needle),
                "merged body must include {needle}; got {body}"
            );
        }
    }
}
