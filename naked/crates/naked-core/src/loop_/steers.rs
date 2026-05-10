//! Steer message draining extracted from `AgentLoop::run`.

use crate::history::ConversationHistory;
use crate::types::{AgentEvent, SteerMessage};
use tokio::sync::mpsc;

impl super::AgentLoop {
    pub(super) async fn drain_steers(
        steer_rx: &mut Option<mpsc::Receiver<SteerMessage>>,
        pending: &mut Vec<SteerMessage>,
        delivered: &mut std::collections::HashSet<i32>,
        history: &mut ConversationHistory,
        tx: &mpsc::Sender<AgentEvent>,
    ) {
        let rx = match steer_rx.as_mut() {
            Some(rx) => rx,
            None => return,
        };

        // Collect new messages from the channel.
        while let Ok(msg) = rx.try_recv() {
            if msg.is_edit {
                // Try to replace in pending queue.
                if let Some(existing) = pending.iter_mut().find(|m| m.msg_id == msg.msg_id) {
                    existing.text = msg.text;
                    continue;
                }
                // Already delivered to LLM — send as correction.
                if delivered.contains(&msg.msg_id) {
                    pending.push(SteerMessage {
                        msg_id: msg.msg_id,
                        text: format!("[correction] {}", msg.text),
                        is_edit: false,
                    });
                    continue;
                }
            }
            pending.push(msg);
        }

        if pending.is_empty() {
            return;
        }

        // Merge all pending into ONE user message.
        let combined: String = pending
            .iter()
            .map(|m| m.text.as_str())
            .collect::<Vec<_>>()
            .join("\n\n");

        // Collect ids in original order before clearing pending. The
        // bot uses these to delete the matching "↩️ Принято"
        // temp confirmation messages once delivery is actually done
        // (PLAN_NEXT_SESSION.md §A.2 S6).
        let msg_ids: Vec<i32> = pending.iter().map(|m| m.msg_id).collect();

        // Track delivered msg_ids.
        for m in pending.iter() {
            delivered.insert(m.msg_id);
        }
        pending.clear();

        history.push_user(&combined);
        // F3: bump observability counter so operators can graph
        // "steers actually delivered to the model".
        crate::types::STEER_DELIVERED_COUNT.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let _ = tx
            .send(AgentEvent::SteerReceived {
                text: combined,
                msg_ids,
            })
            .await;
    }
}
